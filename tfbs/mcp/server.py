"""
MCP server exposing TFBS-NN model loading, prediction, preprocessing,
attribution analysis, and FIMO motif scanning as tools.

Run with:  tfbs-mcp          (stdio transport, for Claude Desktop / VS Code)
           python -m tfbs.mcp.server   (same)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
mcp = FastMCP("tfbs")

# ---------------------------------------------------------------------------
# In-process model registry  (persists for the lifetime of the server)
# ---------------------------------------------------------------------------
_model_registry: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Environment detection — runs once at import time (lightweight, no torch).
# ---------------------------------------------------------------------------
def _detect_environment() -> dict[str, Any]:
    """Detect SLURM cluster and GPU availability.

    Checks, in order:
      1. SLURM_GPUS_ON_NODE env var
      2. CUDA_VISIBLE_DEVICES env var
      3. nvidia-smi (quick subprocess probe)

    Returns a dict describing the runtime:
      - on_slurm: True if running inside a SLURM job
      - slurm_job_id: str or None
      - slurm_partition: str or None
      - gpu_count: detected number of GPUs (0 if none found)
      - default_device: "cuda" or "cpu"
      - default_batch_size: larger on GPU nodes
    """
    on_slurm = "SLURM_JOB_ID" in os.environ
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_partition = os.environ.get("SLURM_JOB_PARTITION")

    # Count GPUs — try multiple detection methods
    gpu_count = 0

    if os.environ.get("SLURM_GPUS_ON_NODE"):
        try:
            gpu_count = int(os.environ["SLURM_GPUS_ON_NODE"])
        except ValueError:
            pass
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        cvd = os.environ["CUDA_VISIBLE_DEVICES"].strip()
        if cvd:
            gpu_count = len(cvd.split(","))

    # Fallback: probe nvidia-smi (works when SLURM doesn't set GPU vars)
    if gpu_count == 0:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
                gpu_count = len(lines)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    has_gpu = gpu_count > 0

    return {
        "on_slurm": on_slurm,
        "slurm_job_id": slurm_job_id,
        "slurm_partition": slurm_partition,
        "gpu_count": gpu_count,
        "default_device": "cuda" if has_gpu else "cpu",
        "default_batch_size": 512 if has_gpu else 64,
    }


_ENV = _detect_environment()


# ---------------------------------------------------------------------------
# Lazy imports — torch and friends are heavy (~13 s).  We defer them so the
# MCP handshake completes instantly and Claude Code doesn't time out.
# ---------------------------------------------------------------------------
_torch = None
_F = None
_np = None
_models = None
_find_best_windows_fast = None


def _ensure_imports():
    """Import heavy dependencies on first tool call, not at server start."""
    global _torch, _F, _np, _models, _find_best_windows_fast
    if _torch is not None:
        return
    import numpy as np
    import torch
    import torch.nn.functional as F
    import tfbs.nn.models as models
    from tfbs.prediction.analysis import find_best_windows_fast

    _torch = torch
    _F = F
    _np = np
    _models = models
    _find_best_windows_fast = find_best_windows_fast

    # Verify the env-var GPU hint against actual torch capability
    if _ENV["default_device"] == "cuda" and not torch.cuda.is_available():
        _ENV["default_device"] = "cpu"
        _ENV["default_batch_size"] = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
_IDX_TO_NUC = ["A", "C", "G", "T"]


def _sequences_to_tensor(sequences: list[str]):
    """Convert DNA strings to a one-hot tensor of shape (N, 4, L)."""
    _ensure_imports()
    indices = _torch.tensor(
        [[_NUC_TO_IDX[c] for c in seq.upper()] for seq in sequences],
        dtype=_torch.long,
    )
    ohe = _F.one_hot(indices, num_classes=4).float()  # (N, L, 4)
    return ohe.permute(0, 2, 1)  # (N, 4, L)


def _tensor_to_sequences(tensor) -> list[str]:
    """Convert a (N, 4, L) one-hot tensor back to DNA strings."""
    indices = tensor.argmax(dim=1).cpu().numpy()  # (N, L)
    return ["".join(_IDX_TO_NUC[i] for i in row) for row in indices]


def _get_model(model_id: str) -> dict[str, Any]:
    """Retrieve a loaded model or raise a clear error."""
    if model_id not in _model_registry:
        available = list(_model_registry.keys()) or ["(none)"]
        raise ValueError(
            f"Model '{model_id}' not loaded. "
            f"Call tfbs_load_model first. Available: {', '.join(available)}"
        )
    return _model_registry[model_id]


def _validate_dna(sequences: list[str]) -> None:
    """Check that every sequence contains only valid DNA characters."""
    valid = set("ACGTacgt")
    for i, seq in enumerate(sequences):
        bad = set(seq) - valid
        if bad:
            raise ValueError(
                f"Sequence {i} contains invalid characters: {bad}. "
                "Only A, C, G, T are allowed."
            )


def _tail_file(path: str, n: int = 100) -> str:
    """Read the last *n* lines of a file efficiently."""
    try:
        with open(path) as f:
            return "\n".join(deque(f, maxlen=n))
    except FileNotFoundError:
        return f"[File not found: {path}]"
    except PermissionError:
        return f"[Permission denied: {path}]"


# ---------------------------------------------------------------------------
# Tool 1: tfbs_load_model
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_load_model",
    annotations={
        "title": "Load TFBS Model",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_load_model(
    model_id: str,
    checkpoint_path: str,
    model_type: str,
    device: str | None = None,
) -> str:
    """Load a trained TFBS model checkpoint into the server's memory.

    The model stays loaded for the lifetime of the server and can be
    referenced by model_id in subsequent predict / analyze calls.

    Args:
        model_id: User-chosen identifier for this model (e.g. "my_vcnn").
        checkpoint_path: Absolute path to a .ckpt file on disk.
        model_type: Architecture class name from tfbs.nn.models
                    (e.g. "VCNNBpnet", "RNN").
        device: "cpu" or "cuda".  Auto-detected if omitted (uses GPU when
                available on SLURM, CPU otherwise).

    Returns:
        JSON with model_id, model_type, input_length, classify flag, and device.
    """
    _ensure_imports()

    if device is None:
        device = _ENV["default_device"]

    path = Path(checkpoint_path).resolve()
    if not path.exists():
        return json.dumps({"error": f"Checkpoint not found: {checkpoint_path}"})

    try:
        model_cls = getattr(_models, model_type)
    except AttributeError:
        available = [
            n for n in dir(_models)
            if isinstance(getattr(_models, n, None), type)
            and issubclass(getattr(_models, n), _torch.nn.Module)
            and n != "BaseModel"
        ]
        return json.dumps({
            "error": f"Unknown model_type '{model_type}'.",
            "available": available,
        })

    model = model_cls.load_from_checkpoint(str(path), map_location=device)
    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    model = model.to(device)

    input_length = getattr(model, "input_length", None)
    classify = getattr(model, "classify", False)

    _model_registry[model_id] = {
        "model": model,
        "model_type": model_type,
        "device": device,
        "input_length": input_length,
        "classify": classify,
    }

    return json.dumps({
        "model_id": model_id,
        "model_type": model_type,
        "input_length": input_length,
        "classify": classify,
        "device": device,
    })


# ---------------------------------------------------------------------------
# Tool 2: tfbs_predict
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_predict",
    annotations={
        "title": "Predict TF Binding",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_predict(
    model_id: str,
    sequences: list[str],
    window_mode: bool = False,
    window_size: int = 70,
    batch_size: int | None = None,
) -> str:
    """Score DNA sequences for transcription-factor binding affinity.

    In normal mode each sequence must match the model's expected input
    length.  In window_mode a sliding window finds the highest-scoring
    sub-sequence within longer inputs.

    Args:
        model_id: Identifier of a model loaded via tfbs_load_model.
        sequences: List of DNA sequences (e.g. ["ACGTACGT..."]).
        window_mode: Use sliding-window scoring for long sequences.
        window_size: Window length for sliding-window mode.
        batch_size: Batch size for processing.  Auto-detected if omitted
                    (512 on GPU, 64 on CPU).

    Returns:
        JSON with "scores" (normal) or "best_scores", "best_sequences",
        and "mean_scores" (window mode).
    """
    _ensure_imports()
    _validate_dna(sequences)

    if batch_size is None:
        batch_size = _ENV["default_batch_size"]

    entry = _get_model(model_id)
    model = entry["model"]
    device = entry["device"]

    X = _sequences_to_tensor(sequences).to(device)

    if window_mode:
        best_windows, best_scores, _, mean_scores = _find_best_windows_fast(
            X, model, window_size=window_size,
            processing_batch_size=batch_size, device=device,
        )
        best_seqs = _tensor_to_sequences(_torch.from_numpy(best_windows))
        return json.dumps({
            "best_scores": best_scores.tolist(),
            "best_sequences": best_seqs,
            "mean_scores": mean_scores.tolist(),
        })

    with _torch.no_grad():
        scores = model(X).cpu().numpy()

    return json.dumps({"scores": scores.squeeze().tolist()})


# ---------------------------------------------------------------------------
# Tool 3: tfbs_preprocess
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_preprocess",
    annotations={
        "title": "Preprocess DNA Sequences",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_preprocess(
    sequences: list[str],
    labels: list[float] | None = None,
    add_rc: bool = False,
    output_path: str | None = None,
) -> str:
    """One-hot encode DNA sequences and optionally add reverse complements.

    Encodes each base as a 4-channel vector (A=0, C=1, G=2, T=3) producing
    tensors of shape (N, 4, L).  If add_rc is set the reverse complement of
    every sequence is appended, doubling N.

    Args:
        sequences: Raw DNA strings (A/C/G/T only).
        labels: Optional numeric labels for each sequence.
        add_rc: Whether to append reverse-complement augmentation.
        output_path: If provided, save tensors (seqs.pt, labels.pt) here.

    Returns:
        JSON with tensor shape and num_sequences.  If output_path was given,
        includes the directory the tensors were saved to.
    """
    _ensure_imports()
    _validate_dna(sequences)

    X = _sequences_to_tensor(sequences)

    if labels is not None:
        y = _torch.tensor(labels, dtype=_torch.float32).unsqueeze(1)
    else:
        y = None

    if add_rc:
        rc = _torch.flip(X, [2])
        rc = _torch.index_select(rc, 1, _torch.tensor([3, 2, 1, 0], dtype=_torch.long))
        X = _torch.cat([X, rc], dim=0)
        if y is not None:
            y = _torch.cat([y, y], dim=0)

    result: dict[str, Any] = {
        "shape": list(X.shape),
        "num_sequences": X.shape[0],
    }

    if output_path:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        _torch.save(X, out / "seqs.pt")
        if y is not None:
            _torch.save(y, out / "labels.pt")
        result["saved_to"] = str(out)
    else:
        result["sequences_encoded"] = True

    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 4: tfbs_analyze (attribution)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_analyze",
    annotations={
        "title": "Attribution Analysis",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_analyze(
    model_id: str,
    sequences: list[str],
    method: str = "deepliftshap",
    n_shuffles: int = 20,
    hypothetical: bool = False,
) -> str:
    """Run attribution analysis to identify important sequence positions.

    Uses Captum to compute per-base importance scores for a loaded model.
    Requires the viz extras: pip install 'tfbs-nn[viz]'

    Args:
        model_id: Identifier of a model loaded via tfbs_load_model.
        sequences: DNA sequences to analyze (A/C/G/T).
        method: "deepliftshap" or "input_x_gradient".
        n_shuffles: Number of shuffled baselines for DeepLiftShap.
        hypothetical: Return hypothetical attributions for all bases
                      (DeepLiftShap only).

    Returns:
        JSON with "attributions" array (shape N×4×L), "shape", and method info.
    """
    try:
        from captum.attr import DeepLiftShap, InputXGradient
        from tangermeme.utils import random_one_hot
        from tangermeme.deep_lift_shap import hypothetical_attributions
    except ImportError:
        return json.dumps({
            "error": "captum and tangermeme required. "
                     "Install with: pip install 'tfbs-nn[viz]'"
        })

    _ensure_imports()
    _validate_dna(sequences)

    entry = _get_model(model_id)
    model = entry["model"]
    device = entry["device"]

    X = _sequences_to_tensor(sequences).to(device)
    N, C, L = X.shape

    if method == "input_x_gradient":
        model.train()  # cuDNN RNN backward fix
        try:
            ixg = InputXGradient(model)
            attr = ixg.attribute(X, target=0)
        finally:
            model.eval()
        attr_np = attr.detach().cpu().numpy()
        return json.dumps({
            "attributions": attr_np.tolist(),
            "shape": list(attr_np.shape),
            "method": "input_x_gradient",
        })

    # DeepLiftShap
    max_pairs = 256
    result_sum = _torch.zeros((N, C, L), dtype=X.dtype, device="cpu")

    i = 0
    while i < N:
        batch_N = min(N - i, max_pairs)
        shuffles_per_call = max(1, max_pairs // batch_N)
        shuffles_per_call = min(shuffles_per_call, n_shuffles)
        X_batch = X[i : i + batch_N]

        s0 = 0
        while s0 < n_shuffles:
            s = min(shuffles_per_call, n_shuffles - s0)
            baselines = random_one_hot((s, C, L)).float().to(device)
            baselines_rep = _torch.cat([baselines] * batch_N, dim=0)
            X_rep = X_batch.repeat_interleave(s, dim=0)

            custom_fn = None if hypothetical else hypothetical_attributions

            model.train()  # cuDNN RNN backward fix
            try:
                dl = DeepLiftShap(model)
                a = dl.attribute(
                    X_rep, baselines=baselines_rep, target=0,
                    return_convergence_delta=False,
                    custom_attribution_func=custom_fn,
                )
            finally:
                model.eval()

            if a.dim() == 3 and a.shape[0] == batch_N * s:
                chunk_sum = a.reshape(batch_N, s, C, L).sum(dim=1)
            elif a.dim() == 3 and a.shape == (batch_N, C, L):
                chunk_sum = a * s
            else:
                chunk_sum = a.reshape(batch_N, s, C, L).sum(dim=1)

            result_sum[i : i + batch_N] += chunk_sum.cpu().detach()
            s0 += s
        i += batch_N

    avg = result_sum / float(n_shuffles)
    if not hypothetical:
        avg = avg.to(device) * X
    attr_np = avg.detach().cpu().numpy()

    return json.dumps({
        "attributions": attr_np.tolist(),
        "shape": list(attr_np.shape),
        "method": "deepliftshap",
        "n_shuffles": n_shuffles,
        "hypothetical": hypothetical,
    })


# ---------------------------------------------------------------------------
# Tool 5: tfbs_mutagenesis (in-silico saturation mutagenesis)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_mutagenesis",
    annotations={
        "title": "Saturation Mutagenesis",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_mutagenesis(
    model_id: str,
    sequences: list[str],
    start: int = 0,
    end: int = -1,
    raw_outputs: bool = False,
    batch_size: int | None = None,
) -> str:
    """Run in-silico saturation mutagenesis (ISM) on DNA sequences.

    For every position in each sequence, substitutes each of the 4 bases
    and records the change in model prediction.  This reveals which
    positions and substitutions most affect the score.

    Requires tangermeme: pip install 'tfbs-nn[viz]'

    Args:
        model_id: Identifier of a model loaded via tfbs_load_model.
        sequences: DNA sequences to mutagenize (A/C/G/T).
        start: First position to mutagenize (0-indexed, default 0).
        end: Last position (exclusive, -1 = full length).
        raw_outputs: If true, return original and per-mutant predictions
                     instead of aggregated attribution scores.
        batch_size: Batch size for predictions.  Auto-detected if omitted.

    Returns:
        JSON with "scores" (shape N x 4 x L attribution matrix) and "shape",
        or with raw_outputs=true: "original_scores" and "mutant_scores".
    """
    try:
        from tangermeme.ism import saturation_mutagenesis
    except ImportError:
        return json.dumps({
            "error": "tangermeme required. Install with: pip install 'tfbs-nn[viz]'"
        })

    _ensure_imports()
    _validate_dna(sequences)

    if batch_size is None:
        batch_size = _ENV["default_batch_size"]

    entry = _get_model(model_id)
    model = entry["model"]
    device = entry["device"]

    X = _sequences_to_tensor(sequences)  # keep on CPU, tangermeme handles device

    result = saturation_mutagenesis(
        model, X,
        start=start, end=end,
        batch_size=batch_size,
        raw_outputs=raw_outputs,
        device=device,
    )

    if raw_outputs:
        # result is (y_orig, y_mutant) tuple
        y_orig, y_mut = result
        return json.dumps({
            "original_scores": y_orig.detach().cpu().numpy().tolist(),
            "mutant_scores": y_mut.detach().cpu().numpy().tolist(),
        })

    attr_np = result.detach().cpu().numpy()
    return json.dumps({
        "scores": attr_np.tolist(),
        "shape": list(attr_np.shape),
        "method": "ism",
    })


# ---------------------------------------------------------------------------
# Tool 6: tfbs_marginalize (motif marginalization)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_marginalize",
    annotations={
        "title": "Motif Marginalization",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_marginalize(
    model_id: str,
    sequences: list[str],
    motif: str,
    start: int | None = None,
    n_background: int | None = None,
) -> str:
    """Measure the marginal effect of inserting a motif into sequences.

    Substitutes the motif consensus into each background sequence and
    reports the change in model prediction.  This isolates the effect of
    a single motif on the model's output.

    If sequences are provided, they are used as backgrounds.  Otherwise
    set n_background to generate random one-hot backgrounds of the
    model's input length.

    Requires tangermeme: pip install 'tfbs-nn[viz]'

    Args:
        model_id: Identifier of a model loaded via tfbs_load_model.
        sequences: Background DNA sequences (A/C/G/T).
        motif: DNA motif consensus to insert (e.g. "GATAAG").
        start: Position to insert motif.  None = center of sequence.
        n_background: If set, ignore sequences and generate this many
                      random one-hot backgrounds instead.

    Returns:
        JSON with per-sequence "delta_predictions" (after - before),
        "mean_delta", "predictions_before", and "predictions_after".
    """
    try:
        from tangermeme.marginalize import marginalize
        from tangermeme.utils import one_hot_encode, random_one_hot
    except ImportError:
        return json.dumps({
            "error": "tangermeme required. Install with: pip install 'tfbs-nn[viz]'"
        })

    _ensure_imports()

    entry = _get_model(model_id)
    model = entry["model"]
    device = entry["device"]
    input_length = entry["input_length"]

    # Build background tensor
    if n_background is not None and input_length is not None:
        X = random_one_hot((n_background, 4, input_length)).float()
    else:
        _validate_dna(sequences)
        X = _sequences_to_tensor(sequences)

    X = X.to(device)

    # One-hot encode the motif string
    motif_ohe = one_hot_encode(motif).unsqueeze(0).float().to(device)

    y_before, y_after = marginalize(
        model, X, motif_ohe,
        start=start,
        device=str(device),
    )

    # Coerce to numpy
    if isinstance(y_before, _torch.Tensor):
        y_before = y_before.detach().cpu().numpy()
    else:
        y_before = _np.asarray(y_before)
    if isinstance(y_after, _torch.Tensor):
        y_after = y_after.detach().cpu().numpy()
    else:
        y_after = _np.asarray(y_after)

    delta = (y_after - y_before).squeeze()

    return json.dumps({
        "predictions_before": y_before.squeeze().tolist(),
        "predictions_after": y_after.squeeze().tolist(),
        "delta_predictions": delta.tolist(),
        "mean_delta": float(delta.mean()),
        "motif": motif,
    })


# ---------------------------------------------------------------------------
# Tool 7: tfbs_classify_metrics (ROC / PR AUC)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_classify_metrics",
    annotations={
        "title": "Classification Metrics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_classify_metrics(
    positive_scores: list[float],
    negative_scores: list[float],
    subsample_negatives: bool = True,
) -> str:
    """Compute ROC-AUC and PR-AUC from positive and negative score lists.

    A generic binary classification evaluator: given prediction scores
    for known-positive and known-negative examples, computes ROC and
    Precision-Recall area under the curve.

    Args:
        positive_scores: Model scores for positive (binding) examples.
        negative_scores: Model scores for negative (non-binding) examples.
        subsample_negatives: If true and negatives outnumber positives,
            randomly subsample negatives to match for balanced ROC
            (default true).

    Returns:
        JSON with "roc_auc", "pr_auc", "num_positive", "num_negative",
        and the ROC curve points ("fpr", "tpr") and PR curve points
        ("precision", "recall").
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    _ensure_imports()

    if not positive_scores or not negative_scores:
        return json.dumps({
            "error": "Both positive_scores and negative_scores must be "
                     "non-empty lists."
        })

    pos = _np.array(positive_scores, dtype=float)
    neg = _np.array(negative_scores, dtype=float)

    if subsample_negatives and len(neg) > len(pos):
        rng = _np.random.RandomState(42)
        neg = rng.choice(neg, size=len(pos), replace=False)

    y_true = _np.concatenate([_np.ones(len(pos)), _np.zeros(len(neg))])
    y_score = _np.concatenate([pos, neg])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc_val = float(auc(fpr, tpr))

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc_val = float(auc(recall, precision))

    return json.dumps({
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
        "num_positive": int(len(pos)),
        "num_negative": int(len(neg)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
    })


# ---------------------------------------------------------------------------
# Tool 8: tfbs_fimo
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_fimo",
    annotations={
        "title": "FIMO Motif Scan",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tfbs_fimo(
    sequences: list[str],
    motif_file: str,
    threshold: float = 1e-4,
    output_dir: str | None = None,
) -> str:
    """Run FIMO motif scanning on DNA sequences.

    Writes sequences to a temporary FASTA file, invokes the FIMO binary
    from the MEME Suite, and returns significant motif hits.

    Requires the ``fimo`` binary on PATH (from MEME Suite).

    Args:
        sequences: DNA sequences to scan (A/C/G/T).
        motif_file: Absolute path to a MEME-format motif file.
        threshold: FIMO p-value threshold (default 1e-4).
        output_dir: Directory for FIMO output.  Uses a temp dir if omitted.

    Returns:
        JSON with "hits" (up to 500), "total_hits", and "output_dir".
    """
    import pandas as pd

    motif_path = Path(motif_file).resolve()
    if not motif_path.exists():
        return json.dumps({"error": f"Motif file not found: {motif_file}"})
    # Basic path validation — motif_file is passed to subprocess
    if not motif_path.is_file():
        return json.dumps({"error": f"Not a regular file: {motif_file}"})

    _validate_dna(sequences)

    # Write sequences to a temp FASTA
    work_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="tfbs_fimo_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = work_dir / "input.fasta"
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq{i}\n{seq}\n")

    fimo_out = work_dir / "fimo_out"

    cmd = [
        "fimo",
        "--thresh", str(float(threshold)),
        "--oc", str(fimo_out),
        "--max-strand",
        "--max-stored-scores", "2147483646",
        str(motif_path),
        str(fasta_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return json.dumps({
            "error": f"FIMO failed (exit {result.returncode})",
            "stderr": result.stderr[:1000],
        })

    # Parse results
    tsv_path = fimo_out / "fimo.tsv"
    if not tsv_path.exists():
        return json.dumps({"error": "FIMO produced no output", "stderr": result.stderr[:500]})

    df = pd.read_csv(tsv_path, sep="\t", comment="#")
    hits = df.to_dict(orient="records")

    return json.dumps({
        "hits": hits[:500],  # cap to avoid huge responses
        "total_hits": len(hits),
        "output_dir": str(fimo_out),
    })


# ---------------------------------------------------------------------------
# Tool 9: tfbs_list_models (utility)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_list_models",
    annotations={
        "title": "List Loaded Models",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_list_models() -> str:
    """List all TFBS models currently loaded in the server.

    Returns:
        JSON object keyed by model_id with model_type, device,
        input_length, and classify flag for each loaded model.
        Returns an empty object if no models are loaded.
    """
    info = {}
    for mid, entry in _model_registry.items():
        info[mid] = {
            "model_type": entry["model_type"],
            "device": entry["device"],
            "input_length": entry["input_length"],
            "classify": entry["classify"],
        }
    return json.dumps(info)


# ---------------------------------------------------------------------------
# Tool 10: tfbs_server_info (environment / diagnostics)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_server_info",
    annotations={
        "title": "Server Environment Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_server_info() -> str:
    """Show the server's runtime environment and auto-detected defaults.

    Reports whether the server is running inside a SLURM job, how many
    GPUs are visible, what default device and batch size are being used,
    and (after the first tool call that triggers imports) the torch and
    CUDA versions.

    Returns:
        JSON with on_slurm, slurm_job_id, slurm_partition, slurm_gpus,
        default_device, default_batch_size, torch_version, cuda_available,
        and cuda_device_name.
    """
    info = dict(_ENV)  # shallow copy

    # Add torch details if already loaded
    if _torch is not None:
        info["torch_version"] = _torch.__version__
        info["cuda_available"] = _torch.cuda.is_available()
        if _torch.cuda.is_available():
            info["cuda_device_name"] = _torch.cuda.get_device_name(0)
            info["cuda_device_count"] = _torch.cuda.device_count()
    else:
        info["torch_loaded"] = False

    info["loaded_models"] = list(_model_registry.keys())

    return json.dumps(info)


# ---------------------------------------------------------------------------
# Tool 11: slurm_submit
# ---------------------------------------------------------------------------
@mcp.tool(
    name="slurm_submit",
    annotations={
        "title": "Submit SLURM Job",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def slurm_submit(
    script_path: str,
    args: list[str] | None = None,
    sbatch_flags: list[str] | None = None,
    working_dir: str | None = None,
) -> str:
    """Submit a batch job to the SLURM scheduler via sbatch.

    Builds and executes: ``sbatch [sbatch_flags...] script_path [args...]``

    Args:
        script_path: Absolute path to the sbatch script to submit.
        args: Extra arguments appended after the script path.
        sbatch_flags: Extra flags inserted before the script path
                      (e.g. ["--partition=gpu", "--gres=gpu:1"]).
        working_dir: Directory to run sbatch from.  Defaults to the
                     script's parent directory.

    Returns:
        JSON with job_id, submit_command, and any stderr from sbatch.
    """
    script = Path(script_path).resolve()
    if not script.is_file():
        return json.dumps({"error": f"Script not found or not a file: {script_path}"})

    if shutil.which("sbatch") is None:
        return json.dumps({"error": "sbatch binary not found on PATH"})

    cmd: list[str] = ["sbatch"]
    if sbatch_flags:
        cmd.extend(sbatch_flags)
    cmd.append(str(script))
    if args:
        cmd.extend(args)

    cwd = working_dir or str(script.parent)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=cwd)

    if result.returncode != 0:
        return json.dumps({
            "error": f"sbatch failed (exit {result.returncode})",
            "stderr": result.stderr.strip(),
            "submit_command": " ".join(cmd),
        })

    # Parse job ID from "Submitted batch job 12345"
    job_id = None
    for word in result.stdout.strip().split():
        if word.isdigit():
            job_id = word
            break

    return json.dumps({
        "job_id": job_id,
        "submit_command": " ".join(cmd),
        "stderr": result.stderr.strip() if result.stderr.strip() else None,
    })


# ---------------------------------------------------------------------------
# Tool 12: slurm_status
# ---------------------------------------------------------------------------
@mcp.tool(
    name="slurm_status",
    annotations={
        "title": "Check SLURM Job Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def slurm_status(
    job_id: str,
) -> str:
    """Check the current state of a SLURM job.

    Tries squeue first (for running/pending jobs), then falls back to
    sacct (for completed/failed jobs) to retrieve status information.

    Args:
        job_id: The SLURM job ID to query.

    Returns:
        JSON with job_id, state, node, elapsed, submit_time, and
        exit_code (exit_code only available from sacct).
    """
    # Try squeue first (running / pending jobs)
    try:
        sq = subprocess.run(
            ["squeue", "-j", job_id, "--noheader", "--format=%i|%T|%N|%M|%V"],
            capture_output=True, text=True, timeout=30,
        )
        line = sq.stdout.strip()
        if line:
            parts = line.split("|")
            return json.dumps({
                "job_id": parts[0].strip() if len(parts) > 0 else job_id,
                "state": parts[1].strip() if len(parts) > 1 else "UNKNOWN",
                "node": parts[2].strip() if len(parts) > 2 else None,
                "elapsed": parts[3].strip() if len(parts) > 3 else None,
                "submit_time": parts[4].strip() if len(parts) > 4 else None,
                "exit_code": None,
            })
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fall back to sacct (completed / historical jobs)
    try:
        sa = subprocess.run(
            [
                "sacct", "-j", job_id, "--noheader", "--parsable2",
                "--format=JobID,State,NodeList,Elapsed,Submit,ExitCode",
            ],
            capture_output=True, text=True, timeout=30,
        )
        for raw_line in sa.stdout.strip().splitlines():
            parts = raw_line.split("|")
            # Filter to main job line (skip .batch / .extern steps)
            if parts and parts[0].strip() == job_id:
                return json.dumps({
                    "job_id": parts[0].strip(),
                    "state": parts[1].strip() if len(parts) > 1 else "UNKNOWN",
                    "node": parts[2].strip() if len(parts) > 2 else None,
                    "elapsed": parts[3].strip() if len(parts) > 3 else None,
                    "submit_time": parts[4].strip() if len(parts) > 4 else None,
                    "exit_code": parts[5].strip() if len(parts) > 5 else None,
                })
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return json.dumps({"job_id": job_id, "state": "NOT_FOUND", "error": "Job not found in squeue or sacct"})


# ---------------------------------------------------------------------------
# Tool 13: slurm_logs
# ---------------------------------------------------------------------------
@mcp.tool(
    name="slurm_logs",
    annotations={
        "title": "Read SLURM Job Logs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def slurm_logs(
    job_id: str,
    log_type: str = "stderr",
    tail: int = 100,
) -> str:
    """Read stdout and/or stderr log files for a SLURM job.

    Uses sacct to locate the log file paths, then reads the last
    ``tail`` lines from each requested file.

    Args:
        job_id: The SLURM job ID whose logs to read.
        log_type: Which log to read: "stdout", "stderr", or "both".
        tail: Number of lines to read from the end of each log file.

    Returns:
        JSON with job_id, log_paths dict, and the content of requested
        log files.  Missing files are reported with an error message.
    """
    stdout_path = None
    stderr_path = None

    try:
        sa = subprocess.run(
            [
                "sacct", "-j", job_id, "--noheader", "--parsable2",
                "--format=JobID,WorkDir,StdOut,StdErr",
            ],
            capture_output=True, text=True, timeout=30,
        )
        for raw_line in sa.stdout.strip().splitlines():
            parts = raw_line.split("|")
            # Filter to main job line (skip .batch / .extern steps)
            if parts and parts[0].strip() == job_id:
                stdout_path = parts[2].strip() if len(parts) > 2 else None
                stderr_path = parts[3].strip() if len(parts) > 3 else None
                break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return json.dumps({"error": "sacct not available or timed out"})

    log_paths: dict[str, str | None] = {
        "stdout": stdout_path,
        "stderr": stderr_path,
    }
    result: dict[str, Any] = {"job_id": job_id, "log_paths": log_paths}

    if log_type in ("stdout", "both") and stdout_path:
        result["stdout"] = _tail_file(stdout_path, tail)
    if log_type in ("stderr", "both") and stderr_path:
        result["stderr"] = _tail_file(stderr_path, tail)

    # Handle case where paths were not found
    if log_type in ("stdout", "both") and not stdout_path:
        result["stdout"] = "[stdout path not found in sacct output]"
    if log_type in ("stderr", "both") and not stderr_path:
        result["stderr"] = "[stderr path not found in sacct output]"

    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 14: slurm_cancel
# ---------------------------------------------------------------------------
@mcp.tool(
    name="slurm_cancel",
    annotations={
        "title": "Cancel SLURM Job",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def slurm_cancel(
    job_id: str,
) -> str:
    """Cancel a running or pending SLURM job.

    Args:
        job_id: The SLURM job ID to cancel.

    Returns:
        JSON with job_id, cancelled (bool), and any stderr from scancel.
    """
    try:
        result = subprocess.run(
            ["scancel", job_id],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        return json.dumps({"error": "scancel binary not found on PATH"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "scancel timed out"})

    return json.dumps({
        "job_id": job_id,
        "cancelled": result.returncode == 0,
        "stderr": result.stderr.strip() if result.stderr.strip() else None,
    })


# ---------------------------------------------------------------------------
# Tool 15: slurm_watch
# ---------------------------------------------------------------------------
@mcp.tool(
    name="slurm_watch",
    annotations={
        "title": "Watch SLURM Job",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def slurm_watch(
    job_id: str,
    check_interval: str = "1h",
    context: str | None = None,
) -> str:
    """Request monitoring of a SLURM job.

    This is a placeholder that returns instructions for the LLM client
    to set up recurring status checks.  The MCP server itself cannot
    create Claude Code scheduled agents — the LLM acts on the returned
    instructions to configure monitoring via CronCreate or /schedule.

    Args:
        job_id: The SLURM job ID to monitor.
        check_interval: How often to check (e.g. "30m", "1h", "2h").
        context: Optional description of what the job is doing, to
                 include in status check messages.

    Returns:
        JSON with job_id, check_interval, context, and instructions
        for the LLM to set up scheduled monitoring.
    """
    instructions = (
        f"Set up a recurring check for SLURM job {job_id} every {check_interval}. "
        "Use CronCreate or /schedule to create a scheduled agent that: "
        f"1) Calls slurm_status with job_id='{job_id}' to get the current state. "
        f"2) Calls slurm_logs with job_id='{job_id}' and log_type='stderr' to check for errors. "
        "3) Reports the status to the user. "
        "4) If the job has completed (COMPLETED, FAILED, CANCELLED, TIMEOUT), "
        "report the final status and exit code, then cancel the scheduled check."
    )

    return json.dumps({
        "job_id": job_id,
        "check_interval": check_interval,
        "context": context,
        "instructions": instructions,
    })


# ---------------------------------------------------------------------------
# Tool 16: tfbs_train
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_train",
    annotations={
        "title": "Submit Training Job",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def tfbs_train(
    script: str | None = None,
    config_path: str | None = None,
    partition: str = "gpu",
    gpu_type: str | None = None,
    max_time: str = "48:00:00",
    job_name: str | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """Submit a TFBS model training job to SLURM.

    Two modes of operation:

    **Script mode** — provide ``script`` to submit an existing sbatch
    script directly.

    **Config mode** — provide ``config_path`` to auto-generate an
    sbatch script that invokes the Lightning CLI trainer with the
    given YAML config.

    Args:
        script: Path to an existing sbatch script (script mode).
        config_path: Path to a Lightning CLI YAML config (config mode).
        partition: SLURM partition (default "gpu").
        gpu_type: GPU type constraint (e.g. "a100", "v100").
        max_time: Maximum wall-clock time (default "48:00:00").
        job_name: SLURM job name.  Auto-generated if omitted.
        extra_args: Additional CLI arguments appended to the training
                    command (config mode only).

    Returns:
        JSON with mode, job_id, submit_command, and (for config mode)
        the generated sbatch script content.
    """
    # Validate: exactly one of script or config_path
    if script and config_path:
        return json.dumps({"error": "Provide exactly one of 'script' or 'config_path', not both."})
    if not script and not config_path:
        return json.dumps({"error": "Provide exactly one of 'script' or 'config_path'."})

    # Script mode
    if script:
        script_p = Path(script).resolve()
        if not script_p.is_file():
            return json.dumps({"error": f"Script not found: {script}"})
        result = await slurm_submit(script_path=str(script_p))
        result_dict = json.loads(result)
        result_dict["mode"] = "script"
        return json.dumps(result_dict)

    # Config mode
    config_p = Path(config_path).resolve()
    if not config_p.is_file():
        return json.dumps({"error": f"Config not found: {config_path}"})

    if job_name is None:
        job_name = f"tfbs_train_{config_p.stem}"

    gres_line = ""
    if gpu_type:
        gres_line = f"#SBATCH --gres=gpu:{gpu_type}:1"
    else:
        gres_line = "#SBATCH --gres=gpu:1"

    train_cmd = f"python /sc-projects/sc-proj-cc17-P09_TFBS/scripts/cli/train_cli.py fit --config {config_p}"
    if extra_args:
        train_cmd += " " + " ".join(extra_args)

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{gres_line}
#SBATCH --time={max_time}
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err
#SBATCH --ntasks=1

{train_cmd}
"""

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, prefix="tfbs_")
    tmp.write(sbatch_content)
    tmp.close()
    os.chmod(tmp.name, 0o755)

    result = await slurm_submit(script_path=tmp.name, working_dir=str(config_p.parent))
    result_dict = json.loads(result)
    result_dict["mode"] = "config"
    result_dict["generated_script"] = sbatch_content
    return json.dumps(result_dict)


# ---------------------------------------------------------------------------
# Tool 17: tfbs_sweep
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_sweep",
    annotations={
        "title": "Submit WandB Sweep",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def tfbs_sweep(
    sweep_config: str,
    project_name: str,
    model: str,
    data_path: str,
    data_name: str,
    scaling_method: str = "standardize",
    n_trials: int = 50,
    partition: str = "gpu",
    gpu_type: str | None = None,
) -> str:
    """Create a WandB sweep and submit an agent job to SLURM.

    Steps:
      1. Run ``wandb sweep`` to create the sweep and obtain a sweep ID.
      2. Generate a temporary sbatch script that launches a sweep agent.
      3. Submit the agent job via sbatch.

    Args:
        sweep_config: Path to a WandB sweep YAML config file.
        project_name: WandB project name for the sweep.
        model: Model architecture name (e.g. "VCNNBpnet").
        data_path: Path to the training data directory.
        data_name: Dataset name (e.g. "all_mean").
        scaling_method: Label scaling method (default "standardize").
        n_trials: Number of sweep trials to run (default 50).
        partition: SLURM partition (default "gpu").
        gpu_type: GPU type constraint (e.g. "a100").

    Returns:
        JSON with sweep_id, job_id, and submit_command.
    """
    sweep_p = Path(sweep_config).resolve()
    if not sweep_p.is_file():
        return json.dumps({"error": f"Sweep config not found: {sweep_config}"})

    if shutil.which("wandb") is None:
        return json.dumps({"error": "wandb binary not found on PATH"})

    # Step 1: Create the sweep
    try:
        sw = subprocess.run(
            ["wandb", "sweep", str(sweep_p), "--project", project_name],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "wandb sweep creation timed out"})

    if sw.returncode != 0:
        return json.dumps({
            "error": f"wandb sweep failed (exit {sw.returncode})",
            "stderr": sw.stderr.strip(),
        })

    # Parse sweep ID from output (check both stdout and stderr)
    sweep_id = None
    for line in (sw.stdout + "\n" + sw.stderr).splitlines():
        if "ID:" in line:
            # Formats: "Creating sweep with ID: xxx" or "wandb: Created sweep with ID: xxx"
            sweep_id = line.split("ID:")[-1].strip()
            break

    if not sweep_id:
        return json.dumps({
            "error": "Could not parse sweep ID from wandb output",
            "stdout": sw.stdout.strip(),
            "stderr": sw.stderr.strip(),
        })

    # Step 2: Build sbatch script for the sweep agent
    gres_line = ""
    if gpu_type:
        gres_line = f"#SBATCH --gres=gpu:{gpu_type}:1"
    else:
        gres_line = "#SBATCH --gres=gpu:1"

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=sweep_{sweep_id}
#SBATCH --partition={partition}
{gres_line}
#SBATCH --time=48:00:00
#SBATCH --output=sweep_{sweep_id}_%j.out
#SBATCH --error=sweep_{sweep_id}_%j.err
#SBATCH --ntasks=1

python scripts/cli/run_sweep.py \\
    --model {model} \\
    --data-path {data_path} \\
    --data-name {data_name} \\
    --scaling-method {scaling_method} \\
    --n-trials {n_trials} \\
    --sweep-id {sweep_id} \\
    --project-name {project_name}
"""

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, prefix="tfbs_")
    tmp.write(sbatch_content)
    tmp.close()
    os.chmod(tmp.name, 0o755)

    # Step 3: Submit via sbatch
    result = await slurm_submit(script_path=tmp.name)
    result_dict = json.loads(result)
    result_dict["sweep_id"] = sweep_id
    return json.dumps(result_dict)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    mcp.run()


if __name__ == "__main__":
    main()
