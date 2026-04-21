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
# Lazy imports — genomics (pyfaidx, tangermeme.io)
# ---------------------------------------------------------------------------
_pyfaidx = None
_extract_loci = None
_pd = None
_genome_cache: dict[str, Any] = {}


def _ensure_genomics_imports():
    """Import genomics dependencies on first use."""
    global _pyfaidx, _extract_loci, _pd
    if _pyfaidx is not None:
        return
    import pyfaidx
    import pandas as pd
    from tangermeme.io import extract_loci

    _pyfaidx = pyfaidx
    _extract_loci = extract_loci
    _pd = pd


def _get_genome(genome_fasta: str):
    """Load a pyfaidx.Fasta genome handle, reopening if stale.

    pyfaidx file handles can close between async calls.  We detect
    this via a probe read and reopen when necessary.
    """
    _ensure_genomics_imports()
    path = str(Path(genome_fasta).resolve())
    genome = _genome_cache.get(path)
    if genome is not None:
        try:
            # Probe: attempt a minimal read to verify handle is alive
            first_chrom = next(iter(genome.keys()))
            _ = genome[first_chrom][0:1]
        except (ValueError, StopIteration):
            # File handle closed — reopen
            genome = None
    if genome is None:
        genome = _pyfaidx.Fasta(path)
        _genome_cache[path] = genome
    return genome


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
    data_path: str | None = None,
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
        data_path: Optional path to the training data directory.  If
                   provided, loads scaling_params.json so predictions
                   can be inverse-transformed to real-world units.

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

    # Load scaling parameters if available
    scaling_params = None
    if data_path:
        sp_path = Path(data_path) / "scaling_params.json"
        if sp_path.exists():
            with open(sp_path) as f:
                scaling_params = json.load(f)

    _model_registry[model_id] = {
        "model": model,
        "model_type": model_type,
        "device": device,
        "input_length": input_length,
        "classify": classify,
        "scaling_params": scaling_params,
    }

    return json.dumps({
        "model_id": model_id,
        "model_type": model_type,
        "input_length": input_length,
        "classify": classify,
        "device": device,
        "scaling_params": scaling_params,
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
    inverse_transform: bool = False,
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
        inverse_transform: If true, reverse any label scaling that was
                           applied during training (requires scaling_params
                           to have been loaded via data_path in
                           tfbs_load_model).  Only affects regression
                           models.

    Returns:
        JSON with "scores" (normal) or "best_scores", "best_sequences",
        and "mean_scores" (window mode).  If inverse_transform is true,
        scores are in the original label space.
    """
    _ensure_imports()
    _validate_dna(sequences)

    if batch_size is None:
        batch_size = _ENV["default_batch_size"]

    entry = _get_model(model_id)
    model = entry["model"]
    device = entry["device"]

    X = _sequences_to_tensor(sequences).to(device)

    # Helper: apply inverse scaling to numpy scores
    def _inverse(scores_np):
        if not inverse_transform:
            return scores_np
        sp = entry.get("scaling_params")
        if sp is None:
            return scores_np  # no scaling params available
        if sp.get("scaling_method") == "standardize":
            return scores_np * sp["std"] + sp["mean"]
        if sp.get("scaling_method") == "normalize":
            return scores_np * (sp["max"] - sp["min"]) + sp["min"]
        return scores_np

    if window_mode:
        best_windows, best_scores, _, mean_scores = _find_best_windows_fast(
            X, model, window_size=window_size,
            processing_batch_size=batch_size, device=device,
        )
        best_seqs = _tensor_to_sequences(_torch.from_numpy(best_windows))
        result = {
            "best_scores": _inverse(best_scores).tolist(),
            "best_sequences": best_seqs,
            "mean_scores": _inverse(mean_scores).tolist(),
        }
        if inverse_transform and entry.get("scaling_params"):
            result["inverse_transformed"] = True
        return json.dumps(result)

    with _torch.no_grad():
        scores = model(X).cpu().numpy()

    scores = _inverse(scores)
    result = {"scores": scores.squeeze().tolist()}
    if inverse_transform and entry.get("scaling_params"):
        result["inverse_transformed"] = True
    return json.dumps(result)


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
# Tool 18: tfbs_validate (model validation on test/val set)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_validate",
    annotations={
        "title": "Submit Validation Job",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def tfbs_validate(
    checkpoint_path: str,
    data_path: str,
    model_name: str = "VCNNBpnet",
    scaling_method: str = "standardize",
    output_file: str | None = None,
    partition: str = "gpu",
    gpu_type: str | None = None,
    max_time: str = "02:00:00",
    job_name: str | None = None,
) -> str:
    """Submit a model validation job to SLURM.

    Runs scripts/validate.py which loads a checkpoint, runs
    trainer.validate() on the validation set, and appends the loss
    to a CSV results file.

    Args:
        checkpoint_path: Path to the .ckpt model checkpoint.
        data_path: Path to the dataset (tensor directory or CSV).
        model_name: Model class name (e.g. "VCNNBpnet", "RNN").
        scaling_method: Label scaling ("standardize", "normalize", "none").
        output_file: CSV file to append results to.  Defaults to
                     results/validation_results.csv in the project dir.
        partition: SLURM partition (default "gpu").
        gpu_type: GPU gres string (e.g. "nvidia_a100_80gb_pcie:1").
                  If omitted, requests 1 generic GPU.
        max_time: SLURM wall time (default "02:00:00").
        job_name: SLURM job name (default "tfbs_validate").

    Returns:
        JSON with job_id, the generated sbatch script, and the
        validation command.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    validate_script = project_root / "scripts" / "validate.py"

    if not validate_script.exists():
        return json.dumps({"error": f"validate.py not found at {validate_script}"})

    ckpt = Path(checkpoint_path).resolve()
    if not ckpt.exists():
        return json.dumps({"error": f"Checkpoint not found: {checkpoint_path}"})

    data = Path(data_path).resolve()
    if not data.exists():
        return json.dumps({"error": f"Data path not found: {data_path}"})

    if output_file is None:
        output_file = str(project_root / "results" / "validation_results.csv")

    gres = f"gpu:{gpu_type}" if gpu_type else "gpu:1"
    name = job_name or "tfbs_validate"

    validate_cmd = (
        f"python {validate_script}"
        f" --checkpoint_path {ckpt}"
        f" --model_name {model_name}"
        f" --data_path {data}"
        f" --scaling_method {scaling_method}"
        f" --output_file {output_file}"
    )

    sbatch_script = (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name={name}\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --cpus-per-task=8\n"
        f"#SBATCH --mem=64G\n"
        f"#SBATCH --gres={gres}\n"
        f"#SBATCH --time={max_time}\n"
        f"#SBATCH --output=outs/validate.o%j\n"
        f"#SBATCH --error=outs/validate.e%j\n"
        f"\ndate\n"
        f"{validate_cmd}\n"
        f"date\n"
    )

    import os as _os
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="tfbs_validate_",
    )
    tmp.write(sbatch_script)
    tmp.close()
    _os.chmod(tmp.name, 0o755)

    # Ensure outs/ directory exists
    outs_dir = project_root / "scripts" / "cli" / "outs"
    outs_dir.mkdir(parents=True, exist_ok=True)

    submit_result = json.loads(await slurm_submit(
        script_path=tmp.name,
        working_dir=str(project_root / "scripts" / "cli"),
    ))

    submit_result["mode"] = "validate"
    submit_result["validate_command"] = validate_cmd
    submit_result["sbatch_script"] = sbatch_script
    return json.dumps(submit_result)


# ---------------------------------------------------------------------------
# Tool 19: tfbs_config (generate LightningCLI YAML config)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_config",
    annotations={
        "title": "Generate Training Config",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_config(
    model_type: str,
    data_path: str,
    output_path: str,
    input_length: int = 24,
    scaling_method: str = "standardize",
    batch_size: int = 64,
    max_epochs: int = 500,
    patience: int = 5,
    learning_rate: float = 0.001,
    classify: bool = False,
    wandb_project: str | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_filename: str | None = None,
    model_params: dict | None = None,
) -> str:
    """Generate a LightningCLI YAML config file for training.

    Creates a complete config with model, data, and trainer sections
    that can be passed directly to tfbs_train.

    Supported model types: VCNNBpnet, RNN.  Model-specific
    hyperparameters are passed via model_params dict.

    VCNNBpnet model_params keys:
      num_channels (int, default 64), kernel_size (int, default 21),
      dilations (list[int], default [1,1,2,4,8]),
      pool_output_size (int, default 128),
      dense_sizes (list[int], default [128,64])

    RNN model_params keys:
      conv_out_channels (int, default 100), kernel_size (int, default 5),
      pool_size (int, default 2), dropout_conv (float, default 0.3),
      gru_hidden_size (int, default 100), gru_num_layers (int, default 1),
      bidirectional (bool, default true),
      dense_size (int, default 100), dropout_fc (float, default 0.6)

    Args:
        model_type: Architecture name ("VCNNBpnet" or "RNN").
        data_path: Path to dataset (tensor directory or CSV).
        output_path: Where to write the YAML config file.
        input_length: Sequence length (default 24).
        scaling_method: Label scaling ("standardize", "normalize", "none").
        batch_size: Training batch size (default 64).
        max_epochs: Maximum training epochs (default 500).
        patience: Early stopping patience (default 5).
        learning_rate: Optimizer learning rate (default 0.001).
        classify: If true, use BCE loss; else MSE (default false).
        wandb_project: WandB project name for logging.  If omitted,
                       no WandB logger is configured.
        checkpoint_dir: Directory to save checkpoints.  Defaults to
                        saved_models/<data_name>/<scaling_method>.
        checkpoint_filename: Checkpoint filename (default "best_<model_type>").
        model_params: Dict of model-specific hyperparameters (see above).

    Returns:
        JSON with "output_path", "config_preview" (first 40 lines),
        and "model_type".
    """
    import yaml

    valid_models = {"VCNNBpnet", "RNN"}
    if model_type not in valid_models:
        return json.dumps({
            "error": f"Unknown model_type '{model_type}'. Must be one of: {sorted(valid_models)}"
        })

    data = Path(data_path).resolve()
    if not data.exists():
        return json.dumps({"error": f"Data path not found: {data_path}"})

    project_root = Path(__file__).resolve().parent.parent.parent
    params = model_params or {}

    # --- Model section ---
    base_args = {
        "input_length": input_length,
        "learning_rate": learning_rate,
        "classify": classify,
        "input_channels": 4,
    }

    if model_type == "VCNNBpnet":
        model_args = {
            "num_channels": params.get("num_channels", 64),
            "kernel_size": params.get("kernel_size", 21),
            "dilations": params.get("dilations", [1, 1, 2, 4, 8]),
            "pool_output_size": params.get("pool_output_size", 128),
            "dense_sizes": params.get("dense_sizes", [128, 64]),
        }
    elif model_type == "RNN":
        model_args = {
            "conv_out_channels": params.get("conv_out_channels", 100),
            "kernel_size": params.get("kernel_size", 5),
            "pool_size": params.get("pool_size", 2),
            "dropout_conv": params.get("dropout_conv", 0.3),
            "gru_hidden_size": params.get("gru_hidden_size", 100),
            "gru_num_layers": params.get("gru_num_layers", 1),
            "bidirectional": params.get("bidirectional", True),
            "dense_size": params.get("dense_size", 100),
            "dropout_fc": params.get("dropout_fc", 0.6),
        }

    model_args.update(base_args)

    # --- Trainer section ---
    data_name = data.name
    ckpt_dir = checkpoint_dir or str(
        project_root / "saved_models" / data_name / scaling_method
    )
    ckpt_filename = checkpoint_filename or f"best_{model_type}"

    callbacks = [
        {
            "class_path": "pytorch_lightning.callbacks.EarlyStopping",
            "init_args": {
                "monitor": "val_loss",
                "patience": patience,
                "mode": "min",
            },
        },
        {
            "class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
            "init_args": {
                "dirpath": ckpt_dir,
                "filename": ckpt_filename,
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": 1,
            },
        },
    ]

    trainer: dict = {
        "accelerator": "auto",
        "devices": "auto",
        "max_epochs": max_epochs,
        "log_every_n_steps": 10,
        "callbacks": callbacks,
    }

    if wandb_project:
        trainer["logger"] = [
            {
                "class_path": "pytorch_lightning.loggers.WandbLogger",
                "init_args": {
                    "project": wandb_project,
                    "log_model": False,
                },
            }
        ]

    # --- Assemble full config ---
    config = {
        "seed_everything": 42,
        "trainer": trainer,
        "model": {
            "class_path": f"tfbs.nn.models.{model_type}",
            "init_args": model_args,
        },
        "data": {
            "data_path": str(data),
            "batch_size": batch_size,
            "num_workers": 24,
            "preprocess": False,
            "scaling_method": scaling_method,
        },
    }

    # --- Write file ---
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Read back for preview
    with open(out) as f:
        lines = f.readlines()
    preview = "".join(lines[:40])

    return json.dumps({
        "output_path": str(out),
        "model_type": model_type,
        "config_preview": preview,
    })


# ===========================================================================
# Pipeline tools (20-23): BED → tensors → train/val/test splits
# ===========================================================================


# ---------------------------------------------------------------------------
# Tool 20: tfbs_extract_loci (BED + genome → one-hot tensor)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_extract_loci",
    annotations={
        "title": "Extract Loci from BED",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_extract_loci(
    bed_path: str,
    output_dir: str,
    genome_fasta: str = "/sc-projects/sc-proj-btg/P09/data/genomes/hg38/hg38.fa",
    in_window: int = 200,
    max_sequences: int | None = None,
    score_column: int | None = 4,
) -> str:
    """Extract DNA sequences from a reference genome at BED peak positions.

    Reads a BED file, centers a window on each peak midpoint, extracts the
    DNA sequence from the genome, one-hot encodes it, and filters out
    sequences containing ambiguous bases (N).  Optionally extracts a
    continuous score column for regression tasks.

    Supports standard BED (3+ columns) and ChIP-Atlas aggregated BED
    format (9 columns with MACS2 score in column 5).

    Requires pyfaidx: pip install 'tfbs-nn[genomics]'

    Args:
        bed_path: Path to a BED file (tab-separated, at least 3 columns).
        output_dir: Directory to save sequences.pt (and scores.pt if scores
                    are extracted).
        genome_fasta: Path to reference genome FASTA file.
        in_window: Window size in bp centered on peak midpoints (default 200).
        max_sequences: If set, randomly sample this many peaks before
                       extraction (useful for large ChIP-Atlas files).
        score_column: 0-indexed BED column to extract as continuous scores
                      (default 4 = column 5).  Set to None to skip.

    Returns:
        JSON with shape, num_peaks_input, num_peaks_extracted,
        num_filtered_n, output_dir, and has_scores.
    """
    _ensure_imports()
    _ensure_genomics_imports()

    bed = Path(bed_path).resolve()
    if not bed.exists():
        return json.dumps({"error": f"BED file not found: {bed_path}"})

    genome_path = Path(genome_fasta).resolve()
    if not genome_path.exists():
        return json.dumps({"error": f"Genome FASTA not found: {genome_fasta}"})

    # Read BED file
    bed_df = _pd.read_csv(bed, sep="\t", header=None, comment="#")
    num_input = len(bed_df)

    peaks_df = _pd.DataFrame({
        "chrom": bed_df.iloc[:, 0],
        "start": bed_df.iloc[:, 1].astype(int),
        "end": bed_df.iloc[:, 2].astype(int),
    })

    # Extract scores if requested
    scores_raw = None
    if score_column is not None and bed_df.shape[1] > score_column:
        try:
            scores_raw = bed_df.iloc[:, score_column].astype(float).values
        except (ValueError, TypeError):
            scores_raw = None

    # Optional subsampling for large files
    if max_sequences and len(peaks_df) > max_sequences:
        idx = _np.random.RandomState(42).choice(
            len(peaks_df), size=max_sequences, replace=False,
        )
        idx.sort()
        peaks_df = peaks_df.iloc[idx].reset_index(drop=True)
        if scores_raw is not None:
            scores_raw = scores_raw[idx]

    # Load genome
    genome = _get_genome(str(genome_path))
    chrom_lengths = {name: len(seq) for name, seq in genome.items()}

    # Pre-filter off-chromosome peaks
    in_width = in_window // 2
    valid_mask = []
    for _, row in peaks_df.iterrows():
        chrom = row["chrom"]
        if chrom not in chrom_lengths:
            valid_mask.append(False)
            continue
        mid = row["start"] + (row["end"] - row["start"]) // 2
        seq_start = mid - in_width
        seq_end = mid + in_width + (in_window % 2)
        valid_mask.append(seq_start >= 0 and seq_end < chrom_lengths[chrom])

    valid_mask = _np.array(valid_mask)
    peaks_df = peaks_df[valid_mask].reset_index(drop=True)
    if scores_raw is not None:
        scores_raw = scores_raw[valid_mask]

    if len(peaks_df) == 0:
        return json.dumps({
            "error": "No valid peaks remaining after filtering",
            "num_peaks_input": num_input,
        })

    # Extract sequences
    seqs_tensor = _extract_loci(
        peaks_df, genome, in_window=in_window, verbose=False,
    ).float()  # type: ignore[union-attr]

    # extract_loci may silently drop peaks — realign scores.
    # If counts don't match, truncate scores to tensor length (extract_loci
    # processes peaks in order, so the first N scores correspond).
    n_extracted = seqs_tensor.shape[0]
    if scores_raw is not None and len(scores_raw) != n_extracted:
        scores_raw = scores_raw[:n_extracted]

    # Filter N-containing sequences
    n_mask = seqs_tensor.sum(dim=(1, 2)) == seqs_tensor.shape[-1]
    num_filtered_n = int((~n_mask).sum())
    seqs_tensor = seqs_tensor[n_mask]
    if scores_raw is not None:
        scores_raw = scores_raw[n_mask.numpy()]
        # Drop any remaining NaN scores
        nan_mask = _np.isnan(scores_raw)
        if nan_mask.any():
            valid = ~nan_mask
            seqs_tensor = seqs_tensor[_torch.from_numpy(valid)]
            scores_raw = scores_raw[valid]

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _torch.save(seqs_tensor, out / "sequences.pt")

    has_scores = scores_raw is not None
    if has_scores:
        scores_t = _torch.tensor(scores_raw, dtype=_torch.float32).unsqueeze(1)
        _torch.save(scores_t, out / "scores.pt")

    return json.dumps({
        "shape": list(seqs_tensor.shape),
        "num_peaks_input": num_input,
        "num_peaks_extracted": int(seqs_tensor.shape[0]),
        "num_filtered_n": num_filtered_n,
        "output_dir": str(out),
        "has_scores": has_scores,
    })


# ---------------------------------------------------------------------------
# Tool 21: tfbs_prepare_dataset (split into train/val/test)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_prepare_dataset",
    annotations={
        "title": "Prepare Train/Val/Test Split",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_prepare_dataset(
    sequences_path: str,
    labels_path: str,
    output_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    add_rc: bool = False,
    random_seed: int = 42,
) -> str:
    """Split sequences and labels into train/val/test directories.

    Creates the directory structure expected by TFBSDataModule:
    {output_dir}/{train,val,test}/{seqs.pt, labels.pt}

    Args:
        sequences_path: Path to sequences.pt tensor (N, 4, L).
        labels_path: Path to labels.pt tensor (N, 1) or (N,).
        output_dir: Base directory for the split dataset.
        test_size: Fraction for test set (default 0.2).
        val_size: Fraction of remaining for validation (default 0.2).
        add_rc: Add reverse complement augmentation (doubles dataset).
        random_seed: Random seed for reproducible splits (default 42).

    Returns:
        JSON with output_dir, train_count, val_count, test_count,
        and label_stats.
    """
    from sklearn.model_selection import train_test_split

    _ensure_imports()

    seqs_p = Path(sequences_path).resolve()
    labels_p = Path(labels_path).resolve()
    if not seqs_p.exists():
        return json.dumps({"error": f"Sequences file not found: {sequences_path}"})
    if not labels_p.exists():
        return json.dumps({"error": f"Labels file not found: {labels_path}"})

    seqs = _torch.load(seqs_p, weights_only=True)
    labels = _torch.load(labels_p, weights_only=True)
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)

    n = len(seqs)
    if n < 3:
        return json.dumps({
            "error": f"Need at least 3 sequences for train/val/test split, got {n}."
        })

    # Split: first test, then val from remaining
    indices = _np.arange(n)
    n_test = max(1, int(n * test_size))
    n_val = max(1, int((n - n_test) * val_size))
    train_idx, test_idx = train_test_split(
        indices, test_size=n_test, shuffle=True, random_state=random_seed,
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=n_val, shuffle=True, random_state=random_seed,
    )

    def _save_split(split_name, idx):
        s = seqs[idx]
        l = labels[idx]
        if add_rc:
            rc = _torch.flip(s, [2])
            rc = _torch.index_select(rc, 1, _torch.tensor([3, 2, 1, 0], dtype=_torch.long))
            s = _torch.cat([s, rc], dim=0)
            l = _torch.cat([l, l], dim=0)
        split_dir = Path(output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        _torch.save(s, split_dir / "seqs.pt")
        _torch.save(l, split_dir / "labels.pt")
        return len(s)

    train_n = _save_split("train", train_idx)
    val_n = _save_split("val", val_idx)
    test_n = _save_split("test", test_idx)

    all_labels = labels.numpy().flatten()
    return json.dumps({
        "output_dir": str(Path(output_dir).resolve()),
        "train_count": train_n,
        "val_count": val_n,
        "test_count": test_n,
        "label_stats": {
            "min": float(all_labels.min()),
            "max": float(all_labels.max()),
            "mean": float(all_labels.mean()),
            "std": float(all_labels.std()),
        },
    })


# ---------------------------------------------------------------------------
# Tool 22: tfbs_prepare_classification (positive + negative → binary)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_prepare_classification",
    annotations={
        "title": "Prepare Classification Dataset",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_prepare_classification(
    positive_path: str,
    negative_path: str,
    output_dir: str,
    balance: bool = True,
    random_seed: int = 42,
) -> str:
    """Combine positive and negative sequence tensors into a binary dataset.

    Loads two sequence tensor files (e.g. from tfbs_extract_loci),
    assigns labels 1.0 (positive) and 0.0 (negative), optionally
    balances by subsampling the larger set, shuffles, and saves.

    Output (sequences.pt + labels.pt) feeds into tfbs_prepare_dataset.

    Args:
        positive_path: Path to positive sequences.pt (N_pos, 4, L).
        negative_path: Path to negative sequences.pt (N_neg, 4, L).
        output_dir: Directory to save combined sequences.pt and labels.pt.
        balance: Subsample larger set to match smaller (default true).
        random_seed: Random seed (default 42).

    Returns:
        JSON with output_dir, num_positive, num_negative, total.
    """
    _ensure_imports()

    pos_p = Path(positive_path).resolve()
    neg_p = Path(negative_path).resolve()
    if not pos_p.exists():
        return json.dumps({"error": f"Positive file not found: {positive_path}"})
    if not neg_p.exists():
        return json.dumps({"error": f"Negative file not found: {negative_path}"})

    pos = _torch.load(pos_p, weights_only=True)
    neg = _torch.load(neg_p, weights_only=True)

    rng = _np.random.RandomState(random_seed)

    if balance:
        n = min(len(pos), len(neg))
        if len(pos) > n:
            idx = rng.choice(len(pos), size=n, replace=False)
            pos = pos[idx]
        if len(neg) > n:
            idx = rng.choice(len(neg), size=n, replace=False)
            neg = neg[idx]

    n_pos, n_neg = len(pos), len(neg)
    seqs = _torch.cat([pos, neg], dim=0)
    labels = _torch.cat([
        _torch.ones(n_pos, 1),
        _torch.zeros(n_neg, 1),
    ], dim=0)

    # Shuffle
    perm = _torch.tensor(rng.permutation(len(seqs)))
    seqs = seqs[perm]
    labels = labels[perm]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _torch.save(seqs, out / "sequences.pt")
    _torch.save(labels, out / "labels.pt")

    return json.dumps({
        "output_dir": str(out.resolve()),
        "num_positive": n_pos,
        "num_negative": n_neg,
        "total": n_pos + n_neg,
    })


# ---------------------------------------------------------------------------
# Tool 23: tfbs_prepare_regression (sequences + scores → transformed)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_prepare_regression",
    annotations={
        "title": "Prepare Regression Dataset",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_prepare_regression(
    sequences_path: str,
    scores_path: str,
    output_dir: str,
    score_transform: str = "log2",
    min_score: float | None = None,
    random_seed: int = 42,
) -> str:
    """Prepare a regression dataset from sequences and continuous scores.

    Loads sequences and their associated scores (e.g. ChIP-seq signal
    values from BED column 5), optionally filters by minimum score,
    applies a transformation, and saves.

    Output (sequences.pt + labels.pt) feeds into tfbs_prepare_dataset.

    Args:
        sequences_path: Path to sequences.pt (N, 4, L).
        scores_path: Path to scores.pt (N, 1) from tfbs_extract_loci.
        output_dir: Directory to save sequences.pt and labels.pt.
        score_transform: Transform to apply: "log2" (log2(x+1)),
                         "log10" (log10(x+1)), "zscore", or "none".
        min_score: If set, filter out sequences with score below this.
        random_seed: Random seed (default 42).

    Returns:
        JSON with output_dir, num_sequences, score_stats_before,
        score_stats_after.
    """
    _ensure_imports()

    seqs_p = Path(sequences_path).resolve()
    scores_p = Path(scores_path).resolve()
    if not seqs_p.exists():
        return json.dumps({"error": f"Sequences file not found: {sequences_path}"})
    if not scores_p.exists():
        return json.dumps({"error": f"Scores file not found: {scores_path}"})

    seqs = _torch.load(seqs_p, weights_only=True)
    scores = _torch.load(scores_p, weights_only=True).float()
    if scores.dim() == 1:
        scores = scores.unsqueeze(1)

    def _stats(t):
        v = t.numpy().flatten()
        return {"min": float(v.min()), "max": float(v.max()),
                "mean": float(v.mean()), "std": float(v.std())}

    stats_before = _stats(scores)

    # Filter by minimum score
    if min_score is not None:
        mask = scores.squeeze() >= min_score
        seqs = seqs[mask]
        scores = scores[mask]

    # Apply transform
    valid_transforms = {"log2", "log10", "zscore", "none"}
    if score_transform not in valid_transforms:
        return json.dumps({
            "error": f"Unknown score_transform '{score_transform}'. "
                     f"Must be one of: {sorted(valid_transforms)}"
        })

    if score_transform == "log2":
        scores = _torch.log2(scores + 1)
    elif score_transform == "log10":
        scores = _torch.log10(scores + 1)
    elif score_transform == "zscore":
        mean = scores.mean()
        std = scores.std()
        scores = (scores - mean) / (std + 1e-8)

    stats_after = _stats(scores)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _torch.save(seqs, out / "sequences.pt")
    _torch.save(scores, out / "labels.pt")

    return json.dumps({
        "output_dir": str(out.resolve()),
        "num_sequences": int(seqs.shape[0]),
        "score_stats_before": stats_before,
        "score_stats_after": stats_after,
    })


# ===========================================================================
# Visualization tools (24-26): seqlets, epistasis, conv filters
# ===========================================================================


# ---------------------------------------------------------------------------
# Tool 24: tfbs_seqlets (motif seqlet discovery from attributions)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_seqlets",
    annotations={
        "title": "Discover Motif Seqlets",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_seqlets(
    model_id: str,
    sequences: list[str],
    n_shuffles: int = 20,
    min_seqlet_len: int = 5,
    max_seqlet_len: int = 15,
    motif_file: str | None = None,
) -> str:
    """Discover recurring motif patterns from attribution scores.

    Computes DeepLiftShap attributions, then finds seqlets (short
    subsequences with consistently high attribution) using tangermeme's
    recursive seqlet discovery.  Optionally annotates seqlets against
    a JASPAR/MEME motif database.

    Requires captum and tangermeme: pip install 'tfbs-nn[viz]'

    Args:
        model_id: Identifier of a loaded model.
        sequences: DNA sequences to analyze.
        n_shuffles: Number of shuffled baselines for DeepLiftShap.
        min_seqlet_len: Minimum seqlet length (default 5).
        max_seqlet_len: Maximum seqlet length (default 15).
        motif_file: Optional path to a MEME-format motif file for
                    annotation (e.g. JASPAR database).

    Returns:
        JSON with num_seqlets, seqlets list (example_idx, start, end,
        mean_attribution), and optionally top_motifs.
    """
    try:
        from tangermeme.seqlet import recursive_seqlets
        from tangermeme.deep_lift_shap import deep_lift_shap as tangermeme_dls
    except ImportError:
        return json.dumps({
            "error": "tangermeme required. Install with: pip install 'tfbs-nn[viz]'"
        })

    _ensure_imports()
    _validate_dna(sequences)

    entry = _get_model(model_id)
    model = entry["model"]
    device = entry["device"]

    X = _sequences_to_tensor(sequences).to(device)

    # Compute attributions
    model.train()
    try:
        attr = tangermeme_dls(model, X, n_shuffles=n_shuffles, device=str(device))
    finally:
        model.eval()

    if isinstance(attr, _torch.Tensor):
        attr_np = attr.detach().cpu().numpy()
    else:
        attr_np = _np.asarray(attr)

    # Sum across channels for seqlet discovery: (N, 4, L) → (N, L)
    attr_sum = attr_np.sum(axis=1)

    # Discover seqlets — recursive_seqlets can fail with too few sequences
    try:
        seqlets = recursive_seqlets(
            attr_sum,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
        )
    except (ZeroDivisionError, ValueError):
        seqlets = []

    seqlet_list = []
    for s in seqlets:
        seqlet_list.append({
            "example_idx": int(s[0]) if hasattr(s, '__getitem__') else 0,
            "start": int(s.start) if hasattr(s, 'start') else int(s[1]),
            "end": int(s.end) if hasattr(s, 'end') else int(s[2]),
        })

    result: dict[str, Any] = {
        "num_seqlets": len(seqlet_list),
        "seqlets": seqlet_list[:200],  # cap output
    }

    # Optional motif annotation
    if motif_file and Path(motif_file).exists():
        try:
            from tangermeme.io import read_meme
            from tangermeme.annotate import annotate_seqlets, count_annotations
            motifs = read_meme(motif_file)
            X_cpu = X.cpu()
            annotations, _ = annotate_seqlets(X_cpu, seqlets, motifs)
            counts = count_annotations(annotations)
            top_motifs = []
            if hasattr(counts, 'items'):
                for name, count in sorted(counts.items(), key=lambda x: -x[1])[:20]:
                    top_motifs.append({"name": str(name), "count": int(count)})
            result["top_motifs"] = top_motifs
        except Exception as e:
            result["annotation_error"] = str(e)

    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 25: tfbs_epistasis (pairwise / 3D mutation interactions)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_epistasis",
    annotations={
        "title": "Epistasis Analysis",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_epistasis(
    model_id: str,
    sequence: str,
    positions: list[int] | None = None,
    output_dir: str | None = None,
) -> str:
    """Compute pairwise epistasis scores for a DNA sequence.

    For each pair of positions, computes all single and double mutant
    predictions, then calculates the epistasis score:
    delta_delta = y_ij - y_i - y_j + y_orig

    Non-zero epistasis indicates the two positions interact (their
    effects are non-additive).

    Args:
        model_id: Identifier of a loaded model.
        sequence: Single DNA sequence to analyze.
        positions: Restrict analysis to these 0-indexed positions.
                   Default: all positions (warning: L^2 predictions).
        output_dir: If set, save CSVs (single_mutations.csv,
                    epistasis_matrix.csv) here.

    Returns:
        JSON with original_prediction, num_pairs_analyzed,
        top_interactions (sorted by |delta_delta|, capped at 500).
    """
    _ensure_imports()
    _validate_dna([sequence])

    entry = _get_model(model_id)
    model = entry["model"]
    device = entry["device"]

    X = _sequences_to_tensor([sequence]).to(device)
    L = X.shape[2]

    with _torch.no_grad():
        y_orig = model(X).cpu().item()

    pos = positions if positions is not None else list(range(L))

    # Single mutants
    single_deltas = {}
    for p in pos:
        for b in range(4):
            X_mut = X.clone()
            X_mut[0, :, p] = 0.0
            X_mut[0, b, p] = 1.0
            with _torch.no_grad():
                y_mut = model(X_mut).cpu().item()
            single_deltas[(p, b)] = y_mut - y_orig

    # Pairwise epistasis
    interactions = []
    for i_idx, pi in enumerate(pos):
        for pj in pos[i_idx + 1:]:
            for bi in range(4):
                for bj in range(4):
                    X_double = X.clone()
                    X_double[0, :, pi] = 0.0
                    X_double[0, bi, pi] = 1.0
                    X_double[0, :, pj] = 0.0
                    X_double[0, bj, pj] = 1.0
                    with _torch.no_grad():
                        y_double = model(X_double).cpu().item()
                    dd = y_double - single_deltas[(pi, bi)] - single_deltas[(pj, bj)]
                    if abs(dd) > 1e-6:
                        interactions.append({
                            "pos_i": pi, "base_i": _IDX_TO_NUC[bi],
                            "pos_j": pj, "base_j": _IDX_TO_NUC[bj],
                            "delta_delta": round(dd, 6),
                        })

    interactions.sort(key=lambda x: -abs(x["delta_delta"]))

    if output_dir:
        import csv as _csv
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "epistasis_top.csv", "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=["pos_i", "base_i", "pos_j", "base_j", "delta_delta"])
            w.writeheader()
            w.writerows(interactions[:500])

    return json.dumps({
        "original_prediction": round(y_orig, 6),
        "num_pairs_analyzed": len(pos) * (len(pos) - 1) // 2 * 16,
        "top_interactions": interactions[:500],
    })


# ---------------------------------------------------------------------------
# Tool 26: tfbs_conv_filters (conv filter visualization + HTML report)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_conv_filters",
    annotations={
        "title": "Visualize Conv Filters",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_conv_filters(
    model_id: str,
    sequences: list[str],
    output_dir: str,
    top_n_filters: int = 50,
) -> str:
    """Visualize what convolutional filters learned.

    Runs input sequences through the model, captures activations at
    each Conv1d layer, identifies maximally-activating subsequences
    per filter, builds position frequency matrices (PFMs), and
    generates a summary.

    Args:
        model_id: Identifier of a loaded model.
        sequences: DNA sequences to use for activation analysis.
        output_dir: Directory to save filter PFMs and summary.
        top_n_filters: Max filters to analyze per layer (default 50).

    Returns:
        JSON with num_layers, num_filters_analyzed, and per-filter
        summary (layer, filter_id, max_activation, num_activating).
    """
    _ensure_imports()
    _validate_dna(sequences)

    entry = _get_model(model_id)
    model = entry["model"]
    device = entry["device"]

    X = _sequences_to_tensor(sequences).to(device)

    # Find Conv1d layers
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, _torch.nn.Conv1d):
            conv_layers.append((name, module))

    if not conv_layers:
        return json.dumps({
            "error": "No Conv1d layers found in model",
            "model_type": entry["model_type"],
        })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Capture activations via hooks
    activations: dict[str, Any] = {}
    hooks = []
    for layer_name, module in conv_layers:
        def _hook(mod, inp, output, name=layer_name):
            activations[name] = output.detach().cpu()
        hooks.append(module.register_forward_hook(_hook))

    with _torch.no_grad():
        model(X)

    for h in hooks:
        h.remove()

    # Analyze filters
    filter_summary = []
    for layer_name, module in conv_layers:
        if layer_name not in activations:
            continue
        act = activations[layer_name]  # (N, C_out, L_out)
        n_filters = min(act.shape[1], top_n_filters)
        kernel_size = module.kernel_size[0]

        for f_idx in range(n_filters):
            f_act = act[:, f_idx, :]  # (N, L_out)
            max_act = float(f_act.max())

            # Find top activating positions
            threshold = max_act * 0.8
            high_mask = f_act >= threshold
            n_activating = int(high_mask.sum())

            # Extract PFM from top-activating windows
            pfm = _np.zeros((4, kernel_size))
            count = 0
            for seq_idx in range(f_act.shape[0]):
                positions = _torch.where(high_mask[seq_idx])[0]
                for pos in positions[:10]:  # cap per sequence
                    p = int(pos)
                    if p + kernel_size <= X.shape[2]:
                        window = X[seq_idx, :, p:p + kernel_size].cpu().numpy()
                        pfm += window
                        count += 1

            if count > 0:
                pfm /= count

            filter_summary.append({
                "layer": layer_name,
                "filter_id": f_idx,
                "max_activation": round(max_act, 4),
                "num_activating": n_activating,
                "kernel_size": kernel_size,
            })

    # Save summary
    with open(out / "filter_summary.json", "w") as f:
        json.dump(filter_summary, f, indent=2)

    return json.dumps({
        "output_dir": str(out.resolve()),
        "num_layers": len(conv_layers),
        "num_filters_analyzed": len(filter_summary),
        "filter_summary": filter_summary[:100],  # cap response
    })


# ---------------------------------------------------------------------------
# Tool 30: tfbs_chipseq_benchmark (ChIP-Atlas benchmark analysis)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_chipseq_benchmark",
    annotations={
        "title": "ChIP-seq Benchmark Analysis",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_chipseq_benchmark(
    intersect_dir: str,
    output_dir: str,
    pos_tf: str = "NKX2-1",
    neg_tfs: list[str] | None = None,
    datasets: list[str] | None = None,
    cutoffs: list[int] | None = None,
    model_type: str = "VCNNBpnet",
    models_dir: str = "saved_models_final",
    genome_fasta: str = "/sc-projects/sc-proj-btg/P09/data/genomes/hg38/hg38.fa",
    max_sequences: int = 15000,
    slurm: bool = False,
    partition: str = "gpu",
    max_time: str = "04:00:00",
) -> str:
    """Run ChIP-seq benchmark analysis comparing NN models against ChIP-Atlas data.

    Evaluates models in two ways:
    1. Classification: distinguish pos TF binding from neg TFs (ROC curves)
    2. Regression: correlate NN score with experiment overlap count

    Input: ChIP-Atlas intersection BED files (*.allintersect.bed).

    Args:
        intersect_dir: Directory with *.allintersect.bed files.
        output_dir: Where to save plots and CSV results.
        pos_tf: Positive transcription factor name (default NKX2-1).
        neg_tfs: Negative TF names (default GATA1, MYOD1, NKX2-5, RXRA).
        datasets: Model dataset names (default all_mean, core_mean, flank_mean).
        cutoffs: Experiment-count cutoffs for classification (default 1,3,5,8).
        model_type: Architecture class name (default VCNNBpnet).
        models_dir: Base checkpoint directory.
        genome_fasta: Reference genome FASTA path.
        max_sequences: Max sequences per TF for speed.
        slurm: Submit as SLURM job instead of running directly.
        partition: SLURM partition.
        max_time: SLURM wall time.

    Returns:
        JSON with output_dir and list of generated files, or job_id if SLURM.
    """
    if neg_tfs is None:
        neg_tfs = ["GATA1", "MYOD1", "NKX2-5", "RXRA"]
    if datasets is None:
        datasets = ["all_mean", "core_mean", "flank_mean"]
    if cutoffs is None:
        cutoffs = [1, 3, 5, 8]

    intersect_path = Path(intersect_dir).resolve()
    if not intersect_path.is_dir():
        return json.dumps({"error": f"Intersect directory not found: {intersect_dir}"})

    project_root = Path(__file__).resolve().parent.parent.parent
    script = project_root / "scripts" / "chipseq_benchmark.py"
    if not script.exists():
        return json.dumps({"error": f"Benchmark script not found: {script}"})

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"python {script}"
        f" --intersect-dir {intersect_path}"
        f" --output-dir {out}"
        f" --pos-tf {pos_tf}"
        f" --neg-tfs {' '.join(neg_tfs)}"
        f" --datasets {' '.join(datasets)}"
        f" --cutoffs {' '.join(str(c) for c in cutoffs)}"
        f" --model-type {model_type}"
        f" --models-dir {models_dir}"
        f" --genome {genome_fasta}"
        f" --max-sequences {max_sequences}"
    )

    if slurm:
        import os as _os
        sbatch_content = (
            f"#!/bin/bash\n"
            f"#SBATCH --job-name=chipseq_bench\n"
            f"#SBATCH --partition={partition}\n"
            f"#SBATCH --nodes=1\n"
            f"#SBATCH --cpus-per-task=8\n"
            f"#SBATCH --mem=64G\n"
            f"#SBATCH --gres=gpu:1\n"
            f"#SBATCH --time={max_time}\n"
            f"#SBATCH --output={out}/bench.o%j\n"
            f"#SBATCH --error={out}/bench.e%j\n"
            f"\ndate\n{cmd}\ndate\n"
        )
        sbatch_path = out / "chipseq_benchmark.sh"
        sbatch_path.write_text(sbatch_content)
        _os.chmod(str(sbatch_path), 0o755)

        submit_result = json.loads(await slurm_submit(
            script_path=str(sbatch_path),
            working_dir=str(project_root),
        ))
        submit_result["mode"] = "slurm"
        submit_result["command"] = cmd
        return json.dumps(submit_result)

    # Direct mode — run the script as subprocess
    result = subprocess.run(
        cmd.split(), capture_output=True, text=True,
        timeout=3600, cwd=str(project_root),
    )

    if result.returncode != 0:
        return json.dumps({
            "error": f"Benchmark failed (exit {result.returncode})",
            "stderr": result.stderr[-1000:],
        })

    # List output files
    files = [str(f.relative_to(out)) for f in out.rglob("*") if f.is_file()]

    return json.dumps({
        "mode": "direct",
        "output_dir": str(out),
        "files": files,
        "stdout": result.stdout[-500:],
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    # Ensure this module is reachable as 'tfbs.mcp.server' even when
    # executed via ``python -m tfbs.mcp.server`` (which sets __name__ to
    # '__main__').  Without this, sub-module imports like foldx_tools would
    # create a *second* copy of the module and register tools on a
    # different ``mcp`` instance.
    import sys
    if __name__ == "__main__" and "tfbs.mcp.server" not in sys.modules:
        sys.modules["tfbs.mcp.server"] = sys.modules[__name__]

    # Register FoldX tools on the shared mcp instance
    import tfbs.mcp.foldx_tools  # noqa: F401

    mcp.run()


if __name__ == "__main__":
    main()
