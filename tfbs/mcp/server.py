"""
MCP server exposing TFBS-NN model loading, prediction, preprocessing,
attribution analysis, and FIMO motif scanning as tools.

Run with:  tfbs-mcp          (stdio transport, for Claude Desktop / VS Code)
           python -m tfbs.mcp.server   (same)
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from mcp.server.fastmcp import FastMCP

import tfbs.nn.models as models
from tfbs.prediction.analysis import find_best_windows_fast

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
mcp = FastMCP("tfbs")

# ---------------------------------------------------------------------------
# In-process model registry  (persists for the lifetime of the server)
# ---------------------------------------------------------------------------
_model_registry: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
_IDX_TO_NUC = ["A", "C", "G", "T"]


def _sequences_to_tensor(sequences: list[str]) -> torch.Tensor:
    """Convert DNA strings to a one-hot tensor of shape (N, 4, L)."""
    indices = torch.tensor(
        [[_NUC_TO_IDX[c] for c in seq.upper()] for seq in sequences],
        dtype=torch.long,
    )
    ohe = F.one_hot(indices, num_classes=4).float()  # (N, L, 4)
    return ohe.permute(0, 2, 1)  # (N, 4, L)


def _tensor_to_sequences(tensor: torch.Tensor) -> list[str]:
    """Convert a (N, 4, L) one-hot tensor back to DNA strings."""
    indices = tensor.argmax(dim=1).cpu().numpy()  # (N, L)
    return ["".join(_IDX_TO_NUC[i] for i in row) for row in indices]


def _get_model(model_id: str) -> dict[str, Any]:
    """Retrieve a loaded model or raise a clear error."""
    if model_id not in _model_registry:
        available = list(_model_registry.keys()) or ["(none)"]
        raise ValueError(
            f"Model '{model_id}' not loaded. "
            f"Call load_model first. Available: {', '.join(available)}"
        )
    return _model_registry[model_id]


# ---------------------------------------------------------------------------
# Tool 1: load_model
# ---------------------------------------------------------------------------
@mcp.tool()
def load_model(
    model_id: str,
    checkpoint_path: str,
    model_type: str,
    device: str = "cpu",
) -> str:
    """Load a trained TFBS model checkpoint into memory.

    Args:
        model_id: User-chosen identifier for this model (e.g. "my_vcnn").
        checkpoint_path: Path to the .ckpt file.
        model_type: Class name from tfbs.nn.models (e.g. "VCNNBpnet", "RNN").
        device: "cpu" or "cuda" (default "cpu").
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return json.dumps({"error": f"Checkpoint not found: {checkpoint_path}"})

    try:
        model_cls = getattr(models, model_type)
    except AttributeError:
        available = [
            n for n in dir(models)
            if isinstance(getattr(models, n, None), type)
            and issubclass(getattr(models, n), torch.nn.Module)
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
# Tool 2: predict
# ---------------------------------------------------------------------------
@mcp.tool()
def predict(
    model_id: str,
    sequences: list[str],
    window_mode: bool = False,
    window_size: int = 70,
    batch_size: int = 128,
) -> str:
    """Score DNA sequences with a loaded model.

    Args:
        model_id: Identifier of a previously loaded model.
        sequences: List of DNA sequences (e.g. ["ACGTACGT..."]).
        window_mode: If true, use sliding-window scoring for long sequences.
        window_size: Window size for sliding-window mode.
        batch_size: Batch size for processing.
    """
    entry = _get_model(model_id)
    model = entry["model"]
    device = entry["device"]

    X = _sequences_to_tensor(sequences).to(device)

    if window_mode:
        best_windows, best_scores, _, mean_scores = find_best_windows_fast(
            X, model, window_size=window_size,
            processing_batch_size=batch_size, device=device,
        )
        best_seqs = _tensor_to_sequences(torch.from_numpy(best_windows))
        return json.dumps({
            "best_scores": best_scores.tolist(),
            "best_sequences": best_seqs,
            "mean_scores": mean_scores.tolist(),
        })

    with torch.no_grad():
        scores = model(X).cpu().numpy()

    return json.dumps({"scores": scores.squeeze().tolist()})


# ---------------------------------------------------------------------------
# Tool 3: preprocess
# ---------------------------------------------------------------------------
@mcp.tool()
def preprocess(
    sequences: list[str],
    labels: list[float] | None = None,
    add_rc: bool = False,
    output_path: str | None = None,
) -> str:
    """One-hot encode DNA sequences and optionally add reverse complements.

    Args:
        sequences: Raw DNA strings.
        labels: Optional numeric labels for each sequence.
        add_rc: Whether to add reverse complement augmentation.
        output_path: If provided, save tensors to this directory.
    """
    X = _sequences_to_tensor(sequences)

    if labels is not None:
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    else:
        y = None

    if add_rc:
        rc = torch.flip(X, [2])
        rc = torch.index_select(rc, 1, torch.tensor([3, 2, 1, 0], dtype=torch.long))
        X = torch.cat([X, rc], dim=0)
        if y is not None:
            y = torch.cat([y, y], dim=0)

    result: dict[str, Any] = {
        "shape": list(X.shape),
        "num_sequences": X.shape[0],
    }

    if output_path:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(X, out / "seqs.pt")
        if y is not None:
            torch.save(y, out / "labels.pt")
        result["saved_to"] = str(out)
    else:
        result["sequences_encoded"] = True

    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 4: analyze (attribution)
# ---------------------------------------------------------------------------
@mcp.tool()
def analyze(
    model_id: str,
    sequences: list[str],
    method: str = "deepliftshap",
    n_shuffles: int = 20,
    hypothetical: bool = False,
) -> str:
    """Run attribution analysis on sequences to identify important positions.

    Requires captum and tangermeme (pip install 'tfbs-nn[viz]').

    Args:
        model_id: Identifier of a previously loaded model.
        sequences: DNA sequences to analyze.
        method: "deepliftshap" or "input_x_gradient".
        n_shuffles: Number of shuffled baselines for DeepLiftShap.
        hypothetical: Return hypothetical attributions for all bases.
    """
    try:
        from captum.attr import DeepLiftShap, InputXGradient
        from tangermeme.utils import random_one_hot
        from tangermeme.deep_lift_shap import hypothetical_attributions
    except ImportError:
        return json.dumps({
            "error": "captum and tangermeme required. Install with: pip install 'tfbs-nn[viz]'"
        })

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
    result_sum = torch.zeros((N, C, L), dtype=X.dtype, device="cpu")

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
            baselines_rep = torch.cat([baselines] * batch_N, dim=0)
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
# Tool 5: fimo
# ---------------------------------------------------------------------------
@mcp.tool()
def fimo(
    sequences: list[str],
    motif_file: str,
    threshold: float = 1e-4,
    output_dir: str | None = None,
) -> str:
    """Run FIMO motif scanning on DNA sequences.

    Requires the FIMO binary from MEME Suite on PATH.

    Args:
        sequences: DNA sequences to scan.
        motif_file: Path to MEME-format motif file.
        threshold: FIMO p-value threshold.
        output_dir: Directory for FIMO output (uses temp dir if omitted).
    """
    import pandas as pd

    if not Path(motif_file).exists():
        return json.dumps({"error": f"Motif file not found: {motif_file}"})

    # Write sequences to a temp FASTA
    use_temp = output_dir is None
    work_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="tfbs_fimo_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = work_dir / "input.fasta"
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq{i}\n{seq}\n")

    fimo_out = work_dir / "fimo_out"

    cmd = [
        "fimo",
        "--thresh", str(threshold),
        "--oc", str(fimo_out),
        "--max-strand",
        "--max-stored-scores", "2147483646",
        str(motif_file),
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
# List loaded models (utility tool)
# ---------------------------------------------------------------------------
@mcp.tool()
def list_models() -> str:
    """List all currently loaded models."""
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
# Entry point
# ---------------------------------------------------------------------------
def main():
    mcp.run()


if __name__ == "__main__":
    main()
