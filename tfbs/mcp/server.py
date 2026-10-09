"""
MCP server exposing TFBS-NN model loading, prediction, preprocessing,
attribution analysis, and FIMO motif scanning as tools.

Run with:  tfbs-mcp          (stdio transport, for Claude Desktop / VS Code)
           python -m tfbs.mcp.server   (same)
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
mcp = FastMCP("tfbs")

# ---------------------------------------------------------------------------
# In-process model registry  (persists for the lifetime of the server)
# ---------------------------------------------------------------------------
_model_registry: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Environment detection: runs once at import time (lightweight, no torch).
# ---------------------------------------------------------------------------
SLURM_PARTITION_ENV = "TFBS_SLURM_PARTITION"
CALL_LOG_ENV = "TFBS_CALL_LOG"


def _detect_environment() -> dict[str, Any]:
    """Detect SLURM cluster and GPU availability.

    Checks, in order:
      1. SLURM_GPUS_ON_NODE env var
      2. CUDA_VISIBLE_DEVICES env var
      3. nvidia-smi (quick subprocess probe)

    Returns a dict describing the runtime:
      - on_slurm: True if running inside a SLURM job
      - slurm_job_id: str or None
      - slurm_partition: partition of the job the server runs in, or None
      - default_partition: partition the SLURM tools submit to unless the
        call passes partition=...; $TFBS_SLURM_PARTITION, else "gpu"
      - gpu_count: detected number of GPUs (0 if none found)
      - default_device: "cuda" or "cpu"
      - default_batch_size: larger on GPU nodes
    """
    on_slurm = "SLURM_JOB_ID" in os.environ
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_partition = os.environ.get("SLURM_JOB_PARTITION")

    # Count GPUs; try multiple detection methods
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
        "default_partition": os.environ.get(SLURM_PARTITION_ENV, "").strip() or "gpu",
        "gpu_count": gpu_count,
        "default_device": "cuda" if has_gpu else "cpu",
        "default_batch_size": 512 if has_gpu else 64,
    }


_ENV = _detect_environment()


def _resolve_partition(partition: str | None) -> str:
    """The caller's partition, else the server default (see tfbs_server_info)."""
    return partition or _ENV["default_partition"]


def _resolve_device(device: str | None, slurm_gpu: bool | None = None) -> str:
    """Device for NN inference: the caller's choice, else where the work runs.

    slurm_gpu is None when the work runs on this machine; the server's
    detected default is then cross-checked against torch.  For a SLURM job
    it says whether the job requests a GPU, since the job does not run on
    the machine the server detected.
    """
    if device:
        return device
    if slurm_gpu is not None:
        return "cuda" if slurm_gpu else "cpu"
    if _ENV["default_device"] == "cuda":
        _ensure_imports()
    return _ENV["default_device"]


# ---------------------------------------------------------------------------
# Tool-call hook: opt-in call log and stdout protection
#
# Every MCP tool call passes through the tool manager, so one wrapper there
# covers all 33 tools (including the FoldX tools registered later).
# ---------------------------------------------------------------------------
def _append_call_log(tool: str, arguments: dict[str, Any]) -> None:
    """Append one JSON line {timestamp, tool, arguments} to $TFBS_CALL_LOG."""
    target = os.environ.get(CALL_LOG_ENV, "").strip()
    if not target:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "tool": tool,
        "arguments": arguments,
    }
    try:
        path = Path(target).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:
        logger.warning("Could not append to %s (%s): %s", target, CALL_LOG_ENV, exc)


_stdout_guard_depth = 0
_stdout_saved: Any = None


@contextlib.contextmanager
def _stdout_to_stderr():
    """Send stray print() output to stderr while a tool runs.

    On the stdio transport stdout carries the JSON-RPC stream (the transport
    holds its own handle on the stdout buffer), so a print() from a tool or a
    library it calls would corrupt it.  The depth counter keeps overlapping
    async tool calls from restoring the wrong stream.
    """
    global _stdout_guard_depth, _stdout_saved
    if _stdout_guard_depth == 0:
        _stdout_saved = sys.stdout
        sys.stdout = sys.stderr
    _stdout_guard_depth += 1
    try:
        yield
    finally:
        _stdout_guard_depth -= 1
        if _stdout_guard_depth == 0:
            sys.stdout = _stdout_saved
            _stdout_saved = None


_tool_manager_call_tool = mcp._tool_manager.call_tool


async def _call_tool_hook(
    name: str,
    arguments: dict[str, Any],
    context: Any = None,
    convert_result: bool = False,
) -> Any:
    _append_call_log(name, arguments)
    with _stdout_to_stderr():
        return await _tool_manager_call_tool(
            name, arguments, context=context, convert_result=convert_result,
        )


mcp._tool_manager.call_tool = _call_tool_hook


# ---------------------------------------------------------------------------
# Reference-data locations
#
# These paths used to be hardcoded to the developers' HPC cluster, so every
# tool signature advertised a filesystem nobody else can reach and the first
# call a new user made failed on someone else's disk.  A path now comes from
# the caller or, failing that, from an environment variable set in the MCP
# client config -- and a tool that cannot find what it needs says which
# argument and which variable would fix it.
# ---------------------------------------------------------------------------
REFERENCE_DATA_ENV = {
    "genome_fasta": "TFBS_GENOME_FASTA",
    "motif_file": "TFBS_MOTIF_FILE",
    "foldx_pdb": "TFBS_FOLDX_PDB",
    "clinvar_vcf": "TFBS_CLINVAR_VCF",
}

_REFERENCE_DATA_LABEL = {
    "genome_fasta": "reference genome FASTA",
    "motif_file": "JASPAR/MEME motif file",
    "foldx_pdb": "FoldX TF-DNA complex PDB",
    "clinvar_vcf": "ClinVar VCF",
}


def _project_root() -> Path:
    """Directory holding data/, saved_models/ and results/.

    $TFBS_PROJECT_ROOT when set, else the repository root when the package
    runs from a checkout, else the server's working directory.  The code the
    tools run ships inside the package (tfbs.pipelines, tfbs.configs), so
    this locates only the user's data, checkpoints and outputs; it is also
    the working directory of the pipeline runs, against which relative paths
    such as models_dir resolve.
    """
    env = os.environ.get("TFBS_PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    checkout = Path(__file__).resolve().parent.parent.parent
    if (checkout / "pyproject.toml").is_file():
        return checkout
    return Path.cwd().resolve()


def _work_dir() -> tuple[Path | None, str | None]:
    """The project root as a pipeline working directory, or an error JSON."""
    root = _project_root()
    if root.is_dir():
        return root, None
    return None, _error(
        f"Project root is not a directory: {root}",
        "Set TFBS_PROJECT_ROOT to an existing directory (the one holding "
        "data/, saved_models/ and results/), or unset it to use the server's "
        "working directory.",
    )


# Pipelines the training and validation tools run, with the optional
# modules each one imports and the pip extra that provides them.
_PIPELINE_EXTRAS = {"pyfaidx": "genomics", "pysam": "genomics", "wandb": "tracking"}
_PIPELINE_MODULES = {
    "validate": (),
    "run_sweep": ("wandb",),
    "chipseq_benchmark": ("pyfaidx",),
    "chipseq_benchmark_with_foldx": ("pyfaidx",),
    "remap_validation": ("pyfaidx",),
    "disease_variant_validation": ("pyfaidx", "pysam"),
}


def _pipeline_argv(name: str) -> list[str]:
    """Command that runs the packaged pipeline tfbs.pipelines.<name>."""
    return [sys.executable, "-m", f"tfbs.pipelines.{name}"]


def _pipeline_pythonpath() -> str | None:
    """Directory to prepend to PYTHONPATH so a pipeline imports this tfbs.

    None for an installed package, which the interpreter finds from any
    working directory.  A source tree run without pip install is found only
    from its own directory, and the pipelines run from the project root.
    """
    parent = Path(__file__).resolve().parent.parent.parent
    if parent.name in ("site-packages", "dist-packages"):
        return None
    return str(parent)


def _pipeline_env() -> dict[str, str]:
    env = dict(os.environ)
    extra = _pipeline_pythonpath()
    if extra:
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join([extra, current]) if current else extra
    return env


def _sbatch_pythonpath_line() -> str:
    extra = _pipeline_pythonpath()
    if not extra:
        return ""
    return f"export PYTHONPATH={shlex.quote(extra)}${{PYTHONPATH:+:$PYTHONPATH}}\n"


def _pipeline_requirements_error(name: str, tool: str) -> str | None:
    """Error JSON when a module the pipeline imports is not installed."""
    import importlib.util

    missing = [m for m in _PIPELINE_MODULES[name] if importlib.util.find_spec(m) is None]
    if not missing:
        return None
    extras = sorted({_PIPELINE_EXTRAS[m] for m in missing})
    fix = (f'pip install "tfbs-mcp[{",".join(extras)}]" in the environment the '
           "server runs in, then restart it.")
    if "pysam" in missing and sys.platform == "win32":
        fix = ("pysam has no Windows build: run this tool from Linux, macOS or "
               "WSL, where " + fix)
    return _error(f"{tool} needs {', '.join(missing)}, which is not installed.", fix)


def _run_pipeline(cmd: list[str], cwd: Path, timeout: int, label: str):
    """Run a pipeline in a subprocess: (CompletedProcess, None) or (None, error JSON)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(cwd), env=_pipeline_env(),
        )
    except subprocess.TimeoutExpired:
        return None, _error(
            f"{label} timed out after {timeout} s.",
            "Run it with slurm=true, which leaves the time limit to the job.",
        )
    except OSError as exc:
        return None, _error(
            f"{label} could not start: {exc}",
            "Check that the server's Python interpreter still exists.",
        )
    return result, None


def _resolve_reference(value: str | None, arg: str) -> tuple[Path | None, str | None]:
    """Resolve a reference-data path from an argument or the environment.

    Returns ``(path, None)`` when the file exists, otherwise
    ``(None, error_json)`` -- a ready-to-return JSON string naming both the
    argument and the environment variable that would satisfy it, so callers
    can write::

        genome, err = _resolve_reference(genome_fasta, "genome_fasta")
        if err:
            return err
    """
    env_var = REFERENCE_DATA_ENV[arg]
    what = _REFERENCE_DATA_LABEL[arg]
    raw = (value or "").strip() or os.environ.get(env_var, "").strip()
    if not raw:
        return None, json.dumps({
            "error": f"No {what} configured.",
            "fix": f"Pass {arg}=/path/to/file, or set {env_var} in the 'env' "
                   f"block of your MCP client config.",
        })
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        return None, json.dumps({
            "error": f"{what} not found: {path}",
            "fix": f"Check the {arg} argument or the {env_var} environment variable.",
        })
    return path, None


# ---------------------------------------------------------------------------
# Lazy imports: torch and friends are heavy (~13 s).  We defer them so the
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
# Lazy imports: genomics (pyfaidx, tangermeme.io)
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
            # File handle closed; reopen
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
    # .contiguous() matters: permute() returns a non-contiguous view, and the
    # models that flatten their input with .view() (MLP, FlexibleMLP) raise
    # "view size is not compatible with input tensor's size and stride" on one.
    # Batches from the DataLoader are contiguous via collation, so this path
    # was the only one that hit it.
    return ohe.permute(0, 2, 1).contiguous()  # (N, 4, L)


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
# Tool results: errors, default output locations and sequence inputs
#
# Output locations follow from the inputs, so a chain of calls needs no
# filenames from the user:
#   data tools     <dir>_<kind>/ next to the input, e.g. peaks.bed ->
#                  peaks_loci/ -> peaks_classification/ -> peaks_dataset/
#   model analyses <results>/<model_id>/<tool>/<digest>/, where <results>
#                  is tfbs_server_info's results_dir and <digest> hashes
#                  the call's inputs, so a repeated call reuses its
#                  directory and different inputs never overwrite each other.
# ---------------------------------------------------------------------------
def _error(error: str, fix: str, **extra: Any) -> str:
    return json.dumps({"error": error, "fix": fix, **extra})


def _model_entry(model_id: str) -> tuple[dict[str, Any] | None, str | None]:
    """(registry entry, None), or (None, error JSON) for an unknown model_id."""
    try:
        return _get_model(model_id), None
    except ValueError as exc:
        return None, _error(
            str(exc),
            "Call tfbs_load_model with this model_id first; tfbs_list_models "
            "lists the loaded models.",
        )


def _results_root() -> Path:
    """Base directory for the model-analysis tools' default outputs.

    <project root>/results for a checkout or $TFBS_PROJECT_ROOT.  After a
    plain pip install the project root would be site-packages, so the
    default is then tfbs_results/ in the server's working directory.
    """
    root = _project_root()
    if os.environ.get("TFBS_PROJECT_ROOT", "").strip() or (root / "pyproject.toml").is_file():
        return root / "results"
    return Path.cwd().resolve() / "tfbs_results"


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._") or "unnamed"


def _digest(*parts: Any) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:10]


def _file_identity(path: str) -> list[Any]:
    """Path, size and mtime: a rewritten input file gets a new digest."""
    p = Path(path).expanduser().resolve()
    try:
        st = p.stat()
        return [str(p), st.st_size, st.st_mtime_ns]
    except OSError:
        return [str(p)]


def _analysis_dir(output_dir: str | None, model_id: str, tool: str, *inputs: Any) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return _results_root() / _safe_name(model_id) / tool / _digest(*inputs)


def _input_key(sequences: list[str] | None, sequences_path: str | None) -> Any:
    return _file_identity(sequences_path) if sequences_path else sequences


_DERIVED_SUFFIXES = ("_loci", "_classification", "_regression", "_dataset")


def _sibling_dir(input_file: str | Path, kind: str) -> Path:
    """<parent>/<name>_<kind> for an input file in <parent>/<name>[_<suffix>]/."""
    d = Path(input_file).expanduser().resolve().parent
    name = d.name
    for suffix in _DERIVED_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
            break
    return d.parent / f"{name or 'tfbs'}_{kind}"


_ONE_HOT_LUT = None


def _one_hot(seqs: list[str]):
    """Equal-length DNA strings to a float (N, 4, L) tensor; non-ACGT columns are zero."""
    global _ONE_HOT_LUT
    _ensure_imports()
    if _ONE_HOT_LUT is None:
        lut = _np.full(256, 4, dtype=_np.int64)
        for i, base in enumerate("ACGT"):
            lut[ord(base)] = i
            lut[ord(base.lower())] = i
        _ONE_HOT_LUT = lut
    if not seqs:
        return _torch.zeros((0, 4, 0))
    raw = _np.frombuffer("".join(seqs).encode("ascii", "replace"), dtype=_np.uint8)
    idx = _ONE_HOT_LUT[raw].reshape(len(seqs), -1)
    table = _np.zeros((5, 4), dtype=_np.float32)
    table[:4] = _np.eye(4, dtype=_np.float32)
    return _torch.from_numpy(table[idx].transpose(0, 2, 1).copy())


_FASTA_SUFFIXES = (".fa", ".fasta", ".fna", ".fas")


def _read_fasta_records(path: Path) -> list[tuple[str, str]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    records: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                fields = line[1:].split()
                name, chunks = (fields[0] if fields else f"seq{len(records)}"), []
            else:
                chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def _load_sequence_input(
    sequences: list[str] | None, sequences_path: str | None,
) -> tuple[Any, list[str] | None, str | None]:
    """(X, ids, None) with X a CPU float (N, 4, L) tensor, or (None, None, error JSON).

    sequences_path takes a one-hot .pt tensor as written by
    tfbs_extract_loci or tfbs_preprocess, or a FASTA file (.fa, .fasta,
    .fna, .fas, optionally .gz); ids are the FASTA record names.
    """
    _ensure_imports()
    if (sequences is None) == (sequences_path is None):
        return None, None, _error(
            "Give exactly one of sequences and sequences_path.",
            "Pass DNA strings as sequences, or a .pt tensor or FASTA file as sequences_path.",
        )
    ids = None
    where = "Sequence {}"
    if sequences is not None:
        seqs = list(sequences)
    else:
        path = Path(sequences_path).expanduser().resolve()
        if not path.is_file():
            return None, None, _error(
                f"sequences_path not found: {sequences_path}",
                "Pass the sequences.pt written by tfbs_extract_loci (or seqs.pt "
                "from tfbs_preprocess), or a FASTA file.",
            )
        suffixes = [s.lower() for s in path.suffixes]
        if suffixes[-1:] == [".pt"]:
            try:
                X = _torch.load(path, map_location="cpu", weights_only=True)
            except Exception as exc:
                return None, None, _error(
                    f"Could not read {path}: {exc}",
                    "Pass a tensor saved with torch.save, e.g. sequences.pt from tfbs_extract_loci.",
                )
            if not isinstance(X, _torch.Tensor):
                return None, None, _error(
                    f"{path} does not hold a tensor.",
                    "Pass a one-hot tensor of shape (N, 4, L).",
                )
            if X.dim() == 2:
                X = X.unsqueeze(0)
            if X.dim() != 3 or X.shape[1] != 4 or X.shape[0] == 0:
                return None, None, _error(
                    f"{path} has shape {list(X.shape)}, expected (N, 4, L) with N > 0.",
                    "Pass a one-hot tensor with the four bases on axis 1, as "
                    "written by tfbs_extract_loci.",
                )
            return X.float().contiguous(), None, None
        core = suffixes[:-1] if suffixes[-1:] == [".gz"] else suffixes
        if core[-1:] == [] or core[-1] not in _FASTA_SUFFIXES:
            return None, None, _error(
                f"Unrecognised sequence file type: {path.name}",
                "Use a .pt tensor or a FASTA file ending in .fa, .fasta, .fna "
                "or .fas (optionally .gz).",
            )
        try:
            records = _read_fasta_records(path)
        except (OSError, UnicodeDecodeError) as exc:
            return None, None, _error(f"Could not read {path}: {exc}",
                                      "Pass a plain-text or gzipped FASTA file.")
        if not records:
            return None, None, _error(f"No FASTA records in {path}.",
                                      "Pass a FASTA file with at least one record.")
        ids = [name for name, _ in records]
        seqs = [seq.upper() for _, seq in records]
        where = "Record {} of " + path.name
    if not seqs:
        return None, None, _error("No sequences given.", "Pass at least one DNA sequence.")
    valid = set("ACGTacgt")
    for i, seq in enumerate(seqs):
        bad = set(seq) - valid
        if bad or not seq:
            return None, None, _error(
                f"{where.format(ids[i] if ids else i)} is empty or contains "
                f"characters other than A, C, G, T: {sorted(bad)}.",
                "Remove or replace N and IUPAC codes; tfbs_extract_loci already "
                "drops windows that contain N.",
            )
    lengths = sorted({len(s) for s in seqs})
    if len(lengths) > 1:
        return None, None, _error(
            f"Sequences differ in length ({lengths[0]} to {lengths[-1]} bp).",
            "Give sequences of one length, e.g. windows from tfbs_extract_loci "
            "with a fixed in_window.",
        )
    return _one_hot(seqs), ids, None


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
    sequences: list[str] | None = None,
    window_mode: bool = False,
    window_size: int = 70,
    batch_size: int | None = None,
    inverse_transform: bool = False,
    sequences_path: str | None = None,
) -> str:
    """Score DNA sequences for transcription-factor binding affinity.

    In normal mode each sequence must match the model's expected input
    length.  In window_mode a sliding window finds the highest-scoring
    sub-sequence within longer inputs.  Sequences come inline
    (sequences) or from a file (sequences_path), so the output of
    tfbs_extract_loci, including its negatives/sequences.pt and the
    ref_sequences.pt / alt_sequences.pt of a variant extraction, can be
    scored without copying sequences into the call.  Inputs are scored
    in batches of batch_size, so large files do not need to fit on the
    device at once.

    Args:
        model_id: Identifier of a model loaded via tfbs_load_model.
        sequences: List of DNA sequences (e.g. ["ACGTACGT..."]).  Give
                   exactly one of sequences and sequences_path.
        window_mode: Use sliding-window scoring for long sequences.
        window_size: Window length for sliding-window mode.
        batch_size: Batch size for processing.  Auto-detected if omitted
                    (512 on GPU, 64 on CPU).
        inverse_transform: If true, reverse any label scaling that was
                           applied during training (requires scaling_params
                           to have been loaded via data_path in
                           tfbs_load_model).  Only affects regression
                           models.  Default false, because scores in the
                           training scale are what tfbs_classify_metrics
                           and ref/alt differences need; pass true to
                           report values in the original label units.
        sequences_path: One-hot .pt tensor (N, 4, L) as written by
                        tfbs_extract_loci or tfbs_preprocess, or a FASTA
                        file (.fa/.fasta/.fna/.fas, optionally .gz) of
                        equal-length A/C/G/T records.

    Returns:
        JSON with "scores" (normal) or "best_scores", "best_sequences",
        and "mean_scores" (window mode), plus "num_sequences"; for a
        FASTA input also "ids" (record names, in score order).  If
        inverse_transform is true, scores are in the original label
        space.
    """
    _ensure_imports()
    X, ids, err = _load_sequence_input(sequences, sequences_path)
    if err:
        return err

    if batch_size is None:
        batch_size = _ENV["default_batch_size"]

    entry, err = _model_entry(model_id)
    if err:
        return err
    model = entry["model"]
    device = entry["device"]

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
            "num_sequences": int(X.shape[0]),
        }
        if ids is not None:
            result["ids"] = ids
        if inverse_transform and entry.get("scaling_params"):
            result["inverse_transformed"] = True
        return json.dumps(result)

    chunks = []
    with _torch.no_grad():
        for k in range(0, X.shape[0], batch_size):
            chunks.append(model(X[k:k + batch_size].to(device)).cpu())
    scores = _torch.cat(chunks).numpy()

    scores = _inverse(scores)
    result = {"scores": scores.squeeze().tolist(), "num_sequences": int(X.shape[0])}
    if ids is not None:
        result["ids"] = ids
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
    every sequence is appended, doubling N.  The tensors are always saved
    as seqs.pt (and labels.pt), which tfbs_prepare_dataset splits into the
    train/val/test layout the DataModule reads, and which tfbs_predict,
    tfbs_analyze and tfbs_mutagenesis accept as sequences_path.

    Args:
        sequences: Raw DNA strings (A/C/G/T only).
        labels: Optional numeric labels for each sequence.
        add_rc: Whether to append reverse-complement augmentation.
        output_path: Directory for seqs.pt and labels.pt.  Default
                     <results>/tfbs_preprocess/<digest>/, with <results>
                     the results_dir of tfbs_server_info and <digest> a
                     hash of the sequences, labels and add_rc, so the same
                     input always lands in the same directory.

    Returns:
        JSON with tensor shape, num_sequences, saved_to (the directory)
        and files (seqs and, with labels, labels: the paths written).
    """
    _ensure_imports()
    X, _, err = _load_sequence_input(sequences, None)
    if err:
        return err

    if labels is not None:
        if len(labels) != len(sequences):
            return _error(
                f"{len(labels)} labels for {len(sequences)} sequences.",
                "Pass one label per sequence, or omit labels.",
            )
        y = _torch.tensor(labels, dtype=_torch.float32).unsqueeze(1)
    else:
        y = None

    if add_rc:
        rc = _torch.flip(X, [2])
        rc = _torch.index_select(rc, 1, _torch.tensor([3, 2, 1, 0], dtype=_torch.long))
        X = _torch.cat([X, rc], dim=0)
        if y is not None:
            y = _torch.cat([y, y], dim=0)

    if output_path:
        out = Path(output_path).expanduser().resolve()
    else:
        out = _results_root() / "tfbs_preprocess" / _digest(sequences, labels, add_rc)
    out.mkdir(parents=True, exist_ok=True)
    files = {"seqs": str(out / "seqs.pt")}
    _torch.save(X, out / "seqs.pt")
    if y is not None:
        _torch.save(y, out / "labels.pt")
        files["labels"] = str(out / "labels.pt")

    return json.dumps({
        "shape": list(X.shape),
        "num_sequences": X.shape[0],
        "saved_to": str(out),
        "files": files,
    })


# ---------------------------------------------------------------------------
# Tool 4: tfbs_analyze (attribution)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_analyze",
    annotations={
        "title": "Attribution Analysis",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_analyze(
    model_id: str,
    sequences: list[str] | None = None,
    method: str = "deepliftshap",
    n_shuffles: int = 20,
    hypothetical: bool = False,
    sequences_path: str | None = None,
    output_dir: str | None = None,
    batch_size: int | None = None,
) -> str:
    """Run attribution analysis to identify important sequence positions.

    Uses Captum to compute per-base importance scores for a loaded model.
    Requires the viz extras: pip install 'tfbs-mcp[viz]'

    The attributions are saved as attributions.pt (N x 4 x L) in
    output_dir.  Inline sequences also get them in the JSON; a
    sequences_path input (possibly thousands of windows) gets only the
    file path, the shape and the per-sequence attribution sums.

    Args:
        model_id: Identifier of a model loaded via tfbs_load_model.
        sequences: DNA sequences to analyze (A/C/G/T).  Give exactly one
                   of sequences and sequences_path.
        method: "deepliftshap" (default, because it gives stable
                attributions for ReLU CNNs by averaging over many
                references) or "input_x_gradient" (one backward pass:
                faster, noisier).
        n_shuffles: Number of shuffled baselines for DeepLiftShap.
        hypothetical: Return hypothetical attributions for all bases
                      (DeepLiftShap only).
        sequences_path: One-hot .pt tensor (N, 4, L) as written by
                        tfbs_extract_loci or tfbs_preprocess, or a FASTA
                        file of equal-length A/C/G/T records.
        output_dir: Directory for attributions.pt.  Default
                    <results>/<model_id>/tfbs_analyze/<digest>/, with
                    <results> the results_dir of tfbs_server_info and
                    <digest> a hash of the inputs and settings.
        batch_size: Sequences per input_x_gradient batch (DeepLiftShap
                    batches sequence/reference pairs, 256 at a time).
                    Auto-detected if omitted (512 on GPU, 64 on CPU).

    Returns:
        JSON with "shape", method info, "files" (attributions: the .pt
        path) and, for inline sequences, "attributions" (N x 4 x L);
        for a sequences_path input "attribution_sums" (one total per
        sequence) and, for FASTA, "ids".
    """
    try:
        from captum.attr import DeepLiftShap, InputXGradient
        from tangermeme.utils import random_one_hot
        from tangermeme.deep_lift_shap import hypothetical_attributions
    except ImportError:
        return _error(
            "captum and tangermeme required.",
            "Install with: pip install 'tfbs-mcp[viz]'",
        )

    _ensure_imports()
    if method not in ("deepliftshap", "input_x_gradient"):
        return _error(f"Unknown method '{method}'.",
                      "Use method='deepliftshap' or method='input_x_gradient'.")
    X, ids, err = _load_sequence_input(sequences, sequences_path)
    if err:
        return err
    entry, err = _model_entry(model_id)
    if err:
        return err
    model = entry["model"]
    device = entry["device"]
    if batch_size is None:
        batch_size = _ENV["default_batch_size"]

    out = _analysis_dir(
        output_dir, model_id, "tfbs_analyze",
        _input_key(sequences, sequences_path), method, n_shuffles, hypothetical,
    )
    N, C, L = X.shape

    if method == "input_x_gradient":
        parts = []
        model.train()  # cuDNN RNN backward fix
        try:
            ixg = InputXGradient(model)
            for k in range(0, N, batch_size):
                parts.append(ixg.attribute(X[k:k + batch_size].to(device), target=0).detach().cpu())
        finally:
            model.eval()
        avg = _torch.cat(parts)
        info: dict[str, Any] = {"method": "input_x_gradient"}
    else:
        max_pairs = 256
        result_sum = _torch.zeros((N, C, L), dtype=X.dtype, device="cpu")

        i = 0
        while i < N:
            batch_N = min(N - i, max_pairs)
            shuffles_per_call = max(1, max_pairs // batch_N)
            shuffles_per_call = min(shuffles_per_call, n_shuffles)
            X_batch = X[i : i + batch_N].to(device)

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

                if a.dim() == 3 and a.shape == (batch_N, C, L):
                    chunk_sum = a * s
                else:
                    chunk_sum = a.reshape(batch_N, s, C, L).sum(dim=1)

                result_sum[i : i + batch_N] += chunk_sum.cpu().detach()
                s0 += s
            i += batch_N

        avg = result_sum / float(n_shuffles)
        if not hypothetical:
            avg = avg * X
        info = {"method": "deepliftshap", "n_shuffles": n_shuffles,
                "hypothetical": hypothetical}

    avg = avg.detach().cpu()
    out.mkdir(parents=True, exist_ok=True)
    attr_file = out / "attributions.pt"
    _torch.save(avg, attr_file)

    result: dict[str, Any] = {"shape": list(avg.shape), **info,
                              "files": {"attributions": str(attr_file)}}
    if sequences is not None:
        result["attributions"] = avg.numpy().tolist()
    else:
        result["attribution_sums"] = avg.sum(dim=(1, 2)).numpy().tolist()
        if ids is not None:
            result["ids"] = ids
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 5: tfbs_mutagenesis (in-silico saturation mutagenesis)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_mutagenesis",
    annotations={
        "title": "Saturation Mutagenesis",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_mutagenesis(
    model_id: str,
    sequences: list[str] | None = None,
    start: int = 0,
    end: int = -1,
    raw_outputs: bool = False,
    batch_size: int | None = None,
    sequences_path: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Run in-silico saturation mutagenesis (ISM) on DNA sequences.

    For every position in each sequence, substitutes each of the 4 bases
    and records the change in model prediction.  This reveals which
    positions and substitutions most affect the score.  Mutants are
    scored in batches of batch_size, so large inputs are fine.

    Requires tangermeme: pip install 'tfbs-mcp[viz]'

    The result is saved in output_dir as ism_scores.pt (or, with
    raw_outputs, original_scores.pt and mutant_scores.pt).  Inline
    sequences also get the values in the JSON; a sequences_path input
    gets only the file paths and the shape.

    Args:
        model_id: Identifier of a model loaded via tfbs_load_model.
        sequences: DNA sequences to mutagenize (A/C/G/T).  Give exactly
                   one of sequences and sequences_path.
        start: First position to mutagenize (0-indexed, default 0).
        end: Last position (exclusive, -1 = full length).
        raw_outputs: If true, return original and per-mutant predictions
                     instead of aggregated attribution scores.
        batch_size: Batch size for predictions.  Auto-detected if omitted.
        sequences_path: One-hot .pt tensor (N, 4, L) as written by
                        tfbs_extract_loci or tfbs_preprocess, or a FASTA
                        file of equal-length A/C/G/T records.
        output_dir: Directory for the .pt results.  Default
                    <results>/<model_id>/tfbs_mutagenesis/<digest>/, with
                    <results> the results_dir of tfbs_server_info and
                    <digest> a hash of the inputs and settings.

    Returns:
        JSON with "scores" (shape N x 4 x L attribution matrix: the
        prediction change of each substitution, mean-centred across the
        four bases at each position) and "shape", or with
        raw_outputs=true: "original_scores" and "mutant_scores"; always
        "files" with the paths written.  For a sequences_path input the
        arrays are only in the files, and FASTA record names are in "ids".
    """
    try:
        # tangermeme >= 1.0 moved this out of tangermeme.ism.
        from tangermeme.saturation_mutagenesis import saturation_mutagenesis
    except ImportError:
        try:
            from tangermeme.ism import saturation_mutagenesis
        except ImportError:
            return _error(
                "tangermeme required.",
                "Install with: pip install 'tfbs-mcp[viz]'",
            )

    _ensure_imports()
    X, ids, err = _load_sequence_input(sequences, sequences_path)
    if err:
        return err

    if batch_size is None:
        batch_size = _ENV["default_batch_size"]

    entry, err = _model_entry(model_id)
    if err:
        return err
    model = entry["model"]
    device = entry["device"]

    out = _analysis_dir(
        output_dir, model_id, "tfbs_mutagenesis",
        _input_key(sequences, sequences_path), start, end, raw_outputs,
    )

    result = saturation_mutagenesis(
        model, X,
        start=start, end=end,
        batch_size=batch_size,
        raw_outputs=raw_outputs,
        device=device,
    )
    out.mkdir(parents=True, exist_ok=True)
    inline = sequences is not None

    if raw_outputs:
        # result is (y_orig, y_mutant) tuple
        y_orig, y_mut = (t.detach().cpu() for t in result)
        files = {"original_scores": str(out / "original_scores.pt"),
                 "mutant_scores": str(out / "mutant_scores.pt")}
        _torch.save(y_orig, files["original_scores"])
        _torch.save(y_mut, files["mutant_scores"])
        payload: dict[str, Any] = {"files": files, "shape": list(y_mut.shape)}
        if inline:
            payload["original_scores"] = y_orig.numpy().tolist()
            payload["mutant_scores"] = y_mut.numpy().tolist()
    else:
        attr = result.detach().cpu()
        files = {"scores": str(out / "ism_scores.pt")}
        _torch.save(attr, files["scores"])
        payload = {"shape": list(attr.shape), "method": "ism", "files": files}
        if inline:
            payload["scores"] = attr.numpy().tolist()
    if ids is not None:
        payload["ids"] = ids
    return json.dumps(payload)


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

    Requires tangermeme: pip install 'tfbs-mcp[viz]'

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
            "error": "tangermeme required. Install with: pip install 'tfbs-mcp[viz]'"
        })

    _ensure_imports()

    entry, err = _model_entry(model_id)
    if err:
        return err
    model = entry["model"]
    device = entry["device"]
    input_length = entry["input_length"]

    # Build background tensor
    if n_background is not None and input_length is not None:
        X = random_one_hot((n_background, 4, input_length)).float()
    else:
        X, _, err = _load_sequence_input(sequences, None)
        if err:
            return err

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
_FIMO_MISSING = {
    "error": "fimo not found on PATH",
    "fix": "Install the MEME Suite (https://meme-suite.org/meme/doc/download.html), "
           "put its bin directory on the PATH of the MCP server, and restart the "
           "MCP client.",
}


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
    motif_file: str = "",
    threshold: float = 1e-4,
    output_dir: str | None = None,
) -> str:
    """Run FIMO motif scanning on DNA sequences.

    Writes sequences to a temporary FASTA file, invokes the FIMO binary
    from the MEME Suite, and returns significant motif hits.

    Requires the ``fimo`` binary on PATH (from MEME Suite).  Without it the
    tool returns an error with the install hint instead of failing.

    Args:
        sequences: DNA sequences to scan (A/C/G/T).
        motif_file: Absolute path to a MEME-format motif file.
        threshold: FIMO p-value threshold (default 1e-4).
        output_dir: Directory for FIMO output.  Uses a temp dir if omitted.

    Returns:
        JSON with "hits" (up to 500), "total_hits", "output_dir" and
        "files" (sequences_fasta: the FASTA FIMO scanned; fimo_tsv: all
        hits).
    """
    import pandas as pd

    motif_path, err = _resolve_reference(motif_file, "motif_file")
    if err:
        return err
    # Basic path validation: motif_file is passed to subprocess
    if not motif_path.is_file():
        return json.dumps({"error": f"Not a regular file: {motif_file}"})

    bad = [i for i, s in enumerate(sequences) if not s or set(s) - set("ACGTacgt")]
    if bad:
        return _error(
            f"Sequences {bad[:10]} are empty or contain characters other than A, C, G, T.",
            "Remove or replace N and IUPAC codes before scanning.",
        )

    if shutil.which("fimo") is None:
        return json.dumps(_FIMO_MISSING)

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

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return json.dumps(_FIMO_MISSING)
    if result.returncode != 0:
        return _error(
            f"FIMO failed (exit {result.returncode})",
            "Read stderr; a malformed MEME motif file is the usual cause.",
            stderr=result.stderr[:1000],
        )

    # Parse results
    tsv_path = fimo_out / "fimo.tsv"
    if not tsv_path.exists():
        return _error("FIMO produced no output", "Read stderr and check the motif file.",
                      stderr=result.stderr[:500])

    df = pd.read_csv(tsv_path, sep="\t", comment="#")
    hits = df.to_dict(orient="records")

    return json.dumps({
        "hits": hits[:500],  # cap to avoid huge responses
        "total_hits": len(hits),
        "output_dir": str(fimo_out),
        "files": {"sequences_fasta": str(fasta_path), "fimo_tsv": str(tsv_path)},
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
    GPUs are visible, the default device and batch size, the default SLURM
    partition, and (after the first tool call that imports PyTorch) the
    torch version and CUDA status.

    Returns:
        JSON with on_slurm, slurm_job_id, slurm_partition (partition of the
        job the server itself runs in, null outside a job),
        default_partition ($TFBS_SLURM_PARTITION, else "gpu"; used by the
        SLURM tools unless a call passes partition=...), gpu_count,
        default_device, default_batch_size, call_log (the $TFBS_CALL_LOG
        file, or null), results_dir (where the model-analysis tools write
        when called without output_dir: <project root>/results for a
        checkout or $TFBS_PROJECT_ROOT, else tfbs_results/ in the server's
        working directory), project_root (working directory of the
        validation and benchmark pipelines, against which their relative
        models_dir resolves: $TFBS_PROJECT_ROOT, else the checkout, else the
        server's working directory) and loaded_models.  Before PyTorch is
        imported it
        adds torch_loaded=false; afterwards torch_version, cuda_version
        (CUDA version of the PyTorch build, null for CPU builds) and
        cuda_available, plus cuda_device_name and cuda_device_count when
        CUDA is available.
    """
    info = dict(_ENV)  # shallow copy
    info["call_log"] = os.environ.get(CALL_LOG_ENV, "").strip() or None
    info["results_dir"] = str(_results_root())
    info["project_root"] = str(_project_root())

    # Add torch details if already loaded
    if _torch is not None:
        info["torch_version"] = _torch.__version__
        info["cuda_version"] = _torch.version.cuda
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
    script_path: str | None = None,
    args: list[str] | None = None,
    sbatch_flags: list[str] | None = None,
    working_dir: str | None = None,
    script_body: str | None = None,
) -> str:
    """Submit a batch job to the SLURM scheduler via sbatch.

    Accepts either an existing script (script_path) or the text of a
    script (script_body), which is written to a temporary .sh file first.
    Builds and executes: ``sbatch [sbatch_flags...] <script> [args...]``

    Args:
        script_path: Absolute path to an existing sbatch script.  Give
                     exactly one of script_path and script_body.
        args: Extra arguments appended after the script path.
        sbatch_flags: Extra flags inserted before the script path
                      (e.g. ["--partition=gpu", "--gres=gpu:1"]).
        working_dir: Directory to run sbatch from, which becomes the job's
                     working directory (where slurm-<job_id>.out lands
                     unless the script sets --output).  Defaults to the
                     script's parent directory for script_path and to the
                     server's current directory for script_body.
        script_body: Full text of a batch script, e.g.
                     "#!/bin/bash\\n#SBATCH --time=01:00:00\\nhostname".
                     A body without a #! first line gets "#!/bin/bash"
                     prepended, since sbatch rejects scripts without one.

    Returns:
        JSON with job_id, script_path (the script actually submitted; for
        script_body, the temporary file), submit_command, and any stderr
        from sbatch.
    """
    if (script_path is None) == (script_body is None):
        return json.dumps({
            "error": "Give exactly one of script_path and script_body.",
            "fix": "Pass script_path for an existing script file, or "
                   "script_body with the text of the script.",
        })

    if shutil.which("sbatch") is None:
        return json.dumps({
            "error": "sbatch binary not found on PATH",
            "fix": "Run the SLURM tools on a cluster login node where sbatch "
                   "is on the PATH of the MCP server.",
        })

    if working_dir is not None and not Path(working_dir).is_dir():
        return json.dumps({
            "error": f"working_dir is not a directory: {working_dir}",
            "fix": "Pass an existing directory, or omit working_dir.",
        })

    if script_body is not None:
        body = script_body if script_body.startswith("#!") else "#!/bin/bash\n" + script_body
        fd, tmp_name = tempfile.mkstemp(prefix="tfbs_job_", suffix=".sh")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        os.chmod(tmp_name, 0o755)
        script = Path(tmp_name).resolve()
        cwd = working_dir or os.getcwd()
    else:
        script = Path(script_path).resolve()
        if not script.is_file():
            return json.dumps({
                "error": f"Script not found or not a file: {script_path}",
                "fix": "Pass the absolute path of an existing script, or pass "
                       "its text as script_body.",
            })
        cwd = working_dir or str(script.parent)

    cmd: list[str] = ["sbatch"]
    if sbatch_flags:
        cmd.extend(sbatch_flags)
    cmd.append(str(script))
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=cwd)
    except subprocess.TimeoutExpired:
        return json.dumps({
            "error": "sbatch timed out after 30 s",
            "fix": "The scheduler may be overloaded; check with squeue whether "
                   "the job was queued before submitting again.",
            "script_path": str(script),
            "submit_command": " ".join(cmd),
        })

    if result.returncode != 0:
        return json.dumps({
            "error": f"sbatch failed (exit {result.returncode})",
            "fix": "Read stderr; usually a partition, account or resource "
                   "request the cluster does not accept.",
            "stderr": result.stderr.strip(),
            "script_path": str(script),
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
        "script_path": str(script),
        "submit_command": " ".join(cmd),
        "stderr": result.stderr.strip() if result.stderr.strip() else None,
    })


# ---------------------------------------------------------------------------
# Tool 12: slurm_status
# ---------------------------------------------------------------------------
def _slurm_unavailable() -> str | None:
    """Error JSON when neither squeue nor sacct is on PATH, else None."""
    if shutil.which("squeue") is None and shutil.which("sacct") is None:
        return json.dumps({
            "error": "SLURM is not available on this machine "
                     "(squeue and sacct are not on PATH).",
            "fix": "Run the SLURM tools on a cluster login node where squeue "
                   "and sacct are on the PATH of the MCP server; the in-process "
                   "tools (prediction, attribution, data preparation) work here.",
        })
    return None


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
        exit_code (exit_code only available from sacct).  On a machine
        without SLURM (neither squeue nor sacct on PATH) it returns an
        error saying so; state NOT_FOUND means SLURM answered but does not
        know the job.
    """
    unavailable = _slurm_unavailable()
    if unavailable:
        return unavailable

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
        return json.dumps({
            "error": "sacct not available or timed out",
            "fix": "Run slurm_logs on a cluster login node where sacct is on "
                   "PATH, or read the job's --output/--error files directly.",
        })

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
        return json.dumps({
            "error": "scancel binary not found on PATH",
            "fix": "Run slurm_cancel on a cluster login node where scancel is "
                   "on the PATH of the MCP server.",
        })
    except subprocess.TimeoutExpired:
        return json.dumps({
            "error": "scancel timed out",
            "fix": "Check the job with slurm_status and retry the cancel.",
        })

    return json.dumps({
        "job_id": job_id,
        "cancelled": result.returncode == 0,
        "stderr": result.stderr.strip() if result.stderr.strip() else None,
    })


# ---------------------------------------------------------------------------
# Tool 15: slurm_watch
# ---------------------------------------------------------------------------
_INTERVAL_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

_SLURM_TERMINAL_STATES = [
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
]


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

    The server does not poll on its own.  This tool returns the status
    call to make next (next_check), the polling cadence (check_interval)
    and the states that end the watch; the MCP client decides when to call
    back, with whatever scheduling it has.

    Args:
        job_id: The SLURM job ID to monitor.
        check_interval: How often to check: a number with unit s, m, h or
                        d (e.g. "30m", "1h", "2h").
        context: Optional description of what the job is doing, to
                 include in status check messages.

    Returns:
        JSON with job_id, check_interval, check_interval_seconds, context,
        next_check ({"tool": "slurm_status", "arguments": {"job_id": ...}}),
        on_finish (the slurm_logs call for the end of the job),
        terminal_states, and plain-language instructions.  Returns an error
        if SLURM is not available on this machine.
    """
    unavailable = _slurm_unavailable()
    if unavailable:
        return unavailable

    match = re.fullmatch(r"\s*(\d+)\s*([smhd])\s*", check_interval.lower())
    seconds = int(match.group(1)) * _INTERVAL_UNIT_SECONDS[match.group(2)] if match else 0
    if seconds <= 0:
        return json.dumps({
            "error": f"Cannot read check_interval {check_interval!r}.",
            "fix": "Use a positive number followed by s, m, h or d, e.g. \"30m\" or \"2h\".",
        })

    next_check = {"tool": "slurm_status", "arguments": {"job_id": job_id}}
    on_finish = {"tool": "slurm_logs", "arguments": {"job_id": job_id, "log_type": "stderr"}}
    instructions = (
        f"Check SLURM job {job_id} every {check_interval}: call slurm_status "
        f"with job_id='{job_id}' (next_check). While the state is PENDING or "
        "RUNNING, check again after the interval. When the state is one of "
        f"terminal_states, call slurm_logs with job_id='{job_id}' and "
        "log_type='stderr' (on_finish), report the final state and exit code "
        "to the user, and stop checking. "
        "Optional: in Claude Code, CronCreate or /schedule can run these "
        "checks on a timer."
    )

    return json.dumps({
        "job_id": job_id,
        "check_interval": check_interval,
        "check_interval_seconds": seconds,
        "context": context,
        "next_check": next_check,
        "on_finish": on_finish,
        "terminal_states": _SLURM_TERMINAL_STATES,
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
    partition: str | None = None,
    gpu_type: str | None = None,
    max_time: str = "48:00:00",
    job_name: str | None = None,
    extra_args: list[str] | None = None,
) -> str:
    """Submit a TFBS model training job to SLURM.

    Two modes of operation:

    **Script mode**: provide ``script`` to submit an existing sbatch
    script directly.

    **Config mode**: provide ``config_path`` to auto-generate an
    sbatch script that invokes the Lightning CLI trainer with the
    given YAML config.

    Args:
        script: Path to an existing sbatch script (script mode).
        config_path: Path to a Lightning CLI YAML config (config mode).
        partition: SLURM partition (config mode).  Defaults to the server's
                   default_partition: $TFBS_SLURM_PARTITION, else "gpu"
                   (see tfbs_server_info), because the job requests a GPU and
                   "gpu" is the usual name of a cluster's GPU partition.
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
    partition = _resolve_partition(partition)

    gres_line = ""
    if gpu_type:
        gres_line = f"#SBATCH --gres=gpu:{gpu_type}:1"
    else:
        gres_line = "#SBATCH --gres=gpu:1"

    train_cmd = f"{sys.executable} -m tfbs.cli.train_cli fit --config {config_p}"
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
# The --model choices of tfbs.pipelines.run_sweep.
_SWEEP_MODELS = ("RNN", "VCNNBpnet", "BPNetReal")


def _config_dir() -> Path:
    """Directory of the packaged training templates (tfbs/configs)."""
    from tfbs.configs import config_dir

    return config_dir()


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
    partition: str | None = None,
    gpu_type: str | None = None,
) -> str:
    """Create a WandB sweep and submit an agent job to SLURM.

    Steps:
      1. Check that sbatch and wandb exist, so no sweep is created that no
         agent can run.
      2. Run ``wandb sweep`` to create the sweep and obtain a sweep ID.
      3. Generate a temporary sbatch script that launches a sweep agent
         (``python -m tfbs.pipelines.run_sweep``, installed with the
         package).
      4. Submit the agent job via sbatch from the project root.

    The search method and space come from the sweep YAML.  The bundled
    one, tfbs/configs/wandb_sweep_config.yaml in the installed package
    (its absolute path is in the error of a call with a missing
    sweep_config), uses Bayesian search over the parameters the agent
    reads for RNN, VCNNBpnet and BPNetReal.  The agent uses the W&B
    entity in $WANDB_ENTITY, else the logged-in user's default.

    Args:
        sweep_config: Path to a WandB sweep YAML config file.
        project_name: WandB project name for the sweep.
        model: Model architecture: "RNN", "VCNNBpnet" or "BPNetReal"
               (checked before any sweep is created).
        data_path: Path to the training data directory.
        data_name: Dataset name (e.g. "all_mean").
        scaling_method: Label scaling method (default "standardize").
        n_trials: Number of sweep trials to run (default 50).
        partition: SLURM partition.  Defaults to the server's
                   default_partition: $TFBS_SLURM_PARTITION, else "gpu"
                   (see tfbs_server_info), because the job requests a GPU and
                   "gpu" is the usual name of a cluster's GPU partition.
        gpu_type: GPU type constraint (e.g. "a100").

    Returns:
        JSON with sweep_id, job_id, script_path, and submit_command.
    """
    sweep_p = Path(sweep_config).resolve()
    if not sweep_p.is_file():
        return json.dumps({
            "error": f"Sweep config not found: {sweep_config}",
            "fix": "Pass the path of a WandB sweep YAML, e.g. the bundled "
                   f"{_config_dir() / 'wandb_sweep_config.yaml'}.",
        })

    if shutil.which("sbatch") is None:
        return json.dumps({
            "error": "sbatch binary not found on PATH",
            "fix": "Run tfbs_sweep on a cluster login node where sbatch is on "
                   "the PATH of the MCP server. No sweep was created.",
        })

    if model not in _SWEEP_MODELS:
        return json.dumps({
            "error": f"The sweep agent does not support model '{model}'.",
            "fix": f"Use one of {list(_SWEEP_MODELS)}, the architectures whose "
                   "hyperparameters the bundled sweep YAML defines. No sweep "
                   "was created.",
        })

    project_root, err = _work_dir()
    if err:
        return err

    if shutil.which("wandb") is None:
        return json.dumps({
            "error": "wandb binary not found on PATH",
            "fix": "pip install 'tfbs-mcp[tracking]' and run 'wandb login'.",
        })

    # Step 2: Create the sweep
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

    # Step 3: Build sbatch script for the sweep agent
    partition = _resolve_partition(partition)
    gres_line = ""
    if gpu_type:
        gres_line = f"#SBATCH --gres=gpu:{gpu_type}:1"
    else:
        gres_line = "#SBATCH --gres=gpu:1"

    q = shlex.quote
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=sweep_{sweep_id}
#SBATCH --partition={partition}
{gres_line}
#SBATCH --time=48:00:00
#SBATCH --output=sweep_{sweep_id}_%j.out
#SBATCH --error=sweep_{sweep_id}_%j.err
#SBATCH --ntasks=1

{_sbatch_pythonpath_line()}{shlex.join(_pipeline_argv("run_sweep"))} \\
    --model {q(model)} \\
    --data-path {q(data_path)} \\
    --data-name {q(data_name)} \\
    --scaling-method {q(scaling_method)} \\
    --n-trials {n_trials} \\
    --sweep-id {q(sweep_id)} \\
    --project-name {q(project_name)}
"""

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="tfbs_", newline="\n",
    )
    tmp.write(sbatch_content)
    tmp.close()
    os.chmod(tmp.name, 0o755)

    # Step 4: Submit via sbatch
    result = await slurm_submit(script_path=tmp.name, working_dir=str(project_root))
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
    partition: str | None = None,
    gpu_type: str | None = None,
    max_time: str = "02:00:00",
    job_name: str | None = None,
) -> str:
    """Submit a model validation job to SLURM.

    The job runs ``python -m tfbs.pipelines.validate`` (installed with
    the package), which loads the checkpoint, runs trainer.test() on the
    held-out test split, and appends test_loss to a CSV results file.
    The job's .o/.e logs are written next to that file.

    Args:
        checkpoint_path: Path to the .ckpt model checkpoint.
        data_path: Path to the dataset (tensor directory or CSV).
        model_name: Model class name (e.g. "VCNNBpnet", "RNN").
        scaling_method: Label scaling ("standardize", "normalize", "none").
        output_file: CSV file to append results to.  Defaults to
                     validation_results.csv in the results directory
                     (results_dir in tfbs_server_info).
        partition: SLURM partition.  Defaults to the server's
                   default_partition: $TFBS_SLURM_PARTITION, else "gpu"
                   (see tfbs_server_info), because the job requests a GPU and
                   "gpu" is the usual name of a cluster's GPU partition.
        gpu_type: GPU gres string (e.g. "nvidia_a100_80gb_pcie:1").
                  If omitted, requests 1 generic GPU.
        max_time: SLURM wall time (default "02:00:00").
        job_name: SLURM job name (default "tfbs_validate").

    Returns:
        JSON with job_id, the generated sbatch script, and the
        validation command.
    """
    ckpt = Path(checkpoint_path).resolve()
    if not ckpt.exists():
        return json.dumps({"error": f"Checkpoint not found: {checkpoint_path}"})

    data = Path(data_path).resolve()
    if not data.exists():
        return json.dumps({"error": f"Data path not found: {data_path}"})

    project_root, err = _work_dir()
    if err:
        return err

    csv_path = (Path(output_file) if output_file
                else _results_root() / "validation_results.csv").resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = str(csv_path)
    log_stem = csv_path.parent / "validate"

    gres = f"gpu:{gpu_type}" if gpu_type else "gpu:1"
    name = job_name or "tfbs_validate"
    partition = _resolve_partition(partition)

    validate_cmd = shlex.join([
        *_pipeline_argv("validate"),
        "--checkpoint_path", str(ckpt),
        "--model_name", model_name,
        "--data_path", str(data),
        "--scaling_method", scaling_method,
        "--output_file", output_file,
    ])

    sbatch_script = (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name={name}\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --nodes=1\n"
        f"#SBATCH --cpus-per-task=8\n"
        f"#SBATCH --mem=64G\n"
        f"#SBATCH --gres={gres}\n"
        f"#SBATCH --time={max_time}\n"
        f"#SBATCH --output={log_stem}.o%j\n"
        f"#SBATCH --error={log_stem}.e%j\n"
        f"\ndate\n"
        f"{_sbatch_pythonpath_line()}"
        f"{validate_cmd}\n"
        f"date\n"
    )

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="tfbs_validate_", newline="\n",
    )
    tmp.write(sbatch_script)
    tmp.close()
    os.chmod(tmp.name, 0o755)

    submit_result = json.loads(await slurm_submit(
        script_path=tmp.name,
        working_dir=str(project_root),
    ))

    submit_result["mode"] = "validate"
    submit_result["validate_command"] = validate_cmd
    submit_result["sbatch_script"] = sbatch_script
    return json.dumps(submit_result)


# ---------------------------------------------------------------------------
# Tool 19: tfbs_config (generate LightningCLI YAML config)
# ---------------------------------------------------------------------------
_SCALING_METHODS = ("standardize", "normalize", "none")

# Shown in the tool result so the LLM can relay architecture caveats.
_ARCH_NOTES = {
    "BPNetReal": "Count head only: predicts one scalar per sequence, no profile track.",
    "EquiNet": "Needs the optional equirc package on the training node (pip install equirc).",
    "FlexibleMLP": "input_length is set to 4 x window: the MLP sizes its first layer "
                   "from the flattened one-hot input.",
    "MultiCNN": "forward(x, x_num) needs a numerical covariate tensor, which "
                "TFBSDataModule does not supply; training with this config fails "
                "until a data module provides covariates.",
}


def _model_yaml_dir() -> Path:
    """Directory holding one LightningCLI model template per architecture."""
    return _config_dir() / "models"


def _model_templates() -> dict[str, Path]:
    d = _model_yaml_dir()
    if not d.is_dir():
        return {}
    return {p.stem: p for p in sorted(d.glob("*.yaml"))}


def _dataset_seq_shape(data: Path) -> tuple[tuple[int, ...] | None, str | None]:
    """Shape of <data>/train/seqs.pt, or (None, reason) when it cannot be read."""
    seqs = data / "train" / "seqs.pt"
    if not seqs.is_file():
        return None, f"{seqs} not found"
    import torch

    try:
        t = torch.load(seqs, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        try:
            t = torch.load(seqs, map_location="cpu", weights_only=True)
        except Exception as exc:
            return None, f"could not read {seqs}: {exc}"
    if not hasattr(t, "shape") or len(t.shape) < 2:
        return None, f"{seqs} is not a (N, 4, L) or (N, L) tensor"
    return tuple(int(s) for s in t.shape), None


def _window_args(model_type: str, window: int) -> dict[str, int]:
    """init_args that depend on the sequence window for this architecture."""
    if model_type == "FlexibleMLP":
        return {"input_length": 4 * window}
    return {"input_length": window}


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
    input_length: int | None = None,
    scaling_method: str | None = None,
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

    Writes a complete config (model, data and trainer sections) that
    tfbs_train runs as is.  The model section is the architecture's
    template tfbs/configs/models/<model_type>.yaml (installed with the
    package), with input_length, learning_rate and classify set from the
    arguments and any model_params merged on top.  "Use a 51 bp window and BPNetReal"
    is model_type="BPNetReal", input_length=51.

    Supported architectures (model_type, case-insensitive):
      VCNNBpnet   dilated CNN with a dense head; recommended default for
                  short (24 bp) windows because it is the architecture of
                  the NKX2-1 study, and its template carries those tuned
                  hyperparameters.
      BPNet       BPNet-style dilated stack with a pooled dense head.
      BPNetReal   reference BPNet layout (residual dilated convolutions);
                  predicts one scalar per sequence, not a profile.
      RNN         convolution into a bidirectional GRU (template carries
                  the tuned NKX2-1 settings).
      CNN         multi-layer CNN with per-layer padding and dilation.
      SimpleCNN   two-layer CNN baseline.
      VCNN        CNN with adaptive pooling to a fixed length.
      FlexibleCNN CNN with global max pooling, per-layer channels/kernels.
      my_cnn      small two-layer CNN from early experiments.
      MLP         fully connected baseline on the flattened one-hot input.
      FlexibleMLP MLP with dropout; input_length is set to 4 x window.
      EquiNet     reverse-complement equivariant network; needs equirc.
      MultiCNN    CNN plus numerical covariates; the default data module
                  does not supply covariates, so it cannot train as is.
    Other templates' values are the class defaults.  The result echoes
    the resolved init_args, so the hyperparameters actually used are
    visible.  An unknown model_type returns the list of valid names.

    Defaults and why:
      input_length: omitted means the length of the sequences in
        <data_path>/train/seqs.pt, because the model must match the
        windows it is trained on.  Pass it to request a window (e.g. 51);
        a mismatch with the dataset is reported as a warning.
      scaling_method: "standardize" (z-score) for regression, because
        zero-mean unit-variance labels keep MSE gradients well scaled;
        "none" for classification, because BCE needs 0/1 labels.
        Combining classify=true with standardize or normalize is an error.
      batch_size 64: fits GPU memory for short one-hot windows and gives
        stable gradients on datasets of a few thousand sequences; lower
        it after an out-of-memory error.
      max_epochs 500: a ceiling high enough that early stopping, not the
        epoch count, ends training.
      patience 5: stops after 5 epochs without val_loss improvement,
        which avoids overfitting small datasets without cutting off
        slow early progress.
      learning_rate 0.001: the Adam default step size, a safe start
        because Adam adapts per-parameter steps from there.
      checkpoint_dir: saved_models/<data_name>/<scaling_method>, with
        filename best_<model_type>, because tfbs_remap_validation and
        tfbs_chipseq_benchmark look for
        <models_dir>/<dataset>/standardize/best_<model_type>.ckpt.

    Args:
        model_type: Architecture name (see list above).
        data_path: Dataset directory with train/val/test seqs.pt and
                   labels.pt (as written by tfbs_prepare_dataset).
        output_path: Where to write the YAML config file.
        input_length: Sequence window in bp (default: from the dataset).
        scaling_method: "standardize", "normalize" or "none" (default:
                        standardize for regression, none for classify).
        batch_size: Training batch size (default 64).
        max_epochs: Maximum training epochs (default 500).
        patience: Early stopping patience on val_loss (default 5).
        learning_rate: Adam learning rate (default 0.001).
        classify: If true, sigmoid output and BCE loss; else MSE
                  regression (default false).
        wandb_project: WandB project name.  If omitted, no WandB logger
                       is configured.
        checkpoint_dir: Directory for the best checkpoint (default above).
        checkpoint_filename: Checkpoint filename (default
                             "best_<model_type>").
        model_params: Architecture hyperparameters merged over the
                      template; keys must exist in the template (an
                      unknown key returns the allowed keys).

    Returns:
        JSON with "output_path", "model_type", "input_length",
        "input_length_source" ("argument" or "dataset"),
        "scaling_method", "model_init_args", "config_preview" (first 40
        lines), and "notes"/"warning" when they apply.
    """
    import yaml

    templates = _model_templates()
    if not templates:
        return json.dumps({
            "error": f"No model templates found in {_model_yaml_dir()}.",
            "fix": "Reinstall tfbs-mcp: the model templates ship inside the "
                   "package as tfbs/configs/models/*.yaml.",
        })
    by_lower = {name.lower(): name for name in templates}
    canonical = by_lower.get(str(model_type).lower())
    if canonical is None:
        return json.dumps({
            "error": f"Unknown model_type '{model_type}'.",
            "fix": f"Use one of: {sorted(templates)}.",
            "valid_model_types": sorted(templates),
        })
    model_type = canonical

    if scaling_method is None:
        scaling_method = "none" if classify else "standardize"
    if scaling_method not in _SCALING_METHODS:
        return json.dumps({
            "error": f"Unknown scaling_method '{scaling_method}'.",
            "fix": f"Use one of: {list(_SCALING_METHODS)}.",
        })
    if classify and scaling_method != "none":
        return json.dumps({
            "error": f"scaling_method '{scaling_method}' cannot be combined "
                     "with classify=true: BCE loss needs 0/1 labels.",
            "fix": "Omit scaling_method (classification defaults to 'none') "
                   "or pass scaling_method='none'.",
        })

    data = Path(data_path).resolve()
    if not data.exists():
        return json.dumps({
            "error": f"Data path not found: {data_path}",
            "fix": "Pass the dataset directory written by tfbs_prepare_dataset.",
        })

    seq_shape, shape_err = _dataset_seq_shape(data)
    warning = None
    if input_length is None:
        if seq_shape is None:
            return json.dumps({
                "error": f"input_length was omitted and could not be inferred: {shape_err}.",
                "fix": "Pass input_length, or point data_path at a dataset "
                       "directory with train/seqs.pt.",
            })
        input_length = seq_shape[-1]
        length_source = "dataset"
    else:
        length_source = "argument"
        if seq_shape is not None and seq_shape[-1] != input_length:
            warning = (
                f"input_length {input_length} differs from the dataset's "
                f"sequence length {seq_shape[-1]}; extract the loci with "
                f"in_window={input_length} or omit input_length."
            )
    if input_length < 1:
        return json.dumps({
            "error": f"input_length must be positive, got {input_length}.",
            "fix": "Pass the sequence window in bp, e.g. input_length=24.",
        })

    with open(templates[model_type]) as f:
        template = yaml.safe_load(f) or {}
    model_args = dict(template.get("init_args") or {})
    params = dict(model_params or {})
    reserved = {"input_length", "learning_rate", "classify"}
    misplaced = sorted(reserved & params.keys())
    if misplaced:
        return json.dumps({
            "error": f"model_params must not set {misplaced}.",
            "fix": "Pass input_length, learning_rate and classify as tool arguments.",
        })
    unknown = sorted(set(params) - set(model_args))
    if unknown:
        return json.dumps({
            "error": f"Unknown model_params for {model_type}: {unknown}.",
            "fix": f"Allowed keys: {sorted(set(model_args) - reserved)}.",
        })
    model_args.update(params)
    model_args.update(_window_args(model_type, input_length))
    model_args["learning_rate"] = learning_rate
    model_args["classify"] = classify

    data_name = data.name
    ckpt_dir = checkpoint_dir or str(
        _project_root() / "saved_models" / data_name / scaling_method
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

    config = {
        "seed_everything": 42,
        "trainer": trainer,
        "model": {
            "class_path": template.get("class_path", f"tfbs.nn.models.{model_type}"),
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

    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    with open(out) as f:
        lines = f.readlines()
    preview = "".join(lines[:40])

    result: dict[str, Any] = {
        "output_path": str(out),
        "model_type": model_type,
        "input_length": input_length,
        "input_length_source": length_source,
        "scaling_method": scaling_method,
        "model_init_args": model_args,
        "config_preview": preview,
    }
    if model_type in _ARCH_NOTES:
        result["notes"] = _ARCH_NOTES[model_type]
    if warning:
        result["warning"] = warning
    return json.dumps(result)

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
    bed_path: str | None = None,
    output_dir: str | None = None,
    genome_fasta: str = "",
    in_window: int = 200,
    max_sequences: int | None = None,
    score_column: int | None = 4,
    negatives: str | None = None,
    variants: list[str] | None = None,
    random_seed: int = 42,
) -> str:
    """Extract one-hot DNA windows from a reference genome: BED peaks,
    matched negatives for them, and ref/alt windows around variants.

    Peaks: reads a BED file, centres a window of in_window bp on each
    peak midpoint, drops windows that leave the chromosome or contain a
    base other than A/C/G/T, one-hot encodes the rest (N, 4, in_window)
    and optionally keeps a continuous score column for regression.
    Standard BED (3+ columns) and ChIP-Atlas aggregated BED (9 columns,
    MACS2 score in column 5) both work.

    Negatives (negatives=...): written to <output_dir>/negatives/
    sequences.pt, ready for tfbs_prepare_classification(positive_path=
    <output_dir>/sequences.pt, negative_path=<that file>).
      "gc_matched": one random genomic window per positive, of the same
        length, from the chromosomes that carry the positives (in
        proportion to their length).  A window is rejected if it
        overlaps a BED interval or a positive window padded by in_window
        bp on each side, overlaps an earlier negative, or contains a
        base other than A/C/G/T.  GC content is matched bin by bin (bins
        of 0.05 GC fraction): each bin gets as many negatives as it has
        positives.  Sampling is seeded with random_seed.  A bin that the
        genome cannot fill within the draw limit (200 candidate windows
        per positive, at least 200,000) is reported in
        negatives.unfilled_bins instead of being padded from other bins.
      "shuffled": a dinucleotide-shuffled copy of every positive
        (tangermeme), which keeps each window's dinucleotide content and
        removes its motifs.

    Variants (variants=[...], "chrom:pos:ref>alt" with a 1-based pos,
    e.g. "chr14:36516392:C>T"): for each variant, a window of in_window
    bp is taken from the genome with the variant at 0-based offset
    in_window // 2.  The genome base(s) at pos must equal ref; a
    mismatch (usually a different genome build) is listed in
    variants.ref_mismatches and the variant is not written.  The alt
    window is the ref window with ref replaced by alt (same-length
    substitutions only; indels are skipped).  Writes ref_sequences.pt
    and alt_sequences.pt (one row per extracted variant, same order)
    plus variants.json (every variant with its status, its row, its
    window coordinates and both window strings).  Score both files
    with tfbs_predict(sequences_path=...) and subtract alt from ref.
    A chromosome missing from the genome is retried with the "chr"
    prefix added or removed.

    Defaults and why:
      in_window 200: covers the positional uncertainty of a ChIP-seq
        summit with flanks to spare, for scoring, FIMO or window_mode
        prediction.  A model is trained and scored on its own input
        length, so pass in_window=input_length (24 for the VCNNBpnet
        models of the NKX2-1 study) when preparing training data or
        variant windows.
      negatives None: a negative set is needed only to train or test a
        classifier.  For that, "gc_matched" is the recommended choice,
        because genomic background of the same length and GC content
        keeps the classifier from separating the classes by base
        composition, and it avoids the cross-TF contamination that
        other TFs' peaks would bring.
      output_dir: <bed_dir>/<bed_name>_loci/ (bed_name without .bed or
        .gz), so the data tools downstream derive their own outputs
        from it; with variants and no BED,
        <results>/variants/<digest>/ (results_dir of tfbs_server_info,
        digest of the variants, window and genome).

    Requires pyfaidx and tangermeme: pip install 'tfbs-mcp[genomics]'

    Args:
        bed_path: Path to a BED file (tab-separated, at least 3 columns).
                  Needed for peaks and negatives; optional with variants.
        output_dir: Output directory (default above).
        genome_fasta: Path to reference genome FASTA file. Falls back
                      to $TFBS_GENOME_FASTA.
        in_window: Window size in bp (default 200, see above).
        max_sequences: If set, randomly sample this many peaks before
                       extraction (useful for large ChIP-Atlas files).
        score_column: 0-indexed BED column to extract as continuous scores
                      (default 4 = column 5).  Set to None to skip.
        negatives: None, "gc_matched" or "shuffled" (see above).
        variants: Variants as "chrom:pos:ref>alt" strings (see above).
        random_seed: Seed for max_sequences sampling and for the
                     negatives (default 42).

    Returns:
        JSON with output_dir and files (every path written).  With a
        BED: shape, num_peaks_input, num_peaks_extracted, num_filtered_n
        (windows with a non-ACGT base), num_filtered_bounds (unknown
        chromosome or window off the end), has_scores and
        sequences_path.  With negatives: negatives_path and negatives
        (mode, num_negatives, num_positives; for gc_matched also
        gc_bin_width, positive_gc_mean, negative_gc_mean, shortfall,
        candidates_drawn, unfilled_bins and seed).  With variants: variants (num_given,
        num_extracted, ref_mismatches as {variant, ref, genome_ref},
        skipped as {variant, reason}, and the per-variant records when
        there are at most 50), ref_sequences_path, alt_sequences_path
        and manifest_path.
    """
    if bed_path is None and not variants:
        return _error(
            "Nothing to extract.",
            "Pass bed_path (peaks), variants (\"chrom:pos:ref>alt\"), or both.",
        )
    if negatives not in (None, "gc_matched", "shuffled"):
        return _error(
            f"Unknown negatives '{negatives}'.",
            "Use negatives='gc_matched' (genomic background, same length and "
            "GC) or negatives='shuffled' (dinucleotide-shuffled positives).",
        )
    if negatives is not None and bed_path is None:
        return _error(
            "negatives needs positives from bed_path.",
            "Pass the peak BED as bed_path together with negatives.",
        )
    if in_window < 1:
        return _error(f"in_window must be positive, got {in_window}.",
                      "Pass the window length in bp, e.g. in_window=24.")

    bed = None
    if bed_path is not None:
        bed = Path(bed_path).expanduser().resolve()
        if not bed.is_file():
            return _error(f"BED file not found: {bed_path}",
                          "Pass the path of an existing BED file.")

    genome_path, err = _resolve_reference(genome_fasta, "genome_fasta")
    if err:
        return err

    try:
        _ensure_imports()
        _ensure_genomics_imports()
    except ImportError as exc:
        return _error(f"Missing dependency: {exc}",
                      "Install with: pip install 'tfbs-mcp[genomics]'")
    genome = _get_genome(str(genome_path))
    chrom_lengths = {str(name): len(rec) for name, rec in genome.items()}

    if output_dir:
        out = Path(output_dir).expanduser().resolve()
    elif bed is not None:
        stem = bed.name
        for suffix in (".gz", ".bed", ".narrowPeak", ".broadPeak", ".txt"):
            if stem.lower().endswith(suffix.lower()) and len(stem) > len(suffix):
                stem = stem[: -len(suffix)]
        out = bed.parent / f"{stem}_loci"
    else:
        out = _results_root() / "variants" / _digest(sorted(variants), in_window, str(genome_path))

    result: dict[str, Any] = {"output_dir": str(out)}
    files: dict[str, str] = {}

    if bed is not None:
        peaks = _extract_bed_windows(
            bed, genome, chrom_lengths, in_window, max_sequences, score_column, random_seed,
        )
        if isinstance(peaks, str):
            return peaks
        out.mkdir(parents=True, exist_ok=True)
        X_pos = _one_hot(peaks["seqs"])
        files["sequences"] = str(out / "sequences.pt")
        _torch.save(X_pos, files["sequences"])
        files["regions"] = str(out / "regions.bed")
        with open(files["regions"], "w", newline="\n") as f:
            for chrom, start, end in peaks["windows"]:
                f.write(f"{chrom}\t{start}\t{end}\n")
        has_scores = peaks["scores"] is not None
        if has_scores:
            files["scores"] = str(out / "scores.pt")
            _torch.save(
                _torch.tensor(peaks["scores"], dtype=_torch.float32).unsqueeze(1),
                files["scores"],
            )
        result.update({
            "shape": list(X_pos.shape),
            "num_peaks_input": peaks["num_input"],
            "num_peaks_extracted": int(X_pos.shape[0]),
            "num_filtered_n": peaks["num_filtered_n"],
            "num_filtered_bounds": peaks["num_filtered_bounds"],
            "has_scores": has_scores,
            "sequences_path": files["sequences"],
        })

        if negatives is not None:
            neg_dir = out / "negatives"
            neg_dir.mkdir(parents=True, exist_ok=True)
            if negatives == "shuffled":
                from tangermeme.ersatz import dinucleotide_shuffle
                X_neg = dinucleotide_shuffle(X_pos, n=1, random_state=random_seed)[:, 0].float()
                info: dict[str, Any] = {"mode": "shuffled"}
            else:
                picks, info = _sample_gc_matched(
                    genome, chrom_lengths, peaks, in_window, random_seed,
                )
                X_neg = _one_hot([p[2] for p in picks])
                files["negative_regions"] = str(neg_dir / "regions.bed")
                with open(files["negative_regions"], "w", newline="\n") as f:
                    for k, (chrom, start, _seq, gc) in enumerate(picks):
                        f.write(f"{chrom}\t{start}\t{start + in_window}\tneg_{k}\t{gc:.4f}\n")
                info = {"mode": "gc_matched", **info}
            files["negatives"] = str(neg_dir / "sequences.pt")
            _torch.save(X_neg, files["negatives"])
            info["num_negatives"] = int(X_neg.shape[0])
            info["num_positives"] = int(X_pos.shape[0])
            result["negatives"] = info
            result["negatives_path"] = files["negatives"]

    if variants:
        out.mkdir(parents=True, exist_ok=True)
        var = _extract_variant_windows(genome, chrom_lengths, variants, in_window)
        manifest = {
            "genome_fasta": str(genome_path),
            "in_window": in_window,
            "variant_offset": in_window // 2,
            "variants": var["records"],
        }
        if var["ref"]:
            files["ref_sequences"] = str(out / "ref_sequences.pt")
            files["alt_sequences"] = str(out / "alt_sequences.pt")
            _torch.save(_one_hot(var["ref"]), files["ref_sequences"])
            _torch.save(_one_hot(var["alt"]), files["alt_sequences"])
            result["ref_sequences_path"] = files["ref_sequences"]
            result["alt_sequences_path"] = files["alt_sequences"]
        files["variant_manifest"] = str(out / "variants.json")
        with open(files["variant_manifest"], "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        result["manifest_path"] = files["variant_manifest"]
        summary: dict[str, Any] = {
            "num_given": len(variants),
            "num_extracted": len(var["ref"]),
            "ref_mismatches": var["mismatches"],
            "skipped": var["skipped"],
        }
        if len(variants) <= 50:
            summary["records"] = var["records"]
        result["variants"] = summary

    result["files"] = files
    return json.dumps(result)


def _extract_bed_windows(bed, genome, chrom_lengths, in_window, max_sequences,
                         score_column, random_seed) -> dict[str, Any] | str:
    """Windows centred on the BED intervals; an error JSON string on failure."""
    try:
        bed_df = _pd.read_csv(bed, sep="\t", header=None, comment="#")
    except (_pd.errors.EmptyDataError, _pd.errors.ParserError, UnicodeDecodeError) as exc:
        return _error(f"Could not read BED file {bed}: {exc}",
                      "Pass a tab-separated BED file with chrom, start and end columns.")
    if bed_df.shape[1] < 3:
        return _error(f"{bed} has {bed_df.shape[1]} column(s), need at least 3.",
                      "Pass a tab-separated BED file with chrom, start and end columns.")
    try:
        chroms = bed_df.iloc[:, 0].astype(str).to_numpy()
        starts = bed_df.iloc[:, 1].astype(int).to_numpy()
        ends = bed_df.iloc[:, 2].astype(int).to_numpy()
    except (ValueError, TypeError) as exc:
        return _error(f"Non-integer start or end in {bed}: {exc}",
                      "Remove header lines that do not start with #, and check columns 2 and 3.")
    num_input = len(bed_df)

    scores_raw = None
    if score_column is not None and bed_df.shape[1] > score_column:
        try:
            scores_raw = bed_df.iloc[:, score_column].astype(float).to_numpy()
        except (ValueError, TypeError):
            scores_raw = None

    order = _np.arange(num_input)
    if max_sequences and num_input > max_sequences:
        order = _np.sort(_np.random.RandomState(random_seed).choice(
            num_input, size=max_sequences, replace=False,
        ))

    half = in_window // 2
    seqs: list[str] = []
    windows: list[tuple[str, int, int]] = []
    kept_scores: list[float] = []
    n_bounds = n_bad = 0
    for k in order:
        chrom = chroms[k]
        if chrom not in chrom_lengths:
            n_bounds += 1
            continue
        mid = int(starts[k]) + (int(ends[k]) - int(starts[k])) // 2
        ws = mid - half
        we = ws + in_window
        if ws < 0 or we > chrom_lengths[chrom]:
            n_bounds += 1
            continue
        if scores_raw is not None and _np.isnan(scores_raw[k]):
            n_bad += 1
            continue
        seq = genome[chrom][ws:we].seq.upper()
        if seq.strip("ACGT"):
            n_bad += 1
            continue
        seqs.append(seq)
        windows.append((chrom, ws, we))
        if scores_raw is not None:
            kept_scores.append(float(scores_raw[k]))

    if not seqs:
        return _error(
            "No valid peaks remaining after filtering.",
            "Check that the BED chromosome names match the genome FASTA and "
            "that the windows lie inside the chromosomes.",
            num_peaks_input=num_input,
            num_filtered_bounds=n_bounds,
            num_filtered_n=n_bad,
        )
    return {
        "seqs": seqs,
        "windows": windows,
        "scores": kept_scores if scores_raw is not None else None,
        "num_input": num_input,
        "num_filtered_n": n_bad,
        "num_filtered_bounds": n_bounds,
        "bed_intervals": list(zip(chroms.tolist(), starts.tolist(), ends.tolist())),
    }


_GC_BIN_WIDTH = 0.05


def _merged_intervals(intervals: list[tuple[int, int]]):
    merged: list[list[int]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    arr = _np.array(merged, dtype=_np.int64).reshape(-1, 2)
    return arr[:, 0], arr[:, 1]


def _sample_gc_matched(genome, chrom_lengths, peaks, window, seed):
    """Random genomic windows matching the positives' GC histogram.

    Returns (picks, info): picks as (chrom, start, sequence, gc) and info
    with the GC summary and any bins the draw limit left unfilled.
    """
    import bisect

    rng = _np.random.RandomState(seed)
    n_bins = int(round(1 / _GC_BIN_WIDTH))

    def _bin(gc):
        return _np.minimum((_np.asarray(gc) / _GC_BIN_WIDTH + 1e-9).astype(int), n_bins - 1)

    pos_gc = _np.array([(s.count("G") + s.count("C")) / window for s in peaks["seqs"]])
    need = _np.bincount(_bin(pos_gc), minlength=n_bins)
    wanted = need.copy()

    excl: dict[str, list[tuple[int, int]]] = {}
    for chrom, s, e in peaks["bed_intervals"]:
        excl.setdefault(chrom, []).append((int(s) - window, int(e) + window))
    for chrom, s, e in peaks["windows"]:
        excl.setdefault(chrom, []).append((s - window, e + window))
    excl_arr = {c: _merged_intervals(v) for c, v in excl.items()}

    chroms = sorted({c for c, _, _ in peaks["windows"] if chrom_lengths[c] >= window})
    lengths = _np.array([chrom_lengths[c] for c in chroms], dtype=float)
    probs = lengths / lengths.sum()
    block = max(100 * window, 100_000)
    max_draws = max(200_000, 200 * len(pos_gc))
    acgt = _np.frombuffer(b"ACGT", dtype=_np.uint8)
    gc_codes = _np.frombuffer(b"GC", dtype=_np.uint8)

    chosen: dict[str, list[int]] = {c: [] for c in chroms}
    picks: list[tuple[str, int, str, float]] = []
    draws = 0
    while need.sum() > 0 and draws < max_draws:
        chrom = chroms[rng.choice(len(chroms), p=probs)]
        clen = chrom_lengths[chrom]
        bstart = int(rng.randint(0, clen - block + 1)) if clen > block else 0
        seq = genome[chrom][bstart:min(clen, bstart + block)].seq.upper()
        arr = _np.frombuffer(seq.encode("ascii", "replace"), dtype=_np.uint8)
        n_win = len(arr) - window + 1
        if n_win <= 0:
            draws += 1
            continue
        gc_cum = _np.concatenate([[0], _np.cumsum(_np.isin(arr, gc_codes))])
        bad_cum = _np.concatenate([[0], _np.cumsum(~_np.isin(arr, acgt))])
        n_cand = max(1, min(n_win, len(arr) // window))
        local = rng.randint(0, n_win, size=n_cand)
        draws += n_cand
        ok = (bad_cum[local + window] - bad_cum[local]) == 0
        starts = local + bstart
        if chrom in excl_arr:
            ex_s, ex_e = excl_arr[chrom]
            idx = _np.searchsorted(ex_e, starts, side="right")
            hit = idx < len(ex_s)
            hit[hit] = ex_s[idx[hit]] < starts[hit] + window
            ok &= ~hit
        gc = (gc_cum[local + window] - gc_cum[local]) / window
        bins = _bin(gc)
        taken = chosen[chrom]
        for j in _np.flatnonzero(ok):
            b = bins[j]
            if need[b] == 0:
                continue
            s = int(starts[j])
            i = bisect.bisect_left(taken, s)
            if (i < len(taken) and taken[i] < s + window) or (i > 0 and taken[i - 1] + window > s):
                continue
            taken.insert(i, s)
            lj = int(local[j])
            picks.append((chrom, s, seq[lj:lj + window], float(gc[j])))
            need[b] -= 1

    unfilled = [
        {"gc_range": [round(b * _GC_BIN_WIDTH, 2), round((b + 1) * _GC_BIN_WIDTH, 2)],
         "wanted": int(wanted[b]), "missing": int(need[b])}
        for b in range(n_bins) if need[b] > 0
    ]
    neg_gc = [p[3] for p in picks]
    info = {
        "gc_bin_width": _GC_BIN_WIDTH,
        "positive_gc_mean": round(float(pos_gc.mean()), 4),
        "negative_gc_mean": round(float(_np.mean(neg_gc)), 4) if neg_gc else None,
        "shortfall": int(need.sum()),
        "candidates_drawn": int(draws),
        "unfilled_bins": unfilled,
        "seed": seed,
    }
    if unfilled:
        info["warning"] = (
            f"{int(need.sum())} positives have no GC-matched negative: the "
            "genome outside the positives had too few windows in those GC bins."
        )
    return picks, info


def _extract_variant_windows(genome, chrom_lengths, variants, window) -> dict[str, Any]:
    """ref/alt windows for "chrom:pos:ref>alt" variants, with per-variant status."""
    offset = window // 2
    ref_seqs: list[str] = []
    alt_seqs: list[str] = []
    records: list[dict[str, Any]] = []
    mismatches: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    def _skip(rec, status, reason):
        rec["status"] = status
        rec["reason"] = reason
        skipped.append({"variant": rec["variant"], "reason": reason})
        records.append(rec)

    for text in variants:
        rec: dict[str, Any] = {"variant": text}
        parts = str(text).strip().rsplit(":", 2)
        alleles = parts[2].split(">") if len(parts) == 3 else []
        if len(parts) != 3 or not parts[1].isdigit() or len(alleles) != 2:
            _skip(rec, "invalid", "expected chrom:pos:ref>alt with a 1-based integer pos")
            continue
        chrom, pos = parts[0], int(parts[1])
        ref, alt = alleles[0].upper(), alleles[1].upper()
        if not ref or not alt or (set(ref) | set(alt)) - set("ACGT"):
            _skip(rec, "invalid", "ref and alt must be A/C/G/T")
            continue
        if len(ref) != len(alt):
            _skip(rec, "unsupported", "indel: only same-length substitutions keep the window length")
            continue
        if chrom not in chrom_lengths:
            other = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
            if other not in chrom_lengths:
                _skip(rec, "unknown_chrom", f"{chrom} is not in the genome FASTA")
                continue
            chrom = other
        rec.update({"chrom": chrom, "pos": pos, "ref": ref, "alt": alt})
        p0 = pos - 1
        ws = p0 - offset
        we = ws + window
        if pos < 1 or ws < 0 or we > chrom_lengths[chrom] or offset + len(ref) > window:
            _skip(rec, "out_of_bounds", "the window leaves the chromosome or cannot hold ref")
            continue
        genome_ref = genome[chrom][p0:p0 + len(ref)].seq.upper()
        rec["genome_ref"] = genome_ref
        if genome_ref != ref:
            rec["status"] = "ref_mismatch"
            records.append(rec)
            mismatches.append({"variant": text, "ref": ref, "genome_ref": genome_ref})
            continue
        ref_window = genome[chrom][ws:we].seq.upper()
        if ref_window.strip("ACGT"):
            _skip(rec, "contains_n", "the window contains a base other than A/C/G/T")
            continue
        alt_window = ref_window[:offset] + alt + ref_window[offset + len(ref):]
        rec.update({
            "status": "ok", "row": len(ref_seqs), "window_start": ws, "window_end": we,
            "ref_window": ref_window, "alt_window": alt_window,
        })
        records.append(rec)
        ref_seqs.append(ref_window)
        alt_seqs.append(alt_window)
    return {"ref": ref_seqs, "alt": alt_seqs, "records": records,
            "mismatches": mismatches, "skipped": skipped}


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
    output_dir: str | None = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    add_rc: bool = False,
    random_seed: int = 42,
) -> str:
    """Split sequences and labels into train/val/test directories.

    Creates the directory structure expected by TFBSDataModule:
    {output_dir}/{train,val,test}/{seqs.pt, labels.pt}

    Defaults and why:
      test_size 0.2, then val_size 0.2 of the rest: a 64/16/20
        train/val/test split, which keeps a fifth of the data unseen for
        a stable test ROC-AUC while leaving most of it for training.
      add_rc false: every split keeps the size of the input; true
        appends the reverse complement of every sequence to its own
        split (doubling each), which helps models that are not
        strand-symmetric, without leaking a sequence across splits.
      random_seed 42: the same input always gives the same split.
      output_dir: <name>_dataset/ next to the input's directory, with
        <name> that directory's name minus a _loci, _classification or
        _regression suffix (peaks_classification/sequences.pt ->
        peaks_dataset/), so tfbs_config names the checkpoint after the
        peak set.

    Args:
        sequences_path: Path to sequences.pt tensor (N, 4, L).
        labels_path: Path to labels.pt tensor (N, 1) or (N,).
        output_dir: Base directory for the split dataset (default above).
        test_size: Fraction for test set (default 0.2).
        val_size: Fraction of remaining for validation (default 0.2).
        add_rc: Add reverse complement augmentation (doubles dataset).
        random_seed: Random seed for reproducible splits (default 42).

    Returns:
        JSON with output_dir, train_count, val_count, test_count,
        label_stats and files (the six seqs.pt / labels.pt paths).
    """
    from sklearn.model_selection import train_test_split

    _ensure_imports()

    seqs_p = Path(sequences_path).resolve()
    labels_p = Path(labels_path).resolve()
    if not seqs_p.exists():
        return _error(f"Sequences file not found: {sequences_path}",
                      "Pass the sequences.pt written by tfbs_prepare_classification "
                      "or tfbs_prepare_regression.")
    if not labels_p.exists():
        return _error(f"Labels file not found: {labels_path}",
                      "Pass the labels.pt written next to the sequences.")
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else _sibling_dir(seqs_p, "dataset")

    seqs = _torch.load(seqs_p, weights_only=True)
    labels = _torch.load(labels_p, weights_only=True)
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)

    n = len(seqs)
    if len(labels) != n:
        return _error(f"{n} sequences but {len(labels)} labels.",
                      "Pass the labels.pt written together with these sequences.")
    if n < 3:
        return _error(
            f"Need at least 3 sequences for train/val/test split, got {n}.",
            "Extract more peaks, or combine positives and negatives first.",
        )

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

    files: dict[str, str] = {}

    def _save_split(split_name, idx):
        s = seqs[idx]
        l = labels[idx]
        if add_rc:
            rc = _torch.flip(s, [2])
            rc = _torch.index_select(rc, 1, _torch.tensor([3, 2, 1, 0], dtype=_torch.long))
            s = _torch.cat([s, rc], dim=0)
            l = _torch.cat([l, l], dim=0)
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        _torch.save(s, split_dir / "seqs.pt")
        _torch.save(l, split_dir / "labels.pt")
        files[f"{split_name}_seqs"] = str(split_dir / "seqs.pt")
        files[f"{split_name}_labels"] = str(split_dir / "labels.pt")
        return len(s)

    train_n = _save_split("train", train_idx)
    val_n = _save_split("val", val_idx)
    test_n = _save_split("test", test_idx)

    all_labels = labels.numpy().flatten()
    return json.dumps({
        "output_dir": str(out_dir),
        "train_count": train_n,
        "val_count": val_n,
        "test_count": test_n,
        "label_stats": {
            "min": float(all_labels.min()),
            "max": float(all_labels.max()),
            "mean": float(all_labels.mean()),
            "std": float(all_labels.std()),
        },
        "files": files,
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
    output_dir: str | None = None,
    balance: bool = True,
    random_seed: int = 42,
) -> str:
    """Combine positive and negative sequence tensors into a binary dataset.

    Loads two sequence tensor files, assigns labels 1.0 (positive) and
    0.0 (negative), optionally balances by subsampling the larger set,
    shuffles, and saves.  The usual inputs come from one
    tfbs_extract_loci call with negatives="gc_matched":
    positive_path=<loci>/sequences.pt and
    negative_path=<loci>/negatives/sequences.pt (its negatives_path).

    Output (sequences.pt + labels.pt) feeds into tfbs_prepare_dataset.

    Defaults and why:
      balance true: equal class sizes keep the 0.5 decision threshold
        and the PR-AUC baseline meaningful.
      output_dir: <name>_classification/ next to the positives'
        directory, with <name> that directory's name minus a _loci
        suffix (peaks_loci/sequences.pt -> peaks_classification/).

    Args:
        positive_path: Path to positive sequences.pt (N_pos, 4, L).
        negative_path: Path to negative sequences.pt (N_neg, 4, L).
        output_dir: Directory to save combined sequences.pt and labels.pt
                    (default above).
        balance: Subsample larger set to match smaller (default true).
        random_seed: Random seed (default 42).

    Returns:
        JSON with output_dir, num_positive, num_negative, total and
        files (sequences, labels).
    """
    _ensure_imports()

    pos_p = Path(positive_path).resolve()
    neg_p = Path(negative_path).resolve()
    if not pos_p.exists():
        return _error(f"Positive file not found: {positive_path}",
                      "Pass the sequences_path returned by tfbs_extract_loci.")
    if not neg_p.exists():
        return _error(f"Negative file not found: {negative_path}",
                      "Pass the negatives_path returned by tfbs_extract_loci "
                      "with negatives='gc_matched' or 'shuffled'.")

    pos = _torch.load(pos_p, weights_only=True)
    neg = _torch.load(neg_p, weights_only=True)
    if pos.shape[1:] != neg.shape[1:]:
        return _error(
            f"Positives {list(pos.shape)} and negatives {list(neg.shape)} differ in shape.",
            "Extract both with the same in_window.",
        )

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

    out = Path(output_dir).expanduser().resolve() if output_dir else _sibling_dir(pos_p, "classification")
    out.mkdir(parents=True, exist_ok=True)
    _torch.save(seqs, out / "sequences.pt")
    _torch.save(labels, out / "labels.pt")

    return json.dumps({
        "output_dir": str(out),
        "num_positive": n_pos,
        "num_negative": n_neg,
        "total": n_pos + n_neg,
        "files": {"sequences": str(out / "sequences.pt"), "labels": str(out / "labels.pt")},
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
    output_dir: str | None = None,
    score_transform: str = "log2",
    min_score: float | None = None,
    random_seed: int = 42,
) -> str:
    """Prepare a regression dataset from sequences and continuous scores.

    Loads sequences and their associated scores (e.g. ChIP-seq signal
    values from BED column 5), optionally filters by minimum score,
    applies a transformation, and saves.

    Output (sequences.pt + labels.pt) feeds into tfbs_prepare_dataset.

    Defaults and why:
      score_transform "log2": log2(x + 1) compresses the long right tail
        of peak signal values so a few very strong peaks do not dominate
        the MSE loss.  The training-time label scaling (tfbs_config
        scaling_method, "standardize" for regression) is applied on top.
      output_dir: <name>_regression/ next to the sequences' directory,
        with <name> that directory's name minus a _loci suffix
        (peaks_loci/sequences.pt -> peaks_regression/).

    Args:
        sequences_path: Path to sequences.pt (N, 4, L).
        scores_path: Path to scores.pt (N, 1) from tfbs_extract_loci.
        output_dir: Directory to save sequences.pt and labels.pt
                    (default above).
        score_transform: Transform to apply: "log2" (log2(x+1)),
                         "log10" (log10(x+1)), "zscore", or "none".
        min_score: If set, filter out sequences with score below this.
        random_seed: Random seed (default 42).

    Returns:
        JSON with output_dir, num_sequences, score_stats_before,
        score_stats_after and files (sequences, labels).
    """
    _ensure_imports()

    seqs_p = Path(sequences_path).resolve()
    scores_p = Path(scores_path).resolve()
    if not seqs_p.exists():
        return _error(f"Sequences file not found: {sequences_path}",
                      "Pass the sequences_path returned by tfbs_extract_loci.")
    if not scores_p.exists():
        return _error(f"Scores file not found: {scores_path}",
                      "Pass the scores.pt that tfbs_extract_loci writes when the "
                      "BED has a numeric score_column.")

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
        return _error(
            f"Unknown score_transform '{score_transform}'.",
            f"Use one of: {sorted(valid_transforms)}.",
        )

    if score_transform == "log2":
        scores = _torch.log2(scores + 1)
    elif score_transform == "log10":
        scores = _torch.log10(scores + 1)
    elif score_transform == "zscore":
        mean = scores.mean()
        std = scores.std()
        scores = (scores - mean) / (std + 1e-8)

    stats_after = _stats(scores)

    out = Path(output_dir).expanduser().resolve() if output_dir else _sibling_dir(seqs_p, "regression")
    out.mkdir(parents=True, exist_ok=True)
    _torch.save(seqs, out / "sequences.pt")
    _torch.save(scores, out / "labels.pt")

    return json.dumps({
        "output_dir": str(out),
        "num_sequences": int(seqs.shape[0]),
        "score_stats_before": stats_before,
        "score_stats_after": stats_after,
        "files": {"sequences": str(out / "sequences.pt"), "labels": str(out / "labels.pt")},
    })


# ===========================================================================
# Visualization tools (24-26): seqlets, epistasis, conv filters
# ===========================================================================


# ---------------------------------------------------------------------------
# Tool 24: tfbs_seqlets (motif seqlet discovery from attributions)
# ---------------------------------------------------------------------------
# TOMTOM always returns a closest motif, even at p = 1; only matches below
# this p-value count as an annotation.
_SEQLET_MOTIF_MAX_P = 0.05


@mcp.tool(
    name="tfbs_seqlets",
    annotations={
        "title": "Discover Motif Seqlets",
        "readOnlyHint": False,
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
    output_dir: str | None = None,
) -> str:
    """Call seqlets (high-attribution spans) in DNA sequences.

    Computes DeepLiftShap attributions with tangermeme (dinucleotide-
    shuffled references), sums them over the four bases, and calls
    seqlets with tangermeme's recursive seqlet caller.  Optionally
    annotates each seqlet with its best TOMTOM match in a MEME motif file
    (e.g. JASPAR) and counts the matched motifs.

    Requires tangermeme: pip install 'tfbs-mcp[viz]'

    Args:
        model_id: Identifier of a loaded model.
        sequences: DNA sequences to analyze.
        n_shuffles: Number of shuffled baselines for DeepLiftShap.
        min_seqlet_len: Minimum seqlet length (default 5).
        max_seqlet_len: Maximum seqlet length (default 15).
        motif_file: Optional path to a MEME-format motif file for
                    annotation (e.g. JASPAR database).
        output_dir: Directory for seqlets.tsv, which lists every seqlet
                    (the JSON returns at most 200).  Default
                    <results>/<model_id>/tfbs_seqlets/<digest>/, with
                    <results> the results_dir of tfbs_server_info and
                    <digest> a hash of the inputs and settings.

    Returns:
        JSON with num_seqlets (all seqlets called), seqlets (the first
        200, lowest p-value first; each with example_idx, start, end,
        attribution = summed attribution over the span, and p-value),
        truncated, and files (seqlets: the TSV of all seqlets, with
        motif and motif_p_value columns when annotated).  With
        motif_file, each returned seqlet also carries
        motif_p_value (TOMTOM p-value of its closest motif) and motif (that
        motif's name when p < 0.05, else null), and top_motifs lists the 20
        motifs annotated on the most seqlets ({name, count}).  Annotation
        problems are reported in annotation_error without dropping the
        seqlets.
    """
    try:
        from tangermeme.seqlet import recursive_seqlets
        from tangermeme.deep_lift_shap import deep_lift_shap as tangermeme_dls
    except ImportError:
        return _error("tangermeme required.", "Install with: pip install 'tfbs-mcp[viz]'")

    _ensure_imports()
    X, _, err = _load_sequence_input(sequences, None)
    if err:
        return err

    entry, err = _model_entry(model_id)
    if err:
        return err
    model = entry["model"]
    device = entry["device"]
    out = _analysis_dir(
        output_dir, model_id, "tfbs_seqlets",
        sequences, n_shuffles, min_seqlet_len, max_seqlet_len, motif_file,
    )

    X = X.to(device)

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

    # recursive_seqlets returns a DataFrame (example_idx, start, end,
    # attribution, p-value) and raises on degenerate input.
    columns = ["example_idx", "start", "end", "attribution", "p-value"]
    try:
        seqlets = recursive_seqlets(
            attr_sum,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
        )
    except (ZeroDivisionError, ValueError) as exc:
        logger.info("tfbs_seqlets: no seqlets called (%s)", exc)
        seqlets = None
    if seqlets is None or len(seqlets) == 0:
        import pandas as pd
        seqlets = pd.DataFrame(columns=columns)

    max_returned = 200
    seqlet_list = [
        {
            "example_idx": int(row["example_idx"]),
            "start": int(row["start"]),
            "end": int(row["end"]),
            "attribution": float(row["attribution"]),
            "p-value": float(row["p-value"]),
        }
        for row in seqlets.head(max_returned).to_dict("records")
    ]

    result: dict[str, Any] = {
        "num_seqlets": int(len(seqlets)),
        "seqlets": seqlet_list,
        "truncated": len(seqlets) > max_returned,
    }
    matched: list[tuple[str | None, float]] = []

    if motif_file:
        if not Path(motif_file).is_file():
            result["annotation_error"] = f"Motif file not found: {motif_file}"
        elif len(seqlets) == 0:
            result["top_motifs"] = []
        else:
            try:
                from tangermeme.io import read_meme
                from tangermeme.annotate import annotate_seqlets
                motifs = read_meme(motif_file)
                names = list(motifs.keys())
                idxs, pvals = annotate_seqlets(
                    X.cpu(), seqlets[["example_idx", "start", "end"]].astype(int), motifs,
                )
                matched = []
                for k in range(len(seqlets)):
                    m, pval = int(idxs[k, 0]), float(pvals[k, 0])
                    significant = 0 <= m < len(names) and pval < _SEQLET_MOTIF_MAX_P
                    matched.append((names[m] if significant else None, pval))
                for record, (name, pval) in zip(seqlet_list, matched):
                    record["motif"] = name
                    record["motif_p_value"] = pval
                counts: dict[str, int] = {}
                for name, _ in matched:
                    if name is not None:
                        counts[name] = counts.get(name, 0) + 1
                result["top_motifs"] = [
                    {"name": name, "count": count}
                    for name, count in sorted(counts.items(), key=lambda kv: -kv[1])[:20]
                ]
            except Exception as e:
                matched = []
                result["annotation_error"] = str(e)

    out.mkdir(parents=True, exist_ok=True)
    table = seqlets[columns].copy()
    if matched:
        table["motif"] = [name or "" for name, _ in matched]
        table["motif_p_value"] = [pval for _, pval in matched]
    tsv = out / "seqlets.tsv"
    table.to_csv(tsv, sep="\t", index=False)
    result["files"] = {"seqlets": str(tsv)}

    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 25: tfbs_epistasis (pairwise / 3D mutation interactions)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_epistasis",
    annotations={
        "title": "Epistasis Analysis",
        "readOnlyHint": False,
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

    For every pair of positions (i, j) and every pair of substitutions
    away from the reference bases (3 x 3 per pair), compares the double
    mutant with the sum of the two single-mutant effects:

        delta_delta = (y_ij - y_orig) - (y_i - y_orig) - (y_j - y_orig)
                    = y_ij - y_i - y_j + y_orig

    Zero means the two substitutions act additively; non-zero means they
    interact.  The reference base is never counted as a mutation.

    Args:
        model_id: Identifier of a loaded model.
        sequence: Single DNA sequence to analyze.
        positions: Restrict analysis to these 0-indexed positions.
                   Default: all L positions, i.e. 9 * L * (L - 1) / 2
                   double-mutant predictions.
        output_dir: Directory for the two CSV files that are always
                    written: single_mutations.csv (one row per single
                    mutant: position, ref_base, base, prediction, delta)
                    and epistasis_matrix.csv (one row per double mutant:
                    pos_i, ref_i, base_i, pos_j, ref_j, base_j,
                    prediction, delta_delta).  Default
                    <results>/<model_id>/tfbs_epistasis/<digest>/, with
                    <results> the results_dir of tfbs_server_info and
                    <digest> a hash of the sequence and positions.

    Returns:
        JSON with original_prediction, num_positions, num_pairs_analyzed
        (position pairs), num_double_mutants, top_interactions (double
        mutants with |delta_delta| > 1e-6, sorted by |delta_delta|,
        capped at 500, same fields as epistasis_matrix.csv), output_dir
        and a "files" object holding the paths of single_mutations and
        epistasis_matrix.
    """
    _ensure_imports()
    X, _, err = _load_sequence_input([sequence], None)
    if err:
        return err

    entry, err = _model_entry(model_id)
    if err:
        return err
    model = entry["model"]
    device = entry["device"]

    X = X.to(device)
    L = X.shape[2]

    pos = sorted(set(positions)) if positions is not None else list(range(L))
    out_of_range = [p for p in pos if not 0 <= p < L]
    if out_of_range:
        return json.dumps({
            "error": f"Positions outside the {L} bp sequence: {out_of_range}",
            "fix": f"Pass 0-indexed positions from 0 to {L - 1}.",
        })

    ref = {p: int(X[0, :, p].argmax()) for p in pos}
    batch_size = _ENV["default_batch_size"]

    def _predict(mutants: list[tuple[tuple[int, int], ...]]) -> list[float]:
        preds: list[float] = []
        for k in range(0, len(mutants), batch_size):
            chunk = mutants[k:k + batch_size]
            X_batch = X.repeat(len(chunk), 1, 1)
            for n, substitutions in enumerate(chunk):
                for p, b in substitutions:
                    X_batch[n, :, p] = 0.0
                    X_batch[n, b, p] = 1.0
            with _torch.no_grad():
                y = model(X_batch).reshape(len(chunk), -1)[:, 0]
            preds.extend(float(v) for v in y.cpu())
        return preds

    y_orig = _predict([()])[0]

    singles = [(p, b) for p in pos for b in range(4) if b != ref[p]]
    single_preds = _predict([(s,) for s in singles])
    single_delta = {s: y - y_orig for s, y in zip(singles, single_preds)}

    doubles = [
        ((pi, bi), (pj, bj))
        for i, pi in enumerate(pos)
        for pj in pos[i + 1:]
        for bi in range(4) if bi != ref[pi]
        for bj in range(4) if bj != ref[pj]
    ]
    double_preds = _predict(doubles)

    pair_rows = []
    for ((pi, bi), (pj, bj)), y_double in zip(doubles, double_preds):
        dd = (y_double - y_orig) - single_delta[(pi, bi)] - single_delta[(pj, bj)]
        pair_rows.append({
            "pos_i": pi, "ref_i": _IDX_TO_NUC[ref[pi]], "base_i": _IDX_TO_NUC[bi],
            "pos_j": pj, "ref_j": _IDX_TO_NUC[ref[pj]], "base_j": _IDX_TO_NUC[bj],
            "prediction": y_double,
            "delta_delta": dd,
        })

    interactions = sorted(
        (r for r in pair_rows if abs(r["delta_delta"]) > 1e-6),
        key=lambda r: -abs(r["delta_delta"]),
    )[:500]

    result: dict[str, Any] = {
        "original_prediction": round(y_orig, 6),
        "num_positions": len(pos),
        "num_pairs_analyzed": len(pos) * (len(pos) - 1) // 2,
        "num_double_mutants": len(doubles),
        "top_interactions": [
            {**r, "prediction": round(r["prediction"], 6),
             "delta_delta": round(r["delta_delta"], 6)}
            for r in interactions
        ],
    }

    import csv as _csv
    out = _analysis_dir(output_dir, model_id, "tfbs_epistasis", sequence, pos)
    out.mkdir(parents=True, exist_ok=True)
    result["output_dir"] = str(out)
    singles_csv = out / "single_mutations.csv"
    with open(singles_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["position", "ref_base", "base", "prediction", "delta"])
        for (p, b), y in zip(singles, single_preds):
            w.writerow([p, _IDX_TO_NUC[ref[p]], _IDX_TO_NUC[b], y, single_delta[(p, b)]])
    pairs_csv = out / "epistasis_matrix.csv"
    with open(pairs_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=[
            "pos_i", "ref_i", "base_i", "pos_j", "ref_j", "base_j",
            "prediction", "delta_delta",
        ])
        w.writeheader()
        w.writerows(pair_rows)
    result["files"] = {
        "single_mutations": str(singles_csv),
        "epistasis_matrix": str(pairs_csv),
    }

    return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 26: tfbs_conv_filters (filter PFMs as MEME, top activating subsequences)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_conv_filters",
    annotations={
        "title": "Visualize Conv Filters",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_conv_filters(
    model_id: str,
    sequences: list[str],
    output_dir: str | None = None,
    top_n_filters: int = 50,
    top_k_subsequences: int = 10,
) -> str:
    """Summarise what the convolutional filters learned.

    Runs input sequences through the model, captures activations at
    each Conv1d layer, finds the maximally activating positions of each
    filter, and for the filters of layers that read the one-hot input
    directly builds a position frequency matrix (PFM) from the input
    windows behind the activations of at least 0.8 x the filter's
    maximum (at most 10 per sequence; the window accounts for the
    layer's padding, dilation and stride).  Deeper layers see features,
    not bases, so they get the activation summary only.  Filters whose
    maximum activation is not positive get no PFM.

    Writes to output_dir:
      filter_summary.json: one record per analysed filter (layer,
        filter_id, kernel_size, max_activation, num_activating,
        pfm_sites, in_meme).
      filters.meme: the PFMs in MEME format (one MOTIF <layer>_f<id>
        per filter with a PFM; letter-probability rows, nsites = the
        number of windows averaged), ready for TOMTOM against JASPAR.
      top_subsequences.json: per filter, the top_k_subsequences input
        windows with the highest activation (sequence_index, start,
        activation, subsequence).

    Args:
        model_id: Identifier of a loaded model.
        sequences: DNA sequences to use for activation analysis.
        output_dir: Directory for the three files.  Default
                    <results>/<model_id>/tfbs_conv_filters/<digest>/,
                    with <results> the results_dir of tfbs_server_info
                    and <digest> a hash of the sequences and settings.
        top_n_filters: Max filters to analyze per layer (default 50).
        top_k_subsequences: Subsequences kept per filter (default 10).

    Returns:
        JSON with output_dir, files (filter_summary, meme,
        top_subsequences), num_layers, num_filters_analyzed,
        num_pfms (MOTIFs in the MEME file) and filter_summary (first
        100 records).
    """
    _ensure_imports()
    X, _, err = _load_sequence_input(sequences, None)
    if err:
        return err

    entry, err = _model_entry(model_id)
    if err:
        return err
    model = entry["model"]
    device = entry["device"]

    X = X.to(device)

    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, _torch.nn.Conv1d):
            conv_layers.append((name, module))

    if not conv_layers:
        return _error(
            "No Conv1d layers found in model",
            "Use a convolutional model (e.g. VCNNBpnet, BPNet, CNN); for "
            "other architectures use tfbs_analyze or tfbs_seqlets.",
            model_type=entry["model_type"],
        )

    out = _analysis_dir(output_dir, model_id, "tfbs_conv_filters",
                        sequences, top_n_filters, top_k_subsequences)
    out.mkdir(parents=True, exist_ok=True)

    activations: dict[str, Any] = {}
    reads_input: dict[str, bool] = {}
    hooks = []
    for layer_name, module in conv_layers:
        def _hook(mod, inp, output, name=layer_name):
            activations[name] = output.detach().cpu()
            x_in = inp[0] if inp else None
            reads_input[name] = x_in is X or (
                isinstance(x_in, _torch.Tensor) and x_in.shape == X.shape and _torch.equal(x_in, X)
            )
        hooks.append(module.register_forward_hook(_hook))

    try:
        with _torch.no_grad():
            model(X)
    finally:
        for h in hooks:
            h.remove()

    X_cpu = X.cpu()
    L = X_cpu.shape[2]
    filter_summary = []
    meme_blocks = []
    top_subsequences = []
    for layer_name, module in conv_layers:
        if layer_name not in activations:
            continue
        act = activations[layer_name]  # (N, C_out, L_out)
        n_filters = min(act.shape[1], top_n_filters)
        kernel_size = module.kernel_size[0]
        dilation = module.dilation[0]
        stride = module.stride[0]
        if isinstance(module.padding, str):
            pad = 0 if module.padding == "valid" else dilation * (kernel_size - 1) // 2
        else:
            pad = module.padding[0]
        span = dilation * (kernel_size - 1) + 1
        on_input = reads_input.get(layer_name, False)

        def _window(seq_idx: int, out_pos: int):
            s = out_pos * stride - pad
            if s < 0 or s + span > L:
                return None
            return X_cpu[seq_idx, :, s:s + span:dilation], s

        for f_idx in range(n_filters):
            f_act = act[:, f_idx, :]  # (N, L_out)
            max_act = float(f_act.max())
            threshold = max_act * 0.8
            high_mask = f_act >= threshold
            n_activating = int(high_mask.sum())

            pfm_sites = 0
            in_meme = False
            if on_input and max_act > 0:
                pfm = _np.zeros((4, kernel_size))
                for seq_idx in range(f_act.shape[0]):
                    for p in _torch.where(high_mask[seq_idx])[0][:10].tolist():
                        w = _window(seq_idx, p)
                        if w is not None:
                            pfm += w[0].numpy()
                            pfm_sites += 1
                if pfm_sites:
                    pfm /= pfm_sites
                    motif = f"{_safe_name(layer_name)}_f{f_idx}"
                    rows = "\n".join(
                        " " + " ".join(f"{v:.6f}" for v in pfm[:, j]) for j in range(kernel_size)
                    )
                    meme_blocks.append(
                        f"MOTIF {motif}\n"
                        f"letter-probability matrix: alength= 4 w= {kernel_size} "
                        f"nsites= {pfm_sites} E= 0\n{rows}\n"
                    )
                    in_meme = True

                flat = f_act.flatten()
                order = _torch.argsort(flat, descending=True).tolist()
                top = []
                for k in order:
                    if len(top) >= top_k_subsequences:
                        break
                    seq_idx, p = divmod(k, f_act.shape[1])
                    w = _window(seq_idx, p)
                    if w is None:
                        continue
                    top.append({
                        "sequence_index": seq_idx,
                        "start": w[1],
                        "activation": round(float(flat[k]), 4),
                        "subsequence": _tensor_to_sequences(w[0].unsqueeze(0))[0],
                    })
                top_subsequences.append({"layer": layer_name, "filter_id": f_idx, "top": top})

            filter_summary.append({
                "layer": layer_name,
                "filter_id": f_idx,
                "max_activation": round(max_act, 4),
                "num_activating": n_activating,
                "kernel_size": kernel_size,
                "pfm_sites": pfm_sites,
                "in_meme": in_meme,
            })

    files = {
        "filter_summary": str(out / "filter_summary.json"),
        "meme": str(out / "filters.meme"),
        "top_subsequences": str(out / "top_subsequences.json"),
    }
    with open(files["filter_summary"], "w") as f:
        json.dump(filter_summary, f, indent=2)
    with open(files["meme"], "w", newline="\n") as f:
        f.write(
            "MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
            "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n"
        )
        f.write("".join(block + "\n" for block in meme_blocks))
    with open(files["top_subsequences"], "w") as f:
        json.dump(top_subsequences, f, indent=2)

    return json.dumps({
        "output_dir": str(out),
        "files": files,
        "num_layers": len(conv_layers),
        "num_filters_analyzed": len(filter_summary),
        "num_pfms": len(meme_blocks),
        "filter_summary": filter_summary[:100],
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
    genome_fasta: str = "",
    max_sequences: int = 15000,
    device: str | None = None,
    slurm: bool = False,
    partition: str | None = None,
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
        models_dir: Base checkpoint directory; a relative path resolves
            against the project root (TFBS_PROJECT_ROOT, see
            tfbs_server_info), where the pipeline runs.
        genome_fasta: Reference genome FASTA path. Falls back to
            $TFBS_GENOME_FASTA.
        max_sequences: Max sequences per TF for speed.
        device: "cuda" or "cpu" for NN inference.  If omitted: "cuda" for
            a SLURM job (the job requests one GPU), otherwise the server's
            detected default device (see tfbs_server_info).
        slurm: Submit as SLURM job instead of running directly.
        partition: SLURM partition.  Defaults to the server's
            default_partition: $TFBS_SLURM_PARTITION, else "gpu"
            (see tfbs_server_info), because the job requests a GPU and
            "gpu" is the usual name of a cluster's GPU partition.
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

    genome_path, err = _resolve_reference(genome_fasta, "genome_fasta")
    if err:
        return err

    err = _pipeline_requirements_error("chipseq_benchmark", "tfbs_chipseq_benchmark")
    if err:
        return err
    project_root, err = _work_dir()
    if err:
        return err

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(device, slurm_gpu=True if slurm else None)

    cmd_parts = [
        *_pipeline_argv("chipseq_benchmark"),
        "--intersect-dir", str(intersect_path),
        "--output-dir", str(out),
        "--pos-tf", pos_tf,
        "--neg-tfs", *neg_tfs,
        "--datasets", *datasets,
        "--cutoffs", *[str(c) for c in cutoffs],
        "--model-type", model_type,
        "--models-dir", models_dir,
        "--genome", str(genome_path),
        "--max-sequences", str(max_sequences),
        "--device", device,
    ]
    cmd = shlex.join(cmd_parts)

    if slurm:
        partition = _resolve_partition(partition)
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
            f"\ndate\n{_sbatch_pythonpath_line()}{cmd}\ndate\n"
        )
        sbatch_path = out / "chipseq_benchmark.sh"
        sbatch_path.write_text(sbatch_content, newline="\n")
        os.chmod(str(sbatch_path), 0o755)

        submit_result = json.loads(await slurm_submit(
            script_path=str(sbatch_path),
            working_dir=str(project_root),
        ))
        submit_result["mode"] = "slurm"
        submit_result["command"] = cmd
        return json.dumps(submit_result)

    result, err = _run_pipeline(cmd_parts, project_root, 3600, "Benchmark")
    if err:
        return err

    if result.returncode != 0:
        return json.dumps({
            "error": f"Benchmark failed (exit {result.returncode})",
            "fix": "Read stderr for the cause; a checkpoint missing under "
                   "models_dir or a genome whose chromosome names differ from "
                   "the BED files are the usual ones.",
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
# Tool 31: tfbs_chipseq_method_comparison (NN + FIMO + FoldX, head-to-head)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_chipseq_method_comparison",
    annotations={
        "title": "ChIP-seq Method Comparison (NN + FIMO + FoldX)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tfbs_chipseq_method_comparison(
    intersect_dir: str,
    output_dir: str,
    motif_file: str = "",
    foldx_pdb: str = "",
    pos_tf: str = "NKX2-1",
    neg_tfs: list[str] | None = None,
    datasets: list[str] | None = None,
    model_type: str = "VCNNBpnet",
    models_dir: str = "saved_models_final",
    genome_fasta: str = "",
    n_per_tf: int = 500,
    foldx_workers: int = 32,
    device: str | None = None,
    slurm: bool = True,
    partition: str | None = None,
    max_time: str = "12:00:00",
) -> str:
    """Compare NN, FIMO (JASPAR PWM), and FoldX (TF-DNA crystal) for
    distinguishing positive TF peaks from negative TF peaks.

    This is the same ChIP-Atlas TF-vs-TF classification task as
    tfbs_chipseq_benchmark, but adds head-to-head ROC-AUC comparison
    against the JASPAR motif (FIMO) and structure-based binding energy
    (FoldX) on the same balanced subsample of sequences.

    Recommended for SLURM (FoldX is ~3 min per sequence; default 500
    sequences per TF takes ~4 hours on 32 cores).

    Args:
        intersect_dir: Directory with *.allintersect.bed files.
        output_dir: Where to save plots, raw scores, and AUC tables.
        motif_file: JASPAR/MEME motif file for FIMO (e.g. MA1994.1.meme
            for NKX2-1). Falls back to $TFBS_MOTIF_FILE.
        foldx_pdb: Repaired TF-DNA complex PDB, or $TFBS_FOLDX_PDB. The
            DNA length in the
            PDB defines the FoldX scoring window (must be <= the
            sequence window, typically 11 bp for NKX2-1).
        pos_tf: Positive transcription factor name.
        neg_tfs: Negative TF names.
        datasets: NN dataset names (subdirectories of models_dir).
        model_type: Model architecture class name.
        models_dir: Base checkpoint directory; a relative path resolves
            against the project root (TFBS_PROJECT_ROOT, see
            tfbs_server_info), where the pipeline runs.
        genome_fasta: Reference genome FASTA. Falls back to
            $TFBS_GENOME_FASTA.
        n_per_tf: Sequences sampled per TF (FoldX bottleneck).
        foldx_workers: Parallel CPU workers for FoldX.
        device: "cuda" or "cpu" for NN inference.  If omitted: "cuda" for
            a SLURM job (the job requests one GPU), otherwise the server's
            detected default device (see tfbs_server_info).
        slurm: Submit as SLURM job (recommended).
        partition: SLURM partition.  Defaults to the server's
            default_partition: $TFBS_SLURM_PARTITION, else "gpu"
            (see tfbs_server_info), because the job requests a GPU and
            "gpu" is the usual name of a cluster's GPU partition.
        max_time: SLURM wall time.

    Returns:
        JSON with output_dir, file list, and (if slurm) job_id.
    """
    if neg_tfs is None:
        neg_tfs = ["GATA1", "MYOD1", "NKX2-5", "RXRA"]
    if datasets is None:
        datasets = ["all_mean", "core_mean", "flank_mean"]

    genome_path, err = _resolve_reference(genome_fasta, "genome_fasta")
    if err:
        return err
    motif_path, err = _resolve_reference(motif_file, "motif_file")
    if err:
        return err
    foldx_path, err = _resolve_reference(foldx_pdb, "foldx_pdb")
    if err:
        return err

    err = _pipeline_requirements_error(
        "chipseq_benchmark_with_foldx", "tfbs_chipseq_method_comparison")
    if err:
        return err
    project_root, err = _work_dir()
    if err:
        return err

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(device, slurm_gpu=True if slurm else None)

    cmd_parts = [
        *_pipeline_argv("chipseq_benchmark_with_foldx"),
        "--intersect-dir", str(Path(intersect_dir).resolve()),
        "--output-dir", str(out),
        "--genome", str(genome_path),
        "--motif-file", str(motif_path),
        "--foldx-pdb", str(foldx_path),
        "--pos-tf", pos_tf,
        "--neg-tfs", *neg_tfs,
        "--datasets", *datasets,
        "--models-dir", models_dir,
        "--model-type", model_type,
        "--n-per-tf", str(n_per_tf),
        "--foldx-workers", str(foldx_workers),
        "--device", device,
    ]
    cmd = shlex.join(cmd_parts)

    if slurm:
        partition = _resolve_partition(partition)
        sbatch_content = (
            f"#!/bin/bash\n"
            f"#SBATCH --job-name=cs_method_cmp\n"
            f"#SBATCH --partition={partition}\n"
            f"#SBATCH --nodes=1\n"
            f"#SBATCH --cpus-per-task={max(foldx_workers, 8)}\n"
            f"#SBATCH --mem=64G\n"
            f"#SBATCH --gres=gpu:1\n"
            f"#SBATCH --time={max_time}\n"
            f"#SBATCH --output={out}/method_cmp.o%j\n"
            f"#SBATCH --error={out}/method_cmp.e%j\n"
            f"\ndate\n{_sbatch_pythonpath_line()}{cmd}\ndate\n"
        )
        sbatch_path = out / "method_comparison.sh"
        sbatch_path.write_text(sbatch_content, newline="\n")
        os.chmod(str(sbatch_path), 0o755)

        submit_result = json.loads(await slurm_submit(
            script_path=str(sbatch_path),
            working_dir=str(project_root),
        ))
        submit_result["mode"] = "slurm"
        submit_result["command"] = cmd
        submit_result["output_dir"] = str(out)
        return json.dumps(submit_result)

    result, err = _run_pipeline(cmd_parts, project_root, 43200, "Method comparison")
    if err:
        return err
    if result.returncode != 0:
        return json.dumps({
            "error": f"Method comparison failed (exit {result.returncode})",
            "fix": "Read stderr for the cause; a checkpoint missing under "
                   "models_dir, fimo or FoldX not on PATH, or a genome whose "
                   "chromosome names differ from the BED files are the usual ones.",
            "stderr": result.stderr[-1500:],
        })
    files = [str(f.relative_to(out)) for f in out.rglob("*") if f.is_file()]
    return json.dumps({
        "mode": "direct", "output_dir": str(out),
        "files": files, "stdout": result.stdout[-1000:],
    })


# ---------------------------------------------------------------------------
# Tool 32: tfbs_remap_validation (ReMap external validation, NN+FIMO+FoldX)
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_remap_validation",
    annotations={
        "title": "ReMap External Validation (NN + FIMO + FoldX)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tfbs_remap_validation(
    remap_bed: str,
    output_dir: str,
    motif_file: str = "",
    foldx_pdb: str = "",
    score_min: int = 4,
    max_positives: int = 500,
    negative_mode: str = "shuffle",
    datasets: list[str] | None = None,
    model_type: str = "VCNNBpnet",
    models_dir: str = "saved_models_final",
    genome_fasta: str = "",
    foldx_workers: int = 32,
    foldx_window: int = 11,
    nn_window: int = 24,
    max_region_len: int = 500,
    skip_foldx: bool = False,
    device: str | None = None,
    seed: int = 42,
    slurm: bool = True,
    partition: str | None = None,
    max_time: str = "12:00:00",
) -> str:
    """External validation of a TFBS model against a ReMap peak set.

    Positives are ReMap peaks at experiment-count >= score_min (column 5
    of the ReMap BED). Negatives are either dinucleotide-shuffled
    positives (cleanest motif-disrupting control) or length+GC matched
    random genomic regions that do not overlap any ReMap peak +/- 1 kb.
    All three method families are scored on the same regions: the NN
    takes the max over nn_window-bp tiles, FIMO scans the full region
    with the supplied JASPAR motif, and FoldX scores the foldx_window-bp
    centre of the region against the supplied TF-DNA crystal.

    This is a complement to tfbs_chipseq_method_comparison, which runs
    positive-TF-peaks vs negative-TF-peaks classification. ReMap
    validation uses shuffle or matched-genomic negatives instead, which
    is the cleanest external test of whether the model has learned the
    motif rather than the cell-type background.

    Args:
        remap_bed: ReMap 2022 non-redundant BED for the target TF.
            Column 5 must encode the experiment count.
        output_dir: Where the script writes summary.csv, the two scores
            CSVs, and roc_all_methods.png.
        motif_file: JASPAR / MEME motif file for FIMO. Falls back to
            $TFBS_MOTIF_FILE.
        foldx_pdb: Repaired TF-DNA complex PDB for FoldX scoring.
            Falls back to $TFBS_FOLDX_PDB. Not required when
            skip_foldx is true.
        score_min: Minimum experiment count in the ReMap BED.
        max_positives: Subsample the high-confidence set to this many
            peaks (FoldX is the bottleneck; default 500).
        negative_mode: "shuffle" for dinucleotide-shuffled positives or
            "genomic" for length+GC matched random regions.
        datasets: NN dataset names (subdirectories of models_dir).
        model_type: Model architecture class name.
        models_dir: Base checkpoint directory; a relative path resolves
            against the project root (TFBS_PROJECT_ROOT, see
            tfbs_server_info), where the pipeline runs.
        genome_fasta: Reference genome FASTA. Falls back to
            $TFBS_GENOME_FASTA.
        foldx_workers: Parallel CPU workers for FoldX.
        foldx_window: DNA window for FoldX (must match the PDB).
        nn_window: Tile size for the NN score.
        max_region_len: Cap on the per-peak region length for tiling.
        skip_foldx: Skip the FoldX leg (useful for quick NN+FIMO runs).
        device: "cuda" or "cpu" for NN inference.  If omitted: "cuda" for
            a SLURM job (the job requests one GPU), otherwise the server's
            detected default device (see tfbs_server_info).
        seed: Random seed for negative sampling and shuffle.
        slurm: Submit as a SLURM job (recommended when FoldX is on).
        partition: SLURM partition.  Defaults to the server's
            default_partition: $TFBS_SLURM_PARTITION, else "gpu"
            (see tfbs_server_info), because the job requests a GPU and
            "gpu" is the usual name of a cluster's GPU partition.
        max_time: SLURM wall time.

    Returns:
        JSON with output_dir, file list, and either job_id (slurm=true)
        or the parsed AUC summary (slurm=false, summary.csv read back
        as a list of {method, auc, n_pos, n_neg} entries).
    """
    if datasets is None:
        datasets = ["all_mean", "core_mean", "flank_mean"]

    genome_path, err = _resolve_reference(genome_fasta, "genome_fasta")
    if err:
        return err
    motif_path, err = _resolve_reference(motif_file, "motif_file")
    if err:
        return err
    # FoldX is the only leg that needs the crystal structure, so skip_foldx
    # runs are not asked for a PDB they will never open.
    foldx_path = None
    if not skip_foldx:
        foldx_path, err = _resolve_reference(foldx_pdb, "foldx_pdb")
        if err:
            return err

    err = _pipeline_requirements_error("remap_validation", "tfbs_remap_validation")
    if err:
        return err
    project_root, err = _work_dir()
    if err:
        return err

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(device, slurm_gpu=True if slurm else None)

    cmd_parts = [
        *_pipeline_argv("remap_validation"),
        "--remap-bed", str(Path(remap_bed).resolve()),
        "--output-dir", str(out),
        "--genome", str(genome_path),
        "--motif-file", str(motif_path),
        "--score-min", str(score_min),
        "--max-positives", str(max_positives),
        "--negative-mode", negative_mode,
        "--datasets", *datasets,
        "--models-dir", models_dir,
        "--model-type", model_type,
        "--device", device,
        "--seed", str(seed),
        "--foldx-workers", str(foldx_workers),
        "--foldx-window", str(foldx_window),
        "--nn-window", str(nn_window),
        "--max-region-len", str(max_region_len),
    ]
    if skip_foldx:
        cmd_parts.append("--skip-foldx")
    else:
        cmd_parts.extend(["--foldx-pdb", str(foldx_path)])
    cmd = shlex.join(cmd_parts)

    if slurm:
        partition = _resolve_partition(partition)
        sbatch_content = (
            f"#!/bin/bash\n"
            f"#SBATCH --job-name=remap_validation\n"
            f"#SBATCH --partition={partition}\n"
            f"#SBATCH --nodes=1\n"
            f"#SBATCH --cpus-per-task={max(foldx_workers, 8)}\n"
            f"#SBATCH --mem=64G\n"
            f"#SBATCH --gres=gpu:1\n"
            f"#SBATCH --time={max_time}\n"
            f"#SBATCH --output={out}/remap_validation.o%j\n"
            f"#SBATCH --error={out}/remap_validation.e%j\n"
            f"\ndate\n{_sbatch_pythonpath_line()}{cmd}\ndate\n"
        )
        sbatch_path = out / "remap_validation.sh"
        sbatch_path.write_text(sbatch_content, newline="\n")
        os.chmod(str(sbatch_path), 0o755)

        submit_result = json.loads(await slurm_submit(
            script_path=str(sbatch_path),
            working_dir=str(project_root),
        ))
        submit_result["mode"] = "slurm"
        submit_result["command"] = cmd
        submit_result["output_dir"] = str(out)
        submit_result["expected_summary"] = str(out / "summary.csv")
        return json.dumps(submit_result)

    result, err = _run_pipeline(cmd_parts, project_root, 43200, "ReMap validation")
    if err:
        return err
    if result.returncode != 0:
        return json.dumps({
            "error": f"ReMap validation failed (exit {result.returncode})",
            "fix": "Read stderr for the cause; a checkpoint missing under "
                   "models_dir, fimo or FoldX not on PATH, or a genome whose "
                   "chromosome names differ from the BED file are the usual ones.",
            "stderr": result.stderr[-1500:],
        })

    # Parse summary.csv if the script produced it, so direct-mode callers
    # see the AUCs in the JSON without a follow-up file read.
    summary = None
    summary_csv = out / "summary.csv"
    if summary_csv.exists():
        try:
            import csv
            with open(summary_csv) as f:
                summary = [
                    {
                        "method": row["method"],
                        "auc": float(row["auc"]),
                        "n_pos": int(row["n_pos"]),
                        "n_neg": int(row["n_neg"]),
                    }
                    for row in csv.DictReader(f)
                ]
        except Exception as exc:
            summary = {"parse_error": str(exc)}

    files = [str(f.relative_to(out)) for f in out.rglob("*") if f.is_file()]
    return json.dumps({
        "mode": "direct",
        "output_dir": str(out),
        "files": files,
        "summary": summary,
        "stdout": result.stdout[-1000:],
    })


# ---------------------------------------------------------------------------
# Tool 33: tfbs_disease_variant_validation
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_disease_variant_validation",
    annotations={
        "title": "Disease-Variant Validation",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tfbs_disease_variant_validation(
    peaks_bed: str,
    output_dir: str,
    clinvar_vcf: str = "",
    genome_fasta: str = "",
    peak_confidence: int = 1,
    nn_models: dict | None = None,
    motif_file: str | None = None,
    foldx_pdb: str | None = None,
    nn_window: int = 24,
    foldx_workers: int = 16,
    device: str | None = None,
    slurm: bool = False,
    partition: str = "compute",
    max_time: str = "06:00:00",
) -> str:
    """Validate TF-binding models against ClinVar disease variants.

    Pipeline (works for any transcription factor):
    1. Merge a BED of TF binding peaks (optional confidence filter).
    2. Pull non-coding pathogenic and benign SNVs from ClinVar that
       overlap those peaks.
    3. Score reference and alternative alleles per variant with one or
       more NN models, with FIMO against a JASPAR PWM, and optionally
       with FoldX on a TF-DNA crystal structure.
    4. Test whether |delta-delta| discriminates pathogenic from benign
       variants (ROC-AUC, Mann-Whitney U).

    Args:
        peaks_bed: Path to a BED file of TF binding peaks. Accepts the
            ChIP-Atlas allintersect format (4th column = experiment
            overlap count) or any BED with 3+ columns.
        output_dir: Directory to write results, plots, and summary CSV.
        clinvar_vcf: Indexed ClinVar VCF (.vcf.gz with .tbi). Falls
            back to $TFBS_CLINVAR_VCF.
        genome_fasta: Reference genome FASTA file. Falls back to
            $TFBS_GENOME_FASTA.
        peak_confidence: Minimum value of the BED 4th column to keep
            (e.g. ChIP-Atlas overlap >= 3). Default 1.
        nn_models: Dict mapping model name to checkpoint path, e.g.
            {"flank_mean": "/path/to/best_VCNNBpnet.ckpt"}.
        motif_file: JASPAR / MEME motif file for FIMO scoring.
        foldx_pdb: Repaired TF-DNA complex PDB for FoldX scoring.
            FoldX is computationally expensive; consider slurm=true.
        nn_window: Window size around the variant for NN scoring
            (default 24, matches model input).
        foldx_workers: Parallel CPU workers for FoldX.
        device: "cuda" or "cpu" for NN inference.  If omitted: "cpu" for a
            SLURM job (the job requests no GPU), otherwise the server's
            detected default device (see tfbs_server_info).
        slurm: Submit as a SLURM job rather than running directly.
            Strongly recommended when FoldX is enabled.
        partition: SLURM partition (default "compute").  This job requests
            no GPU, so unlike the GPU tools it does not default to the
            server's default_partition; pass the name of a CPU partition
            if the cluster has no partition called "compute".
        max_time: SLURM wall time (max 48:00:00).

    Returns:
        JSON with output_dir, file list, and (if slurm=true) job_id.
    """
    clinvar_path, err = _resolve_reference(clinvar_vcf, "clinvar_vcf")
    if err:
        return err
    genome_path, err = _resolve_reference(genome_fasta, "genome_fasta")
    if err:
        return err

    err = _pipeline_requirements_error(
        "disease_variant_validation", "tfbs_disease_variant_validation")
    if err:
        return err
    project_root, err = _work_dir()
    if err:
        return err

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(device, slurm_gpu=False if slurm else None)

    cmd_parts = [
        *_pipeline_argv("disease_variant_validation"),
        "--peaks-bed", str(Path(peaks_bed).resolve()),
        "--clinvar-vcf", str(clinvar_path),
        "--genome", str(genome_path),
        "--output-dir", str(out),
        "--peak-confidence", str(peak_confidence),
        "--nn-window", str(nn_window),
        "--foldx-workers", str(foldx_workers),
        "--device", device,
    ]
    if nn_models:
        model_args = [f"{name}={ckpt}" for name, ckpt in nn_models.items()]
        cmd_parts.append("--models")
        cmd_parts.extend(model_args)
    if motif_file:
        cmd_parts.extend(["--motif-file", str(Path(motif_file).resolve())])
    if foldx_pdb:
        cmd_parts.extend(["--foldx-pdb", str(Path(foldx_pdb).resolve())])

    cmd = shlex.join(cmd_parts)

    if slurm:
        sbatch_content = (
            f"#!/bin/bash\n"
            f"#SBATCH --job-name=disease_validation\n"
            f"#SBATCH --partition={partition}\n"
            f"#SBATCH --nodes=1\n"
            f"#SBATCH --cpus-per-task={max(foldx_workers, 8)}\n"
            f"#SBATCH --mem=64G\n"
            f"#SBATCH --time={max_time}\n"
            f"#SBATCH --output={out}/disease_validation.o%j\n"
            f"#SBATCH --error={out}/disease_validation.e%j\n"
            f"\ndate\n{_sbatch_pythonpath_line()}{cmd}\ndate\n"
        )
        sbatch_path = out / "disease_validation.sh"
        sbatch_path.write_text(sbatch_content, newline="\n")
        os.chmod(str(sbatch_path), 0o755)

        submit_result = json.loads(await slurm_submit(
            script_path=str(sbatch_path),
            working_dir=str(project_root),
        ))
        submit_result["mode"] = "slurm"
        submit_result["command"] = cmd
        submit_result["output_dir"] = str(out)
        return json.dumps(submit_result)

    result, err = _run_pipeline(cmd_parts, project_root, 14400, "Disease validation")
    if err:
        return err

    if result.returncode != 0:
        return json.dumps({
            "error": f"Disease validation failed (exit {result.returncode})",
            "fix": "Read stderr for the cause; a ClinVar VCF without its .tbi "
                   "index, an nn_models checkpoint that does not exist, or "
                   "fimo or FoldX not on PATH are the usual ones.",
            "stderr": result.stderr[-1500:],
            "stdout": result.stdout[-500:],
        })

    files = [str(f.relative_to(out)) for f in out.rglob("*") if f.is_file()]
    return json.dumps({
        "mode": "direct",
        "output_dir": str(out),
        "files": files,
        "stdout": result.stdout[-1000:],
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
