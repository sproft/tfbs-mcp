"""
MCP server exposing TFBS-NN model loading, prediction, preprocessing,
attribution analysis, and FIMO motif scanning as tools.

Run with:  tfbs-mcp          (stdio transport, for Claude Desktop / VS Code)
           python -m tfbs.mcp.server   (same)
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import hashlib
import importlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from tfbs.bedio import BedReadError, read_bed
from tfbs.mcp.cli import SETUP_VARIABLES

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
                capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=5,
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
# A result whose JSON is larger than this is logged as its size and first
# characters only, so one huge answer cannot swell the log.
CALL_LOG_MAX_RESULT_BYTES = 1024 * 1024
CALL_LOG_HEAD_CHARS = 4096


def _call_result_text(result: Any) -> str | None:
    """The text a tool call returned, from any form the tool manager hands back.

    Without conversion a tool returns its own value (a JSON string here);
    with it, a list of content blocks, or a (content, structured) tuple when
    the tool has an output schema.
    """
    if isinstance(result, tuple) and result:
        result = result[0]
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)) and result:
        texts = [getattr(block, "text", None) for block in result]
        if all(isinstance(t, str) for t in texts):
            return "\n".join(texts)
    return None


def _call_log_result(result: Any) -> tuple[Any, bool]:
    """(logged result, is_error) for one tool call's return value.

    The logged result is the parsed JSON, {"text": ...} when the return value
    is not JSON, or {"truncated": true, "bytes": N, "head": ...} when the
    returned text exceeds CALL_LOG_MAX_RESULT_BYTES (UTF-8).  The size is
    checked on the text before it is parsed, so a large answer is never
    parsed or serialised again just to be cut; its is_error is read from
    the head (an error answer starts with {"error": ...).
    """
    text = _call_result_text(result)
    if text is None:
        text = str(result)
    if len(text) > CALL_LOG_MAX_RESULT_BYTES or (
            4 * len(text) > CALL_LOG_MAX_RESULT_BYTES
            and len(text.encode("utf-8")) > CALL_LOG_MAX_RESULT_BYTES):
        head = text[:CALL_LOG_HEAD_CHARS]
        is_error = re.match(r'\s*\{\s*"error"\s*:', head) is not None
        return {"truncated": True, "bytes": len(text.encode("utf-8")), "head": head}, is_error
    try:
        parsed: Any = json.loads(text)
    except ValueError:
        parsed = {"text": text}
    is_error = isinstance(parsed, dict) and "error" in parsed
    return parsed, is_error


def _append_call_log(tool: str, arguments: dict[str, Any], started: datetime,
                     seconds: float, result: Any, is_error: bool) -> None:
    """Append one JSON line per finished call to $TFBS_CALL_LOG.

    The line holds timestamp (start of the call, UTC), tool, arguments as
    received, seconds (wall time), is_error and result.  A failure to write
    is logged as a warning and never reaches the caller.
    """
    target = os.environ.get(CALL_LOG_ENV, "").strip()
    if not target:
        return
    try:
        record = {
            "timestamp": started.isoformat(timespec="milliseconds"),
            "tool": tool,
            "arguments": arguments,
            "seconds": round(seconds, 3),
            "is_error": bool(is_error),
            "result": result,
        }
        line = json.dumps(record, default=str, ensure_ascii=False) + "\n"
        path = Path(target).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line)
    except Exception as exc:  # logging must never break a tool call
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
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        with _stdout_to_stderr():
            result = await _tool_manager_call_tool(
                name, arguments, context=context, convert_result=convert_result,
            )
    except BaseException as exc:
        # Logged, then raised unchanged: the MCP layer reports it as before.
        try:
            message = f"{type(exc).__name__}: {exc}"
        except Exception:  # an exception whose __str__ itself raises
            message = type(exc).__name__
        _append_call_log(name, arguments, started, time.perf_counter() - t0,
                         {"error": message}, True)
        raise
    seconds = time.perf_counter() - t0
    if os.environ.get(CALL_LOG_ENV, "").strip():
        try:
            logged, is_error = _call_log_result(result)
        except Exception as exc:  # logging must never break a tool call
            logged, is_error = {"error": f"result not loggable: {exc}"}, False
        _append_call_log(name, arguments, started, seconds, logged, is_error)
    return result


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


def _pipeline_popen_kwargs(cwd: Path) -> dict[str, Any]:
    """Popen arguments of a pipeline run: no stdin, piped UTF-8 output, own process group."""
    return {
        "stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
        "text": True, "encoding": "utf-8", "errors": "replace", "cwd": str(cwd),
        "env": {**_pipeline_env(), "PYTHONIOENCODING": "utf-8"}, **_new_process_group(),
    }


def _spawn_pipeline(cmd: list[str], cwd: Path):
    """Start a pipeline process (tests replace this with a fake process)."""
    return subprocess.Popen(cmd, **_pipeline_popen_kwargs(cwd))


class _PipelineRun:
    """The process of one direct-mode pipeline run, shared with its worker thread."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc = None
        self.cancelled = False

    def start(self, cmd: list[str], cwd: Path):
        with self.lock:
            if self.cancelled:
                raise subprocess.SubprocessError("cancelled before it started")
            self.proc = _spawn_pipeline(cmd, cwd)
            return self.proc

    def cancel(self):
        """Mark the run cancelled; the process to stop, if it has started."""
        with self.lock:
            self.cancelled = True
            return self.proc


def _pipeline_blocking(cmd: list[str], cwd: Path, timeout: int,
                       run: _PipelineRun) -> subprocess.CompletedProcess:
    """Run the pipeline to the end (in a worker thread); kills its group on timeout."""
    proc = run.start(cmd, cwd)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        try:
            proc.communicate(timeout=30)
        except Exception:
            pass
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout or "", stderr or "")


async def _run_pipeline(cmd: list[str], cwd: Path, timeout: int, label: str):
    """Run a pipeline in a subprocess: (CompletedProcess, None) or (None, error JSON).

    The process runs in a worker thread, so the server keeps answering
    other calls meanwhile; when the MCP call is cancelled, the pipeline's
    whole process group (process tree on Windows) is killed.
    """
    run = _PipelineRun()
    try:
        result = await asyncio.to_thread(_pipeline_blocking, cmd, cwd, timeout, run)
    except asyncio.CancelledError:
        proc = run.cancel()
        if proc is not None:
            await asyncio.shield(asyncio.to_thread(_stop_process_group, proc))
        raise
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
            "fix": f"Pass {arg}=/path/to/file, or register it as {env_var}: run "
                   f"`{_setup_command(env_var)}` (with the flags of the files "
                   f"already registered, since each run replaces the whole "
                   f"registration), apply the Claude Code command or Claude "
                   f"Desktop block it prints, and restart the client.",
        })
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        return None, json.dumps({
            "error": f"{what} not found: {path}",
            "fix": f"Check the {arg} argument or the {env_var} environment variable.",
        })
    return path, None


def _setup_command(env_var: str) -> str:
    """The ``tfbs-mcp --setup`` call that registers ``env_var``, for fix texts."""
    flag = next(v.flag for v in SETUP_VARIABLES if v.env == env_var)
    return f"tfbs-mcp --setup --non-interactive {flag} /path/to/file"


def _reference_files() -> dict[str, dict]:
    """The registered reference files, as tfbs_server_info reports them.

    One entry per reference variable (TFBS_GENOME_FASTA, TFBS_MOTIF_FILE,
    TFBS_CLINVAR_VCF, TFBS_FOLDX_PDB): ``path`` is the variable's value
    expanded and made absolute the way the tools resolve it, or None when it
    is unset or empty; ``exists`` says whether that path exists, the same
    check the tools make, and is None when no path is set.
    """
    files: dict[str, dict] = {}
    for arg in ("genome_fasta", "motif_file", "clinvar_vcf", "foldx_pdb"):
        raw = os.environ.get(REFERENCE_DATA_ENV[arg], "").strip()
        path, exists = None, None
        if raw:
            try:
                resolved = Path(raw).expanduser().resolve()
                path, exists = str(resolved), resolved.exists()
            except (OSError, RuntimeError, ValueError):
                path, exists = raw, False
        files[REFERENCE_DATA_ENV[arg]] = {"path": path, "exists": exists}
    return files


def _optional_reference(value: str | None, arg: str) -> tuple[Path | None, str | None]:
    """Like _resolve_reference for data a tool can do without.

    (None, None) when neither the argument nor its environment variable
    names a file (the step that needs it is skipped); otherwise the result
    of _resolve_reference, so a configured path that does not exist is
    still an error.
    """
    if not (value or "").strip() and not os.environ.get(REFERENCE_DATA_ENV[arg], "").strip():
        return None, None
    return _resolve_reference(value, arg)


def _fimo_binary_error(tool: str, skip_hint: str) -> str | None:
    """Error JSON when the MEME Suite's fimo program is not on PATH, else None.

    The ReMap, ClinVar and method-comparison pipelines run fimo for their
    FIMO leg and, unlike tfbs_fimo, have no tangermeme/memelite fallback.
    """
    if shutil.which("fimo") is not None:
        return None
    return _error(
        f"{tool} runs the MEME Suite's fimo program for its FIMO leg, and fimo is "
        "not on the PATH of the MCP server. This pipeline has no fallback: the "
        "tangermeme/memelite FIMO reimplementation is used only by tfbs_fimo.",
        "Install the MEME Suite (https://meme-suite.org/meme/doc/download.html, or "
        "conda install -c bioconda meme), put its bin directory on the PATH of the "
        f"MCP server and restart it{skip_hint}.",
    )


# ---------------------------------------------------------------------------
# Native libraries load at startup: main() calls _preload_native_modules()
# before the stdio loop starts, because on Windows a native extension that is
# imported for the first time after that loop is running never finishes
# loading.  The _ensure_* helpers bind the modules to these globals; they are
# idempotent, so in-process callers (tests, library use) can call them too.
# ---------------------------------------------------------------------------
_torch = None
_F = None
_np = None
_models = None
_find_best_windows_fast = None

# Every module with native code that a tool or a library it calls imports
# inside a function, so nothing native is imported for the first time while
# a tool call runs.  Optional extras are skipped when not installed.
_NATIVE_MODULES = (
    "numpy", "pandas", "yaml", "csv", "bisect",
    "scipy.linalg", "scipy.signal", "scipy.sparse", "scipy.stats",
    "sklearn.metrics", "sklearn.model_selection",
    "torchmetrics", "pytorch_lightning", "pytorch_lightning.cli",
    "pytorch_lightning.loggers", "jsonargparse",
    "captum.attr",
    "tangermeme.utils", "tangermeme.deep_lift_shap",
    "tangermeme.saturation_mutagenesis", "tangermeme.ism",
    "tangermeme.marginalize", "tangermeme.ersatz", "tangermeme.seqlet",
    "tangermeme.io", "tangermeme.annotate", "tangermeme.tools.fimo",
    "memelite.fimo",
    "pyfaidx", "pyBigWig", "pysam",
)


def _preload_native_modules() -> dict[str, str]:
    """Import torch, the genomics stack and every module in _NATIVE_MODULES.

    Returns {module: reason} for the modules that could not be imported
    (an optional extra that is not installed, typically); the tools that
    need one report the missing install when called.
    """
    missing: dict[str, str] = {}
    for name, loader in (("torch", _ensure_imports), ("pyfaidx", _ensure_genomics_imports)):
        try:
            loader()
        except Exception as exc:
            missing[name] = f"{type(exc).__name__}: {exc}"
    for name in _NATIVE_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            missing[name] = f"{type(exc).__name__}: {exc}"
    try:
        _warm_up_python_fimo()
    except Exception as exc:
        logger.info("FIMO reimplementation warm-up failed: %s: %s", type(exc).__name__, exc)
    return missing


_WARMUP_MOTIF = """MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF warmup
letter-probability matrix: alength= 4 w= 8 nsites= 1 E= 0
 0.97 0.01 0.01 0.01
 0.01 0.97 0.01 0.01
 0.01 0.01 0.97 0.01
 0.01 0.01 0.01 0.97
 0.97 0.01 0.01 0.01
 0.01 0.97 0.01 0.01
 0.01 0.01 0.97 0.01
 0.01 0.01 0.01 0.97

"""


def _warm_up_python_fimo() -> None:
    """Scan one sequence with the FIMO reimplementation, if installed.

    The first scan compiles memelite's numba functions and loads numba's
    threading layer (numba.np.ufunc.omppool or another native extension),
    which must happen before the stdio loop starts.
    """
    fimo, _ = _python_fimo()
    if fimo is None:
        return
    with tempfile.TemporaryDirectory(prefix="tfbs_fimo_warmup_", ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)
        motif = work / "warmup.meme"
        with open(motif, "w", encoding="utf-8", newline="\n") as f:
            f.write(_WARMUP_MOTIF)
        _python_fimo_hits(fimo, motif, work / "warmup.fasta", work / "fimo_out", 1e-3,
                          ["TTACGTACGTTT"])


_MODULE_ALTERNATIVES = (
    ("tangermeme.ism", "tangermeme.saturation_mutagenesis"),
    ("tangermeme.tools.fimo", "memelite.fimo"),
)


def _unavailable_modules(missing: dict[str, str]) -> list[str]:
    """The modules of missing that no loaded alternative stands in for, sorted."""
    unavailable = []
    for name in sorted(missing):
        group = next((g for g in _MODULE_ALTERNATIVES if name in g), (name,))
        if all(member in missing for member in group):
            unavailable.append(name)
    return unavailable


def _detach_child_stdin() -> None:
    """Point the process-level standard input handle at NUL (Windows only).

    The stdio transport keeps reading the protocol through fd 0 and
    sys.stdin, which hold their own handle.  A child process started
    without an explicit stdin (e.g. by jsonargparse inside
    load_from_checkpoint) would otherwise inherit the protocol pipe and
    stall.
    """
    if sys.platform != "win32":
        return
    import ctypes
    import msvcrt

    handle = msvcrt.get_osfhandle(os.open(os.devnull, os.O_RDONLY))
    ctypes.windll.kernel32.SetStdHandle(-10, handle)


def _ensure_imports():
    """Import torch, numpy and the model classes (once; later calls return at once)."""
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
# Genomics imports (pyfaidx, pandas), loaded at startup like the above
# ---------------------------------------------------------------------------
_pyfaidx = None
_pd = None
_genome_cache: dict[str, Any] = {}


def _ensure_genomics_imports():
    """Import pyfaidx and pandas (once; later calls return at once)."""
    global _pyfaidx, _pd
    if _pyfaidx is not None:
        return
    import pyfaidx
    import pandas as pd

    _pyfaidx = pyfaidx
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
    # models that flatten their input with .view() (e.g. MLP) raise
    # "view size is not compatible with input tensor's size and stride" on one.
    # Batches from the DataLoader are contiguous via collation, so this path
    # was the only one that hit it.
    return ohe.permute(0, 2, 1).contiguous()  # (N, 4, L)


def _tensor_to_sequences(tensor) -> list[str]:
    """Convert a (N, 4, L) one-hot tensor back to DNA strings."""
    indices = tensor.argmax(dim=1).cpu().numpy()  # (N, L)
    return ["".join(_IDX_TO_NUC[i] for i in row) for row in indices]


def _get_model(model_id: str) -> tuple[dict[str, Any] | None, str | None]:
    """(registry entry, None), or (None, error JSON) for an unknown model_id."""
    if model_id in _model_registry:
        return _model_registry[model_id], None
    available = list(_model_registry.keys()) or ["(none)"]
    return None, _error(
        f"Model '{model_id}' not loaded. Available: {', '.join(available)}.",
        "Call tfbs_load_model with this model_id first; tfbs_list_models "
        "lists the loaded models.",
    )


def _validate_dna(sequences: list[str], what: str = "Sequence {}",
                  ids: list[str] | None = None) -> str | None:
    """Error JSON for the first empty or non-ACGT sequence, else None.

    ``what`` names a sequence in the message; ``{}`` becomes its index, or
    its entry in ``ids`` when given (FASTA record names).
    """
    valid = set("ACGTacgt")
    for i, seq in enumerate(sequences):
        if not isinstance(seq, str):
            return _error(f"{what.format(ids[i] if ids else i)} is not a string.",
                          "Pass DNA sequences as strings of A, C, G and T.")
        bad = set(seq) - valid
        if bad or not seq:
            return _error(
                f"{what.format(ids[i] if ids else i)} is empty or contains "
                f"characters other than A, C, G, T: {sorted(bad)}.",
                "Remove or replace N and IUPAC codes; tfbs_extract_loci already "
                "drops windows that contain N.",
            )
    return None


def _warning_messages(caught: list, limit: int = 10, width: int = 300) -> list[dict[str, Any]]:
    """Distinct recorded warnings as [{"message", "count"}], in first-seen order.

    A warning is identified by its category and the text of its message
    before the first ": " (colon and space), so tangermeme's "Convergence
    deltas too high: tensor([...])" counts as one message whatever the
    tensor holds, while a path such as C:/data/x.pt stays whole.
    """
    counts: dict[str, int] = {}
    for w in caught:
        head = " ".join(str(w.message).split(": ", 1)[0].split())
        text = f"{w.category.__name__}: {head}"
        text = text if len(text) <= width else text[: width - 3] + "..."
        counts[text] = counts.get(text, 0) + 1
    return [{"message": m, "count": c} for m, c in counts.items()][:limit]


def _no_sequences_error(tool: str, fix: str) -> str:
    return _error(f"{tool} got no sequences.", fix)


def _tail_file(path: str, n: int = 100) -> str:
    """Read the last *n* lines of a file efficiently."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
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
#                  peaks_loci_24bp_gc/ -> peaks_24bp_gc_classification/ ->
#                  peaks_24bp_gc_dataset/
#   model analyses <results>/<model_id>/<tool>/<digest>/, where <results>
#                  is tfbs_server_info's results_dir and <digest> hashes
#                  the call's inputs, so a repeated call reuses its
#                  directory and different inputs never overwrite each other.
# ---------------------------------------------------------------------------
def _error(error: str, fix: str, **extra: Any) -> str:
    return json.dumps({"error": error, "fix": fix, **extra})


def _available_model_types() -> list[str]:
    """The model classes of tfbs.nn.models a model_type may name.

    The rule is tfbs.nn.loading.model_class_names, the one the loader
    applies to a model_type and to an inference checkpoint's arch.
    """
    from tfbs.nn.loading import model_class_names

    _ensure_imports()
    return model_class_names()


def _model_type_error(model_type: str, argument: str = "model_type") -> str | None:
    """{error, fix} JSON when model_type names no model class in tfbs.nn.models.

    Checked before a pipeline runs or is submitted, so an unknown class
    fails at once instead of when the SLURM job starts.  argument is the
    parameter's name in the error.
    """
    available = _available_model_types()
    if str(model_type) in available:
        return None
    return _error(f"Unknown {argument} '{model_type}'.",
                  f"Use one of: {available}.", available=available)


def _mkdir_error(path: Path, param: str = "output_dir") -> str | None:
    """Create the directory path (and its parents); {error, fix} JSON when that fails.

    Every tool that writes into a directory it was given creates it through
    this, so a path naming an existing file, or one the server cannot write
    to, comes back as an error instead of an exception.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if path.exists() and not path.is_dir():
            reason = "a file of that name already exists"
        else:
            reason = exc.strerror or str(exc)
        return _error(
            f"Cannot create the directory {path} ({param}): {reason}.",
            f"Check {param}: it must lead to a directory the server can create and write "
            "to (an existing directory, or a path that does not exist yet), not to an "
            "existing file.",
        )
    return None


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
_NEGATIVES_SUFFIX = {"gc_matched": "_gc", "shuffled": "_shuffled"}
_LOCI_DIR_RE = re.compile(r"^(.+)_loci(_\d+bp(?:_gc|_shuffled)?)$")


def _sibling_dir(input_file: str | Path, kind: str) -> Path:
    """<parent>/<name>_<kind> for an input file in <parent>/<name>[_<suffix>]/.

    A tfbs_extract_loci directory <stem>_loci_<n>bp[_gc|_shuffled] keeps
    its window and negatives mode: <stem>_<n>bp[_gc|_shuffled]_<kind>.
    """
    d = Path(input_file).expanduser().resolve().parent
    name = d.name
    loci = _LOCI_DIR_RE.match(name)
    if loci:
        return d.parent / f"{loci.group(1)}{loci.group(2)}_{kind}"
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
    err = _validate_dna(seqs, where, ids)
    if err:
        return None, None, err
    lengths = sorted({len(s) for s in seqs})
    if len(lengths) > 1:
        return None, None, _error(
            f"Sequences differ in length ({lengths[0]} to {lengths[-1]} bp).",
            "Give sequences of one length, e.g. windows from tfbs_extract_loci "
            "with a fixed in_window.",
        )
    return _one_hot(seqs), ids, None


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------
def _load_weights_only(path: Path) -> Any:
    """The object torch.load(weights_only=True) reads from path, or None."""
    _ensure_imports()
    for kwargs in ({"mmap": True}, {}):
        try:
            return _torch.load(path, map_location="cpu", weights_only=True, **kwargs)
        except Exception:
            continue
    return None


def _checkpoint_meta(path: Path, ckpt: Any = None) -> dict[str, dict]:
    """hyper_parameters and datamodule_hyper_parameters of a checkpoint ({} each if absent)."""
    if ckpt is None:
        ckpt = _load_weights_only(path)
    meta: dict[str, dict] = {}
    for key in ("hyper_parameters", "datamodule_hyper_parameters"):
        value = ckpt.get(key) if isinstance(ckpt, dict) else None
        meta[key] = dict(value) if isinstance(value, dict) else {}
    return meta


def _checkpoint_class_path(path: Path, meta: dict | None = None) -> str | None:
    """The model class path a LightningCLI checkpoint records, or None.

    LightningCLI stores it as hyper_parameters["_class_path"], e.g.
    "classes.nn.models.VCNNBpnet" for the NKX2-1 study's checkpoints.
    """
    meta = meta if meta is not None else _checkpoint_meta(path)
    value = meta["hyper_parameters"].get("_class_path")
    return value if isinstance(value, str) and "." in value else None


def _recorded_scaling(ckpt: Path, meta: dict) -> str | None:
    """The label scaling a checkpoint was trained with, as far as it can be told.

    LightningCLI records the data module's scaling_method; checkpoints
    without it are read from their folder name, which tfbs_config and the
    NKX2-1 study set to the scaling method.
    """
    dm = meta.get("datamodule_hyper_parameters") or {}
    if "scaling_method" in dm:
        return dm["scaling_method"] or "none"
    return ckpt.parent.name if ckpt.parent.name in _SCALING_METHODS else None


def _checkpoint_training_scaling(ckpt: Path) -> tuple[str | None, bool | None]:
    """(scaling_method, classify) a checkpoint was trained with; None where it records nothing.

    Reads the records tfbs_load_model trusts, without building the model:
    an inference checkpoint's own scaling block and hyper_parameters, or, for
    a Lightning checkpoint, the data module's recorded scaling_method or the
    scaling folder it was saved in (_recorded_scaling) and the model's
    classify flag.  A file torch cannot read gives (None, None).
    """
    payload = _load_weights_only(ckpt)
    if not isinstance(payload, dict):
        return None, None
    if _is_inference_payload(payload):
        own = payload.get("scaling")
        scaling = own.get("scaling_method") if isinstance(own, dict) else None
        hparams = payload.get("hyper_parameters")
    else:
        meta = _checkpoint_meta(ckpt, payload)
        scaling = _recorded_scaling(ckpt, meta)
        hparams = meta["hyper_parameters"]
    classify = hparams.get("classify") if isinstance(hparams, dict) else None
    # Release the (possibly memory-mapped) file before returning.
    del payload, hparams
    return (scaling if scaling in _SCALING_METHODS else None,
            classify if isinstance(classify, bool) else None)


def _validate_scaling(ckpt: Path, data: Path, requested: str | None
                      ) -> tuple[str | None, str | None, list[str], str | None]:
    """(scaling_method, source, cautions, None), or (None, None, [], error) for tfbs_validate.

    trainer.test() scores the model's outputs, which are on its training
    scale, against labels put on scaling_method's scale, so the two must
    agree: otherwise test_loss, and for a classifier test_acc and test_auc,
    compare two different scales.  Without a requested method the
    checkpoint's own is used.  When the checkpoint records none, the task
    decides as in tfbs_config: "none" for a classifier (its classify flag,
    else 0/1 labels in the dataset) and "standardize" otherwise.
    """
    if requested is not None and requested not in _SCALING_METHODS:
        return None, None, [], _error(
            f"Unknown scaling_method '{requested}'.",
            f"Pass one of {list(_SCALING_METHODS)}, or omit scaling_method to use the "
            "scaling the checkpoint was trained with. Nothing was submitted.")
    recorded, classify = _checkpoint_training_scaling(ckpt)
    binary = None
    if classify is None and data.is_dir():
        for split in ("test", "train"):
            binary = _binary_labels(data / split / "labels.pt")
            if binary is not None:
                break
    cautions: list[str] = []
    if classify and recorded not in (None, "none"):
        cautions.append(
            f"{ckpt.name} is a classifier but records label scaling '{recorded}', so it was "
            "trained against scaled 0/1 labels. It is validated on that scale, and its "
            "test_acc and test_auc are not those of a classifier trained on 0/1 labels.")
    if requested is None:
        if recorded:
            return recorded, "checkpoint", cautions, None
        if classify is not None:
            return ("none" if classify else "standardize"), "checkpoint classify flag", cautions, None
        if binary:
            return "none", "labels", cautions, None
        cautions.append(
            "The scaling could not be read from the checkpoint or from 0/1 labels, so "
            "'standardize' is assumed; pass scaling_method if the model was trained otherwise.")
        return "standardize", "default", cautions, None
    if recorded and requested != recorded:
        return None, None, [], _error(
            f"scaling_method '{requested}' differs from '{recorded}', the scaling "
            f"{ckpt.name} was trained with: trainer.test() scores the model's outputs on "
            "its training scale, so the metrics would compare two different scales.",
            f"Omit scaling_method, or pass scaling_method='{recorded}'. Nothing was submitted.",
            checkpoint_scaling=recorded)
    if recorded is None and classify and requested != "none":
        return None, None, [], _error(
            f"scaling_method '{requested}' cannot validate {ckpt.name}: it is a classifier, "
            "whose binary cross-entropy loss, accuracy and ROC-AUC need the 0/1 labels unscaled.",
            "Omit scaling_method (a classifier is validated with 'none') or pass "
            "scaling_method='none'. Nothing was submitted.")
    if recorded is None and classify is None and binary and requested != "none":
        cautions.append(
            f"Every label in {data} is 0 or 1, but scaling_method '{requested}' validates the "
            "checkpoint as a regression model; omit scaling_method if it is a classifier.")
    return requested, "argument", cautions, None


def _load_checkpoint(path: Path, model_type: str, device: str, meta: dict):
    """(model, None) or (None, error JSON) for a Lightning checkpoint.

    The model is built by tfbs.nn.loading.load_checkpoint, the loader the
    pipelines share.  Lightning resolves the class path stored in a
    LightningCLI checkpoint and, depending on its version, refuses one
    whose module is not imported yet.  The module named in the checkpoint
    is imported first; when loading still fails over a class path whose
    module was not imported, that module is imported and the load retried
    once.
    """
    from tfbs.nn.loading import UnimportableClassError, load_checkpoint

    class_path = _checkpoint_class_path(path, meta)
    try:
        # Only the hyper_parameters are passed on: a frame holding the
        # memory-mapped checkpoint would keep its file mapped (see
        # tfbs.nn.loading.load_checkpoint).
        model, _ = load_checkpoint(path, model_type, map_location=device,
                                   payload={"hyper_parameters": meta["hyper_parameters"]})
        return model, None
    except UnimportableClassError as exc:
        return None, _error(
            f"The checkpoint was saved with the class {exc.class_path}, and its "
            f"module {exc.module} cannot be imported by the server.",
            f"Put the package that defines {exc.module} on the PYTHONPATH of the "
            "MCP server (the 'env' block of the client config) and restart it; "
            "for the NKX2-1 study checkpoints that is the TF-NN workspace, whose "
            "classes package maps classes.* to tfbs.*. Or re-save the checkpoint "
            "with hyper_parameters['_class_path'] set to the matching tfbs class, "
            f"tfbs.nn.models.{model_type}.",
            class_path=exc.class_path,
        )
    except Exception as exc:
        failure = exc
    return None, _error(
        f"Could not load {path} as {model_type}: {type(failure).__name__}: {str(failure)[:1500]}",
        "Check that model_type is the architecture the checkpoint was trained "
        "with (its hyper_parameters name the class) and that the file is a "
        "PyTorch Lightning checkpoint.",
        class_path=class_path,
    )


_SCALING_FILE = "scaling_params.json"


def _scaling_key(ckpt: Path, model_type: str) -> str:
    """<dataset>_<scaling>_<model_type> for <dataset>/<scaling>/best_<model>.ckpt.

    The NKX2-1 study names its per-model files this way, e.g.
    all_mean_standardize_VCNNBpnet.json, and uses the same names as keys
    of its combined file.
    """
    return f"{ckpt.parent.parent.name}_{ckpt.parent.name}_{model_type}"


def _read_scaling_file(path: Path, key: str) -> tuple[dict | None, str | None]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        return None, _error(f"Could not read scaling parameters from {path}: {exc}",
                            "Pass the scaling_params.json written next to the training data.")
    if isinstance(data, dict) and "scaling_method" not in data and isinstance(data.get(key), dict):
        data = data[key]
    if not isinstance(data, dict) or "scaling_method" not in data:
        keys = sorted(data) if isinstance(data, dict) else []
        return None, _error(
            f"{path} has no scaling_method" + (f" and no entry {key}" if keys else "") + ".",
            "Pass a scaling_params.json as written by training "
            '({"scaling_method": ..., "mean": ..., "std": ...}), or a combined '
            "file keyed by <dataset>_<scaling>_<model_type>."
            + (f" Keys in this file: {keys[:20]}." if keys else ""),
        )
    err = _incomplete_scaling(data, path)
    return (None, err) if err else (data, None)


_SCALING_VALUES = {"standardize": ("mean", "std"), "normalize": ("min", "max")}


def _incomplete_scaling(data: dict, source: Path) -> str | None:
    """Error JSON unless the values its scaling_method needs are usable, else None.

    mean and std (standardize) or min and max (normalize) must be finite
    numbers other than bool, with std != 0 and max != min, because the
    inverse transform multiplies by std (or max - min).
    """
    method = data["scaling_method"]
    needed = _SCALING_VALUES.get(method, ()) if isinstance(method, str) else ()
    fix = "Pass the scaling_params.json written by the training run."
    missing = [k for k in needed if data.get(k) is None]
    if missing:
        return _error(f"{source} uses scaling_method {method} but lacks {missing}.", fix)
    bad = [k for k in needed
           if isinstance(data[k], bool) or not isinstance(data[k], (int, float))
           or not math.isfinite(data[k])]
    if bad:
        shown = ", ".join(f"{k}={data[k]!r}" for k in bad)
        return _error(
            f"{source} uses scaling_method {method}, but {shown} is not a finite number.",
            fix)
    if method == "standardize" and data["std"] == 0:
        return _error(f"{source} has std 0, so its scores cannot be converted to label units.",
                      fix)
    if method == "normalize" and data["max"] == data["min"]:
        return _error(f"{source} has max equal to min ({data['min']!r}), so its scores "
                      "cannot be converted to label units.", fix)
    return None


def _training_data_dir(ckpt: Path, meta: dict) -> Path | None:
    """The data_path the checkpoint's data module recorded, as an absolute path."""
    dm = meta.get("datamodule_hyper_parameters") or {}
    raw = dm.get("data_path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    p = Path(raw).expanduser()
    return p if p.is_absolute() else _project_root() / p


def _default_scaling_candidates(ckpt: Path, meta: dict, key: str) -> list[Path]:
    """Where scaling parameters are looked for without data_path, in order."""
    candidates = [ckpt.parent / _SCALING_FILE]
    data_dir = _training_data_dir(ckpt, meta)
    if data_dir is not None:
        candidates.append(data_dir / _SCALING_FILE)
    study = ckpt.parent.parent.parent / "scaling_params"
    candidates += [study / f"{key}.json", study / "scaling_params_all_models.json"]
    return candidates


def _keyed_entry(path: Path, key: str) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and isinstance(data.get(key), dict)


def _find_scaling_params(data_path: str | None, ckpt: Path, model_type: str,
                         meta: dict | None = None) -> tuple[dict | None, str | None, str | None]:
    """(params, source file, None); (None, None, None) when there are none; or an error.

    data_path is a scaling JSON, or a directory holding scaling_params.json,
    <dataset>_<scaling>_<model_type>.json or exactly one *scaling*.json.
    Without data_path the candidates of _default_scaling_candidates are
    tried in order.  The file's scaling_method must be the one the
    checkpoint was trained with.
    """
    key = _scaling_key(ckpt, model_type)
    meta = meta if meta is not None else _checkpoint_meta(ckpt)
    if not data_path:
        found = [p for p in _default_scaling_candidates(ckpt, meta, key) if p.is_file()]
        found = [p for p in found
                 if p.name != "scaling_params_all_models.json" or _keyed_entry(p, key)]
        if not found:
            return None, None, None
        candidate = found[0]
    else:
        p = Path(data_path).expanduser().resolve()
        if p.is_file():
            candidate = p
        elif not p.is_dir():
            return None, None, _error(
                f"data_path not found: {data_path}",
                "Pass the training data directory (with scaling_params.json), "
                "or the scaling JSON itself.")
        else:
            named = [p / n for n in (_SCALING_FILE, f"{key}.json") if (p / n).is_file()]
            matches = sorted(p.glob("*scaling*.json"))
            if named:
                candidate = named[0]
            elif len(matches) == 1:
                candidate = matches[0]
            else:
                return None, None, _error(
                    f"No scaling parameters for this checkpoint in {p}: no "
                    f"{_SCALING_FILE}, no {key}.json, and {len(matches)} files "
                    "match *scaling*.json.",
                    "Pass the scaling JSON itself as data_path, or the training "
                    "data directory whose scaling_params.json the training run wrote.",
                    json_files=sorted(x.name for x in p.glob("*.json"))[:50],
                )
    params, err = _read_scaling_file(candidate, key)
    if err:
        return None, None, err
    recorded = _recorded_scaling(ckpt, meta)
    if recorded and params["scaling_method"] != recorded:
        return None, None, _error(
            f"{candidate} holds scaling_method '{params['scaling_method']}', but the "
            f"checkpoint was trained with scaling_method '{recorded}'.",
            "Pass as data_path the scaling_params.json written by the training run "
            "of this checkpoint (it is in that run's data directory).",
            scaling_source=str(candidate), checkpoint_scaling_method=recorded,
        )
    return params, str(candidate), None


def _json_safe(value: Any) -> Any:
    """value with tensors and numpy arrays or scalars turned into lists and numbers.

    Other objects are left as they are; json.dumps(default=str) renders them.
    """
    if isinstance(value, dict):
        return {k if isinstance(k, (str, int, float, bool)) or k is None else str(k): _json_safe(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    for convert in ("tolist", "item"):
        method = getattr(value, convert, None)
        if callable(method):
            try:
                converted = method()
            except Exception:
                continue
            if converted is not value:
                return _json_safe(converted)
    return value


def _is_inference_payload(ckpt: Any) -> bool:
    """A released inference checkpoint: {arch, hyper_parameters, state_dict, ...}."""
    from tfbs.nn.loading import is_inference_payload

    return is_inference_payload(ckpt)


def _build_inference_model(payload: dict, path: Path, model_type: str, device: str):
    """(model, None) or (None, error JSON) for an inference checkpoint.

    The model is built by tfbs.nn.loading.load_checkpoint, the loader the
    pipelines share, on device and in eval mode.
    """
    from tfbs.nn.loading import load_checkpoint

    arch = payload["arch"]
    if arch != model_type:
        return None, _error(
            f"{path} holds a {arch} model, but model_type is {model_type}.",
            f"Pass model_type='{arch}', the architecture the file records.", arch=arch)
    try:
        model, _ = load_checkpoint(path, model_type, map_location=device, payload=payload)
    except Exception as exc:
        return None, _error(
            f"Could not build {arch} from {path}: {type(exc).__name__}: {str(exc)[:1500]}",
            f"The file's hyper_parameters and state_dict must fit tfbs.nn.models.{arch} "
            "of the installed tfbs-mcp; update tfbs-mcp, or check that the file is an "
            "unmodified release.", arch=arch)
    return model, None


def _inference_scaling(payload: dict, path: Path, model_type: str, data_path: str | None
                       ) -> tuple[dict | None, str | None, str | None]:
    """(params, source, None) or an error, for an inference checkpoint.

    The file's own scaling block is used; data_path is then optional and,
    if given, must hold the same values.  A file without a scaling block
    falls back to the lookup of _find_scaling_params.
    """
    try:
        own = _json_safe(payload.get("scaling"))
    except RecursionError:
        return None, None, _error(
            f"The scaling block of {path} refers to itself or is nested too deeply to read.",
            "Save scaling as a plain dict such as "
            '{"scaling_method": "standardize", "mean": ..., "std": ...}.')
    if not (isinstance(own, dict) and isinstance(own.get("scaling_method"), str)):
        return _find_scaling_params(data_path, path, model_type,
                                    {"hyper_parameters": {}, "datamodule_hyper_parameters": {}})
    err = _incomplete_scaling(own, path)
    if err:
        return None, None, err
    own = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
           for k, v in own.items()}
    if data_path:
        meta = {"hyper_parameters": {},
                "datamodule_hyper_parameters": {"scaling_method": own["scaling_method"]}}
        given, source, err = _find_scaling_params(data_path, path, model_type, meta)
        if err:
            return None, None, err
        if given is not None:
            differ = [k for k in _SCALING_VALUES.get(own["scaling_method"], ())
                      if abs(float(given[k]) - own[k]) > 1e-6 * max(1.0, abs(own[k]))]
            if differ:
                got = {k: given[k] for k in differ}
                stored = {k: own[k] for k in differ}
                return None, None, _error(
                    f"{source} disagrees with the scaling stored in {path}: {got} vs {stored}.",
                    "Omit data_path: the released checkpoint carries its own scaling.",
                    scaling_source=source, checkpoint_scaling=own)
    return own, str(path), None


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

    Two checkpoint formats are accepted.  An inference checkpoint is a
    dict saved with torch.save that holds arch (the tfbs.nn.models class
    name), hyper_parameters (its constructor arguments), state_dict,
    scaling ({scaling_method, mean, std}) and optionally library and
    metadata, such as the NKX2-1 study's released vcnnbpnet.pt files
    (Zenodo, doi:10.5281/zenodo.21399578; one each for the core, flank
    and all libraries).  It is read with weights_only=True, rebuilt from
    its hyper_parameters and loaded strictly; its scaling comes from the
    file, so data_path is optional and, when given, must hold the same
    values.  Any other file is loaded as a PyTorch Lightning checkpoint.

    A checkpoint written by LightningCLI records the class it was trained
    as (e.g. classes.nn.models.VCNNBpnet for the NKX2-1 study); its module
    is imported before loading, and a module that cannot be imported is
    reported with the PYTHONPATH fix instead of a traceback.

    Label scaling: a regression model trained on standardized or
    normalized labels predicts in that scale.  When scaling parameters
    are found, tfbs_predict (and tfbs_mutagenesis, tfbs_marginalize and
    tfbs_epistasis) report the original label units by default.  They
    are looked up in data_path; without it, in this order:
    scaling_params.json next to the checkpoint; scaling_params.json in
    the training data directory the checkpoint records (its
    datamodule_hyper_parameters.data_path, where training writes it, so
    tfbs_config -> tfbs_train -> tfbs_load_model needs no data_path);
    and the study layout <ckpt dir>/../../scaling_params/
    <dataset>_<scaling>_<model_type>.json or the combined
    scaling_params_all_models.json keyed by that name.  The file's
    scaling_method must match the one the checkpoint was trained with
    (recorded by LightningCLI, else the checkpoint's folder name), or the
    call returns an error naming both.  A regression checkpoint trained
    on scaled labels for which none are found loads with a "warning".
    A scaling block or file must hold finite numbers (mean and std, or
    min and max) with std != 0 (max != min), else the call returns an
    error.  A regression inference checkpoint with no scaling block and
    no scaling file loads with units "unknown" and a "warning" naming
    data_path, because nothing says whether it was trained on scaled
    labels; the analysis tools then report units "unknown" too.  Units
    "label" for a model with no recorded scaling assume it was trained
    on unscaled labels.

    Args:
        model_id: User-chosen identifier for this model (e.g. "my_vcnn").
        checkpoint_path: Absolute path to a Lightning .ckpt file or an
                         inference checkpoint (e.g. a released
                         vcnnbpnet.pt).
        model_type: Architecture class name from tfbs.nn.models
                    (e.g. "VCNNBpnet", "RNN"); for an inference
                    checkpoint it must equal the file's arch.
        device: "cpu" or "cuda".  Auto-detected if omitted (uses GPU when
                available on SLURM, CPU otherwise).
        data_path: Where the scaling parameters are: the training data
                   directory holding scaling_params.json; a directory
                   holding <dataset>_<scaling>_<model_type>.json (named
                   after the checkpoint's <dataset>/<scaling>/ folders, as
                   in the study's saved_models_final/scaling_params/) or
                   exactly one *scaling*.json; or the JSON file itself (a
                   combined file keyed by those names works too).

    Returns:
        JSON with model_id, model_type, checkpoint_format ("inference" or
        "lightning"), input_length, classify, device, scaling_params
        (null when none were found), scaling_source (the file they came
        from: the checkpoint itself for an inference checkpoint),
        checkpoint_scaling_method (the scaling the checkpoint was trained
        with, null if unknown), library and metadata when an inference
        checkpoint has them (tensors and numpy values converted to lists
        and numbers), and "warning" when a regression model's scaling
        parameters are missing, with "units": "unknown" for an inference
        checkpoint.  The model is registered only once this result is
        built.
    """
    _ensure_imports()

    if device is None:
        device = _ENV["default_device"]

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        return _error(f"Checkpoint not found: {checkpoint_path}",
                      "Pass the absolute path of a .ckpt file, e.g. the checkpoint_path "
                      "that tfbs_config reported.")

    err = _model_type_error(model_type)
    if err:
        return err

    from tfbs.nn.loading import checkpoint_problem

    ckpt = _load_weights_only(path)
    problem = checkpoint_problem(path, ckpt)
    if problem:
        del ckpt
        return _error(
            problem,
            "Pass a Lightning .ckpt or an inference checkpoint saved with torch.save as "
            "{arch, hyper_parameters, state_dict}, such as a released vcnnbpnet.pt; "
            "nothing was registered.")
    extra: dict[str, Any] = {}
    if _is_inference_payload(ckpt):
        checkpoint_format = "inference"
        scaling_params, scaling_source, err = _inference_scaling(ckpt, path, model_type, data_path)
        if err:
            return err
        model, err = _build_inference_model(ckpt, path, model_type, device)
        if err:
            return err
        hparams = ckpt["hyper_parameters"]
        input_length = hparams.get("input_length", getattr(model, "input_length", None))
        classify = bool(hparams.get("classify", getattr(model, "classify", False)))
        recorded = (scaling_params or {}).get("scaling_method")
        for key in ("library", "metadata"):
            if ckpt.get(key) is not None:
                extra[key] = ckpt[key]
        meta = None
    else:
        checkpoint_format = "lightning"
        meta = _checkpoint_meta(path, ckpt)
        scaling_params, scaling_source, err = _find_scaling_params(
            data_path, path, model_type, meta)
        if err:
            return err
        model, err = _load_checkpoint(path, model_type, device, meta)
        if err:
            return err
        input_length = getattr(model, "input_length", None)
        classify = getattr(model, "classify", False)
        recorded = _recorded_scaling(path, meta)
    del ckpt
    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    model = model.to(device)
    units_unknown = checkpoint_format == "inference" and not classify and scaling_params is None

    result = {
        "model_id": model_id,
        "model_type": model_type,
        "checkpoint_format": checkpoint_format,
        "input_length": input_length,
        "classify": classify,
        "device": device,
        "scaling_params": scaling_params,
        "scaling_source": scaling_source,
        "checkpoint_scaling_method": recorded,
        **extra,
    }
    if not classify and scaling_params is None and recorded not in (None, "none") and meta:
        data_dir = _training_data_dir(path, meta)
        result["warning"] = (
            f"This regression model was trained on '{recorded}'-scaled labels, but no "
            "scaling parameters were found, so tfbs_predict reports scores in the "
            "training scale. Load it again with data_path set to the training data "
            "directory holding scaling_params.json"
            + (f" (the checkpoint records {data_dir}, which has none)" if data_dir else "")
            + ", or to that JSON file."
        )
    if units_unknown:
        result["units"] = "unknown"
        result["warning"] = (
            "This inference checkpoint of a regression model carries no scaling block "
            "and no scaling parameters were found, so it is unknown whether its scores "
            "are in label units or in a training scale; the analysis tools report units "
            "\"unknown\". If it was trained on scaled labels, load it again with "
            "data_path set to the scaling_params.json of its training run (or the "
            "directory holding it)."
        )
    try:
        result.update({key: _json_safe(value) for key, value in extra.items()})
        text = json.dumps(result, default=str)
    except (TypeError, ValueError, RecursionError) as exc:
        return _error(f"The checkpoint's library or metadata cannot be reported as JSON: {exc}",
                      "Save library and metadata as plain strings, numbers, lists and dicts.")
    _model_registry[model_id] = {
        "model": model,
        "model_type": model_type,
        "device": device,
        "input_length": input_length,
        "classify": classify,
        "scaling_params": scaling_params,
        "checkpoint_scaling_method": recorded,
        "checkpoint_format": checkpoint_format,
        "units_unknown": units_unknown,
    }
    return text


# ---------------------------------------------------------------------------
# Tool 2: tfbs_predict
# ---------------------------------------------------------------------------
def _inverse_scaling(entry: dict[str, Any], inverse_transform: bool | None):
    """The function that maps model outputs to label units, or None to leave them."""
    sp = entry.get("scaling_params")
    if inverse_transform is False or not sp or entry.get("classify"):
        return None
    if sp.get("scaling_method") == "standardize":
        return lambda v: v * sp["std"] + sp["mean"]
    if sp.get("scaling_method") == "normalize":
        return lambda v: v * (sp["max"] - sp["min"]) + sp["min"]
    return None


def _units(entry: dict[str, Any], inverse) -> str:
    """The units of a model's scores: label units, the training scale, or unknown.

    "label" when the model has no label scaling (classifiers, scaling_method
    "none", or none recorded, which assumes it was trained on unscaled
    labels) or the scaling was undone; "training_scale" for a scaled
    regression model whose scores were left as the model returns them;
    "unknown" for a regression inference checkpoint that carries no
    scaling block and for which no scaling file was found.
    """
    if inverse is not None or entry.get("classify"):
        return "label"
    if entry.get("units_unknown"):
        return "unknown"
    method = ((entry.get("scaling_params") or {}).get("scaling_method")
              or entry.get("checkpoint_scaling_method"))
    return "label" if method in (None, "none") else "training_scale"


def _delta_scale(entry: dict[str, Any]) -> float | None:
    """Factor that turns a difference of model outputs into label units.

    None exactly when _inverse_scaling(entry, None) is None.
    """
    sp = entry.get("scaling_params")
    if not sp or entry.get("classify"):
        return None
    if sp.get("scaling_method") == "standardize":
        return float(sp["std"])
    if sp.get("scaling_method") == "normalize":
        return float(sp["max"] - sp["min"])
    return None


def _sequence_length(entry: dict[str, Any]) -> int | None:
    """Window length in bp the model takes (FlexibleMLP counts 4 inputs per base)."""
    length = entry.get("input_length")
    if not isinstance(length, int) or length < 1:
        return None
    if entry.get("model_type") == "FlexibleMLP":
        return length // 4 if length % 4 == 0 else None
    return length


def _output_dim(model, X_probe, device) -> int:
    with _torch.no_grad():
        out = model(X_probe.to(device))
    return int(_torch.as_tensor(out).reshape(X_probe.shape[0], -1).shape[1])


def _missing_scaling_error(model_id: str) -> str:
    return _error(
        f"inverse_transform=true, but model '{model_id}' was loaded without scaling "
        "parameters, so its scores cannot be converted to the original label units.",
        "Call tfbs_load_model again with data_path set to the training data directory "
        "holding scaling_params.json (or to that JSON file), or pass "
        "inverse_transform=null to get the training scale.",
    )


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
    window_size: int | None = None,
    batch_size: int | None = None,
    inverse_transform: bool | None = None,
    sequences_path: str | None = None,
) -> str:
    """Score DNA sequences for transcription-factor binding affinity.

    In normal mode each sequence must match the model's expected input
    length; a different length returns an error that suggests
    window_mode.  In window_mode a sliding window of the model's input
    length finds the highest-scoring sub-sequence within longer inputs.
    A model with more than one output per sequence is reported as an
    error (output_dim) rather than reduced to its first output.
    Sequences come inline
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
        window_size: Window length for sliding-window mode.  Default: the
                     model's input length (70 when the model reports
                     none).  It must equal the input length when that is
                     known, and must not exceed the sequence length.
        batch_size: Batch size for processing.  Auto-detected if omitted
                    (512 on GPU, 64 on CPU).
        inverse_transform: Default null, because predictions are meant
                           to be read in the original label units (e.g.
                           log2 fold change): the scores of a regression
                           model are converted back whenever
                           tfbs_load_model found its scaling parameters,
                           and stay in the training scale when it found
                           none.  false is the explicit opt-out that keeps
                           the training scale (z-scores for standardize);
                           true insists on the original units and returns
                           an error, with the data_path fix, for a
                           regression model loaded without scaling
                           parameters.  Classifier scores are never
                           transformed.
        sequences_path: One-hot .pt tensor (N, 4, L) as written by
                        tfbs_extract_loci or tfbs_preprocess, or a FASTA
                        file (.fa/.fasta/.fna/.fas, optionally .gz) of
                        equal-length A/C/G/T records.

    Returns:
        JSON with "scores" (normal mode; always a list with one value per
        sequence) or "best_scores", "best_sequences" and "mean_scores"
        (window mode), plus "num_sequences", "inverse_transformed" (true
        when a label scaling was undone; false for a model trained with
        scaling_method "none" or without scaling parameters) and "units"
        ("label" when the scores are in label units, because the model
        has no label scaling or it was undone, which for a model with no
        recorded scaling assumes it was trained on unscaled labels;
        "training_scale" when a label scaling was left in place; "unknown"
        for a regression inference checkpoint loaded without any scaling
        block or file);
        for a FASTA input also "ids" (record names, in score order).
    """
    _ensure_imports()
    X, ids, err = _load_sequence_input(sequences, sequences_path)
    if err:
        return err

    if batch_size is None:
        batch_size = _ENV["default_batch_size"]

    entry, err = _get_model(model_id)
    if err:
        return err
    model = entry["model"]
    device = entry["device"]
    n, L = int(X.shape[0]), int(X.shape[2])
    if inverse_transform is True and not entry.get("classify") and not entry.get("scaling_params"):
        return _missing_scaling_error(model_id)
    inverse = _inverse_scaling(entry, inverse_transform)
    expected = _sequence_length(entry)

    if window_mode:
        if window_size is None:
            window_size = expected or 70
        if window_size > L:
            return _error(
                f"window_size {window_size} is longer than the sequences ({L} bp).",
                f"Pass sequences of at least {window_size} bp, or score {L} bp "
                "sequences with window_mode=false if that is the model's input length.",
            )
        if expected is not None and window_size != expected:
            return _error(
                f"window_size {window_size} differs from the input length of model "
                f"'{model_id}' ({expected} bp).",
                f"Pass window_size={expected}, or omit window_size.",
            )
    elif expected is not None and L != expected:
        return _error(
            f"The sequences are {L} bp, but model '{model_id}' takes {expected} bp.",
            f"Pass window_mode=true with window_size={expected} to score the best "
            f"{expected} bp window of each sequence, or extract the loci again with "
            f"in_window={expected}.",
        )

    output_dim = _output_dim(model, X[:1, :, :window_size if window_mode else L], device)
    if output_dim != 1:
        return _error(
            f"Model '{model_id}' returns output_dim={output_dim} values per sequence; "
            "tfbs_predict scores models with one output per sequence.",
            "Load a checkpoint with a single-output head (every tfbs.nn.models "
            "architecture has one), or reduce the outputs in a wrapper model.",
            output_dim=output_dim,
        )

    def _per_sequence(values) -> list[float]:
        arr = _np.asarray(values, dtype=float).reshape(n, -1)[:, 0]
        return (inverse(arr) if inverse else arr).tolist()

    if window_mode:
        best_windows, best_scores, _, mean_scores = _find_best_windows_fast(
            X, model, window_size=window_size,
            processing_batch_size=batch_size, device=device,
        )
        result = {
            "best_scores": _per_sequence(best_scores),
            "best_sequences": _tensor_to_sequences(_torch.from_numpy(best_windows)),
            "mean_scores": _per_sequence(mean_scores),
            "num_sequences": n,
        }
    else:
        chunks = []
        with _torch.no_grad():
            for k in range(0, n, batch_size):
                chunks.append(model(X[k:k + batch_size].to(device)).cpu())
        result = {"scores": _per_sequence(_torch.cat(chunks).numpy()), "num_sequences": n}
    result["inverse_transformed"] = inverse is not None
    result["units"] = _units(entry, inverse)
    if ids is not None:
        result["ids"] = ids
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
    if not sequences:
        return _no_sequences_error(
            "tfbs_preprocess",
            "Pass the DNA strings to encode as sequences (A/C/G/T, one length). "
            "Sequences already in a FASTA or .pt file need no preprocessing: "
            "tfbs_predict, tfbs_analyze and tfbs_mutagenesis read them through "
            "sequences_path.",
        )
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
    mkdir_err = _mkdir_error(out, "output_path")
    if mkdir_err:
        return mkdir_err
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
        sequence) and, for FASTA, "ids"; "warnings" lists the distinct
        warnings captum or tangermeme raised during the attribution,
        each once as {message, count}, with message the warning's
        category and its text before the first ": " (colon and space),
        so tensor values do not split one warning into many while a
        path such as C:/data/x.pt stays whole.
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
    entry, err = _get_model(model_id)
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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # captum says this on every call; it sets requires_grad itself.
        warnings.filterwarnings(
            "ignore", message="Input Tensor .* did not already require gradients")
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
    mkdir_err = _mkdir_error(out, "output_dir")
    if mkdir_err:
        return mkdir_err
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
    raised = _warning_messages(caught)
    if raised:
        result["warnings"] = raised
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

    Units: the same as tfbs_predict.  For a regression model loaded with
    scaling parameters, predictions are converted to the original label
    units (standardize: x * std + mean; normalize: x * (max - min) +
    min) and prediction changes are multiplied by std (or max - min);
    "inverse_transformed" says whether that happened, and "units" is
    "label", "training_scale" or "unknown" as in tfbs_predict.  Classifier
    outputs and models without scaling parameters stay as the model
    returns them.

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
        "files" with the paths written, "inverse_transformed" and
        "units".  For a
        sequences_path input the arrays are only in the files, and FASTA
        record names are in "ids".
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

    entry, err = _get_model(model_id)
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
    mkdir_err = _mkdir_error(out, "output_dir")
    if mkdir_err:
        return mkdir_err
    inline = sequences is not None
    inverse = _inverse_scaling(entry, None)
    scale = _delta_scale(entry) if inverse else 1.0

    if raw_outputs:
        # result is (y_orig, y_mutant) tuple
        y_orig, y_mut = (t.detach().cpu() for t in result)
        if inverse:
            y_orig, y_mut = inverse(y_orig), inverse(y_mut)
        files = {"original_scores": str(out / "original_scores.pt"),
                 "mutant_scores": str(out / "mutant_scores.pt")}
        _torch.save(y_orig, files["original_scores"])
        _torch.save(y_mut, files["mutant_scores"])
        payload: dict[str, Any] = {"files": files, "shape": list(y_mut.shape)}
        if inline:
            payload["original_scores"] = y_orig.numpy().tolist()
            payload["mutant_scores"] = y_mut.numpy().tolist()
    else:
        attr = result.detach().cpu() * scale
        files = {"scores": str(out / "ism_scores.pt")}
        _torch.save(attr, files["scores"])
        payload = {"shape": list(attr.shape), "method": "ism", "files": files}
        if inline:
            payload["scores"] = attr.numpy().tolist()
    payload["inverse_transformed"] = inverse is not None
    payload["units"] = _units(entry, inverse)
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
    sequences: list[str] | None,
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

    Units: the same as tfbs_predict.  For a regression model loaded with
    scaling parameters, predictions_before and predictions_after are in
    the original label units (standardize: x * std + mean; normalize:
    x * (max - min) + min), and the deltas are their difference, i.e.
    the model-scale delta times std (or max - min);
    "inverse_transformed" says whether that happened, and "units" is
    "label", "training_scale" or "unknown" as in tfbs_predict.

    Args:
        model_id: Identifier of a model loaded via tfbs_load_model.
        sequences: Background DNA sequences (A/C/G/T, one length), or
                   null when n_background is set.
        motif: DNA motif consensus to insert (e.g. "GATAAG"; A/C/G/T).
        start: 0-indexed position to insert the motif, from 0 to
               L - len(motif).  None = center of sequence.
        n_background: If set, ignore sequences and generate this many
                      random one-hot backgrounds instead.

    Returns:
        JSON with per-sequence lists "delta_predictions" (after - before),
        "predictions_before" and "predictions_after", plus "mean_delta",
        "motif", "inverse_transformed" and "units".
    """
    try:
        from tangermeme.marginalize import marginalize
        from tangermeme.utils import one_hot_encode, random_one_hot
    except ImportError:
        return _error("tangermeme required.", "Install with: pip install 'tfbs-mcp[viz]'")

    _ensure_imports()

    entry, err = _get_model(model_id)
    if err:
        return err
    model = entry["model"]
    device = entry["device"]
    input_length = _sequence_length(entry)

    err = _validate_dna([motif], "The motif")
    if err:
        return err

    if n_background is not None:
        if n_background < 1:
            return _error(f"n_background must be at least 1, got {n_background}.",
                          "Pass the number of random backgrounds, e.g. n_background=100.")
        if input_length is None:
            return _error(
                f"n_background needs the model's input length, which model "
                f"'{model_id}' does not report.",
                "Pass background sequences of the model's input length as sequences instead.",
            )
        X = random_one_hot((n_background, 4, input_length)).float()
    elif not sequences:
        return _no_sequences_error(
            "tfbs_marginalize",
            "Pass background DNA sequences as sequences (A/C/G/T, one length), "
            "or set n_background to insert the motif into that many random "
            "backgrounds of the model's input length.",
        )
    else:
        X, _, err = _load_sequence_input(sequences, None)
        if err:
            return err
    L = int(X.shape[2])
    if len(motif) > L:
        return _error(f"The motif ({len(motif)} bp) is longer than the backgrounds "
                      f"({L} bp).", "Pass a shorter motif or longer backgrounds.")
    if start is not None and not 0 <= start <= L - len(motif):
        return _error(
            f"start {start} does not fit the {len(motif)} bp motif into {L} bp backgrounds.",
            f"Pass start from 0 to {L - len(motif)}, or omit it to insert at the centre.",
        )

    X = X.to(device)

    # One-hot encode the motif string
    motif_ohe = one_hot_encode(motif.upper()).unsqueeze(0).float().to(device)

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

    n = int(X.shape[0])
    before = _np.asarray(y_before, dtype=float).reshape(n, -1)[:, 0]
    after = _np.asarray(y_after, dtype=float).reshape(n, -1)[:, 0]
    inverse = _inverse_scaling(entry, None)
    if inverse:
        before, after = inverse(before), inverse(after)
    delta = after - before

    return json.dumps({
        "predictions_before": before.tolist(),
        "predictions_after": after.tolist(),
        "delta_predictions": delta.tolist(),
        "mean_delta": float(delta.mean()),
        "motif": motif,
        "inverse_transformed": inverse is not None,
        "units": _units(entry, inverse),
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
    "error": "fimo not found on PATH, and no FIMO reimplementation (tangermeme or "
             "memelite) to fall back on",
    "fix": "Install tangermeme (pip install 'tfbs-mcp[viz]'; tangermeme 1.x brings "
           "memelite, its numba-based FIMO reimplementation), or install the MEME Suite "
           "(https://meme-suite.org/meme/doc/download.html) and put its bin directory "
           "on the PATH of the MCP server; then restart the MCP client.",
}

_FIMO_COLUMNS = ["motif", "sequence", "start", "end", "strand", "score", "p_value",
                 "matched_sequence"]


_PYTHON_FIMO_BACKENDS = {"tangermeme.tools.fimo": "tangermeme", "memelite.fimo": "memelite"}


def _python_fimo():
    """(fimo function, module name) of tangermeme's FIMO reimplementation, or (None, None).

    That is memelite (numba-based, shipped with tangermeme 1.x) as
    memelite.fimo, or tangermeme.tools.fimo in older tangermeme releases.
    """
    for module in _PYTHON_FIMO_BACKENDS:
        try:
            return importlib.import_module(module).fimo, module
        except (ImportError, AttributeError):
            continue
    return None, None


def _reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]


def _meme_fimo_hits(motif_path: Path, fasta_path: Path, fimo_out: Path, threshold: float):
    """(hits, None), (None, error JSON), or (None, None) when the binary is gone."""
    import pandas as pd

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
        result = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        return None, None
    if result.returncode != 0:
        return None, _error(
            f"FIMO failed (exit {result.returncode})",
            "Read stderr; a malformed MEME motif file is the usual cause.",
            stderr=result.stderr[:1000],
        )
    tsv_path = fimo_out / "fimo.tsv"
    if not tsv_path.exists():
        return None, _error("FIMO produced no output", "Read stderr and check the motif file.",
                            stderr=result.stderr[:500])
    try:
        df = pd.read_csv(tsv_path, sep="\t", comment="#")
    except pd.errors.EmptyDataError:
        return [], None
    hits = [
        {
            "motif": str(row["motif_id"]),
            "sequence": str(row["sequence_name"]),
            "start": int(row["start"]),
            "end": int(row["stop"]),
            "strand": str(row["strand"]),
            "score": float(row["score"]),
            "p_value": float(row["p-value"]),
            "matched_sequence": str(row["matched_sequence"]),
        }
        for row in df.to_dict(orient="records")
    ]
    return hits, None


def _terminated_motif_copy(motif_path: Path, work_dir: Path) -> Path:
    """A copy of motif_path in work_dir that ends with a blank line.

    memelite's read_meme stores a motif only when a line follows its
    last matrix row, so a file that ends right after that row would lose
    its last motif.
    """
    with open(motif_path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    copy = work_dir / "motifs.meme"
    with open(copy, "w", encoding="utf-8", newline="\n") as f:
        f.write(text if text.endswith("\n") else text + "\n")
        f.write("\n")
    return copy


def _python_fimo_hits(fimo, motif_path: Path, fasta_path: Path, fimo_out: Path,
                      threshold: float, sequences: list[str]) -> list[dict]:
    """Hits of tangermeme's FIMO reimplementation, in the MEME columns, written to fimo_out/fimo.tsv."""
    import csv

    motif_path = _terminated_motif_copy(motif_path, fasta_path.parent)
    # memelite's scan loop stops one window short of the end of every
    # sequence; one trailing N (scored as no base) brings the last window in.
    padded = fasta_path.with_name(fasta_path.stem + "_padded.fasta")
    with open(padded, "w", encoding="utf-8", newline="\n") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq{i}\n{seq}N\n")
    padded.with_name(padded.name + ".fai").unlink(missing_ok=True)
    frames = fimo(str(motif_path), str(padded), threshold=threshold,
                  reverse_complement=True)
    by_name = {f"seq{i}": seq.upper() for i, seq in enumerate(sequences)}
    hits = []
    for frame in frames:
        for row in frame.to_dict(orient="records"):
            name = str(row["sequence_name"])
            start, end = int(row["start"]), int(row["end"])
            if name not in by_name or end > len(by_name[name]):
                continue
            site = by_name[name][start:end]
            strand = str(row["strand"])
            hits.append({
                "motif": str(row["motif_name"]).split()[0],
                "sequence": name,
                "start": start + 1,
                "end": end,
                "strand": strand,
                "score": float(row["score"]),
                "p_value": float(row["p-value"]),
                "matched_sequence": _reverse_complement(site) if strand == "-" else site,
            })
    hits.sort(key=lambda h: (h["p_value"], h["sequence"], h["start"]))
    fimo_out.mkdir(parents=True, exist_ok=True)
    with open(fimo_out / "fimo.tsv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIMO_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(hits)
    return hits


@mcp.tool(
    name="tfbs_fimo",
    annotations={
        "title": "FIMO Motif Scan",
        "readOnlyHint": False,
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

    Writes the sequences to a FASTA file and scans them with FIMO: the
    MEME Suite's ``fimo`` binary when it is on PATH, otherwise
    tangermeme's FIMO reimplementation (memelite, numba-based, shipped
    with tangermeme 1.x; tangermeme.tools.fimo in older tangermeme), so
    the tool works without the MEME Suite, e.g. on Windows.  Both use
    the same p-value threshold and scan every window of both strands, up
    to and including the one that ends on the last base; the binary
    keeps only the better strand at each position (--max-strand), the
    reimplementation reports a hit on each strand.  The reimplementation
    reads a copy of motif_file, motifs.meme in the output directory,
    ending in a blank line, because memelite drops a last motif that
    ends the file.  Without either, the tool returns an error with the
    install hint.

    Args:
        sequences: DNA sequences to scan (A/C/G/T).
        motif_file: Absolute path to a MEME-format motif file.
        threshold: FIMO p-value threshold (default 1e-4).
        output_dir: Directory for the FASTA and the hit table.  Default
                    <results>/fimo/<digest>/, with <results> the
                    results_dir of tfbs_server_info and <digest> a hash
                    of the sequences, motif file and threshold.

    Returns:
        JSON with "hits" (up to 500, lowest p-value first; each with
        motif (the motif ID), sequence (seq0, seq1, ... in input order),
        start and end (1-based, inclusive, as FIMO reports them), strand,
        score, p_value and matched_sequence (read on the hit's strand)),
        "total_hits", "backend" ("meme" for the binary, "memelite" when
        memelite.fimo ran, "tangermeme" when tangermeme.tools.fimo ran),
        "backend_module" (the module, for the last two), "output_dir"
        and "files" (sequences_fasta: the sequences scanned; fimo_tsv:
        all hits, MEME's own fimo.tsv for the binary and a TSV with the
        columns above for the reimplementation).
    """
    motif_path, err = _resolve_reference(motif_file, "motif_file")
    if err:
        return err
    # Basic path validation: motif_file is passed to subprocess
    if not motif_path.is_file():
        return _error(f"Not a regular file: {motif_file}",
                      "Pass the path of a MEME-format motif file.")

    if not sequences:
        return _no_sequences_error("tfbs_fimo", "Pass the DNA sequences to scan as sequences.")
    err = _validate_dna(sequences)
    if err:
        return err

    python_fimo, module = (None, None)
    if shutil.which("fimo") is None:
        python_fimo, module = _python_fimo()
        if python_fimo is None:
            return json.dumps(_FIMO_MISSING)

    work_dir = (Path(output_dir).expanduser().resolve() if output_dir
                else _results_root() / "fimo" / _digest(
                    [s.upper() for s in sequences], _file_identity(str(motif_path)),
                    float(threshold)))
    mkdir_err = _mkdir_error(work_dir, "output_dir")
    if mkdir_err:
        return mkdir_err
    fasta_path = work_dir / "input.fasta"
    with open(fasta_path, "w", newline="\n") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq{i}\n{seq}\n")
    fimo_out = work_dir / "fimo_out"

    hits = None
    if python_fimo is None:
        hits, err = _meme_fimo_hits(motif_path, fasta_path, fimo_out, threshold)
        if err:
            return err
        backend = "meme"
        if hits is None:
            python_fimo, module = _python_fimo()
            if python_fimo is None:
                return json.dumps(_FIMO_MISSING)
    if hits is None:
        try:
            hits = _python_fimo_hits(python_fimo, motif_path, fasta_path, fimo_out,
                                     threshold, sequences)
        except (ValueError, KeyError, IndexError, OSError) as exc:
            return _error(f"The FIMO reimplementation ({module}) failed: "
                          f"{type(exc).__name__}: {exc}",
                          "Check that motif_file is a MEME-format motif file.")
        backend = _PYTHON_FIMO_BACKENDS.get(module, module)

    hits.sort(key=lambda h: h["p_value"])
    result: dict[str, Any] = {
        "hits": hits[:500],  # cap to avoid huge responses
        "total_hits": len(hits),
        "backend": backend,
        "output_dir": str(fimo_out),
        "files": {"sequences_fasta": str(fasta_path), "fimo_tsv": str(fimo_out / "fimo.tsv")},
    }
    if backend != "meme":
        result["backend_module"] = module
    return json.dumps(result)


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
    """Show the server's runtime environment, defaults and registered reference files.

    Reports whether the server is running inside a SLURM job, how many
    GPUs are visible, the default device and batch size, the default SLURM
    partition, the reference files registered for the server (genome,
    motif, ClinVar VCF and FoldX structure) and whether each exists, and
    the torch version and CUDA status.  The server imports
    PyTorch and the other native libraries at startup, so every call,
    this one included, answers at once.

    To register or change the reference files and the project folder, run
    `tfbs-mcp --setup --non-interactive` in a shell with the flags
    --genome, --motif, --clinvar, --foldx-pdb and --project-root.  It
    asks nothing, checks each path and prints the `claude mcp add` command
    for Claude Code and the env block for Claude Desktop.  Each run
    replaces the whole registration, so pass the paths already registered
    too (`claude mcp get tfbs` shows them), and keep any other variable
    such as TFBS_SLURM_PARTITION.  For Claude Code, run
    `claude mcp remove tfbs --scope user`, then the printed command; the
    server reads the paths when it starts, so restart the client.

    Returns:
        JSON with on_slurm, slurm_job_id, slurm_partition (partition of the
        job the server itself runs in, null outside a job),
        default_partition ($TFBS_SLURM_PARTITION, else "gpu"; used by the
        SLURM tools unless a call passes partition=...), gpu_count,
        default_device, default_batch_size, call_log (the $TFBS_CALL_LOG
        file, or null; when set, every tool call appends one JSON line
        with its timestamp, tool, arguments, seconds, is_error and result),
        results_dir (where the model-analysis tools write
        when called without output_dir: <project root>/results for a
        checkout or $TFBS_PROJECT_ROOT, else tfbs_results/ in the server's
        working directory), project_root (working directory of the
        validation and benchmark pipelines, against which their relative
        models_dir resolves: $TFBS_PROJECT_ROOT, else the checkout, else the
        server's working directory; it shows that fallback when
        TFBS_PROJECT_ROOT is unset), reference_files (one entry per
        reference variable, TFBS_GENOME_FASTA, TFBS_MOTIF_FILE,
        TFBS_CLINVAR_VCF and TFBS_FOLDX_PDB, each {"path": the registered
        path, expanded and made absolute as the tools resolve it, or null,
        "exists": whether that path exists, null when no path is set}; the
        tools that read a variable fall back to it when a call passes no
        path),
        loaded_models, torch_version,
        cuda_version (CUDA version of the PyTorch build, null for CPU
        builds) and cuda_available, plus cuda_device_name and
        cuda_device_count when CUDA is available.
    """
    try:
        _ensure_imports()
    except ImportError as exc:
        return _error(f"PyTorch cannot be imported: {exc}",
                      "Install PyTorch in the server's environment (see pytorch.org), "
                      "then restart the MCP client.")
    info = dict(_ENV)  # shallow copy
    info["call_log"] = os.environ.get(CALL_LOG_ENV, "").strip() or None
    info["results_dir"] = str(_results_root())
    info["project_root"] = str(_project_root())
    info["reference_files"] = _reference_files()
    info["torch_version"] = _torch.__version__
    info["cuda_version"] = _torch.version.cuda
    info["cuda_available"] = _torch.cuda.is_available()
    if info["cuda_available"]:
        info["cuda_device_name"] = _torch.cuda.get_device_name(0)
        info["cuda_device_count"] = _torch.cuda.device_count()

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
        result = subprocess.run(cmd, capture_output=True, text=True,
                                stdin=subprocess.DEVNULL, timeout=30, cwd=cwd)
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
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30,
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
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30,
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

    Finds each log file in this order, taking the first that exists:
    the StdOut/StdErr path sacct reports (with %j, %x and %% expanded
    and a relative path joined to the job's WorkDir); the StdOut=/StdErr=
    path of ``scontrol show job`` (which knows recent jobs only); then
    <WorkDir>/<JobName>_<job_id>.out (.err for stderr), the names
    tfbs_train's scripts use.  Then it reads the last ``tail`` lines of
    each requested file.

    Args:
        job_id: The SLURM job ID whose logs to read.
        log_type: Which log to read: "stdout", "stderr", or "both".
        tail: Number of lines to read from the end of each log file.

    Returns:
        JSON with job_id, log_paths ({stdout, stderr}: the file found, or
        the best candidate when none exists, or null), log_sources (where
        each path came from: "sacct", "scontrol" or "default name"), and
        the content of the requested logs.  A file that is not found is
        reported in place of its content; neither sacct nor scontrol on
        PATH returns an error.
    """
    if log_type not in ("stdout", "stderr", "both"):
        return _error(f"Unknown log_type {log_type!r}.",
                      'Pass log_type "stdout", "stderr" or "both".')
    sacct, sacct_ok = _sacct_job_fields(job_id)
    scontrol: dict[str, str] | None = None
    scontrol_ok = True

    log_paths: dict[str, str | None] = {"stdout": None, "stderr": None}
    log_sources: dict[str, str | None] = {"stdout": None, "stderr": None}
    for stream, key, suffix in (("stdout", "StdOut", ".out"), ("stderr", "StdErr", ".err")):
        if log_type not in (stream, "both"):
            continue
        candidates: list[tuple[str, Path]] = []
        path = _slurm_log_path(sacct.get(key), sacct.get("WorkDir"), job_id, sacct.get("JobName"))
        if path is not None:
            candidates.append(("sacct", path))
        if not any(c.is_file() for _, c in candidates):
            if scontrol is None:
                scontrol, scontrol_ok = _scontrol_job_fields(job_id)
            work_dir = scontrol.get("WorkDir") or sacct.get("WorkDir")
            job_name = scontrol.get("JobName") or sacct.get("JobName")
            path = _slurm_log_path(scontrol.get(key), work_dir, job_id, job_name)
            if path is not None:
                candidates.append(("scontrol", path))
            if work_dir and job_name:
                candidates.append(("default name", Path(work_dir) / f"{job_name}_{job_id}{suffix}"))
        found = next(((src, c) for src, c in candidates if c.is_file()), None)
        source, path = found or (candidates[0] if candidates else (None, None))
        log_paths[stream] = str(path) if path is not None else None
        log_sources[stream] = source

    if not sacct_ok and not scontrol_ok:
        return _error(
            "Neither sacct nor scontrol is available (not on PATH, or timed out).",
            "Run slurm_logs on a cluster login node where sacct or scontrol is on "
            "PATH, or read the job's --output/--error files directly.",
        )

    result: dict[str, Any] = {"job_id": job_id, "log_paths": log_paths,
                              "log_sources": log_sources}
    for stream in ("stdout", "stderr"):
        if log_type not in (stream, "both"):
            continue
        if log_paths[stream]:
            result[stream] = _tail_file(log_paths[stream], tail)
        else:
            result[stream] = f"[{stream} path not found in sacct or scontrol output]"
    return json.dumps(result)


def _expand_slurm_pattern(pattern: str, job_id: str, job_name: str | None) -> str:
    """%j (job ID), %x (job name) and %% in a SLURM --output/--error pattern."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern[i] == "%" and i + 1 < len(pattern):
            code = pattern[i + 1]
            if code == "j":
                out.append(job_id)
                i += 2
                continue
            if code == "x" and job_name:
                out.append(job_name)
                i += 2
                continue
            if code == "%":
                out.append("%")
                i += 2
                continue
        out.append(pattern[i])
        i += 1
    return "".join(out)


def _slurm_log_path(raw: str | None, work_dir: str | None, job_id: str,
                    job_name: str | None) -> Path | None:
    """A reported StdOut/StdErr as a path: patterns expanded, relative paths under WorkDir."""
    raw = (raw or "").strip()
    if not raw or raw.lower() in ("(null)", "none", "/dev/null"):
        return None
    path = Path(_expand_slurm_pattern(raw, job_id, job_name)).expanduser()
    if not path.is_absolute():
        if not work_dir:
            # Relative to a WorkDir nobody reported: resolving it against the
            # server's own working directory could read an unrelated file.
            return None
        path = Path(work_dir) / path
    return path


def _sacct_job_fields(job_id: str) -> tuple[dict[str, str], bool]:
    """({JobName, WorkDir, StdOut, StdErr}, sacct ran) for the job's main line.

    sacct releases without the StdOut and StdErr fields reject them, so a
    second query asks for JobName and WorkDir only.
    """
    ran = False
    for fields in (("JobID", "JobName", "WorkDir", "StdOut", "StdErr"),
                   ("JobID", "JobName", "WorkDir")):
        try:
            sa = subprocess.run(
                ["sacct", "-j", job_id, "--noheader", "--parsable2",
                 f"--format={','.join(fields)}"],
                capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}, ran
        ran = True
        for raw_line in (sa.stdout or "").strip().splitlines():
            parts = raw_line.split("|")
            # The main job line, not its .batch / .extern steps.
            if parts and parts[0].strip() == job_id:
                return {f: parts[n].strip() for n, f in enumerate(fields)
                        if n < len(parts) and n > 0 and parts[n].strip()}, ran
    return {}, ran


def _scontrol_job_fields(job_id: str) -> tuple[dict[str, str], bool]:
    """({JobName, WorkDir, StdOut, StdErr}, scontrol ran) from ``scontrol show job``."""
    try:
        sc = subprocess.run(["scontrol", "show", "job", job_id], capture_output=True,
                            text=True, stdin=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return {}, False
    fields: dict[str, str] = {}
    text = sc.stdout or ""
    name = re.search(r"\bJobName=(\S+)", text)
    if name:
        fields["JobName"] = name.group(1)
    for line in text.splitlines():
        line = line.strip()
        for key in ("WorkDir", "StdOut", "StdErr"):
            if line.startswith(key + "="):
                fields[key] = line[len(key) + 1:].strip()
    return fields, True


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
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30,
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
        "title": "Train Model",
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
    local: bool = False,
    local_timeout: int = 3600,
    cpus_per_task: int | None = None,
    mem: str | None = None,
) -> str:
    """Train a TFBS model: submit a SLURM job, or train on this machine.

    Three modes of operation:

    **Script mode**: provide ``script`` to submit an existing sbatch
    script directly.

    **Config mode**: provide ``config_path`` to auto-generate an
    sbatch script that invokes the Lightning CLI trainer with the
    given YAML config.

    **Local mode**: ``config_path`` with ``local=true`` runs the same
    trainer here, ``python -m tfbs.cli.train_cli fit --config <config>
    --data.num_workers 0``, from the config's directory.  The call
    blocks until training finishes, so it is meant for small datasets
    and laptops (e.g. a few thousand 24 bp windows for a few epochs);
    larger runs belong on SLURM, which stays the default and is
    unchanged.  Its output goes to <config dir>/<config stem>_train.log.

    Args:
        script: Path to an existing sbatch script (script mode).
        config_path: Path to a Lightning CLI YAML config (config and
                     local mode), e.g. the output_path of tfbs_config.
        partition: SLURM partition (config mode).  Defaults to the server's
                   default_partition: $TFBS_SLURM_PARTITION, else "gpu"
                   (see tfbs_server_info), because the job requests a GPU and
                   "gpu" is the usual name of a cluster's GPU partition.
        gpu_type: GPU type constraint (e.g. "a100", "v100").
        max_time: Maximum wall-clock time (default "48:00:00").
        job_name: SLURM job name.  Auto-generated if omitted.
        extra_args: Additional CLI arguments appended to the training
                    command (config and local mode).  Each element is one
                    or more argv tokens: an element with whitespace or
                    quotes is split like a shell would split it (e.g.
                    "--trainer.max_epochs 3"; quotes are removed once, as
                    a POSIX shell removes them, on every platform, and on
                    Windows backslashes are not escapes), any other
                    element is one token as given, so a Windows path keeps its
                    backslashes.  The same tokenization applies in SLURM
                    config mode.  An element with unbalanced quotes
                    returns an error naming it.
        local: Train on this machine instead of submitting to SLURM
               (needs config_path; default false).
        local_timeout: Seconds local mode may run (default 3600, at
                       least 1); the trainer runs in a process group of
                       its own, and that whole group (the process tree,
                       on Windows) is killed after that, and also when
                       the MCP call is cancelled.
        cpus_per_task: CPUs the SLURM job requests (config mode,
                       #SBATCH --cpus-per-task).  Default: the config's
                       data.num_workers (the DataLoader worker processes;
                       an extra_args --data.num_workers overrides it)
                       plus 1 for the main training process, so 24
                       workers request 25 CPUs.
        mem: Memory the SLURM job requests (config mode, #SBATCH --mem),
             a number of at least 1 with an optional K, M, G or T
             suffix, e.g. "64G" ("0" is refused: SLURM reads --mem=0 as
             all of the node's memory).
             Default null: no --mem line, so the cluster's default
             memory per CPU or node applies.

    Returns:
        JSON with mode, job_id, submit_command, and (for config mode)
        the generated sbatch script content as generated_script, plus
        config_path, checkpoint_path (where the job's ModelCheckpoint
        writes, <dirpath>/<filename>.ckpt from the config, relative
        paths against the config's directory, which is the job's working
        directory; null when the config names none), cpus_per_task and
        mem as requested.  Local mode returns mode
        "local", command (the argv list), exit_code, log_path, log_tail
        (last 20 lines), checkpoint_path and checkpoint_exists (from the
        config's ModelCheckpoint dirpath and filename; true only for a
        file this run created or rewrote, i.e. one that did not exist
        before the run or whose modification time changed during it),
        and "metrics" from the metrics.csv this run's CSVLogger wrote
        (<save_dir>/<name>/version_<n>/ of the config's trainer.logger;
        likewise a file that is new or was modified during the run): the
        row of the best epoch, the one with the lowest finite val_loss
        that ModelCheckpoint keeps ({val_loss, and for classifiers
        val_acc and val_auc, as floats}), plus best_epoch and epochs_run
        (last epoch + 1), with the file as metrics_path; a row whose
        epoch or val_loss is nan or inf is skipped.  A CSVLogger with
        a fixed version reuses its metrics.csv, and some Lightning
        releases append to it, so only the rows this run added are read:
        rows the file held when the run started are skipped when the
        rows after them start again at an earlier or equal (epoch, step),
        as an appended run does.  The file is read whole when its new
        rows continue past the old ones (Lightning 2.6 deletes the file,
        so a rerun logs the same epochs again and then goes on), when it
        holds no more rows than before, or when its first rows differ
        from the old ones.  metrics is null, with a "note" saying why,
        when the config has no CSVLogger, this run wrote no metrics.csv,
        or the file cannot be read;
        val_auc is null, with a note, when the validation split holds a
        single class (AUROC is undefined there).  For a classifier
        (model.init_args.classify true, or val_acc or val_auc logged) a
        "warning" says the model did not learn (constant output) when
        val_loss lies within 0.005 of ln 2 at every logged epoch, or
        when the best epoch is the first, a later epoch was logged and
        the best val_loss lies within 0.005 of ln 2, and, when val_auc
        is logged, the best epoch's val_auc also lies within 0.05 of 0.5
        (a run whose loss is still near ln 2 in its first epochs while
        its AUC already rises is learning); it suggests another seed,
        more data, or a shallower stack through tfbs_config's
        model_params (for VCNNBpnet, the template's dilations
        [1, 1, 2]).  A failed run returns
        error, fix, exit_code, log_path and log_tail; a run killed after
        local_timeout returns error, fix and log_path.  In local mode a
        config whose trainer, or the init_args of its ModelCheckpoint or
        CSVLogger, is not a mapping returns error and fix before anything
        runs.
    """
    if script and config_path:
        return _error("Provide exactly one of 'script' or 'config_path', not both.",
                      "Pass script for an existing sbatch script, or config_path "
                      "for a Lightning CLI config.")
    if not script and not config_path:
        return _error("Provide exactly one of 'script' or 'config_path'.",
                      "Pass config_path (e.g. the output_path of tfbs_config), or "
                      "script for an existing sbatch script.")
    if local and script:
        return _error("local=true trains from a config, not from an sbatch script.",
                      "Pass config_path with local=true, or submit the script "
                      "with local=false.")
    tokens: list[str] = []
    for arg in extra_args or []:
        split, err = _argv_tokens(arg)
        if err:
            return err
        tokens += split

    if script:
        script_p = Path(script).resolve()
        if not script_p.is_file():
            return _error(f"Script not found: {script}",
                          "Pass the path of an existing sbatch script.")
        if shutil.which("sbatch") is None:
            return _no_sbatch_for_training()
        result = await slurm_submit(script_path=str(script_p))
        result_dict = json.loads(result)
        result_dict["mode"] = "script"
        return json.dumps(result_dict)

    config_p = Path(config_path).expanduser().resolve()
    if not config_p.is_file():
        return _error(f"Config not found: {config_path}",
                      "Pass the output_path that tfbs_config reported.")

    if local:
        if local_timeout < 1:
            return _error(f"local_timeout must be at least 1 second, got {local_timeout}.",
                          "Pass the seconds local training may run, e.g. local_timeout=3600.")
        return await _train_locally(config_p, tokens, local_timeout)
    if shutil.which("sbatch") is None:
        return _no_sbatch_for_training()
    if cpus_per_task is not None and cpus_per_task < 1:
        return _error(f"cpus_per_task must be at least 1, got {cpus_per_task}.",
                      "Pass the number of CPUs the job needs, or omit it to use the "
                      "config's data.num_workers.")
    mem_match = re.fullmatch(r"(\d+)[KMGTkmgt]?", mem.strip()) if mem is not None else None
    if mem is not None and (mem_match is None or int(mem_match.group(1)) < 1):
        return _error(f"Cannot read mem {mem!r}." if mem_match is None
                      else f"mem must be at least 1, got {mem!r}.",
                      'Pass a number of at least 1 with an optional K, M, G or T suffix, '
                      'e.g. "64G", or omit it to leave the memory to the cluster default '
                      "(--mem=0 would ask SLURM for all of the node's memory).")

    import yaml

    try:
        with open(config_p, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        return _error(f"Could not read the config {config_p}: {exc}",
                      "Pass a Lightning CLI YAML config, e.g. from tfbs_config.")
    config = config if isinstance(config, dict) else {}
    try:
        target = _checkpoint_target(config, config_p.parent)
    except (ValueError, OSError) as exc:
        return _error(f"The config {config_p} is not a usable Lightning CLI config: {exc}.",
                      "Write trainer, and the init_args of its ModelCheckpoint, as YAML "
                      "mappings (key: value), as tfbs_config does.")
    if cpus_per_task is None:
        # The DataLoader workers plus the main training process.
        cpus_per_task = max(0, _config_num_workers(config, tokens) or 0) + 1
    mem = mem.strip().upper() if mem is not None else None

    if job_name is None:
        job_name = f"tfbs_train_{config_p.stem}"
    partition = _resolve_partition(partition)

    gres_line = ""
    if gpu_type:
        gres_line = f"#SBATCH --gres=gpu:{gpu_type}:1"
    else:
        gres_line = "#SBATCH --gres=gpu:1"

    train_cmd = shlex.join([
        sys.executable, "-m", "tfbs.cli.train_cli", "fit", "--config", str(config_p),
        *tokens,
    ])
    mem_line = f"#SBATCH --mem={mem}\n" if mem else ""

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{gres_line}
#SBATCH --time={max_time}
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
{mem_line}
{_sbatch_pythonpath_line()}{train_cmd}
"""

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="tfbs_", newline="\n",
    )
    tmp.write(sbatch_content)
    tmp.close()
    os.chmod(tmp.name, 0o755)

    result = await slurm_submit(script_path=tmp.name, working_dir=str(config_p.parent))
    result_dict = json.loads(result)
    result_dict["mode"] = "config"
    result_dict["generated_script"] = sbatch_content
    result_dict["config_path"] = str(config_p)
    result_dict["checkpoint_path"] = str(target) if target else None
    result_dict["cpus_per_task"] = cpus_per_task
    result_dict["mem"] = mem
    return json.dumps(result_dict)


def _config_num_workers(config: dict, tokens: list[str]) -> int | None:
    """data.num_workers of a Lightning CLI config, as the training run will see it.

    An extra_args --data.num_workers N (or =N) overrides the config's value.
    None when neither gives an integer.
    """
    data = config.get("data")
    value: Any = data.get("num_workers") if isinstance(data, dict) else None
    for i, token in enumerate(tokens):
        if token == "--data.num_workers" and i + 1 < len(tokens):
            value = tokens[i + 1]
        elif token.startswith("--data.num_workers="):
            value = token.split("=", 1)[1]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _argv_tokens(arg: str, posix: bool | None = None) -> tuple[list[str], str | None]:
    """(argv tokens, None) for one extra_args element, or ([], error JSON).

    Only an element with whitespace or quotes is split: posix shlex would
    also drop the backslashes of a Windows path given as a single token.
    posix defaults to os.name != "nt".  posix=False (Windows) splits with
    shell quoting but without backslash escapes, because shlex's own
    non-posix mode leaves a quote inside a token (--x='a b') unsplit;
    quotes are removed once on both platforms.
    """
    if not any(c.isspace() or c in "'\"" for c in arg):
        return [arg], None
    posix = os.name != "nt" if posix is None else posix
    try:
        if posix:
            tokens = shlex.split(arg, posix=True)
        else:
            lexer = shlex.shlex(arg, posix=True)
            lexer.whitespace_split = True
            lexer.escape = ""
            lexer.commenters = ""
            tokens = list(lexer)
    except ValueError as exc:
        return [], _error(
            f"extra_args element {arg!r} cannot be split into arguments: {exc}.",
            "Balance the quotes in that element, or pass each argument as its own "
            "element, e.g. ['--trainer.max_epochs', '3'].")
    return tokens, None


def _no_sbatch_for_training() -> str:
    return _error(
        "sbatch binary not found on PATH",
        "Run tfbs_train on a cluster login node where sbatch is on the PATH of "
        "the MCP server, or train on this machine with config_path and "
        "local=true (blocks until training ends; meant for small datasets).",
    )


def _mapping(parent: dict, key: str, where: str) -> dict:
    """parent[key] as a dict ({} when absent); ValueError naming where if it is not a mapping."""
    value = parent.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{where} is a {type(value).__name__} ({value!r:.80}), not a mapping")
    return value


def _checkpoint_target(config: dict, base: Path) -> Path | None:
    """<dirpath>/<filename>.ckpt of the config's ModelCheckpoint callback.

    Raises ValueError when trainer or the callback's init_args is not a mapping.
    """
    callbacks = _mapping(config, "trainer", "trainer").get("callbacks") or []
    for cb in callbacks if isinstance(callbacks, list) else [callbacks]:
        if not isinstance(cb, dict) or not str(cb.get("class_path", "")).endswith("ModelCheckpoint"):
            continue
        args = _mapping(cb, "init_args", f"init_args of {cb['class_path']}")
        if not args.get("dirpath"):
            return None
        dirpath = Path(str(args["dirpath"])).expanduser()
        if not dirpath.is_absolute():
            dirpath = base / dirpath
        name = str(args.get("filename") or "best")
        return dirpath.resolve() / f"{name}.ckpt"
    return None


def _csv_logger_dir(config: dict, base: Path) -> Path | None:
    """<save_dir>/<name> of the config's CSVLogger (relative save_dir against base), or None.

    Raises ValueError when trainer or the logger's init_args is not a mapping.
    """
    loggers = _mapping(config, "trainer", "trainer").get("logger")
    for lg in loggers if isinstance(loggers, list) else [loggers]:
        if not isinstance(lg, dict) or not str(lg.get("class_path", "")).endswith("CSVLogger"):
            continue
        args = _mapping(lg, "init_args", f"init_args of {lg['class_path']}")
        if not args.get("save_dir"):
            return None
        save_dir = Path(str(args["save_dir"])).expanduser()
        if not save_dir.is_absolute():
            save_dir = base / save_dir
        name = args.get("name", "lightning_logs")
        return (save_dir / str(name) if name else save_dir).resolve()
    return None


def _as_float(value: Any) -> float | None:
    """value as a finite float, or None (blank, not a number, nan or inf)."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _mtime_ns(path: Path | None) -> int | None:
    """path's st_mtime_ns, or None when it does not exist (or cannot be read)."""
    if path is None:
        return None
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _read_metrics_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _metrics_snapshot(log_dir: Path | None) -> dict[Path, tuple[int, list[dict[str, str]] | None]]:
    """{metrics.csv: (st_mtime_ns, rows)} under log_dir before a run starts.

    rows is None for a file that cannot be read.
    """
    import csv

    snapshot: dict[Path, tuple[int, list[dict[str, str]] | None]] = {}
    if log_dir is None:
        return snapshot
    try:
        paths = list(log_dir.glob("*/metrics.csv")) if log_dir.is_dir() else []
    except OSError:
        return snapshot
    for path in paths:
        stamp = _mtime_ns(path)
        if stamp is None:
            continue
        try:
            rows = _read_metrics_rows(path)
        except (OSError, UnicodeDecodeError, csv.Error):
            rows = None
        snapshot[path] = (stamp, rows)
    return snapshot


def _rows_of_this_run(rows: list[dict[str, str]], before: list[dict[str, str]] | None
                      ) -> list[dict[str, str]]:
    """rows without the leading rows the file already held when the run started.

    A CSVLogger with a fixed version reuses its metrics.csv: Lightning
    releases that append to it (and rewrite it with a wider header when
    new metrics appear) keep the earlier run's rows first, and the
    appended rows restart at a lower (epoch, step).  Lightning 2.6
    deletes the file instead, so a rerun that logs the same rows again
    continues past them, or matches them exactly, and is read whole.
    """
    if not before or len(rows) <= len(before):
        return rows
    for row, old in zip(rows, before):
        if any(row.get(k, "") != v for k, v in old.items() if k is not None):
            return rows

    def position(row: dict[str, str]) -> tuple[float, float]:
        return (_as_float(row.get("epoch")) or 0.0, _as_float(row.get("step")) or 0.0)

    if position(rows[len(before)]) > position(rows[len(before) - 1]):
        return rows
    return rows[len(before):]


def _best_epoch_metrics(log_dir: Path,
                        before: dict[Path, tuple[int, list[dict[str, str]] | None]],
                        ) -> tuple[dict | None, Path | None, str | None,
                                   list[tuple[int, float]]]:
    """(metrics, metrics.csv, None, val_losses) for the best epoch of the run.

    On failure (None, path or None, note, []).  Only a metrics.csv under
    log_dir that is new or whose st_mtime_ns changed since before (from
    _metrics_snapshot) counts, and only the rows it gained during the
    run, so an earlier run's log is never reported.  A row whose epoch
    or val_loss is blank, nan or inf is not a candidate for the best
    epoch.  val_losses holds (epoch, val_loss) for every row of the run
    with a finite val_loss and epoch, in the order logged.
    """
    import csv

    path = None
    try:
        stamps = {p: _mtime_ns(p) for p in log_dir.glob("*/metrics.csv")} \
            if log_dir.is_dir() else {}
        fresh = {p: ns for p, ns in stamps.items()
                 if ns is not None and (p not in before or before[p][0] != ns)}
        if not fresh:
            return None, None, (f"This run wrote no metrics.csv under {log_dir} (the config's "
                                "CSVLogger directory), so no validation metrics are "
                                "reported."), []
        path = max(fresh, key=fresh.__getitem__)
        rows = _rows_of_this_run(_read_metrics_rows(path), before.get(path, (0, None))[1])
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return None, path, (f"Could not read {path or log_dir} after the run "
                            f"({type(exc).__name__}: {exc}), so no validation metrics "
                            "are reported."), []
    numbered = [(r, _as_float(r.get("epoch"))) for r in rows]
    epochs = [int(e) for _, e in numbered if e is not None]
    scored = [(r, e) for r, e in numbered
              if e is not None and _as_float(r.get("val_loss")) is not None]
    if not scored:
        return None, path, (f"{path} has no row with a finite val_loss and epoch, so no "
                            "validation metrics are reported."), []
    best, best_epoch = min(scored, key=lambda pair: _as_float(pair[0]["val_loss"]))
    metrics: dict[str, Any] = {}
    for key in ("val_loss", "val_acc", "val_auc"):
        value = _as_float(best.get(key))
        if value is not None:
            metrics[key] = value
    metrics["best_epoch"] = int(best_epoch)
    metrics["epochs_run"] = max(epochs) + 1
    return metrics, path, None, [(int(e), _as_float(r["val_loss"])) for r, e in scored]


# A classifier whose validation BCE stays within this distance of ln 2 (the
# loss of a constant output of 0.5 on balanced classes) has not learned.
_DEAD_RUN_TOLERANCE = 0.005
# A run whose best-epoch val_auc lies this far or further from 0.5 is
# learning even when its val_loss is still near ln 2: the default
# three-layer VCNNBpnet stays within 0.005 of ln 2 for its first 1 to 3
# epochs while its AUC already reaches 0.6 to 0.8.
_DEAD_RUN_AUC_TOLERANCE = 0.05


def _dead_classifier_warning(metrics: dict, val_losses: list[tuple[int, float]],
                             classify: bool) -> str | None:
    """A warning when a classifier run shows a constant output, else None.

    Two patterns count: the val_loss of every logged epoch lies within
    _DEAD_RUN_TOLERANCE of ln 2, or the best epoch is the run's first
    epoch, at least one later epoch was logged without improving on it,
    and its val_loss lies within _DEAD_RUN_TOLERANCE of ln 2.  When
    metrics holds a finite val_auc (the best epoch's), the run also
    needs that val_auc within _DEAD_RUN_AUC_TOLERANCE of 0.5; a val_auc
    that is None (a single-class validation split) counts as not
    logged.  metrics and val_losses are as _best_epoch_metrics returns
    them.
    """
    if not classify or not val_losses:
        return None
    ln2 = math.log(2.0)
    epochs = [e for e, _ in val_losses]
    if all(abs(v - ln2) <= _DEAD_RUN_TOLERANCE for _, v in val_losses):
        pattern = "at every logged epoch"
    elif (metrics.get("best_epoch") == min(epochs) and max(epochs) > min(epochs)
          and abs(metrics.get("val_loss", math.inf) - ln2) <= _DEAD_RUN_TOLERANCE):
        pattern = "at its best epoch, the first, and no later epoch improved on it"
    else:
        return None
    auc = metrics.get("val_auc")
    auc_note = ""
    if isinstance(auc, (int, float)) and math.isfinite(auc):
        if abs(auc - 0.5) >= _DEAD_RUN_AUC_TOLERANCE:
            return None
        auc_note = f", and val_auc at the best epoch was {auc:.3f}, near chance (0.5)"
    return (f"The model did not learn: val_loss stayed within {_DEAD_RUN_TOLERANCE} of "
            f"ln 2 ({ln2:.4f}), the loss of a constant output of 0.5, {pattern}"
            f"{auc_note}. "
            "Train again with another seed (extra_args ['--seed_everything 7'] or "
            "seed_everything in the config), with more data, or with a shallower stack "
            "through tfbs_config's model_params (for VCNNBpnet with the study's five "
            "layers [1, 1, 2, 4, 8], the template's three-layer default "
            "{\"dilations\": [1, 1, 2]}).")


def _config_classify(config: dict) -> bool:
    """Whether the config's model.init_args.classify is true."""
    model = config.get("model")
    args = model.get("init_args") if isinstance(model, dict) else None
    return isinstance(args, dict) and args.get("classify") is True


def _single_class_validation(config: dict, base: Path) -> bool:
    """Whether val/labels.pt of the config's dataset holds one class only."""
    raw = (config.get("data") or {}).get("data_path")
    if not isinstance(raw, str) or not raw.strip():
        return False
    data = Path(raw).expanduser()
    labels_path = (data if data.is_absolute() else base / data) / "val" / "labels.pt"
    try:
        _ensure_imports()
        labels = _torch.load(labels_path, map_location="cpu", weights_only=True)
        return int(_torch.unique(_torch.as_tensor(labels)).numel()) == 1
    except Exception:
        return False


def _new_process_group(posix: bool | None = None) -> dict[str, Any]:
    """Popen arguments that start the child in a process group of its own."""
    posix = os.name != "nt" if posix is None else posix
    if posix:
        return {"start_new_session": True}
    return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)}


def _kill_process_group(proc, posix: bool | None = None) -> None:
    """Kill proc and every process it started, falling back to proc.kill().

    POSIX kills the process group that _new_process_group started;
    Windows has no group kill, so taskkill /T walks the process tree.
    """
    posix = os.name != "nt" if posix is None else posix
    try:
        if posix:
            os.killpg(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            return
        done = subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                              stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
        if done.returncode == 0:
            return
    except Exception:
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _stop_process_group(proc, timeout: float = 30.0) -> None:
    """Kill proc's process group, then wait up to timeout seconds for proc to exit."""
    _kill_process_group(proc)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Process %s had not exited %s s after it was killed.", proc.pid, timeout)


def _wait_polling(proc, timeout: float, poll: float = 0.5) -> int:
    """proc's exit code; kills its process group and raises TimeoutExpired after timeout seconds."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return proc.wait(timeout=max(0.0, min(poll, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline:
                _kill_process_group(proc)
                proc.wait()
                raise


async def _train_locally(config_p: Path, tokens: list[str], timeout: int) -> str:
    import yaml

    try:
        with open(config_p, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        return _error(f"Could not read the config {config_p}: {exc}",
                      "Pass a Lightning CLI YAML config, e.g. from tfbs_config.")

    config = config if isinstance(config, dict) else {}
    try:
        target = _checkpoint_target(config, config_p.parent)
        log_dir = _csv_logger_dir(config, config_p.parent)
    except (ValueError, OSError) as exc:
        return _error(f"The config {config_p} is not a usable Lightning CLI config: {exc}.",
                      "Write trainer, and the init_args of its ModelCheckpoint and "
                      "CSVLogger, as YAML mappings (key: value), as tfbs_config does.")
    cmd = [sys.executable, "-m", "tfbs.cli.train_cli", "fit", "--config", str(config_p),
           "--data.num_workers", "0", *tokens]
    log_path = config_p.parent / f"{config_p.stem}_train.log"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    env = {**_pipeline_env(), "PYTHONIOENCODING": "utf-8"}
    target_before = _mtime_ns(target)
    before = _metrics_snapshot(log_dir)
    try:
        with open(log_path, "a", encoding="utf-8", newline="\n") as log:
            log.write(f"\n# {stamp} {shlex.join(cmd)}\n")
            log.flush()
            proc = subprocess.Popen(
                cmd, cwd=str(config_p.parent), env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                **_new_process_group(),
            )
            try:
                returncode = await asyncio.to_thread(_wait_polling, proc, timeout)
            except asyncio.CancelledError:
                await asyncio.shield(asyncio.to_thread(_stop_process_group, proc))
                raise
    except subprocess.TimeoutExpired:
        return _error(
            f"Local training did not finish within local_timeout={timeout} s; "
            "the process was killed.",
            "Pass a larger local_timeout, train for fewer epochs (max_epochs in "
            "tfbs_config, or extra_args ['--trainer.max_epochs 5']), or submit it "
            "to SLURM with local=false.",
            log_path=str(log_path),
        )
    except OSError as exc:
        return _error(f"Local training could not start: {exc}",
                      "Check that the config directory is writable and the server's "
                      "Python interpreter still exists.", log_path=str(log_path))
    tail = _tail_file(str(log_path), 20)
    if returncode != 0:
        return _error(
            f"Local training failed (exit {returncode}).",
            "Read log_tail for the cause; a data_path that does not exist or an "
            "out-of-memory error (lower batch_size in tfbs_config) are the usual ones.",
            exit_code=returncode, log_path=str(log_path), log_tail=tail,
        )
    notes: list[str] = []
    fresh = bool(target and target.is_file() and _mtime_ns(target) != target_before)
    if target and target.is_file() and not fresh:
        notes.append(f"{target} predates this run, so it is not reported as written by it.")
    result: dict[str, Any] = {
        "mode": "local",
        "command": cmd,
        "exit_code": returncode,
        "log_path": str(log_path),
        "log_tail": tail,
        "checkpoint_path": str(target) if target else None,
        "checkpoint_exists": fresh,
        "metrics": None,
    }
    if log_dir is None:
        notes.append("The config has no CSVLogger with a save_dir (trainer.logger), so no "
                     "validation metrics are reported; tfbs_config adds one.")
    else:
        metrics, metrics_path, note, val_losses = _best_epoch_metrics(log_dir, before)
        if note:
            notes.append(note)
        if metrics is not None and "val_auc" in metrics \
                and _single_class_validation(config, config_p.parent):
            metrics["val_auc"] = None
            notes.append("validation split contains one class, so AUROC is undefined "
                         "and val_auc is null (torchmetrics logs 0.0 for it).")
        if metrics is not None:
            warning = _dead_classifier_warning(
                metrics, val_losses,
                _config_classify(config) or "val_acc" in metrics or "val_auc" in metrics)
            if warning:
                result["warning"] = warning
        result["metrics"] = metrics
        if metrics_path:
            result["metrics_path"] = str(metrics_path)
    if notes:
        result["note"] = " ".join(notes)
    return json.dumps(result)


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

    Needs the tracking extra (pip install "tfbs-mcp[tracking]", which
    installs wandb) and a prior "wandb login".

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

    err = _pipeline_requirements_error("run_sweep", "tfbs_sweep")
    if err:
        return err

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
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60,
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
    scaling_method: str | None = None,
    output_file: str | None = None,
    partition: str | None = None,
    gpu_type: str | None = None,
    max_time: str = "02:00:00",
    job_name: str | None = None,
) -> str:
    """Submit a model validation job to SLURM.

    The job runs ``python -m tfbs.pipelines.validate`` (installed with
    the package), which loads the checkpoint, runs trainer.test() on the
    held-out test split, and appends one row per run to a CSV results
    file: model, scaling and dataset, then every metric the test
    reports, which is test_loss for a regression model and test_loss,
    test_acc (accuracy) and test_auc (ROC-AUC) for a classifier.  The
    job's .o/.e logs are written next to that file.

    Args:
        checkpoint_path: Path to the model checkpoint.  Both formats are
            accepted: a PyTorch Lightning .ckpt, or an inference
            checkpoint such as the NKX2-1 study's released
            <models>/flank/vcnnbpnet.pt (Zenodo,
            doi:10.5281/zenodo.21399578).  An inference checkpoint whose
            arch is not model_name, and a bare state_dict (a file
            saved with torch.save(model.state_dict())), are an error
            before anything is submitted.
        data_path: Path to the dataset (tensor directory or CSV).
        model_name: Model class name (e.g. "VCNNBpnet", "RNN"); one that
            names no model class in tfbs.nn.models is an error before
            anything is submitted.
        scaling_method: Label scaling the test labels are put on before
            trainer.test() scores the model ("standardize", "normalize",
            "none").  It must be the scaling the checkpoint was trained
            with, because the model's outputs are on that scale.  Default:
            read from the checkpoint (an inference checkpoint's scaling
            block; a Lightning checkpoint's recorded data-module scaling,
            or the scaling folder it was saved in).  When the checkpoint
            records none: "none" for a classifier (its classify flag, else
            0/1 labels in data_path) and "standardize" otherwise.  A value
            that contradicts the checkpoint, or label scaling for a
            classifier, is an error before anything is submitted.
        output_file: CSV file to append results to.  Defaults to
                     validation_results.csv in the results directory
                     (results_dir in tfbs_server_info).  Jobs that may
                     run at the same time should each get their own
                     output_file (e.g. validation_<model>_<dataset>.csv)
                     and be combined afterwards: one shared file is
                     appended to by whichever job finishes, so a row can
                     be lost when two finish together.
        partition: SLURM partition.  Defaults to the server's
                   default_partition: $TFBS_SLURM_PARTITION, else "gpu"
                   (see tfbs_server_info), because the job requests a GPU and
                   "gpu" is the usual name of a cluster's GPU partition.
        gpu_type: GPU gres string (e.g. "nvidia_a100_80gb_pcie:1").
                  If omitted, requests 1 generic GPU.
        max_time: SLURM wall time (default "02:00:00").
        job_name: SLURM job name (default "tfbs_validate").

    Returns:
        JSON with job_id, the generated sbatch script, the validation
        command, scaling_method and scaling_source (checkpoint, checkpoint
        classify flag, labels, argument or default), and cautions when
        there are any.
    """
    ckpt = Path(checkpoint_path).resolve()
    if not ckpt.exists():
        return json.dumps({"error": f"Checkpoint not found: {checkpoint_path}"})

    data = Path(data_path).resolve()
    if not data.exists():
        return json.dumps({"error": f"Data path not found: {data_path}"})

    err = _model_type_error(model_name, "model_name") or await asyncio.to_thread(
        _inference_checkpoint_error, {"checkpoint_path": str(ckpt)}, model_name, None,
        "model_name")
    if err:
        return err

    scaling_method, scaling_source, cautions, err = await asyncio.to_thread(
        _validate_scaling, ckpt, data, scaling_method)
    if err:
        return err

    project_root, err = _work_dir()
    if err:
        return err

    csv_path = (Path(output_file) if output_file
                else _results_root() / "validation_results.csv").resolve()
    mkdir_err = _mkdir_error(csv_path.parent, "output_file")
    if mkdir_err:
        return mkdir_err
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
    submit_result["scaling_method"] = scaling_method
    submit_result["scaling_source"] = scaling_source
    if cautions:
        submit_result["cautions"] = cautions
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
    "FlexibleMLP": "input_length is set to 4 x window: the MLP flattens the one-hot "
                   "(batch, 4, window) input and sizes its first layer from input_length.",
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


def _binary_labels(labels: Path) -> bool | None:
    """Whether every value in a labels.pt tensor is 0 or 1; None if it cannot be read."""
    if not labels.is_file():
        return None
    import torch

    try:
        t = torch.load(labels, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    return bool(((t == 0) | (t == 1)).all())


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
    output_path: str | None = None,
    input_length: int | None = None,
    scaling_method: str | None = None,
    batch_size: int = 64,
    max_epochs: int = 500,
    patience: int = 5,
    learning_rate: float = 0.001,
    classify: bool | None = None,
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
                  hyperparameters.  Its template uses a three-layer stack,
                  dilations [1, 1, 2], instead of the study's five layers,
                  because the five-layer stack can stay at a constant
                  output on small classification sets at batch size 64,
                  and model_params={"dilations": [1, 1, 2, 4, 8]} restores
                  the study's five-layer stack.
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
      FlexibleMLP MLP with dropout on the flattened one-hot input;
                  input_length is set to 4 x window.
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
      classify: omitted means true when every label in
        <data_path>/train/labels.pt is 0 or 1 (the result then carries a
        "note" saying so), because such labels come from
        tfbs_prepare_classification, and false otherwise.  An explicit
        classify=false on 0/1 labels is kept and reported as a "warning".
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
      checkpoint_dir: <project root>/saved_models/<data_name>/
        <scaling_method> (a relative checkpoint_dir resolves against the
        project root), with filename best_<model_type>, because
        tfbs_chipseq_benchmark, tfbs_chipseq_method_comparison and
        tfbs_remap_validation look for
        <models_dir>/<dataset>/<scaling_method>/best_<model_type>.ckpt
        (and fall back to the one <dataset>/*/best_<model_type>.ckpt).
        data_name is the dataset directory's name without a trailing
        _dataset, so tfbs_prepare_dataset's peaks_dataset/ trains to
        saved_models/peaks/.  The result's checkpoint_path is the file
        training will write; pass it to tfbs_load_model and
        tfbs_validate.  Retraining into the same folder overwrites it
        (no best_<model_type>-v1.ckpt), so checkpoint_path stays true.
      logger: without wandb_project a CSVLogger writes
        <checkpoint_dir>/logs/<checkpoint_filename>/version_<n>/metrics.csv,
        because every training run should leave its loss curve on disk
        (tfbs_train in local mode reports its last row); with
        wandb_project the WandbLogger replaces it.
      output_path: <name>_config/<model_type>.yaml next to the dataset
        directory (<name> as for data_name), like the other data tools'
        outputs, so configs never land inside the dataset.

    Args:
        model_type: Architecture name (see list above).
        data_path: Dataset directory with train/val/test seqs.pt and
                   labels.pt (as written by tfbs_prepare_dataset).
        output_path: Where to write the YAML config file (default above).
        input_length: Sequence window in bp (default: from the dataset).
        scaling_method: "standardize", "normalize" or "none" (default:
                        standardize for regression, none for classify).
        batch_size: Training batch size (default 64).
        max_epochs: Maximum training epochs (default 500).
        patience: Early stopping patience on val_loss (default 5).
        learning_rate: Adam learning rate (default 0.001).
        classify: If true, sigmoid output and BCE loss; if false, MSE
                  regression; omitted, decided from the labels (above).
        wandb_project: WandB project name.  If omitted, no WandB logger
                       is configured and a CSVLogger is (above).
        checkpoint_dir: Directory for the best checkpoint (default above).
        checkpoint_filename: Checkpoint filename (default
                             "best_<model_type>").
        model_params: Architecture hyperparameters merged over the
                      template; keys must exist in the template (an
                      unknown key returns the allowed keys).

    Returns:
        JSON with "output_path", "checkpoint_path" (the .ckpt training
        will write), "files" (config and checkpoint), "metrics_dir"
        (where the CSVLogger writes, null with wandb_project),
        "model_type", "input_length", "input_length_source" ("argument"
        or "dataset"), "classify", "scaling_method", "model_init_args",
        "config_preview" (first 40 lines), and "notes" (architecture
        caveats), "note" (classify inferred from the labels) and
        "warning" when they apply.
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

    if scaling_method is not None and scaling_method not in _SCALING_METHODS:
        return json.dumps({
            "error": f"Unknown scaling_method '{scaling_method}'.",
            "fix": f"Use one of: {list(_SCALING_METHODS)}.",
        })

    data = Path(data_path).resolve()
    if not data.exists():
        return json.dumps({
            "error": f"Data path not found: {data_path}",
            "fix": "Pass the dataset directory written by tfbs_prepare_dataset.",
        })

    cautions: list[str] = []
    note = None
    labels_file = data / "train" / "labels.pt"
    binary = _binary_labels(labels_file)
    if classify is None:
        classify = bool(binary) and scaling_method in (None, "none")
        if classify:
            note = ("classify was set to true and scaling_method to 'none' because "
                    f"every label in {labels_file} is 0 or 1; pass classify=false "
                    "to train a regression model on them.")
        elif binary:
            cautions.append(
                f"Every label in {labels_file} is 0 or 1, but scaling_method "
                f"'{scaling_method}' asks for regression, so classify stays false; "
                "omit scaling_method to train a classifier.")
    elif classify is False and binary:
        cautions.append(
            f"classify=false, but every label in {labels_file} is 0 or 1: this "
            "trains a regression model on class labels. Omit classify (or pass "
            "classify=true) to train a classifier with BCE loss.")
    if scaling_method is None:
        scaling_method = "none" if classify else "standardize"
    if classify and scaling_method != "none":
        return json.dumps({
            "error": f"scaling_method '{scaling_method}' cannot be combined "
                     "with classify=true: BCE loss needs 0/1 labels.",
            "fix": "Omit scaling_method (classification defaults to 'none') "
                   "or pass scaling_method='none'.",
        })

    seq_shape, shape_err = _dataset_seq_shape(data)
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
            cautions.append(
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
    if data_name.endswith("_dataset") and len(data_name) > len("_dataset"):
        data_name = data_name[: -len("_dataset")]
    if checkpoint_dir:
        ckpt_path = Path(checkpoint_dir).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = _project_root() / ckpt_path
    else:
        ckpt_path = _project_root() / "saved_models" / data_name / scaling_method
    ckpt_path = ckpt_path.resolve()
    ckpt_dir = str(ckpt_path)
    ckpt_filename = checkpoint_filename or f"best_{model_type}"
    checkpoint_path = ckpt_path / f"{ckpt_filename}.ckpt"
    metrics_dir = None if wandb_project else ckpt_path / "logs" / ckpt_filename

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
                "enable_version_counter": False,
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
    else:
        trainer["logger"] = [
            {
                "class_path": "pytorch_lightning.loggers.CSVLogger",
                "init_args": {"save_dir": str(ckpt_path / "logs"), "name": ckpt_filename},
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

    if output_path:
        out = Path(output_path).expanduser().resolve()
    else:
        out = _sibling_dir(data / "train", "config") / f"{model_type}.yaml"
    mkdir_err = _mkdir_error(out.parent, "output_path")
    if mkdir_err:
        return mkdir_err
    with open(out, "w", newline="\n") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    with open(out) as f:
        lines = f.readlines()
    preview = "".join(lines[:40])

    result: dict[str, Any] = {
        "output_path": str(out),
        "checkpoint_path": str(checkpoint_path),
        "files": {"config": str(out), "checkpoint": str(checkpoint_path)},
        "metrics_dir": str(metrics_dir) if metrics_dir else None,
        "model_type": model_type,
        "input_length": input_length,
        "input_length_source": length_source,
        "classify": classify,
        "scaling_method": scaling_method,
        "model_init_args": model_args,
        "config_preview": preview,
    }
    if model_type in _ARCH_NOTES:
        result["notes"] = _ARCH_NOTES[model_type]
    if note:
        result["note"] = note
    if cautions:
        result["warning"] = " ".join(cautions)
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
    MACS2 score in column 5) both work.  Blank lines and lines that start
    with #, track or browser are skipped; a # inside a line is data (such
    as "Catalog#: ..." in column 4 of a ChIP-Atlas BED), so the columns
    after it are kept.

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
      output_dir: <bed_dir>/<bed_name>_loci_<in_window>bp[_gc|_shuffled]/
        (bed_name without .bed or .gz; _gc for negatives="gc_matched",
        _shuffled for "shuffled"), e.g. demo_peaks_loci_51bp_gc, so
        extracting the same BED at another window or with negatives
        never overwrites or mixes files, and the data tools downstream
        derive their own outputs from it (demo_peaks_51bp_gc_classification/,
        demo_peaks_51bp_gc_dataset/); for a BED inside the installed
        package (the bundled demo, tfbs/data/demo/demo_peaks.bed), the
        same name under <results>/ instead, so nothing is written into
        the package; with variants and no BED,
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
        chromosome or window off the end), num_filtered_score (peaks
        whose score_column value is missing or NaN), has_scores and
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
        package_dir = Path(__file__).resolve().parents[1]
        base = _results_root() if bed.is_relative_to(package_dir) else bed.parent
        mode = _NEGATIVES_SUFFIX.get(negatives, "")
        out = base / f"{stem}_loci_{in_window}bp{mode}"
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
        mkdir_err = _mkdir_error(out, "output_dir")
        if mkdir_err:
            return mkdir_err
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
            "num_filtered_score": peaks["num_filtered_score"],
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
        mkdir_err = _mkdir_error(out, "output_dir")
        if mkdir_err:
            return mkdir_err
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
        with open(files["variant_manifest"], "w", encoding="utf-8", newline="\n") as f:
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
    """Windows centred on the BED intervals; an error JSON string on failure.

    Blank lines and lines starting with #, track or browser are skipped
    (tfbs.bedio); a "#" inside a line, such as "Catalog#: ..." in column 4
    of a ChIP-Atlas BED, is data.  gzip, bzip2, xz, single-file zip and
    zstd files are decompressed; one that cannot be gives its own error.
    """
    try:
        bed_df = read_bed(bed)
    except BedReadError as exc:
        return _error(f"Could not read BED file {bed}: {exc.reason}",
                      "Decompress the file yourself (gunzip, bunzip2, xz -d, unzip or "
                      "zstd -d) and pass the plain BED file. If that fails too, the file "
                      "is damaged: download it again.")
    except (_pd.errors.EmptyDataError, _pd.errors.ParserError, OSError) as exc:
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
                      "Remove header lines that do not start with #, track or browser, "
                      "and check columns 2 and 3.")
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
    n_bounds = n_bad = n_nan = 0
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
            n_nan += 1
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
            num_filtered_score=n_nan,
        )
    return {
        "seqs": seqs,
        "windows": windows,
        "scores": kept_scores if scores_raw is not None else None,
        "num_input": num_input,
        "num_filtered_n": n_bad,
        "num_filtered_bounds": n_bounds,
        "num_filtered_score": n_nan,
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
        peaks_dataset/).  tfbs_config drops the _dataset suffix again,
        so the checkpoint is named after the peak set
        (saved_models/peaks/...).

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

    mkdir_err = _mkdir_error(out_dir, "output_dir")
    if mkdir_err:
        return mkdir_err
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
        directory, with <name> that directory's name minus its _loci
        part (peaks_loci_24bp_gc/sequences.pt ->
        peaks_24bp_gc_classification/).

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
    mkdir_err = _mkdir_error(out, "output_dir")
    if mkdir_err:
        return mkdir_err
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
        with <name> that directory's name minus its _loci part
        (peaks_loci_200bp/sequences.pt -> peaks_200bp_regression/).

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
    mkdir_err = _mkdir_error(out, "output_dir")
    if mkdir_err:
        return mkdir_err
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
    seed: int = 42,
) -> str:
    """Call seqlets (high-attribution spans) in DNA sequences.

    Computes DeepLiftShap attributions with tangermeme (dinucleotide-
    shuffled references drawn with seed, so a repeated call returns the
    same seqlets), sums them over the four bases, and calls seqlets with
    tangermeme's recursive seqlet caller.  Optionally annotates each
    seqlet with its best TOMTOM match in a MEME motif file (e.g. JASPAR)
    and counts the matched motifs.  Warnings tangermeme raises during
    the attribution (e.g. "Convergence deltas too high", which the
    NKX2-1 VCNNBpnet models trigger) are returned under "warnings"
    instead of being lost on the server's stderr.

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
        seed: Random seed of the shuffled references (default 42).

    Returns:
        JSON with num_seqlets (all seqlets called), seqlets (the first
        200, lowest p-value first; each with example_idx, start, end,
        attribution = summed attribution over the span, and p-value),
        truncated, and files (seqlets: the TSV of all seqlets, with
        motif and motif_p_value columns when annotated).  "warnings"
        lists the distinct warnings raised during the attribution, each
        once as {message, count} as in tfbs_analyze; when
        no seqlet is called, "note" says why (the caller found no
        significant span, or raised on flat attributions).  With
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
    if not sequences:
        return _no_sequences_error(
            "tfbs_seqlets",
            "Pass the DNA sequences to call seqlets in as sequences (A/C/G/T, "
            "one length); this tool takes them inline only.",
        )
    X, _, err = _load_sequence_input(sequences, None)
    if err:
        return err

    entry, err = _get_model(model_id)
    if err:
        return err
    model = entry["model"]
    device = entry["device"]
    out = _analysis_dir(
        output_dir, model_id, "tfbs_seqlets",
        sequences, n_shuffles, min_seqlet_len, max_seqlet_len, motif_file, seed,
    )

    X = X.to(device)

    model.train()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            attr = tangermeme_dls(model, X, n_shuffles=n_shuffles, device=str(device),
                                  random_state=seed)
    finally:
        model.eval()
    raised = _warning_messages(caught)

    if isinstance(attr, _torch.Tensor):
        attr_np = attr.detach().cpu().numpy()
    else:
        attr_np = _np.asarray(attr)

    # Sum across channels for seqlet discovery: (N, 4, L) → (N, L)
    attr_sum = attr_np.sum(axis=1)

    # recursive_seqlets returns a DataFrame (example_idx, start, end,
    # attribution, p-value) and raises on degenerate input.
    columns = ["example_idx", "start", "end", "attribution", "p-value"]
    note = None
    try:
        seqlets = recursive_seqlets(
            attr_sum,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
        )
    except (ZeroDivisionError, ValueError) as exc:
        seqlets = None
        note = (f"No seqlets were called: tangermeme's seqlet caller raised "
                f"{type(exc).__name__} ({exc}), which happens when the attributions "
                "are flat, e.g. a model that ignores these sequences.")
    if seqlets is None or len(seqlets) == 0:
        import pandas as pd
        seqlets = pd.DataFrame(columns=columns)
        note = note or (
            "No seqlets were called: no span of the summed attributions passed the "
            "seqlet caller's significance threshold. Try more or longer sequences, "
            "or a smaller min_seqlet_len.")

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
    if note:
        result["note"] = note
    if raised:
        result["warnings"] = raised
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

    mkdir_err = _mkdir_error(out, "output_dir")
    if mkdir_err:
        return mkdir_err
    table = seqlets[columns].copy()
    if matched:
        table["motif"] = [name or "" for name, _ in matched]
        table["motif_p_value"] = [pval for _, pval in matched]
    tsv = out / "seqlets.tsv"
    table.to_csv(tsv, sep="\t", index=False, lineterminator="\n")
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

    Units: the same as tfbs_predict.  For a regression model loaded with
    scaling parameters, every prediction is converted to the original
    label units (standardize: x * std + mean; normalize: x * (max - min)
    + min) and delta and delta_delta are multiplied by std (or max -
    min); "inverse_transformed" says whether that happened, and "units"
    is "label", "training_scale" or "unknown" as in tfbs_predict.

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
                    prediction, delta_delta; earlier versions wrote only
                    the top interactions, as epistasis_top.csv).  Default
                    <results>/<model_id>/tfbs_epistasis/<digest>/, with
                    <results> the results_dir of tfbs_server_info and
                    <digest> a hash of the sequence and positions.

    Returns:
        JSON with original_prediction, num_positions,
        num_pairs_analyzed (the number of position pairs,
        num_positions * (num_positions - 1) / 2), num_double_mutants
        (the double mutants scored, 9 per position pair),
        top_interactions (double mutants with |delta_delta| > 1e-6,
        sorted by |delta_delta|, capped at 500, same fields as
        epistasis_matrix.csv), inverse_transformed, units, output_dir and a
        "files" object holding the paths of single_mutations and
        epistasis_matrix.
    """
    _ensure_imports()
    if not sequence:
        return _error("tfbs_epistasis got no sequence.",
                      "Pass one DNA sequence (A/C/G/T) as sequence.")
    X, _, err = _load_sequence_input([sequence], None)
    if err:
        return err

    entry, err = _get_model(model_id)
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

    inverse = _inverse_scaling(entry, None)
    to_units = inverse or (lambda v: v)
    y_orig = to_units(_predict([()])[0])

    singles = [(p, b) for p in pos for b in range(4) if b != ref[p]]
    single_preds = [to_units(y) for y in _predict([(s,) for s in singles])]
    single_delta = {s: y - y_orig for s, y in zip(singles, single_preds)}

    doubles = [
        ((pi, bi), (pj, bj))
        for i, pi in enumerate(pos)
        for pj in pos[i + 1:]
        for bi in range(4) if bi != ref[pi]
        for bj in range(4) if bj != ref[pj]
    ]
    double_preds = [to_units(y) for y in _predict(doubles)]

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
        "inverse_transformed": inverse is not None,
        "units": _units(entry, inverse),
    }

    import csv as _csv
    out = _analysis_dir(output_dir, model_id, "tfbs_epistasis", sequence, pos)
    mkdir_err = _mkdir_error(out, "output_dir")
    if mkdir_err:
        return mkdir_err
    result["output_dir"] = str(out)
    singles_csv = out / "single_mutations.csv"
    with open(singles_csv, "w", newline="") as f:
        w = _csv.writer(f, lineterminator="\n")
        w.writerow(["position", "ref_base", "base", "prediction", "delta"])
        for (p, b), y in zip(singles, single_preds):
            w.writerow([p, _IDX_TO_NUC[ref[p]], _IDX_TO_NUC[b], y, single_delta[(p, b)]])
    pairs_csv = out / "epistasis_matrix.csv"
    with open(pairs_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, lineterminator="\n", fieldnames=[
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

    A window that reaches past the ends of the input (padding) contributes
    only its in-bounds columns, and each PFM column is averaged over the
    windows that cover it.  The PFM spans the kernel from its first to
    its last covered column, so it can be narrower than the kernel;
    uncovered columns between them are filled with a uniform 0.25 and
    reported in a "warning".  When a layer's receptive span
    (dilation x (kernel_size - 1) + 1) is longer than the input, e.g. a
    42-wide kernel on 24 bp windows, every window is clipped this way and
    the result carries a "warning" naming that receptive field.

    Writes to output_dir:
      filter_summary.json: one record per analysed filter (layer,
        filter_id, kernel_size, max_activation, num_activating,
        pfm_sites, pfm_width, in_meme).
      filters.meme: the PFMs in MEME format (one MOTIF <layer>_f<id>
        per filter with a PFM; letter-probability rows, nsites = the
        fewest windows behind any covered column), ready for TOMTOM
        against JASPAR.
      top_subsequences.json: per filter, the top_k_subsequences input
        windows with the highest activation (sequence_index, start,
        activation, subsequence; the in-bounds part of the window).

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
        num_pfms (MOTIFs in the MEME file), filter_summary (first 100
        records) and "warning" when a layer that reads the input has a
        receptive span longer than the input or a PFM has uncovered
        columns.
    """
    _ensure_imports()
    if not sequences:
        return _no_sequences_error(
            "tfbs_conv_filters",
            "Pass DNA sequences to summarise the filters on as sequences "
            "(A/C/G/T, the model's input length), e.g. a few hundred peak windows.",
        )
    X, _, err = _load_sequence_input(sequences, None)
    if err:
        return err

    entry, err = _get_model(model_id)
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
    mkdir_err = _mkdir_error(out, "output_dir")
    if mkdir_err:
        return mkdir_err

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
    oversized: list[str] = []
    gapped: list[str] = []
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
        if on_input and span > L:
            oversized.append(f"{layer_name} (receptive field {span} bp: kernel "
                             f"{kernel_size}, dilation {dilation})")
        offsets = _np.arange(kernel_size) * dilation

        def _window(seq_idx: int, out_pos: int):
            """(kernel columns inside the input, their input positions), or None."""
            cols = out_pos * stride - pad + offsets
            inside = (cols >= 0) & (cols < L)
            if not inside.any():
                return None
            return _np.flatnonzero(inside), cols[inside]

        for f_idx in range(n_filters):
            f_act = act[:, f_idx, :]  # (N, L_out)
            max_act = float(f_act.max())
            threshold = max_act * 0.8
            high_mask = f_act >= threshold
            n_activating = int(high_mask.sum())

            pfm_sites = 0
            pfm_width = 0
            in_meme = False
            if on_input and max_act > 0:
                counts = _np.zeros((4, kernel_size))
                coverage = _np.zeros(kernel_size)
                for seq_idx in range(f_act.shape[0]):
                    for p in _torch.where(high_mask[seq_idx])[0][:10].tolist():
                        w = _window(seq_idx, p)
                        if w is None:
                            continue
                        kcols, positions = w
                        counts[:, kcols] += X_cpu[seq_idx][:, _torch.as_tensor(positions)].numpy()
                        coverage[kcols] += 1
                        pfm_sites += 1
                covered = _np.flatnonzero(coverage)
                if pfm_sites and covered.size:
                    span = _np.arange(covered[0], covered[-1] + 1)
                    seen = coverage[span] > 0
                    pfm = _np.full((4, span.size), 0.25)
                    pfm[:, seen] = counts[:, span[seen]] / coverage[span[seen]]
                    pfm_width = int(span.size)
                    if not seen.all():
                        gapped.append(f"{layer_name} filter {f_idx}")
                    nsites = int(coverage[covered].min())
                    motif = f"{_safe_name(layer_name)}_f{f_idx}"
                    rows = "\n".join(
                        " " + " ".join(f"{v:.6f}" for v in pfm[:, j]) for j in range(pfm_width)
                    )
                    meme_blocks.append(
                        f"MOTIF {motif}\n"
                        f"letter-probability matrix: alength= 4 w= {pfm_width} "
                        f"nsites= {nsites} E= 0\n{rows}\n"
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
                    positions = w[1]
                    top.append({
                        "sequence_index": seq_idx,
                        "start": int(positions[0]),
                        "activation": round(float(flat[k]), 4),
                        "subsequence": _tensor_to_sequences(
                            X_cpu[seq_idx][:, _torch.as_tensor(positions)].unsqueeze(0))[0],
                    })
                top_subsequences.append({"layer": layer_name, "filter_id": f_idx, "top": top})

            filter_summary.append({
                "layer": layer_name,
                "filter_id": f_idx,
                "max_activation": round(max_act, 4),
                "num_activating": n_activating,
                "kernel_size": kernel_size,
                "pfm_sites": pfm_sites,
                "pfm_width": pfm_width,
                "in_meme": in_meme,
            })

    files = {
        "filter_summary": str(out / "filter_summary.json"),
        "meme": str(out / "filters.meme"),
        "top_subsequences": str(out / "top_subsequences.json"),
    }
    with open(files["filter_summary"], "w", newline="\n") as f:
        json.dump(filter_summary, f, indent=2)
    with open(files["meme"], "w", newline="\n") as f:
        f.write(
            "MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n"
            "Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n"
        )
        f.write("".join(block + "\n" for block in meme_blocks))
    with open(files["top_subsequences"], "w", newline="\n") as f:
        json.dump(top_subsequences, f, indent=2)

    result: dict[str, Any] = {
        "output_dir": str(out),
        "files": files,
        "num_layers": len(conv_layers),
        "num_filters_analyzed": len(filter_summary),
        "num_pfms": len(meme_blocks),
        "filter_summary": filter_summary[:100],
    }
    cautions = []
    if oversized:
        cautions.append(
            f"The receptive field of {'; '.join(oversized)} is longer than the "
            f"{L} bp input, so every window is clipped to the input: the PFMs "
            "hold only the kernel columns that fall on sequence, and each column "
            "is averaged over the windows that cover it."
        )
    if gapped:
        shown = ", ".join(gapped[:10]) + (f" and {len(gapped) - 10} more" if len(gapped) > 10 else "")
        cautions.append(
            f"No activating window covers some kernel columns inside the PFM of {shown}; "
            "those columns are filled with a uniform 0.25 so the PFM keeps the "
            "kernel's spacing from its first to its last covered column."
        )
    if cautions:
        result["warning"] = " ".join(cautions)
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Checkpoints of the benchmark and validation tools
# ---------------------------------------------------------------------------
def _loaded_from(module: str, root: Path) -> bool:
    """Whether the server itself imported module from a file under root."""
    loaded = sys.modules.get(module)
    if loaded is None:
        return False
    files = [getattr(loaded, "__file__", None), *list(getattr(loaded, "__path__", None) or [])]
    root = root.resolve()
    for f in files:
        if f:
            try:
                if Path(f).resolve().is_relative_to(root):
                    return True
            except (OSError, ValueError):
                continue
    return False


_IMPORT_PROBE_TIMEOUT = 120


def _module_importable(module: str, project_root: Path,
                       timeout: int = _IMPORT_PROBE_TIMEOUT) -> bool | None:
    """Whether a pipeline started from project_root can import module; None if the check timed out.

    Checked in a child interpreter with the pipelines' environment and
    working directory, so the server never imports a checkpoint's
    package itself; a module the server already imported from under
    project_root counts without one.  The probe blocks, so async tools
    call it through asyncio.to_thread.
    """
    if _loaded_from(module, project_root):
        return True
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import importlib, sys; importlib.import_module(sys.argv[1])",
             module],
            cwd=str(project_root), env=_pipeline_env(), stdin=subprocess.DEVNULL,
            capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    except OSError:
        return False
    return probe.returncode == 0


def _checkpoint_summary(path: str, cache: dict[str, dict]) -> dict:
    """{"arch", "class_path", "bare"} of a checkpoint, told by its content and read once per call.

    arch is set for an inference checkpoint (a dict with arch,
    hyper_parameters and state_dict, such as the NKX2-1 study's released
    vcnnbpnet.pt files) and None otherwise; class_path is the class a
    Lightning checkpoint records in hyper_parameters _class_path, or None;
    bare is True for a bare state_dict (torch.save(model.state_dict())),
    which neither loader can rebuild a model from.  cache maps a path to
    its summary for the length of one tool call.
    """
    if path not in cache:
        from tfbs.nn.loading import is_bare_state_dict

        ckpt = _load_weights_only(Path(path))
        if _is_inference_payload(ckpt):
            cache[path] = {"arch": ckpt["arch"], "class_path": None, "bare": False}
        elif is_bare_state_dict(ckpt):
            cache[path] = {"arch": None, "class_path": None, "bare": True}
        else:
            meta = _checkpoint_meta(Path(path), ckpt if ckpt is not None else {})
            cache[path] = {"arch": None, "class_path": _checkpoint_class_path(Path(path), meta),
                           "bare": False}
        del ckpt
    return cache[path]


def _inference_checkpoint_error(found: dict[str, str], model_type: str | None,
                                cache: dict[str, dict] | None = None,
                                argument: str = "model_type") -> str | None:
    """Error JSON for the first checkpoint that cannot be loaded as given, else None.

    An inference checkpoint's arch must be a class in tfbs.nn.models and,
    when model_type is given, equal it, because the pipelines rebuild the
    model from that class.  A bare state_dict is refused, because neither
    loader can rebuild a model from it.  Lightning checkpoints are left to
    the class-path probe.
    """
    cache = {} if cache is None else cache
    known = _available_model_types()
    for name, path in found.items():
        summary = _checkpoint_summary(path, cache)
        if summary.get("bare"):
            return _error(
                f"The checkpoint for {name} ({path}) is a bare state_dict (a dict of "
                "tensors only, as torch.save(model.state_dict()) writes it), which records "
                "neither the architecture nor its hyper_parameters.",
                "Save the model as {arch, hyper_parameters, state_dict} with torch.save "
                "(the format of the released vcnnbpnet.pt files) or pass a Lightning .ckpt. "
                "Nothing was run or submitted.",
                checkpoint=path,
            )
        arch = summary["arch"]
        if arch is None:
            continue
        if arch not in known:
            return _error(
                f"The checkpoint for {name} ({path}) holds a {arch} model, which is not a "
                "model class in tfbs.nn.models of the installed tfbs-mcp.",
                "Update tfbs-mcp to a version that has this architecture, or check that "
                "the file is an unmodified release. Nothing was run or submitted.",
                arch=arch, checkpoint=path, available=known,
            )
        if model_type and arch != model_type:
            return _error(
                f"The checkpoint for {name} ({path}) holds a {arch} model, but "
                f"{argument} is {model_type}.",
                f"Pass {argument}='{arch}', the architecture the file records, or name "
                f"a checkpoint of a {model_type} model. Nothing was run or submitted.",
                arch=arch, checkpoint=path,
            )
    return None


def _unimportable_checkpoint_class(found: dict[str, str], project_root: Path,
                                   cache: dict[str, dict] | None = None) -> str | None:
    """Error JSON for the first checkpoint whose recorded class module the pipeline cannot import.

    Each module is probed once per call, however many checkpoints name it.
    An inference checkpoint records no class path and is not probed: the
    pipelines rebuild it from its arch in tfbs.nn.models.
    """
    cache = {} if cache is None else cache
    verdicts: dict[str, bool | None] = {}
    for name, path in found.items():
        class_path = _checkpoint_summary(path, cache)["class_path"]
        if not class_path:
            continue
        module = class_path.rpartition(".")[0]
        if module not in verdicts:
            verdicts[module] = _module_importable(module, project_root)
        if verdicts[module]:
            continue
        if verdicts[module] is None:
            return _error(
                f"The checkpoint for {name} ({path}) was saved with the class {class_path}, "
                f"and whether the pipeline, which runs from the project root {project_root}, "
                f"can import its module {module} could not be checked within "
                f"{_IMPORT_PROBE_TIMEOUT} s.",
                "Retry the call (the first import of a large package can be slow, e.g. "
                "on a busy login node); if it times out again, check that "
                f"'python -c \"import {module}\"' finishes when run from {project_root}. "
                "Nothing was run or submitted.",
                class_path=class_path, checkpoint=path,
            )
        return _error(
            f"The checkpoint for {name} ({path}) was saved with the class {class_path}, "
            f"and its module {module} cannot be imported by the pipeline, which runs "
            f"from the project root {project_root}.",
            f"Put the package that defines {module} on the PYTHONPATH of the MCP server "
            "(the 'env' block of the client config) and restart it, or set "
            "TFBS_PROJECT_ROOT to the directory that holds it; for the NKX2-1 study "
            "checkpoints that is the TF-NN workspace, whose classes package maps "
            "classes.* to tfbs.*. Nothing was run or submitted.",
            class_path=class_path, checkpoint=path,
        )
    return None


def _benchmark_checkpoints(
    models_dir: str, datasets: list[str], model_type: str, scaling_method: str | None,
    checkpoints: dict[str, str] | None, project_root: Path,
) -> tuple[dict[str, str] | None, str | None, str | None]:
    """({dataset: absolute checkpoint path}, warning or None, None) or (None, None, error JSON)."""
    from tfbs.pipelines.checkpoints import PREFERRED_SCALING, resolve_checkpoints

    if scaling_method is not None and scaling_method not in _SCALING_METHODS:
        return None, None, _error(f"Unknown scaling_method '{scaling_method}'.",
                                  f"Use one of: {list(_SCALING_METHODS)}.")

    def _absolute(path: str) -> Path:
        q = Path(path).expanduser()
        return q if q.is_absolute() else project_root / q

    explicit = {name: str(_absolute(path)) for name, path in (checkpoints or {}).items()}
    found, problems = resolve_checkpoints(
        _absolute(models_dir), datasets, model_type, scaling_method, explicit or None,
    )
    if problems:
        several = [p for p in problems if len(p["candidates"]) > 1]
        elsewhere = [p for p in problems if len(p["candidates"]) == 1]
        fix = ("Pass checkpoints={dataset: path} to name each checkpoint, "
               "or point models_dir and datasets at <models_dir>/<dataset>/"
               "<scaling_method>/best_<model_type>.ckpt (tfbs_config reports "
               "the checkpoint_path it trains to). Nothing was run or submitted: "
               "the lookup is all-or-nothing.")
        if several:
            fix = ("Several checkpoints match: pass scaling_method naming the "
                   "folder to use, or checkpoints={dataset: path}. " + fix)
        elif elsewhere and scaling_method is not None:
            fix = (f"scaling_method '{scaling_method}' names the only folder used; "
                   "pass the scaling_method of the candidate's folder, or omit "
                   "scaling_method to fall back to it. " + fix)
        return None, None, _error(
            "No single checkpoint for "
            + ", ".join(p["dataset"] for p in problems) + ".",
            fix,
            checkpoint_problems=problems,
        )
    resolved = {name: str(path.resolve()) for name, path in found.items()}
    requested = scaling_method or PREFERRED_SCALING
    differ = [
        f"{name} ({Path(path).parent.name}/)" for name, path in resolved.items()
        if Path(path).parent.name != requested
        and (not explicit or (scaling_method is not None
                              and Path(path).parent.name in _SCALING_METHODS))
    ]
    warning = None
    if differ:
        warning = (
            f"The checkpoint of {', '.join(differ)} is not in a '{requested}' folder"
            + ("" if scaling_method else " (the default when scaling_method is omitted)")
            + ", so it was probably trained with another label scaling; check that "
            "this is the model you mean, or pass scaling_method or checkpoints."
        )
    cache: dict[str, dict] = {}
    err = (_inference_checkpoint_error(resolved, model_type, cache)
           or _unimportable_checkpoint_class(resolved, project_root, cache))
    if err:
        return None, None, err
    return resolved, warning, None


def _checkpoint_args(found: dict[str, str]) -> list[str]:
    return ["--checkpoints", *[f"{name}={path}" for name, path in found.items()]]


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
    scaling_method: str | None = None,
    checkpoints: dict[str, str] | None = None,
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
            tfbs_server_info), where the pipeline runs.  Each dataset's
            checkpoint is <models_dir>/<dataset>/<scaling_method>/
            best_<model_type>.ckpt, the layout tfbs_config writes; when
            that file is missing and scaling_method is omitted, the one
            <dataset>/*/best_<model_type>.ckpt is used if there is exactly
            one.  The lookup is all-or-nothing: when any dataset has no
            single checkpoint, a Lightning checkpoint's recorded class
            (hyper_parameters _class_path) cannot be imported from the
            project root, or an inference checkpoint's arch is not
            model_type, the call returns an error and nothing is run
            or submitted.
        scaling_method: Folder of the checkpoint under each dataset.
            Default null: "standardize" (the regression models of the
            NKX2-1 study), falling back to the one other folder that
            holds best_<model_type>.ckpt (a classifier from tfbs_config
            is under "none"), and the result then carries a "warning".
            An explicit value names the only folder used and never falls
            back.
        checkpoints: Explicit {dataset name: checkpoint path}; replaces
            the models_dir lookup, and its names replace datasets.  Both
            checkpoint formats are accepted, here and in the lookup: a
            PyTorch Lightning .ckpt, or an inference checkpoint such as
            the NKX2-1 study's released <models>/flank/vcnnbpnet.pt
            (Zenodo, doi:10.5281/zenodo.21399578), whose arch must equal
            model_type.  The NN scores are the model's raw outputs, in
            the training scale for a standardized model, which leaves
            the AUCs unchanged.
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
        JSON with output_dir, checkpoints (the file used per dataset) and
        the list of generated files, or job_id if SLURM.
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
    found, ckpt_warning, err = await asyncio.to_thread(
        _benchmark_checkpoints,
        models_dir, datasets, model_type, scaling_method, checkpoints, project_root)
    if err:
        return err

    out = Path(output_dir).resolve()
    mkdir_err = _mkdir_error(out, "output_dir")
    if mkdir_err:
        return mkdir_err

    device = _resolve_device(device, slurm_gpu=True if slurm else None)

    cmd_parts = [
        *_pipeline_argv("chipseq_benchmark"),
        "--intersect-dir", str(intersect_path),
        "--output-dir", str(out),
        "--pos-tf", pos_tf,
        "--neg-tfs", *neg_tfs,
        "--cutoffs", *[str(c) for c in cutoffs],
        "--model-type", model_type,
        *_checkpoint_args(found),
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
        submit_result["checkpoints"] = found
        if ckpt_warning:
            submit_result["warning"] = ckpt_warning
        return json.dumps(submit_result)

    result, err = await _run_pipeline(cmd_parts, project_root, 3600, "Benchmark")
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
        "checkpoints": found,
        **({"warning": ckpt_warning} if ckpt_warning else {}),
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
    scaling_method: str | None = None,
    checkpoints: dict[str, str] | None = None,
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
            for NKX2-1). Falls back to $TFBS_MOTIF_FILE. The FIMO leg runs
            the MEME Suite's fimo program (no tangermeme fallback here), so
            the call returns an error before anything runs or is submitted
            when fimo is not on the server's PATH.
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
            tfbs_server_info), where the pipeline runs.  Each dataset's
            checkpoint is <models_dir>/<dataset>/<scaling_method>/
            best_<model_type>.ckpt, the layout tfbs_config writes; when
            that file is missing and scaling_method is omitted, the one
            <dataset>/*/best_<model_type>.ckpt is used if there is exactly
            one.  The lookup is all-or-nothing: when any dataset has no
            single checkpoint, a Lightning checkpoint's recorded class
            (hyper_parameters _class_path) cannot be imported from the
            project root, or an inference checkpoint's arch is not
            model_type, the call returns an error and nothing is run
            or submitted.
        scaling_method: Folder of the checkpoint under each dataset.
            Default null: "standardize" (the regression models of the
            NKX2-1 study), falling back to the one other folder that
            holds best_<model_type>.ckpt (a classifier from tfbs_config
            is under "none"), and the result then carries a "warning".
            An explicit value names the only folder used and never falls
            back.
        checkpoints: Explicit {dataset name: checkpoint path}; replaces
            the models_dir lookup, and its names replace datasets.  Both
            checkpoint formats are accepted, here and in the lookup: a
            PyTorch Lightning .ckpt, or an inference checkpoint such as
            the NKX2-1 study's released <models>/flank/vcnnbpnet.pt
            (Zenodo, doi:10.5281/zenodo.21399578), whose arch must equal
            model_type.  The NN scores are the model's raw outputs, in
            the training scale for a standardized model, which leaves
            the AUCs unchanged.
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
        JSON with output_dir, checkpoints (the file used per dataset),
        file list, and (if slurm) job_id.
    """
    if neg_tfs is None:
        neg_tfs = ["GATA1", "MYOD1", "NKX2-5", "RXRA"]
    if datasets is None:
        datasets = ["all_mean", "core_mean", "flank_mean"]

    genome_path, err = _resolve_reference(genome_fasta, "genome_fasta")
    if err:
        return err
    err = _fimo_binary_error("tfbs_chipseq_method_comparison", "")
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
    found, ckpt_warning, err = await asyncio.to_thread(
        _benchmark_checkpoints,
        models_dir, datasets, model_type, scaling_method, checkpoints, project_root)
    if err:
        return err

    out = Path(output_dir).resolve()
    mkdir_err = _mkdir_error(out, "output_dir")
    if mkdir_err:
        return mkdir_err

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
        *_checkpoint_args(found),
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
        submit_result["checkpoints"] = found
        if ckpt_warning:
            submit_result["warning"] = ckpt_warning
        return json.dumps(submit_result)

    result, err = await _run_pipeline(cmd_parts, project_root, 43200, "Method comparison")
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
        "mode": "direct", "output_dir": str(out), "checkpoints": found,
        **({"warning": ckpt_warning} if ckpt_warning else {}),
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
    scaling_method: str | None = None,
    checkpoints: dict[str, str] | None = None,
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
            $TFBS_MOTIF_FILE.  The FIMO leg always runs and needs the
            MEME Suite's fimo program on the server's PATH (also on the
            compute node for slurm=true); without it the call returns an
            error before anything runs or is submitted.  The tangermeme
            fallback of tfbs_fimo does not apply here.
        foldx_pdb: Repaired TF-DNA complex PDB for FoldX scoring.
            Falls back to $TFBS_FOLDX_PDB. Not required when
            skip_foldx is true.
        score_min: Minimum experiment count in the ReMap BED.
        max_positives: Subsample the high-confidence set to this many
            peaks (FoldX is the bottleneck; default 500).
        negative_mode: "shuffle" for dinucleotide-shuffled positives or
            "genomic" for length+GC matched random regions.
        datasets: NN dataset names (subdirectories of models_dir).
        model_type: Model architecture class name in tfbs.nn.models; an
            unknown one is an error before anything runs or is submitted.
        models_dir: Base checkpoint directory; a relative path resolves
            against the project root (TFBS_PROJECT_ROOT, see
            tfbs_server_info), where the pipeline runs.  Each dataset's
            checkpoint is <models_dir>/<dataset>/<scaling_method>/
            best_<model_type>.ckpt, the layout tfbs_config writes; when
            that file is missing and scaling_method is omitted, the one
            <dataset>/*/best_<model_type>.ckpt is used if there is exactly
            one.  The lookup is all-or-nothing: when any dataset has no
            single checkpoint, a Lightning checkpoint's recorded class
            (hyper_parameters _class_path) cannot be imported from the
            project root, or an inference checkpoint's arch is not
            model_type, the call returns an error and nothing is run
            or submitted.
        scaling_method: Folder of the checkpoint under each dataset.
            Default null: "standardize" (the regression models of the
            NKX2-1 study), falling back to the one other folder that
            holds best_<model_type>.ckpt (a classifier from tfbs_config
            is under "none"), and the result then carries a "warning".
            An explicit value names the only folder used and never falls
            back.
        checkpoints: Explicit {dataset name: checkpoint path}; replaces
            the models_dir lookup, and its names replace datasets.  Both
            checkpoint formats are accepted, here and in the lookup: a
            PyTorch Lightning .ckpt, or an inference checkpoint such as
            the NKX2-1 study's released <models>/flank/vcnnbpnet.pt
            (Zenodo, doi:10.5281/zenodo.21399578), whose arch must equal
            model_type.  The NN scores are the model's raw outputs, in
            the training scale for a standardized model, which leaves
            the AUCs unchanged.
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
        JSON with output_dir, checkpoints (the file used per dataset),
        file list, and either job_id (slurm=true) or the parsed AUC summary (slurm=false, summary.csv read back
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

    err = _fimo_binary_error("tfbs_remap_validation", "")
    if err:
        return err
    err = _pipeline_requirements_error("remap_validation", "tfbs_remap_validation")
    if err:
        return err
    err = _model_type_error(model_type)
    if err:
        return err
    project_root, err = _work_dir()
    if err:
        return err
    found, ckpt_warning, err = await asyncio.to_thread(
        _benchmark_checkpoints,
        models_dir, datasets, model_type, scaling_method, checkpoints, project_root)
    if err:
        return err

    out = Path(output_dir).resolve()
    mkdir_err = _mkdir_error(out, "output_dir")
    if mkdir_err:
        return mkdir_err

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
        *_checkpoint_args(found),
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
        submit_result["checkpoints"] = found
        if ckpt_warning:
            submit_result["warning"] = ckpt_warning
        return json.dumps(submit_result)

    result, err = await _run_pipeline(cmd_parts, project_root, 43200, "ReMap validation")
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
        "checkpoints": found,
        **({"warning": ckpt_warning} if ckpt_warning else {}),
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
    model_type: str = "VCNNBpnet",
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
        peaks_bed: Path to a BED file of TF binding peaks with 3 or more
            tab-separated columns.  Only chrom, start and end are read,
            plus the 4th column as the confidence count when it is a
            number on every line: the 9-column ChIP-Atlas allintersect
            format keeps the experiment overlap count there.  Otherwise
            every peak counts 1.
        output_dir: Directory to write results, plots, and summary CSV.
        clinvar_vcf: Indexed ClinVar VCF (.vcf.gz with its .tbi, or .csi,
            next to it). Falls back to $TFBS_CLINVAR_VCF.  Without the
            index the call returns an error before anything runs.
        genome_fasta: Reference genome FASTA file. Falls back to
            $TFBS_GENOME_FASTA.
        peak_confidence: Minimum value of the BED 4th column to keep
            (e.g. ChIP-Atlas overlap >= 3). Default 1.  Above 1 every
            line needs a number in column 4; otherwise the run stops
            with an error naming the first line without one.
        nn_models: Dict mapping model name to checkpoint path, e.g.
            {"flank_mean": "/path/to/best_VCNNBpnet.ckpt"}.  Both
            checkpoint formats are accepted: a PyTorch Lightning .ckpt,
            or an inference checkpoint such as the NKX2-1 study's
            released <models>/flank/vcnnbpnet.pt (Zenodo,
            doi:10.5281/zenodo.21399578), e.g. {"flank":
            "<models>/flank/vcnnbpnet.pt"}.  A relative path resolves
            against the project root (TFBS_PROJECT_ROOT, see
            tfbs_server_info); a checkpoint that does not exist, or a
            name with "=" or whitespace, is an error.  The NN scores are
            the model's raw outputs, in the training scale for a
            standardized model, which leaves the AUCs unchanged.
            Omitted: no NN leg.
        model_type: Class in tfbs.nn.models that every nn_models
            checkpoint is loaded as (default "VCNNBpnet"; e.g. "RNN" or
            "BPNetReal" for a model trained with that architecture).  An
            unknown class, or an inference checkpoint whose arch is
            another class, is an error before anything runs or is
            submitted.
        motif_file: JASPAR / MEME motif file for the FIMO leg.  Empty or
            null uses $TFBS_MOTIF_FILE; when that is unset too, the FIMO
            leg is skipped.  The FIMO leg needs the MEME Suite's fimo
            program on the server's PATH (also on the compute node for
            slurm=true); without it the call returns an error before
            anything runs or is submitted.  The tangermeme fallback of
            tfbs_fimo does not apply here.
        foldx_pdb: Repaired TF-DNA complex PDB for the FoldX leg.  Empty
            or null uses $TFBS_FOLDX_PDB; when that is unset too, the
            FoldX leg is skipped.  FoldX is computationally expensive;
            consider slurm=true.
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
        JSON with mode, output_dir, clinvar_vcf, nn_models (the name to
        checkpoint map used, absolute paths), model_type, motif_file and
        foldx_pdb (the files used, null for a skipped leg), and command.
        slurm=true adds job_id and expected_summary (the
        summary_all_methods.csv the job writes).  slurm=false adds files,
        stdout (its last 1000 characters) and summary, the
        summary_all_methods.csv read back as a list of {method, auc_abs,
        auc_raw, mwu_p, n_path, n_benign} (method is "NN <name>", "FIMO"
        or "FoldX"; a method with fewer than 10 scored variants or one
        class only has no row).  When no ClinVar variant falls in the
        peaks the pipeline stops and the call returns an error saying so.
    """
    clinvar_path, err = _resolve_reference(clinvar_vcf, "clinvar_vcf")
    if err:
        return err
    if not any(Path(str(clinvar_path) + ext).is_file() for ext in (".tbi", ".csi")):
        return _error(
            f"The ClinVar VCF has no tabix index: {clinvar_path}.tbi not found.",
            f"Index it with htslib's tabix (tabix -p vcf {clinvar_path}), or download "
            "the .tbi that ClinVar publishes next to the VCF, and keep it in the same "
            "directory. The pipeline reads variants by region and finds none without it.",
        )
    genome_path, err = _resolve_reference(genome_fasta, "genome_fasta")
    if err:
        return err
    peaks_path = Path(peaks_bed).expanduser().resolve()
    if not peaks_path.is_file():
        return _error(f"peaks_bed not found: {peaks_path}",
                      "Pass the path of a BED file of TF binding peaks.")
    motif_path, err = _optional_reference(motif_file, "motif_file")
    if err:
        return err
    foldx_path, err = _optional_reference(foldx_pdb, "foldx_pdb")
    if err:
        return err
    if motif_path is not None:
        err = _fimo_binary_error(
            "tfbs_disease_variant_validation",
            ", or skip the FIMO leg: leave motif_file empty and unset TFBS_MOTIF_FILE")
        if err:
            return err

    err = _pipeline_requirements_error(
        "disease_variant_validation", "tfbs_disease_variant_validation")
    if err:
        return err
    project_root, err = _work_dir()
    if err:
        return err

    bad_names = [str(n) for n in (nn_models or {})
                 if not str(n) or "=" in str(n) or any(c.isspace() for c in str(n))]
    if bad_names:
        return _error(
            f"nn_models names that cannot be passed to the pipeline: {bad_names}.",
            "Use names without '=' or whitespace (the pipeline reads each model as "
            "name=checkpoint and splits at the first '='), e.g. {\"flank_mean\": "
            "\"/path/best_VCNNBpnet.ckpt\"}.",
        )
    if nn_models:
        err = _model_type_error(model_type)
        if err:
            return err

    models_used: dict[str, str] = {}
    for name, ckpt in (nn_models or {}).items():
        ckpt_path = Path(str(ckpt)).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = project_root / ckpt_path
        models_used[str(name)] = str(ckpt_path.resolve())
    missing = {n: p for n, p in models_used.items() if not Path(p).is_file()}
    if missing:
        return _error(
            "nn_models names checkpoints that do not exist: "
            + ", ".join(f"{n}={p}" for n, p in missing.items()),
            "Pass existing checkpoint files, a Lightning .ckpt or an inference "
            "checkpoint such as a released vcnnbpnet.pt (absolute, or relative to the "
            "project root reported by tfbs_server_info), e.g. the checkpoint_path of "
            "tfbs_config.",
        )
    if models_used:
        err = await asyncio.to_thread(_inference_checkpoint_error, models_used, model_type)
        if err:
            return err

    out = Path(output_dir).resolve()
    mkdir_err = _mkdir_error(out, "output_dir")
    if mkdir_err:
        return mkdir_err

    device = _resolve_device(device, slurm_gpu=False if slurm else None)

    cmd_parts = [
        *_pipeline_argv("disease_variant_validation"),
        "--peaks-bed", str(peaks_path),
        "--clinvar-vcf", str(clinvar_path),
        "--genome", str(genome_path),
        "--output-dir", str(out),
        "--peak-confidence", str(peak_confidence),
        "--nn-window", str(nn_window),
        "--foldx-workers", str(foldx_workers),
        "--device", device,
        "--model-type", model_type,
    ]
    if models_used:
        cmd_parts.append("--models")
        cmd_parts.extend(f"{name}={ckpt}" for name, ckpt in models_used.items())
    if motif_path is not None:
        cmd_parts.extend(["--motif-file", str(motif_path)])
    if foldx_path is not None:
        cmd_parts.extend(["--foldx-pdb", str(foldx_path)])

    cmd = shlex.join(cmd_parts)
    used = {
        "output_dir": str(out),
        "clinvar_vcf": str(clinvar_path),
        "nn_models": models_used,
        "model_type": model_type,
        "motif_file": str(motif_path) if motif_path is not None else None,
        "foldx_pdb": str(foldx_path) if foldx_path is not None else None,
        "command": cmd,
    }

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
        submit_result.update(used)
        submit_result["expected_summary"] = str(out / "summary_all_methods.csv")
        return json.dumps(submit_result)

    result, err = await _run_pipeline(cmd_parts, project_root, 14400, "Disease validation")
    if err:
        return err

    if result.returncode != 0:
        if "No ClinVar variants found" in result.stderr:
            return _error(
                "No ClinVar variant falls in the peaks, so there is nothing to score.",
                "Check that the ClinVar VCF is on the same genome build as the peaks "
                "(GRCh38 for hg38 peaks) and that it holds the BED's chromosomes (chr1 "
                "in the BED finds 1 or chr1 in the VCF, and the other way round); lower "
                "peak_confidence if few peaks pass it.",
                stderr=result.stderr[-1500:], stdout=result.stdout[-500:], **used,
            )
        if "Cannot read --peaks-bed" in result.stderr:
            reason = result.stderr.split("Cannot read --peaks-bed:", 1)[1].strip()
            return _error(
                f"The peaks BED cannot be used: {reason[:600]}",
                "Fix the line it names. A peak_confidence above 1 needs the confidence "
                "count as a number in column 4 on every line; with any other BED pass "
                "peak_confidence=1.",
                stderr=result.stderr[-1500:], stdout=result.stdout[-500:], **used,
            )
        return json.dumps({
            "error": f"Disease validation failed (exit {result.returncode})",
            "fix": "Read stderr for the cause; a model_type that does not match the "
                   "nn_models checkpoints, a peaks BED whose start or end is not an "
                   "integer, or an out-of-memory error are the usual ones.",
            "stderr": result.stderr[-1500:],
            "stdout": result.stdout[-500:],
            **used,
        })

    files = [str(f.relative_to(out)) for f in out.rglob("*") if f.is_file()]
    return json.dumps({
        "mode": "direct",
        **used,
        "files": files,
        "summary": _read_disease_summary(out / "summary_all_methods.csv"),
        "stdout": result.stdout[-1000:],
    })


def _read_disease_summary(path: Path) -> list[dict[str, Any]] | dict[str, str] | None:
    """summary_all_methods.csv as [{method, auc_abs, auc_raw, mwu_p, n_path, n_benign}].

    None when the file is missing, {"parse_error": ...} when it cannot be
    read; a value that is not a finite number becomes null.
    """
    if not path.exists():
        return None

    def number(value: Any, kind):
        parsed = _as_float(value)
        return None if parsed is None else kind(parsed)

    try:
        import csv
        with open(path, encoding="utf-8", newline="") as f:
            return [
                {
                    "method": row["method"],
                    "auc_abs": number(row["auc_abs"], float),
                    "auc_raw": number(row["auc_raw"], float),
                    "mwu_p": number(row["mwu_p"], float),
                    "n_path": number(row["n_path"], int),
                    "n_benign": number(row["n_benign"], int),
                }
                for row in csv.DictReader(f)
            ]
    except Exception as exc:
        return {"parse_error": f"{type(exc).__name__}: {exc}"}


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

    with _stdout_to_stderr():
        unavailable = _unavailable_modules(_preload_native_modules())
    if unavailable:
        logger.info("Optional modules not available: %s", ", ".join(unavailable))
    _detach_child_stdin()
    mcp.run()


if __name__ == "__main__":
    main()
