"""Command-line pipelines that the training and validation MCP tools run.

Each module runs as ``python -m tfbs.pipelines.<name>`` and takes ``--help``.
They ship with the package, so a plain pip install can run every tool;
``scripts/`` in the repository keeps thin wrappers around them.
"""

import os
import shutil

PIPELINES = (
    "validate",
    "run_sweep",
    "chipseq_benchmark",
    "chipseq_benchmark_with_foldx",
    "remap_validation",
    "disease_variant_validation",
)

# The ReMap and ClinVar validations run the MEME Suite's fimo program for
# their FIMO leg; unlike the tfbs_fimo tool they have no Python fallback.
FIMO_REQUIRED = (
    "fimo (MEME Suite) is not on PATH. This pipeline runs the fimo program for "
    "its FIMO leg and has no fallback; the tangermeme/memelite reimplementation "
    "is used only by the tfbs_fimo tool. Install the MEME Suite "
    "(https://meme-suite.org/meme/doc/download.html, or conda install -c bioconda "
    "meme) and put its bin directory on the PATH."
)

# The ChIP-seq method comparison, the ReMap validation and the ClinVar
# validation score a structure with FoldX.  Given a structure, each stops at
# startup without a FoldX binary instead of silently dropping the FoldX leg;
# the MCP tools that run them take skip_foldx to run without it.  The
# missing-binary error and fix are also those of the MCP structure tools
# (tfbs.mcp.foldx_tools imports them), and live here so that a pipeline can
# report them without importing the MCP server.
FOLDX_MISSING = "FoldX binary not found on PATH."
FOLDX_MISSING_FIX = (
    "Install FoldX (https://foldxsuite.crg.eu/, free academic licence), put "
    "'foldx' or 'foldx5' on the PATH of the MCP server process, and restart "
    "the server."
)
SKIP_FOLDX_HINT = (
    "Or call the tool with skip_foldx=true to run the other scores without FoldX."
)


def find_foldx(explicit=None):
    """The FoldX binary: explicit (--foldx-bin), else $TFBS_FOLDX_BIN, else foldx or foldx5 on PATH."""
    return (explicit or os.environ.get("TFBS_FOLDX_BIN", "").strip()
            or shutil.which("foldx") or shutil.which("foldx5"))


def foldx_required_message():
    """The exit message of a pipeline that needs FoldX and finds no binary.

    The MCP structure tools' FoldX-missing error and fix, plus the
    skip_foldx hint.
    """
    return f"{FOLDX_MISSING} {FOLDX_MISSING_FIX} {SKIP_FOLDX_HINT}"
