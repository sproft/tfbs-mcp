"""Command-line pipelines that the training and validation MCP tools run.

Each module runs as ``python -m tfbs.pipelines.<name>`` and takes ``--help``.
They ship with the package, so a plain pip install can run every tool;
``scripts/`` in the repository keeps thin wrappers around them.
"""

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
