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
