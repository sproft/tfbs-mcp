"""Backwards-compatible shim.

The pipeline moved into the installed package as ``tfbs.pipelines.chipseq_benchmark``,
so the MCP tools can run it after a plain pip install
(``python -m tfbs.pipelines.chipseq_benchmark``). This file keeps the old
repository path working, both as a script and as an import.
"""
from tfbs.pipelines.chipseq_benchmark import *  # noqa: F401,F403
from tfbs.pipelines.chipseq_benchmark import main

if __name__ == "__main__":
    main()
