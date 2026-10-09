"""Backwards-compatible shim.

The pipeline moved into the installed package as ``tfbs.pipelines.run_sweep``,
so the MCP tools can run it after a plain pip install
(``python -m tfbs.pipelines.run_sweep``). This file keeps the old
repository path working, both as a script and as an import.
"""
from tfbs.pipelines.run_sweep import *  # noqa: F401,F403
from tfbs.pipelines.run_sweep import main

if __name__ == "__main__":
    main()
