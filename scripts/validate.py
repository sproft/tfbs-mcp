"""Backwards-compatible shim.

The pipeline moved into the installed package as ``tfbs.pipelines.validate``,
so the MCP tools can run it after a plain pip install
(``python -m tfbs.pipelines.validate``). This file keeps the old
repository path working, both as a script and as an import.
"""
from tfbs.pipelines.validate import *  # noqa: F401,F403
from tfbs.pipelines.validate import main

if __name__ == "__main__":
    main()
