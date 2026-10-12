"""Backwards-compatible shim.

The pipeline moved into the installed package as ``tfbs.pipelines.disease_variant_validation``,
so the MCP tools can run it after a plain pip install
(``python -m tfbs.pipelines.disease_variant_validation``). This file keeps the old
repository path working, both as a script and as an import.
"""
from tfbs.pipelines.disease_variant_validation import *  # noqa: F401,F403
from tfbs.pipelines.disease_variant_validation import main

if __name__ == "__main__":
    main()
