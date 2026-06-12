"""Backwards-compatible shim.

The training CLI moved into the installed package at ``tfbs.cli.train_cli`` so the
``tfbs-train`` console script resolves from a pip-installed wheel (the repo-local
``scripts`` directory is not packaged). This re-export keeps ``scripts.cli.train_cli``
working for repo-local Weights & Biases sweeps that still reference it.
"""
from tfbs.cli.train_cli import cli_main

__all__ = ["cli_main"]


if __name__ == "__main__":
    cli_main()
