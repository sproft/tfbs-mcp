"""LightningCLI entry point for training tfbs models.

This lives inside the installed ``tfbs`` package (not the repo-only ``scripts``
directory, which is not packaged), so the ``tfbs-train`` console script resolves
from a pip-installed wheel.
"""
import torch
from pytorch_lightning.cli import LightningCLI

from tfbs.dataloaders.SeqDataset import TFBSDataModule
from tfbs.nn.models import BaseModel

torch.set_float32_matmul_precision("high")


def cli_main():
    # Build the CLI but do not call fit() here; LightningCLI dispatches the
    # subcommand (fit/validate/test/predict) given on the command line.
    LightningCLI(
        BaseModel,
        TFBSDataModule,
        seed_everything_default=42,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
