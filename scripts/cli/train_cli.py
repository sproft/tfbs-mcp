from pytorch_lightning.cli import LightningCLI
import torch

torch.set_float32_matmul_precision('high')

from tfbs.nn.models import BaseModel
from tfbs.dataloaders.SeqDataset import TFBSDataModule

def cli_main():
    cli = LightningCLI(BaseModel, TFBSDataModule, seed_everything_default=42, subclass_mode_model=True, save_config_kwargs={"overwrite": True})
    # note: don't call fit!!

if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block