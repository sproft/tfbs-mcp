from pytorch_lightning.cli import LightningCLI
import os
import sys
import torch

# be able to run from any location
    
torch.set_float32_matmul_precision('high')

project_root = "/sc-projects/sc-proj-cc17-P09_TFBS" # Go up one level from the notebook location
sys.path.append(project_root)

# import modules
from classes.nn.models import BaseModel
from classes.dataloaders.SeqDataset import TFBSDataModule

def cli_main():
    cli = LightningCLI(BaseModel, TFBSDataModule, seed_everything_default=42, subclass_mode_model=True, save_config_kwargs={"overwrite": True})
    # note: don't call fit!!

if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block