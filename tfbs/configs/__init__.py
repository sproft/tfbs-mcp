"""Training configuration templates shipped as package data.

models/<Architecture>.yaml   one LightningCLI model block per architecture
                             (read by the tfbs_config tool)
trainers/, datas/            trainer and data blocks for ``tfbs-train``
config.example.yaml          a complete example config for ``tfbs-train fit``
wandb_sweep_config.yaml      the search space used by tfbs.pipelines.run_sweep
"""

from importlib.resources import files
from pathlib import Path


def config_dir() -> Path:
    """Directory holding the installed templates."""
    return Path(str(files(__name__)))
