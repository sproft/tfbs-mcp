"""Weights & Biases sweep agent for the tfbs models.

Run as ``python -m tfbs.pipelines.run_sweep --help``; the tfbs_sweep MCP tool
creates the sweep and submits this module as the agent job. The search space
is the sweep YAML (tfbs/configs/wandb_sweep_config.yaml is the bundled one);
the model, data and scaling given on the command line override its
placeholder values in every trial.
"""
from __future__ import annotations

# Standard Library Imports
import argparse
import functools
import os
import sys

# Third-party Imports
import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Local Imports
from tfbs.dataloaders.SeqDataset import TFBSDataModule
from tfbs.nn import models

# --- Configuration Constants ---

# Default training settings
MAX_EPOCHS = 100
SEQUENCE_LENGTH = 24  # Assuming this is the fixed input length

# --- Utility Functions ---

def get_model_params(model_name: str, config: wandb.config) -> dict:
    """
    Extracts and structures model-specific hyperparameters from wandb.config.
    
    This function handles the necessary parameter name mapping and list creation
    to correctly initialize each model type.
    """
    if model_name == 'RNN':
        # RNN parameters are mostly flat, but use prefixed names
        return {
            'conv_out_channels': config.conv_out_channels,
            'kernel_size': config.RNN_kernel_size,
            'pool_size': config.pool_size,
            'dropout_conv': config.dropout_conv,
            'gru_hidden_size': config.gru_hidden_size,
            'gru_num_layers': config.gru_num_layers,
            'bidirectional': config.bidirectional,
            'dense_size': config.RNN_dense_size,
            'dropout_fc': config.dropout_fc,
        }
    
    elif model_name == 'VCNNBpnet':
        # VCNNBpnet requires combining two dense sizes into a list
        dense_sizes = [config.VCNN_dense_size_1, config.VCNN_dense_size_2]
        return {
            'num_channels': config.VCNN_num_channels,
            'kernel_size': config.VCNN_kernel_size,
            'pool_output_size': config.VCNN_pool_output_size,
            'dense_sizes': dense_sizes,
        }
        
    elif model_name == 'BPNetReal':
        # BPNetReal parameters use specific names
        return {
            'n_filters': config.BPNet_n_filters,
            'n_layers': config.BPNet_n_layers,
            # n_control_tracks is typically fixed (0 for this use case)
            'n_control_tracks': 0, 
            'count_output_bias': config.BPNet_count_output_bias,
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def get_model_class(model_name: str):
    """Dynamically retrieves the model class based on its name."""
    if model_name == 'RNN':
        return models.RNN
    elif model_name == 'VCNNBpnet':
        return models.VCNNBpnet
    elif model_name == 'BPNetReal':
        return models.BPNetReal
    else:
        raise ValueError(f"Model class not found for: {model_name}")

# --- Core Training Logic (Agent Function) ---

def train_sweep_agent(fixed_config: dict | None = None):
    """
    This function is executed by the WandB agent for each trial.
    It initializes the model, data, optimizer, and trainer based on the sampled config.
    """
    import wandb

    # 1. Start the trial's run; the agent supplies the sampled hyperparameters,
    # and the task's fixed values replace the sweep YAML's placeholders.
    run = wandb.init()
    if fixed_config:
        wandb.config.update(fixed_config, allow_val_change=True)
    config = wandb.config

    try:
        # 2. Extract specific model parameters
        model_name = config.model
        model_class = get_model_class(model_name)
        
        model_params = get_model_params(model_name, config)
        
        # Add common parameters
        model_params.update({
            'input_channels': 4, # Fixed for sequence data
            'input_length': SEQUENCE_LENGTH,
            'dropout': config.dropout,
            'learning_rate': config.lr,
        })

        # 3. Initialize Model and Data Module
        model = model_class(**model_params)
        
        # Instantiate the DataModule
        data_module = TFBSDataModule(
            data_path=config.data_path,
            batch_size=config.batch_size,
            # num_workers is often specified in the shell/trainer YAML
            num_workers=os.cpu_count() // 2, 
            scaling_method=config.scaling_method
        )

        # 4. Configure Optimizer (where weight_decay is applied)
        # Note: PyTorch Lightning allows overriding configure_optimizers() in the model,
        # but for consistent weight decay application, we'll demonstrate it here 
        # (assuming TFBSModel uses standard Adam or AdamW)
        if hasattr(model, 'configure_optimizers'):
            # If the model handles its own optimizer, it must apply weight_decay
            # We'll rely on the model for now.
            pass 
        else:
            # Placeholder for models that don't define it (typically base TFBSModel)
            model.optimizer = optim.AdamW
            model.optimizer_kwargs = {'lr': config.lr, 'weight_decay': config.weight_decay}
            # The actual implementation relies on the model's configure_optimizers() method
            # to pick up model.optimizer_kwargs.

        # 5. Define Callbacks
        
        # Checkpointing: Save the best model based on validation loss
        checkpoint_dir = os.path.join(
            config.checkpoint_dir,
            f"{config.data_name}", 
            f"{config.scaling_method}",
            f"{model_name}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Use wandb run name to create unique filename
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=checkpoint_dir,
            filename=f'best-{model_name}-{run.name}',
            save_top_k=1,
            mode='min',
            save_last=True
        )
        
        # Early Stopping
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )

        # 6. Initialize Trainer
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator='gpu',
            devices=1,
            logger=WandbLogger(experiment=run),
            callbacks=[checkpoint_callback, early_stopping_callback],
            # Enables strict checking for the model's forward pass
            fast_dev_run=False, 
            enable_progress_bar=False,
            log_every_n_steps=5,
        )

        # 7. Start Training
        trainer.fit(model, data_module)

    except Exception as e:
        # Log error to WandB and re-raise
        print(f"Error during training sweep: {e}", file=sys.stderr)
        if wandb.run is not None:
            wandb.log({"error": str(e), "status": "failed"})
        raise


# --- Main Execution (SLURM Task Logic) ---

def run_agent(args):
    """
    Launches the agent for an existing sweep and a specific model/dataset task.
    """
    import wandb
    
    # 1. Setup WandB Environment and Project Name
    # Use the project name passed from the SLURM script
    project_name = args.project_name
    # None lets wandb use the logged-in user's default entity.
    entity_name = os.environ.get("WANDB_ENTITY") or None

    # 2. The sweep YAML defines the search space. The task's fixed values
    # (model, data, scaling) come from the command line and are applied to
    # every trial's config by train_sweep_agent.
    fixed_config = {
        'model': args.model,
        'data_path': args.data_path,
        'data_name': args.data_name,
        'scaling_method': args.scaling_method,
        'batch_size': args.batch_size,
        'checkpoint_dir': args.checkpoint_dir,
        'wandb_project': args.project_name,
    }

    # 4. Initialize Sweep (The SLURM script now handles creation and provides the ID)
    sweep_id = args.sweep_id
    
    # Since the SLURM script ensures the sweep is created and provides the full ID, 
    # the creation logic inside the Python script is commented out/removed.
    if not sweep_id:
        print("ERROR: Sweep ID must be provided by the SLURM script.")
        sys.exit(1)

    # 5. Run the WandB Agent
    # The agent will pull a new config (trial) from the server for this sweep_id
    # and execute the train_sweep_agent function until n_trials are complete.
    print(f"Starting WandB agent for Sweep ID: {sweep_id} (Task: {args.model}/{args.data_name}, Project: {project_name})")
    
    # The `count` argument ensures the agent only runs for the specified number of trials
    wandb.agent(
        sweep_id, 
        function=functools.partial(train_sweep_agent, fixed_config),
        count=args.n_trials,
        project=project_name,  # Use the dynamic project name here
        entity=entity_name,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WandB sweep agent for TFBS models")
    
    # Fixed task parameters (passed from SLURM script)
    parser.add_argument('--model', type=str, required=True, choices=['RNN', 'VCNNBpnet', 'BPNetReal'],
                        help='Model type to train.')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the dataset directory.')
    parser.add_argument('--data-name', type=str, required=True,
                        help='Name of the dataset (for logging/folder structure).')
    parser.add_argument('--scaling-method', type=str, required=True, 
                        choices=['standardize', 'normalize', 'none'],
                        help='Label scaling method.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Fixed batch size for all trials in this run.')
    parser.add_argument('--checkpoint-dir', type=str, default='saved_models_tuned/',
                        help='Base directory for saving model checkpoints.')
    parser.add_argument('--project-name', type=str, required=True,
                        help='WandB project name for this sweep (must match SLURM setting).') # <--- NEW ARGUMENT

    # Sweep control parameters
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials (runs) for this model/dataset combination.')
    parser.add_argument('--sweep-id', type=str, default=None,
                        help='Existing WandB sweep ID to attach to. If None, a new sweep is created.')

    return parser


def main(argv=None):
    run_agent(build_parser().parse_args(argv))


if __name__ == '__main__':
    main()
