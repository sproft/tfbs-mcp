"""Evaluate a checkpoint on the held-out test split and append its metrics to a CSV.

The row holds test_loss and, for a classifier, test_acc and test_auc.

Run as ``python -m tfbs.pipelines.validate --help``; the tfbs_validate MCP tool
submits this module as a SLURM job.

A new row is appended under the file's header.  When a run reports a metric
column the header lacks, the whole file is rewritten through a temporary file
in the same directory and swapped in with os.replace, so a reader never sees a
half-written CSV.  Neither path locks the file: concurrent jobs should write
separate output files (one per job) and merge them afterwards, because a row
another job appends during a rewrite is lost.
"""
import argparse
import csv
import os
import shutil
import sys
import tempfile

import pytorch_lightning as pl
import torch

from tfbs.dataloaders.SeqDataset import TFBSDataModule
import tfbs.nn.models as models


# --- Helper function to dynamically retrieve the Model class ---
def get_model_class(model_name: str):
    """Retrieves the model class from the imported models module."""
    try:
        # We assume the model class is defined in 'models.py' or globally available.
        return getattr(models, model_name)
    except AttributeError:
        print(f"Error: Model class '{model_name}' not found in 'models.py'.")
        print("Please ensure your model is correctly defined and imported.")
        sys.exit(1)


BASE_FIELDS = ['modeltype', 'scaling_method', 'dataset_path']


def save_results(args, metrics):
    """Append one row (the run's settings plus every test metric) to the CSV.

    ``metrics`` is the dict trainer.test() returns, e.g. test_loss and, for a
    classifier, test_acc and test_auc; a bare number is taken as test_loss.
    A file whose header lacks a new metric column is rewritten with the
    wider header, so earlier rows keep their values.
    """
    if not isinstance(metrics, dict):
        metrics = {'test_loss': metrics}
    row = {
        'modeltype': args.model_name,
        'scaling_method': args.scaling_method,
        'dataset_path': os.path.basename(os.path.normpath(args.data_path)),
        **{k: float(v) for k, v in metrics.items()},
    }
    metric_fields = ['test_loss'] + sorted(k for k in metrics if k != 'test_loss')

    existing_rows, existing_fields = [], []
    if os.path.exists(args.output_file):
        with open(args.output_file, newline='') as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)
    fieldnames = list(existing_fields or BASE_FIELDS)
    for name in BASE_FIELDS + metric_fields:
        if name not in fieldnames:
            fieldnames.append(name)

    try:
        if existing_fields and fieldnames != existing_fields:
            _rewrite_atomically(args.output_file, fieldnames, existing_rows + [row])
        else:
            with open(args.output_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
                if not existing_fields:
                    writer.writeheader()
                writer.writerow(row)
        print(f"Successfully wrote results to: {args.output_file}")
    except OSError as e:
        print(f"Error writing to output file {args.output_file}: {e}", file=sys.stderr)


def _rewrite_atomically(path, fieldnames, rows):
    """Write header and rows to a temporary sibling of path, then os.replace it."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix='.' + os.path.basename(path) + '.', suffix='.tmp',
                               dir=directory)
    try:
        with os.fdopen(fd, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
            writer.writeheader()
            writer.writerows(rows)
        try:
            # mkstemp creates the file owner-only; keep the original's permissions.
            shutil.copymode(path, tmp)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def run(args):
    """Load the model and evaluate it on the test split."""
    
    # 1. Instantiate the DataModule
    dm = TFBSDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocess=args.preprocess,
        scaling_method=args.scaling_method
    )
    
    # Call setup to load and preprocess data (and apply scaling)
    dm.setup(stage="validate") 
    
    # Determine the expected input length from the processed data
    # Input shape: (Batch, Channels, Length) -> we need the Length dimension (index 2)
    # The BaseModel expects 'input_length' which, for the one-hot encoded sequence, 
    # is the sequence length (the last dimension of the tensor after permute).
    if dm.train_dataset:
        sample_input, _ = dm.train_dataset[0]
        # In TFBSDataModule, shape is (C, L), so L is sample_input.shape[1]
        input_length = sample_input.shape[1] 
        print(f"Inferred sequence length (input_length): {input_length}")
    else:
        print("Error: DataModule setup failed to create datasets.")
        return

    # 2. Load Model from Checkpoint
    model_class = get_model_class(args.model_name)
    print(f"Loading model '{args.model_name}' from checkpoint: {args.checkpoint_path}")

    # Use load_from_checkpoint. This automatically reads all hyperparameters 
    # (including input_length, classify, learning_rate) saved in the checkpoint file.
    # We pass the class reference but the actual parameters are loaded from the file.
    try:
        model = model_class.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            # We explicitly pass input_length in case the checkpoint was trained 
            # with a different sequence length than what is inferred now, 
            # though usually it should match.
            input_length=input_length 
        )
        model.eval() # Set model to evaluation mode
    except Exception as e:
        print(f"Failed to load model from checkpoint: {e}")
        print("Check if the checkpoint file exists and if the model class matches the one used during training.")
        return

    # 3. Instantiate the PyTorch Lightning Trainer
    # We use a minimal setup for validation/prediction
    trainer = pl.Trainer(
        accelerator="auto", 
        devices=1, 
        logger=False
    )

    # 4. Run Test (on the held-out test set, NOT the validation set)
    # trainer.test() uses test_step which evaluates on data never seen
    # during training or early stopping / checkpoint selection.
    print("\nRunning evaluation on the held-out test set...")
    results = trainer.test(model, datamodule=dm, verbose=True)

    print("\n--- Test Results ---")
    print(results)
    print("--------------------")

    # 5. Save every metric trainer.test() reports: test_loss, and for a
    # classifier also test_acc and test_auc.
    metrics = dict(results[0]) if results else {}
    if 'test_loss' in metrics:
        save_results(args, metrics)
    else:
        print(f"Warning: no 'test_loss' in the test results {metrics}; nothing saved.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run validation on a PyTorch Lightning model checkpoint.")
    
    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint file (.ckpt).')
    parser.add_argument('--model_name', type=str, default='FlexibleMLP',
                        help='The name of the model class used during training (e.g., FlexibleMLP, CNN, BPNet).')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the CSV file where the results will be appended.')

    # DataModule arguments (must match the DataModule's __init__)
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the data (CSV file or directory of tensors).')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for prediction.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers.')
    parser.add_argument('--preprocess', default=False, action=argparse.BooleanOptionalAction,
                        help='If set, preprocesses data from CSV; otherwise loads tensors from directory.')
    parser.add_argument('--scaling_method', type=str, default='standardize',
                        choices=['standardize', 'normalize', 'none'],
                        help="Label scaling method: 'standardize', 'normalize', or 'none'.")

    return parser


def main(argv=None):
    """Run from a list of command-line arguments, or from parsed arguments.

    ``main(args)`` with an argparse.Namespace is the signature of the
    former scripts/validate.py and still works.
    """
    if isinstance(argv, argparse.Namespace):
        return run(argv)
    return run(build_parser().parse_args(argv))


if __name__ == '__main__':
    main()
