import argparse
import os
import sys
import torch
import pytorch_lightning as pl
import csv

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


def save_results(args, loss_value):
    """Saves the prediction results and metadata to a CSV file."""
    
    # Define the output data row
    result_data = {
        'modeltype': args.model_name,
        'scaling_method': args.scaling_method,
        'dataset_path': os.path.basename(args.data_path),
        'test_loss': loss_value
    }

    fieldnames = ['modeltype', 'scaling_method', 'dataset_path', 'test_loss']
    
    # Check if the file exists to determine if we need to write the header
    file_exists = os.path.exists(args.output_file)
    
    try:
        with open(args.output_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write the header only if the file did not exist before
            if not file_exists:
                writer.writeheader()
                
            # Write the data row
            writer.writerow(result_data)
            
        print(f"\nSuccessfully wrote results to: {args.output_file}")
        
    except Exception as e:
        print(f"Error writing to output file {args.output_file}: {e}")


def main(args):
    """Main function to load the model and run validation."""
    
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

    # 5. Extract and Save Results
    test_loss_key = 'test_loss'
    test_loss = results[0].get(test_loss_key)

    if test_loss is not None:
        save_results(args, test_loss)
    else:
        print(f"Warning: Could not find '{test_loss_key}' in test results to save.")


if __name__ == '__main__':
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
    parser.add_argument('--preprocess', type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help='If set, preprocesses data from CSV; otherwise loads tensors from directory.')
    parser.add_argument('--scaling_method', type=str, default='standardize',
                        choices=['standardize', 'normalize', 'none'],
                        help="Label scaling method: 'standardize', 'normalize', or 'none'.")

    args = parser.parse_args()
    main(args)
