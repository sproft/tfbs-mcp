import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from tangermeme.io import extract_loci

import tfbs.nn.models as models
from tfbs.prediction import analysis, utils


def main() -> None:
    """
    Main function to orchestrate the ChIP-seq peak analysis workflow.
    """
    parser = argparse.ArgumentParser(
        description="Process ChIP-seq peak files, score them with a trained model, and generate sequence logos."
    )
    parser.add_argument(
        '--peak_file',
        type=Path,
        required=True,
        help='Path to the ChIP-seq peak file (BED format).'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Name of the model to use for scoring (must be a key in config.yml).'
    )
    parser.add_argument(
        '--config_file',
        type=Path,
        default=Path('config.yml'),
        help='Path to the configuration file.'
    )
    parser.add_argument(
        '--peak_window',
        type=int,
        default=200,
        help='Window size in bp for extracting sequences around peak centers.'
    )
    parser.add_argument(
        '--motif_window_size',
        type=int,
        default=70,
        help='Sliding window size for model input (motif scanning).'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('results'),
        help='Directory to save the output files.'
    )
    args = parser.parse_args()

    # --- 1. Load Configuration and Model ---
    print(f"Loading configuration from: {args.config_file}")
    config = utils.load_config(args.config_file)
    model_info = config['models'].get(args.model_name)

    if not model_info:
        raise ValueError(f"Model '{args.model_name}' not found in {args.config_file}. "
                         f"Available models: {', '.join(config['models'].keys())}")

    print(f"Loading model '{args.model_name}' from: {model_info['path']}")
    model = utils.load_model(model_info, models_module=models)
    print("Model loaded successfully.")

    # --- 2. Load and Prepare Sequence Data ---
    print(f"Processing peaks from: {args.peak_file}")
    peaks_df = pd.read_csv(
        args.peak_file,
        sep="\t",
        usecols=(0, 1, 2),
        names=['chrom', 'start', 'end']
    )

    print(f"Extracting DNA loci from genome: {config['genome_fasta_path']}")
    # Extract loci (one-hot encoded DNA windows)
    X = extract_loci(
        peaks_df,
        config['genome_fasta_path'],
        verbose=True,
        in_window=args.peak_window
    ).float()

    # Filter out sequences containing ambiguous bases (N)
    # A valid one-hot encoded base sums to 1. The total sum should equal the sequence length.
    initial_count = X.shape[0]
    X = X[X.sum(dim=(1, 2)) == X.shape[-1]]
    filtered_count = initial_count - X.shape[0]
    print(f"Removed {filtered_count} sequences with ambiguous bases.")
    print(f"Proceeding with {X.shape[0]} sequences.")

    # --- 3. Run Analysis and Save Results ---
    # Create a descriptive file prefix for output files
    peak_file_base = args.peak_file.stem.split('.')[0] # More robust way to get base name
    file_prefix = f"{args.output_dir}/{peak_file_base}_{args.model_name}"
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting analysis for model '{args.model_name}'...")
    analysis.process_sequences_and_save(
        X=X,
        model=model,
        model_name=args.model_name,
        file_prefix=file_prefix,
        motif_window_size=args.motif_window_size
    )
    print(f"\nAnalysis complete. Results saved with prefix: {file_prefix}")


if __name__ == "__main__":
    main()
