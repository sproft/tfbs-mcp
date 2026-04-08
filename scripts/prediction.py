"""
Sequence Analysis and Motif Discovery Script

This script processes DNA sequences from a ChIP-seq peak file using a trained
PyTorch model. For each sequence, it identifies the window with the highest
model score, saves these "best" windows and their scores to a CSV file, and
can optionally generate a sequence logo for the discovered motifs.

Key functionalities:
- Load a trained PyTorch model for sequence scoring.
- Extract DNA sequences (loci) from a genome based on a BED file of peaks.
- Efficiently scan sequences to find the highest-scoring window using batch processing
  to manage GPU memory.
- Convert one-hot encoded sequences to DNA strings.
- Save results, including sequences and scores, to a CSV file.
- Generate and save sequence logo plots from the best-scoring windows.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from logomaker import Logo
from tangermeme.io import extract_loci
import pyfaidx  # Import pyfaidx


def extract_best_windows(
    x_sequences: torch.Tensor,
    model: torch.nn.Module,
    window_size: int = 70,
    batch_size: int = 128,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the window with the highest model score for each sequence in a batch.

    This function processes sequences in chunks to manage VRAM usage, making it
    suitable for large datasets. It balances performance and memory efficiency.

    Args:
        x_sequences (torch.Tensor): A tensor of one-hot encoded sequences with
                                    shape (N, C, L), where N is the number of
                                    sequences, C is the number of channels (e.g., 4
                                    for DNA), and L is the sequence length.
        model (torch.nn.Module): The PyTorch model to use for scoring. It should
                                 accept a batch of windows of shape (B, C, window_size).
        window_size (int): The size of the sliding window.
        batch_size (int): The number of sequences to process in a single chunk.
                                  Lower this value if you encounter out-of-memory errors.
        device (str): The device to run computations on ('cuda' or 'cpu').

    Returns:
        A tuple containing:
        - np.ndarray: The best-scoring windows, shape (N, C, window_size).
        - np.ndarray: The scores of the best windows, shape (N,).
        - np.ndarray: The scores of all windows for each sequence, shape (N, num_windows).
        - np.ndarray: The mean score of all windows for each sequence, shape (N,).
        - np.ndarray: The start indices of the best windows within their original sequence, shape (N,).
    """
    model.eval()
    model.to(device)

    n_sequences, n_channels, seq_len = x_sequences.shape
    num_windows = seq_len - window_size + 1

    # Lists to store results from each processed chunk
    all_results = {
        "best_windows": [],
        "best_scores": [],
        "all_scores": [],
        "mean_scores": [],
        "best_indices": [],
    }

    with torch.no_grad():
        for i in range(0, n_sequences, batch_size):
            x_chunk = x_sequences[i:i + batch_size].to(device)
            chunk_n = x_chunk.shape[0]

            # 1. Create a view of all possible windows for the chunk
            windows = x_chunk.unfold(dimension=2, size=window_size, step=1)
            windows_permuted = windows.permute(0, 2, 1, 3)
            windows_batch = windows_permuted.reshape(-1, n_channels, window_size)

            # 2. Score all windows in a single pass for the chunk
            scores_chunk = model(windows_batch).view(chunk_n, num_windows)

            # 3. Find the best score and index for each sequence in the chunk
            best_scores_chunk, best_indices_chunk = torch.max(scores_chunk, dim=1)

            # 4. Gather the best windows using the indices
            best_indices_expanded = best_indices_chunk.view(chunk_n, 1, 1, 1).expand(-1, n_channels, -1, window_size)
            best_windows_tensor_chunk = windows.gather(2, best_indices_expanded).squeeze(2)

            # 5. Calculate mean scores
            mean_scores_chunk = torch.mean(scores_chunk, dim=1)

            # 6. Store results on CPU to free up GPU memory
            all_results["best_windows"].append(best_windows_tensor_chunk.cpu())
            all_results["best_scores"].append(best_scores_chunk.cpu())
            all_results["all_scores"].append(scores_chunk.cpu())
            all_results["mean_scores"].append(mean_scores_chunk.cpu())
            all_results["best_indices"].append(best_indices_chunk.cpu())

    # Concatenate results from all chunks into final numpy arrays
    best_windows_np = torch.cat(all_results["best_windows"]).numpy()
    best_scores_np = torch.cat(all_results["best_scores"]).numpy()
    all_scores_np = torch.cat(all_results["all_scores"]).numpy()
    mean_scores_np = torch.cat(all_results["mean_scores"]).numpy()
    best_indices_np = torch.cat(all_results["best_indices"]).numpy()

    return best_windows_np, best_scores_np, all_scores_np, mean_scores_np, best_indices_np


def onehot_to_seq(onehot_matrix: np.ndarray) -> str:
    """
    Convert a one-hot encoded DNA sequence to a string.

    Args:
        onehot_matrix (np.ndarray): A (4, L) numpy array representing the one-hot
                                    encoded sequence.

    Returns:
        str: The DNA sequence as a string (e.g., "ACGT...").
    """
    bases = np.array(['A', 'C', 'G', 'T'])
    indices = np.argmax(onehot_matrix, axis=0)
    return "".join(bases[indices])


def plot_sequence_logo(sequences: List[str], title: str, save_path: Path) -> None:
    """
    Generate and save an information content sequence logo from a list of sequences.

    Args:
        sequences (List[str]): A list of DNA sequences of the same length.
        title (str): The title for the plot.
        save_path (Path): The file path to save the plot image.
    """
    if not sequences:
        print("Warning: Cannot generate sequence logo from an empty list of sequences.")
        return

    counts_df = logomaker.alignment_to_matrix(sequences, to_type='counts')
    info_df = logomaker.transform_matrix(counts_df, from_type='counts', to_type='information')

    info_df = info_df.astype(float)

    plt.figure(figsize=(12, 3))
    logo = Logo(info_df)
    logo.ax.set_title(title)
    logo.ax.set_ylabel("bits")
    logo.ax.set_xlabel("Position")
    plt.ylim(0, 1)
    plt.tight_layout()
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Sequence logo saved to {save_path}")


def analyze_and_save_sequences(
    sequences: torch.Tensor,
    peaks_info: pd.DataFrame,
    model: torch.nn.Module,
    model_name: str,
    output_dir: Path,
    peak_name: str,
    window_size: int = 70,
    generate_logo: bool = True
) -> None:
    """
    Run the full analysis pipeline for a given model and set of sequences.

    Args:
        sequences (torch.Tensor): The input sequences.
        peaks_info (pd.DataFrame): DataFrame with 'chrom', 'start', 'end' for each sequence.
        model (torch.nn.Module): The trained model for scoring.
        model_name (str): The name of the model (for file naming).
        output_dir (Path): Directory to save the output files.
        peak_name (str): A descriptive name from the input peak file.
        window_size (int): The sliding window size for the model.
        generate_logo (bool): Whether to generate and save a sequence logo plot.
    """
    print(f"Processing model: {model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_windows, best_scores, all_scores, mean_scores, best_indices = extract_best_windows(
        sequences, model, window_size=window_size
    )

    dna_sequences = [onehot_to_seq(window) for window in best_windows]

    # Calculate the genomic coordinates of the best-scoring windows
    peak_centers = (peaks_info['start'] + peaks_info['end']) // 2
    # The start coordinate of the larger window extracted by extract_loci
    extracted_seq_starts = peak_centers - (sequences.shape[2] // 2)
    # The start coordinate of the best window relative to the chromosome
    best_window_starts = extracted_seq_starts.to_numpy() + best_indices
    best_window_ends = best_window_starts + window_size

    # Create a string representation for the original peak position
    peak_position_str = peaks_info['chrom'].astype(str) + ':' + \
                        best_window_starts.astype(str) + '-' + \
                        best_window_ends.astype(str)

    # Save sequences and scores to a CSV file
    results_df = pd.DataFrame({
        'chrom': peaks_info['chrom'],
        'start': peaks_info['start'],
        'end': peaks_info['end'],
        'best_peak': peak_position_str,
        'sequence': dna_sequences,
        'max_score': best_scores,
        'mean_score': mean_scores,
        "all_scores": all_scores.tolist()
    })
    csv_path = output_dir / f"sequences_{peak_name}_{model_name}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Plot and save sequence logo
    if generate_logo:
        logo_path = output_dir / "logos" / f"logo_{peak_name}_{model_name}.png"
        plot_title = f"Sequence Logo for {model_name} on {peak_name}"
        plot_sequence_logo(dna_sequences, plot_title, logo_path)


def main() -> None:
    """
    Main function to parse arguments and run the analysis pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Analyze ChIP-seq peaks to find high-scoring sequence motifs using a trained model."
    )
    parser.add_argument('--peak_file', type=Path, required=True,
                        help='Path to the ChIP-seq peak file in BED format.')
    parser.add_argument('--genome_file', type=Path, required=True,
                        help='Path to the reference genome file in FASTA format.')
    parser.add_argument('--model_path', type=Path, required=True,
                        help='Path to the trained PyTorch Lightning model checkpoint.')
    parser.add_argument('--model_type', type=str, required=True,
                        help='Name of the model class to load (e.g., "SimpleCNN").')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='Directory to save output CSV files and sequence logos.')
    parser.add_argument('--peak_window', type=int, default=200,
                        help='Total window size for extracting sequences around peak summits.')
    parser.add_argument('--window_size', type=int, default=70,
                        help='Sliding window size for model input.')
    
    args = parser.parse_args()

    import tfbs.nn.models as models

    # --- Load Model ---
    print(f"Loading model '{args.model_type}' from '{args.model_path}'")
    try:
        model_cls = getattr(models, args.model_type)
        model = model_cls.load_from_checkpoint(args.model_path)
        model.eval()
        if hasattr(model, "freeze"):
            model.freeze()
        print("Model loaded successfully.")
    except AttributeError:
        print(f"Error: Model type '{args.model_type}' not found in the models module.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        sys.exit(1)

    # --- Load and Process Peaks ---
    print(f"Processing peaks from '{args.peak_file}'")
    try:
        original_peaks_df = pd.read_csv(
            args.peak_file, sep="\t", usecols=[0, 1, 2],
            names=['chrom', 'start', 'end'], header=None
        )
    except FileNotFoundError:
        print(f"Error: Peak file not found at '{args.peak_file}'")
        sys.exit(1)

    # --- Pre-filter Peaks to Match extract_loci Logic ---
    print("Loading genome for sequence validation...")
    try:
        genome = pyfaidx.Fasta(str(args.genome_file))
        chrom_lengths = {name: len(seq) for name, seq in genome.items()}
    except pyfaidx.FastaIndexingError:
        print(f"Error: Could not read or index genome file '{args.genome_file}'.")
        print("Please ensure it's a valid FASTA file and the .fai index is present or buildable.")
        sys.exit(1)

    in_window = args.peak_window
    in_width = in_window // 2
    
    valid_rows = []
    dropped_count = 0
    
    for index, row in original_peaks_df.iterrows():
        chrom = row['chrom']
        start = row['start']
        end = row['end']

        if chrom not in chrom_lengths:
            # This chromosome from the BED file is not in the FASTA file.
            dropped_count += 1
            continue 

        chrom_len = chrom_lengths[chrom]
        mid = start + (end - start) // 2

        # Replicate extract_loci's exact window calculation
        # (based on in_window, 0 jitter, and no out_window)
        seq_start = mid - in_width
        seq_end = mid + in_width + (in_window % 2) # This handles odd-sized windows

        # Replicate extract_loci's exact filtering condition
        # The 'continue' (filter out) condition is:
        # if seq_start < 0 or seq_end >= chrom_len:
        #
        # So the 'keep' condition is:
        if seq_start >= 0 and seq_end < chrom_len:
            valid_rows.append(row)
        else:
            # This locus falls off the edge of the chromosome.
            dropped_count += 1

    peaks_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    
    print(f"Filtered {len(original_peaks_df)} original peaks down to {len(peaks_df)} valid loci.")
    print(f"({dropped_count} loci were dropped for being off-chromosome or not in FASTA).")
    
    if len(peaks_df) == 0:
        print("Error: No valid loci remaining after filtering. Exiting.")
        genome.close()
        sys.exit(1)

    # --- Extract Sequences ---
    print("Extracting sequences from genome...")
    sequences_tensor = extract_loci(
        peaks_df,
        genome,  # Pass the loaded pyfaidx.Fasta object directly
        verbose=True,
        in_window=args.peak_window
    ).float()

    # We are done with the genome file
    genome.close()

    # This assertion should now pass, as we pre-filtered peaks_df
    # using the same logic as extract_loci.
    print(f"Successfully extracted {len(sequences_tensor)} sequences.")
    
    # We remove the old, broken filtering block
    
    assert len(sequences_tensor) == len(peaks_df), (
        f"Sequence extraction length mismatch AFTER pre-filtering: "
        f"got {len(sequences_tensor)}, expected {len(peaks_df)}. "
        f"This should not happen."
    )

    # Filter out sequences containing ambiguous bases (N)
    valid_mask = sequences_tensor.sum(dim=(1, 2)) == sequences_tensor.shape[-1]
    filtered_sequences = sequences_tensor[valid_mask]
    filtered_peaks_df = peaks_df[valid_mask.numpy()].copy().reset_index(drop=True)
    print(f"Retained {len(filtered_sequences)} of {len(sequences_tensor)} sequences after filtering.")

    # --- Run Analysis ---
    peak_name = args.peak_file.stem.split('.')[-2] if '.' in args.peak_file.stem else args.peak_file.stem
    analyze_and_save_sequences(
        sequences=filtered_sequences,
        peaks_info=filtered_peaks_df,
        model=model,
        model_name=args.model_type,
        output_dir=args.output_dir,
        peak_name=peak_name,
        window_size=args.window_size
    )

    print("Analysis complete.")


if __name__ == "__main__":
    main()