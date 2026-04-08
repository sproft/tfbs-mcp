import numpy as np
import pandas as pd
import torch
from torch.nn import Module
from typing import Tuple

from . import utils


def find_best_windows_fast(
    X: torch.Tensor,
    model: Module,
    window_size: int = 70,
    processing_batch_size: int = 128,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each sequence in X, efficiently find the window with the highest model score.

    This version processes sequences in chunks to manage memory usage, making it
    suitable for large datasets. It uses a vectorized approach within each chunk for speed.

    Args:
        X: Input tensor of shape (N, C, L), where N is number of sequences,
           C is channels (4 for one-hot DNA), and L is sequence length.
        model: The model used to score windows.
        window_size: The size of the sliding window.
        processing_batch_size: The number of sequences to process at a time.
                               Lower this number if you run out of memory.
        device: The device to run computations on ('cuda' or 'cpu').

    Returns:
        A tuple containing:
            - np.ndarray: The best windows, shape (N, C, window_size).
            - np.ndarray: The scores of the best windows, shape (N,).
            - np.ndarray: The scores of all windows for each sequence.
            - np.ndarray: The mean score of all windows for each sequence.
    """
    model.eval()
    model = model.to(device)

    N, C, L = X.shape
    num_windows = L - window_size + 1

    final_best_windows, final_best_scores = [], []
    final_all_scores, final_mean_scores = [], []

    print(f"Processing {N} sequences on device '{device}'...")
    with torch.no_grad():
        for i in range(0, N, processing_batch_size):
            X_chunk = X[i:i + processing_batch_size].to(device)
            chunk_N = X_chunk.shape[0]

            # 1. Create all sliding windows for the chunk in a vectorized way.
            # `unfold` creates a view of the tensor with an added dimension for windows.
            # Shape: (chunk_N, C, num_windows, window_size)
            windows = X_chunk.unfold(dimension=2, size=window_size, step=1)

            # 2. Reshape for batch processing by the model.
            # We want a single batch of all windows from all sequences in the chunk.
            # (chunk_N, C, num_windows, window_size) -> (chunk_N, num_windows, C, window_size)
            windows_permuted = windows.permute(0, 2, 1, 3)
            # -> (chunk_N * num_windows, C, window_size)
            windows_batch = windows_permuted.reshape(-1, C, window_size)

            # 3. Score all windows in the batch.
            all_scores_chunk = model(windows_batch)

            # 4. Reshape scores back to match the original chunk structure.
            # Shape: (chunk_N, num_windows)
            all_scores_chunk = all_scores_chunk.view(chunk_N, num_windows)

            # 5. Find the best score and its index for each sequence in the chunk.
            best_scores_chunk, best_indices_chunk = torch.max(all_scores_chunk, dim=1)

            # 6. Use the indices to gather the corresponding best windows.
            # `gather` selects windows along dim=2 using the indices from `best_indices_chunk`.
            best_indices_expanded = best_indices_chunk.view(chunk_N, 1, 1, 1).expand(-1, C, -1, window_size)
            best_windows_tensor_chunk = windows.gather(2, best_indices_expanded).squeeze(2)

            # 7. Calculate mean scores for the chunk.
            mean_scores_chunk = torch.mean(all_scores_chunk, dim=1)

            # 8. Move results to CPU to free up GPU memory for the next chunk.
            final_best_windows.append(best_windows_tensor_chunk.cpu())
            final_best_scores.append(best_scores_chunk.cpu())
            final_all_scores.append(all_scores_chunk.cpu())
            final_mean_scores.append(mean_scores_chunk.cpu())

    # Concatenate results from all chunks into single NumPy arrays.
    best_windows_np = torch.cat(final_best_windows, dim=0).numpy()
    best_scores_np = torch.cat(final_best_scores, dim=0).numpy()
    all_scores_np = torch.cat(final_all_scores, dim=0).numpy()
    mean_scores_np = torch.cat(final_mean_scores, dim=0).numpy()

    return best_windows_np, best_scores_np, all_scores_np, mean_scores_np


def process_sequences_and_save(
    X: torch.Tensor,
    model: Module,
    model_name: str,
    file_prefix: str,
    motif_window_size: int = 70
) -> None:
    """
    Orchestrates the analysis: finds best windows, saves them to a CSV file,
    and plots a sequence logo.

    Args:
        X: The input tensor of one-hot encoded sequences.
        model: The PyTorch model to use for scoring.
        model_name: The name of the model, used for titling plots.
        file_prefix: The base path and name for output files.
        motif_window_size: The size of the sliding window for motif discovery.
    """
    print(f"Finding best windows for model: {model_name}")
    best_windows, best_scores, all_scores, mean_scores = find_best_windows_fast(
        X, model, window_size=motif_window_size
    )

    # Convert one-hot encoded windows back to DNA sequences
    seqs = [utils.onehot_to_seq(win) for win in best_windows]

    # --- Save results to CSV ---
    csv_path = f"{file_prefix}_sequences.csv"
    print(f"Saving sequences and scores to: {csv_path}")
    seqs_df = pd.DataFrame({
        'sequence': seqs,
        'max_score': best_scores,
        'mean_score': mean_scores,
        "all_scores": all_scores.tolist()
    })
    seqs_df.to_csv(csv_path, index=False)

    # --- Plot and save sequence logo ---
    logo_path = f"{file_prefix}_logo.png"
    print(f"Saving sequence logo to: {logo_path}")
    utils.plot_sequence_logo(seqs, model_name, save_path=logo_path)
