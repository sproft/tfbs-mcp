import logging

import numpy as np
import pandas as pd
import torch
from torch.nn import Module
from typing import Tuple

from . import utils

logger = logging.getLogger(__name__)


#: Upper bound on the number of elements in any single activation of one forward pass.
#: Several CUDA kernels index with 32-bit integers and fail with "an illegal memory access
#: was encountered" once one tensor holds more than 2**31 elements.  Window-mode scoring of
#: 500 bp sequences in batches of 512 reached 3.3e9 to 4.4e9 elements in the pooled output of
#: the NKX2.1 VCNNBpnet models (num_channels x pool_output_size values per window).  A quarter
#: of the limit leaves a margin for kernels whose internal buffers are larger than their output.
MAX_ACTIVATION_ELEMENTS = 2 ** 29


try:  # sees every tensor an operator returns, in functional code and TorchScript models too
    from torch.utils._python_dispatch import TorchDispatchMode as _TorchDispatchMode
except ImportError:  # pragma: no cover - torch without dispatch modes; forward hooks are used
    _TorchDispatchMode = None

if _TorchDispatchMode is not None:
    class _OutputSizes(_TorchDispatchMode):
        """Records the number of elements of every tensor an operator returns."""

        def __init__(self):
            super().__init__()
            self.sizes = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in (out if isinstance(out, (tuple, list)) else (out,)):
                if isinstance(t, torch.Tensor):
                    self.sizes.append(t.numel())
            return out


def _output_sizes(model: Module, x: torch.Tensor) -> list:
    """Numbers of elements of the tensors produced in one forward pass of ``model`` on ``x``."""
    if _TorchDispatchMode is not None:
        with _OutputSizes() as recorder:
            model(x)
        return recorder.sizes
    sizes = []

    def _hook(_module, _inputs, output):
        for t in (output if isinstance(output, (tuple, list)) else (output,)):
            if torch.is_tensor(t):
                sizes.append(t.numel())

    handles = [m.register_forward_hook(_hook) for m in model.modules()]
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
    return sizes


def largest_activation_per_sample(model: Module, sample: torch.Tensor) -> int:
    """Largest number of elements per input in the input or any tensor of one forward pass.

    Runs ``model`` on 2 and on 4 copies of ``sample[:1]`` and records the size of every
    tensor an operator returns (module outputs only, through forward hooks, if this torch
    has no dispatch modes).  The size per input of a tensor is its growth between the two
    passes, halved, so that weights and other tensors that do not grow with the batch do not
    count; if the two passes record different numbers of tensors, the largest tensor of the
    second pass divided by 4 is used instead.  This assumes activations grow linearly with
    the batch.  Both passes run in eval mode without gradients, and every submodule's mode
    is restored afterwards.  Returns at least 1.
    """
    one = sample[:1]
    x2 = one.expand(2, *one.shape[1:]).contiguous()
    x4 = one.expand(4, *one.shape[1:]).contiguous()
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            s2 = _output_sizes(model, x2)
            s4 = _output_sizes(model, x4)
    finally:
        for m, training in modes:
            m.training = training
    if s4 and len(s2) == len(s4):
        per = max(-(-(b - a) // 2) for a, b in zip(s2, s4))
    else:
        per = -(-max(s4, default=0) // 4)
    return max(1, per, one.numel())


def safe_forward_batch_size(model: Module, sample: torch.Tensor,
                            max_elements: int = MAX_ACTIVATION_ELEMENTS) -> int:
    """Largest batch size whose activations stay at or below ``max_elements`` elements each.

    Uses :func:`largest_activation_per_sample` on ``sample[:1]``, so it holds for
    activations that grow linearly with the batch.  Returns at least 1.
    """
    return max(1, int(max_elements) // largest_activation_per_sample(model, sample))


def forward_in_chunks(model: Module, x: torch.Tensor, max_batch: int) -> torch.Tensor:
    """``model(x)``, computed in slices of at most ``max_batch`` inputs along dim 0 and concatenated.

    The model must return one tensor whose first dimension is the batch; the output of a
    slice that has no dimensions left (a batch of one squeezed to a scalar) counts as one
    row.  Results equal a single pass up to the floating-point rounding of the kernels
    chosen for each batch size.
    """
    max_batch = max(1, int(max_batch))
    if x.shape[0] <= max_batch:
        return model(x)
    outs = [model(x[i:i + max_batch]) for i in range(0, x.shape[0], max_batch)]
    return torch.cat([o.reshape(1) if o.dim() == 0 else o for o in outs], dim=0)


def find_best_windows_fast(
    X: torch.Tensor,
    model: Module,
    window_size: int = 70,
    processing_batch_size: int = 128,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    max_activation_elements: int = MAX_ACTIVATION_ELEMENTS,
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
                               Each sequence contributes L - window_size + 1
                               windows; the windows of a chunk are scored in
                               forward passes small enough that no activation
                               exceeds max_activation_elements, so the scores do
                               not depend on this value beyond floating-point
                               rounding.
        device: The device to run computations on ('cuda' or 'cpu').
        max_activation_elements: Largest number of elements any activation may
                                 hold in one forward pass (default 2**29).

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

    logger.info("Processing %d sequences on device '%s'...", N, device)
    max_windows = None
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

            # 3. Score all windows of the chunk, in forward passes small enough that no
            # activation reaches 2**31 elements (CUDA's 32-bit index limit).
            if max_windows is None:
                max_windows = safe_forward_batch_size(model, windows_batch[:1], max_activation_elements)
                logger.debug("Scoring at most %d windows per forward pass.", max_windows)
            all_scores_chunk = forward_in_chunks(model, windows_batch, max_windows)

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
    logger.info("Finding best windows for model: %s", model_name)
    best_windows, best_scores, all_scores, mean_scores = find_best_windows_fast(
        X, model, window_size=motif_window_size
    )

    # Convert one-hot encoded windows back to DNA sequences
    seqs = [utils.onehot_to_seq(win) for win in best_windows]

    # --- Save results to CSV ---
    csv_path = f"{file_prefix}_sequences.csv"
    logger.info("Saving sequences and scores to: %s", csv_path)
    seqs_df = pd.DataFrame({
        'sequence': seqs,
        'max_score': best_scores,
        'mean_score': mean_scores,
        "all_scores": all_scores.tolist()
    })
    seqs_df.to_csv(csv_path, index=False)

    # --- Plot and save sequence logo ---
    logo_path = f"{file_prefix}_logo.png"
    logger.info("Saving sequence logo to: %s", logo_path)
    utils.plot_sequence_logo(seqs, model_name, save_path=logo_path)
