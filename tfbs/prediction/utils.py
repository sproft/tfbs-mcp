from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from logomaker import Logo, transform_matrix
from pathlib import Path
from torch.nn import Module

# Define a type alias for clarity
ModelInfo = Dict[str, Any]


def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(model_info: ModelInfo, models_module: Any) -> Module:
    """
    Loads a PyTorch model from a checkpoint file based on its type.

    Args:
        model_info: A dictionary containing the model's 'path' and 'type'.
        models_module: The imported module containing model classes (e.g., SimpleCNN).

    Returns:
        The loaded, frozen model in evaluation mode.
    """
    model_path = model_info['path']
    model_type = model_info['type']

    # Every model in tfbs.nn.models subclasses BaseModel and calls
    # save_hyperparameters(), so load_from_checkpoint reconstructs any of them
    # from the checkpoint's stored hyperparameters. Look the class up by name
    # instead of hardcoding the supported types.
    model_cls = getattr(models_module, model_type, None)
    if model_cls is None:
        available = [n for n in dir(models_module) if n[:1].isupper()]
        raise ValueError(
            f"Unknown model type '{model_type}' in config. Available: {available}"
        )
    model = model_cls.load_from_checkpoint(model_path)

    model.eval()
    model.freeze()
    return model


def onehot_to_seq(onehot: np.ndarray) -> str:
    """
    Converts a one-hot encoded DNA sequence back to a string of bases.
    Assumes shape (4, L).
    """
    bases = np.array(['A', 'C', 'G', 'T'])
    indices = np.argmax(onehot, axis=0)
    return ''.join(bases[indices])


def plot_sequence_logo(sequences: List[str], model_name: str, save_path: Path = None) -> None:
    """
    Generates and optionally saves an information content sequence logo.

    Args:
        sequences: A list of DNA sequences of the same length.
        model_name: The name of the model, for the plot title.
        save_path: The file path to save the plot. If None, plot is not saved.
    """
    if not sequences:
        print("Warning: Cannot generate a logo for an empty list of sequences.")
        return

    seq_len = len(sequences[0])
    counts_mat = np.zeros((seq_len, 4))
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}

    for seq in sequences:
        for i, base in enumerate(seq.upper()):
            if base in base_map:
                counts_mat[i, base_map[base]] += 1

    counts_df = pd.DataFrame(counts_mat, columns=list("ACGT"))

    # Convert counts to information content
    info_df = transform_matrix(
        counts_df,
        from_type='counts',
        to_type='information'
    )

    plt.figure(figsize=(12, 3))
    Logo(info_df)
    plt.title(f"Sequence Logo for Best Windows ({model_name})")
    plt.xlabel("Position")
    plt.ylabel("bits")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close() # Close the figure to free up memory


def generate_random_onehot(num_seqs: int, seq_len: int, seed: int = None) -> torch.Tensor:
    """
    Generates random one-hot encoded DNA sequences.

    Returns:
        A torch.Tensor of shape (num_seqs, 4, seq_len).
    """
    if seed is not None:
        np.random.seed(seed)
    
    indices = np.random.randint(0, 4, size=(num_seqs, seq_len))
    onehot = np.zeros((num_seqs, 4, seq_len), dtype=np.float32)
    
    for i in range(num_seqs):
        onehot[i, indices[i], np.arange(seq_len)] = 1.0
        
    return torch.from_numpy(onehot)
