"""Tests for synthetic dataset creation and data module loading."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from tfbs.dataloaders.SeqDataset import TFBSDataModule


def create_artificial_tfbs_dataset(file_path, num_samples=1000, seq_length=14, motif="CACTTGA", motif_insertion_rate=0.5):
    """Generate an artificial TFBS dataset with random sequences and optional motif insertion."""
    nucleotides = ['A', 'C', 'G', 'T']
    sequences = []
    log2fc_values = []

    motif_length = len(motif)
    rev_comp_motif = motif.translate(str.maketrans('ACGT', 'TGCA'))[::-1]

    for _ in range(num_samples):
        seq = ''.join(np.random.choice(nucleotides, seq_length))

        if np.random.rand() < motif_insertion_rate:
            insert_pos = np.random.randint(0, seq_length - motif_length + 1)
            if np.random.rand() < 0.5:
                seq = seq[:insert_pos] + motif + seq[insert_pos + motif_length:]
            else:
                seq = seq[:insert_pos] + rev_comp_motif + seq[insert_pos + motif_length:]
            log2fc = np.random.uniform(1.0, 3.0)
        else:
            log2fc = np.random.uniform(-1.0, 1.0)

        sequences.append(seq)
        log2fc_values.append(log2fc)

    df = pd.DataFrame({'seq': sequences, 'log2FC': log2fc_values})
    df.to_csv(file_path, index=False)
    return df


@pytest.fixture
def dataset_csv(tmp_path):
    """Create a temporary CSV dataset for testing."""
    csv_path = str(tmp_path / "test_dataset.csv")
    create_artificial_tfbs_dataset(csv_path, num_samples=200, seq_length=14)
    return csv_path


def test_create_dataset_shape(tmp_path):
    csv_path = str(tmp_path / "test.csv")
    df = create_artificial_tfbs_dataset(csv_path, num_samples=100, seq_length=14)
    assert len(df) == 100
    assert list(df.columns) == ['seq', 'log2FC']
    assert all(len(s) == 14 for s in df['seq'])


def test_create_dataset_motif_insertion(tmp_path):
    csv_path = str(tmp_path / "test.csv")
    df = create_artificial_tfbs_dataset(csv_path, num_samples=500, motif_insertion_rate=1.0)
    # All sequences should have motif or reverse complement → high log2FC
    assert all(df['log2FC'] >= 1.0)


def test_create_dataset_no_motif(tmp_path):
    csv_path = str(tmp_path / "test.csv")
    df = create_artificial_tfbs_dataset(csv_path, num_samples=500, motif_insertion_rate=0.0)
    # No motifs → all log2FC between -1 and 1
    assert all(df['log2FC'] <= 1.0)
    assert all(df['log2FC'] >= -1.0)


def test_datamodule_from_csv(dataset_csv, tmp_path):
    """Test that TFBSDataModule can preprocess a CSV into tensors."""
    dm = TFBSDataModule(
        data_path=dataset_csv,
        batch_size=32,
        num_workers=1,
        preprocess=True,
        scaling_method=None,
    )
    dm.setup(stage="fit")
    assert dm.train_dataloader() is not None
    batch = next(iter(dm.train_dataloader()))
    X, y = batch[0], batch[1]
    assert X.ndim == 3  # (batch, channels=4, seq_len)
    assert X.shape[1] == 4  # one-hot encoded
    assert y.ndim in (1, 2)  # labels may be (batch,) or (batch, 1)
