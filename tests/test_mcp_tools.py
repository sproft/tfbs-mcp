"""Tests for MCP server tools that don't require real model checkpoints."""
import asyncio
import json
import os
import tempfile
import pytest
import torch
import numpy as np


class FakeModel(torch.nn.Module):
    """Minimal model that returns a scalar per sequence."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        self.input_length = 10
        self.classify = False

    def forward(self, x):
        # x: (N, 4, L) -> mean-pool -> linear -> (N, 1)
        return self.linear(x.mean(dim=2))


@pytest.fixture(autouse=True)
def _patch_registry(monkeypatch):
    """Inject a FakeModel into the server's model registry."""
    from tfbs.mcp import server
    fake = FakeModel()
    fake.eval()
    server._model_registry["test_model"] = {
        "model": fake,
        "model_type": "FakeModel",
        "device": "cpu",
        "input_length": 10,
        "classify": False,
    }
    server._ensure_imports()
    yield
    server._model_registry.pop("test_model", None)


def test_tfbs_mutagenesis_returns_scores():
    """Verify ISM returns scores with shape (1, 4, 10)."""
    from tfbs.mcp.server import tfbs_mutagenesis

    result = json.loads(asyncio.run(tfbs_mutagenesis(
        model_id="test_model",
        sequences=["ACGTACGTAC"],
    )))
    assert "scores" in result
    assert result["shape"] == [1, 4, 10]
    scores = np.array(result["scores"])
    assert scores.shape == (1, 4, 10)


def test_tfbs_mutagenesis_raw_outputs():
    """Verify ISM with raw_outputs returns original_scores and mutant_scores."""
    from tfbs.mcp.server import tfbs_mutagenesis

    result = json.loads(asyncio.run(tfbs_mutagenesis(
        model_id="test_model",
        sequences=["ACGTACGTAC"],
        raw_outputs=True,
    )))
    assert "original_scores" in result
    assert "mutant_scores" in result


def test_tfbs_mutagenesis_bad_model():
    """Verify ValueError for unknown model_id."""
    from tfbs.mcp.server import tfbs_mutagenesis

    with pytest.raises(ValueError, match="not loaded"):
        asyncio.run(tfbs_mutagenesis(
            model_id="nonexistent_model",
            sequences=["ACGTACGTAC"],
        ))


def test_tfbs_marginalize_with_consensus():
    """Verify marginalize returns delta_predictions and mean_delta."""
    from tfbs.mcp.server import tfbs_marginalize

    result = json.loads(asyncio.run(tfbs_marginalize(
        model_id="test_model",
        sequences=["ACGTACGTAC"],
        motif="ACGT",
    )))
    assert "delta_predictions" in result
    assert "mean_delta" in result
    assert isinstance(result["mean_delta"], float)


def test_tfbs_marginalize_with_start():
    """Verify marginalize accepts a start position."""
    from tfbs.mcp.server import tfbs_marginalize

    result = json.loads(asyncio.run(tfbs_marginalize(
        model_id="test_model",
        sequences=["ACGTACGTAC"],
        motif="AC",
        start=2,
    )))
    assert "delta_predictions" in result


def test_tfbs_marginalize_bad_model():
    """Verify ValueError for unknown model_id."""
    from tfbs.mcp.server import tfbs_marginalize

    with pytest.raises(ValueError, match="not loaded"):
        asyncio.run(tfbs_marginalize(
            model_id="nonexistent",
            sequences=["ACGT"],
            motif="AC",
        ))


def test_classify_metrics_roc():
    from tfbs.mcp.server import tfbs_classify_metrics

    result = json.loads(asyncio.run(tfbs_classify_metrics(
        positive_scores=[0.9, 0.8, 0.7, 0.6],
        negative_scores=[0.3, 0.2, 0.4, 0.1],
    )))
    assert "roc_auc" in result
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert "pr_auc" in result
    assert "num_positive" in result
    assert result["num_positive"] == 4
    assert result["num_negative"] == 4


def test_classify_metrics_perfect():
    from tfbs.mcp.server import tfbs_classify_metrics

    result = json.loads(asyncio.run(tfbs_classify_metrics(
        positive_scores=[1.0, 1.0, 1.0],
        negative_scores=[0.0, 0.0, 0.0],
    )))
    assert result["roc_auc"] == 1.0


def test_classify_metrics_empty():
    from tfbs.mcp.server import tfbs_classify_metrics

    result = json.loads(asyncio.run(tfbs_classify_metrics(
        positive_scores=[],
        negative_scores=[0.1, 0.2],
    )))
    assert "error" in result


# ---------------------------------------------------------------------------
# Pipeline tool tests
# ---------------------------------------------------------------------------

def test_prepare_dataset_splits():
    """Verify train+val+test counts sum correctly and RC doubles them."""
    from tfbs.mcp.server import tfbs_prepare_dataset

    with tempfile.TemporaryDirectory() as td:
        # Create synthetic data: 100 sequences of shape (100, 4, 20)
        seqs = torch.randn(100, 4, 20)
        labels = torch.randn(100, 1)
        seqs_path = os.path.join(td, "seqs.pt")
        labels_path = os.path.join(td, "labels.pt")
        torch.save(seqs, seqs_path)
        torch.save(labels, labels_path)

        # Without RC
        result = json.loads(asyncio.run(tfbs_prepare_dataset(
            sequences_path=seqs_path,
            labels_path=labels_path,
            output_dir=os.path.join(td, "split"),
        )))
        assert "error" not in result
        assert result["train_count"] + result["val_count"] + result["test_count"] == 100
        # Verify files exist
        for split in ("train", "val", "test"):
            assert os.path.exists(os.path.join(td, "split", split, "seqs.pt"))
            assert os.path.exists(os.path.join(td, "split", split, "labels.pt"))

        # With RC — should double each split
        result_rc = json.loads(asyncio.run(tfbs_prepare_dataset(
            sequences_path=seqs_path,
            labels_path=labels_path,
            output_dir=os.path.join(td, "split_rc"),
            add_rc=True,
        )))
        assert result_rc["train_count"] == result["train_count"] * 2
        assert result_rc["val_count"] == result["val_count"] * 2
        assert result_rc["test_count"] == result["test_count"] * 2


def test_prepare_dataset_small():
    """Verify that < 3 sequences returns an error, not a crash."""
    from tfbs.mcp.server import tfbs_prepare_dataset

    with tempfile.TemporaryDirectory() as td:
        seqs = torch.randn(2, 4, 20)
        labels = torch.randn(2, 1)
        torch.save(seqs, os.path.join(td, "seqs.pt"))
        torch.save(labels, os.path.join(td, "labels.pt"))

        result = json.loads(asyncio.run(tfbs_prepare_dataset(
            sequences_path=os.path.join(td, "seqs.pt"),
            labels_path=os.path.join(td, "labels.pt"),
            output_dir=os.path.join(td, "split"),
        )))
        assert "error" in result
        assert "at least 3" in result["error"]


def test_prepare_classification_balance():
    """Verify subsampling produces equal pos/neg counts."""
    from tfbs.mcp.server import tfbs_prepare_classification

    with tempfile.TemporaryDirectory() as td:
        pos = torch.randn(60, 4, 20)
        neg = torch.randn(40, 4, 20)
        torch.save(pos, os.path.join(td, "pos.pt"))
        torch.save(neg, os.path.join(td, "neg.pt"))

        result = json.loads(asyncio.run(tfbs_prepare_classification(
            positive_path=os.path.join(td, "pos.pt"),
            negative_path=os.path.join(td, "neg.pt"),
            output_dir=os.path.join(td, "combined"),
            balance=True,
        )))
        assert result["num_positive"] == 40  # subsampled to match neg
        assert result["num_negative"] == 40
        assert result["total"] == 80

        # Verify labels are correct
        labels = torch.load(os.path.join(td, "combined/labels.pt"), weights_only=True)
        assert labels.shape == (80, 1)
        assert float(labels.sum()) == 40.0  # 40 ones, 40 zeros


def test_prepare_classification_no_balance():
    """Verify unbalanced mode keeps all sequences."""
    from tfbs.mcp.server import tfbs_prepare_classification

    with tempfile.TemporaryDirectory() as td:
        pos = torch.randn(60, 4, 20)
        neg = torch.randn(40, 4, 20)
        torch.save(pos, os.path.join(td, "pos.pt"))
        torch.save(neg, os.path.join(td, "neg.pt"))

        result = json.loads(asyncio.run(tfbs_prepare_classification(
            positive_path=os.path.join(td, "pos.pt"),
            negative_path=os.path.join(td, "neg.pt"),
            output_dir=os.path.join(td, "combined"),
            balance=False,
        )))
        assert result["num_positive"] == 60
        assert result["num_negative"] == 40
        assert result["total"] == 100


def test_extract_loci_missing_bed():
    """Verify graceful error on missing BED file."""
    from tfbs.mcp.server import tfbs_extract_loci

    result = json.loads(asyncio.run(tfbs_extract_loci(
        bed_path="/nonexistent/peaks.bed",
        output_dir="/tmp/test_extract_missing",
    )))
    assert "error" in result
    assert "not found" in result["error"]


def test_predict_inverse_transform():
    """Verify inverse transform reverses standardization correctly."""
    from tfbs.mcp import server

    # Set up a model with known scaling params
    server._model_registry["test_scaled"] = {
        **server._model_registry["test_model"],
        "scaling_params": {
            "scaling_method": "standardize",
            "mean": 5.0,
            "std": 2.0,
        },
    }

    try:
        # Raw prediction (z-score space)
        raw = json.loads(asyncio.run(server.tfbs_predict(
            model_id="test_scaled",
            sequences=["ACGTACGTAC"],
            inverse_transform=False,
        )))
        # Inverse-transformed prediction
        inv = json.loads(asyncio.run(server.tfbs_predict(
            model_id="test_scaled",
            sequences=["ACGTACGTAC"],
            inverse_transform=True,
        )))

        raw_score = raw["scores"]
        inv_score = inv["scores"]

        # inverse = raw * std + mean
        if isinstance(raw_score, list):
            raw_score = raw_score[0] if isinstance(raw_score[0], (int, float)) else raw_score
        if isinstance(inv_score, list):
            inv_score = inv_score[0] if isinstance(inv_score[0], (int, float)) else inv_score
        expected = float(raw_score) * 2.0 + 5.0
        assert abs(float(inv_score) - expected) < 1e-4, (
            f"Expected {expected}, got {inv_score}"
        )
        assert inv.get("inverse_transformed") is True
    finally:
        server._model_registry.pop("test_scaled", None)


class FakeConvModel(torch.nn.Module):
    """Model with a Conv1d layer for testing conv_filters tool."""
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(4, 8, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(8, 1)
        self.input_length = 10
        self.classify = False

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


def test_conv_filters_with_conv():
    """Verify conv_filters works on a model with Conv1d layers."""
    from tfbs.mcp import server

    fake_conv = FakeConvModel()
    fake_conv.eval()
    server._model_registry["test_conv"] = {
        "model": fake_conv,
        "model_type": "FakeConvModel",
        "device": "cpu",
        "input_length": 10,
        "classify": False,
        "scaling_params": None,
    }

    try:
        with tempfile.TemporaryDirectory() as td:
            result = json.loads(asyncio.run(server.tfbs_conv_filters(
                model_id="test_conv",
                sequences=["ACGTACGTAC"],
                output_dir=td,
                top_n_filters=5,
            )))
            assert "error" not in result
            assert result["num_layers"] == 1
            assert result["num_filters_analyzed"] > 0
            assert os.path.exists(os.path.join(td, "filter_summary.json"))
    finally:
        server._model_registry.pop("test_conv", None)


def test_conv_filters_no_conv():
    """Verify conv_filters returns error for model without Conv1d."""
    from tfbs.mcp import server

    with tempfile.TemporaryDirectory() as td:
        result = json.loads(asyncio.run(server.tfbs_conv_filters(
            model_id="test_model",  # FakeModel has only Linear, no Conv1d
            sequences=["ACGTACGTAC"],
            output_dir=td,
        )))
        assert "error" in result
        assert "No Conv1d" in result["error"]


def test_prepare_dataset_loadable_by_datamodule():
    """Verify the output of prepare_dataset can be loaded by TFBSDataModule."""
    from tfbs.mcp.server import tfbs_prepare_dataset
    from tfbs.dataloaders.SeqDataset import TFBSDataModule

    with tempfile.TemporaryDirectory() as td:
        seqs = torch.randn(50, 4, 24)
        labels = torch.randn(50, 1)
        torch.save(seqs, os.path.join(td, "seqs.pt"))
        torch.save(labels, os.path.join(td, "labels.pt"))

        split_dir = os.path.join(td, "split")
        asyncio.run(tfbs_prepare_dataset(
            sequences_path=os.path.join(td, "seqs.pt"),
            labels_path=os.path.join(td, "labels.pt"),
            output_dir=split_dir,
        ))

        # TFBSDataModule should load this without errors
        dm = TFBSDataModule(
            data_path=split_dir,
            batch_size=8,
            scaling_method="none",
        )
        dm.setup()
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None
        x, y = dm.train_dataset[0]
        assert x.shape == (4, 24)
        assert y.shape == (1,)
