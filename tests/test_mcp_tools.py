"""Tests for MCP server tools that don't require real model checkpoints."""
import asyncio
import json
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
