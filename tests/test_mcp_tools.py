"""Tests for MCP server tools that don't require real model checkpoints."""
import ast
import asyncio
import csv
import importlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import torch
import numpy as np

from tests.fake_pipeline import fake_pipelines


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
def _patch_registry(monkeypatch, tmp_path):
    """Inject a FakeModel into the server's model registry.

    Default outputs of the analysis tools go to tmp_path, not the checkout.
    """
    from tfbs.mcp import server
    monkeypatch.setattr(server, "_results_root", lambda: tmp_path / "results")
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
    """An unknown model_id comes back as a JSON error, not an exception."""
    from tfbs.mcp.server import tfbs_mutagenesis

    result = json.loads(asyncio.run(tfbs_mutagenesis(
        model_id="nonexistent_model",
        sequences=["ACGTACGTAC"],
    )))
    assert "not loaded" in result["error"]
    assert "tfbs_load_model" in result["fix"]


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
    """An unknown model_id comes back as a JSON error, not an exception."""
    from tfbs.mcp.server import tfbs_marginalize

    result = json.loads(asyncio.run(tfbs_marginalize(
        model_id="nonexistent",
        sequences=["ACGT"],
        motif="AC",
    )))
    assert "not loaded" in result["error"]
    assert result["fix"]


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

        # With RC: should double each split
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


# ---------------------------------------------------------------------------
# Helpers for the server tests below
# ---------------------------------------------------------------------------

def _call_via_mcp(name, arguments):
    """Call a tool through the MCP tool manager, the way a client does."""
    from tfbs.mcp import server

    out = asyncio.run(server.mcp.call_tool(name, arguments))
    content = out[0] if isinstance(out, tuple) else out
    return json.loads(content[0].text)


def _register(model_id, model, input_length):
    from tfbs.mcp import server

    model.eval()
    server._model_registry[model_id] = {
        "model": model,
        "model_type": type(model).__name__,
        "device": "cpu",
        "input_length": input_length,
        "classify": False,
        "scaling_params": None,
    }


class FakeSlurm:
    """Stands in for sbatch, squeue, sacct and wandb on a machine without them."""

    def __init__(self, monkeypatch, available=("sbatch", "fimo"), sweep_id="abc123", job_id="4242"):
        from tfbs.mcp import server

        self.calls = []
        self.sweep_id = sweep_id
        self.job_id = job_id
        monkeypatch.setattr(
            server.shutil, "which",
            lambda name, *a, **k: f"/usr/bin/{name}" if name in available else None,
        )
        monkeypatch.setattr(server.subprocess, "run", self._run)
        fake_pipelines(monkeypatch, self._run)

    def _run(self, cmd, *args, **kwargs):
        self.calls.append({"cmd": list(cmd), "cwd": kwargs.get("cwd")})
        if cmd[0] == "wandb":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="", stderr=f"wandb: Created sweep with ID: {self.sweep_id}\n")
        if cmd[0] == "sbatch":
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"Submitted batch job {self.job_id}\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def sbatch_calls(self):
        return [c for c in self.calls if c["cmd"][0] == "sbatch"]


def _no_binaries(monkeypatch):
    from tfbs.mcp import server

    monkeypatch.setattr(server.shutil, "which", lambda name, *a, **k: None)


def test_exactly_33_tools_are_registered():
    from tfbs.mcp import server
    import tfbs.mcp.foldx_tools  # noqa: F401

    assert len(asyncio.run(server.mcp.list_tools())) == 33


# ---------------------------------------------------------------------------
# tfbs_seqlets
# ---------------------------------------------------------------------------

SEQLET_COLUMNS = ["example_idx", "start", "end", "attribution", "p-value"]
SEQS_30 = ["ACGTACGTACGTACGTACGTACGTACGTAC", "TTGACTCAGGCATTGACTCAGGCATTGACT"]


class TinyConvModel(torch.nn.Module):
    """Conv1d, ReLU, global average and Linear: small enough for DeepLiftShap."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv1d(4, 4, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(self.pool(self.relu(self.conv(x))).squeeze(-1))


@pytest.fixture
def tiny_conv():
    from tfbs.mcp import server

    _register("tiny_conv", TinyConvModel(), 30)
    yield "tiny_conv"
    server._model_registry.pop("tiny_conv", None)


def _seqlet_frame(n):
    import pandas as pd

    return pd.DataFrame({
        "example_idx": [i % 2 for i in range(n)],
        "start": [i % 20 for i in range(n)],
        "end": [i % 20 + 5 for i in range(n)],
        "attribution": [1.0 + i for i in range(n)],
        "p-value": [1e-4 * (i + 1) for i in range(n)],
    }, columns=SEQLET_COLUMNS)


def _patch_seqlet_caller(monkeypatch, frame_or_error):
    """Replace tangermeme's attribution and seqlet calls with fixed outputs."""
    dls = importlib.import_module("tangermeme.deep_lift_shap")
    seqlet = importlib.import_module("tangermeme.seqlet")
    monkeypatch.setattr(dls, "deep_lift_shap", lambda model, X, **kw: torch.zeros_like(X))

    def fake_recursive_seqlets(*args, **kwargs):
        if isinstance(frame_or_error, Exception):
            raise frame_or_error
        return frame_or_error

    monkeypatch.setattr(seqlet, "recursive_seqlets", fake_recursive_seqlets)


def test_seqlets_rows_come_from_the_dataframe_and_are_capped(monkeypatch, tiny_conv):
    from tfbs.mcp import server

    _patch_seqlet_caller(monkeypatch, _seqlet_frame(250))
    result = json.loads(asyncio.run(server.tfbs_seqlets(model_id=tiny_conv, sequences=SEQS_30)))

    assert result["num_seqlets"] == 250
    assert result["truncated"] is True
    assert len(result["seqlets"]) == 200
    assert result["seqlets"][0] == {
        "example_idx": 0, "start": 0, "end": 5, "attribution": 1.0, "p-value": 1e-4,
    }
    assert result["seqlets"][199]["attribution"] == 200.0


def test_seqlets_reports_an_empty_result(monkeypatch, tiny_conv):
    from tfbs.mcp import server

    _patch_seqlet_caller(monkeypatch, _seqlet_frame(0))
    result = json.loads(asyncio.run(server.tfbs_seqlets(model_id=tiny_conv, sequences=SEQS_30)))
    files = result.pop("files")
    note = result.pop("note")
    assert result == {"num_seqlets": 0, "seqlets": [], "truncated": False}
    assert "No seqlets were called" in note and "significance threshold" in note
    lines = Path(files["seqlets"]).read_text().splitlines()
    assert lines == ["	".join(SEQLET_COLUMNS)]


def test_seqlets_survives_a_seqlet_caller_error(monkeypatch, tiny_conv):
    from tfbs.mcp import server

    _patch_seqlet_caller(monkeypatch, ValueError("recursive_seqlets requires X to be non-empty"))
    result = json.loads(asyncio.run(server.tfbs_seqlets(model_id=tiny_conv, sequences=SEQS_30)))
    assert result["num_seqlets"] == 0
    assert result["seqlets"] == []
    assert "ValueError" in result["note"] and "non-empty" in result["note"]
    _patch_seqlet_caller(monkeypatch, ZeroDivisionError("division by zero"))
    result = json.loads(asyncio.run(server.tfbs_seqlets(model_id=tiny_conv, sequences=SEQS_30)))
    assert result["num_seqlets"] == 0 and "ZeroDivisionError" in result["note"]


def test_seqlets_return_tangermeme_warnings_and_pass_the_seed(monkeypatch, tiny_conv):
    import warnings as _warnings
    from tfbs.mcp import server

    dls = importlib.import_module("tangermeme.deep_lift_shap")
    seen = {}

    def warning_dls(model, X, **kw):
        seen.update(kw)
        for k in range(3):
            _warnings.warn(f"Convergence deltas too high: tensor([0.{k}, 0.7])", RuntimeWarning)
        return torch.zeros_like(X)

    _patch_seqlet_caller(monkeypatch, _seqlet_frame(2))
    monkeypatch.setattr(dls, "deep_lift_shap", warning_dls)
    result = json.loads(asyncio.run(server.tfbs_seqlets(
        model_id=tiny_conv, sequences=SEQS_30, seed=7)))
    assert seen["random_state"] == 7
    assert result["warnings"] == [
        {"message": "RuntimeWarning: Convergence deltas too high", "count": 3}]
    assert "note" not in result


class PlantedMotifModel(torch.nn.Module):
    """Max over positions of a Conv1d that matches CACTTG exactly (score 6)."""

    MOTIF = "CACTTG"

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 1, kernel_size=len(self.MOTIF))
        with torch.no_grad():
            self.conv.weight.fill_(-1.0)
            for k, base in enumerate(self.MOTIF):
                self.conv.weight[0, "ACGT".index(base), k] = 1.0
            self.conv.bias.fill_(-4.0)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x)).sum(dim=2)


def _planted(n=16, length=100, seed=3):
    import random
    rng = random.Random(seed)
    seqs = []
    for i in range(n):
        background = [rng.choice("AGT") for _ in range(length)]
        at = 5 + (i * 7) % (length - 12)
        background[at:at + 6] = list(PlantedMotifModel.MOTIF)
        seqs.append("".join(background))
    return seqs


def test_seqlets_find_a_planted_motif_and_repeat_with_the_same_seed(tmp_path):
    from tfbs.mcp import server

    _register("planted", PlantedMotifModel(), 100)
    try:
        seqs = _planted()
        first = json.loads(asyncio.run(server.tfbs_seqlets(
            model_id="planted", sequences=seqs, n_shuffles=4, min_seqlet_len=3,
            max_seqlet_len=10, output_dir=str(tmp_path / "a"))))
        second = json.loads(asyncio.run(server.tfbs_seqlets(
            model_id="planted", sequences=seqs, n_shuffles=4, min_seqlet_len=3,
            max_seqlet_len=10, output_dir=str(tmp_path / "b"))))
    finally:
        server._model_registry.pop("planted", None)
    assert "error" not in first, first
    assert first["num_seqlets"] > 0 and "note" not in first
    assert first["seqlets"] == second["seqlets"]
    hits = 0
    for s in first["seqlets"]:
        at = 5 + (s["example_idx"] * 7) % 88
        hits += s["start"] < at + 6 and s["end"] > at
    assert hits == len(first["seqlets"])


MEME_TWO_MOTIFS = "\n".join([
    "MEME version 4",
    "",
    "ALPHABET= ACGT",
    "",
    "strands: + -",
    "",
    "Background letter frequencies",
    "A 0.25 C 0.25 G 0.25 T 0.25",
    "",
    "MOTIF M1 first",
    "letter-probability matrix: alength= 4 w= 3 nsites= 1 E= 0",
    "1.0 0.0 0.0 0.0",
    "0.0 1.0 0.0 0.0",
    "0.0 0.0 1.0 0.0",
    "",
    "MOTIF M2 second",
    "letter-probability matrix: alength= 4 w= 3 nsites= 1 E= 0",
    "0.0 0.0 0.0 1.0",
    "0.0 0.0 1.0 0.0",
    "0.0 1.0 0.0 0.0",
    "",
    "",
])


def test_seqlets_are_annotated_with_motif_names(monkeypatch, tiny_conv, tmp_path):
    from tfbs.mcp import server
    from tangermeme.io import read_meme

    motif_file = tmp_path / "two.meme"
    motif_file.write_text(MEME_TWO_MOTIFS)
    names = list(read_meme(str(motif_file)).keys())

    annotate = importlib.import_module("tangermeme.annotate")
    monkeypatch.setattr(
        annotate, "annotate_seqlets",
        lambda X, seqlets, motifs, **kw: (
            torch.tensor([[1], [0], [1], [0]], dtype=torch.int32),
            torch.tensor([[1e-3], [2e-3], [3e-3], [0.5]], dtype=torch.float64),
        ),
    )
    _patch_seqlet_caller(monkeypatch, _seqlet_frame(4))
    result = json.loads(asyncio.run(server.tfbs_seqlets(
        model_id=tiny_conv, sequences=SEQS_30, motif_file=str(motif_file),
    )))

    assert "annotation_error" not in result
    assert [s["motif"] for s in result["seqlets"]] == [names[1], names[0], names[1], None]
    assert result["seqlets"][0]["motif_p_value"] == pytest.approx(1e-3)
    assert result["seqlets"][3]["motif_p_value"] == pytest.approx(0.5)
    assert result["top_motifs"] == [
        {"name": names[1], "count": 2}, {"name": names[0], "count": 1},
    ]

    import pandas as pd
    table = pd.read_csv(result["files"]["seqlets"], sep="	", keep_default_na=False)
    assert list(table.columns) == SEQLET_COLUMNS + ["motif", "motif_p_value"]
    assert list(table["motif"]) == [names[1], names[0], names[1], ""]


def test_seqlets_runs_tangermeme_on_a_conv_model(tiny_conv):
    from tfbs.mcp import server

    result = json.loads(asyncio.run(server.tfbs_seqlets(
        model_id=tiny_conv, sequences=SEQS_30 * 2, n_shuffles=2,
        min_seqlet_len=3, max_seqlet_len=6,
    )))

    assert "error" not in result
    assert result["num_seqlets"] >= len(result["seqlets"])
    assert result["truncated"] == (result["num_seqlets"] > 200)
    for s in result["seqlets"]:
        assert set(s) == set(SEQLET_COLUMNS)
        assert 0 <= s["example_idx"] < 4
        assert 0 <= s["start"] < s["end"] <= 30


# ---------------------------------------------------------------------------
# tfbs_epistasis
# ---------------------------------------------------------------------------

class AdditiveModel(torch.nn.Module):
    """Sum of one weight per position and base: every pair is additive."""

    def __init__(self, length):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.weight = torch.nn.Parameter(
            torch.randn(4, length, generator=g, dtype=torch.float64))

    def forward(self, x):
        return (x.double() * self.weight).sum(dim=(1, 2)).unsqueeze(1)


class PairModel(torch.nn.Module):
    """1 only when position 1 is A and position 3 is C: one interacting pair."""

    def forward(self, x):
        return (x[:, 0, 1] * x[:, 1, 3]).unsqueeze(1)


def test_epistasis_is_zero_for_an_additive_model(tmp_path):
    from tfbs.mcp import server

    seq = "ACGTTGCAAC"
    L = len(seq)
    _register("additive", AdditiveModel(L), L)
    try:
        result = json.loads(asyncio.run(server.tfbs_epistasis(
            model_id="additive", sequence=seq, output_dir=str(tmp_path),
        )))
    finally:
        server._model_registry.pop("additive", None)

    assert result["num_pairs_analyzed"] == L * (L - 1) // 2
    assert result["num_double_mutants"] == 9 * result["num_pairs_analyzed"]
    assert "num_position_pairs" not in result
    assert result["inverse_transformed"] is False
    assert result["top_interactions"] == []

    with open(result["files"]["epistasis_matrix"], newline="") as f:
        pairs = list(csv.DictReader(f))
    assert len(pairs) == result["num_double_mutants"]
    assert max(abs(float(r["delta_delta"])) for r in pairs) < 1e-6
    assert all(r["base_i"] != r["ref_i"] and r["base_j"] != r["ref_j"] for r in pairs)

    with open(result["files"]["single_mutations"], newline="") as f:
        singles = list(csv.DictReader(f))
    assert len(singles) == 3 * L
    assert all(r["base"] != r["ref_base"] for r in singles)


def test_epistasis_finds_the_one_interacting_pair(tmp_path):
    from tfbs.mcp import server

    _register("pair", PairModel(), 6)
    try:
        result = json.loads(asyncio.run(server.tfbs_epistasis(model_id="pair", sequence="GGGGGG")))
    finally:
        server._model_registry.pop("pair", None)

    assert len(result["top_interactions"]) == 1
    top = result["top_interactions"][0]
    assert (top["pos_i"], top["base_i"], top["pos_j"], top["base_j"]) == (1, "A", 3, "C")
    assert (top["ref_i"], top["ref_j"]) == ("G", "G")
    assert top["delta_delta"] == 1.0
    # Without output_dir the CSVs go to <results>/<model_id>/tfbs_epistasis/<digest>/.
    out = Path(result["output_dir"])
    assert out.parent == tmp_path / "results" / "pair" / "tfbs_epistasis"
    assert {Path(p).name for p in result["files"].values()} == {
        "single_mutations.csv", "epistasis_matrix.csv",
    }
    assert all(Path(p).is_file() for p in result["files"].values())


def test_epistasis_rejects_positions_outside_the_sequence():
    from tfbs.mcp import server

    result = json.loads(asyncio.run(server.tfbs_epistasis(
        model_id="test_model", sequence="ACGTACGTAC", positions=[0, 12],
    )))
    assert "12" in result["error"]
    assert result["fix"]


# ---------------------------------------------------------------------------
# Device and partition defaults of the validation tools
# ---------------------------------------------------------------------------

def _reference_files(tmp_path):
    files = {}
    for name in ("genome.fa", "motif.meme", "peaks.bed", "clinvar.vcf.gz", "clinvar.vcf.gz.tbi"):
        path = tmp_path / name
        path.write_text("placeholder\n")
        files[name] = str(path)
    models = tmp_path / "models"
    for dataset in ("all_mean", "core_mean", "flank_mean"):
        ckpt = models / dataset / "standardize" / "best_VCNNBpnet.ckpt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_text("x")
    files["models_dir"] = str(models)
    return files


def _flag_value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def _remap(tmp_path, files, **kwargs):
    from tfbs.mcp import server

    return json.loads(asyncio.run(server.tfbs_remap_validation(
        remap_bed=files["peaks.bed"], output_dir=str(tmp_path / "remap"),
        motif_file=files["motif.meme"], genome_fasta=files["genome.fa"],
        skip_foldx=True, models_dir=files["models_dir"], **kwargs,
    )))


def _disease(tmp_path, files, **kwargs):
    from tfbs.mcp import server

    return json.loads(asyncio.run(server.tfbs_disease_variant_validation(
        peaks_bed=files["peaks.bed"], output_dir=str(tmp_path / "disease"),
        clinvar_vcf=files["clinvar.vcf.gz"], genome_fasta=files["genome.fa"],
        **kwargs,
    )))


def test_remap_validation_direct_run_uses_the_server_device(monkeypatch, tmp_path):
    from tfbs.mcp import server

    files = _reference_files(tmp_path)
    fake = FakeSlurm(monkeypatch)
    monkeypatch.setitem(server._ENV, "default_device", "cpu")

    result = _remap(tmp_path, files, slurm=False)
    assert result["mode"] == "direct"
    assert _flag_value(fake.calls[-1]["cmd"], "--device") == "cpu"


def test_remap_validation_slurm_job_uses_cuda_and_the_default_partition(monkeypatch, tmp_path):
    from tfbs.mcp import server

    files = _reference_files(tmp_path)
    FakeSlurm(monkeypatch)
    monkeypatch.setitem(server._ENV, "default_device", "cpu")
    monkeypatch.setitem(server._ENV, "default_partition", "gpu-a100")

    result = _remap(tmp_path, files, slurm=True)
    assert result["job_id"] == "4242"
    script = (tmp_path / "remap" / "remap_validation.sh").read_text()
    assert "#SBATCH --partition=gpu-a100" in script
    assert "--device cuda" in script

    _remap(tmp_path, files, slurm=True, device="cpu", partition="short")
    script = (tmp_path / "remap" / "remap_validation.sh").read_text()
    assert "#SBATCH --partition=short" in script
    assert "--device cpu" in script


def test_disease_validation_device_and_partition_defaults(monkeypatch, tmp_path):
    from tfbs.mcp import server

    files = _reference_files(tmp_path)
    fake = FakeSlurm(monkeypatch)
    monkeypatch.setitem(server._ENV, "default_device", "cpu")
    monkeypatch.setattr(server, "_pipeline_requirements_error", lambda *a: None)

    result = _disease(tmp_path, files, slurm=False)
    assert result["mode"] == "direct"
    assert _flag_value(fake.calls[-1]["cmd"], "--device") == "cpu"

    result = _disease(tmp_path, files, slurm=True)
    assert result["job_id"] == "4242"
    script = (tmp_path / "disease" / "disease_validation.sh").read_text()
    assert "#SBATCH --partition=compute" in script
    assert "--device cpu" in script


# ---------------------------------------------------------------------------
# SLURM tools
# ---------------------------------------------------------------------------

def test_slurm_status_says_when_slurm_is_missing(monkeypatch):
    from tfbs.mcp import server

    _no_binaries(monkeypatch)
    result = json.loads(asyncio.run(server.slurm_status(job_id="123")))
    assert "SLURM is not available" in result["error"]
    assert result["fix"]
    assert "state" not in result


def test_slurm_watch_says_when_slurm_is_missing(monkeypatch):
    from tfbs.mcp import server

    _no_binaries(monkeypatch)
    result = json.loads(asyncio.run(server.slurm_watch(job_id="123")))
    assert "SLURM is not available" in result["error"]
    assert result["fix"]


def test_slurm_watch_returns_the_next_status_call(monkeypatch):
    from tfbs.mcp import server

    FakeSlurm(monkeypatch, available=("squeue", "sacct"))
    result = json.loads(asyncio.run(server.slurm_watch(job_id="123", check_interval="30m")))

    assert result["next_check"] == {"tool": "slurm_status", "arguments": {"job_id": "123"}}
    assert result["check_interval"] == "30m"
    assert result["check_interval_seconds"] == 1800
    assert "COMPLETED" in result["terminal_states"]
    required, _, optional = result["instructions"].partition("Optional:")
    assert "slurm_status" in required
    assert "CronCreate" not in required
    assert "CronCreate" in optional


def test_slurm_watch_rejects_an_unreadable_interval(monkeypatch):
    from tfbs.mcp import server

    FakeSlurm(monkeypatch, available=("squeue", "sacct"))
    result = json.loads(asyncio.run(server.slurm_watch(job_id="123", check_interval="soon")))
    assert "soon" in result["error"]
    assert result["fix"]


def test_slurm_submit_writes_a_script_body_and_submits_it(monkeypatch, tmp_path):
    from tfbs.mcp import server

    fake = FakeSlurm(monkeypatch)
    result = json.loads(asyncio.run(server.slurm_submit(
        script_body="#SBATCH --time=00:05:00\nhostname\n", working_dir=str(tmp_path),
    )))
    script = Path(result["script_path"])
    try:
        assert result["job_id"] == "4242"
        assert script.read_text() == "#!/bin/bash\n#SBATCH --time=00:05:00\nhostname\n"
        call = fake.sbatch_calls()[-1]
        assert call["cmd"][-1] == str(script)
        assert call["cwd"] == str(tmp_path)
    finally:
        script.unlink()


def test_slurm_submit_keeps_a_body_with_its_own_shebang(monkeypatch):
    from tfbs.mcp import server

    FakeSlurm(monkeypatch)
    body = "#!/bin/sh\necho hi\n"
    result = json.loads(asyncio.run(server.slurm_submit(script_body=body)))
    script = Path(result["script_path"])
    try:
        assert script.read_text() == body
    finally:
        script.unlink()


def test_slurm_submit_needs_exactly_one_script_source(monkeypatch, tmp_path):
    from tfbs.mcp import server

    FakeSlurm(monkeypatch)
    existing = tmp_path / "job.sh"
    existing.write_text("#!/bin/bash\nhostname\n")
    neither = json.loads(asyncio.run(server.slurm_submit()))
    both = json.loads(asyncio.run(server.slurm_submit(
        script_path=str(existing), script_body="hostname",
    )))
    for result in (neither, both):
        assert "exactly one" in result["error"]
        assert result["fix"]


def test_slurm_submit_without_sbatch(monkeypatch):
    from tfbs.mcp import server

    _no_binaries(monkeypatch)
    result = json.loads(asyncio.run(server.slurm_submit(script_body="hostname")))
    assert "sbatch" in result["error"]
    assert result["fix"]


def test_train_config_mode_uses_the_server_partition(monkeypatch, tmp_path):
    from tfbs.mcp import server

    cfg = tmp_path / "train.yaml"
    cfg.write_text("seed_everything: 42\n")
    FakeSlurm(monkeypatch)
    monkeypatch.setitem(server._ENV, "default_partition", "gpu-h100")

    default = json.loads(asyncio.run(server.tfbs_train(config_path=str(cfg))))
    explicit = json.loads(asyncio.run(server.tfbs_train(config_path=str(cfg), partition="short")))
    try:
        assert "#SBATCH --partition=gpu-h100" in default["generated_script"]
        assert "#SBATCH --partition=short" in explicit["generated_script"]
    finally:
        Path(default["script_path"]).unlink()
        Path(explicit["script_path"]).unlink()


def test_partition_parameters_default_to_the_server_partition():
    from tfbs.mcp import server
    import tfbs.mcp.foldx_tools  # noqa: F401

    tools = asyncio.run(server.mcp.list_tools())
    defaults = {
        t.name: t.inputSchema["properties"]["partition"].get("default")
        for t in tools if "partition" in t.inputSchema.get("properties", {})
    }
    assert defaults.pop("tfbs_disease_variant_validation") == "compute"
    assert set(defaults) >= {
        "tfbs_train", "tfbs_sweep", "tfbs_validate", "tfbs_chipseq_benchmark",
        "tfbs_chipseq_method_comparison", "tfbs_remap_validation",
    }
    assert all(value is None for value in defaults.values())


def test_sweep_checks_sbatch_before_creating_a_sweep(monkeypatch, tmp_path):
    from tfbs.mcp import server

    cfg = tmp_path / "sweep.yaml"
    cfg.write_text("method: bayes\n")
    fake = FakeSlurm(monkeypatch, available=("wandb",))
    result = json.loads(asyncio.run(server.tfbs_sweep(
        sweep_config=str(cfg), project_name="p", model="VCNNBpnet",
        data_path=str(tmp_path), data_name="d",
    )))
    assert "sbatch" in result["error"]
    assert result["fix"]
    assert fake.calls == []


def test_sweep_agent_runs_the_packaged_pipeline_from_the_project_root(monkeypatch, tmp_path):
    from tfbs.mcp import server

    cfg = tmp_path / "sweep.yaml"
    cfg.write_text("method: bayes\n")
    fake = FakeSlurm(monkeypatch, available=("sbatch", "wandb"))
    monkeypatch.setitem(server._ENV, "default_partition", "gpu-test")
    monkeypatch.setattr(server, "_pipeline_requirements_error", lambda *a: None)

    result = json.loads(asyncio.run(server.tfbs_sweep(
        sweep_config=str(cfg), project_name="p", model="VCNNBpnet",
        data_path=str(tmp_path), data_name="d",
    )))
    script_path = Path(result["script_path"])
    try:
        assert result["sweep_id"] == "abc123"
        assert result["job_id"] == "4242"
        root = server._project_root()
        script = script_path.read_text()
        assert shlex.join([sys.executable, "-m", "tfbs.pipelines.run_sweep"]) in script
        assert "#SBATCH --partition=gpu-test" in script
        assert fake.sbatch_calls()[-1]["cwd"] == str(root)
    finally:
        script_path.unlink()


# ---------------------------------------------------------------------------
# tfbs_fimo and tfbs_server_info
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
NKX_MOTIF = Path(__file__).resolve().parent.parent / "tfbs" / "data" / "demo" / "MA1994.1.meme"
TG_24MER = "GGGCTCCTCACTTGACCTTTGCTG"


def _expected_backend() -> str:
    from tfbs.mcp import server

    return server._PYTHON_FIMO_BACKENDS[server._python_fimo()[1]]


def test_fimo_without_binary_or_python_fimo_returns_an_error(monkeypatch, tmp_path):
    from tfbs.mcp import server

    motif = tmp_path / "m.meme"
    motif.write_text("MEME version 4\n")
    _no_binaries(monkeypatch)
    monkeypatch.setattr(server, "_python_fimo", lambda: (None, None))
    result = json.loads(asyncio.run(server.tfbs_fimo(sequences=["ACGT"], motif_file=str(motif))))
    assert result["error"] == server._FIMO_MISSING["error"]
    assert "MEME Suite" in result["fix"] and "tangermeme" in result["fix"]


def test_fimo_falls_back_to_the_pytorch_fimo(monkeypatch, tmp_path):
    from tfbs.mcp import server

    _no_binaries(monkeypatch)
    rc = TG_24MER.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    result = json.loads(asyncio.run(server.tfbs_fimo(
        sequences=[TG_24MER, rc, "ACGTACGTACGTAC"], motif_file=str(NKX_MOTIF),
        output_dir=str(tmp_path / "fimo"))))
    assert result["backend"] == _expected_backend(), result
    assert result["backend_module"] == server._python_fimo()[1]
    hits = {(h["sequence"], h["strand"]): h for h in result["hits"]}
    plus = hits[("seq0", "+")]
    assert plus["motif"] == "MA1994.1" and plus["matched_sequence"] == "CTCACTTGACCTT"
    assert (plus["start"], plus["end"]) == (7, 19)
    assert TG_24MER[plus["start"] - 1:plus["end"]] == "CTCACTTGACCTT"
    assert plus["p_value"] < 1e-4 and plus["score"] > 0
    minus = hits[("seq1", "-")]
    assert minus["matched_sequence"] == "CTCACTTGACCTT"
    assert minus["p_value"] == pytest.approx(plus["p_value"])
    assert all(h["sequence"] != "seq2" for h in result["hits"])
    assert all(set(h) == set(server._FIMO_COLUMNS) for h in result["hits"])
    tsv = Path(result["files"]["fimo_tsv"]).read_bytes()
    assert b"\r\n" not in tsv
    assert tsv.splitlines()[0].decode().split("\t") == server._FIMO_COLUMNS
    assert Path(result["files"]["sequences_fasta"]).read_text().startswith(">seq0\n" + TG_24MER)


def test_fimo_binary_output_comes_back_in_the_same_columns(monkeypatch, tmp_path):
    from tfbs.mcp import server

    monkeypatch.setattr(server.shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")

    def meme_fimo(cmd, **kwargs):
        assert kwargs["stdin"] is subprocess.DEVNULL
        out = Path(cmd[cmd.index("--oc") + 1])
        out.mkdir(parents=True)
        (out / "fimo.tsv").write_text(
            "motif_id\tmotif_alt_id\tsequence_name\tstart\tstop\tstrand\tscore\tp-value"
            "\tq-value\tmatched_sequence\n"
            "MA1994.1\tNkx2-1\tseq0\t7\t19\t+\t12.28\t3.9e-05\t0.002\tCTCACTTGACCTT\n"
            "# fimo --thresh 0.0001\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(server.subprocess, "run", meme_fimo)
    result = json.loads(asyncio.run(server.tfbs_fimo(
        sequences=[TG_24MER], motif_file=str(NKX_MOTIF), output_dir=str(tmp_path / "f"))))
    assert result["backend"] == "meme" and "backend_module" not in result
    assert result["hits"] == [{
        "motif": "MA1994.1", "sequence": "seq0", "start": 7, "end": 19, "strand": "+",
        "score": 12.28, "p_value": 3.9e-05, "matched_sequence": "CTCACTTGACCTT"}]


def test_fimo_binary_vanishing_mid_call_falls_back(monkeypatch, tmp_path):
    from tfbs.mcp import server

    monkeypatch.setattr(server.shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")

    def missing(*args, **kwargs):
        raise FileNotFoundError("fimo")

    monkeypatch.setattr(server.subprocess, "run", missing)
    result = json.loads(asyncio.run(server.tfbs_fimo(
        sequences=[TG_24MER], motif_file=str(NKX_MOTIF), output_dir=str(tmp_path / "fimo"),
    )))
    assert result["backend"] == _expected_backend() and result["total_hits"] >= 1
    monkeypatch.setattr(server, "_python_fimo", lambda: (None, None))
    result = json.loads(asyncio.run(server.tfbs_fimo(
        sequences=[TG_24MER], motif_file=str(NKX_MOTIF), output_dir=str(tmp_path / "fimo2"),
    )))
    assert result["error"] == server._FIMO_MISSING["error"]


def test_fimo_fallback_scores_the_window_that_ends_on_the_last_base(monkeypatch, tmp_path):
    from tfbs.mcp import server

    _no_binaries(monkeypatch)
    site = "CTCACTTGACCTT"
    tail = "ACGTGCAGTA" + site
    result = json.loads(asyncio.run(server.tfbs_fimo(
        sequences=[site, tail, site.lower()], motif_file=str(NKX_MOTIF),
        output_dir=str(tmp_path / "fimo"))))
    assert "error" not in result, result
    by_seq = {}
    for h in result["hits"]:
        by_seq.setdefault(h["sequence"], []).append(h)
    assert len(by_seq["seq0"]) == 1 and len(by_seq["seq2"]) == 1
    assert (by_seq["seq0"][0]["start"], by_seq["seq0"][0]["end"]) == (1, 13)
    end_hit = next(h for h in by_seq["seq1"] if h["strand"] == "+")
    assert (end_hit["start"], end_hit["end"]) == (len(tail) - 12, len(tail))
    lengths = {"seq0": len(site), "seq1": len(tail), "seq2": len(site)}
    for h in result["hits"]:
        assert h["end"] <= lengths[h["sequence"]] and "N" not in h["matched_sequence"]
        assert h["matched_sequence"] == site
    fasta = Path(result["files"]["sequences_fasta"]).read_text()
    assert "N" not in fasta.replace(">", "").replace("seq", "")


def test_fimo_fallback_reads_padded_sequences_and_names_its_backend(monkeypatch, tmp_path):
    import pandas as pd
    from tfbs.mcp import server

    _no_binaries(monkeypatch)
    seen = {}

    def fake_fimo(motifs, fasta, threshold, reverse_complement):
        seen["fasta"] = Path(fasta).read_text()
        return [pd.DataFrame([
            {"motif_name": "M1 alt", "sequence_name": "seq0", "start": 0, "end": 4,
             "strand": "+", "score": 5.0, "p-value": 1e-5},
            {"motif_name": "M1 alt", "sequence_name": "seq0", "start": 2, "end": 6,
             "strand": "+", "score": 4.0, "p-value": 2e-5},
        ])]

    monkeypatch.setattr(server, "_python_fimo", lambda: (fake_fimo, "tangermeme.tools.fimo"))
    result = json.loads(asyncio.run(server.tfbs_fimo(
        sequences=["ACGTA"], motif_file=str(NKX_MOTIF), output_dir=str(tmp_path / "f"))))
    assert seen["fasta"] == ">seq0" + chr(10) + "ACGTAN" + chr(10)
    assert result["backend"] == "tangermeme"
    assert result["backend_module"] == "tangermeme.tools.fimo"
    assert [(h["start"], h["end"], h["matched_sequence"]) for h in result["hits"]] == [(1, 4, "ACGT")]


def test_fimo_writes_to_the_results_dir_by_default(monkeypatch, tmp_path):
    from tfbs.mcp import server

    _no_binaries(monkeypatch)
    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    first = json.loads(asyncio.run(server.tfbs_fimo(
        sequences=[TG_24MER], motif_file=str(NKX_MOTIF))))
    out = Path(first["output_dir"])
    assert out.parent.parent == tmp_path / "results" / "fimo"
    again = json.loads(asyncio.run(server.tfbs_fimo(
        sequences=[TG_24MER], motif_file=str(NKX_MOTIF))))
    assert again["output_dir"] == first["output_dir"]
    other = json.loads(asyncio.run(server.tfbs_fimo(
        sequences=[TG_24MER], motif_file=str(NKX_MOTIF), threshold=1e-3)))
    assert other["output_dir"] != first["output_dir"]
    doc = " ".join(server.tfbs_fimo.__doc__.split())
    assert "<results>/fimo/<digest>/" in doc and "temp" not in doc


def test_fimo_descriptions_name_memelite_not_pytorch():
    from tfbs.mcp import server

    doc = " ".join(server.tfbs_fimo.__doc__.split())
    assert "memelite, numba-based, shipped with tangermeme 1.x" in doc
    assert '"memelite" when memelite.fimo ran' in doc
    for text in (server.__doc__ or "", doc, json.dumps(server._FIMO_MISSING),
                 (REPO / "README.md").read_text(encoding="utf-8"),
                 (REPO / "docs" / "development.md").read_text(encoding="utf-8")):
        assert "PyTorch FIMO" not in text and "pure-PyTorch" not in text


def test_startup_warms_up_the_fimo_reimplementation(monkeypatch):
    from tfbs.mcp import server

    calls = []
    monkeypatch.setattr(server, "_warm_up_python_fimo", lambda: calls.append(1))
    server._preload_native_modules()
    assert calls == [1]

    def failing():
        raise RuntimeError("no threading layer")

    monkeypatch.setattr(server, "_warm_up_python_fimo", failing)
    server._preload_native_modules()


def test_fimo_warm_up_compiles_without_error():
    from tfbs.mcp import server

    if server._python_fimo()[0] is None:
        pytest.skip("no FIMO reimplementation installed")
    server._warm_up_python_fimo()


def test_startup_lists_a_module_only_when_no_alternative_loaded():
    from tfbs.mcp import server

    missing = {"tangermeme.ism": "x", "tangermeme.tools.fimo": "x", "pyBigWig": "x", "pysam": "x"}
    assert server._unavailable_modules(missing) == ["pyBigWig", "pysam"]
    both = {**missing, "memelite.fimo": "x"}
    assert server._unavailable_modules(both) == [
        "memelite.fimo", "pyBigWig", "pysam", "tangermeme.tools.fimo"]


def test_server_info_reports_cuda_version_and_default_partition(monkeypatch):
    from tfbs.mcp import server

    monkeypatch.delenv("TFBS_CALL_LOG", raising=False)
    info = json.loads(asyncio.run(server.tfbs_server_info()))
    assert info["cuda_version"] == torch.version.cuda
    assert info["default_partition"] == server._ENV["default_partition"]
    assert info["call_log"] is None
    assert "torch_loaded" not in info
    for key in info:
        assert key in server.tfbs_server_info.__doc__, key

    # Imported in-process too, without main()'s startup preload.
    monkeypatch.setattr(server, "_torch", None)
    fresh = json.loads(asyncio.run(server.tfbs_server_info()))
    assert fresh["torch_version"] == torch.__version__
    assert fresh["cuda_version"] == torch.version.cuda
    assert "cuda_available" in fresh and "torch_loaded" not in fresh


def test_server_info_reports_registered_reference_files(monkeypatch, tmp_path):
    """reference_files gives each reference variable's path and whether it exists."""
    from tfbs.mcp import server

    genome = tmp_path / "hg38.fa"
    genome.write_text(">chr1\nACGT\n")
    missing = tmp_path / "MA1994.1.meme"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TFBS_GENOME_FASTA", "hg38.fa")  # relative, as a user might
    monkeypatch.setenv("TFBS_MOTIF_FILE", str(missing))
    monkeypatch.delenv("TFBS_CLINVAR_VCF", raising=False)
    monkeypatch.setenv("TFBS_FOLDX_PDB", "   ")  # blank counts as unset

    info = json.loads(asyncio.run(server.tfbs_server_info()))
    files = info["reference_files"]
    assert list(files) == ["TFBS_GENOME_FASTA", "TFBS_MOTIF_FILE",
                           "TFBS_CLINVAR_VCF", "TFBS_FOLDX_PDB"]
    # Reported as the tools resolve it: expanded and absolute.
    assert files["TFBS_GENOME_FASTA"] == {"path": str(genome.resolve()), "exists": True}
    assert files["TFBS_MOTIF_FILE"] == {"path": str(missing.resolve()), "exists": False}
    assert files["TFBS_CLINVAR_VCF"] == {"path": None, "exists": None}
    assert files["TFBS_FOLDX_PDB"] == {"path": None, "exists": None}
    # The existing fields are still there.
    for key in ("on_slurm", "default_partition", "call_log", "results_dir",
                "project_root", "loaded_models", "torch_version", "cuda_available"):
        assert key in info, key
    assert "reference_files" in server.tfbs_server_info.__doc__
    # The description tells Claude how to register files.
    assert "tfbs-mcp --setup --non-interactive" in server.tfbs_server_info.__doc__


@pytest.mark.parametrize("arg", ["genome_fasta", "motif_file", "clinvar_vcf", "foldx_pdb"])
def test_missing_reference_fix_names_the_setup_flag(monkeypatch, arg):
    """The fix for an unconfigured reference file names the setup call that registers it."""
    from tfbs.mcp import server
    from tfbs.mcp.cli import SETUP_VARIABLES

    env_var = server.REFERENCE_DATA_ENV[arg]
    monkeypatch.delenv(env_var, raising=False)
    path, err = server._resolve_reference(None, arg)
    assert path is None
    payload = json.loads(err)
    flag = next(v.flag for v in SETUP_VARIABLES if v.env == env_var)
    assert payload["error"].startswith("No ")
    assert f"Pass {arg}=" in payload["fix"]
    assert f"tfbs-mcp --setup --non-interactive {flag} /path/to/file" in payload["fix"]
    assert "replaces the whole registration" in payload["fix"]


def test_startup_preloads_every_native_module_the_tools_import():
    """main() imports these before the stdio loop; a tool importing more would hang on Windows."""
    from tfbs.mcp import server

    for name in ("numpy", "pandas", "sklearn.metrics", "scipy.linalg", "torchmetrics",
                 "pytorch_lightning", "captum.attr", "tangermeme.deep_lift_shap",
                 "tangermeme.seqlet", "tangermeme.annotate", "pyfaidx", "pyBigWig", "pysam"):
        assert name in server._NATIVE_MODULES, name
    sources = [Path(server.__file__), Path(server.__file__).with_name("foldx_tools.py")]
    pure = {"importlib.util", "tfbs.configs", "tfbs.pipelines.checkpoints", "tfbs.mcp",
            "ctypes", "msvcrt", "sys", "csv", "bisect", "torch", "torch.nn.functional",
            "tfbs.nn.models", "tfbs.prediction.analysis", "tfbs.mcp.foldx_tools", "shlex",
            "tfbs.mcp.server", "tfbs.nn.loading"}
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            for fn in ([node] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else []):
                for inner in ast.walk(fn):
                    if isinstance(inner, ast.Import):
                        names = [a.name for a in inner.names]
                    elif isinstance(inner, ast.ImportFrom) and inner.level == 0:
                        names = [inner.module]
                    else:
                        continue
                    for name in names:
                        assert name in pure or name in server._NATIVE_MODULES, (
                            f"{path.name}:{inner.lineno} imports {name}")
    missing = server._preload_native_modules()
    assert "torch" not in missing and "numpy" not in missing
    assert server._torch is not None and server._pd is not None


def test_default_partition_comes_from_the_environment(monkeypatch):
    from tfbs.mcp import server

    monkeypatch.setenv("TFBS_SLURM_PARTITION", "a100")
    assert server._detect_environment()["default_partition"] == "a100"
    monkeypatch.delenv("TFBS_SLURM_PARTITION")
    assert server._detect_environment()["default_partition"] == "gpu"


# ---------------------------------------------------------------------------
# stdout stays clean (it is the MCP stdio channel)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["tfbs.mcp.server", "tfbs.prediction.analysis"])
def test_module_has_no_print_calls(module):
    path = Path(importlib.import_module(module).__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    prints = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert prints == []


def test_window_mode_prediction_writes_nothing_to_stdout(capsys):
    from tfbs.mcp import server

    result = json.loads(asyncio.run(server.tfbs_predict(
        model_id="test_model", sequences=["ACGTACGTAC" * 2], window_mode=True, window_size=10,
    )))
    assert len(result["best_scores"]) == 1
    assert capsys.readouterr().out == ""


def test_stray_prints_inside_a_tool_call_go_to_stderr(monkeypatch, capsys):
    from tfbs.mcp import server

    real = server._find_best_windows_fast

    def noisy(*args, **kwargs):
        print("noisy library output")
        return real(*args, **kwargs)

    monkeypatch.setattr(server, "_find_best_windows_fast", noisy)
    stdout_before = sys.stdout
    result = _call_via_mcp("tfbs_predict", {
        "model_id": "test_model", "sequences": ["ACGTACGTAC" * 2],
        "window_mode": True, "window_size": 10,
    })
    captured = capsys.readouterr()
    assert "best_scores" in result
    assert captured.out == ""
    assert "noisy library output" in captured.err
    assert sys.stdout is stdout_before


def test_startup_preload_prints_go_to_stderr(monkeypatch, capsys):
    from tfbs.mcp import server

    seen: dict = {}

    def noisy_preload():
        print("library banner at import")
        return {}

    monkeypatch.setattr(server, "_preload_native_modules", noisy_preload)
    monkeypatch.setattr(server, "_detach_child_stdin", lambda: None)
    monkeypatch.setattr(server.mcp, "run", lambda *a, **k: seen.setdefault("stdout", sys.stdout))
    stdout_before = sys.stdout
    server.main()
    captured = capsys.readouterr()
    assert captured.out == "" and "library banner at import" in captured.err
    assert seen["stdout"] is stdout_before


# ---------------------------------------------------------------------------
# Opt-in call log (TFBS_CALL_LOG)
# ---------------------------------------------------------------------------

def test_call_log_records_each_tool_call(monkeypatch, tmp_path):
    log = tmp_path / "logs" / "calls.jsonl"
    monkeypatch.setenv("TFBS_CALL_LOG", str(log))
    args = {"positive_scores": [0.9, 0.8], "negative_scores": [0.1, 0.2]}

    _call_via_mcp("tfbs_classify_metrics", args)
    _call_via_mcp("tfbs_list_models", {})

    raw = log.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"\n")
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    assert [r["tool"] for r in records] == ["tfbs_classify_metrics", "tfbs_list_models"]
    assert records[0]["arguments"] == args
    assert records[1]["arguments"] == {}
    for record in records:
        assert set(record) == {"timestamp", "tool", "arguments", "seconds", "is_error", "result"}
        started = datetime.fromisoformat(record["timestamp"])
        assert started.utcoffset() == timedelta(0)
        assert len(record["timestamp"].split("T")[1].split("+")[0]) == len("HH:MM:SS.mmm")
        assert isinstance(record["seconds"], float) and record["seconds"] >= 0
        assert record["is_error"] is False


def test_call_log_records_the_result_the_tool_returned(monkeypatch, tmp_path):
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("TFBS_CALL_LOG", str(log))
    args = {"positive_scores": [0.9, 0.8], "negative_scores": [0.1, 0.2]}

    returned = _call_via_mcp("tfbs_classify_metrics", args)

    (record,) = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert record["result"] == returned
    assert record["result"]["roc_auc"] == 1.0


def test_call_log_marks_an_error_result(monkeypatch, tmp_path):
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("TFBS_CALL_LOG", str(log))

    returned = _call_via_mcp("tfbs_predict", {"model_id": "no_such_model", "sequences": ["ACGT"]})

    (record,) = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert "error" in returned
    assert record["is_error"] is True
    assert record["result"] == returned


def test_call_log_records_a_call_that_raises_and_the_exception_still_propagates(
        monkeypatch, tmp_path):
    from mcp.server.fastmcp.exceptions import ToolError
    from tfbs.mcp import server

    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("TFBS_CALL_LOG", str(log))

    with pytest.raises(ToolError, match="Unknown tool"):
        asyncio.run(server.mcp.call_tool("tfbs_no_such_tool", {"x": 1}))

    (record,) = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert record["tool"] == "tfbs_no_such_tool"
    assert record["arguments"] == {"x": 1}
    assert record["is_error"] is True
    assert "Unknown tool" in record["result"]["error"]


def test_call_log_raise_with_an_unwritable_log_still_raises_the_tool_error(monkeypatch, tmp_path):
    from mcp.server.fastmcp.exceptions import ToolError
    from tfbs.mcp import server

    monkeypatch.setenv("TFBS_CALL_LOG", str(tmp_path))
    with pytest.raises(ToolError, match="Unknown tool"):
        asyncio.run(server.mcp.call_tool("tfbs_no_such_tool", {}))


def test_call_log_truncates_a_result_over_the_size_limit(monkeypatch, tmp_path):
    from tfbs.mcp import server

    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("TFBS_CALL_LOG", str(log))
    monkeypatch.setattr(server, "CALL_LOG_MAX_RESULT_BYTES", 20)
    args = {"positive_scores": [0.9, 0.8], "negative_scores": [0.1, 0.2]}

    returned = _call_via_mcp("tfbs_classify_metrics", args)

    (record,) = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    full = json.dumps(returned, ensure_ascii=False)
    assert record["result"] == {"truncated": True, "bytes": len(full.encode("utf-8")),
                                "head": full[:server.CALL_LOG_HEAD_CHARS]}
    assert record["is_error"] is False


def test_call_log_result_limit_is_one_megabyte_with_a_4096_character_head():
    from tfbs.mcp import server

    small = json.dumps({"scores": [0.5] * 1000})
    assert server._call_log_result(small) == (json.loads(small), False)

    big = json.dumps({"error": "x", "blob": "A" * (1024 * 1024)})
    logged, is_error = server._call_log_result(big)
    assert is_error is True
    assert logged["truncated"] is True
    assert logged["bytes"] == len(big.encode("utf-8"))
    assert logged["head"] == big[:4096]


def test_call_log_wraps_a_result_that_is_not_json():
    from types import SimpleNamespace

    from tfbs.mcp import server

    assert server._call_log_result("plain text") == ({"text": "plain text"}, False)
    blocks = [SimpleNamespace(type="text", text='{"a": 1}')]
    assert server._call_log_result((blocks, {"result": '{"a": 1}'})) == ({"a": 1}, False)


def test_call_log_is_off_without_the_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("TFBS_CALL_LOG", raising=False)
    monkeypatch.chdir(tmp_path)
    _call_via_mcp("tfbs_list_models", {})
    assert list(tmp_path.iterdir()) == []


def test_unwritable_call_log_does_not_break_the_tool_call(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("TFBS_CALL_LOG", str(tmp_path))
    with caplog.at_level("WARNING", logger="tfbs.mcp.server"):
        assert "test_model" in _call_via_mcp("tfbs_list_models", {})
    assert any("Could not append" in r.getMessage() for r in caplog.records)


def test_a_result_hook_that_raises_still_returns_the_tool_value(monkeypatch, tmp_path):
    from tfbs.mcp import server

    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("TFBS_CALL_LOG", str(log))

    def broken(result):
        raise RuntimeError("cannot serialise")

    monkeypatch.setattr(server, "_call_log_result", broken)
    returned = _call_via_mcp("tfbs_classify_metrics",
                             {"positive_scores": [0.9], "negative_scores": [0.1]})
    assert returned["roc_auc"] == 1.0

    (record,) = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert record["tool"] == "tfbs_classify_metrics"
    assert record["result"] == {"error": "result not loggable: cannot serialise"}
    assert record["is_error"] is False


def test_a_result_over_the_limit_is_cut_without_being_parsed(monkeypatch):
    from tfbs.mcp import server

    def no_parse(*a, **k):
        raise AssertionError("json.loads must not run on an oversized result")

    big = '{"hits": [' + ", ".join(["1"] * 400_000) + "]}"
    monkeypatch.setattr(server.json, "loads", no_parse)
    logged, is_error = server._call_log_result(big)
    assert logged == {"truncated": True, "bytes": len(big.encode("utf-8")),
                      "head": big[:server.CALL_LOG_HEAD_CHARS]}
    assert is_error is False


def test_an_exception_whose_str_raises_is_logged_by_type_and_re_raised(monkeypatch, tmp_path):
    from tfbs.mcp import server

    class Unprintable(Exception):
        def __str__(self):
            raise ValueError("no text")

    async def explode(*a, **k):
        raise Unprintable()

    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("TFBS_CALL_LOG", str(log))
    monkeypatch.setattr(server, "_tool_manager_call_tool", explode)
    with pytest.raises(Unprintable):
        asyncio.run(server.mcp.call_tool("tfbs_list_models", {}))
    (record,) = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert record["result"] == {"error": "Unprintable"} and record["is_error"] is True


def test_fimo_fallback_with_a_sequence_shorter_than_the_motif_has_no_hits(monkeypatch, tmp_path):
    from tfbs.mcp import server

    _no_binaries(monkeypatch)
    if server._python_fimo()[0] is None:
        pytest.skip("no FIMO reimplementation installed")
    result = json.loads(asyncio.run(server.tfbs_fimo(
        sequences=["ACGT", "ACGTACGTAC"], motif_file=str(NKX_MOTIF),
        output_dir=str(tmp_path / "fimo"))))
    assert "error" not in result, result
    assert result["total_hits"] == 0 and result["hits"] == []


def test_fimo_fallback_finds_a_motif_that_ends_the_file(monkeypatch, tmp_path):
    from tfbs.mcp import server

    _no_binaries(monkeypatch)
    if server._python_fimo()[0] is None:
        pytest.skip("no FIMO reimplementation installed")
    text = NKX_MOTIF.read_text()
    rows = text[:text.index("URL")].rstrip("\n")
    motif = tmp_path / "last.meme"
    motif.write_bytes(rows.encode())
    original = motif.read_bytes()
    result = json.loads(asyncio.run(server.tfbs_fimo(
        sequences=[TG_24MER], motif_file=str(motif), output_dir=str(tmp_path / "fimo"))))
    assert "error" not in result, result
    plus = [h for h in result["hits"] if h["strand"] == "+"]
    assert [(h["motif"], h["start"], h["end"]) for h in plus] == [("MA1994.1", 7, 19)]
    assert motif.read_bytes() == original
    copy = tmp_path / "fimo" / "motifs.meme"
    assert copy.read_bytes().endswith(b"\n\n") and b"\r" not in copy.read_bytes()


def test_fimo_binary_reads_the_original_motif_file(monkeypatch, tmp_path):
    from tfbs.mcp import server

    monkeypatch.setattr(server.shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")
    seen = {}

    def meme_fimo(cmd, **kwargs):
        seen["motif"] = cmd[-2]
        out = Path(cmd[cmd.index("--oc") + 1])
        out.mkdir(parents=True)
        (out / "fimo.tsv").write_text("motif_id\tmotif_alt_id\tsequence_name\tstart\tstop"
                                      "\tstrand\tscore\tp-value\tq-value\tmatched_sequence\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(server.subprocess, "run", meme_fimo)
    result = json.loads(asyncio.run(server.tfbs_fimo(
        sequences=[TG_24MER], motif_file=str(NKX_MOTIF), output_dir=str(tmp_path / "f"))))
    assert result["backend"] == "meme" and seen["motif"] == str(NKX_MOTIF.resolve())
    assert not (tmp_path / "f" / "motifs.meme").exists()


def test_fimo_is_not_annotated_read_only():
    from tfbs.mcp import server

    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert tools["tfbs_fimo"].annotations.readOnlyHint is False


def test_mkdir_error_returns_json_for_a_path_naming_a_file(tmp_path):
    from tfbs.mcp import server

    blocker = tmp_path / "taken"
    blocker.write_text("x")
    assert server._mkdir_error(tmp_path / "new" / "dir") is None
    assert (tmp_path / "new" / "dir").is_dir()
    for path in (blocker, blocker / "below"):
        result = json.loads(server._mkdir_error(path, "output_dir"))
        assert str(path) in result["error"] and "output_dir" in result["fix"]
    assert "already exists" in json.loads(server._mkdir_error(blocker))["error"]


def test_fimo_output_dir_that_is_a_file_is_an_error(monkeypatch, tmp_path):
    from tfbs.mcp import server

    monkeypatch.setattr(server.shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")
    blocker = tmp_path / "taken"
    blocker.write_text("x")
    result = json.loads(asyncio.run(server.tfbs_fimo(
        sequences=[TG_24MER], motif_file=str(NKX_MOTIF), output_dir=str(blocker))))
    assert "Cannot create the directory" in result["error"] and result["fix"]
