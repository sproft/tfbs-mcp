"""tfbs_validate validates a checkpoint on the label scale it was trained with.

trainer.test() scores the model's outputs, which are on its training scale,
against labels put on scaling_method's scale, so a mismatch produces metrics
that compare two different scales. The old default, "standardize" whatever
the checkpoint, did exactly that for every classifier.
"""
import asyncio
import json
import subprocess

import pytest
import torch

from tfbs.mcp import server


def run(coro):
    return json.loads(asyncio.run(coro))


@pytest.fixture
def submitted(tmp_path, monkeypatch):
    """Stand-in sbatch: records each submission instead of running it."""
    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    calls = []
    monkeypatch.setattr(server.subprocess, "run",
                        lambda cmd, *a, **k: calls.append(cmd) or subprocess.CompletedProcess(
                            cmd, 0, "Submitted batch job 9\n", ""))
    monkeypatch.setattr(server.shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")
    return calls


def _model(classify):
    from tfbs.nn import models

    torch.manual_seed(0)
    return models.VCNNBpnet(num_channels=4, kernel_size=3, dilations=[1, 2],
                            pool_output_size=5, dense_sizes=[8], input_length=10,
                            classify=classify)


def _inference(path, scaling="standardize", classify=False):
    model = _model(classify)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"arch": "VCNNBpnet", "hyper_parameters": dict(model.hparams),
               "state_dict": model.state_dict()}
    if scaling is not None:
        payload["scaling"] = ({"scaling_method": "standardize", "mean": 0.0, "std": 1.0}
                              if scaling == "standardize" else {"scaling_method": scaling})
    torch.save(payload, path)
    return path


def _lightning(path, classify, datamodule_scaling=None):
    model = _model(classify)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(),
               "hyper_parameters": {**dict(model.hparams),
                                    "_class_path": "tfbs.nn.models.VCNNBpnet"}}
    if datamodule_scaling is not None:
        payload["datamodule_hyper_parameters"] = {"scaling_method": datamodule_scaling}
    torch.save(payload, path)
    return path


def _dataset(root, labels):
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True)
        torch.save(torch.zeros(len(labels), 4, 10), root / split / "seqs.pt")
        torch.save(torch.tensor(labels, dtype=torch.float32).reshape(-1, 1),
                   root / split / "labels.pt")
    return root


BINARY = [0.0, 1.0, 1.0, 0.0]
REAL = [0.3, -1.2, 2.5, 0.8]


def _validate(ckpt, data, **kw):
    return run(server.tfbs_validate(checkpoint_path=str(ckpt), data_path=str(data),
                                    output_file=str(data.parent / "v.csv"), **kw))


def test_default_is_the_inference_checkpoints_own_scaling(tmp_path, submitted):
    ckpt = _inference(tmp_path / "released" / "flank" / "vcnnbpnet.pt", scaling="standardize")
    result = _validate(ckpt, _dataset(tmp_path / "data", REAL))
    assert "error" not in result, result
    assert (result["scaling_method"], result["scaling_source"]) == ("standardize", "checkpoint")
    assert "--scaling_method standardize" in result["validate_command"]
    assert len(submitted) == 1


def test_a_lightning_classifier_is_validated_unscaled_by_default(tmp_path, submitted):
    ckpt = _lightning(tmp_path / "trained" / "best.ckpt", classify=True, datamodule_scaling="none")
    data = _dataset(tmp_path / "data", BINARY)
    result = _validate(ckpt, data)
    assert (result["scaling_method"], result["scaling_source"]) == ("none", "checkpoint")
    assert "--scaling_method none" in result["validate_command"]
    # the old default, now an explicit and contradictory request, is refused
    refused = _validate(ckpt, data, scaling_method="standardize")
    assert "differs from 'none'" in refused["error"]
    assert "scaling_method='none'" in refused["fix"] and "Nothing was submitted" in refused["fix"]
    assert len(submitted) == 1


def test_the_scaling_folder_is_read_when_the_data_module_records_none(tmp_path, submitted):
    ckpt = _lightning(tmp_path / "saved_models" / "ds" / "none" / "best_VCNNBpnet.ckpt",
                      classify=True)
    result = _validate(ckpt, _dataset(tmp_path / "data", BINARY))
    assert (result["scaling_method"], result["scaling_source"]) == ("none", "checkpoint")


def test_a_classifier_that_records_no_scaling_refuses_label_scaling(tmp_path, submitted):
    ckpt = _lightning(tmp_path / "trained" / "best.ckpt", classify=True)
    data = _dataset(tmp_path / "data", BINARY)
    result = _validate(ckpt, data)
    assert (result["scaling_method"], result["scaling_source"]) == ("none", "checkpoint classify flag")
    refused = _validate(ckpt, data, scaling_method="standardize")
    assert "is a classifier" in refused["error"] and "0/1 labels unscaled" in refused["error"]
    assert len(submitted) == 1


def test_a_regression_checkpoint_without_records_keeps_standardize(tmp_path, submitted):
    ckpt = _lightning(tmp_path / "trained" / "best.ckpt", classify=False)
    result = _validate(ckpt, _dataset(tmp_path / "data", REAL))
    assert (result["scaling_method"], result["scaling_source"]) == (
        "standardize", "checkpoint classify flag")


def test_an_unreadable_checkpoint_falls_back_to_the_labels(tmp_path, submitted):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("x")
    binary = _validate(ckpt, _dataset(tmp_path / "bin", BINARY))
    assert (binary["scaling_method"], binary["scaling_source"]) == ("none", "labels")
    real = _validate(ckpt, _dataset(tmp_path / "real", REAL))
    assert (real["scaling_method"], real["scaling_source"]) == ("standardize", "default")
    assert any("could not be read" in c for c in real["cautions"])


def test_a_classifier_saved_under_standardize_is_validated_as_trained(tmp_path, submitted):
    """A classifier trained under the old standardize default: no dead end, a caution."""
    ckpt = _lightning(tmp_path / "trained" / "best.ckpt", classify=True,
                      datamodule_scaling="standardize")
    data = _dataset(tmp_path / "data", BINARY)
    result = _validate(ckpt, data)
    assert result["scaling_method"] == "standardize"
    assert any("trained against scaled 0/1 labels" in c for c in result["cautions"])
    refused = _validate(ckpt, data, scaling_method="none")
    assert "differs from 'standardize'" in refused["error"]


def test_an_explicit_scaling_that_agrees_is_used(tmp_path, submitted):
    ckpt = _inference(tmp_path / "released" / "flank" / "vcnnbpnet.pt", scaling="standardize")
    result = _validate(ckpt, _dataset(tmp_path / "data", REAL), scaling_method="standardize")
    assert (result["scaling_method"], result["scaling_source"]) == ("standardize", "argument")


def test_an_unknown_scaling_method_is_rejected_before_submission(tmp_path, submitted):
    ckpt = _inference(tmp_path / "released" / "flank" / "vcnnbpnet.pt")
    result = _validate(ckpt, _dataset(tmp_path / "data", REAL), scaling_method="zscore")
    assert result["error"] == "Unknown scaling_method 'zscore'."
    assert submitted == []


def test_the_description_states_the_checkpoint_default():
    doc = " ".join(server.tfbs_validate.__doc__.split())
    assert "It must be the scaling the checkpoint was trained with" in doc
    assert "label scaling for a classifier, is an error" in doc
