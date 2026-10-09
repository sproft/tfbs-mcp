"""tfbs_train local=true: training on this machine, end to end and its error paths."""
import asyncio
import json
import subprocess
from pathlib import Path

import pytest
import torch

from tfbs.mcp import server


def run(coro):
    return json.loads(asyncio.run(coro))


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    server._ensure_imports()
    yield
    for key in [k for k in server._model_registry if k.startswith("lt_")]:
        server._model_registry.pop(key)


def _dataset(tmp_path: Path, labels: torch.Tensor, name: str = "peaks") -> Path:
    """tfbs_prepare_dataset output for random 24 bp windows with these labels."""
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, 4, (len(labels), 24), generator=g)
    seqs = torch.nn.functional.one_hot(idx, 4).permute(0, 2, 1).float().contiguous()
    loci = tmp_path / f"{name}_loci"
    loci.mkdir()
    torch.save(seqs, loci / "sequences.pt")
    torch.save(labels, loci / "labels.pt")
    ds = run(server.tfbs_prepare_dataset(sequences_path=str(loci / "sequences.pt"),
                                         labels_path=str(loci / "labels.pt")))
    assert "error" not in ds, ds
    return Path(ds["output_dir"])


def _train(config: dict) -> dict:
    return run(server.tfbs_train(config_path=config["output_path"], local=True,
                                 local_timeout=900))


def test_local_regression_training_feeds_load_model_without_data_path(tmp_path):
    labels = torch.randn(48, 1, generator=torch.Generator().manual_seed(1)) * 2 + 3
    data = _dataset(tmp_path, labels)
    config = run(server.tfbs_config(model_type="SimpleCNN", data_path=str(data),
                                    max_epochs=1, batch_size=8))
    assert config["classify"] is False and config["scaling_method"] == "standardize"

    result = _train(config)
    assert result.get("mode") == "local", result
    assert result["exit_code"] == 0
    assert result["checkpoint_path"] == config["checkpoint_path"]
    assert result["checkpoint_exists"] is True
    assert isinstance(result["metrics"]["val_loss"], float)
    assert "val_acc" not in result["metrics"]
    assert Path(result["metrics_path"]).is_relative_to(Path(config["checkpoint_path"]).parent)
    assert result["command"][-2:] == ["--data.num_workers", "0"]
    log = Path(result["log_path"])
    assert log == Path(config["output_path"]).with_name("SimpleCNN_train.log") and log.is_file()

    loaded = run(server.tfbs_load_model("lt_reg", result["checkpoint_path"], "SimpleCNN",
                                        device="cpu"))
    assert Path(loaded["scaling_source"]) == data / "scaling_params.json"
    scores = run(server.tfbs_predict("lt_reg", sequences=["ACGT" * 6]))
    assert scores["inverse_transformed"] is True

    again = _train(config)
    assert again["exit_code"] == 0
    ckpts = sorted(p.name for p in Path(config["checkpoint_path"]).parent.glob("*.ckpt"))
    assert ckpts == ["best_SimpleCNN.ckpt"]


def test_local_classifier_training_reports_accuracy_and_auc(tmp_path):
    labels = torch.tensor([[1.0], [0.0]] * 24)
    data = _dataset(tmp_path, labels, name="bound")
    config = run(server.tfbs_config(model_type="SimpleCNN", data_path=str(data),
                                    max_epochs=1, batch_size=8))
    assert config["classify"] is True and "note" in config

    result = _train(config)
    assert result.get("exit_code") == 0, result
    assert result["checkpoint_exists"] is True
    metrics = result["metrics"]
    assert {"val_loss", "val_acc", "val_auc"} <= set(metrics)
    assert all(isinstance(metrics[k], float) for k in ("val_loss", "val_acc", "val_auc"))


# ---------------------------------------------------------------------------
# Error paths, without starting a trainer
# ---------------------------------------------------------------------------

@pytest.fixture
def config_file(tmp_path):
    cfg = tmp_path / "cfg" / "RNN.yaml"
    cfg.parent.mkdir()
    cfg.write_text(
        "trainer:\n  callbacks:\n"
        "  - class_path: pytorch_lightning.callbacks.ModelCheckpoint\n"
        "    init_args: {dirpath: ckpts, filename: best_RNN}\n")
    return cfg


def test_local_timeout_kills_and_reports_the_log(monkeypatch, config_file):
    def slow(cmd, **kwargs):
        assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["timeout"] == 5
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(server.subprocess, "run", slow)
    result = run(server.tfbs_train(config_path=str(config_file), local=True, local_timeout=5))
    assert "local_timeout=5" in result["error"] and "local_timeout" in result["fix"]
    assert result["log_path"] == str(config_file.with_name("RNN_train.log"))


def test_local_failure_returns_the_exit_code_and_log_tail(monkeypatch, config_file):
    def failing(cmd, **kwargs):
        kwargs["stdout"].write("Traceback: boom\n")
        assert kwargs["cwd"] == str(config_file.parent)
        assert "tfbs.cli.train_cli" in cmd and "--trainer.max_epochs" in cmd
        return subprocess.CompletedProcess(cmd, 3)

    monkeypatch.setattr(server.subprocess, "run", failing)
    result = run(server.tfbs_train(config_path=str(config_file), local=True,
                                   extra_args=["--trainer.max_epochs 2"]))
    assert result["exit_code"] == 3 and "boom" in result["log_tail"]
    assert result["error"] and result["fix"] and result["log_path"]


def test_local_success_reads_the_checkpoint_from_the_config(monkeypatch, config_file):
    def ok(cmd, **kwargs):
        ckpts = config_file.parent / "ckpts"
        (ckpts / "logs" / "version_0").mkdir(parents=True)
        (ckpts / "best_RNN.ckpt").write_text("x")
        (ckpts / "logs" / "version_0" / "metrics.csv").write_text(
            "epoch,step,train_loss,val_loss\n0,5,1.5,\n0,5,,0.75\n1,11,0.9,\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(server.subprocess, "run", ok)
    result = run(server.tfbs_train(config_path=str(config_file), local=True))
    assert result["checkpoint_path"] == str((config_file.parent / "ckpts" / "best_RNN.ckpt").resolve())
    assert result["checkpoint_exists"] is True
    assert result["metrics"] == {"val_loss": 0.75, "epoch": 0}


def test_without_sbatch_the_fix_offers_local_mode(monkeypatch, config_file):
    monkeypatch.setattr(server.shutil, "which", lambda *a, **k: None)
    result = run(server.tfbs_train(config_path=str(config_file)))
    assert "sbatch" in result["error"] and "local=true" in result["fix"]


def test_local_mode_needs_a_config(tmp_path):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n")
    result = run(server.tfbs_train(script=str(script), local=True))
    assert "config" in result["error"] and "config_path" in result["fix"]


def test_train_description_states_local_mode():
    doc = " ".join(server.tfbs_train.__doc__.split())
    assert "local=true" in doc and "blocks until training finishes" in doc
    assert "small datasets and laptops" in doc and "one or more argv tokens" in doc
