"""Training a classifier end to end, and the data module under it.

The dataset comes from the server's own tools (tfbs_prepare_classification,
tfbs_prepare_dataset, tfbs_config), so the label shapes are the ones a user
gets: labels.pt holds (N, 1) float tensors.  Training runs in-process through
LightningCLI, the same entry point as tfbs.cli.train_cli.
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tfbs.dataloaders.SeqDataset import TFBSDataModule
from tfbs.mcp import server
from tfbs.nn import models

WINDOW = 51
SMALL_VCNNBPNET = {"num_channels": 8, "kernel_size": 5, "dilations": [1, 2],
                   "pool_output_size": 8, "dense_sizes": [16]}


def _call(tool, **kwargs):
    result = json.loads(asyncio.run(tool(**kwargs)))
    assert "error" not in result, result
    return result


def _random_one_hot(n, generator):
    idx = torch.randint(0, 4, (n, WINDOW), generator=generator)
    return F.one_hot(idx, num_classes=4).float().permute(0, 2, 1).contiguous()


def _classification_dataset(tmp_path, val_size=0.2):
    gen = torch.Generator().manual_seed(0)
    loci = tmp_path / "peaks_loci"
    (loci / "negatives").mkdir(parents=True)
    torch.save(_random_one_hot(20, gen), loci / "sequences.pt")
    torch.save(_random_one_hot(20, gen), loci / "negatives" / "sequences.pt")
    cls = _call(server.tfbs_prepare_classification,
                positive_path=str(loci / "sequences.pt"),
                negative_path=str(loci / "negatives" / "sequences.pt"))
    assert cls["total"] == 40
    return _call(server.tfbs_prepare_dataset,
                 sequences_path=cls["files"]["sequences"],
                 labels_path=cls["files"]["labels"],
                 val_size=val_size)


def _fit(tmp_path, data_dir, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tfbs-train"])
    from pytorch_lightning.cli import LightningCLI

    config = _call(server.tfbs_config, model_type="VCNNBpnet", data_path=data_dir,
                   output_path=str(tmp_path / "cfg.yaml"), classify=True,
                   batch_size=8, max_epochs=1, checkpoint_dir=str(tmp_path / "ckpt"),
                   model_params=SMALL_VCNNBPNET)
    cli = LightningCLI(
        models.BaseModel,
        TFBSDataModule,
        subclass_mode_model=True,
        save_config_callback=None,
        args=["fit", "--config", config["output_path"], "--data.num_workers=0",
              "--trainer.accelerator=cpu", "--trainer.logger=false",
              "--trainer.enable_progress_bar=false",
              f"--trainer.default_root_dir={tmp_path}"],
    )
    assert cli.trainer.state.finished
    return cli, config


def test_a_classifier_trains_and_tests_on_the_prepared_dataset(tmp_path, monkeypatch):
    dataset = _classification_dataset(tmp_path)
    assert (dataset["train_count"], dataset["val_count"], dataset["test_count"]) == (26, 6, 8)
    assert torch.load(dataset["files"]["val_labels"]).shape == (6, 1)

    cli, config = _fit(tmp_path, dataset["output_dir"], monkeypatch)
    assert {"val_loss", "val_acc", "val_auc"} <= set(cli.trainer.callback_metrics)

    results = cli.trainer.test(cli.model, datamodule=cli.datamodule, verbose=False)
    assert set(results[0]) == {"test_loss", "test_acc", "test_auc"}
    assert 0.0 <= results[0]["test_acc"] <= 1.0

    assert Path(config["checkpoint_path"]).is_file()
    params = json.loads((Path(dataset["output_dir"]) / "scaling_params.json").read_text())
    assert params == {"scaling_method": "none"}


def test_the_validate_pipeline_saves_a_trained_classifiers_metrics(tmp_path, monkeypatch):
    import csv

    from tfbs.pipelines import validate

    dataset = _classification_dataset(tmp_path)
    _, config = _fit(tmp_path, dataset["output_dir"], monkeypatch)
    out = tmp_path / "results.csv"
    monkeypatch.chdir(tmp_path)
    validate.main(["--checkpoint_path", config["checkpoint_path"], "--model_name", "VCNNBpnet",
                   "--output_file", str(out), "--data_path", dataset["output_dir"],
                   "--batch_size", "8", "--num_workers", "0", "--scaling_method", "none"])
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert all(rows[0][k] != "" for k in ("test_loss", "test_acc", "test_auc"))


def test_a_validation_split_of_one_sequence_trains(tmp_path, monkeypatch):
    dataset = _classification_dataset(tmp_path, val_size=0.02)
    assert dataset["val_count"] == 1

    cli, _ = _fit(tmp_path, dataset["output_dir"], monkeypatch)
    assert {"val_loss", "val_acc"} <= set(cli.trainer.callback_metrics)
    validated = cli.trainer.validate(cli.model, datamodule=cli.datamodule, verbose=False)
    assert {"val_loss", "val_acc", "val_auc"} <= set(validated[0])


@pytest.mark.parametrize("label_shape", [(1, 1), (1,)])
@pytest.mark.parametrize("step", ["validation_step", "test_step"])
def test_classifier_steps_take_a_batch_of_one(label_shape, step):
    model = models.VCNNBpnet(input_length=WINDOW, classify=True, **SMALL_VCNNBPNET)
    x = _random_one_hot(1, torch.Generator().manual_seed(1))
    y = torch.ones(label_shape)
    loss = getattr(model, step)((x, y), 0)
    assert loss.dim() == 0 and torch.isfinite(loss)
    assert model.accuracy.compute() in (0.0, 1.0)


@pytest.mark.parametrize("label_shape", [(3, 1), (3,), (1, 1), (1,)])
@pytest.mark.parametrize("classify", [True, False])
def test_the_loss_takes_column_or_flat_targets(label_shape, classify):
    model = models.VCNNBpnet(input_length=WINDOW, classify=classify, **SMALL_VCNNBPNET)
    n = label_shape[0]
    x = _random_one_hot(n, torch.Generator().manual_seed(2))
    y = torch.ones(label_shape)
    loss, y_hat, _ = model._handle_batch((x, y))
    expected = (F.binary_cross_entropy if classify else F.mse_loss)(y_hat, torch.ones(n, 1))
    assert torch.allclose(loss, expected)


def _tensor_dataset(root, n=6):
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True)
        torch.save(torch.zeros(n, 4, WINDOW), root / split / "seqs.pt")
        torch.save(torch.arange(n, dtype=torch.float32).unsqueeze(1), root / split / "labels.pt")
    return root


def test_the_data_module_loads_with_no_worker_processes(tmp_path):
    dm = TFBSDataModule(data_path=str(_tensor_dataset(tmp_path / "ds")), batch_size=4,
                        num_workers=0, scaling_method="standardize")
    dm.setup("fit")
    for loader in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
        assert loader.num_workers == 0 and loader.persistent_workers is False
        assert loader.pin_memory == torch.cuda.is_available()
        x, y = next(iter(loader))
        assert x.shape[1:] == (4, WINDOW) and y.shape[1:] == (1,)


def test_worker_processes_persist_between_epochs(tmp_path):
    dm = TFBSDataModule(data_path=str(_tensor_dataset(tmp_path / "ds")), num_workers=2)
    dm.setup("fit")
    for loader in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
        assert loader.persistent_workers is True


def test_the_data_module_logs_instead_of_printing(tmp_path, capsys, caplog):
    data = _tensor_dataset(tmp_path / "ds")
    with caplog.at_level(logging.INFO, logger="tfbs.dataloaders.SeqDataset"):
        TFBSDataModule(data_path=str(data), num_workers=0,
                       scaling_method="standardize").setup("fit")
    assert capsys.readouterr().out == ""
    assert "standardize" in caplog.text
    params = json.loads((data / "scaling_params.json").read_text())
    assert params["scaling_method"] == "standardize" and params["std"] > 0
    assert b"\r\n" not in (data / "scaling_params.json").read_bytes()
