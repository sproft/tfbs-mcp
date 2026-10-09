"""Tests for tfbs_config: every model template, input_length inference, scaling rules."""
import asyncio
import importlib
import json
import sys

import pytest
import torch
import yaml

from tfbs.mcp import server
from tfbs.nn import models

TEMPLATE_NAMES = sorted(server._model_templates())
NEEDS_EQUIRC = pytest.mark.skipif(
    models._EQUIRC_IMPORT_ERROR is not None, reason="equirc is not installed"
)


def _dataset(tmp_path, length, n=8, name="ds"):
    root = tmp_path / name
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True)
        seqs = torch.zeros(n, 4, length)
        seqs[:, 0, :] = 1.0
        torch.save(seqs, root / split / "seqs.pt")
        torch.save(torch.randn(n), root / split / "labels.pt")
    return root


def _config(**kwargs):
    return json.loads(asyncio.run(server.tfbs_config(**kwargs)))


def _marks(name):
    return [NEEDS_EQUIRC] if name == "EquiNet" else []


def test_there_is_one_template_per_architecture():
    assert len(TEMPLATE_NAMES) == 13
    assert {"VCNNBpnet", "RNN", "BPNet", "BPNetReal"} <= set(TEMPLATE_NAMES)


@pytest.mark.parametrize(
    "name", [pytest.param(n, marks=_marks(n)) for n in TEMPLATE_NAMES]
)
def test_every_architecture_builds_and_runs_on_a_51_bp_window(tmp_path, name):
    data = _dataset(tmp_path, 51)
    out = tmp_path / "cfg.yaml"
    result = _config(model_type=name, data_path=str(data), output_path=str(out),
                     input_length=51)
    assert "error" not in result, result
    assert result["model_type"] == name
    assert result["input_length"] == 51

    cfg = yaml.safe_load(out.read_text())
    module_name, cls_name = cfg["model"]["class_path"].rsplit(".", 1)
    assert module_name == "tfbs.nn.models" and cls_name == name
    cls = getattr(importlib.import_module(module_name), cls_name)
    model = cls(**cfg["model"]["init_args"]).eval()

    x = torch.zeros(2, 4, 51)
    x[:, 1, :] = 1.0
    with torch.no_grad():
        y = model(x, torch.zeros(2, 1)) if name == "MultiCNN" else model(x)
    assert tuple(y.shape) == (2, 1)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(
    "name", [pytest.param(n, marks=_marks(n)) for n in TEMPLATE_NAMES]
)
def test_lightningcli_parses_every_generated_config(tmp_path, monkeypatch, name):
    monkeypatch.setattr(sys, "argv", ["tfbs-train"])
    from pytorch_lightning.cli import LightningCLI

    from tfbs.dataloaders.SeqDataset import TFBSDataModule

    data = _dataset(tmp_path, 51)
    out = tmp_path / "cfg.yaml"
    result = _config(model_type=name, data_path=str(data), output_path=str(out),
                     input_length=51, checkpoint_dir=str(tmp_path / "ckpt"))
    assert "error" not in result, result

    cli = LightningCLI(
        models.BaseModel,
        TFBSDataModule,
        subclass_mode_model=True,
        run=False,
        save_config_callback=None,
        args=["--config", str(out)],
    )
    assert type(cli.model).__name__ == name
    assert cli.datamodule.hparams.scaling_method == "standardize"


def test_application_note_example_trains_through_lightningcli(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tfbs-train"])
    from pytorch_lightning.cli import LightningCLI

    from tfbs.dataloaders.SeqDataset import TFBSDataModule

    data = _dataset(tmp_path, 51)
    out = tmp_path / "cfg.yaml"
    _config(model_type="BPNetReal", data_path=str(data), output_path=str(out),
            input_length=51, checkpoint_dir=str(tmp_path / "ckpt"))
    cli = LightningCLI(
        models.BaseModel,
        TFBSDataModule,
        subclass_mode_model=True,
        save_config_callback=None,
        args=["fit", "--config", str(out), "--data.num_workers=1",
              "--trainer.fast_dev_run=true", "--trainer.accelerator=cpu",
              "--trainer.enable_progress_bar=false"],
    )
    assert cli.trainer.state.finished
    assert cli.model.input_length == 51


def test_application_note_example_51_bp_bpnetreal(tmp_path):
    data = _dataset(tmp_path, 51)
    out = tmp_path / "cfg.yaml"
    result = _config(model_type="BPNetReal", data_path=str(data),
                     output_path=str(out), input_length=51)
    cfg = yaml.safe_load(out.read_text())
    assert cfg["model"]["class_path"] == "tfbs.nn.models.BPNetReal"
    assert cfg["model"]["init_args"]["input_length"] == 51
    assert "warning" not in result
    assert result["notes"]


def test_model_type_is_case_insensitive(tmp_path):
    data = _dataset(tmp_path, 24)
    result = _config(model_type="bpnetreal", data_path=str(data),
                     output_path=str(tmp_path / "c.yaml"))
    assert result["model_type"] == "BPNetReal"


def test_unknown_model_type_lists_the_valid_names(tmp_path):
    data = _dataset(tmp_path, 24)
    out = tmp_path / "c.yaml"
    result = _config(model_type="Transformer", data_path=str(data), output_path=str(out))
    assert result["error"] and result["fix"]
    assert result["valid_model_types"] == TEMPLATE_NAMES
    assert not out.exists()


def test_templates_come_from_the_model_yaml_dir_helper(tmp_path, monkeypatch):
    tdir = tmp_path / "templates"
    tdir.mkdir()
    (tdir / "SimpleCNN.yaml").write_text(
        "class_path: tfbs.nn.models.SimpleCNN\n"
        "init_args:\n  input_length: 24\n  dense_size: 7\n"
    )
    monkeypatch.setattr(server, "_model_yaml_dir", lambda: tdir)
    data = _dataset(tmp_path, 24)
    ok = _config(model_type="SimpleCNN", data_path=str(data),
                 output_path=str(tmp_path / "a.yaml"))
    assert ok["model_init_args"]["dense_size"] == 7
    missing = _config(model_type="VCNNBpnet", data_path=str(data),
                      output_path=str(tmp_path / "b.yaml"))
    assert missing["valid_model_types"] == ["SimpleCNN"]


def test_missing_template_directory_is_a_json_error(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_model_yaml_dir", lambda: tmp_path / "nowhere")
    data = _dataset(tmp_path, 24)
    result = _config(model_type="VCNNBpnet", data_path=str(data),
                     output_path=str(tmp_path / "c.yaml"))
    assert result["error"] and "tfbs/configs/models" in result["fix"]


def test_input_length_is_inferred_from_the_dataset(tmp_path):
    data = _dataset(tmp_path, 37)
    out = tmp_path / "c.yaml"
    result = _config(model_type="VCNNBpnet", data_path=str(data), output_path=str(out))
    assert result["input_length"] == 37
    assert result["input_length_source"] == "dataset"
    cfg = yaml.safe_load(out.read_text())
    assert cfg["model"]["init_args"]["input_length"] == 37


def test_flexible_mlp_input_length_covers_the_flattened_one_hot(tmp_path):
    data = _dataset(tmp_path, 30)
    result = _config(model_type="FlexibleMLP", data_path=str(data),
                     output_path=str(tmp_path / "c.yaml"))
    assert result["input_length"] == 30
    assert result["model_init_args"]["input_length"] == 120


def test_explicit_input_length_that_differs_from_the_dataset_warns(tmp_path):
    data = _dataset(tmp_path, 24)
    result = _config(model_type="RNN", data_path=str(data),
                     output_path=str(tmp_path / "c.yaml"), input_length=51)
    assert result["input_length"] == 51
    assert result["input_length_source"] == "argument"
    assert "24" in result["warning"]


def test_omitted_input_length_without_tensors_is_a_json_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _config(model_type="RNN", data_path=str(empty),
                     output_path=str(tmp_path / "c.yaml"))
    assert result["error"] and "input_length" in result["fix"]


@pytest.mark.parametrize("method", ["standardize", "normalize"])
def test_classify_with_label_scaling_is_a_json_error(tmp_path, method):
    data = _dataset(tmp_path, 24)
    out = tmp_path / "c.yaml"
    result = _config(model_type="VCNNBpnet", data_path=str(data), output_path=str(out),
                     classify=True, scaling_method=method)
    assert result["error"] and result["fix"]
    assert not out.exists()


def test_scaling_defaults_follow_the_task(tmp_path):
    data = _dataset(tmp_path, 24)
    reg_out, cls_out = tmp_path / "reg.yaml", tmp_path / "cls.yaml"
    reg = _config(model_type="VCNNBpnet", data_path=str(data), output_path=str(reg_out))
    cls = _config(model_type="VCNNBpnet", data_path=str(data), output_path=str(cls_out),
                  classify=True)
    assert reg["scaling_method"] == "standardize"
    assert cls["scaling_method"] == "none"
    reg_cfg = yaml.safe_load(reg_out.read_text())
    cls_cfg = yaml.safe_load(cls_out.read_text())
    assert reg_cfg["data"]["scaling_method"] == "standardize"
    assert cls_cfg["data"]["scaling_method"] == "none"
    assert cls_cfg["model"]["init_args"]["classify"] is True
    ckpt = reg_cfg["trainer"]["callbacks"][1]["init_args"]
    assert ckpt["dirpath"].replace("\\", "/").endswith("saved_models/ds/standardize")
    assert ckpt["filename"] == "best_VCNNBpnet"


def test_unknown_scaling_method_is_a_json_error(tmp_path):
    data = _dataset(tmp_path, 24)
    result = _config(model_type="RNN", data_path=str(data),
                     output_path=str(tmp_path / "c.yaml"), scaling_method="log")
    assert result["error"] and result["fix"]


def test_model_params_override_the_template_and_reject_unknown_keys(tmp_path):
    data = _dataset(tmp_path, 24)
    ok = _config(model_type="BPNetReal", data_path=str(data),
                 output_path=str(tmp_path / "a.yaml"), model_params={"n_layers": 3})
    assert ok["model_init_args"]["n_layers"] == 3
    bad = _config(model_type="BPNetReal", data_path=str(data),
                  output_path=str(tmp_path / "b.yaml"), model_params={"num_channels": 3})
    assert bad["error"] and "n_filters" in bad["fix"]
    reserved = _config(model_type="BPNetReal", data_path=str(data),
                       output_path=str(tmp_path / "c.yaml"),
                       model_params={"input_length": 99})
    assert reserved["error"] and reserved["fix"]


def test_tool_description_lists_architectures_and_default_rationale():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    desc = tools["tfbs_config"].description
    for name in TEMPLATE_NAMES:
        assert name in desc
    for default in ("input_length:", "scaling_method:", "batch_size 64",
                    "max_epochs 500", "patience 5", "checkpoint_dir:"):
        assert default in desc
    assert desc.count("because") >= 5
    assert chr(0x2014) not in desc and chr(0x2013) not in desc


def test_tfbs_config_writes_nothing_to_stdout(tmp_path, capfd):
    data = _dataset(tmp_path, 24)
    _config(model_type="VCNNBpnet", data_path=str(data), output_path=str(tmp_path / "c.yaml"))
    assert capfd.readouterr().out == ""
