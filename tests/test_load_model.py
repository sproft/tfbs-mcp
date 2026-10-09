"""tfbs_load_model: class paths of LightningCLI checkpoints, scaling parameters,
and tfbs_predict returning scores in the original label units."""
import asyncio
import json
import sys
import types
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch

from tfbs.mcp import server
from tfbs.nn import models

SEQ = "ACGTACGTAC"
INSTANTIATOR = "pytorch_lightning.cli.instantiate_module"


def run(coro):
    return json.loads(asyncio.run(coro))


@pytest.fixture(autouse=True)
def _cleanup():
    server._ensure_imports()
    yield
    for key in [k for k in server._model_registry if k.startswith("lm_")]:
        server._model_registry.pop(key)


def _save_ckpt(path: Path, class_path: str | None = None, input_length: int = 10,
               datamodule: dict | None = None) -> Path:
    """A checkpoint as LightningCLI writes it, of a SimpleCNN."""
    torch.manual_seed(0)
    model = models.SimpleCNN(input_length=input_length)
    hparams = dict(model.hparams)
    if class_path:
        hparams["_class_path"] = class_path
        hparams["_instantiator"] = INSTANTIATOR
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"state_dict": model.state_dict(), "hyper_parameters": hparams,
            "pytorch-lightning_version": pl.__version__}
    if datamodule is not None:
        ckpt["datamodule_hyper_parameters"] = datamodule
    torch.save(ckpt, path)
    return path


def _shim(name: str) -> None:
    """Register <name>, <name>.nn and <name>.nn.models mapping to tfbs.nn.models."""
    pkg, nn = types.ModuleType(name), types.ModuleType(f"{name}.nn")
    pkg.__path__, nn.__path__ = [], []
    pkg.nn, nn.models = nn, models
    sys.modules[name] = pkg
    sys.modules[f"{name}.nn"] = nn
    sys.modules[f"{name}.nn.models"] = models


# ---------------------------------------------------------------------------
# Class paths
# ---------------------------------------------------------------------------

def test_checkpoint_with_a_shim_class_path_loads(tmp_path, monkeypatch):
    for key in ("fakeshim", "fakeshim.nn", "fakeshim.nn.models"):
        monkeypatch.delitem(sys.modules, key, raising=False)
    _shim("fakeshim")
    try:
        ckpt = _save_ckpt(tmp_path / "ds" / "standardize" / "best_SimpleCNN.ckpt",
                          "fakeshim.nn.models.SimpleCNN")
        result = run(server.tfbs_load_model("lm_shim", str(ckpt), "SimpleCNN", device="cpu"))
        assert "error" not in result, result
        assert result["input_length"] == 10
        assert result["scaling_params"] is None
    finally:
        for key in ("fakeshim", "fakeshim.nn", "fakeshim.nn.models"):
            sys.modules.pop(key, None)


def test_unimportable_class_path_is_a_json_error(tmp_path):
    ckpt = _save_ckpt(tmp_path / "best.ckpt", "nosuchshim_xyz.nn.models.SimpleCNN")
    result = run(server.tfbs_load_model("lm_bad", str(ckpt), "SimpleCNN", device="cpu"))
    assert "nosuchshim_xyz.nn.models" in result["error"]
    assert "PYTHONPATH" in result["fix"] and "tfbs.nn.models.SimpleCNN" in result["fix"]
    assert result["class_path"] == "nosuchshim_xyz.nn.models.SimpleCNN"
    assert "lm_bad" not in server._model_registry


class StrictLoader(torch.nn.Module):
    """Loads like Lightning 2.5: refuses a class path whose module is not imported."""

    calls = 0
    named = "strictshim.models.StrictLoader"

    def __init__(self):
        super().__init__()
        self.input_length = 10
        self.classify = False
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x.mean(dim=2))

    @classmethod
    def load_from_checkpoint(cls, path, map_location=None):
        cls.calls += 1
        module = cls.named.rpartition(".")[0]
        if module not in sys.modules:
            raise ValueError(f"Import path {cls.named} does not correspond to an imported class")
        return cls()


@pytest.fixture
def strict(tmp_path, monkeypatch):
    pkg = tmp_path / "pkgs" / "strictshim"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "models.py").write_text("from tfbs.nn.models import *\n")
    monkeypatch.syspath_prepend(str(tmp_path / "pkgs"))
    for key in ("strictshim", "strictshim.models"):
        monkeypatch.delitem(sys.modules, key, raising=False)
    monkeypatch.setattr(server._models, "StrictLoader", StrictLoader, raising=False)
    monkeypatch.setattr(StrictLoader, "calls", 0)
    yield
    for key in ("strictshim", "strictshim.models"):
        sys.modules.pop(key, None)


def test_module_in_the_hyper_parameters_is_imported_before_loading(tmp_path, strict):
    ckpt = _save_ckpt(tmp_path / "best.ckpt", StrictLoader.named)
    result = run(server.tfbs_load_model("lm_strict", str(ckpt), "StrictLoader", device="cpu"))
    assert "error" not in result, result
    assert StrictLoader.calls == 1
    assert "strictshim.models" in sys.modules


def test_class_path_named_by_the_error_is_imported_and_retried_once(tmp_path, strict):
    ckpt = _save_ckpt(tmp_path / "best.ckpt")
    result = run(server.tfbs_load_model("lm_retry", str(ckpt), "StrictLoader", device="cpu"))
    assert "error" not in result, result
    assert StrictLoader.calls == 2


def test_class_path_named_by_the_error_that_cannot_be_imported(tmp_path, strict, monkeypatch):
    monkeypatch.setattr(StrictLoader, "named", "absentshim_q.models.StrictLoader")
    ckpt = _save_ckpt(tmp_path / "best.ckpt")
    result = run(server.tfbs_load_model("lm_absent", str(ckpt), "StrictLoader", device="cpu"))
    assert "absentshim_q.models" in result["error"] and "PYTHONPATH" in result["fix"]
    assert StrictLoader.calls == 1


def test_wrong_model_type_is_a_json_error(tmp_path):
    ckpt = _save_ckpt(tmp_path / "best.ckpt")
    result = run(server.tfbs_load_model("lm_wrong", str(ckpt), "VCNNBpnet", device="cpu"))
    assert "Could not load" in result["error"] and "model_type" in result["fix"]


def test_unknown_model_type_and_missing_file_are_json_errors(tmp_path):
    ckpt = _save_ckpt(tmp_path / "best.ckpt")
    result = run(server.tfbs_load_model("lm_x", str(ckpt), "Transformer"))
    assert "Transformer" in result["error"] and "SimpleCNN" in result["available"]
    result = run(server.tfbs_load_model("lm_x", str(tmp_path / "none.ckpt"), "SimpleCNN"))
    assert "not found" in result["error"] and result["fix"]


def test_description_states_the_class_path_handling():
    doc = " ".join(server.tfbs_load_model.__doc__.split())
    assert "classes.nn.models.VCNNBpnet" in doc and "PYTHONPATH" in doc


# ---------------------------------------------------------------------------
# Scaling parameters
# ---------------------------------------------------------------------------

STD = {"scaling_method": "standardize", "mean": 5.0, "std": 2.0}


def _write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def study(tmp_path):
    """The study's layout: saved_models_final/<name>/standardize/best_<model>.ckpt."""
    root = tmp_path / "saved_models_final"
    ckpt = _save_ckpt(root / "all_mean" / "standardize" / "best_SimpleCNN.ckpt")
    _save_ckpt(root / "core_mean" / "standardize" / "best_SimpleCNN.ckpt")
    sp = root / "scaling_params"
    _write(sp / "all_mean_standardize_SimpleCNN.json", STD)
    _write(sp / "core_mean_standardize_SimpleCNN.json", {**STD, "mean": -1.0})
    _write(sp / "scaling_params_all_models.json", {
        "all_mean_standardize_SimpleCNN": STD,
        "core_mean_standardize_SimpleCNN": {**STD, "mean": -1.0},
    })
    return root, ckpt


def _load(ckpt, data_path=None, model_id="lm_sp"):
    return run(server.tfbs_load_model(model_id, str(ckpt), "SimpleCNN", device="cpu",
                                      data_path=data_path))


def test_study_directory_picks_the_file_named_after_the_checkpoint(study):
    root, ckpt = study
    result = _load(ckpt, str(root / "scaling_params"))
    assert result["scaling_params"] == STD
    assert result["scaling_source"].endswith("all_mean_standardize_SimpleCNN.json")
    core = _load(root / "core_mean" / "standardize" / "best_SimpleCNN.ckpt",
                 str(root / "scaling_params"))
    assert core["scaling_params"]["mean"] == -1.0


def test_combined_file_is_read_by_the_checkpoint_key(study):
    root, ckpt = study
    result = _load(ckpt, str(root / "scaling_params" / "scaling_params_all_models.json"))
    assert result["scaling_params"] == STD


def test_direct_json_path(tmp_path):
    ckpt = _save_ckpt(tmp_path / "run" / "best_SimpleCNN.ckpt")
    path = _write(tmp_path / "anything.json", {"scaling_method": "normalize", "min": 1.0, "max": 3.0})
    assert _load(ckpt, str(path))["scaling_params"]["scaling_method"] == "normalize"


def test_scaling_file_must_match_the_checkpoint_scaling(study, tmp_path):
    _, ckpt = study  # .../all_mean/standardize/best_SimpleCNN.ckpt
    path = _write(tmp_path / "anything.json", {"scaling_method": "normalize", "min": 1.0, "max": 3.0})
    result = _load(ckpt, str(path))
    assert "'normalize'" in result["error"] and "'standardize'" in result["error"]
    assert result["checkpoint_scaling_method"] == "standardize" and result["fix"]
    assert "lm_sp" not in server._model_registry


def test_directory_with_scaling_params_json(tmp_path):
    ckpt = _save_ckpt(tmp_path / "ckpt" / "best.ckpt")
    data = tmp_path / "dataset"
    _write(data / "scaling_params.json", STD)
    _write(data / "other.json", {"x": 1})
    assert _load(ckpt, str(data))["scaling_params"] == STD


def test_directory_with_exactly_one_scaling_json(tmp_path):
    ckpt = _save_ckpt(tmp_path / "ckpt" / "best.ckpt")
    data = tmp_path / "dataset"
    _write(data / "run7_scaling.json", STD)
    assert _load(ckpt, str(data))["scaling_source"].endswith("run7_scaling.json")


def test_ambiguous_directory_is_a_json_error(tmp_path):
    ckpt = _save_ckpt(tmp_path / "ckpt" / "best.ckpt")
    data = tmp_path / "dataset"
    _write(data / "a_scaling.json", STD)
    _write(data / "b_scaling.json", STD)
    result = _load(ckpt, str(data))
    assert "2 files match" in result["error"] and result["fix"]
    assert result["json_files"] == ["a_scaling.json", "b_scaling.json"]


def test_unusable_scaling_files_are_json_errors(tmp_path):
    ckpt = _save_ckpt(tmp_path / "ckpt" / "best.ckpt")
    assert "not found" in _load(ckpt, str(tmp_path / "missing"))["error"]
    bad = _write(tmp_path / "bad.json", {"something": {}})
    assert "no scaling_method" in _load(ckpt, str(bad))["error"]
    partial = _write(tmp_path / "partial.json", {"scaling_method": "standardize", "mean": 1.0})
    assert "std" in _load(ckpt, str(partial))["error"]


def test_scaling_params_next_to_the_checkpoint_are_found_by_default(tmp_path):
    ckpt = _save_ckpt(tmp_path / "ds" / "standardize" / "best_SimpleCNN.ckpt")
    _write(ckpt.parent / "scaling_params.json", STD)
    assert _load(ckpt)["scaling_params"] == STD


def test_no_scaling_params_is_not_an_error(tmp_path):
    ckpt = _save_ckpt(tmp_path / "best.ckpt")
    result = _load(ckpt)
    assert result["scaling_params"] is None and result["scaling_source"] is None


# ---------------------------------------------------------------------------
# tfbs_predict: original units by default, explicit opt-out, always a list
# ---------------------------------------------------------------------------

def test_predict_inverts_by_default_once_scaling_params_are_loaded(study):
    root, ckpt = study
    _load(ckpt, str(root / "scaling_params"), model_id="lm_pred")
    default = run(server.tfbs_predict("lm_pred", sequences=[SEQ]))
    raw = run(server.tfbs_predict("lm_pred", sequences=[SEQ], inverse_transform=False))
    explicit = run(server.tfbs_predict("lm_pred", sequences=[SEQ], inverse_transform=True))
    assert default["inverse_transformed"] is True and raw["inverse_transformed"] is False
    assert isinstance(default["scores"], list) and len(default["scores"]) == 1
    assert default["scores"][0] == pytest.approx(raw["scores"][0] * 2.0 + 5.0)
    assert explicit["scores"] == default["scores"]


def test_predict_without_scaling_params_reports_the_training_scale(tmp_path):
    ckpt = _save_ckpt(tmp_path / "best.ckpt")
    _load(ckpt, model_id="lm_plain")
    result = run(server.tfbs_predict("lm_plain", sequences=[SEQ]))
    assert result["inverse_transformed"] is False
    assert isinstance(result["scores"], list)


def test_classifier_scores_are_never_transformed():
    server._model_registry["lm_cls"] = {
        "model": torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(40, 1)).eval(),
        "model_type": "Seq", "device": "cpu", "input_length": 10, "classify": True,
        "scaling_params": STD,
    }
    result = run(server.tfbs_predict("lm_cls", sequences=[SEQ]))
    assert result["inverse_transformed"] is False


def test_single_sequence_scores_are_a_list_in_both_modes(tmp_path):
    ckpt = _save_ckpt(tmp_path / "best.ckpt")
    _load(ckpt, model_id="lm_one")
    normal = run(server.tfbs_predict("lm_one", sequences=[SEQ]))
    assert isinstance(normal["scores"], list) and len(normal["scores"]) == 1
    window = run(server.tfbs_predict("lm_one", sequences=[SEQ * 3], window_mode=True,
                                     window_size=10))
    assert isinstance(window["best_scores"], list) and len(window["best_scores"]) == 1
    assert isinstance(window["mean_scores"], list) and len(window["mean_scores"]) == 1


def test_predict_description_states_the_default():
    doc = " ".join(server.tfbs_predict.__doc__.split())
    assert "Default null, because" in doc and "explicit opt-out" in doc
    assert "always a list" in doc


# ---------------------------------------------------------------------------
# Scaling parameters without data_path: the lookup order
# ---------------------------------------------------------------------------

def test_training_data_dir_recorded_in_the_checkpoint_is_searched(tmp_path):
    data = tmp_path / "peaks_dataset"
    _write(data / "scaling_params.json", STD)
    ckpt = _save_ckpt(tmp_path / "saved_models" / "peaks" / "standardize" / "best_SimpleCNN.ckpt",
                      datamodule={"data_path": str(data), "scaling_method": "standardize"})
    result = _load(ckpt)
    assert result["scaling_params"] == STD
    assert Path(result["scaling_source"]) == data / "scaling_params.json"
    assert result["checkpoint_scaling_method"] == "standardize" and "warning" not in result


def test_scaling_params_next_to_the_checkpoint_come_first(tmp_path):
    data = tmp_path / "data"
    _write(data / "scaling_params.json", {**STD, "mean": 1.0})
    ckpt = _save_ckpt(tmp_path / "m" / "best_SimpleCNN.ckpt",
                      datamodule={"data_path": str(data), "scaling_method": "standardize"})
    _write(ckpt.parent / "scaling_params.json", STD)
    assert _load(ckpt)["scaling_params"] == STD


def test_study_layout_is_found_without_data_path(study):
    root, ckpt = study
    result = _load(ckpt)
    assert result["scaling_params"] == STD
    assert Path(result["scaling_source"]) == root / "scaling_params" / "all_mean_standardize_SimpleCNN.json"
    (root / "scaling_params" / "core_mean_standardize_SimpleCNN.json").unlink()
    core = _load(root / "core_mean" / "standardize" / "best_SimpleCNN.ckpt", model_id="lm_core")
    assert core["scaling_params"]["mean"] == -1.0
    assert core["scaling_source"].endswith("scaling_params_all_models.json")


def test_recorded_scaling_method_must_match_the_file(tmp_path):
    data = tmp_path / "data"
    _write(data / "scaling_params.json", STD)
    ckpt = _save_ckpt(tmp_path / "m" / "best_SimpleCNN.ckpt",
                      datamodule={"data_path": str(data), "scaling_method": "normalize"})
    result = _load(ckpt)
    assert "'standardize'" in result["error"] and "'normalize'" in result["error"]
    assert result["scaling_source"] == str(data / "scaling_params.json")


def test_regression_checkpoint_without_scaling_params_warns(tmp_path):
    ckpt = _save_ckpt(tmp_path / "ds" / "standardize" / "best_SimpleCNN.ckpt",
                      datamodule={"data_path": "/cluster/only/data", "scaling_method": "standardize"})
    result = _load(ckpt)
    assert result["scaling_params"] is None
    assert "data_path" in result["warning"] and "standardize" in result["warning"]
    assert "/cluster/only/data" in result["warning"].replace("\\", "/")
    unscaled = _save_ckpt(tmp_path / "ds" / "none" / "best_SimpleCNN.ckpt")
    assert "warning" not in _load(unscaled, model_id="lm_none")


def test_predict_insisting_on_original_units_needs_scaling_params(tmp_path):
    ckpt = _save_ckpt(tmp_path / "best.ckpt")
    _load(ckpt, model_id="lm_plain")
    result = run(server.tfbs_predict("lm_plain", sequences=[SEQ], inverse_transform=True))
    assert "scaling parameters" in result["error"] and "data_path" in result["fix"]
    assert "scores" in run(server.tfbs_predict("lm_plain", sequences=[SEQ]))


STUDY = Path("D:/claude_code/TF-NN")
STUDY_CKPT = STUDY / "saved_models_final" / "flank_mean" / "standardize" / "best_VCNNBpnet.ckpt"


@pytest.mark.skipif(not STUDY_CKPT.is_file(), reason="the NKX2-1 study checkpoints are not on this machine")
def test_study_checkpoint_scores_in_log2_fold_change(monkeypatch):
    monkeypatch.syspath_prepend(str(STUDY))
    result = run(server.tfbs_load_model("lm_study", str(STUDY_CKPT), "VCNNBpnet", device="cpu"))
    assert "error" not in result, result
    expected = STUDY / "saved_models_final" / "scaling_params" / "flank_mean_standardize_VCNNBpnet.json"
    assert Path(result["scaling_source"]) == expected
    scores = run(server.tfbs_predict("lm_study", sequences=["GGGCTCCTCACTTGACCTTTGCTG"]))
    assert scores["inverse_transformed"] is True
    assert scores["scores"][0] == pytest.approx(1.3907, abs=1e-3)


def test_a_checkpoint_saved_under_a_shim_package_on_disk_loads_with_lightning(tmp_path, monkeypatch):
    """The TF-NN situation: classes.nn.models re-exports tfbs.nn.models from a workspace."""
    root = tmp_path / "workspace"
    pkg = root / "diskshim_w" / "nn"
    pkg.mkdir(parents=True)
    (root / "diskshim_w" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "models.py").write_text("from tfbs.nn.models import SimpleCNN  # noqa: F401\n")
    monkeypatch.syspath_prepend(str(root))
    for key in ("diskshim_w", "diskshim_w.nn", "diskshim_w.nn.models"):
        monkeypatch.delitem(sys.modules, key, raising=False)
    ckpt = _save_ckpt(tmp_path / "ds" / "none" / "best_SimpleCNN.ckpt",
                      "diskshim_w.nn.models.SimpleCNN",
                      datamodule={"data_path": str(tmp_path / "gone"), "scaling_method": "none"})
    try:
        result = run(server.tfbs_load_model("lm_disk", str(ckpt), "SimpleCNN", device="cpu"))
        assert "error" not in result, result
        assert "diskshim_w.nn.models" in sys.modules
        model = server._model_registry["lm_disk"]["model"]
        assert isinstance(model, models.SimpleCNN) and isinstance(model, pl.LightningModule)
        reference = models.SimpleCNN(input_length=10)
        reference.load_state_dict(torch.load(ckpt, weights_only=True)["state_dict"])
        x = torch.zeros(2, 4, 10)
        x[:, 2, :] = 1.0
        assert torch.allclose(model(x), reference.eval()(x))
        assert run(server.tfbs_predict("lm_disk", sequences=[SEQ]))["inverse_transformed"] is False
    finally:
        for key in ("diskshim_w", "diskshim_w.nn", "diskshim_w.nn.models"):
            sys.modules.pop(key, None)


# ---------------------------------------------------------------------------
# Inference checkpoints (the NKX2-1 study's released vcnnbpnet.pt format)
# ---------------------------------------------------------------------------

def _tiny_vcnn() -> models.VCNNBpnet:
    torch.manual_seed(0)
    return models.VCNNBpnet(num_channels=4, kernel_size=3, dilations=[1, 2],
                            pool_output_size=5, dense_sizes=[8], input_length=10).eval()


def _save_inference(path: Path, model=None, scaling=STD, **extra) -> Path:
    model = model or _tiny_vcnn()
    payload = {"arch": type(model).__name__, "library": "demo",
               "hyper_parameters": dict(model.hparams), "state_dict": model.state_dict(),
               "metadata": {"study": "test", "epoch": 3}, **extra}
    if scaling is not None:
        payload["scaling"] = scaling
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _load_vcnn(path, data_path=None, model_id="lm_inf"):
    return run(server.tfbs_load_model(model_id, str(path), "VCNNBpnet", device="cpu",
                                      data_path=data_path))


def test_inference_checkpoint_loads_with_its_own_scaling(tmp_path):
    model = _tiny_vcnn()
    path = _save_inference(tmp_path / "rel" / "vcnnbpnet.pt", model)
    result = _load_vcnn(path)
    assert "error" not in result, result
    assert result["checkpoint_format"] == "inference"
    assert result["input_length"] == 10 and result["classify"] is False
    assert result["scaling_params"] == STD and result["scaling_source"] == str(path.resolve())
    assert result["checkpoint_scaling_method"] == "standardize"
    assert result["library"] == "demo" and result["metadata"] == {"study": "test", "epoch": 3}
    assert "warning" not in result

    scores = run(server.tfbs_predict("lm_inf", sequences=[SEQ]))
    x = server._sequences_to_tensor([SEQ])
    with torch.no_grad():
        expected = float(model(x).flatten()[0]) * 2.0 + 5.0
    assert scores["scores"][0] == pytest.approx(expected, rel=1e-5)
    assert scores["inverse_transformed"] is True and scores["units"] == "label"


def test_inference_checkpoint_takes_an_agreeing_data_path_only(tmp_path):
    path = _save_inference(tmp_path / "rel" / "vcnnbpnet.pt")
    same = _write(tmp_path / "same.json", STD)
    assert _load_vcnn(path, str(same))["scaling_params"] == STD
    other = _write(tmp_path / "other.json", {**STD, "mean": 1.0})
    result = _load_vcnn(path, str(other))
    assert "disagrees" in result["error"] and "Omit data_path" in result["fix"]
    method = _write(tmp_path / "method.json", {"scaling_method": "normalize", "min": 0.0, "max": 1.0})
    assert "'standardize'" in _load_vcnn(path, str(method))["error"]


def test_inference_checkpoint_without_scaling_uses_the_lookup(tmp_path):
    path = _save_inference(tmp_path / "rel" / "vcnnbpnet.pt", scaling=None)
    assert _load_vcnn(path)["scaling_params"] is None
    _write(path.parent / "scaling_params.json", STD)
    assert _load_vcnn(path)["scaling_params"] == STD


def test_inference_checkpoint_errors_are_json(tmp_path):
    path = _save_inference(tmp_path / "rel" / "vcnnbpnet.pt")
    wrong = run(server.tfbs_load_model("lm_inf", str(path), "SimpleCNN", device="cpu"))
    assert "VCNNBpnet" in wrong["error"] and "model_type='VCNNBpnet'" in wrong["fix"]
    model = _tiny_vcnn()
    state = model.state_dict()
    state.pop(next(iter(state)))
    broken = _save_inference(tmp_path / "broken.pt", model, state_dict=state)
    result = _load_vcnn(broken)
    assert "Could not build VCNNBpnet" in result["error"] and result["fix"]
    partial = _save_inference(tmp_path / "partial.pt", scaling={"scaling_method": "standardize"})
    assert "lacks" in _load_vcnn(partial)["error"]
    assert "lm_inf" not in server._model_registry


def test_lightning_checkpoints_report_their_format(tmp_path):
    ckpt = _save_ckpt(tmp_path / "best.ckpt")
    assert _load(ckpt)["checkpoint_format"] == "lightning"


def test_load_model_description_names_both_formats():
    doc = " ".join(server.tfbs_load_model.__doc__.split())
    assert "Two checkpoint formats are accepted" in doc
    assert "doi:10.5281/zenodo.21399578" in doc and "Lightning checkpoint" in doc


RELEASED_FLANK = Path("D:/claude_code/nkx2-1-tfbs-variants/models/flank/vcnnbpnet.pt")


@pytest.mark.skipif(not RELEASED_FLANK.is_file(), reason="the released NKX2-1 checkpoints are not on this machine")
def test_released_flank_checkpoint_scores_in_log2_fold_change():
    result = _load_vcnn(RELEASED_FLANK, model_id="lm_flank")
    assert "error" not in result, result
    assert result["checkpoint_format"] == "inference" and result["library"] == "flank"
    assert result["input_length"] == 24
    assert result["scaling_params"]["mean"] == pytest.approx(-0.2345, abs=1e-4)
    assert result["scaling_params"]["std"] == pytest.approx(1.3395, abs=1e-4)
    scores = run(server.tfbs_predict("lm_flank", sequences=["GGGCTCCTCACTTGACCTTTGCTG"]))
    assert scores["scores"][0] == pytest.approx(1.3907, abs=1e-3)
    assert scores["units"] == "label"


# ---------------------------------------------------------------------------
# units and inverse_transformed in predict, mutagenesis, marginalize, epistasis
# ---------------------------------------------------------------------------

def _units_of_every_tool(model_id: str) -> list[tuple[bool, str]]:
    calls = [
        server.tfbs_predict(model_id, sequences=[SEQ]),
        server.tfbs_mutagenesis(model_id, sequences=[SEQ]),
        server.tfbs_marginalize(model_id, sequences=[SEQ], motif="ACG"),
        server.tfbs_epistasis(model_id, sequence=SEQ, positions=[0, 1]),
    ]
    results = [run(c) for c in calls]
    for r in results:
        assert "error" not in r, r
    return [(r["inverse_transformed"], r["units"]) for r in results]


def test_unscaled_model_is_in_label_units_without_an_inverse(tmp_path):
    ckpt = _save_ckpt(tmp_path / "ds" / "none" / "best_SimpleCNN.ckpt")
    _write(ckpt.parent / "scaling_params.json", {"scaling_method": "none"})
    assert _load(ckpt, model_id="lm_none")["scaling_params"] == {"scaling_method": "none"}
    assert _units_of_every_tool("lm_none") == [(False, "label")] * 4
    insisted = run(server.tfbs_predict("lm_none", sequences=[SEQ], inverse_transform=True))
    assert insisted["inverse_transformed"] is False and insisted["units"] == "label"


def test_scaled_model_reports_label_or_training_scale(study, tmp_path):
    root, ckpt = study
    _load(ckpt, str(root / "scaling_params"), model_id="lm_std")
    assert _units_of_every_tool("lm_std") == [(True, "label")] * 4
    raw = run(server.tfbs_predict("lm_std", sequences=[SEQ], inverse_transform=False))
    assert raw["inverse_transformed"] is False and raw["units"] == "training_scale"

    missing = _save_ckpt(tmp_path / "x" / "standardize" / "best_SimpleCNN.ckpt",
                         datamodule={"data_path": str(tmp_path / "gone"),
                                     "scaling_method": "standardize"})
    _load(missing, model_id="lm_missing")
    assert _units_of_every_tool("lm_missing") == [(False, "training_scale")] * 4


def test_units_are_described_in_every_tool():
    for tool in (server.tfbs_predict, server.tfbs_mutagenesis, server.tfbs_marginalize,
                 server.tfbs_epistasis):
        doc = " ".join(tool.__doc__.split())
        assert '"units"' in doc or "units," in doc, tool.__name__
    doc = " ".join(server.tfbs_predict.__doc__.split())
    assert '"training_scale"' in doc and 'scaling_method "none"' in doc


def test_readme_says_the_released_checkpoints_load_directly():
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    text = " ".join(readme.split())
    assert "released checkpoints" in text and "load directly with `tfbs_load_model`" in text
    assert "doi:10.5281/zenodo.21399578" in text


def test_inference_metadata_with_tensors_is_reported_as_json(tmp_path):
    metadata = {"epoch": torch.tensor(3), "curve": torch.tensor([0.5, 0.25]),
                "nested": {"ids": (torch.tensor(1), 2)}}
    path = _save_inference(tmp_path / "rel" / "vcnnbpnet.pt", metadata=metadata,
                           library=torch.tensor([1, 2]))
    result = _load_vcnn(path)
    assert "error" not in result, result
    assert result["metadata"] == {"epoch": 3, "curve": [0.5, 0.25], "nested": {"ids": [1, 2]}}
    assert result["library"] == [1, 2]
    assert "lm_inf" in server._model_registry


def test_unreportable_metadata_leaves_no_model_registered(tmp_path):
    loop: dict = {"name": "loop"}
    loop["self"] = loop
    path = _save_inference(tmp_path / "rel" / "vcnnbpnet.pt", metadata=loop)
    result = _load_vcnn(path, model_id="lm_loop")
    assert "metadata" in result["error"] and result["fix"]
    assert "lm_loop" not in server._model_registry


@pytest.mark.parametrize("scaling, expected", [
    ({**STD, "std": float("nan")}, "not a finite number"),
    ({**STD, "mean": float("inf")}, "not a finite number"),
    ({**STD, "mean": True}, "not a finite number"),
    ({**STD, "std": "2"}, "not a finite number"),
    ({**STD, "std": 0.0}, "std 0"),
    ({"scaling_method": "normalize", "min": 1.0, "max": 1.0}, "max equal to min"),
])
def test_unusable_scaling_values_are_json_errors(tmp_path, scaling, expected):
    path = _save_inference(tmp_path / "rel" / "vcnnbpnet.pt", scaling=scaling)
    result = _load_vcnn(path)
    assert expected in result["error"] and result["fix"], result
    assert "lm_inf" not in server._model_registry
    as_file = tmp_path / "scaling_params.json"
    as_file.write_text(json.dumps(scaling))
    plain = _save_inference(tmp_path / "plain" / "vcnnbpnet.pt", scaling=None)
    result = _load_vcnn(plain, str(as_file))
    assert expected in result["error"], result


def test_inference_regressor_without_any_scaling_has_unknown_units(tmp_path):
    path = _save_inference(tmp_path / "rel" / "vcnnbpnet.pt", scaling=None)
    result = _load_vcnn(path)
    assert result["scaling_params"] is None and result["units"] == "unknown"
    assert "data_path" in result["warning"]
    assert _units_of_every_tool("lm_inf") == [(False, "unknown")] * 4

    _write(path.parent / "scaling_params.json", STD)
    found = _load_vcnn(path)
    assert "warning" not in found and "units" not in found
    assert _units_of_every_tool("lm_inf") == [(True, "label")] * 4


def test_load_model_description_states_unknown_units_and_the_label_assumption():
    doc = " ".join(server.tfbs_load_model.__doc__.split())
    assert 'units "unknown"' in doc and "assume it was trained on unscaled labels" in doc
    doc = " ".join(server.tfbs_predict.__doc__.split())
    assert '"unknown"' in doc and "assumes it was trained on unscaled labels" in doc


@pytest.mark.parametrize("scaling_params, classify", [
    (STD, False), ({"scaling_method": "normalize", "min": 0.0, "max": 4.0}, False),
    ({"scaling_method": "none"}, False), (None, False), (STD, True),
])
def test_delta_scale_exists_exactly_when_the_inverse_does(scaling_params, classify):
    entry = {"scaling_params": scaling_params, "classify": classify}
    assert (server._delta_scale(entry) is None) == (server._inverse_scaling(entry, None) is None)
