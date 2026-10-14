"""Pipeline, model and version fixes of the paper audit (application note, 1.0.0).

- The sweep agent reads only keys that the bundled sweep YAML or the task's
  fixed values define, so the bundled sweep trains RNN, VCNNBpnet and
  BPNetReal (tfbs/pipelines/run_sweep.py).
- FlexibleCNN's class default reads the one-hot input (input_channels=4),
  so every template other than VCNNBpnet's and RNN's holds the class
  defaults apart from input_length.
- The ChIP-seq method comparison and the ClinVar validation, given a
  structure, stop at startup without a FoldX binary, as the ReMap
  validation does, and all three name the tools' skip_foldx.
- The package version is 1.0.1.
"""
import ast
import inspect
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import tfbs
import tfbs.configs
import tfbs.pipelines
from tfbs.nn import models
from tfbs.nn.loading import load_checkpoint
from tfbs.pipelines import chipseq_benchmark_with_foldx as cbf
from tfbs.pipelines import disease_variant_validation as dv
from tfbs.pipelines import remap_validation as rv
from tfbs.pipelines import run_sweep

REPO = Path(__file__).resolve().parent.parent
SWEEP_YAML = tfbs.configs.config_dir() / "wandb_sweep_config.yaml"
SWEEP_MODELS = ("RNN", "VCNNBpnet", "BPNetReal")


# ---------------------------------------------------------------------------
# B1: the bundled sweep YAML and the keys the agent reads
# ---------------------------------------------------------------------------

class _StrictConfig(dict):
    """wandb.config stand-in: a key the sweep does not define is an error."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"the sweep config has no key {name!r}") from exc

    def update(self, values, allow_val_change=False):
        super().update(values)


class _SweepConfig(_StrictConfig):
    """A trial's wandb.config: the keys the sweep defines are locked.

    wandb ignores an update of a locked key (with a warning), also with
    allow_val_change=True, so only new keys are added.
    """

    def update(self, values, allow_val_change=False):
        super().update({k: v for k, v in values.items() if k not in self})


def _fake_wandb(config):
    fake = types.ModuleType("wandb")
    fake.run = None
    fake.config = config
    fake.agent_calls = []
    fake.logged = []

    def init():
        fake.run = types.SimpleNamespace(name="trial-1")
        return fake.run

    fake.init = init
    fake.log = fake.logged.append
    fake.agent = lambda sweep_id, **kw: fake.agent_calls.append((sweep_id, kw))
    return fake


def _sweep_parameters() -> dict:
    with open(SWEEP_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["parameters"]


def _sample(spec: dict, pick: str):
    """One value of a sweep parameter: its lowest ("low") or highest ("high") choice."""
    if "value" in spec:
        return spec["value"]
    if "values" in spec:
        return spec["values"][0 if pick == "low" else -1]
    return spec["min" if pick == "low" else "max"]


def _fixed_config(monkeypatch, model: str) -> dict:
    """The task's fixed values, as run_agent hands them to every trial."""
    fake = _fake_wandb(_StrictConfig())
    monkeypatch.setitem(sys.modules, "wandb", fake)
    run_sweep.main(["--model", model, "--data-path", "/d", "--data-name", "n",
                    "--scaling-method", "standardize", "--project-name", "p",
                    "--sweep-id", "sw1", "--n-trials", "1"])
    return dict(fake.agent_calls[-1][1]["function"].args[0])


def _trial_config(monkeypatch, model: str, pick: str) -> _StrictConfig:
    config = _StrictConfig({k: _sample(v, pick) for k, v in _sweep_parameters().items()})
    config.update(_fixed_config(monkeypatch, model))
    return config


def _config_keys_read(source: str) -> set[str]:
    """Every name read as config.<name> in a module's source."""
    return {node.attr for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "config"}


def test_every_key_the_agent_reads_is_in_the_bundled_yaml_or_the_fixed_config(monkeypatch):
    source = (REPO / "tfbs" / "pipelines" / "run_sweep.py").read_text(encoding="utf-8")
    read = _config_keys_read(source)
    assert {"num_channels", "pool_output_size", "VCNN_kernel_size"} <= read
    assert not {"VCNN_num_channels", "VCNN_pool_output_size"} & read
    defined = set(_sweep_parameters()) | set(_fixed_config(monkeypatch, "VCNNBpnet"))
    assert read <= defined, sorted(read - defined)


@pytest.mark.parametrize("pick", ["low", "high"])
@pytest.mark.parametrize("model", SWEEP_MODELS)
def test_get_model_params_builds_each_model_from_the_bundled_yaml(model, pick, monkeypatch):
    config = _trial_config(monkeypatch, model, pick)
    params = run_sweep.get_model_params(model, config)
    if model == "VCNNBpnet":
        assert params["num_channels"] == config["num_channels"]
        assert params["pool_output_size"] == config["pool_output_size"]
    params.update({"input_channels": 4, "input_length": run_sweep.SEQUENCE_LENGTH,
                   "learning_rate": config.lr})
    net = run_sweep.get_model_class(model)(**params).eval()
    x = torch.nn.functional.one_hot(torch.randint(0, 4, (3, run_sweep.SEQUENCE_LENGTH)),
                                    num_classes=4).float().permute(0, 2, 1)
    with torch.no_grad():
        assert net(x).shape == (3, 1)


@pytest.mark.parametrize("model", SWEEP_MODELS)
def test_a_trial_of_the_bundled_sweep_trains_each_model(model, tmp_path, monkeypatch):
    fixed = _fixed_config(monkeypatch, model)
    fixed["checkpoint_dir"] = str(tmp_path / "ckpt")
    # The trial's config holds every key of the YAML, the placeholders of the
    # task's fixed values (data_path /path/, ...) included, locked as wandb
    # locks sweep parameters; the agent must still use the task's values.
    sampled = {k: _sample(v, "low") for k, v in _sweep_parameters().items()}
    assert {"model", "data_path", "checkpoint_dir"} <= set(sampled)
    fake = _fake_wandb(_SweepConfig(sampled))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    seen = {}

    class DataModule:
        def __init__(self, **kwargs):
            seen["data"] = kwargs

    class Trainer:
        def __init__(self, **kwargs):
            seen["trainer"] = kwargs

        def fit(self, net, data):
            seen["fit"] = net

    monkeypatch.setattr(run_sweep, "TFBSDataModule", DataModule)
    monkeypatch.setattr(run_sweep.pl, "Trainer", Trainer)
    monkeypatch.setattr(run_sweep, "WandbLogger", lambda experiment: ("logger", experiment))
    run_sweep.train_sweep_agent(fixed)

    net = seen["fit"]
    assert type(net).__name__ == model
    assert net.hparams["input_length"] == 24 and not net.classify
    assert seen["trainer"]["max_epochs"] == 100
    early = [c for c in seen["trainer"]["callbacks"]
             if type(c).__name__ == "EarlyStopping"]
    assert early and early[0].patience == 10
    assert (tmp_path / "ckpt" / "n" / "standardize" / model).is_dir()
    assert seen["data"]["data_path"] == "/d"
    assert seen["data"]["scaling_method"] == "standardize"
    assert fake.config["task_data_path"] == "/d" and fake.config["data_path"] == sampled["data_path"]
    assert fake.logged == []


def test_the_agent_exits_nonzero_when_a_trial_failed(monkeypatch):
    """wandb.agent swallows a trial's exception, so the job must fail on its own."""
    fake = _fake_wandb(_SweepConfig())

    def agent(sweep_id, function, count, **kw):
        for _ in range(count):
            try:
                function()
            except Exception:
                pass

    fake.agent = agent
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setattr(run_sweep, "get_model_class",
                        lambda name: (_ for _ in ()).throw(RuntimeError("boom")))
    argv = ["--model", "VCNNBpnet", "--data-path", "/d", "--data-name", "n",
            "--scaling-method", "none", "--project-name", "p", "--sweep-id", "sw1",
            "--n-trials", "2"]
    with pytest.raises(SystemExit) as stop:
        run_sweep.main(argv)
    assert "2 of 2 trials failed" in str(stop.value) and "RuntimeError: boom" in str(stop.value)

    monkeypatch.setattr(run_sweep, "train_sweep_agent", lambda fixed: None)
    run_sweep.main(argv)  # every trial finished: no exit
    assert run_sweep.FAILED_TRIALS == []


def test_the_agent_defaults_are_the_documented_ones():
    parser = run_sweep.build_parser()
    defaults = {a.dest: a.default for a in parser._actions}
    assert defaults["n_trials"] == 50
    assert defaults["checkpoint_dir"] == "saved_models_tuned/"
    assert run_sweep.MAX_EPOCHS == 100 and run_sweep.SEQUENCE_LENGTH == 24


# ---------------------------------------------------------------------------
# B2: FlexibleCNN reads the one-hot input by default
# ---------------------------------------------------------------------------

def _norm(value):
    return [_norm(v) for v in value] if isinstance(value, (list, tuple)) else value


def test_flexible_cnn_default_reads_four_channels():
    default = inspect.signature(models.FlexibleCNN.__init__).parameters["input_channels"].default
    assert default == 4
    net = models.FlexibleCNN(input_length=24).eval()
    assert net.convs[0].in_channels == 4
    with torch.no_grad():
        assert net(torch.zeros(2, 4, 24)).shape == (2, 1)


def test_templates_other_than_vcnnbpnet_and_rnn_hold_the_class_defaults():
    folder = tfbs.configs.config_dir() / "models"
    checked = []
    for path in sorted(folder.glob("*.yaml")):
        template = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = template["class_path"].rsplit(".", 1)[1]
        if name in ("VCNNBpnet", "RNN"):
            continue
        params = inspect.signature(getattr(models, name).__init__).parameters
        differ = {k: (params[k].default if k in params else None, v)
                  for k, v in template["init_args"].items()
                  if k != "input_length" and (k not in params
                                              or _norm(params[k].default) != _norm(v))}
        assert differ == {}, f"{path.name}: {differ}"
        checked.append(name)
    assert "FlexibleCNN" in checked and len(checked) == 11


def test_a_checkpoint_of_a_one_channel_flexible_cnn_still_loads(tmp_path):
    import pytorch_lightning as pl

    torch.manual_seed(0)
    old = models.FlexibleCNN(input_length=24, input_channels=1)
    hparams = dict(old.hparams)
    assert hparams["input_channels"] == 1
    path = tmp_path / "flexible.ckpt"
    torch.save({"state_dict": old.state_dict(), "hyper_parameters": hparams,
                "pytorch-lightning_version": pl.__version__}, path)
    net, _ = load_checkpoint(str(path), "FlexibleCNN")
    assert net.convs[0].in_channels == 1


# ---------------------------------------------------------------------------
# A6: a structure without FoldX stops the pipeline at startup
# ---------------------------------------------------------------------------

class _Reached(Exception):
    """Raised by a stub of the first step after the startup checks."""


def _no_foldx(monkeypatch, module):
    monkeypatch.delenv("TFBS_FOLDX_BIN", raising=False)
    monkeypatch.setattr(module.shutil, "which",
                        lambda name, *a, **k: "/usr/bin/fimo" if name == "fimo" else None)


def _tool_message() -> str:
    from tfbs.mcp.foldx_tools import _foldx_missing_error

    error = json.loads(_foldx_missing_error())
    return f"{error['error']} {error['fix']} {tfbs.pipelines.SKIP_FOLDX_HINT}"


def test_the_three_pipelines_share_one_foldx_lookup_and_message():
    assert rv.find_foldx is tfbs.pipelines.find_foldx
    assert cbf.find_foldx is tfbs.pipelines.find_foldx
    assert dv.find_foldx is tfbs.pipelines.find_foldx
    message = tfbs.pipelines.foldx_required_message()
    assert message == _tool_message()
    assert message.endswith(
        "Or call the tool with skip_foldx=true to run the other scores without FoldX.")
    assert chr(0x2014) not in message and chr(0x2013) not in message


def test_the_foldx_missing_message_does_not_import_the_mcp_server():
    """A pipeline run by hand on a library-only install (no mcp) can still say why it stops."""
    import os
    import subprocess

    code = ("import sys, tfbs.pipelines as p; p.foldx_required_message(); "
            "print('tfbs.mcp.server' in sys.modules, 'mcp' in sys.modules)")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, cwd=str(REPO), timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False"]


def test_find_foldx_prefers_the_flag_then_the_environment_then_path(monkeypatch):
    on_path = {}
    monkeypatch.setattr(tfbs.pipelines.shutil, "which", on_path.get)
    monkeypatch.delenv("TFBS_FOLDX_BIN", raising=False)
    find = tfbs.pipelines.find_foldx
    assert find() is None
    on_path["foldx5"] = "/opt/foldx5"
    assert find() == "/opt/foldx5"
    on_path["foldx"] = "/opt/foldx"
    assert find() == "/opt/foldx"
    monkeypatch.setenv("TFBS_FOLDX_BIN", "/env/foldx")
    assert find() == "/env/foldx"
    assert find("/flag/foldx") == "/flag/foldx"


def _intersect(tmp_path: Path) -> Path:
    folder = tmp_path / "intersect"
    folder.mkdir()
    for tf, offset in (("NKX2-1", 50), ("GATA1", 250)):
        lines = ["chrom\tstart\tend\tnum"]
        lines += [f"chr1\t{offset + 200 * i}\t{offset + 200 * i + 30}\t3" for i in range(12)]
        (folder / f"{tf}.allintersect.bed").write_text("\n".join(lines) + "\n")
    return folder


def _comparison_args(tmp_path: Path, *extra: str) -> list[str]:
    return ["--intersect-dir", str(tmp_path / "intersect"), "--output-dir", str(tmp_path / "out"),
            "--genome", str(tmp_path / "genome.fa"), "--motif-file", "motif.meme",
            "--neg-tfs", "GATA1", "--checkpoints", f"flank={tmp_path / 'm.pt'}", *extra]


def test_method_comparison_with_a_structure_and_no_foldx_stops_before_any_work(tmp_path,
                                                                             monkeypatch):
    _intersect(tmp_path)
    _no_foldx(monkeypatch, cbf)

    def must_not_run(*a, **k):
        raise AssertionError("the sequences were read before the FoldX check")

    monkeypatch.setattr(cbf, "merge_intervals", must_not_run)
    monkeypatch.setattr(cbf, "resolve_checkpoints", must_not_run)
    with pytest.raises(SystemExit) as stop:
        cbf.main(_comparison_args(tmp_path, "--foldx-pdb", str(tmp_path / "complex.pdb")))
    assert stop.value.code == _tool_message()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("foldx_args", [[], ["--foldx-pdb", ""]])
def test_method_comparison_without_a_structure_needs_no_foldx(foldx_args, tmp_path, monkeypatch):
    _intersect(tmp_path)
    _no_foldx(monkeypatch, cbf)

    def reached(*a, **k):
        raise _Reached

    monkeypatch.setattr(cbf, "merge_intervals", reached)
    with pytest.raises(_Reached):
        cbf.main(_comparison_args(tmp_path, *foldx_args))


def test_method_comparison_finds_foldx_in_the_environment(tmp_path, monkeypatch):
    _intersect(tmp_path)
    _no_foldx(monkeypatch, cbf)
    monkeypatch.setenv("TFBS_FOLDX_BIN", "/env/foldx")

    def reached(*a, **k):
        raise _Reached

    monkeypatch.setattr(cbf, "merge_intervals", reached)
    with pytest.raises(_Reached):
        cbf.main(_comparison_args(tmp_path, "--foldx-pdb", str(tmp_path / "complex.pdb")))


def _genome(tmp_path: Path, length: int = 3000) -> Path:
    rng = np.random.RandomState(7)
    seq = "".join(rng.choice(list("ACGT"), length))
    fa = tmp_path / "genome.fa"
    fa.write_text(">chr1\n" + "\n".join(seq[i:i + 60] for i in range(0, length, 60)) + "\n")
    return fa


def _tiny_checkpoint(path: Path) -> Path:
    torch.manual_seed(0)
    net = models.VCNNBpnet(num_channels=4, kernel_size=3, dilations=[1, 2],
                           pool_output_size=5, dense_sizes=[8], input_length=10).eval()
    torch.save({"arch": "VCNNBpnet", "hyper_parameters": dict(net.hparams),
                "state_dict": net.state_dict()}, path)
    return path


@pytest.mark.parametrize("structure", [False, True])
def test_method_comparison_runs_the_foldx_leg_only_with_a_structure(structure, tmp_path,
                                                                    monkeypatch):
    _intersect(tmp_path)
    _genome(tmp_path)
    _tiny_checkpoint(tmp_path / "m.pt")
    _no_foldx(monkeypatch, cbf)
    calls = []
    if structure:
        monkeypatch.setenv("TFBS_FOLDX_BIN", "/env/foldx")
    monkeypatch.setattr(cbf, "score_fimo", lambda seqs, motif: np.zeros(len(seqs)))
    monkeypatch.setattr(cbf, "score_foldx_batch",
                        lambda seqs, pdb, binary, n_workers: calls.append(binary)
                        or np.arange(len(seqs), dtype=float))
    extra = ["--foldx-pdb", str(tmp_path / "complex.pdb")] if structure else []
    cbf.main(_comparison_args(tmp_path, "--model-type", "VCNNBpnet", "--window", "10",
                              "--n-per-tf", "12", "--device", "cpu", *extra))
    raw = pd.read_csv(tmp_path / "out" / "raw_scores.csv")
    methods = set(pd.read_csv(tmp_path / "out" / "auc_summary.csv")["method"])
    assert ("FoldX" in raw.columns) is structure
    assert ("FoldX" in methods) is structure
    assert calls == (["/env/foldx", "/env/foldx"] if structure else [])


def _clinvar(tmp_path: Path) -> list[str]:
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t100\t200\n")
    vcf = tmp_path / "clinvar.vcf.gz"
    vcf.write_text("x")
    (tmp_path / "clinvar.vcf.gz.tbi").write_text("x")
    return ["--peaks-bed", str(bed), "--clinvar-vcf", str(vcf), "--genome", "g.fa",
            "--output-dir", str(tmp_path / "out")]


def test_disease_validation_with_a_structure_and_no_foldx_stops_before_any_work(tmp_path,
                                                                              monkeypatch):
    args = _clinvar(tmp_path)
    _no_foldx(monkeypatch, dv)

    def must_not_run(*a, **k):
        raise AssertionError("the peaks were read before the FoldX check")

    monkeypatch.setattr(dv, "merge_bed_intervals", must_not_run)
    with pytest.raises(SystemExit) as stop:
        dv.main(args + ["--foldx-pdb", str(tmp_path / "complex.pdb")])
    assert stop.value.code == _tool_message()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("with_binary", [False, True])
def test_disease_validation_checks_foldx_only_for_a_structure(with_binary, tmp_path, monkeypatch):
    args = _clinvar(tmp_path)
    _no_foldx(monkeypatch, dv)
    if with_binary:
        monkeypatch.setenv("TFBS_FOLDX_BIN", "/env/foldx")
        args += ["--foldx-pdb", str(tmp_path / "complex.pdb")]

    def reached(*a, **k):
        raise _Reached

    monkeypatch.setattr(dv, "merge_bed_intervals", reached)
    with pytest.raises(_Reached):
        dv.main(args)


def test_disease_foldx_leg_stops_instead_of_returning_empty_columns(monkeypatch):
    _no_foldx(monkeypatch, dv)
    df = pd.DataFrame([{"chrom": "chr1", "pos": 11, "ref": "T", "alt": "G",
                        "label": "pathogenic"}])
    with pytest.raises(SystemExit) as stop:
        dv.score_with_foldx(df, genome=None, foldx_pdb="complex.pdb", n_workers=1)
    assert stop.value.code == _tool_message()


def test_disease_foldx_leg_runs_the_binary_it_is_given(monkeypatch):
    from tfbs.mcp import foldx_tools

    _no_foldx(monkeypatch, dv)

    class Genome(dict):
        def __getitem__(self, chrom):
            return dict.__getitem__(self, chrom)

    genome = Genome({"chr1": "ACGTACGTAC" + "T" + "GCATGCATGC" + "A" * 30})
    binaries = []
    monkeypatch.setattr(foldx_tools, "_parse_dna_from_pdb",
                        lambda pdb: {"C": "GGTCAAGTGTT", "D": "AACACTTGACC"})
    monkeypatch.setattr(foldx_tools, "_build_foldx_mutation_string",
                        lambda ref, seq, chain: f"X{seq[5]}6;")
    monkeypatch.setattr(foldx_tools, "_run_foldx_energy",
                        lambda binary, pdb, wd, ms: binaries.append(binary) or -8.0)
    df = pd.DataFrame([{"chrom": "chr1", "pos": 11, "ref": "T", "alt": "G",
                        "label": "pathogenic"}])
    out = dv.score_with_foldx(df, genome, "complex.pdb", n_workers=1, foldx_bin="/flag/foldx")
    assert set(binaries) == {"/flag/foldx"} and len(binaries) == 2
    assert out.loc[0, "foldx_ddG"] == pytest.approx(0.0)


def test_remap_validation_names_skip_foldx(tmp_path, monkeypatch):
    _no_foldx(monkeypatch, rv)
    with pytest.raises(SystemExit) as stop:
        rv.main(["--remap-bed", str(tmp_path / "peaks.bed"), "--output-dir", str(tmp_path / "out"),
                 "--foldx-pdb", str(tmp_path / "complex.pdb")])
    assert stop.value.code == _tool_message()
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# B3: version 1.0.1
# ---------------------------------------------------------------------------

def test_the_package_version_is_1_0_0():
    tomllib = pytest.importorskip("tomllib")
    with open(REPO / "pyproject.toml", "rb") as fh:
        project = tomllib.load(fh)["project"]
    assert project["version"] == tfbs.__version__ == "1.0.1"
