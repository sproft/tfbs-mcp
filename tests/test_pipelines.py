"""The training and validation pipelines ship inside the package.

Six MCP tools run tfbs.pipelines.<name> as ``python -m``; these tests check
that every module imports and answers --help as a module, that the YAML
templates are package data, that a built wheel contains both, and that the
tools invoke the packaged modules instead of files under scripts/.
"""
import ast
import asyncio
import importlib
import json
import os
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest
import torch

import tfbs.configs
import tfbs.pipelines
from tfbs.mcp import cli
from tfbs.mcp import server

REPO = Path(__file__).resolve().parent.parent
PIPELINES = tfbs.pipelines.PIPELINES
SHIMS = {
    "validate": "scripts/validate.py",
    "run_sweep": "scripts/cli/run_sweep.py",
    "chipseq_benchmark": "scripts/chipseq_benchmark.py",
    "chipseq_benchmark_with_foldx": "scripts/chipseq_benchmark_with_foldx.py",
    "remap_validation": "scripts/remap_validation.py",
    "disease_variant_validation": "scripts/disease_variant_validation.py",
}


def run(coro):
    return json.loads(asyncio.run(coro))


def _env_with_repo_on_path():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    return env


# ---------------------------------------------------------------------------
# The modules themselves
# ---------------------------------------------------------------------------

def test_pipeline_lists_agree():
    assert tuple(cli.PIPELINES) == PIPELINES
    assert tuple(server._PIPELINE_MODULES) == PIPELINES
    assert set(SHIMS) == set(PIPELINES)


@pytest.mark.parametrize("name", PIPELINES)
def test_every_pipeline_module_imports(name):
    module = importlib.import_module(f"tfbs.pipelines.{name}")
    assert callable(module.main)


@pytest.mark.parametrize("name", PIPELINES)
def test_pipeline_answers_help_as_a_module(name, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", f"tfbs.pipelines.{name}", "--help"],
        capture_output=True, text=True, timeout=600, cwd=tmp_path,
        env=_env_with_repo_on_path(),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout


@pytest.mark.parametrize("name, optional", [
    ("disease_variant_validation", "pysam"),
    ("run_sweep", "wandb"),
])
def test_optional_dependencies_are_imported_only_when_used(name, optional):
    source = (REPO / "tfbs" / "pipelines" / f"{name}.py").read_text(encoding="utf-8")
    top_level = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert optional not in top_level


def test_remap_validation_imports_its_helpers_from_the_package():
    source = (REPO / "tfbs" / "pipelines" / "remap_validation.py").read_text(encoding="utf-8")
    assert "sys.path.insert" not in source
    assert "from tfbs.pipelines.chipseq_benchmark_with_foldx import" in source


@pytest.mark.parametrize("name", PIPELINES)
def test_repository_script_is_a_shim_for_the_packaged_module(name):
    text = (REPO / SHIMS[name]).read_text(encoding="utf-8")
    assert f"from tfbs.pipelines.{name} import main" in text
    assert len(text.splitlines()) < 20


def test_repository_shim_still_runs_as_a_script(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "remap_validation.py"), "--help"],
        capture_output=True, text=True, timeout=600, cwd=tmp_path,
        env=_env_with_repo_on_path(),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "--remap-bed" in result.stdout


# ---------------------------------------------------------------------------
# Templates as package data, and the built wheel
# ---------------------------------------------------------------------------

def test_model_templates_are_package_data():
    config_dir = tfbs.configs.config_dir()
    assert config_dir == Path(tfbs.configs.__file__).resolve().parent
    assert server._model_yaml_dir() == config_dir / "models"
    names = {p.stem for p in (config_dir / "models").glob("*.yaml")}
    assert {"VCNNBpnet", "RNN", "BPNetReal"} <= names
    for name in ("config.example.yaml", "wandb_sweep_config.yaml", "trainers/default.yaml"):
        assert (config_dir / name).is_file(), name
    assert not (REPO / "scripts" / "cli" / "yamls").exists()


def test_repository_scripts_point_at_the_packaged_templates():
    cli_dir = REPO / "scripts" / "cli"
    for path in list(cli_dir.glob("*.sh")) + list((cli_dir / "sweeps").rglob("*.yaml")):
        assert "yamls/" not in path.read_text(encoding="utf-8"), path
    trainer = (cli_dir / ".." / ".." / "tfbs" / "configs" / "trainers" / "default.yaml")
    assert trainer.resolve().is_file()


def test_pyproject_declares_the_package_data_and_pysam():
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]["tfbs.configs"]
    assert "models/*.yaml" in package_data and "*.yaml" in package_data
    genomics = data["project"]["optional-dependencies"]["genomics"]
    pysam = [req for req in genomics if req.startswith("pysam")]
    assert pysam and "sys_platform != 'win32'" in pysam[0]
    assert {"pyfaidx>=0.7", "tangermeme>=0.4"} <= set(genomics)
    assert "tfbs-mcp[viz,genomics,tracking,mcp,dev,foldx,equinet]" in \
        data["project"]["optional-dependencies"]["all"]


def test_built_wheel_contains_the_pipelines_and_templates(tmp_path):
    pytest.importorskip("setuptools")
    src = tmp_path / "src"
    src.mkdir()
    shutil.copytree(REPO / "tfbs", src / "tfbs",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(REPO / name, src / name)
    dist = tmp_path / "dist"
    dist.mkdir()
    # Through the PEP 517 backend directly: a uv venv has no pip.
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, setuptools.build_meta as b; b.build_wheel(sys.argv[1])", str(dist)],
        cwd=src, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    wheel = next(dist.glob("tfbs_mcp-*.whl"))
    names = set(zipfile.ZipFile(wheel).namelist())
    for pipeline in PIPELINES:
        assert f"tfbs/pipelines/{pipeline}.py" in names
    assert "tfbs/pipelines/__init__.py" in names
    assert "tfbs/pipelines/checkpoints.py" in names
    assert "tfbs/configs/__init__.py" in names
    templates = {n for n in names if n.startswith("tfbs/configs/models/") and n.endswith(".yaml")}
    assert len(templates) == len(list((REPO / "tfbs" / "configs" / "models").glob("*.yaml")))
    for name in ("config.example.yaml", "wandb_sweep_config.yaml",
                 "trainers/default.yaml", "datas/core_0.1.yaml"):
        assert f"tfbs/configs/{name}" in names
    assert not any(n.startswith("scripts/") for n in names)


# ---------------------------------------------------------------------------
# How the server runs them
# ---------------------------------------------------------------------------

class Recorder:
    """Stands in for subprocess.run and shutil.which inside the server."""

    def __init__(self, monkeypatch, available=("sbatch",), returncode=0):
        self.calls = []
        self.returncode = returncode
        monkeypatch.setattr(
            server.shutil, "which",
            lambda name, *a, **k: f"/usr/bin/{name}" if name in available else None,
        )
        monkeypatch.setattr(server.subprocess, "run", self._run)

    def _run(self, cmd, *args, **kwargs):
        self.calls.append({"cmd": list(cmd), **kwargs})
        if cmd[0] == "sbatch":
            return subprocess.CompletedProcess(cmd, 0, "Submitted batch job 77\n", "")
        if cmd[0] == "wandb":
            return subprocess.CompletedProcess(cmd, 0, "", "wandb: Created sweep with ID: sw1\n")
        return subprocess.CompletedProcess(cmd, self.returncode, "", "boom")

    def pipeline_calls(self):
        return [c for c in self.calls if c["cmd"][0] == sys.executable]


@pytest.fixture
def refs(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_pipeline_requirements_error", lambda *a: None)
    monkeypatch.setitem(server._ENV, "default_device", "cpu")
    files = {}
    for name in ("genome.fa", "motif.meme", "peaks.bed", "clinvar.vcf.gz", "complex.pdb"):
        path = tmp_path / name
        path.write_text("placeholder\n")
        files[name] = str(path)
    (tmp_path / "intersect").mkdir()
    files["intersect"] = str(tmp_path / "intersect")
    files["models_dir"] = str(_fake_checkpoints(tmp_path / "models"))
    return files


def _fake_checkpoints(models, datasets=("all_mean", "core_mean", "flank_mean"),
                      scaling="standardize", model="VCNNBpnet"):
    for dataset in datasets:
        ckpt = models / dataset / scaling / f"best_{model}.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("x")
    return models


def _call(tool, files, tmp_path, **kwargs):
    out = str(tmp_path / f"out_{tool}")
    if tool != "disease_variant_validation" and "checkpoints" not in kwargs:
        kwargs.setdefault("models_dir", files["models_dir"])
    if tool == "chipseq_benchmark":
        return run(server.tfbs_chipseq_benchmark(
            intersect_dir=files["intersect"], output_dir=out,
            genome_fasta=files["genome.fa"], **kwargs))
    if tool == "chipseq_benchmark_with_foldx":
        return run(server.tfbs_chipseq_method_comparison(
            intersect_dir=files["intersect"], output_dir=out,
            genome_fasta=files["genome.fa"], motif_file=files["motif.meme"],
            foldx_pdb=files["complex.pdb"], **kwargs))
    if tool == "remap_validation":
        return run(server.tfbs_remap_validation(
            remap_bed=files["peaks.bed"], output_dir=out,
            genome_fasta=files["genome.fa"], motif_file=files["motif.meme"],
            skip_foldx=True, **kwargs))
    return run(server.tfbs_disease_variant_validation(
        peaks_bed=files["peaks.bed"], output_dir=out,
        clinvar_vcf=files["clinvar.vcf.gz"], genome_fasta=files["genome.fa"], **kwargs))


VALIDATION_PIPELINES = ("chipseq_benchmark", "chipseq_benchmark_with_foldx",
                        "remap_validation", "disease_variant_validation")


@pytest.mark.parametrize("name", VALIDATION_PIPELINES)
def test_direct_run_invokes_the_packaged_module_from_the_project_root(
        name, refs, tmp_path, monkeypatch):
    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    rec = Recorder(monkeypatch)
    result = _call(name, refs, tmp_path, slurm=False)
    assert result["mode"] == "direct", result
    call = rec.pipeline_calls()[-1]
    assert call["cmd"][:3] == [sys.executable, "-m", f"tfbs.pipelines.{name}"]
    assert call["cwd"] == str(tmp_path.resolve())
    if server._pipeline_pythonpath():
        assert call["env"]["PYTHONPATH"].split(os.pathsep)[0] == server._pipeline_pythonpath()
    assert not any(part.endswith(f"{name}.py") for part in call["cmd"])


@pytest.mark.parametrize("name", VALIDATION_PIPELINES)
def test_slurm_job_invokes_the_packaged_module(name, refs, tmp_path, monkeypatch):
    rec = Recorder(monkeypatch)
    result = _call(name, refs, tmp_path, slurm=True)
    assert result["job_id"] == "77", result
    assert result["command"].startswith(
        " ".join([server.shlex.quote(sys.executable), "-m", f"tfbs.pipelines.{name}"]))
    script = Path(rec.calls[-1]["cmd"][-1]).read_text()
    assert f"-m tfbs.pipelines.{name}" in script
    assert f"scripts/{name}.py" not in script
    assert rec.calls[-1]["cwd"] == str(server._project_root())


@pytest.mark.parametrize("name", ("chipseq_benchmark", "chipseq_benchmark_with_foldx"))
def test_benchmarks_pass_a_device(name, refs, tmp_path, monkeypatch):
    rec = Recorder(monkeypatch)
    _call(name, refs, tmp_path, slurm=False)
    cmd = rec.pipeline_calls()[-1]["cmd"]
    assert cmd[cmd.index("--device") + 1] == "cpu"
    result = _call(name, refs, tmp_path, slurm=True)
    assert "--device cuda" in result["command"]
    result = _call(name, refs, tmp_path, slurm=True, device="cpu")
    assert "--device cpu" in result["command"]


def test_pipeline_failure_and_timeout_are_json_errors(refs, tmp_path, monkeypatch):
    Recorder(monkeypatch, returncode=1)
    result = _call("remap_validation", refs, tmp_path, slurm=False)
    assert "exit 1" in result["error"] and result["fix"] and result["stderr"] == "boom"

    def too_slow(cmd, *a, **k):
        raise subprocess.TimeoutExpired(cmd, k.get("timeout"))

    monkeypatch.setattr(server.subprocess, "run", too_slow)
    result = _call("chipseq_benchmark", refs, tmp_path, slurm=False)
    assert "timed out" in result["error"] and "slurm=true" in result["fix"]


def test_missing_project_root_is_a_json_error(refs, tmp_path, monkeypatch):
    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path / "nowhere"))
    rec = Recorder(monkeypatch)
    result = _call("remap_validation", refs, tmp_path, slurm=False)
    assert "not a directory" in result["error"] and "TFBS_PROJECT_ROOT" in result["fix"]
    assert rec.pipeline_calls() == []


def test_missing_pysam_is_reported_before_anything_runs(tmp_path, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "pysam" else real(name, *a))
    rec = Recorder(monkeypatch)
    files = {}
    for name in ("genome.fa", "peaks.bed", "clinvar.vcf.gz"):
        (tmp_path / name).write_text("x\n")
        files[name] = str(tmp_path / name)
    result = _call("disease_variant_validation", files, tmp_path, slurm=False)
    assert "pysam" in result["error"]
    assert "genomics" in result["fix"] or "Windows" in result["fix"]
    assert rec.calls == []


def test_validate_job_runs_the_packaged_module(tmp_path, monkeypatch):
    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    rec = Recorder(monkeypatch)
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("x")
    data = tmp_path / "dataset"
    data.mkdir()
    result = run(server.tfbs_validate(checkpoint_path=str(ckpt), data_path=str(data)))
    assert result["job_id"] == "77", result
    csv_path = tmp_path.resolve() / "results" / "validation_results.csv"
    assert csv_path.parent.is_dir()
    assert "-m tfbs.pipelines.validate" in result["validate_command"]
    assert str(csv_path) in result["validate_command"]
    script = result["sbatch_script"]
    assert f"#SBATCH --output={csv_path.parent / 'validate'}.o%j" in script
    assert "scripts/validate.py" not in script
    assert rec.calls[-1]["cwd"] == str(tmp_path.resolve())
    assert "trainer.test()" in server.tfbs_validate.__doc__


def test_sweep_rejects_a_model_the_agent_cannot_train(tmp_path, monkeypatch):
    cfg = tmp_path / "sweep.yaml"
    cfg.write_text("method: bayes\n")
    rec = Recorder(monkeypatch, available=("sbatch", "wandb"))
    result = run(server.tfbs_sweep(sweep_config=str(cfg), project_name="p",
                                   model="CNN", data_path=str(tmp_path), data_name="d"))
    assert "CNN" in result["error"] and "No sweep was created" in result["fix"]
    assert rec.calls == []


def test_sweep_config_error_names_the_bundled_config(tmp_path, monkeypatch):
    Recorder(monkeypatch, available=("sbatch", "wandb"))
    result = run(server.tfbs_sweep(sweep_config=str(tmp_path / "missing.yaml"),
                                   project_name="p", model="RNN",
                                   data_path=str(tmp_path), data_name="d"))
    bundled = tfbs.configs.config_dir() / "wandb_sweep_config.yaml"
    assert str(bundled) in result["fix"] and bundled.is_file()


# ---------------------------------------------------------------------------
# Project root and PYTHONPATH outside a checkout
# ---------------------------------------------------------------------------

def test_project_root_outside_a_checkout_is_the_working_directory(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    (site / "tfbs" / "mcp").mkdir(parents=True)
    monkeypatch.setattr(server, "__file__", str(site / "tfbs" / "mcp" / "server.py"))
    monkeypatch.delenv("TFBS_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert server._project_root() == tmp_path.resolve()
    assert server._pipeline_pythonpath() is None
    assert server._sbatch_pythonpath_line() == ""
    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path / "data"))
    assert server._project_root() == (tmp_path / "data").resolve()


def test_source_checkout_is_put_on_the_pipeline_pythonpath(monkeypatch):
    if Path(server.__file__).resolve().parents[2] != REPO:
        pytest.skip("the server under test is an installed copy, not this checkout")
    monkeypatch.setenv("PYTHONPATH", "elsewhere")
    assert server._pipeline_pythonpath() == str(REPO)
    assert server._pipeline_env()["PYTHONPATH"] == os.pathsep.join([str(REPO), "elsewhere"])
    assert server._sbatch_pythonpath_line().startswith("export PYTHONPATH=")


def test_server_info_reports_the_project_root():
    info = run(server.tfbs_server_info())
    assert info["project_root"] == str(server._project_root())


# ---------------------------------------------------------------------------
# tfbs-mcp --check
# ---------------------------------------------------------------------------

def test_check_reports_the_packaged_pipelines(capsys, monkeypatch):
    monkeypatch.setattr(cli, "find_executable", lambda: "/opt/env/bin/tfbs-mcp")
    monkeypatch.setattr(cli, "_module_found",
                        lambda name: not name.startswith(("pysam", "wandb")))
    assert cli.check() == 0
    out = capsys.readouterr().out
    assert "6 training and validation pipelines and 13 model templates installed" in out
    assert "pysam not installed: tfbs_disease_variant_validation" in out
    assert 'wandb not installed: tfbs_sweep needs it (pip install "tfbs-mcp[tracking]")' in out
    assert "clone" not in out


def test_check_fails_when_a_pipeline_is_missing(capsys, monkeypatch):
    monkeypatch.setattr(cli, "find_executable", lambda: "/opt/env/bin/tfbs-mcp")
    monkeypatch.setattr(cli, "packaged_pipelines", lambda: (["validate"], 13))
    assert cli.check() == 1
    assert "[FAIL] training and validation pipelines incomplete (missing: validate)" \
        in capsys.readouterr().out


def test_packaged_pipelines_finds_everything_here():
    assert cli.packaged_pipelines() == ([], 13)


# ---------------------------------------------------------------------------
# The sweep agent (with a stand-in wandb module)
# ---------------------------------------------------------------------------

class _Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def update(self, values, allow_val_change=False):
        super().update(values)


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


def test_sweep_agent_starts_a_run_and_uses_the_task_values(tmp_path, monkeypatch):
    from tfbs.pipelines import run_sweep

    config = _Config(
        model="RNN", data_path="/path/", data_name="data", scaling_method="none",
        batch_size=8, checkpoint_dir="x", dropout=0.1, lr=1e-3, weight_decay=0.0,
        conv_out_channels=8, RNN_kernel_size=3, pool_size=2, dropout_conv=0.1,
        gru_hidden_size=4, gru_num_layers=1, bidirectional=False, RNN_dense_size=8,
        dropout_fc=0.1,
    )
    fake = _fake_wandb(config)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    seen = {}

    class DataModule:
        def __init__(self, **kwargs):
            seen["data"] = kwargs

    class Trainer:
        def __init__(self, **kwargs):
            seen["trainer"] = kwargs

        def fit(self, model, data):
            seen["fit"] = (model, data)

    monkeypatch.setattr(run_sweep, "TFBSDataModule", DataModule)
    monkeypatch.setattr(run_sweep.pl, "Trainer", Trainer)
    monkeypatch.setattr(run_sweep, "WandbLogger", lambda experiment: ("logger", experiment))

    fixed = {"data_path": str(tmp_path / "tensors"), "data_name": "nkx",
             "scaling_method": "standardize", "checkpoint_dir": str(tmp_path / "ckpt"),
             "model": "RNN", "batch_size": 16}
    run_sweep.train_sweep_agent(fixed)

    assert seen["data"]["data_path"] == str(tmp_path / "tensors")
    assert seen["data"]["batch_size"] == 16
    assert seen["trainer"]["logger"] == ("logger", fake.run)
    assert isinstance(seen["fit"][0], run_sweep.models.RNN)
    assert (tmp_path / "ckpt" / "nkx" / "standardize" / "RNN").is_dir()


@pytest.mark.parametrize("entity", [None, "my-lab"])
def test_sweep_agent_uses_the_wandb_entity_from_the_environment(entity, monkeypatch):
    from tfbs.pipelines import run_sweep

    fake = _fake_wandb(_Config())
    monkeypatch.setitem(sys.modules, "wandb", fake)
    if entity:
        monkeypatch.setenv("WANDB_ENTITY", entity)
    else:
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
    run_sweep.main(["--model", "VCNNBpnet", "--data-path", "/d", "--data-name", "n",
                    "--scaling-method", "standardize", "--project-name", "p",
                    "--sweep-id", "sw1", "--n-trials", "3"])
    sweep_id, kwargs = fake.agent_calls[-1]
    assert sweep_id == "sw1"
    assert kwargs["entity"] == entity
    assert kwargs["count"] == 3
    assert kwargs["function"].args[0]["data_path"] == "/d"


# ---------------------------------------------------------------------------
# Old call signatures and the validation CSV
# ---------------------------------------------------------------------------

def _load_shim(name):
    import importlib.util as iu

    spec = iu.spec_from_file_location(f"shim_{name}", REPO / SHIMS[name])
    module = iu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name, runner", [("validate", "run"), ("run_sweep", "run_agent")])
def test_shim_main_still_accepts_a_namespace(name, runner, monkeypatch):
    import argparse

    pipeline = importlib.import_module(f"tfbs.pipelines.{name}")
    seen = []
    monkeypatch.setattr(pipeline, runner, seen.append)
    args = argparse.Namespace(anything=1)
    _load_shim(name).main(args)
    assert seen == [args]


def test_validate_main_still_parses_a_list(monkeypatch, tmp_path):
    from tfbs.pipelines import validate

    seen = []
    monkeypatch.setattr(validate, "run", seen.append)
    validate.main(["--checkpoint_path", "c.ckpt", "--output_file", "o.csv",
                   "--data_path", str(tmp_path)])
    assert seen[0].checkpoint_path == "c.ckpt" and seen[0].scaling_method == "standardize"


def _args(tmp_path, **kw):
    import argparse

    base = dict(model_name="VCNNBpnet", scaling_method="none",
                data_path=str(tmp_path / "peaks_dataset") + os.sep,
                output_file=str(tmp_path / "results.csv"))
    base.update(kw)
    return argparse.Namespace(**base)


def _csv_rows(path):
    import csv

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_validate_appends_every_test_metric_of_a_classifier(tmp_path):
    from tfbs.pipelines import validate

    validate.save_results(_args(tmp_path), {"test_loss": 0.4, "test_acc": 0.81, "test_auc": 0.9})
    rows = _csv_rows(tmp_path / "results.csv")
    assert rows == [{"modeltype": "VCNNBpnet", "scaling_method": "none",
                     "dataset_path": "peaks_dataset", "test_loss": "0.4",
                     "test_acc": "0.81", "test_auc": "0.9"}]


def test_validate_widens_an_older_csv_and_keeps_its_rows(tmp_path):
    from tfbs.pipelines import validate

    validate.save_results(_args(tmp_path, scaling_method="standardize"), 1.5)
    validate.save_results(_args(tmp_path), {"test_loss": 0.4, "test_acc": 0.8, "test_auc": 0.9})
    rows = _csv_rows(tmp_path / "results.csv")
    assert [r["test_loss"] for r in rows] == ["1.5", "0.4"]
    assert rows[0]["test_auc"] == "" and rows[1]["test_auc"] == "0.9"
    validate.save_results(_args(tmp_path), {"test_loss": 0.3})
    assert len(_csv_rows(tmp_path / "results.csv")) == 3


def test_validate_widens_the_csv_through_an_atomic_replace(tmp_path, monkeypatch):
    from tfbs.pipelines import validate

    replaced = []
    real_replace = os.replace

    def recording_replace(src, dst):
        replaced.append((os.path.dirname(os.path.abspath(src)), dst))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", recording_replace)
    validate.save_results(_args(tmp_path), 1.5)
    assert replaced == []
    validate.save_results(_args(tmp_path), {"test_loss": 0.4, "test_auc": 0.9})
    assert replaced == [(str(tmp_path), str(tmp_path / "results.csv"))]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["results.csv"]
    assert [r["test_auc"] for r in _csv_rows(tmp_path / "results.csv")] == ["", "0.9"]


def test_a_failed_widening_leaves_the_csv_and_no_temporary_file(tmp_path, monkeypatch):
    from tfbs.pipelines import validate

    validate.save_results(_args(tmp_path), 1.5)
    before = (tmp_path / "results.csv").read_bytes()

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)
    validate.save_results(_args(tmp_path), {"test_loss": 0.4, "test_auc": 0.9})
    assert (tmp_path / "results.csv").read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["results.csv"]


def test_validate_appends_under_the_existing_header(tmp_path):
    from tfbs.pipelines import validate

    validate.save_results(_args(tmp_path), {"test_loss": 0.4, "test_acc": 0.8, "test_auc": 0.9})
    before = (tmp_path / "results.csv").read_bytes()
    validate.save_results(_args(tmp_path), 0.3)
    after = (tmp_path / "results.csv").read_bytes()
    assert after.startswith(before)
    assert _csv_rows(tmp_path / "results.csv")[1] == {
        "modeltype": "VCNNBpnet", "scaling_method": "none", "dataset_path": "peaks_dataset",
        "test_loss": "0.3", "test_acc": "", "test_auc": ""}


def test_validate_run_saves_the_metrics_trainer_test_reports(tmp_path, monkeypatch):
    from tfbs.pipelines import validate

    class DM:
        def __init__(self, **kw):
            self.train_dataset = [(torch.zeros(4, 24), torch.zeros(1))]

        def setup(self, stage=None):
            pass

    class Model:
        @classmethod
        def load_from_checkpoint(cls, **kw):
            return cls()

        def eval(self):
            return self

    class Trainer:
        def __init__(self, **kw):
            pass

        def test(self, model, datamodule=None, verbose=True):
            return [{"test_loss": 0.2, "test_acc": 0.75, "test_auc": 0.85}]

    monkeypatch.setattr(validate, "TFBSDataModule", DM)
    monkeypatch.setattr(validate, "get_model_class", lambda name: Model)
    monkeypatch.setattr(validate.pl, "Trainer", Trainer)
    args = _args(tmp_path, checkpoint_path="x.ckpt", batch_size=8, num_workers=0,
                 preprocess=False)
    validate.run(args)
    row = _csv_rows(tmp_path / "results.csv")[0]
    assert (row["test_loss"], row["test_acc"], row["test_auc"]) == ("0.2", "0.75", "0.85")


def test_validate_writes_lf_line_endings_on_append_and_rewrite(tmp_path):
    from tfbs.pipelines import validate

    validate.save_results(_args(tmp_path), 0.5)
    validate.save_results(_args(tmp_path), 0.6)
    validate.save_results(_args(tmp_path), {"test_loss": 0.4, "test_auc": 0.9})
    data = (tmp_path / "results.csv").read_bytes()
    assert b"\r" not in data and data.count(b"\n") == 4


@pytest.mark.parametrize("module", ["chipseq_benchmark", "chipseq_benchmark_with_foldx",
                                    "remap_validation"])
def test_scaling_method_help_says_the_fallback_needs_the_flag_omitted(module, capsys):
    pipeline = importlib.import_module(f"tfbs.pipelines.{module}")
    with pytest.raises(SystemExit):
        pipeline.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "Only when this flag is omitted and that file is missing" in text
    assert "an explicit value never falls back to another folder" in text
    assert "(default: the standardize folder)" in text
