"""One checkpoint convention for tfbs_config, the benchmark tools and the pipelines."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tfbs.mcp import server
from tfbs.pipelines import checkpoints as ck

REPO = Path(__file__).resolve().parent.parent


def run(coro):
    return json.loads(asyncio.run(coro))


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


def _dataset(root: Path, length: int = 24, n: int = 6) -> Path:
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True)
        torch.save(torch.zeros(n, 4, length), root / split / "seqs.pt")
        torch.save(torch.zeros(n, 1), root / split / "labels.pt")
    return root


# ---------------------------------------------------------------------------
# The shared resolver
# ---------------------------------------------------------------------------

def test_expected_checkpoint_wins(tmp_path):
    expected = _touch(tmp_path / "ds" / "standardize" / "best_RNN.ckpt")
    _touch(tmp_path / "ds" / "none" / "best_RNN.ckpt")
    path, candidates = ck.resolve_checkpoint(tmp_path, "ds", "RNN")
    assert path == expected and candidates == [expected]


def test_a_single_other_scaling_folder_is_used_only_without_scaling_method(tmp_path):
    only = _touch(tmp_path / "ds" / "none" / "best_RNN.ckpt")
    path, _ = ck.resolve_checkpoint(tmp_path, "ds", "RNN")
    assert path == only
    path, candidates = ck.resolve_checkpoint(tmp_path, "ds", "RNN", "standardize")
    assert path is None and candidates == [only]
    found, problems = ck.resolve_checkpoints(tmp_path, ["ds"], "RNN", "standardize")
    assert found == {} and "another scaling folder" in ck.describe_problems(problems)
    assert ck.DEFAULT_SCALING is None
    assert ck.expected_checkpoint(tmp_path, "ds", "RNN") == tmp_path / "ds" / "standardize" / "best_RNN.ckpt"


def test_several_candidates_resolve_to_none(tmp_path):
    a = _touch(tmp_path / "ds" / "none" / "best_RNN.ckpt")
    b = _touch(tmp_path / "ds" / "normalize" / "best_RNN.ckpt")
    path, candidates = ck.resolve_checkpoint(tmp_path, "ds", "RNN")
    assert path is None and sorted(candidates) == sorted([a, b])
    found, problems = ck.resolve_checkpoints(tmp_path, ["ds"], "RNN")
    assert found == {} and len(problems[0]["candidates"]) == 2
    assert "several candidates" in ck.describe_problems(problems)


def test_explicit_checkpoints_replace_the_lookup(tmp_path):
    ckpt = _touch(tmp_path / "anywhere" / "model.ckpt")
    found, problems = ck.resolve_checkpoints(
        tmp_path / "nothing", ["ignored"], "RNN", explicit={"mine": str(ckpt), "gone": "x.ckpt"})
    assert found == {"mine": ckpt}
    assert [p["dataset"] for p in problems] == ["gone"]


def test_named_paths_parse_and_reject_garbage():
    assert ck.parse_named_paths(["a=/x/y.ckpt", "b=c=d"]) == {"a": "/x/y.ckpt", "b": "c=d"}
    assert ck.parse_named_paths(None) == {}
    with pytest.raises(ValueError):
        ck.parse_named_paths(["no-equals"])


@pytest.mark.parametrize("name", ["chipseq_benchmark", "chipseq_benchmark_with_foldx",
                                  "remap_validation"])
def test_pipelines_take_scaling_method_and_explicit_checkpoints(name):
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    result = subprocess.run([sys.executable, "-m", f"tfbs.pipelines.{name}", "--help"],
                            capture_output=True, text=True, timeout=600, env=env)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "--scaling-method" in result.stdout and "--checkpoints" in result.stdout


@pytest.mark.parametrize("name", ["chipseq_benchmark", "chipseq_benchmark_with_foldx",
                                  "remap_validation"])
def test_pipelines_no_longer_hard_code_standardize(name):
    source = (REPO / "tfbs" / "pipelines" / f"{name}.py").read_text(encoding="utf-8")
    assert '/ "standardize" /' not in source
    assert "resolve_checkpoints(" in source


# ---------------------------------------------------------------------------
# tfbs_config: the checkpoint it will produce, and where the config goes
# ---------------------------------------------------------------------------

def test_config_reports_the_checkpoint_and_drops_the_dataset_suffix(tmp_path, monkeypatch):
    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    data = _dataset(tmp_path / "data" / "peaks_dataset")
    result = run(server.tfbs_config(model_type="VCNNBpnet", data_path=str(data), classify=True))
    assert "error" not in result, result
    expected = (tmp_path / "saved_models" / "peaks" / "none" / "best_VCNNBpnet.ckpt").resolve()
    assert result["checkpoint_path"] == str(expected)
    assert result["files"] == {"config": result["output_path"], "checkpoint": str(expected)}


def test_config_output_path_defaults_next_to_the_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    data = _dataset(tmp_path / "data" / "peaks_dataset")
    result = run(server.tfbs_config(model_type="RNN", data_path=str(data)))
    out = Path(result["output_path"])
    assert out == (tmp_path / "data" / "peaks_config" / "RNN.yaml").resolve()
    assert out.is_file() and b"\r\n" not in out.read_bytes()
    assert not any(p.suffix == ".yaml" for p in data.rglob("*"))


def test_config_description_names_the_convention():
    doc = " ".join(server.tfbs_config.__doc__.split())
    assert "<models_dir>/<dataset>/<scaling_method>/best_<model_type>.ckpt" in doc
    assert "checkpoint_path" in doc and "without a trailing _dataset" in doc
    prep = " ".join(server.tfbs_prepare_dataset.__doc__.split())
    assert "tfbs_config drops the _dataset suffix" in prep


def test_a_classifier_from_tfbs_config_is_found_by_the_benchmark(tmp_path, monkeypatch):
    """tfbs_config -> training -> tfbs_chipseq_benchmark without naming the scaling."""
    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(server, "_pipeline_requirements_error", lambda *a: None)
    monkeypatch.setitem(server._ENV, "default_device", "cpu")
    data = _dataset(tmp_path / "data" / "peaks_dataset")
    config = run(server.tfbs_config(model_type="RNN", data_path=str(data), classify=True))
    _touch(Path(config["checkpoint_path"]))

    genome = _touch(tmp_path / "genome.fa")
    (tmp_path / "intersect").mkdir()
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    result = run(server.tfbs_chipseq_benchmark(
        intersect_dir=str(tmp_path / "intersect"), output_dir=str(tmp_path / "bench"),
        datasets=["peaks"], model_type="RNN", models_dir="saved_models",
        genome_fasta=str(genome)))
    assert "error" not in result, result
    assert result["checkpoints"] == {"peaks": config["checkpoint_path"]}
    cmd = calls[-1]
    assert cmd[cmd.index("--checkpoints") + 1] == f"peaks={config['checkpoint_path']}"
    assert "--models-dir" not in cmd


# ---------------------------------------------------------------------------
# The benchmark and validation tools
# ---------------------------------------------------------------------------

@pytest.fixture
def bench(tmp_path, monkeypatch):
    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(server, "_pipeline_requirements_error", lambda *a: None)
    monkeypatch.setitem(server._ENV, "default_device", "cpu")
    files = {name: str(_touch(tmp_path / name)) for name in
             ("genome.fa", "motif.meme", "peaks.bed", "complex.pdb")}
    (tmp_path / "intersect").mkdir()
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))
        if cmd[0] == "sbatch":
            return subprocess.CompletedProcess(cmd, 0, "Submitted batch job 5\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    monkeypatch.setattr(server.shutil, "which",
                        lambda name, *a, **k: f"/usr/bin/{name}" if name == "sbatch" else None)
    return tmp_path, files, calls


def _tool(name, tmp_path, files, **kwargs):
    out = str(tmp_path / f"out_{name}")
    if name == "tfbs_chipseq_benchmark":
        return run(server.tfbs_chipseq_benchmark(
            intersect_dir=str(tmp_path / "intersect"), output_dir=out,
            genome_fasta=files["genome.fa"], **kwargs))
    if name == "tfbs_chipseq_method_comparison":
        return run(server.tfbs_chipseq_method_comparison(
            intersect_dir=str(tmp_path / "intersect"), output_dir=out,
            genome_fasta=files["genome.fa"], motif_file=files["motif.meme"],
            foldx_pdb=files["complex.pdb"], slurm=False, **kwargs))
    return run(server.tfbs_remap_validation(
        remap_bed=files["peaks.bed"], output_dir=out, genome_fasta=files["genome.fa"],
        motif_file=files["motif.meme"], skip_foldx=True, slurm=False, **kwargs))


TOOLS = ("tfbs_chipseq_benchmark", "tfbs_chipseq_method_comparison", "tfbs_remap_validation")


@pytest.mark.parametrize("name", TOOLS)
def test_tools_resolve_relative_models_dir_against_the_project_root(name, bench):
    tmp_path, files, calls = bench
    ckpt = _touch(tmp_path / "saved_models_final" / "all_mean" / "standardize"
                  / "best_VCNNBpnet.ckpt")
    result = _tool(name, tmp_path, files, datasets=["all_mean"])
    assert result["checkpoints"] == {"all_mean": str(ckpt.resolve())}, result
    cmd = calls[-1]
    assert f"all_mean={ckpt.resolve()}" in cmd


@pytest.mark.parametrize("name", TOOLS)
def test_tools_take_scaling_method(name, bench):
    tmp_path, files, calls = bench
    models = tmp_path / "m"
    _touch(models / "ds" / "standardize" / "best_VCNNBpnet.ckpt")
    norm = _touch(models / "ds" / "normalize" / "best_VCNNBpnet.ckpt")
    result = _tool(name, tmp_path, files, datasets=["ds"], models_dir=str(models),
                   scaling_method="normalize")
    assert result["checkpoints"] == {"ds": str(norm.resolve())}


@pytest.mark.parametrize("name", TOOLS)
def test_tools_list_the_candidates_when_none_is_unique(name, bench):
    tmp_path, files, calls = bench
    models = tmp_path / "m"
    a = _touch(models / "ds" / "none" / "best_VCNNBpnet.ckpt")
    b = _touch(models / "ds" / "normalize" / "best_VCNNBpnet.ckpt")
    result = _tool(name, tmp_path, files, datasets=["ds"], models_dir=str(models))
    assert "ds" in result["error"] and "scaling_method" in result["fix"]
    problem = result["checkpoint_problems"][0]
    assert sorted(problem["candidates"]) == sorted([str(a), str(b)])
    assert problem["expected"].replace("\\", "/").endswith("ds/standardize/best_VCNNBpnet.ckpt")
    assert calls == []


@pytest.mark.parametrize("name", TOOLS)
def test_tools_report_a_missing_checkpoint(name, bench):
    tmp_path, files, calls = bench
    result = _tool(name, tmp_path, files, datasets=["nothing"], models_dir=str(tmp_path / "m"))
    assert "nothing" in result["error"] and "checkpoints=" in result["fix"]
    assert result["checkpoint_problems"][0]["candidates"] == []
    assert calls == []


@pytest.mark.parametrize("name", TOOLS)
def test_tools_accept_explicit_checkpoints(name, bench):
    tmp_path, files, calls = bench
    ckpt = _touch(tmp_path / "elsewhere" / "my.ckpt")
    result = _tool(name, tmp_path, files, checkpoints={"mine": "elsewhere/my.ckpt"})
    assert result["checkpoints"] == {"mine": str(ckpt.resolve())}
    cmd = calls[-1]
    assert cmd[cmd.index("--checkpoints") + 1] == f"mine={ckpt.resolve()}"
    assert "--datasets" not in cmd


def test_unknown_scaling_method_is_a_json_error(bench):
    tmp_path, files, calls = bench
    result = _tool("tfbs_chipseq_benchmark", tmp_path, files, scaling_method="zscore")
    assert "zscore" in result["error"] and result["fix"]


@pytest.mark.parametrize("name", TOOLS)
def test_tool_descriptions_state_the_convention(name):
    doc = " ".join(getattr(server, name).__doc__.split())
    assert "<models_dir>/<dataset>/<scaling_method>/ best_<model_type>.ckpt" in doc
    assert "scaling_method:" in doc and "checkpoints:" in doc


@pytest.mark.parametrize("name", TOOLS)
def test_tools_fall_back_with_a_warning_only_when_scaling_method_is_omitted(name, bench):
    tmp_path, files, calls = bench
    models = tmp_path / "m"
    only = _touch(models / "ds" / "none" / "best_VCNNBpnet.ckpt")
    result = _tool(name, tmp_path, files, datasets=["ds"], models_dir=str(models))
    assert result["checkpoints"] == {"ds": str(only.resolve())}
    assert "none/" in result["warning"] and "'standardize'" in result["warning"]
    calls.clear()
    strict = _tool(name, tmp_path, files, datasets=["ds"], models_dir=str(models),
                   scaling_method="standardize")
    assert "ds" in strict["error"] and "omit scaling_method" in strict["fix"]
    assert strict["checkpoint_problems"][0]["candidates"] == [str(only)]
    assert calls == []


@pytest.mark.parametrize("name", TOOLS)
def test_tools_have_no_warning_for_the_requested_folder(name, bench):
    tmp_path, files, calls = bench
    models = tmp_path / "m"
    _touch(models / "ds" / "standardize" / "best_VCNNBpnet.ckpt")
    assert "warning" not in _tool(name, tmp_path, files, datasets=["ds"], models_dir=str(models))


def _class_path_ckpt(path: Path, class_path: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {}, "hyper_parameters": {"_class_path": class_path}}, path)
    return path


@pytest.mark.parametrize("name", TOOLS)
def test_tools_stop_when_the_checkpoint_class_cannot_be_imported(name, bench):
    tmp_path, files, calls = bench
    models = tmp_path / "m"
    _class_path_ckpt(models / "ds" / "standardize" / "best_VCNNBpnet.ckpt",
                     "noshim_xyz.nn.models.VCNNBpnet")
    result = _tool(name, tmp_path, files, datasets=["ds"], models_dir=str(models))
    assert "noshim_xyz.nn.models" in result["error"] and "PYTHONPATH" in result["fix"]
    assert result["class_path"] == "noshim_xyz.nn.models.VCNNBpnet"
    assert calls == []


def test_a_class_module_under_the_project_root_is_importable(bench):
    tmp_path, files, calls = bench
    pkg = tmp_path / "rootshim_q" / "nn"
    pkg.mkdir(parents=True)
    (tmp_path / "rootshim_q" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "models.py").write_text("from tfbs.nn.models import *\n")
    ckpt = _class_path_ckpt(tmp_path / "m" / "ds" / "standardize" / "best_VCNNBpnet.ckpt",
                            "rootshim_q.nn.models.VCNNBpnet")
    try:
        result = _tool("tfbs_chipseq_benchmark", tmp_path, files, datasets=["ds"],
                       models_dir=str(tmp_path / "m"))
    finally:
        for key in [k for k in sys.modules if k.startswith("rootshim_q")]:
            sys.modules.pop(key)
    assert result["checkpoints"] == {"ds": str(ckpt.resolve())}, result
    assert str(tmp_path) not in sys.path


def test_tool_descriptions_say_the_lookup_is_all_or_nothing():
    for name in TOOLS:
        doc = " ".join(getattr(server, name).__doc__.split())
        assert "all-or-nothing" in doc and "never falls back" in doc, name
