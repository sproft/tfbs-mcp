"""Server behaviour that the application note and its supplement describe.

Each test pins one statement of the paper audit (A1 to A8) to the code:
saturation mutagenesis returns the mean-centred change of all four bases;
DeepLiftShap explains each sequence against seeded dinucleotide shuffles
of itself; tfbs_epistasis takes batch_size; slurm_status tells NOT_FOUND
from a failed lookup; tfbs_conv_filters keeps the strongest filters; the
FoldX legs need FoldX unless skip_foldx; and every tool description
states each default its schema gives.
"""
import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tfbs.mcp import server
from tests.fake_pipeline import fake_pipelines

SEQS_20 = ["ACGTTGCAAGCTTAGCCGAT", "GGATCCATTGACGTCAGTAC", "TTAGGCATCGATCGGATACC",
           "CAGTGGTACCATGATCGCAA", "ATCGGCTAGCTTACGGATCA", "GCATGCCTAGGATTCAAGTC"]


def run(coro):
    return json.loads(asyncio.run(coro))


class Linear(torch.nn.Module):
    """f(x) = sum over positions of W[base, position]: every effect is additive."""

    def __init__(self, length=20, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = torch.nn.Parameter(torch.randn(4, length, generator=g))

    def forward(self, x):
        return (x * self.W).sum(dim=(1, 2)).unsqueeze(1)


class ReluConv(torch.nn.Module):
    """A small non-linear model, so attributions depend on the references."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.conv = torch.nn.Conv1d(4, 3, kernel_size=5, padding=2)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(3, 1)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g))
            self.conv.bias.copy_(torch.randn(self.conv.bias.shape, generator=g))
            self.fc.weight.copy_(torch.randn(self.fc.weight.shape, generator=g))

    def forward(self, x):
        return self.fc(self.relu(self.conv(x)).mean(dim=2))


def _register(model_id, model, length=20):
    server._model_registry[model_id] = {
        "model": model.eval(), "model_type": type(model).__name__, "device": "cpu",
        "input_length": length, "classify": False, "scaling_params": None,
    }


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_results_root", lambda: tmp_path / "results")
    server._ensure_imports()
    _register("lin", Linear())
    _register("relu", ReluConv())
    yield
    for name in ("lin", "relu", "spy", "ranked"):
        server._model_registry.pop(name, None)


# ---------------------------------------------------------------------------
# A1 tfbs_mutagenesis: mean-centred changes of all four bases
# ---------------------------------------------------------------------------

def test_mutagenesis_scores_are_mean_centred_changes_of_all_four_bases():
    W = server._model_registry["lin"]["model"].W.detach()
    expected = W - W.mean(dim=0, keepdim=True)
    result = run(server.tfbs_mutagenesis(model_id="lin", sequences=SEQS_20[:2], batch_size=7))
    scores = torch.tensor(result["scores"])
    assert result["shape"] == [2, 4, 20]
    for i in range(2):
        # Not projected: the three unobserved bases carry values too.
        assert torch.allclose(scores[i], expected, atol=1e-5)
        assert torch.allclose(scores[i].sum(dim=0), torch.zeros(20), atol=1e-5)


def test_mutagenesis_window_is_end_minus_start_wide():
    W = server._model_registry["lin"]["model"].W.detach()
    expected = (W - W.mean(dim=0, keepdim=True))[:, 3:11]
    result = run(server.tfbs_mutagenesis(model_id="lin", sequences=SEQS_20[:1], start=3, end=11))
    assert result["shape"] == [1, 4, 8]
    assert torch.allclose(torch.tensor(result["scores"][0]), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# A2 tfbs_analyze: DeepLiftShap against seeded dinucleotide shuffles
# ---------------------------------------------------------------------------

def _dinucleotides(seq: str) -> list[str]:
    return sorted(seq[k:k + 2] for k in range(len(seq) - 1))


def test_deepliftshap_references_are_dinucleotide_shuffles_of_each_input(monkeypatch, tmp_path):
    import tangermeme.ersatz as ersatz

    real = ersatz.dinucleotide_shuffle
    calls = []

    def spy(X, *args, **kwargs):
        out = real(X, *args, **kwargs)
        calls.append((X.clone(), kwargs, out))
        return out

    monkeypatch.setattr(ersatz, "dinucleotide_shuffle", spy)
    result = run(server.tfbs_analyze(model_id="relu", sequences=SEQS_20[:3], n_shuffles=6,
                                     seed=7, output_dir=str(tmp_path / "a")))
    assert result["seed"] == 7 and result["n_shuffles"] == 6
    (X, kwargs, refs), = calls
    assert kwargs["random_state"] == 7 and kwargs["n"] == 6
    assert refs.shape == (3, 6, 4, 20)
    for i, seq in enumerate(server._tensor_to_sequences(X)):
        assert seq == SEQS_20[i]
        shuffles = server._tensor_to_sequences(refs[i])
        assert any(s != seq for s in shuffles)
        for s in shuffles:
            assert _dinucleotides(s) == _dinucleotides(seq)
            assert s[0] == seq[0] and s[-1] == seq[-1]


def test_projected_attributions_are_the_hypothetical_ones_times_the_input(tmp_path):
    kw = dict(model_id="relu", sequences=SEQS_20[:3], n_shuffles=8)
    projected = run(server.tfbs_analyze(**kw, output_dir=str(tmp_path / "p")))
    hypothetical = run(server.tfbs_analyze(**kw, hypothetical=True, output_dir=str(tmp_path / "h")))
    assert projected["hypothetical"] is False and hypothetical["hypothetical"] is True
    X = server._sequences_to_tensor(SEQS_20[:3])
    H = torch.tensor(hypothetical["attributions"])
    assert (H * (1 - X)).abs().sum() > 0  # unobserved bases are kept
    assert torch.allclose(H * X, torch.tensor(projected["attributions"]), atol=1e-6)


def test_the_seed_not_the_batching_decides_the_result(tmp_path):
    # 6 sequences x 50 shuffles: batch_size 1 scores 256 pairs per pass (5
    # sequences per shuffle call), batch_size 1024 all of them in one.
    kw = dict(model_id="relu", sequences=SEQS_20, n_shuffles=50)
    small = run(server.tfbs_analyze(**kw, batch_size=1, output_dir=str(tmp_path / "s")))
    large = run(server.tfbs_analyze(**kw, batch_size=1024, output_dir=str(tmp_path / "l")))
    assert torch.allclose(torch.tensor(small["attributions"]),
                          torch.tensor(large["attributions"]), atol=1e-6)
    other = run(server.tfbs_analyze(**kw, seed=3, output_dir=str(tmp_path / "o")))
    assert not torch.allclose(torch.tensor(other["attributions"]),
                              torch.tensor(large["attributions"]), atol=1e-6)


def test_linear_model_attributions_sum_to_the_difference_from_the_shuffles(tmp_path):
    from tangermeme.ersatz import dinucleotide_shuffle

    model = server._model_registry["lin"]["model"]
    result = run(server.tfbs_analyze(model_id="lin", sequences=SEQS_20[:3], n_shuffles=20,
                                     output_dir=str(tmp_path / "a")))
    X = server._sequences_to_tensor(SEQS_20[:3])
    with torch.no_grad():
        for i in range(3):
            refs = dinucleotide_shuffle(X[i:i + 1], start=0, end=20, n=20, random_state=42 + i)[0]
            expected = model(X[i:i + 1]).item() - model(refs).mean().item()
            total = torch.tensor(result["attributions"][i]).sum().item()
            assert total == pytest.approx(expected, abs=1e-4)


def test_seed_is_part_of_the_default_output_directory():
    a = run(server.tfbs_analyze(model_id="relu", sequences=SEQS_20[:1], n_shuffles=4))
    b = run(server.tfbs_analyze(model_id="relu", sequences=SEQS_20[:1], n_shuffles=4, seed=1))
    assert Path(a["files"]["attributions"]).parent != Path(b["files"]["attributions"]).parent


def test_a_sequence_without_distinct_shuffles_is_an_error_suggesting_input_x_gradient(tmp_path):
    result = run(server.tfbs_analyze(model_id="relu", sequences=[SEQS_20[0], "ACGT" * 5],
                                     output_dir=str(tmp_path / "a")))
    assert "Sequence 1 cannot be shuffled" in result["error"]
    assert "input_x_gradient" in result["fix"] and result["sequence_index"] == 1


def test_a_column_that_is_not_one_hot_is_an_error(tmp_path):
    X = server._sequences_to_tensor(SEQS_20[:2])
    X[1, :, 5] = 0.0  # an N
    path = tmp_path / "seqs.pt"
    torch.save(X, path)
    result = run(server.tfbs_analyze(model_id="relu", sequences_path=str(path),
                                     output_dir=str(tmp_path / "a")))
    assert "Sequence 1 cannot be shuffled" in result["error"]
    assert "input_x_gradient" in result["fix"]
    ixg = run(server.tfbs_analyze(model_id="relu", sequences_path=str(path),
                                  method="input_x_gradient", output_dir=str(tmp_path / "b")))
    assert ixg["shape"] == [2, 4, 20]


@pytest.mark.parametrize("kwargs", [{"n_shuffles": 0}, {"seed": -1}, {"batch_size": 0}])
def test_analyze_rejects_settings_out_of_range(kwargs):
    result = run(server.tfbs_analyze(model_id="relu", sequences=SEQS_20[:1], **kwargs))
    assert result["error"] and result["fix"]


# ---------------------------------------------------------------------------
# A3 tfbs_epistasis: batch_size
# ---------------------------------------------------------------------------

class BatchSpy(Linear):
    def __init__(self):
        super().__init__(length=6)
        self.batches = []

    def forward(self, x):
        self.batches.append(x.shape[0])
        return super().forward(x)


def test_epistasis_scores_mutants_in_batches_of_batch_size(monkeypatch):
    spy = BatchSpy()
    _register("spy", spy, length=6)
    explicit = run(server.tfbs_epistasis(model_id="spy", sequence="ACGTAC", batch_size=5))
    assert max(spy.batches) == 5
    spy.batches.clear()
    monkeypatch.setitem(server._ENV, "default_batch_size", 3)
    default = run(server.tfbs_epistasis(model_id="spy", sequence="ACGTAC"))
    assert max(spy.batches) == 3
    assert default["top_interactions"] == explicit["top_interactions"]
    assert default["original_prediction"] == explicit["original_prediction"]
    assert "error" in run(server.tfbs_epistasis(model_id="spy", sequence="ACGTAC", batch_size=0))


def test_epistasis_schema_has_batch_size_defaulting_to_null():
    tools = {t.name: t.inputSchema for t in asyncio.run(server.mcp.list_tools())}
    assert tools["tfbs_epistasis"]["properties"]["batch_size"].get("default") is None
    assert tools["tfbs_analyze"]["properties"]["seed"]["default"] == 42


# ---------------------------------------------------------------------------
# A4 slurm_status: NOT_FOUND only when SLURM answered
# ---------------------------------------------------------------------------

def _slurm(monkeypatch, squeue, sacct, available=("squeue", "sacct")):
    """squeue and sacct as (returncode, stdout, stderr) or an exception to raise."""
    monkeypatch.setattr(server.shutil, "which",
                        lambda name, *a, **k: f"/usr/bin/{name}" if name in available else None)

    def fake_run(cmd, *args, **kwargs):
        answer = {"squeue": squeue, "sacct": sacct}[cmd[0]]
        if isinstance(answer, BaseException):
            raise answer
        code, out, err = answer
        return subprocess.CompletedProcess(cmd, code, out, err)

    monkeypatch.setattr(server.subprocess, "run", fake_run)


def test_status_not_found_when_both_answer_without_the_job(monkeypatch):
    _slurm(monkeypatch, (1, "", "slurm_load_jobs error: Invalid job id specified"), (0, "", ""))
    assert run(server.slurm_status(job_id="77"))["state"] == "NOT_FOUND"


def test_status_of_a_purged_job_comes_from_sacct(monkeypatch):
    _slurm(monkeypatch, (1, "", "Invalid job id specified"),
           (0, "77|COMPLETED|node1|00:01:00|2026-10-14T10:00:00|0:0\n77.batch|COMPLETED|||||\n", ""))
    result = run(server.slurm_status(job_id="77"))
    assert result["state"] == "COMPLETED" and result["exit_code"] == "0:0"


def test_status_is_an_error_when_sacct_fails(monkeypatch):
    _slurm(monkeypatch, (0, "", ""), (1, "", "sacct: error: Problem talking to the database"))
    result = run(server.slurm_status(job_id="77"))
    assert "state" not in result
    assert "sacct failed (exit 1)" in result["error"]
    assert result["stderr"] == "sacct: error: Problem talking to the database"
    assert result["fix"] and result["job_id"] == "77"


def test_status_is_an_error_when_sacct_times_out(monkeypatch):
    _slurm(monkeypatch, subprocess.TimeoutExpired(["squeue"], 30),
           subprocess.TimeoutExpired(["sacct"], 30, stderr=b"slow"))
    result = run(server.slurm_status(job_id="77"))
    assert "sacct timed out" in result["error"] and "squeue timed out" in result["error"]
    assert result["stderr"] == "slow"


def test_status_is_an_error_when_sacct_is_missing_and_squeue_has_no_job(monkeypatch):
    _slurm(monkeypatch, (0, "", ""), FileNotFoundError("sacct"), available=("squeue",))
    result = run(server.slurm_status(job_id="77"))
    assert "sacct is not on PATH" in result["error"]


def test_status_running_job_comes_from_squeue(monkeypatch):
    _slurm(monkeypatch, (0, "77|RUNNING|node2|00:05:00|2026-10-14T10:00:00\n", ""),
           (1, "", "should not be asked"))
    assert run(server.slurm_status(job_id="77"))["state"] == "RUNNING"


# ---------------------------------------------------------------------------
# A5 tfbs_conv_filters: the strongest filters of each layer
# ---------------------------------------------------------------------------

class BiasConv(torch.nn.Module):
    """Five filters whose activation is their bias, so their ranking is known."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 5, kernel_size=3, padding=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.copy_(torch.tensor([0.1, 0.9, 0.5, 0.7, 0.3]))

    def forward(self, x):
        return self.conv(x).mean(dim=(1, 2)).unsqueeze(1)


def test_conv_filters_keeps_the_filters_with_the_highest_activation(tmp_path):
    _register("ranked", BiasConv())
    result = run(server.tfbs_conv_filters(model_id="ranked", sequences=SEQS_20[:2],
                                          top_n_filters=3, output_dir=str(tmp_path / "f")))
    assert [r["filter_id"] for r in result["filter_summary"]] == [1, 3, 2]
    assert [r["max_activation"] for r in result["filter_summary"]] == [0.9, 0.7, 0.5]
    tops = json.loads(Path(result["files"]["top_subsequences"]).read_text())
    assert [t["filter_id"] for t in tops] == [1, 3, 2]
    meme = Path(result["files"]["meme"]).read_text()
    assert "MOTIF conv_f1" in meme and "MOTIF conv_f0" not in meme


# ---------------------------------------------------------------------------
# A6 skip_foldx and the FoldX-missing stop
# ---------------------------------------------------------------------------

def _foldx_missing_stderr():
    from tfbs.mcp.foldx_tools import _FOLDX_MISSING, _FOLDX_MISSING_FIX
    return f"{_FOLDX_MISSING} {_FOLDX_MISSING_FIX}\n"


class Pipeline:
    """Fakes which(), sbatch and the pipeline process; records the pipeline commands."""

    def __init__(self, monkeypatch, returncode=0, stderr=""):
        self.cmds = []
        self.returncode, self.stderr = returncode, stderr
        monkeypatch.setattr(server.shutil, "which", lambda name, *a, **k: (
            f"/usr/bin/{name}" if name in ("sbatch", "fimo") else None))
        monkeypatch.setattr(server.subprocess, "run", self._run)
        monkeypatch.setattr(server, "_pipeline_requirements_error", lambda *a: None)
        fake_pipelines(monkeypatch, self._run)

    def _run(self, cmd, *args, **kwargs):
        if cmd[0] == "sbatch":
            return subprocess.CompletedProcess(cmd, 0, "Submitted batch job 9\n", "")
        self.cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, self.returncode, "", self.stderr)


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.delenv("TFBS_FOLDX_PDB", raising=False)
    out = {}
    for name in ("genome.fa", "motif.meme", "peaks.bed", "complex.pdb",
                 "clinvar.vcf.gz", "clinvar.vcf.gz.tbi"):
        (tmp_path / name).write_text("placeholder\n")
        out[name] = str(tmp_path / name)
    models = tmp_path / "models"
    for dataset in ("all_mean", "core_mean", "flank_mean"):
        ckpt = models / dataset / "standardize" / "best_VCNNBpnet.ckpt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_text("x")
    out["models_dir"] = str(models)
    (tmp_path / "intersect").mkdir()
    return out


def _comparison(tmp_path, files, **kwargs):
    return run(server.tfbs_chipseq_method_comparison(
        intersect_dir=str(tmp_path / "intersect"), output_dir=str(tmp_path / "cmp"),
        motif_file=files["motif.meme"], genome_fasta=files["genome.fa"],
        models_dir=files["models_dir"], **kwargs))


def _disease(tmp_path, files, **kwargs):
    return run(server.tfbs_disease_variant_validation(
        peaks_bed=files["peaks.bed"], output_dir=str(tmp_path / "dis"),
        clinvar_vcf=files["clinvar.vcf.gz"], genome_fasta=files["genome.fa"], **kwargs))


def test_method_comparison_with_skip_foldx_passes_no_structure(monkeypatch, tmp_path, files):
    pipe = Pipeline(monkeypatch)
    monkeypatch.setenv("TFBS_FOLDX_PDB", files["complex.pdb"])
    result = _comparison(tmp_path, files, skip_foldx=True, slurm=False)
    assert result["mode"] == "direct"
    assert result["skip_foldx"] is True and result["foldx_pdb"] is None
    assert "--foldx-pdb" not in pipe.cmds[-1]

    result = _comparison(tmp_path, files, skip_foldx=True, slurm=True)
    script = (tmp_path / "cmp" / "method_comparison.sh").read_text()
    assert result["job_id"] == "9" and "--foldx-pdb" not in script


def test_method_comparison_passes_the_structure_without_skip_foldx(monkeypatch, tmp_path, files):
    pipe = Pipeline(monkeypatch)
    result = _comparison(tmp_path, files, foldx_pdb=files["complex.pdb"], slurm=False)
    cmd = pipe.cmds[-1]
    assert cmd[cmd.index("--foldx-pdb") + 1] == str(Path(files["complex.pdb"]).resolve())
    assert result["foldx_pdb"] == str(Path(files["complex.pdb"]).resolve())
    assert result["skip_foldx"] is False


def test_method_comparison_without_a_structure_suggests_skip_foldx(monkeypatch, tmp_path, files):
    Pipeline(monkeypatch)
    result = _comparison(tmp_path, files, slurm=False)
    assert "FoldX" in result["error"] or "foldx" in result["error"].lower()
    assert "skip_foldx=true" in result["fix"]


@pytest.mark.parametrize("tool", ["comparison", "remap", "disease"])
def test_a_pipeline_stopped_for_lack_of_foldx_returns_that_error(monkeypatch, tmp_path, files, tool):
    from tfbs.mcp.foldx_tools import _FOLDX_MISSING

    Pipeline(monkeypatch, returncode=1, stderr=_foldx_missing_stderr())
    if tool == "comparison":
        result = _comparison(tmp_path, files, foldx_pdb=files["complex.pdb"], slurm=False)
    elif tool == "remap":
        result = run(server.tfbs_remap_validation(
            remap_bed=files["peaks.bed"], output_dir=str(tmp_path / "remap"),
            motif_file=files["motif.meme"], genome_fasta=files["genome.fa"],
            foldx_pdb=files["complex.pdb"], models_dir=files["models_dir"], slurm=False))
    else:
        result = _disease(tmp_path, files, foldx_pdb=files["complex.pdb"])
    assert result["error"].startswith(_FOLDX_MISSING)
    assert "skip_foldx=true" in result["fix"] and "TFBS_FOLDX_BIN" in result["fix"]


def test_disease_validation_skip_foldx_ignores_a_given_structure(monkeypatch, tmp_path, files):
    pipe = Pipeline(monkeypatch)
    result = _disease(tmp_path, files, foldx_pdb=files["complex.pdb"], skip_foldx=True)
    assert result["mode"] == "direct" and "--foldx-pdb" not in pipe.cmds[-1]
    assert result["foldx_pdb"] is None and result["skip_foldx"] is True

    result = _disease(tmp_path, files, foldx_pdb=files["complex.pdb"])
    cmd = pipe.cmds[-1]
    assert cmd[cmd.index("--foldx-pdb") + 1] == str(Path(files["complex.pdb"]).resolve())
    assert result["skip_foldx"] is False


def test_disease_validation_without_a_structure_skips_the_foldx_leg(monkeypatch, tmp_path, files):
    pipe = Pipeline(monkeypatch)
    result = _disease(tmp_path, files)
    assert "--foldx-pdb" not in pipe.cmds[-1] and result["foldx_pdb"] is None


# ---------------------------------------------------------------------------
# A7 every non-null schema default is stated in the tool's description
# ---------------------------------------------------------------------------

def _args_entries(doc: str) -> dict[str, str]:
    """{parameter: its text} from the Args section of a tool description."""
    match = re.search(r"\n[ \t]*Args:[ \t]*\n(.*?)(?=\n[ \t]*(?:Returns|Raises):|\Z)", doc, re.S)
    if not match:
        return {}
    entries: dict[str, list[str]] = {}
    indent = None
    current = None
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        lead = len(line) - len(line.lstrip())
        head = re.match(r"\s*([A-Za-z_]\w*)\s*(?:\([^)]*\))?:\s?(.*)", line)
        if head and (indent is None or lead == indent):
            indent, current = lead, head.group(1)
            entries[current] = [head.group(2)]
        elif current:
            entries[current].append(line.strip())
    return {name: " ".join(" ".join(parts).split()) for name, parts in entries.items()}


def _renderings(value) -> list[str]:
    """Ways a description may write a default value."""
    if isinstance(value, bool):
        return [str(value).lower()]
    if isinstance(value, (int, float)):
        out = [str(value), repr(value)]
        if isinstance(value, float):
            out += [f"{value:g}", f"{value:.0e}".replace("e-0", "e-")]
        return out
    if value == "":
        return ['""']
    if isinstance(value, list):
        return [", ".join(map(str, value)), ",".join(map(str, value))]
    return [str(value)]


def _tools():
    import tfbs.mcp.foldx_tools  # noqa: F401  (registers the structure tools)
    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


# Tools whose descriptions do not state every default yet (strict xfail);
# empty since foldx_tools.py states those of tfbs_binding_energy and
# tfbs_binding_scan.
_PENDING: set[str] = set()


def _tool_names():
    names = sorted(_tools())
    assert len(names) == 33
    return [pytest.param(n, marks=pytest.mark.xfail(strict=True, reason="foldx_tools.py defaults"))
            if n in _PENDING else n for n in names]


@pytest.mark.parametrize("name", _tool_names())
def test_every_non_null_default_is_stated_in_the_args_line(name):
    tool = _tools()[name]
    entries = _args_entries(tool.description or "")
    unstated = []
    for param, schema in tool.inputSchema.get("properties", {}).items():
        value = schema.get("default")
        if value is None:
            continue
        text = entries.get(param, "")
        if "default" not in text.lower() or not any(r in text for r in _renderings(value)):
            unstated.append(f"{param}={value!r}: {text[:80]!r}")
    assert not unstated, f"{name} leaves defaults unstated: {unstated}"


def test_the_contract_parser_reads_a_known_description():
    entries = _args_entries(server.tfbs_analyze.__doc__)
    assert "default 20" in entries["n_shuffles"]
    assert "default 42" in entries["seed"]
    assert set(entries) >= {"model_id", "method", "hypothetical", "batch_size"}


# ---------------------------------------------------------------------------
# A8 corrected descriptions
# ---------------------------------------------------------------------------

def _doc(tool) -> str:
    return " ".join(tool.__doc__.split())


def test_corrected_descriptions():
    config = _doc(server.tfbs_config)
    assert "is 0 or 1 and scaling_method is omitted or \"none\"" in config
    assert "with standardize or normalize on 0/1 labels it stays false" in config
    validate = _doc(server.tfbs_validate)
    assert "classifier whose checkpoint records none, is an error" in validate
    assert "8 CPUs and 64 GB" in validate
    bench = _doc(server.tfbs_chipseq_benchmark)
    assert "in three ways" in bench and "random subsample of this many (seed 42)" in bench
    assert "supported by at least cutoff experiments against the rest" in bench
    fimo = _args_entries(server.tfbs_fimo.__doc__)["motif_file"]
    assert "$TFBS_MOTIF_FILE" in fimo
    info = _doc(server.tfbs_server_info)
    assert "slurm_submit adds no partition" in info
    conv = _args_entries(server.tfbs_conv_filters.__doc__)["top_n_filters"]
    assert "highest maximum activation" in conv


class DropoutGRU(torch.nn.Module):
    """Dropout before a two-layer GRU; records the modes each forward pass saw."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.drop = torch.nn.Dropout(0.5)
        self.gru = torch.nn.GRU(4, 3, num_layers=2, dropout=0.5, batch_first=True)
        self.fc = torch.nn.Linear(3, 1)
        self.seen = []

    def forward(self, x):
        self.seen.append((self.drop.training, self.gru.training, self.gru.dropout))
        h, _ = self.gru(self.drop(x).permute(0, 2, 1))
        return self.fc(h[:, -1])


@pytest.mark.parametrize("method", ["deepliftshap", "input_x_gradient"])
def test_attribution_samples_no_dropout_and_runs_recurrent_layers_in_train_mode(method):
    model = DropoutGRU()
    _register("drop", model)
    try:
        calls = [run(server.tfbs_analyze("drop", sequences=SEQS_20[:3], method=method,
                                         n_shuffles=5))
                 for _ in range(2)]
    finally:
        server._model_registry.pop("drop", None)
    assert calls[0]["attributions"] == calls[1]["attributions"]
    assert model.seen and set(model.seen) == {(False, True, 0.0)}
    assert not model.training and not model.gru.training and model.gru.dropout == 0.5
