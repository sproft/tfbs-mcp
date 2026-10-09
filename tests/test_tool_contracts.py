"""Every tool answers with JSON: unknown models, bad DNA and missing inputs are
{"error", "fix"}, never an exception.  Also the job writers, tfbs_sweep's
tracking extra, clipped conv-filter PFMs and the FlexibleMLP template."""
import asyncio
import importlib.util
import inspect
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from tfbs.mcp import server

SEQ = "ACGTACGTAC"
REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "tfbs" / "data" / "demo"


def run(coro):
    return json.loads(asyncio.run(coro))


class TinyConv(torch.nn.Module):
    def __init__(self, kernel_size=3, padding=1, out_channels=2):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, out_channels, kernel_size=kernel_size, padding=padding)
        torch.nn.init.constant_(self.conv.weight, 0.1)
        torch.nn.init.constant_(self.conv.bias, 0.5)
        self.input_length = 10
        self.classify = False

    def forward(self, x):
        return torch.relu(self.conv(x)).mean(dim=(1, 2)).unsqueeze(1)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_results_root", lambda: tmp_path / "results")
    server._ensure_imports()
    model = TinyConv().eval()
    server._model_registry["tc"] = {"model": model, "model_type": "TinyConv", "device": "cpu",
                                    "input_length": 10, "classify": False, "scaling_params": None}
    yield
    server._model_registry.pop("tc", None)
    server._model_registry.pop("tc_nolen", None)
    server._model_registry.pop("tc_wide", None)


MODEL_TOOLS = {
    "tfbs_predict": lambda mid, s: server.tfbs_predict(model_id=mid, sequences=[s]),
    "tfbs_analyze": lambda mid, s: server.tfbs_analyze(model_id=mid, sequences=[s], n_shuffles=2),
    "tfbs_mutagenesis": lambda mid, s: server.tfbs_mutagenesis(model_id=mid, sequences=[s]),
    "tfbs_marginalize": lambda mid, s: server.tfbs_marginalize(model_id=mid, sequences=[s],
                                                               motif="AC"),
    "tfbs_seqlets": lambda mid, s: server.tfbs_seqlets(model_id=mid, sequences=[s], n_shuffles=2),
    "tfbs_epistasis": lambda mid, s: server.tfbs_epistasis(model_id=mid, sequence=s),
    "tfbs_conv_filters": lambda mid, s: server.tfbs_conv_filters(model_id=mid, sequences=[s]),
}


@pytest.mark.parametrize("tool", sorted(MODEL_TOOLS))
def test_unknown_model_id_is_a_json_error(tool):
    result = run(MODEL_TOOLS[tool]("no_such_model", SEQ))
    assert "no_such_model" in result["error"] and "not loaded" in result["error"]
    assert "tfbs_load_model" in result["fix"]


@pytest.mark.parametrize("bad", ["ACGTNNACGT", "acgtxyzacg", ""])
@pytest.mark.parametrize("tool", sorted(MODEL_TOOLS))
def test_invalid_dna_is_a_json_error(tool, bad):
    result = run(MODEL_TOOLS[tool]("tc", bad))
    assert result.get("error") and result.get("fix"), result


def test_get_model_returns_an_error_instead_of_raising():
    entry, err = server._get_model("missing")
    assert entry is None and json.loads(err)["fix"]
    entry, err = server._get_model("tc")
    assert err is None and entry["model_type"] == "TinyConv"


def test_validate_dna_is_the_shared_validator():
    assert server._validate_dna(["ACGT", "acgt"]) is None
    err = json.loads(server._validate_dna(["ACGT", "ACNT"]))
    assert "Sequence 1" in err["error"] and "'N'" in err["error"] and err["fix"]
    named = json.loads(server._validate_dna(["AC-"], "Record {} of x.fa", ["chr1"]))
    assert "Record chr1 of x.fa" in named["error"]


def test_no_unused_genomics_import():
    source = Path(server.__file__).read_text(encoding="utf-8")
    assert "extract_loci" not in source.split("def _ensure_genomics_imports")[1].split("def ")[0]
    assert not hasattr(server, "_extract_loci")


def test_marginalize_rejects_a_bad_motif():
    result = run(server.tfbs_marginalize(model_id="tc", sequences=[SEQ], motif="ACNNT"))
    assert "motif" in result["error"] and result["fix"]
    result = run(server.tfbs_marginalize(model_id="tc", sequences=[SEQ], motif="A" * 12))
    assert "longer" in result["error"]


def test_fimo_uses_the_shared_validator(tmp_path):
    motif = tmp_path / "m.meme"
    motif.write_text("MEME version 4\n")
    result = run(server.tfbs_fimo(sequences=["ACGT", "ACRT"], motif_file=str(motif)))
    assert "Sequence 1" in result["error"] and result["fix"]
    result = run(server.tfbs_fimo(sequences=[], motif_file=str(motif)))
    assert "tfbs_fimo got no sequences" in result["error"]


# ---------------------------------------------------------------------------
# Tools that take sequences inline only say so when they get none
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool, call, words", [
    ("tfbs_preprocess", lambda: server.tfbs_preprocess(sequences=[]), ["sequences_path"]),
    ("tfbs_marginalize", lambda: server.tfbs_marginalize(model_id="tc", sequences=None,
                                                         motif="AC"), ["n_background"]),
    ("tfbs_seqlets", lambda: server.tfbs_seqlets(model_id="tc", sequences=[]), ["inline"]),
    ("tfbs_conv_filters", lambda: server.tfbs_conv_filters(model_id="tc", sequences=[]),
     ["input length"]),
])
def test_tools_without_sequences_path_name_themselves(tool, call, words):
    result = run(call())
    assert result["error"] == f"{tool} got no sequences."
    assert "sequences_path" not in result["fix"] or tool == "tfbs_preprocess"
    for word in words:
        assert word in result["fix"]


def test_marginalize_backgrounds_need_an_input_length():
    server._model_registry["tc_nolen"] = {**server._model_registry["tc"], "input_length": None}
    result = run(server.tfbs_marginalize(model_id="tc_nolen", sequences=None, motif="AC",
                                         n_background=4))
    assert "input length" in result["error"] and "sequences" in result["fix"]
    ok = run(server.tfbs_marginalize(model_id="tc", sequences=None, motif="AC", n_background=3))
    assert len(ok["delta_predictions"]) == 3


def test_marginalize_returns_lists_for_one_sequence():
    result = run(server.tfbs_marginalize(model_id="tc", sequences=[SEQ], motif="ac"))
    for key in ("predictions_before", "predictions_after", "delta_predictions"):
        assert isinstance(result[key], list) and len(result[key]) == 1


# ---------------------------------------------------------------------------
# tfbs_conv_filters: receptive field longer than the input
# ---------------------------------------------------------------------------

def test_conv_filters_clip_a_receptive_field_longer_than_the_input(tmp_path):
    """The all_mean template: kernel 42 on 24 bp windows still gives PFMs."""
    model = TinyConv(kernel_size=42, padding=21).eval()
    server._model_registry["tc_wide"] = {"model": model, "model_type": "TinyConv",
                                         "device": "cpu", "input_length": 24,
                                         "classify": False, "scaling_params": None}
    seqs = ["ACGTTGCAACGTAGCTAGCTTACG", "TTGCAACGTAGCTAGCTTACGACG"]
    result = run(server.tfbs_conv_filters(model_id="tc_wide", sequences=seqs,
                                          output_dir=str(tmp_path)))
    assert "error" not in result, result
    assert result["num_pfms"] == 2
    assert "receptive field 42 bp" in result["warning"] and "24 bp input" in result["warning"]
    meme = (tmp_path / "filters.meme").read_text()
    widths = [int(line.split("w= ")[1].split()[0]) for line in meme.splitlines()
              if line.startswith("letter-probability")]
    assert widths and all(0 < w <= 42 for w in widths)
    for record in result["filter_summary"]:
        assert record["pfm_sites"] > 0 and record["pfm_width"] == widths[0]
    rows = [line.split() for line in meme.splitlines() if line.startswith(" ")]
    assert all(abs(sum(float(v) for v in row) - 1.0) < 1e-5 for row in rows)
    top = json.loads((tmp_path / "top_subsequences.json").read_text())
    assert all(set(t["subsequence"]) <= set("ACGT") for f in top for t in f["top"])


def test_conv_filters_inside_the_input_have_no_warning(tmp_path):
    result = run(server.tfbs_conv_filters(model_id="tc", sequences=[SEQ],
                                          output_dir=str(tmp_path)))
    assert "warning" not in result
    assert all(r["pfm_width"] == 3 for r in result["filter_summary"] if r["in_meme"])


# ---------------------------------------------------------------------------
# tfbs_train config mode and tfbs_sweep
# ---------------------------------------------------------------------------

class Sbatch:
    def __init__(self, monkeypatch, available=("sbatch",)):
        self.calls = []
        monkeypatch.setattr(server.shutil, "which",
                            lambda name, *a, **k: f"/usr/bin/{name}" if name in available else None)
        monkeypatch.setattr(server.subprocess, "run", self._run)

    def _run(self, cmd, *a, **k):
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "Submitted batch job 9\n", "")


def test_train_config_mode_quotes_the_command_and_sets_pythonpath(tmp_path, monkeypatch):
    rec = Sbatch(monkeypatch)
    config = tmp_path / "my config.yaml"
    config.write_text("model: {}\n")
    extra = ["--trainer.max_epochs=3", "--data.data_path=/data/with space"]
    result = run(server.tfbs_train(config_path=str(config), extra_args=extra))
    assert result["job_id"] == "9"
    script = result["generated_script"]
    expected = shlex.join([sys.executable, "-m", "tfbs.cli.train_cli", "fit", "--config",
                           str(config.resolve()), *extra])
    assert expected in script
    assert server._sbatch_pythonpath_line() in script
    written = Path(rec.calls[-1][-1]).read_bytes()
    assert b"\r\n" not in written and expected.encode() in written


def test_sweep_without_wandb_names_the_tracking_extra(tmp_path, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "wandb" else real(name, *a))
    rec = Sbatch(monkeypatch, available=("sbatch", "wandb"))
    cfg = tmp_path / "sweep.yaml"
    cfg.write_text("method: bayes\n")
    result = run(server.tfbs_sweep(sweep_config=str(cfg), project_name="p", model="RNN",
                                   data_path=str(tmp_path), data_name="d"))
    assert "wandb" in result["error"]
    assert 'pip install "tfbs-mcp[tracking]"' in result["fix"]
    assert rec.calls == []


def test_sweep_description_mentions_the_tracking_extra():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert 'pip install "tfbs-mcp[tracking]"' in tools["tfbs_sweep"].description


# ---------------------------------------------------------------------------
# tfbs_epistasis, tfbs_validate and tfbs_extract_loci contracts
# ---------------------------------------------------------------------------

def test_epistasis_description_names_both_csv_names_and_counts():
    doc = " ".join(server.tfbs_epistasis.__doc__.split())
    assert "epistasis_matrix.csv" in doc and "epistasis_top.csv" in doc
    assert "num_position_pairs" in doc and "num_pairs_analyzed (the number of double mutants" in doc


def test_validate_description_lists_the_classifier_metrics():
    doc = " ".join(server.tfbs_validate.__doc__.split())
    assert "test_acc" in doc and "test_auc" in doc and "ROC-AUC" in doc


def test_nan_scores_are_counted_apart_from_non_acgt_windows(tmp_path):
    bed = tmp_path / "peaks" / "p.bed"
    bed.parent.mkdir()
    rows = [line.split("\t") for line in (DEMO / "demo_peaks.bed").read_text().splitlines()]
    rows[1][4] = "nan"
    bed.write_text("".join("\t".join(r) + "\n" for r in rows))
    result = run(server.tfbs_extract_loci(bed_path=str(bed),
                                          genome_fasta=str(DEMO / "demo_genome.fa")))
    assert result["num_filtered_score"] == 1
    assert result["num_filtered_n"] == 0
    assert result["num_peaks_extracted"] == 3
    doc = " ".join(server.tfbs_extract_loci.__doc__.split())
    assert "num_filtered_score" in doc


# ---------------------------------------------------------------------------
# FlexibleMLP: template, server note and developer guide agree with the code
# ---------------------------------------------------------------------------

def test_flexible_mlp_template_runs_on_one_hot_windows():
    from tfbs.nn import models

    template = yaml.safe_load((REPO / "tfbs" / "configs" / "models" / "FlexibleMLP.yaml").read_text())
    args = template["init_args"]
    assert args["input_length"] == 4 * 24
    model = models.FlexibleMLP(**args).eval()
    with torch.no_grad():
        assert tuple(model(torch.zeros(2, 4, 24)).shape) == (2, 1)


def test_flexible_mlp_descriptions_agree():
    template = (REPO / "tfbs" / "configs" / "models" / "FlexibleMLP.yaml").read_text()
    guide = " ".join((REPO / "docs" / "development.md").read_text(encoding="utf-8").split())
    note = server._ARCH_NOTES["FlexibleMLP"]
    assert "4 x L" in template and "flattens" in template
    assert "flattens" in note and "4 x window" in note
    assert "FlexibleMLP" in guide and "4 x" in guide
    assert "FlexibleMLP` expects tabular features" not in guide


# ---------------------------------------------------------------------------
# README and developer guide match the code
# ---------------------------------------------------------------------------

def _doc(name):
    return " ".join((REPO / name).read_text(encoding="utf-8").split())


def test_readme_troubleshooting_quotes_the_fimo_error_the_tool_returns():
    readme = _doc("README.md")
    assert "fails with `FileNotFoundError`" not in readme
    assert server._FIMO_MISSING["error"] in readme


def test_docs_say_where_the_clinvar_tool_runs():
    for name in ("README.md", "docs/development.md"):
        text = _doc(name)
        assert "Linux, macOS or WSL" in text, name
        assert "no pysam wheel" in text or "no Windows wheel" in text, name


def test_readme_lists_every_setup_flag():
    from tfbs.mcp import cli

    readme = _doc("README.md")
    flags = [v.flag for v in cli.SETUP_VARIABLES] + ["--non-interactive", "--demo"]
    section = readme.split("Flags prefill answers")[1].split("For example")[0]
    for flag in flags:
        assert f"`{flag}" in section, flag


def test_readme_states_the_checkpoint_convention():
    readme = _doc("README.md")
    assert "<models_dir>/<dataset>/<scaling_method>/best_<model_type>.ckpt" in readme
    assert "/standardize/best_<model_type>.ckpt`" not in readme
    guide = _doc("docs/development.md")
    assert "checkpoint_path" in guide and "--checkpoints NAME=PATH" in guide


def test_docs_do_not_say_a_clone_is_needed_for_the_validation_tools():
    for name in ("README.md", "docs/development.md"):
        text = _doc(name).lower()
        assert "clone the repository" not in text and "git clone" not in text, name


def test_no_dashes_in_changed_text():
    for name in ("README.md", "docs/development.md", "tfbs/mcp/server.py",
                 "tfbs/mcp/foldx_tools.py", "tfbs/pipelines/checkpoints.py",
                 "tfbs/pipelines/validate.py", "tfbs/configs/models/FlexibleMLP.yaml"):
        text = (REPO / name).read_text(encoding="utf-8")
        assert chr(0x2014) not in text and chr(0x2013) not in text, name
