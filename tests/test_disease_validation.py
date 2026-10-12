"""tfbs_disease_variant_validation and its pipeline, tfbs.pipelines.disease_variant_validation.

The tool tests fake the pipeline process (tests/fake_pipeline.py); the
pipeline tests stub the ClinVar reader (pysam), FIMO and FoldX.
"""
import asyncio
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tfbs.mcp import server
from tfbs.pipelines import FIMO_REQUIRED
from tfbs.pipelines import disease_variant_validation as dv
from tests.fake_pipeline import fake_pipelines

SUMMARY_ROWS = [
    {"method": "NN flank_mean", "auc_abs": 0.71, "auc_raw": 0.64, "mwu_p": 0.003,
     "median_abs_path": 0.5, "median_abs_benign": 0.2, "n_path": 40, "n_benign": 60},
    {"method": "FIMO", "auc_abs": 0.55, "auc_raw": 0.5, "mwu_p": float("nan"),
     "median_abs_path": 0.1, "median_abs_benign": 0.1, "n_path": 38, "n_benign": 58},
]


def run(coro):
    return json.loads(asyncio.run(coro))


class Pipeline:
    """Fakes which(), sbatch and the pipeline process; records every command."""

    def __init__(self, monkeypatch, available=("sbatch", "fimo"), returncode=0,
                 stdout="", stderr="", write_summary=True):
        self.calls = []
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr
        self.write_summary = write_summary
        monkeypatch.setattr(server.shutil, "which",
                            lambda name, *a, **k: f"/usr/bin/{name}" if name in available else None)
        monkeypatch.setattr(server.subprocess, "run", self._run)
        fake_pipelines(monkeypatch, self._run)

    def _run(self, cmd, *args, **kwargs):
        self.calls.append(list(cmd))
        if cmd[0] == "sbatch":
            return subprocess.CompletedProcess(cmd, 0, "Submitted batch job 31\n", "")
        if self.write_summary and self.returncode == 0:
            out = Path(cmd[cmd.index("--output-dir") + 1])
            pd.DataFrame(SUMMARY_ROWS).to_csv(out / "summary_all_methods.csv", index=False)
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)

    def pipeline_cmd(self):
        (cmd,) = [c for c in self.calls if c[0] == sys.executable]
        return cmd


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(server, "_pipeline_requirements_error", lambda *a: None)
    monkeypatch.setitem(server._ENV, "default_device", "cpu")
    for var in ("TFBS_MOTIF_FILE", "TFBS_FOLDX_PDB", "TFBS_CLINVAR_VCF", "TFBS_GENOME_FASTA"):
        monkeypatch.delenv(var, raising=False)
    found = {}
    for name in ("genome.fa", "motif.meme", "peaks.bed", "clinvar.vcf.gz", "clinvar.vcf.gz.tbi",
                 "complex.pdb"):
        (tmp_path / name).write_text("placeholder\n")
        found[name] = str(tmp_path / name)
    ckpt = tmp_path / "saved_models" / "peaks" / "none" / "best_VCNNBpnet.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("x")
    found["ckpt"] = str(ckpt)
    return found


def _disease(files, tmp_path, **kwargs):
    kwargs.setdefault("clinvar_vcf", files["clinvar.vcf.gz"])
    return run(server.tfbs_disease_variant_validation(
        peaks_bed=files["peaks.bed"], output_dir=str(tmp_path / "out"),
        genome_fasta=files["genome.fa"], **kwargs))


def _flag(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


# ---------------------------------------------------------------------------
# (a) summary, nn_models and clinvar_vcf in the result
# ---------------------------------------------------------------------------

def test_direct_run_reads_the_summary_back(files, tmp_path, monkeypatch):
    Pipeline(monkeypatch)
    result = _disease(files, tmp_path, slurm=False,
                      nn_models={"flank_mean": "saved_models/peaks/none/best_VCNNBpnet.ckpt"})
    assert result["mode"] == "direct", result
    assert result["summary"] == [
        {"method": "NN flank_mean", "auc_abs": 0.71, "auc_raw": 0.64, "mwu_p": 0.003,
         "n_path": 40, "n_benign": 60},
        {"method": "FIMO", "auc_abs": 0.55, "auc_raw": 0.5, "mwu_p": None,
         "n_path": 38, "n_benign": 58},
    ]
    assert result["nn_models"] == {"flank_mean": str(Path(files["ckpt"]).resolve())}
    assert result["clinvar_vcf"] == str(Path(files["clinvar.vcf.gz"]).resolve())
    assert result["model_type"] == "VCNNBpnet"
    assert "summary_all_methods.csv" in result["files"]


def test_direct_run_without_a_summary_file_reports_null(files, tmp_path, monkeypatch):
    Pipeline(monkeypatch, write_summary=False)
    result = _disease(files, tmp_path, slurm=False)
    assert result["mode"] == "direct" and result["summary"] is None
    assert result["nn_models"] == {}


def test_slurm_run_names_the_summary_it_will_write(files, tmp_path, monkeypatch):
    Pipeline(monkeypatch)
    result = _disease(files, tmp_path, slurm=True, nn_models={"m": files["ckpt"]})
    assert result["job_id"] == "31", result
    assert result["expected_summary"] == str((tmp_path / "out" / "summary_all_methods.csv").resolve())
    assert result["nn_models"] == {"m": str(Path(files["ckpt"]).resolve())}
    assert result["clinvar_vcf"] == str(Path(files["clinvar.vcf.gz"]).resolve())


def test_a_missing_nn_models_checkpoint_is_an_error(files, tmp_path, monkeypatch):
    rec = Pipeline(monkeypatch)
    result = _disease(files, tmp_path, slurm=False, nn_models={"m": "nope/best.ckpt"})
    assert "do not exist" in result["error"] and "m=" in result["error"] and result["fix"]
    assert rec.calls == []


# ---------------------------------------------------------------------------
# (b) motif_file and foldx_pdb fall back to the environment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("given", ["", None])
def test_motif_and_structure_fall_back_to_the_environment(files, tmp_path, monkeypatch, given):
    monkeypatch.setenv("TFBS_MOTIF_FILE", files["motif.meme"])
    monkeypatch.setenv("TFBS_FOLDX_PDB", files["complex.pdb"])
    rec = Pipeline(monkeypatch)
    result = _disease(files, tmp_path, slurm=False, motif_file=given, foldx_pdb=given)
    cmd = rec.pipeline_cmd()
    assert _flag(cmd, "--motif-file") == str(Path(files["motif.meme"]).resolve())
    assert _flag(cmd, "--foldx-pdb") == str(Path(files["complex.pdb"]).resolve())
    assert result["motif_file"] == _flag(cmd, "--motif-file")
    assert result["foldx_pdb"] == _flag(cmd, "--foldx-pdb")


def test_without_argument_or_variable_the_legs_are_skipped(files, tmp_path, monkeypatch):
    rec = Pipeline(monkeypatch)
    result = _disease(files, tmp_path, slurm=False)
    cmd = rec.pipeline_cmd()
    assert "--motif-file" not in cmd and "--foldx-pdb" not in cmd
    assert result["motif_file"] is None and result["foldx_pdb"] is None


def test_an_argument_wins_over_the_variable(files, tmp_path, monkeypatch):
    other = tmp_path / "other.meme"
    other.write_text("x\n")
    monkeypatch.setenv("TFBS_MOTIF_FILE", files["motif.meme"])
    rec = Pipeline(monkeypatch)
    _disease(files, tmp_path, slurm=False, motif_file=str(other))
    assert _flag(rec.pipeline_cmd(), "--motif-file") == str(other.resolve())


def test_a_variable_naming_a_missing_file_is_an_error(files, tmp_path, monkeypatch):
    monkeypatch.setenv("TFBS_FOLDX_PDB", str(tmp_path / "missing.pdb"))
    rec = Pipeline(monkeypatch)
    result = _disease(files, tmp_path, slurm=False)
    assert "not found" in result["error"] and "TFBS_FOLDX_PDB" in result["fix"]
    assert rec.calls == []


# ---------------------------------------------------------------------------
# (d) ClinVar index, and an empty variant set
# ---------------------------------------------------------------------------

def test_a_vcf_without_its_index_is_an_error_before_anything_runs(files, tmp_path, monkeypatch):
    Path(files["clinvar.vcf.gz.tbi"]).unlink()
    rec = Pipeline(monkeypatch)
    for slurm in (False, True):
        result = _disease(files, tmp_path, slurm=slurm)
        assert "no tabix index" in result["error"] and "tabix -p vcf" in result["fix"]
    assert rec.calls == []


def test_a_csi_index_is_accepted(files, tmp_path, monkeypatch):
    Path(files["clinvar.vcf.gz.tbi"]).rename(files["clinvar.vcf.gz"] + ".csi")
    Pipeline(monkeypatch)
    assert _disease(files, tmp_path, slurm=False)["mode"] == "direct"


def test_no_variants_in_the_peaks_is_an_error_not_an_empty_success(files, tmp_path, monkeypatch):
    Pipeline(monkeypatch, returncode=1, stdout="Peaks at confidence >= 1: 5\n",
             stderr="No ClinVar variants found in the 5 peaks at confidence >= 1. Check ...")
    result = _disease(files, tmp_path, slurm=False)
    assert "No ClinVar variant" in result["error"]
    assert "genome build" in result["fix"] and "peak_confidence" in result["fix"]
    assert "mode" not in result


def test_pipeline_exits_non_zero_when_no_variant_is_found(tmp_path, monkeypatch, capsys):
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t100\t200\n")
    vcf = tmp_path / "clinvar.vcf.gz"
    vcf.write_text("x")
    (tmp_path / "clinvar.vcf.gz.tbi").write_text("x")
    monkeypatch.setattr(dv, "extract_clinvar_snvs", lambda peaks, path: pd.DataFrame())
    with pytest.raises(SystemExit) as stop:
        dv.main(["--peaks-bed", str(bed), "--clinvar-vcf", str(vcf), "--genome", "g.fa",
                 "--output-dir", str(tmp_path / "out")])
    assert "No ClinVar variants found in the 1 peaks" in str(stop.value.code)
    assert stop.value.code not in (0, None)


def test_pipeline_stops_on_a_vcf_without_index(tmp_path):
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t100\t200\n")
    with pytest.raises(SystemExit) as stop:
        dv.main(["--peaks-bed", str(bed), "--clinvar-vcf", str(tmp_path / "c.vcf.gz"),
                 "--genome", "g.fa", "--output-dir", str(tmp_path / "out")])
    assert "no tabix index" in stop.value.code and "tabix -p vcf" in stop.value.code
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# (c) BED files with 3 or more columns
# ---------------------------------------------------------------------------

def test_chip_atlas_allintersect_bed_reads_the_4th_column_as_the_count(tmp_path):
    bed = tmp_path / "NKX2-1.allintersect.bed"
    bed.write_text(
        "track name=x\n"
        "chr1\t100\t200\t3\tNKX2-1\t0\t.\t100\t200\n"
        "chr1\t150\t260\t5\tNKX2-1\t0\t.\t150\t260\n"
        "chr2\t10\t20\t1\tNKX2-1\t0\t.\t10\t20\n")
    peaks = dv.read_peaks_bed(str(bed))
    assert list(peaks.columns) == ["chrom", "start", "end", "num"]
    assert peaks.values.tolist() == [["chr1", 100, 200, 3], ["chr1", 150, 260, 5],
                                     ["chr2", 10, 20, 1]]
    merged = dv.merge_bed_intervals(str(bed), min_confidence=3)
    assert merged.values.tolist() == [["chr1", 100, 260, 5]]


def test_a_three_column_bed_counts_every_peak_once(tmp_path):
    bed = tmp_path / "peaks.bed"
    bed.write_text("# comment\nchr1\t100\t200\nchr1\t300\t400\n")
    merged = dv.merge_bed_intervals(str(bed))
    assert merged.values.tolist() == [["chr1", 100, 200, 1], ["chr1", 300, 400, 1]]
    with pytest.raises(ValueError, match="line 2 has no 4th column"):
        dv.merge_bed_intervals(str(bed), min_confidence=2)


def test_a_bed_with_a_name_in_column_4_ignores_it(tmp_path):
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t100\t200\tpeak1\t500\t+\nchr1\t300\t400\tpeak2\t20\t-\n")
    peaks = dv.read_peaks_bed(str(bed))
    assert peaks["num"].tolist() == [1, 1]


def test_a_bed_line_with_too_few_columns_names_the_line(tmp_path):
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t100\t200\nchr1\t300\n")
    with pytest.raises(ValueError, match="line 2 has fewer than 3 columns"):
        dv.read_peaks_bed(str(bed))


# ---------------------------------------------------------------------------
# (e) method labels
# ---------------------------------------------------------------------------

def test_summary_labels_read_nn_fimo_and_foldx(tmp_path):
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "label": ["pathogenic"] * (n // 2) + ["benign"] * (n // 2),
        "nn_flank_mean_dd": rng.normal(size=n),
        "fimo_dd": rng.normal(size=n),
        "foldx_ref_dG": rng.normal(size=n),
        "foldx_ddG": rng.normal(size=n),
    })
    summary = dv.evaluate(df, tmp_path)
    assert summary["method"].tolist() == ["NN flank_mean", "FIMO", "FoldX"]
    with open(tmp_path / "summary_all_methods.csv", encoding="utf-8") as f:
        assert [r["method"] for r in csv.DictReader(f)] == ["NN flank_mean", "FIMO", "FoldX"]
    assert not any("foldxG" in m for m in summary["method"])


def test_method_label_keeps_a_model_name_with_underscores():
    assert dv.method_label("nn_my_dd_model_dd") == "NN my_dd_model"
    assert dv.method_label("nn_ref") is None


# ---------------------------------------------------------------------------
# (f) model_type
# ---------------------------------------------------------------------------

def test_model_type_reaches_the_pipeline(files, tmp_path, monkeypatch):
    rec = Pipeline(monkeypatch)
    result = _disease(files, tmp_path, slurm=False, model_type="RNN",
                      nn_models={"m": files["ckpt"]})
    assert _flag(rec.pipeline_cmd(), "--model-type") == "RNN"
    assert result["model_type"] == "RNN"
    assert _flag(_disease(files, tmp_path, slurm=True)["command"].split(), "--model-type") == "VCNNBpnet"


class FakeNet(torch.nn.Module):
    # A model class is one defined in tfbs.nn.models under its own name.
    __module__ = "tfbs.nn.models"
    loaded = []

    @classmethod
    def load_from_checkpoint(cls, path, map_location=None):
        cls.loaded.append((path, map_location))
        return cls()

    def forward(self, x):
        return x[:, 0, :].sum(dim=1)


def test_pipeline_loads_every_checkpoint_as_the_model_type(monkeypatch):
    import tfbs.nn.models as models_mod

    monkeypatch.setattr(models_mod, "FakeNet", FakeNet, raising=False)
    FakeNet.loaded = []
    genome = {"chr1": "ACGTACGTAC" * 10}
    df = pd.DataFrame([{"chrom": "chr1", "pos": 50, "ref": genome["chr1"][49], "alt": "A",
                        "label": "benign"}])
    scored = dv.score_with_nn(df, {"m": "/ckpt/m.ckpt"}, genome, window=24, device="cpu",
                              model_type="FakeNet")
    assert FakeNet.loaded == [("/ckpt/m.ckpt", "cpu")]
    assert {"nn_m_ref", "nn_m_alt", "nn_m_dd"} <= set(scored.columns)


def test_pipeline_rejects_an_unknown_model_type():
    with pytest.raises(SystemExit) as stop:
        dv.model_class("NoSuchNet")
    assert "NoSuchNet" in stop.value.code and "VCNNBpnet" in stop.value.code


@pytest.mark.parametrize("name", ["BaseModel", "IrrepBatchNorm", "_MissingEquirc"])
def test_pipeline_rejects_a_helper_class_as_the_model_type(name):
    """Only the model classes the server lists are a model type, not its helpers."""
    with pytest.raises(SystemExit) as stop:
        dv.model_class(name)
    known = stop.value.code.split("(known: ")[1]
    assert name not in known.split(", ") and "VCNNBpnet" in known


# ---------------------------------------------------------------------------
# fimo for the FIMO leg (and for ReMap, which always runs it)
# ---------------------------------------------------------------------------

def test_missing_fimo_is_an_error_when_the_fimo_leg_runs(files, tmp_path, monkeypatch):
    rec = Pipeline(monkeypatch, available=("sbatch",))
    for slurm in (False, True):
        result = _disease(files, tmp_path, slurm=slurm, motif_file=files["motif.meme"])
        assert "fimo" in result["error"] and "tfbs_fimo" in result["error"]
        assert "MEME Suite" in result["fix"] and "TFBS_MOTIF_FILE" in result["fix"]
    assert rec.calls == []


def test_missing_fimo_does_not_matter_without_a_fimo_leg(files, tmp_path, monkeypatch):
    Pipeline(monkeypatch, available=("sbatch",))
    assert _disease(files, tmp_path, slurm=False)["mode"] == "direct"


@pytest.mark.parametrize("slurm", [False, True])
def test_remap_without_fimo_returns_an_error_before_running(files, tmp_path, monkeypatch, slurm):
    rec = Pipeline(monkeypatch, available=("sbatch",))
    result = run(server.tfbs_remap_validation(
        remap_bed=files["peaks.bed"], output_dir=str(tmp_path / "remap"),
        genome_fasta=files["genome.fa"], motif_file=files["motif.meme"], skip_foldx=True,
        checkpoints={"peaks": files["ckpt"]}, slurm=slurm))
    assert "fimo" in result["error"] and "tfbs_fimo" in result["error"]
    assert "MEME Suite" in result["fix"]
    assert rec.calls == []


def test_pipeline_with_a_motif_and_no_fimo_stops(tmp_path, monkeypatch):
    vcf = tmp_path / "clinvar.vcf.gz"
    vcf.write_text("x")
    (tmp_path / "clinvar.vcf.gz.tbi").write_text("x")
    monkeypatch.setattr(dv.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as stop:
        dv.main(["--peaks-bed", "p.bed", "--clinvar-vcf", str(vcf), "--genome", "g.fa",
                 "--output-dir", str(tmp_path / "out"), "--motif-file", "m.meme"])
    assert stop.value.code == FIMO_REQUIRED


def test_docstrings_name_the_fimo_requirement():
    for tool in (server.tfbs_remap_validation, server.tfbs_disease_variant_validation):
        doc = " ".join(tool.__doc__.split())
        assert "fimo program" in doc and "tangermeme fallback of tfbs_fimo does not apply" in doc


# ---------------------------------------------------------------------------
# (g) a confidence threshold needs a count in column 4 on every line
# ---------------------------------------------------------------------------

CHIP_ATLAS_PEAK_BROWSER = (
    "chr1\t100\t200\tID=SRX100;Name=NKX2-1%20(@%20A549);Title=x;\t500\t.\t100\t200\t0,0,255\n"
    "chr1\t300\t400\tID=SRX101;Name=NKX2-1%20(@%20A549);Title=y;\t800\t.\t300\t400\t0,0,255\n")


@pytest.mark.parametrize("text, line, shown", [
    (CHIP_ATLAS_PEAK_BROWSER, 1, "ID=SRX100"),
    ("chr1\t100\t200\t3\nchr1\t300\t400\t.\nchr2\t5\t9\t4\n", 2, "'.'"),
])
def test_a_threshold_on_a_bed_without_counts_names_the_line(tmp_path, text, line, shown):
    bed = tmp_path / "peaks.bed"
    bed.write_text(text)
    with pytest.raises(ValueError) as exc:
        dv.merge_bed_intervals(str(bed), min_confidence=3)
    message = str(exc.value)
    assert f"line {line} has" in message and shown in message
    assert "cannot be applied" in message and "column 4" in message
    # Without a threshold the file is still usable: every peak counts 1.
    assert dv.merge_bed_intervals(str(bed))["num"].tolist() == [1] * len(
        dv.merge_bed_intervals(str(bed)))


def test_pipeline_stops_on_a_threshold_the_bed_cannot_meet(tmp_path):
    bed = tmp_path / "peaks.bed"
    bed.write_text(CHIP_ATLAS_PEAK_BROWSER)
    vcf = tmp_path / "clinvar.vcf.gz"
    vcf.write_text("x")
    (tmp_path / "clinvar.vcf.gz.tbi").write_text("x")
    with pytest.raises(SystemExit) as stop:
        dv.main(["--peaks-bed", str(bed), "--clinvar-vcf", str(vcf), "--genome", "g.fa",
                 "--output-dir", str(tmp_path / "out"), "--peak-confidence", "3"])
    assert str(stop.value.code).startswith("Cannot read --peaks-bed:")
    assert "line 1 has" in stop.value.code and "No ClinVar" not in stop.value.code


def test_the_tool_reports_a_threshold_the_bed_cannot_meet(files, tmp_path, monkeypatch):
    Pipeline(monkeypatch, returncode=1,
             stderr="Cannot read --peaks-bed: /p.bed line 2 has '.' in column 4, not a number, "
                    "so peak confidence >= 3 cannot be applied: ...")
    result = _disease(files, tmp_path, slurm=False, peak_confidence=3)
    assert "line 2 has '.' in column 4" in result["error"]
    assert "peak_confidence=1" in result["fix"] and "column 4" in result["fix"]


def test_fractional_counts_are_kept_not_truncated(tmp_path):
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t100\t200\t2.9\nchr1\t150\t260\t1.5\nchr2\t10\t20\t2\n")
    assert dv.read_peaks_bed(str(bed))["num"].tolist() == [2.9, 1.5, 2.0]
    merged = dv.merge_bed_intervals(str(bed), min_confidence=2.5)
    assert merged.values.tolist() == [["chr1", 100, 260, 2.9]]


def test_integer_counts_stay_integers(tmp_path):
    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t100\t200\t3.0\nchr2\t10\t20\t2\n")
    nums = dv.read_peaks_bed(str(bed))["num"].tolist()
    assert nums == [3, 2] and all(isinstance(n, int) for n in nums)


def test_a_byte_order_mark_is_not_part_of_the_first_chromosome(tmp_path):
    bed = tmp_path / "peaks.bed"
    bed.write_bytes(b"\xef\xbb\xbfchr1\t100\t200\t3\nchr2\t10\t20\t4\n")
    peaks = dv.read_peaks_bed(str(bed))
    assert peaks["chrom"].tolist() == ["chr1", "chr2"]
    assert peaks["num"].tolist() == [3, 4]


# ---------------------------------------------------------------------------
# (h) chromosome names: chr1 or 1 in the BED, chr1 or 1 in the VCF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bed_chrom, contigs, expected", [
    ("chr1", {"1", "2"}, "1"),
    ("1", {"1", "2"}, "1"),
    ("chr1", {"chr1", "chr2"}, "chr1"),
    ("1", {"chr1", "chr2"}, "chr1"),
    ("chrM", {"MT"}, "MT"),
    ("MT", {"chrM"}, "chrM"),
    ("chr3", {"1", "2"}, None),
])
def test_vcf_contig_finds_either_naming(bed_chrom, contigs, expected):
    assert dv.vcf_contig(bed_chrom, contigs) == expected


class _Rec:
    def __init__(self, pos, clnsig):
        self.pos, self.ref, self.alts, self.id = pos, "A", ("G",), "rs1"
        self.info = {"CLNSIG": (clnsig,), "MC": ("SO:0001627|intron_variant",),
                     "GENEINFO": "NKX2-1:7080"}


class _FakeVcf:
    def __init__(self, contigs, records, header=True):
        self.records = records
        self.fetched = []

        class _Header:
            pass

        self.header = _Header()
        self.header.contigs = {c: None for c in contigs} if header else {}
        self._contigs = set(contigs)

    def fetch(self, chrom, start, end):
        self.fetched.append(chrom)
        if chrom not in self._contigs:
            raise ValueError(f"invalid contig {chrom}")
        return [r for r in self.records.get(chrom, []) if start <= r.pos <= end]

    def close(self):
        pass


def _fake_pysam(monkeypatch, vcf):
    import types

    module = types.ModuleType("pysam")
    module.VariantFile = lambda path: vcf
    monkeypatch.setitem(sys.modules, "pysam", module)


@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("bed_chrom", ["chr1", "1"])
def test_extraction_finds_variants_in_a_chr_prefixed_vcf(monkeypatch, bed_chrom, header):
    vcf = _FakeVcf(["chr1", "chr2"], {"chr1": [_Rec(150, "Pathogenic"), _Rec(160, "Benign")]},
                   header=header)
    _fake_pysam(monkeypatch, vcf)
    peaks = pd.DataFrame([{"chrom": bed_chrom, "start": 100, "end": 200, "num": 3}])
    hits = dv.extract_clinvar_snvs(peaks, "clinvar.vcf.gz")
    assert hits["label"].tolist() == ["pathogenic", "benign"]
    assert hits["chrom"].tolist() == [bed_chrom, bed_chrom]
    assert vcf.fetched[-1] == "chr1"


@pytest.mark.parametrize("bed_chrom", ["chr1", "1"])
def test_extraction_finds_variants_in_an_ncbi_named_vcf(monkeypatch, bed_chrom):
    vcf = _FakeVcf(["1", "2"], {"1": [_Rec(150, "Likely_pathogenic")]})
    _fake_pysam(monkeypatch, vcf)
    peaks = pd.DataFrame([{"chrom": bed_chrom, "start": 100, "end": 200, "num": 1}])
    hits = dv.extract_clinvar_snvs(peaks, "clinvar.vcf.gz")
    assert hits["label"].tolist() == ["pathogenic"]
    assert vcf.fetched == ["1"]


def test_extraction_skips_records_without_an_alt_allele(monkeypatch):
    """ClinVar writes ALT "." for some records; pysam reports alts None for them."""
    no_alt = _Rec(155, "Pathogenic")
    no_alt.alts = None
    none_inside = _Rec(156, "Pathogenic")
    none_inside.alts = (None,)
    vcf = _FakeVcf(["chr1"], {"chr1": [_Rec(150, "Pathogenic"), no_alt, none_inside, _Rec(160, "Benign")]})
    _fake_pysam(monkeypatch, vcf)
    peaks = pd.DataFrame([{"chrom": "chr1", "start": 100, "end": 200, "num": 1}])
    hits = dv.extract_clinvar_snvs(peaks, "clinvar.vcf.gz")
    assert hits["pos"].tolist() == [150, 160]
    assert hits["label"].tolist() == ["pathogenic", "benign"]


class _Genome(dict):
    """A pyfaidx-like genome: genome[name][s:e] gives a sequence slice."""


def test_extract_seq_pair_cuts_an_odd_window_centred_on_the_variant():
    genome = _Genome({"chr1": "ACGTACGTAC" + "T" + "GCATGCATGC"})  # variant T at 0-based 10
    ref, alt = dv.extract_seq_pair(genome, "chr1", 11, "T", "G", half_window=5, length=11)
    assert ref == "CGTACTGCATG"
    assert alt == "CGTACGGCATG"


def test_extract_seq_pair_default_length_is_twice_half_window_and_tries_both_contig_names():
    genome = _Genome({"1": "ACGTACGTACTGCATGCATGCATG"})
    ref, alt = dv.extract_seq_pair(genome, "chr1", 11, "T", "G", half_window=5)
    assert len(ref) == 10 and ref[5] == "T" and alt[5] == "G"
    assert dv.extract_seq_pair(genome, "chr1", 11, "A", "G", half_window=5) == (None, None)
    assert dv.extract_seq_pair(genome, "chr9", 11, "T", "G", half_window=5) == (None, None)


def test_foldx_leg_scores_an_eleven_bp_duplex_and_keeps_every_row(monkeypatch):
    """The 11 bp crystal duplex needs an 11 bp window; unscorable rows stay with NaN."""
    from tfbs.mcp import foldx_tools

    genome = _Genome({"chr1": "ACGTACGTAC" + "T" + "GCATGCATGC" + "A" * 30})
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/foldx" if name == "foldx" else None)
    monkeypatch.setattr(foldx_tools, "_parse_dna_from_pdb", lambda pdb: {"C": "GGTCAAGTGTT", "D": "AACACTTGACC"})
    # The fake mutation string carries the window's centre base, so the fake
    # energy can tell the reference window (T) from the variant window (G).
    monkeypatch.setattr(foldx_tools, "_build_foldx_mutation_string",
                        lambda ref, seq, chain: f"X{seq[5]}6;")
    monkeypatch.setattr(foldx_tools, "_run_foldx_energy",
                        lambda binary, pdb, wd, ms: -8.0 if ms and "XT6" in ms else -7.5)

    # The real thread pool runs here: the fakes above are module attributes
    # that worker threads share, which a process pool could not pickle.
    df = pd.DataFrame([
        {"chrom": "chr1", "pos": 11, "ref": "T", "alt": "G", "label": "pathogenic", "nn_m_dd": 0.5},
        {"chrom": "chr1", "pos": 2, "ref": "C", "alt": "A", "label": "benign", "nn_m_dd": 0.1},  # window runs off the contig start
    ])
    out = dv.score_with_foldx(df, genome, "complex_Repair.pdb", n_workers=1)
    assert len(out) == 2, "no row may be dropped"
    assert out.loc[0, "foldx_ddG"] == pytest.approx(0.5)
    assert np.isnan(out.loc[1, "foldx_ddG"]) and out.loc[1, "nn_m_dd"] == pytest.approx(0.1)


def test_no_variant_messages_do_not_claim_the_vcf_may_use_any_names(files, tmp_path, monkeypatch):
    Pipeline(monkeypatch, returncode=1, stderr="No ClinVar variants found in the 5 peaks")
    fix = _disease(files, tmp_path, slurm=False)["fix"]
    assert "(chr1 and 1 both work)" not in fix
    assert "chr1 in the BED finds 1 or chr1 in the VCF" in fix


# ---------------------------------------------------------------------------
# (i) arguments checked before anything runs or is submitted
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["a=b", "two words", ""])
@pytest.mark.parametrize("slurm", [False, True])
def test_an_nn_models_name_the_pipeline_cannot_parse_is_an_error(files, tmp_path, monkeypatch,
                                                                 name, slurm):
    rec = Pipeline(monkeypatch)
    result = _disease(files, tmp_path, slurm=slurm, nn_models={name: files["ckpt"]})
    assert "nn_models" in result["error"] and repr(name) in result["error"]
    assert "'='" in result["fix"]
    assert rec.calls == []


@pytest.mark.parametrize("slurm", [False, True])
def test_an_unknown_model_type_is_an_error_before_submitting(files, tmp_path, monkeypatch, slurm):
    rec = Pipeline(monkeypatch)
    result = _disease(files, tmp_path, slurm=slurm, model_type="Nope",
                      nn_models={"m": files["ckpt"]})
    assert result["error"] == "Unknown model_type 'Nope'."
    assert "VCNNBpnet" in result["available"] and "VCNNBpnet" in result["fix"]
    assert rec.calls == []


def test_model_type_is_not_checked_without_an_nn_leg(files, tmp_path, monkeypatch):
    Pipeline(monkeypatch)
    assert _disease(files, tmp_path, slurm=True, model_type="Nope")["mode"] == "slurm"


@pytest.mark.parametrize("slurm", [False, True])
def test_remap_rejects_an_unknown_model_type_before_submitting(files, tmp_path, monkeypatch,
                                                               slurm):
    rec = Pipeline(monkeypatch)
    result = run(server.tfbs_remap_validation(
        remap_bed=files["peaks.bed"], output_dir=str(tmp_path / "remap"),
        genome_fasta=files["genome.fa"], motif_file=files["motif.meme"], skip_foldx=True,
        checkpoints={"peaks": files["ckpt"]}, model_type="Nope", slurm=slurm))
    assert result["error"] == "Unknown model_type 'Nope'."
    assert rec.calls == []


def test_an_output_dir_that_is_a_file_is_an_error_not_an_exception(files, tmp_path, monkeypatch):
    rec = Pipeline(monkeypatch)
    blocker = tmp_path / "taken"
    blocker.write_text("x")
    result = run(server.tfbs_disease_variant_validation(
        peaks_bed=files["peaks.bed"], output_dir=str(blocker), clinvar_vcf=files["clinvar.vcf.gz"],
        genome_fasta=files["genome.fa"], slurm=True))
    assert "Cannot create the directory" in result["error"] and "already exists" in result["error"]
    assert "output_dir" in result["fix"]
    result = run(server.tfbs_remap_validation(
        remap_bed=files["peaks.bed"], output_dir=str(blocker / "sub"),
        genome_fasta=files["genome.fa"], motif_file=files["motif.meme"], skip_foldx=True,
        checkpoints={"peaks": files["ckpt"]}, slurm=True))
    assert "Cannot create the directory" in result["error"] and "output_dir" in result["fix"]
    assert rec.calls == []


# ---------------------------------------------------------------------------
# Inference checkpoints (the NKX2-1 study's released vcnnbpnet.pt format)
# ---------------------------------------------------------------------------

def _released_like(path: Path) -> Path:
    from tfbs.nn import models

    torch.manual_seed(0)
    model = models.VCNNBpnet(num_channels=4, kernel_size=3, dilations=[1, 2],
                             pool_output_size=5, dense_sizes=[8], input_length=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": "VCNNBpnet", "hyper_parameters": dict(model.hparams),
                "state_dict": model.state_dict(), "library": "flank",
                "scaling": {"scaling_method": "standardize", "mean": 0.0, "std": 1.0}}, path)
    return path


@pytest.mark.parametrize("slurm", [False, True])
def test_nn_models_take_an_inference_checkpoint(files, tmp_path, monkeypatch, slurm):
    rec = Pipeline(monkeypatch)
    ckpt = _released_like(tmp_path / "models" / "flank" / "vcnnbpnet.pt")
    result = _disease(files, tmp_path, slurm=slurm, nn_models={"flank": "models/flank/vcnnbpnet.pt"})
    assert "error" not in result, result
    assert result["nn_models"] == {"flank": str(ckpt.resolve())}
    cmd = rec.pipeline_cmd() if not slurm else result["command"].split()
    assert f"flank={ckpt.resolve()}" in " ".join(cmd).replace("'", "")


def test_nn_models_reject_an_inference_checkpoint_of_another_model_type(files, tmp_path,
                                                                       monkeypatch):
    rec = Pipeline(monkeypatch)
    ckpt = _released_like(tmp_path / "models" / "flank" / "vcnnbpnet.pt")
    result = _disease(files, tmp_path, slurm=False, nn_models={"flank": str(ckpt)},
                      model_type="RNN")
    assert "holds a VCNNBpnet model, but model_type is RNN" in result["error"]
    assert "model_type='VCNNBpnet'" in result["fix"] and result["arch"] == "VCNNBpnet"
    assert rec.calls == []


@pytest.mark.parametrize("slurm", [False, True])
def test_nn_models_reject_a_bare_state_dict_before_running(files, tmp_path, monkeypatch, slurm):
    rec = Pipeline(monkeypatch)
    released = _released_like(tmp_path / "models" / "flank" / "vcnnbpnet.pt")
    bare = tmp_path / "models" / "flank" / "weights.pt"
    torch.save(torch.load(released, weights_only=True)["state_dict"], bare)
    result = _disease(files, tmp_path, slurm=slurm, nn_models={"flank": str(bare)})
    assert "bare state_dict" in result["error"] and str(bare.resolve()) in result["error"]
    assert "{arch, hyper_parameters, state_dict}" in result["fix"]
    assert "Nothing was run or submitted" in result["fix"]
    assert rec.calls == []


def test_disease_pipeline_exits_cleanly_on_a_file_it_cannot_load(tmp_path):
    genome = {"chr1": "".join(np.random.RandomState(3).choice(list("ACGT"), 100))}
    ref = genome["chr1"][49]
    df = pd.DataFrame([{"chrom": "chr1", "pos": 50, "ref": ref,
                        "alt": "A" if ref != "A" else "C", "label": "benign"}])
    released = _released_like(tmp_path / "flank" / "vcnnbpnet.pt")
    bare = tmp_path / "flank" / "weights.pt"
    torch.save(torch.load(released, weights_only=True)["state_dict"], bare)
    corrupt = tmp_path / "flank" / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    for path, expected in ((bare, "bare state_dict"), (corrupt, "not a PyTorch checkpoint")):
        with pytest.raises(SystemExit) as stop:
            dv.score_with_nn(df, {"flank": str(path)}, genome, window=10, device="cpu",
                             model_type="VCNNBpnet")
        assert f"--models flank={path}" in stop.value.code and expected in stop.value.code
