"""tfbs.nn.loading.load_checkpoint, the one loader of tfbs_load_model and the pipelines.

An inference checkpoint (the format of the NKX2-1 study's released
vcnnbpnet.pt files) and a Lightning checkpoint both load through it; the
pipeline tests stub FIMO and FoldX as tests/test_pipelines.py does.
"""
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch

from tfbs.nn import models
from tfbs.nn.loading import (
    UnimportableClassError, is_bare_state_dict, is_inference_payload, load_checkpoint,
    model_class, model_class_names, read_payload,
)
from tfbs.pipelines import chipseq_benchmark as cb
from tfbs.pipelines import chipseq_benchmark_with_foldx as cbf
from tfbs.pipelines import disease_variant_validation as dv
from tfbs.pipelines import remap_validation as rv
from tfbs.pipelines import validate

STD = {"scaling_method": "standardize", "mean": 5.0, "std": 2.0}
# The released models (Zenodo, doi:10.5281/zenodo.21399578): a folder holding
# <library>/vcnnbpnet.pt, from $TFBS_RELEASED_MODELS or the author's copy.
RELEASED_MODELS = Path(os.environ.get("TFBS_RELEASED_MODELS")
                       or "D:/claude_code/nkx2-1-tfbs-variants/models")
RELEASED_FLANK = RELEASED_MODELS / "flank" / "vcnnbpnet.pt"
FLANK_SEQ = "GGGCTCCTCACTTGACCTTTGCTG"


def _tiny_vcnn(seed: int = 0) -> models.VCNNBpnet:
    torch.manual_seed(seed)
    return models.VCNNBpnet(num_channels=4, kernel_size=3, dilations=[1, 2],
                            pool_output_size=5, dense_sizes=[8], input_length=10).eval()


def _save_inference(path: Path, model=None, **extra) -> Path:
    """An inference checkpoint as the study released it: arch, hyper_parameters, state_dict, scaling."""
    model = model or _tiny_vcnn()
    payload = {"arch": type(model).__name__, "library": "demo",
               "hyper_parameters": dict(model.hparams), "state_dict": model.state_dict(),
               "scaling": dict(STD), "metadata": {"study": "test"}, **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _save_lightning(path: Path, class_path: str | None = None) -> Path:
    torch.manual_seed(0)
    model = models.SimpleCNN(input_length=10)
    hparams = dict(model.hparams)
    if class_path:
        hparams["_class_path"] = class_path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": hparams,
                "pytorch-lightning_version": pl.__version__}, path)
    return path


def _raw(model, seqs):
    with torch.no_grad():
        return cbf.score_nn(model, seqs, device="cpu")


def _random_seqs(n: int, length: int, seed: int = 1) -> list[str]:
    rng = np.random.RandomState(seed)
    return ["".join(rng.choice(list("ACGT"), length)) for _ in range(n)]


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------

def test_inference_checkpoint_loads_strictly_with_its_info(tmp_path):
    model = _tiny_vcnn()
    path = _save_inference(tmp_path / "rel" / "vcnnbpnet.pt", model)
    assert is_inference_payload(read_payload(path))

    loaded, info = load_checkpoint(path, "VCNNBpnet")
    assert info.format == "inference" and info.arch == "VCNNBpnet"
    assert info.scaling == STD and info.library == "demo" and info.metadata == {"study": "test"}
    assert info.class_path is None and info.hyper_parameters["input_length"] == 10
    assert isinstance(loaded, models.VCNNBpnet) and not loaded.training
    state = loaded.state_dict()
    assert set(state) == set(model.state_dict())
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    seqs = _random_seqs(5, 10)
    np.testing.assert_allclose(_raw(loaded, seqs), _raw(model, seqs), rtol=1e-6)

    # Without model_type the file's arch is used.
    _, info = load_checkpoint(str(path))
    assert info.format == "inference" and info.arch == "VCNNBpnet"


def test_inference_state_dict_is_loaded_strictly(tmp_path):
    model = _tiny_vcnn()
    missing = dict(model.state_dict())
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(_save_inference(tmp_path / "missing.pt", model, state_dict=missing))
    extra = {**model.state_dict(), "stray.weight": torch.zeros(1)}
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_checkpoint(_save_inference(tmp_path / "extra.pt", model, state_dict=extra))


def test_inference_arch_must_be_a_model_class_and_the_model_type(tmp_path):
    path = _save_inference(tmp_path / "vcnnbpnet.pt")
    with pytest.raises(ValueError, match="VCNNBpnet model, but model_type is SimpleCNN"):
        load_checkpoint(path, "SimpleCNN")
    unknown = _save_inference(tmp_path / "unknown.pt", arch="NoSuchNet")
    with pytest.raises(ValueError, match="NoSuchNet"):
        load_checkpoint(unknown)


def test_overrides_replace_hyper_parameters_of_an_inference_checkpoint(tmp_path):
    path = _save_inference(tmp_path / "vcnnbpnet.pt")
    loaded, info = load_checkpoint(path, "VCNNBpnet", overrides={"input_length": 10})
    assert info.hyper_parameters["input_length"] == 10 and loaded.input_length == 10


def test_lightning_checkpoint_loads_by_model_type_or_by_its_class_path(tmp_path):
    path = _save_lightning(tmp_path / "best.ckpt", "tfbs.nn.models.SimpleCNN")
    model, info = load_checkpoint(path, "SimpleCNN")
    assert info.format == "lightning" and info.arch == "SimpleCNN"
    assert info.class_path == "tfbs.nn.models.SimpleCNN" and info.scaling is None
    assert info.library is None and info.metadata is None
    assert isinstance(model, models.SimpleCNN) and not model.training
    model, info = load_checkpoint(path)
    assert info.arch == "SimpleCNN" and isinstance(model, models.SimpleCNN)


def test_lightning_checkpoint_without_a_class_needs_model_type(tmp_path):
    path = _save_lightning(tmp_path / "best.ckpt")
    with pytest.raises(ValueError, match="model_type"):
        load_checkpoint(path)
    with pytest.raises(ValueError, match="NoSuchNet"):
        load_checkpoint(path, "NoSuchNet")
    absent = _save_lightning(tmp_path / "absent.ckpt", "nosuchshim_lq.models.SimpleCNN")
    with pytest.raises(UnimportableClassError) as err:
        load_checkpoint(absent)
    assert err.value.module == "nosuchshim_lq.models"


def _bare_and_corrupt(tmp_path: Path) -> dict[str, Path]:
    """A bare state_dict, a file of garbage and an empty file, under tmp_path."""
    bare = tmp_path / "bad" / "weights.pt"
    bare.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_tiny_vcnn().state_dict(), bare)
    corrupt = tmp_path / "bad" / "corrupt.pt"
    corrupt.write_bytes(b"not a checkpoint")
    empty = tmp_path / "bad" / "empty.ckpt"
    empty.write_bytes(b"")
    return {"bare state_dict": bare, "not a PyTorch checkpoint": corrupt, "empty": empty}


def test_a_bare_state_dict_or_a_corrupt_file_is_a_value_error(tmp_path):
    files = _bare_and_corrupt(tmp_path)
    assert is_bare_state_dict(read_payload(files["bare state_dict"]))
    assert not is_bare_state_dict(read_payload(_save_inference(tmp_path / "ok.pt")))
    assert not is_bare_state_dict(read_payload(_save_lightning(tmp_path / "ok.ckpt")))
    for expected, path in files.items():
        expected = "not a PyTorch checkpoint" if expected == "empty" else expected
        for model_type in (None, "VCNNBpnet"):
            with pytest.raises(ValueError, match=expected) as err:
                load_checkpoint(path, model_type)
            assert str(path) in str(err.value)
    # A missing file is the loader's to report, not a "not a checkpoint" error.
    with pytest.raises(Exception) as err:
        load_checkpoint(tmp_path / "missing.ckpt", "SimpleCNN")
    assert "not a PyTorch checkpoint" not in str(err.value)


def test_an_inference_checkpoint_is_not_left_mapped(tmp_path):
    """Overwriting the file right after loading it works (on Windows a mapped file cannot)."""
    import gc

    path = _save_inference(tmp_path / "rel" / "vcnnbpnet.pt")
    enabled = gc.isenabled()
    gc.disable()
    try:
        model, info = load_checkpoint(path, "VCNNBpnet")
        _save_inference(path, _tiny_vcnn(seed=1))
    finally:
        if enabled:
            gc.enable()
    assert info.format == "inference" and isinstance(model, models.VCNNBpnet)
    reloaded, _ = load_checkpoint(path, "VCNNBpnet")
    first = next(iter(reloaded.state_dict()))
    assert torch.equal(reloaded.state_dict()[first], _tiny_vcnn(seed=1).state_dict()[first])


def test_model_classes_are_the_ones_the_server_lists():
    from tfbs.mcp import server

    names = model_class_names()
    assert names == server._available_model_types()
    assert "VCNNBpnet" in names and "SimpleCNN" in names
    for helper in ("BaseModel", "IrrepBatchNorm", "IrrepActivationLayer", "_MissingEquirc"):
        assert helper not in names
        with pytest.raises(ValueError) as err:
            model_class(helper)
        assert helper not in str(err.value).split("(known: ")[1].split(", ")
    assert model_class("VCNNBpnet") is models.VCNNBpnet


# ---------------------------------------------------------------------------
# The pipelines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [rv, cb, cbf], ids=["remap", "chipseq", "method_comparison"])
def test_pipeline_loader_exits_with_one_line_for_any_load_failure(module, tmp_path):
    for expected, path in _bare_and_corrupt(tmp_path).items():
        expected = "not a PyTorch checkpoint" if expected == "empty" else expected
        with pytest.raises(SystemExit) as stop:
            module.load_model(str(path), "VCNNBpnet", "cpu")
        assert stop.value.code.startswith(f"Cannot load {path}: ValueError: ")
        assert expected in stop.value.code
    model = _tiny_vcnn()
    missing = dict(model.state_dict())
    missing.pop(next(iter(missing)))
    broken = _save_inference(tmp_path / "broken.pt", model, state_dict=missing)
    with pytest.raises(SystemExit) as stop:
        module.load_model(str(broken), "VCNNBpnet", "cpu")
    assert stop.value.code.startswith(f"Cannot load {broken}: RuntimeError: ")
    absent = _save_lightning(tmp_path / "absent.ckpt", "nosuchshim_pq.models.SimpleCNN")
    with pytest.raises(SystemExit) as stop:
        module.load_model(str(absent), None, "cpu")
    assert stop.value.code.startswith(f"Cannot load {absent}: UnimportableClassError: ")
    assert "nosuchshim_pq.models" in stop.value.code


@pytest.mark.parametrize("module", [rv, cb, cbf], ids=["remap", "chipseq", "method_comparison"])
def test_pipeline_loader_takes_an_inference_checkpoint(module, tmp_path):
    model = _tiny_vcnn()
    path = _save_inference(tmp_path / "flank" / "vcnnbpnet.pt", model)
    loaded = module.load_model(str(path), "VCNNBpnet", "cpu")
    assert isinstance(loaded, models.VCNNBpnet) and not loaded.training
    assert not any(p.requires_grad for p in loaded.parameters())
    seqs = _random_seqs(4, 10)
    np.testing.assert_allclose(_raw(loaded, seqs), _raw(model, seqs), rtol=1e-6)
    with pytest.raises(SystemExit) as stop:
        module.load_model(str(path), "SimpleCNN", "cpu")
    assert "VCNNBpnet" in str(stop.value.code)


@pytest.mark.parametrize("module", [rv, cb, cbf], ids=["remap", "chipseq", "method_comparison"])
def test_pipeline_loader_still_takes_a_lightning_checkpoint(module, tmp_path):
    path = _save_lightning(tmp_path / "best_SimpleCNN.ckpt")
    assert isinstance(module.load_model(str(path), "SimpleCNN", "cpu"), models.SimpleCNN)


def _genome(tmp_path: Path, length: int = 3000) -> Path:
    rng = np.random.RandomState(7)
    seq = "".join(rng.choice(list("ACGT"), length))
    fa = tmp_path / "genome.fa"
    fa.write_text(">chr1\n" + "\n".join(seq[i:i + 60] for i in range(0, length, 60)) + "\n")
    return fa


def test_remap_pipeline_scores_an_inference_checkpoint(tmp_path, monkeypatch):
    model = _tiny_vcnn()
    ckpt = _save_inference(tmp_path / "models" / "flank" / "vcnnbpnet.pt", model)
    fa = _genome(tmp_path)
    rows = []
    for i in range(6):
        start = 100 + 400 * i
        rows.append(["chr1", start, start + 40, f"peak{i}", 5, ".", start + 20, start + 21, "0"])
    bed = tmp_path / "remap.bed"
    bed.write_text("".join("\t".join(map(str, r)) + "\n" for r in rows))
    monkeypatch.setattr(rv.shutil, "which", lambda name: "/usr/bin/fimo" if name == "fimo" else None)
    monkeypatch.setattr(rv, "fimo_max_per_region", lambda seqs, motif: np.zeros(len(seqs)))
    out = tmp_path / "out"
    rv.main(["--remap-bed", str(bed), "--genome", str(fa), "--motif-file", "motif.meme",
             "--checkpoints", f"flank={ckpt}", "--model-type", "VCNNBpnet",
             "--nn-window", "10", "--negative-mode", "shuffle", "--skip-foldx",
             "--device", "cpu", "--output-dir", str(out)])

    pos = pd.read_csv(out / "positive_scores.csv")
    import pyfaidx

    genome = pyfaidx.Fasta(str(fa))
    regions = [rv.extract_full_region(pd.Series(dict(zip(
        ["chrom", "start", "end", "name", "score", "strand", "thickStart", "thickEnd", "itemRgb"],
        r))), genome, max_len=500) for r in rows]
    genome.close()
    expected = rv.nn_max_per_region(model, regions, 10, "cpu")
    np.testing.assert_allclose(pos["nn_flank_max"].to_numpy(), expected, rtol=1e-5)
    summary = pd.read_csv(out / "summary.csv")
    assert "NN flank" in set(summary.iloc[:, 0])


def test_method_comparison_pipeline_scores_an_inference_checkpoint(tmp_path, monkeypatch):
    model = _tiny_vcnn()
    ckpt = _save_inference(tmp_path / "models" / "flank" / "vcnnbpnet.pt", model)
    fa = _genome(tmp_path)
    intersect = tmp_path / "intersect"
    intersect.mkdir()
    for tf, offset in (("NKX2-1", 50), ("GATA1", 250)):
        lines = ["chrom\tstart\tend\tnum"]
        lines += [f"chr1\t{offset + 200 * i}\t{offset + 200 * i + 30}\t3" for i in range(12)]
        (intersect / f"{tf}.allintersect.bed").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(cbf, "score_fimo", lambda seqs, motif: np.zeros(len(seqs)))
    monkeypatch.setattr(cbf.shutil, "which", lambda name: None)
    out = tmp_path / "out"
    cbf.main(["--intersect-dir", str(intersect), "--output-dir", str(out), "--genome", str(fa),
              "--motif-file", "motif.meme", "--foldx-pdb", "", "--neg-tfs", "GATA1",
              "--checkpoints", f"flank={ckpt}", "--model-type", "VCNNBpnet",
              "--window", "10", "--n-per-tf", "12", "--device", "cpu"])
    raw = pd.read_csv(out / "raw_scores.csv")
    np.testing.assert_allclose(raw["NN_flank"].to_numpy(),
                               _raw(model, list(raw["sequence"])), rtol=1e-5)
    assert "NN_flank" in set(pd.read_csv(out / "auc_summary.csv")["method"])


def test_chipseq_benchmark_loads_its_checkpoints_through_the_shared_loader(tmp_path, monkeypatch):
    ckpt = _save_inference(tmp_path / "models" / "flank" / "vcnnbpnet.pt")
    loaded = []
    real = cb.load_model

    def spy(path, model_type, device):
        model = real(path, model_type, device)
        loaded.append((path, type(model).__name__))
        return model

    monkeypatch.setattr(cb, "load_model", spy)
    # The models load before the BED files are read, so a missing one stops the run after.
    with pytest.raises(FileNotFoundError):
        cb.main(["--intersect-dir", str(tmp_path / "none"), "--output-dir", str(tmp_path / "out"),
                 "--genome", "genome.fa", "--checkpoints", f"flank={ckpt}",
                 "--model-type", "VCNNBpnet", "--device", "cpu"])
    assert loaded == [(str(ckpt), "VCNNBpnet")]


def test_disease_pipeline_scores_an_inference_checkpoint(tmp_path):
    model = _tiny_vcnn()
    ckpt = _save_inference(tmp_path / "flank" / "vcnnbpnet.pt", model)
    genome = {"chr1": "".join(np.random.RandomState(3).choice(list("ACGT"), 100))}
    ref = genome["chr1"][49]
    alt = "A" if ref != "A" else "C"
    df = pd.DataFrame([{"chrom": "chr1", "pos": 50, "ref": ref, "alt": alt, "label": "benign"}])
    scored = dv.score_with_nn(df, {"flank": str(ckpt)}, genome, window=10, device="cpu",
                              model_type="VCNNBpnet")
    ref_seq, alt_seq = dv.extract_seq_pair(genome, "chr1", 50, ref, alt, 5)
    expected_ref, expected_alt = _raw(model, [ref_seq, alt_seq])
    assert scored["nn_flank_ref"][0] == pytest.approx(expected_ref, rel=1e-5)
    assert scored["nn_flank_dd"][0] == pytest.approx(expected_ref - expected_alt, rel=1e-4, abs=1e-6)
    with pytest.raises(SystemExit) as stop:
        dv.score_with_nn(df, {"flank": str(ckpt)}, genome, window=10, device="cpu",
                         model_type="SimpleCNN")
    assert "VCNNBpnet" in str(stop.value.code) and "flank" in str(stop.value.code)


def _tensor_dataset(root: Path, length: int = 10, n: int = 8) -> Path:
    gen = torch.Generator().manual_seed(0)
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True)
        idx = torch.randint(0, 4, (n, length), generator=gen)
        seqs = torch.nn.functional.one_hot(idx, 4).float().permute(0, 2, 1).contiguous()
        torch.save(seqs, root / split / "seqs.pt")
        torch.save(torch.randn(n, 1, generator=gen), root / split / "labels.pt")
    return root


def test_validate_runs_trainer_test_on_an_inference_checkpoint(tmp_path):
    ckpt = _save_inference(tmp_path / "flank" / "vcnnbpnet.pt")
    data = _tensor_dataset(tmp_path / "data" / "flank_dataset")
    out = tmp_path / "results.csv"
    validate.main(["--checkpoint_path", str(ckpt), "--model_name", "VCNNBpnet",
                   "--data_path", str(data), "--output_file", str(out),
                   "--batch_size", "4", "--num_workers", "0", "--scaling_method", "none"])
    with open(out, newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert row["modeltype"] == "VCNNBpnet" and row["dataset_path"] == "flank_dataset"
    assert np.isfinite(float(row["test_loss"]))


# ---------------------------------------------------------------------------
# The released NKX2-1 flank model
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RELEASED_FLANK.is_file(),
                    reason="the released NKX2-1 checkpoints are not on this machine")
def test_released_flank_checkpoint_through_the_shared_and_the_remap_loader():
    model, info = load_checkpoint(RELEASED_FLANK, "VCNNBpnet")
    assert info.format == "inference" and info.arch == "VCNNBpnet" and info.library == "flank"
    raw = float(_raw(model, [FLANK_SEQ])[0])
    assert raw == pytest.approx(1.2133, abs=1e-3)
    scaled = raw * info.scaling["std"] + info.scaling["mean"]
    assert info.scaling["scaling_method"] == "standardize"
    assert scaled == pytest.approx(1.3907, abs=1e-3)

    remap_model = rv.load_model(os.fspath(RELEASED_FLANK), "VCNNBpnet", "cpu")
    assert float(rv.nn_max_per_region(remap_model, [FLANK_SEQ], 24, "cpu")[0]) == \
        pytest.approx(1.2133, abs=1e-3)
