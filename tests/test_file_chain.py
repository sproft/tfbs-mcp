"""Tools chain through files: derived output paths, matched negatives,
file inputs for the model tools, variant windows, conv-filter PFMs and
documented defaults."""
import asyncio
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from tfbs.mcp import server

DEMO = Path(server.__file__).resolve().parent.parent / "data" / "demo"
GENOME = DEMO / "demo_genome.fa"


def run(coro):
    return json.loads(asyncio.run(coro))


class MeanModel(torch.nn.Module):
    """Linear on the per-base mean: accepts any length, scores differ by base."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.1, 0.7, -0.3, 0.2]]))
            self.linear.bias.fill_(0.05)

    def forward(self, x):
        return self.linear(x.mean(dim=2))


class ACGDetector(torch.nn.Module):
    """One Conv1d filter that fires (activation 1) only on the 3-mer ACG."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 1, kernel_size=3, padding=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0, 0] = 1.0  # A at offset 0
            self.conv.weight[0, 1, 1] = 1.0  # C at offset 1
            self.conv.weight[0, 2, 2] = 1.0  # G at offset 2
            self.conv.bias.fill_(-2.0)

    def forward(self, x):
        return torch.relu(self.conv(x)).amax(dim=2)


def _register(model_id, model):
    model.eval()
    server._model_registry[model_id] = {
        "model": model, "model_type": type(model).__name__, "device": "cpu",
        "input_length": None, "classify": False, "scaling_params": None,
    }


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_results_root", lambda: tmp_path / "results")
    server._ensure_imports()
    _register("mean", MeanModel())
    _register("acg", ACGDetector())
    yield
    server._model_registry.pop("mean", None)
    server._model_registry.pop("acg", None)


@pytest.fixture
def demo_bed(tmp_path):
    bed = tmp_path / "peaks" / "demo_peaks.bed"
    bed.parent.mkdir()
    shutil.copy(DEMO / "demo_peaks.bed", bed)
    return bed


def _bed_rows(path):
    return [
        (c, int(s), int(e)) for c, s, e, *_ in
        (line.split("\t") for line in Path(path).read_text().splitlines() if line.strip())
    ]


def _genome_seq():
    lines = GENOME.read_text().splitlines()
    return "".join(line.strip() for line in lines[1:]).upper()


def _decode(t):
    return ["".join("ACGT"[i] for i in row) for row in t.argmax(dim=1).tolist()]


# ---------------------------------------------------------------------------
# tfbs_extract_loci: derived output directory, files, matched negatives
# ---------------------------------------------------------------------------

def test_extract_loci_derives_its_output_dir_from_the_bed(demo_bed):
    result = run(server.tfbs_extract_loci(bed_path=str(demo_bed), genome_fasta=str(GENOME)))

    out = demo_bed.parent / "demo_peaks_loci"
    assert result["output_dir"] == str(out)
    assert result["num_peaks_extracted"] == 4
    assert result["shape"] == [4, 4, 200]
    assert set(result["files"]) == {"sequences", "regions", "scores"}
    assert all(Path(p).is_file() and Path(p).parent == out for p in result["files"].values())
    assert result["sequences_path"] == result["files"]["sequences"]

    genome = _genome_seq()
    seqs = _decode(torch.load(result["files"]["sequences"], weights_only=True))
    for (chrom, s, e), seq in zip(_bed_rows(result["files"]["regions"]), seqs):
        assert seq == genome[s:e]
    scores = torch.load(result["files"]["scores"], weights_only=True)
    assert scores.flatten().tolist() == pytest.approx([10.37, 12.06, 10.36, 8.99])


def test_extract_loci_explicit_output_dir_wins(demo_bed, tmp_path):
    out = tmp_path / "mine"
    result = run(server.tfbs_extract_loci(
        bed_path=str(demo_bed), genome_fasta=str(GENOME), output_dir=str(out)))
    assert result["output_dir"] == str(out.resolve())
    assert Path(result["files"]["sequences"]).parent == out.resolve()


def test_extract_loci_needs_peaks_or_variants():
    result = run(server.tfbs_extract_loci(genome_fasta=str(GENOME)))
    assert result["error"] and result["fix"]
    result = run(server.tfbs_extract_loci(genome_fasta=str(GENOME), variants=["x:1:A>G"],
                                          negatives="gc_matched"))
    assert "bed_path" in result["fix"]
    result = run(server.tfbs_extract_loci(genome_fasta=str(GENOME), variants=["x:1:A>G"],
                                          negatives="other"))
    assert "gc_matched" in result["fix"]


def _gc(seq):
    return (seq.count("G") + seq.count("C")) / len(seq)


@pytest.mark.parametrize("window", [200, 24])
def test_gc_matched_negatives_on_the_demo_genome(demo_bed, window):
    result = run(server.tfbs_extract_loci(
        bed_path=str(demo_bed), genome_fasta=str(GENOME), in_window=window,
        negatives="gc_matched"))

    neg = result["negatives"]
    assert neg["mode"] == "gc_matched"
    assert neg["shortfall"] == 0 and neg["unfilled_bins"] == []
    assert neg["num_negatives"] == neg["num_positives"] == 4
    assert result["negatives_path"] == result["files"]["negatives"]
    assert Path(result["negatives_path"]) == Path(result["output_dir"]) / "negatives" / "sequences.pt"

    genome = _genome_seq()
    pos_seqs = _decode(torch.load(result["files"]["sequences"], weights_only=True))
    neg_tensor = torch.load(result["negatives_path"], weights_only=True)
    assert neg_tensor.shape == (4, 4, window)
    assert torch.all(neg_tensor.sum(dim=1) == 1)
    neg_seqs = _decode(neg_tensor)
    neg_rows = _bed_rows(result["files"]["negative_regions"])

    # Same GC histogram, bin by bin (bins of 0.05).
    def bins(seqs):
        return sorted(min(int(_gc(s) / 0.05 + 1e-9), 19) for s in seqs)
    assert bins(neg_seqs) == bins(pos_seqs)

    # Windows come from the genome, contain no N, and keep clear of the
    # positives (BED interval and window, padded by in_window) and each other.
    blocked = []
    for (_, s, e), (_, ws, we) in zip(_bed_rows(demo_bed), _bed_rows(result["files"]["regions"])):
        blocked.append((min(s, ws) - window, max(e, we) + window))
    for (chrom, s, e), seq in zip(neg_rows, neg_seqs):
        assert chrom == "demo_chr8" and e - s == window
        assert genome[s:e] == seq and set(seq) <= set("ACGT")
        assert all(e <= bs or s >= be for bs, be in blocked)
    starts = sorted(s for _, s, _ in neg_rows)
    assert all(b - a >= window for a, b in zip(starts, starts[1:]))


def test_gc_matched_negatives_are_seeded(demo_bed, tmp_path):
    def regions(out, seed):
        r = run(server.tfbs_extract_loci(
            bed_path=str(demo_bed), genome_fasta=str(GENOME), in_window=50,
            negatives="gc_matched", output_dir=str(tmp_path / out), random_seed=seed))
        return Path(r["files"]["negative_regions"]).read_text()

    assert regions("a", 7) == regions("b", 7)
    assert regions("c", 7) != regions("d", 8)


def test_shuffled_negatives_keep_dinucleotide_counts(demo_bed):
    result = run(server.tfbs_extract_loci(
        bed_path=str(demo_bed), genome_fasta=str(GENOME), in_window=60, negatives="shuffled"))
    assert result["negatives"]["mode"] == "shuffled"
    pos = _decode(torch.load(result["files"]["sequences"], weights_only=True))
    neg = _decode(torch.load(result["negatives_path"], weights_only=True))
    assert len(neg) == len(pos) == 4

    def dinucs(s):
        return sorted(s[i:i + 2] for i in range(len(s) - 1))
    for p, n in zip(pos, neg):
        assert dinucs(p) == dinucs(n)
    assert pos != neg


def test_chain_from_bed_to_dataset_needs_no_paths_from_the_user(demo_bed):
    loci = run(server.tfbs_extract_loci(
        bed_path=str(demo_bed), genome_fasta=str(GENOME), in_window=24, negatives="gc_matched"))
    clf = run(server.tfbs_prepare_classification(
        positive_path=loci["sequences_path"], negative_path=loci["negatives_path"]))
    assert clf["output_dir"] == str(demo_bed.parent / "demo_peaks_classification")
    assert (clf["num_positive"], clf["num_negative"]) == (4, 4)
    assert set(clf["files"]) == {"sequences", "labels"}

    ds = run(server.tfbs_prepare_dataset(
        sequences_path=clf["files"]["sequences"], labels_path=clf["files"]["labels"]))
    assert ds["output_dir"] == str(demo_bed.parent / "demo_peaks_dataset")
    assert ds["train_count"] + ds["val_count"] + ds["test_count"] == 8
    assert len(ds["files"]) == 6 and all(Path(p).is_file() for p in ds["files"].values())

    reg = run(server.tfbs_prepare_regression(
        sequences_path=loci["sequences_path"],
        scores_path=str(Path(loci["output_dir"]) / "scores.pt")))
    assert reg["output_dir"] == str(demo_bed.parent / "demo_peaks_regression")
    assert all(Path(p).is_file() for p in reg["files"].values())


def test_negatives_from_extract_loci_score_through_predict_and_metrics(demo_bed):
    loci = run(server.tfbs_extract_loci(
        bed_path=str(demo_bed), genome_fasta=str(GENOME), in_window=24, negatives="shuffled"))
    pos = run(server.tfbs_predict(model_id="mean", sequences_path=loci["sequences_path"]))
    neg = run(server.tfbs_predict(model_id="mean", sequences_path=loci["negatives_path"]))
    assert pos["num_sequences"] == neg["num_sequences"] == 4
    metrics = run(server.tfbs_classify_metrics(
        positive_scores=pos["scores"], negative_scores=neg["scores"]))
    assert "error" not in metrics


# ---------------------------------------------------------------------------
# Variant windows
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_genome(tmp_path):
    rng = np.random.RandomState(3)
    seq = "".join(rng.choice(list("ACGT"), size=300))
    fasta = tmp_path / "synthetic.fa"
    fasta.write_text(">chr1\n" + "\n".join(seq[i:i + 60] for i in range(0, 300, 60)) + "\n",
                     newline="\n")
    return fasta, seq


def test_variants_give_ref_and_alt_windows_and_flag_mismatches(synthetic_genome, tmp_path):
    fasta, seq = synthetic_genome
    pos = 101  # 1-based
    ref = seq[pos - 1]
    alt = "A" if ref != "A" else "C"
    wrong = "G" if ref != "G" else "T"
    variants = [
        f"chr1:{pos}:{ref}>{alt}",
        f"1:150:{seq[149]}>{'T' if seq[149] != 'T' else 'G'}",   # chr prefix added
        f"chr1:{pos}:{wrong}>{alt}",                               # wrong reference
        f"chr1:{pos}:{ref}>{ref}{alt}",                            # insertion
        "chr1:5:A>G",                                              # window off the start
        "chr2:50:A>G",                                             # unknown chromosome
        "not a variant",
    ]
    out = tmp_path / "var"
    result = run(server.tfbs_extract_loci(
        genome_fasta=str(fasta), variants=variants, in_window=20, output_dir=str(out)))

    v = result["variants"]
    assert (v["num_given"], v["num_extracted"]) == (7, 2)
    assert v["ref_mismatches"] == [
        {"variant": variants[2], "ref": wrong, "genome_ref": ref},
    ]
    assert [s["variant"] for s in v["skipped"]] == variants[3:]

    ref_t = torch.load(result["ref_sequences_path"], weights_only=True)
    alt_t = torch.load(result["alt_sequences_path"], weights_only=True)
    assert ref_t.shape == alt_t.shape == (2, 4, 20)
    ref_w, alt_w = _decode(ref_t), _decode(alt_t)
    # Window start = (pos - 1) - 10, the variant at offset 10.
    assert ref_w[0] == seq[90:110]
    assert alt_w[0] == seq[90:100] + alt + seq[101:110]
    assert ref_w[1] == seq[139:159]
    assert [i for i in range(20) if ref_w[1][i] != alt_w[1][i]] == [10]

    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["variant_offset"] == 10
    assert [r["status"] for r in manifest["variants"]] == [
        "ok", "ok", "ref_mismatch", "unsupported", "out_of_bounds", "unknown_chrom", "invalid",
    ]
    assert manifest["variants"][1]["chrom"] == "chr1"
    assert set(result["files"]) == {"ref_sequences", "alt_sequences", "variant_manifest"}


def test_variant_windows_score_with_predict(synthetic_genome):
    fasta, seq = synthetic_genome
    result = run(server.tfbs_extract_loci(
        genome_fasta=str(fasta), variants=[f"chr1:120:{seq[119]}>{'A' if seq[119] != 'A' else 'C'}"],
        in_window=24))
    assert Path(result["output_dir"]).parent.name == "variants"
    ref = run(server.tfbs_predict(model_id="mean", sequences_path=result["ref_sequences_path"]))
    alt = run(server.tfbs_predict(model_id="mean", sequences_path=result["alt_sequences_path"]))
    assert ref["scores"] != alt["scores"]


# ---------------------------------------------------------------------------
# sequences_path for tfbs_predict, tfbs_analyze and tfbs_mutagenesis
# ---------------------------------------------------------------------------

SEQS = ["ACGTACGTAC", "TTTTGGGGCC", "ACGACGACGA", "GGGCCCAAAT", "CATGCATGCA"]


def _inline_scores(**kw):
    return run(server.tfbs_predict(model_id="mean", sequences=SEQS, **kw))["scores"]


def test_predict_reads_a_tensor_file_in_batches(tmp_path):
    path = tmp_path / "seqs.pt"
    torch.save(server._sequences_to_tensor(SEQS), path)
    result = run(server.tfbs_predict(model_id="mean", sequences_path=str(path), batch_size=2))
    assert result["num_sequences"] == 5
    assert result["scores"] == pytest.approx(_inline_scores())


def test_predict_reads_a_fasta_file(tmp_path):
    fasta = tmp_path / "seqs.fa"
    fasta.write_text("".join(f">s{i} note\n{s[:5]}\n{s[5:].lower()}\n" for i, s in enumerate(SEQS)))
    result = run(server.tfbs_predict(model_id="mean", sequences_path=str(fasta), batch_size=3))
    assert result["ids"] == [f"s{i}" for i in range(5)]
    assert result["scores"] == pytest.approx(_inline_scores())
    window = run(server.tfbs_predict(model_id="mean", sequences_path=str(fasta),
                                     window_mode=True, window_size=4))
    assert len(window["best_scores"]) == 5 and window["ids"] == result["ids"]


def test_predict_reads_seqs_written_by_preprocess():
    pre = run(server.tfbs_preprocess(sequences=SEQS))
    result = run(server.tfbs_predict(model_id="mean", sequences_path=pre["files"]["seqs"]))
    assert result["scores"] == pytest.approx(_inline_scores())


@pytest.mark.parametrize("kwargs, needle", [
    ({"sequences": SEQS, "sequences_path": "x.pt"}, "exactly one"),
    ({}, "exactly one"),
    ({"sequences_path": "missing.pt"}, "not found"),
    ({"sequences": ["ACGN"]}, "other than A, C, G, T"),
    ({"sequences": ["ACGT", "ACG"]}, "differ in length"),
])
def test_sequence_input_errors_are_json(kwargs, needle, tmp_path):
    for tool in (server.tfbs_predict, server.tfbs_analyze, server.tfbs_mutagenesis):
        result = run(tool(model_id="mean", **kwargs))
        assert needle in result["error"] and result["fix"]


def test_sequence_file_errors_are_json(tmp_path):
    bad_shape = tmp_path / "bad.pt"
    torch.save(torch.zeros(3, 5, 10), bad_shape)
    assert "expected (N, 4, L)" in run(server.tfbs_predict(
        model_id="mean", sequences_path=str(bad_shape)))["error"]
    other = tmp_path / "seqs.txt"
    other.write_text("ACGT\n")
    assert "Unrecognised" in run(server.tfbs_predict(
        model_id="mean", sequences_path=str(other)))["error"]
    fasta = tmp_path / "n.fasta"
    fasta.write_text(">a\nACGT\n>b\nACNT\n")
    assert "Record b" in run(server.tfbs_predict(
        model_id="mean", sequences_path=str(fasta)))["error"]


def test_invalid_dna_in_marginalize_and_fimo_is_a_json_error():
    result = run(server.tfbs_marginalize(model_id="mean", sequences=["ACGN"], motif="AC"))
    assert "other than A, C, G, T" in result["error"] and result["fix"]
    result = run(server.tfbs_fimo(sequences=["ACGN"], motif_file=str(DEMO / "MA1994.1.meme")))
    assert "other than A, C, G, T" in result["error"] and result["fix"]


def test_unknown_model_is_a_json_error():
    for tool in (server.tfbs_predict, server.tfbs_analyze, server.tfbs_mutagenesis,
                 server.tfbs_seqlets, server.tfbs_conv_filters):
        result = run(tool(model_id="nope", sequences=["ACGT"]))
        assert "not loaded" in result["error"] and "tfbs_load_model" in result["fix"]
    result = run(server.tfbs_epistasis(model_id="nope", sequence="ACGT"))
    assert "not loaded" in result["error"]


def test_analyze_file_input_writes_attributions(tmp_path):
    path = tmp_path / "seqs.pt"
    X = server._sequences_to_tensor(SEQS)
    torch.save(X, path)
    inline = run(server.tfbs_analyze(model_id="mean", sequences=SEQS, method="input_x_gradient"))
    from_file = run(server.tfbs_analyze(model_id="mean", sequences_path=str(path),
                                        method="input_x_gradient", batch_size=2))
    assert "attributions" not in from_file
    saved = torch.load(from_file["files"]["attributions"], weights_only=True)
    assert saved.shape == (5, 4, 10) and from_file["shape"] == [5, 4, 10]
    assert saved.numpy() == pytest.approx(np.array(inline["attributions"]), abs=1e-6)
    assert from_file["attribution_sums"] == pytest.approx(saved.sum(dim=(1, 2)).tolist(), abs=1e-6)
    # Default location: <results>/<model_id>/tfbs_analyze/<digest>/.
    out = Path(from_file["files"]["attributions"]).parent
    assert out.parent == tmp_path / "results" / "mean" / "tfbs_analyze"
    assert Path(inline["files"]["attributions"]).parent != out


def test_analyze_deepliftshap_reads_a_fasta(tmp_path):
    fasta = tmp_path / "s.fa"
    fasta.write_text("".join(f">r{i}\n{s}\n" for i, s in enumerate(SEQS[:2])))
    result = run(server.tfbs_analyze(model_id="acg", sequences_path=str(fasta), n_shuffles=2,
                                     output_dir=str(tmp_path / "attr")))
    assert result["ids"] == ["r0", "r1"]
    assert result["files"]["attributions"] == str((tmp_path / "attr" / "attributions.pt").resolve())


def test_mutagenesis_file_input_writes_scores(tmp_path):
    path = tmp_path / "seqs.pt"
    torch.save(server._sequences_to_tensor(SEQS), path)
    inline = run(server.tfbs_mutagenesis(model_id="mean", sequences=SEQS))
    result = run(server.tfbs_mutagenesis(model_id="mean", sequences_path=str(path), batch_size=4))
    assert "scores" not in result and result["shape"] == [5, 4, 10]
    saved = torch.load(result["files"]["scores"], weights_only=True)
    assert saved.numpy() == pytest.approx(np.array(inline["scores"]), abs=1e-6)
    raw = run(server.tfbs_mutagenesis(model_id="mean", sequences_path=str(path), raw_outputs=True))
    assert set(raw["files"]) == {"original_scores", "mutant_scores"}
    assert all(Path(p).is_file() for p in raw["files"].values())


# ---------------------------------------------------------------------------
# Other tools that write files
# ---------------------------------------------------------------------------

def test_preprocess_always_saves_and_reports_files(tmp_path):
    a = run(server.tfbs_preprocess(sequences=SEQS, labels=[1, 0, 1, 0, 1]))
    b = run(server.tfbs_preprocess(sequences=SEQS, labels=[1, 0, 1, 0, 1]))
    assert a["saved_to"] == b["saved_to"]
    assert Path(a["saved_to"]).parent == tmp_path / "results" / "tfbs_preprocess"
    assert set(a["files"]) == {"seqs", "labels"}
    explicit = run(server.tfbs_preprocess(sequences=SEQS, output_path=str(tmp_path / "pp")))
    assert explicit["files"] == {"seqs": str((tmp_path / "pp" / "seqs.pt").resolve())}
    assert "fix" in run(server.tfbs_preprocess(sequences=SEQS, labels=[1.0]))


def test_conv_filters_write_meme_pfms_and_top_subsequences(tmp_path):
    from tangermeme.io import read_meme

    seqs = ["TTACGTTTTT", "GGGGACGGGG", "ACGTTTTTTT"]
    result = run(server.tfbs_conv_filters(model_id="acg", sequences=seqs))
    out = Path(result["output_dir"])
    assert out.parent == tmp_path / "results" / "acg" / "tfbs_conv_filters"
    assert set(result["files"]) == {"filter_summary", "meme", "top_subsequences"}
    assert result["num_pfms"] == 1

    motifs = read_meme(result["files"]["meme"])
    assert len(motifs) == 1
    pfm = next(iter(motifs.values())).numpy()
    # The window behind each activation accounts for padding=1, so the PFM
    # is exactly A, C, G.
    assert pfm == pytest.approx(np.eye(4)[:, :3])

    top = json.loads(Path(result["files"]["top_subsequences"]).read_text())
    assert top[0]["layer"] == "conv" and top[0]["filter_id"] == 0
    hits = top[0]["top"]
    assert [h["subsequence"] for h in hits[:3]] == ["ACG"] * 3
    assert sorted((h["sequence_index"], h["start"]) for h in hits[:3]) == [(0, 2), (1, 4), (2, 0)]
    summary = json.loads(Path(result["files"]["filter_summary"]).read_text())
    assert summary[0]["pfm_sites"] == 3 and summary[0]["in_meme"] is True


def test_conv_filters_skip_pfms_for_deeper_layers(tmp_path):
    class TwoLayers(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = torch.nn.Conv1d(4, 3, kernel_size=3)
            self.c2 = torch.nn.Conv1d(3, 2, kernel_size=3)

        def forward(self, x):
            return self.c2(torch.relu(self.c1(x))).mean(dim=2)

    torch.manual_seed(0)
    _register("two", TwoLayers())
    try:
        result = run(server.tfbs_conv_filters(model_id="two", sequences=SEQS,
                                              output_dir=str(tmp_path / "cf")))
    finally:
        server._model_registry.pop("two", None)
    by_layer = {}
    for rec in result["filter_summary"]:
        by_layer.setdefault(rec["layer"], []).append(rec)
    assert all(r["pfm_sites"] == 0 and not r["in_meme"] for r in by_layer["c2"])
    assert result["num_pfms"] == sum(r["in_meme"] for r in by_layer["c1"])
    assert Path(result["files"]["meme"]).read_text().startswith("MEME version 4")


def test_server_info_reports_the_results_dir(tmp_path):
    info = run(server.tfbs_server_info())
    assert info["results_dir"] == str(tmp_path / "results")


def test_results_root_outside_a_checkout(monkeypatch, tmp_path):
    monkeypatch.undo()
    site = tmp_path / "site-packages"
    site.mkdir()
    monkeypatch.delenv("TFBS_PROJECT_ROOT", raising=False)
    monkeypatch.setattr(server, "_project_root", lambda: site)
    monkeypatch.chdir(tmp_path)
    assert server._results_root() == tmp_path.resolve() / "tfbs_results"
    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path / "clone"))
    assert server._results_root() == site / "results"


def test_sibling_dir_strips_pipeline_suffixes(tmp_path):
    assert server._sibling_dir(tmp_path / "x_loci" / "sequences.pt", "dataset") == \
        tmp_path.resolve() / "x_dataset"
    assert server._sibling_dir(tmp_path / "data" / "seqs.pt", "dataset") == \
        tmp_path.resolve() / "data_dataset"


# ---------------------------------------------------------------------------
# Defaults are stated with a reason in the tool descriptions
# ---------------------------------------------------------------------------

def _descriptions():
    import tfbs.mcp.foldx_tools  # noqa: F401
    return {t.name: t.description for t in asyncio.run(server.mcp.list_tools())}


@pytest.mark.parametrize("tool, phrases", [
    ("tfbs_extract_loci", ["in_window 200", "negatives None", "gc_matched", "because"]),
    ("tfbs_prepare_dataset", ["64/16/20", "add_rc false", "random_seed 42"]),
    ("tfbs_prepare_classification", ["balance true", "negatives_path"]),
    ("tfbs_prepare_regression", ["log2", "standardize"]),
    ("tfbs_config", ["scaling_method", "standardize", "because"]),
    ("tfbs_analyze", ["deepliftshap", "default, because"]),
    ("tfbs_predict", ["Default false, because"]),
    ("tfbs_fetch_structure", ["co-crystal", "fallback", "because"]),
    ("tfbs_binding_scan", ["scheduler's default", "because FoldX runs on CPUs"]),
    ("tfbs_disease_variant_validation", ["default \"compute\"", "no GPU"]),
] + [
    (tool, ["$TFBS_SLURM_PARTITION", "because the job requests a GPU"])
    for tool in ("tfbs_train", "tfbs_sweep", "tfbs_validate", "tfbs_chipseq_benchmark",
                 "tfbs_chipseq_method_comparison", "tfbs_remap_validation")
])
def test_defaults_carry_their_rationale(tool, phrases):
    text = " ".join(_descriptions()[tool].split())
    for phrase in phrases:
        assert phrase in text, (tool, phrase)


def test_file_writing_tools_document_their_default_output():
    desc = _descriptions()
    for tool in ("tfbs_extract_loci", "tfbs_prepare_dataset", "tfbs_prepare_classification",
                 "tfbs_prepare_regression", "tfbs_preprocess", "tfbs_conv_filters",
                 "tfbs_epistasis", "tfbs_seqlets", "tfbs_analyze", "tfbs_mutagenesis"):
        text = " ".join(desc[tool].split())
        assert "files" in text and ("Default" in text or "default" in text), tool


def test_sources_have_no_long_dashes():
    for module in ("tfbs.mcp.server", "tfbs.mcp.foldx_tools"):
        import importlib
        text = Path(importlib.import_module(module).__file__).read_text(encoding="utf-8")
        assert chr(0x2014) not in text and chr(0x2013) not in text, module
