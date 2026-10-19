"""Tool replies stay under the MCP message size limit (TFBS_MAX_INLINE_BYTES).

On 19 October 2026 a tfbs_predict reply for 220,293 window-mode sequences (about 15.5 MB)
made Claude Code close the stdio server ("wrote >16MB to stdout without a JSON-RPC message
boundary"), losing every loaded model.  Per-item results above the reply limit now go to a
file, and the reply names the file with its row count, checksum and summary statistics.
"""
import asyncio
import csv
import hashlib
import json
import random

import pytest
import torch


class FakeModel(torch.nn.Module):
    """Minimal model that returns a scalar per sequence."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = torch.nn.Linear(4, 1)
        self.input_length = 10
        self.classify = False

    def forward(self, x):
        return self.linear(x.mean(dim=2))


@pytest.fixture(autouse=True)
def _registry(monkeypatch, tmp_path):
    from tfbs.mcp import server

    monkeypatch.setattr(server, "_results_root", lambda: tmp_path / "results")
    monkeypatch.delenv("TFBS_MAX_INLINE_BYTES", raising=False)
    server._model_registry["bounded_model"] = {"model": FakeModel().eval(), "model_type": "FakeModel",
                                               "device": "cpu", "input_length": 10, "classify": False}
    server._ensure_imports()
    yield server
    server._model_registry.pop("bounded_model", None)


def _run(coro):
    return json.loads(asyncio.run(coro))


def _seqs(n, length=10, seed=0):
    rnd = random.Random(seed)
    return ["".join(rnd.choice("ACGT") for _ in range(length)) for _ in range(n)]


def _table(desc):
    with open(desc["path"], newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_small_prediction_stays_inline(_registry):
    r = _run(_registry.tfbs_predict(model_id="bounded_model", sequences=_seqs(5)))
    assert len(r["scores"]) == 5
    assert "results_file" not in r and "inline" not in r


def test_large_prediction_goes_to_a_file(_registry, monkeypatch):
    seqs = _seqs(300)
    full = _run(_registry.tfbs_predict(model_id="bounded_model", sequences=seqs))
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "1000")
    r = _run(_registry.tfbs_predict(model_id="bounded_model", sequences=seqs))
    assert "scores" not in r and r["inline"] is False
    desc = r["results_file"]
    header, rows = _table(desc)
    assert header == ["scores"] == desc["columns"]
    assert desc["rows"] == len(rows) == 300 == r["num_sequences"]
    assert [float(row[0]) for row in rows] == full["scores"]  # full precision, same order
    assert desc["sha256"] == _sha(desc["path"])
    assert r["summary"]["scores"]["n"] == 300
    assert r["summary"]["scores"]["min"] == min(full["scores"])
    assert r["summary"]["scores"]["max"] == max(full["scores"])
    assert r["units"] == full["units"] and r["inverse_transformed"] == full["inverse_transformed"]
    assert len(json.dumps(r)) < 2000


def test_window_mode_with_fasta_ids_goes_to_a_file(_registry, monkeypatch, tmp_path):
    fa = tmp_path / "peaks.fa"
    seqs = _seqs(50, length=30, seed=3)
    fa.write_text("".join(f">peak{i}\n{s}\n" for i, s in enumerate(seqs)))
    kw = dict(model_id="bounded_model", sequences_path=str(fa), window_mode=True, window_size=10)
    full = _run(_registry.tfbs_predict(**kw))
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "500")
    r = _run(_registry.tfbs_predict(**kw))
    header, rows = _table(r["results_file"])
    assert header == ["ids", "best_scores", "mean_scores", "best_sequences"]
    assert [row[0] for row in rows] == full["ids"]
    assert [float(row[1]) for row in rows] == full["best_scores"]
    assert [float(row[2]) for row in rows] == full["mean_scores"]
    assert [row[3] for row in rows] == full["best_sequences"]
    assert "best_sequences" not in r["summary"] and "ids" not in r["summary"]


def test_limit_zero_always_writes_a_file(_registry, monkeypatch):
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "0")
    r = _run(_registry.tfbs_predict(model_id="bounded_model", sequences=_seqs(2)))
    assert r["results_file"]["rows"] == 2 and "scores" not in r


def test_unreadable_limit_falls_back_to_the_default(_registry, monkeypatch):
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "not-a-number")
    assert _registry._max_inline_bytes() == 512 * 1024
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "-5")
    assert _registry._max_inline_bytes() == 0


def test_server_info_reports_the_limit(_registry, monkeypatch):
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "12345")
    assert _run(_registry.tfbs_server_info())["max_inline_bytes"] == 12345


def test_mutagenesis_drops_inline_arrays_kept_in_files(_registry, monkeypatch):
    seqs = _seqs(20)
    full = _run(_registry.tfbs_mutagenesis(model_id="bounded_model", sequences=seqs))
    assert "scores" in full
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "2000")
    r = _run(_registry.tfbs_mutagenesis(model_id="bounded_model", sequences=seqs))
    assert "scores" not in r and r["inline"] is False and r["omitted_inline"] == ["scores"]
    saved = torch.load(r["files"]["scores"], weights_only=True)
    assert list(saved.shape) == r["shape"] == full["shape"]
    assert torch.allclose(saved, torch.tensor(full["scores"], dtype=saved.dtype))


def test_marginalize_lists_go_to_a_file(_registry, monkeypatch):
    seqs = _seqs(100, seed=5)
    full = _run(_registry.tfbs_marginalize(model_id="bounded_model", sequences=seqs, motif="ACGT"))
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "1000")
    r = _run(_registry.tfbs_marginalize(model_id="bounded_model", sequences=seqs, motif="ACGT"))
    header, rows = _table(r["results_file"])
    assert header == ["predictions_before", "predictions_after", "delta_predictions"]
    assert len(rows) == 100
    assert [float(row[2]) for row in rows] == full["delta_predictions"]
    assert r["mean_delta"] == full["mean_delta"]


def test_classify_metrics_curves_go_to_files(_registry, monkeypatch):
    rnd = random.Random(7)
    pos = [rnd.gauss(1.0, 1.0) for _ in range(2000)]
    neg = [rnd.gauss(0.0, 1.0) for _ in range(2000)]
    kw = dict(positive_scores=pos, negative_scores=neg, subsample_negatives=False)
    full = _run(_registry.tfbs_classify_metrics(**kw))
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "2000")
    r = _run(_registry.tfbs_classify_metrics(**kw))
    assert r["roc_auc"] == full["roc_auc"] and r["pr_auc"] == full["pr_auc"]
    roc_h, roc = _table(r["results_files"]["roc_curve"])
    pr_h, pr = _table(r["results_files"]["pr_curve"])
    assert roc_h == ["fpr", "tpr"] and len(roc) == len(full["fpr"])
    assert pr_h == ["precision", "recall"] and len(pr) == len(full["precision"])
    assert [float(x[1]) for x in roc] == full["tpr"]
    assert "fpr" not in r and r["inline"] is False


def test_analyze_bounded(_registry, monkeypatch, tmp_path):
    pytest.importorskip("captum")
    seqs = _seqs(30, seed=9)
    full = _run(_registry.tfbs_analyze(model_id="bounded_model", sequences=seqs, method="input_x_gradient"))
    if "error" in full:
        pytest.skip(full["error"])
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "1500")
    r = _run(_registry.tfbs_analyze(model_id="bounded_model", sequences=seqs, method="input_x_gradient"))
    assert "attributions" not in r and r["omitted_inline"] == ["attributions"]
    saved = torch.load(r["files"]["attributions"], weights_only=True)
    assert list(saved.shape) == r["shape"]
    pt = tmp_path / "x.pt"
    idx = torch.randint(0, 4, (200, 10), generator=torch.Generator().manual_seed(1))
    torch.save(torch.nn.functional.one_hot(idx, 4).float().permute(0, 2, 1).contiguous(), pt)
    r2 = _run(_registry.tfbs_analyze(model_id="bounded_model", sequences_path=str(pt), method="input_x_gradient"))
    header, rows = _table(r2["results_file"])
    assert header == ["attribution_sums"] and len(rows) == 200


def test_results_files_are_named_after_their_content(_registry, monkeypatch):
    seqs = _seqs(40)
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "0")
    first = _run(_registry.tfbs_predict(model_id="bounded_model", sequences=seqs))["results_file"]
    other = FakeModel()
    with torch.no_grad():
        other.linear.weight.add_(1.0)
    _registry._model_registry["bounded_model"]["model"] = other.eval()  # same model_id, other weights
    second = _run(_registry.tfbs_predict(model_id="bounded_model", sequences=seqs))["results_file"]
    assert first["path"] != second["path"]
    assert first["sha256"] == _sha(first["path"]) and second["sha256"] == _sha(second["path"])
    assert first["sha256"][:12] in first["path"]


def test_mutagenesis_fasta_ids_go_to_a_file(_registry, monkeypatch, tmp_path):
    fa = tmp_path / "many.fa"
    seqs = _seqs(300, seed=11)
    fa.write_text("".join(f">chr3:{100000000 + 200 * i}-{100000200 + 200 * i}\n{s}\n" for i, s in enumerate(seqs)))
    full = _run(_registry.tfbs_mutagenesis(model_id="bounded_model", sequences_path=str(fa)))
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "2000")
    r = _run(_registry.tfbs_mutagenesis(model_id="bounded_model", sequences_path=str(fa)))
    assert "ids" not in r and r["inline"] is False and "omitted_inline" not in r
    header, rows = _table(r["results_file"])
    assert header == ["ids"] and [row[0] for row in rows] == full["ids"]
    assert len(json.dumps(r)) < 4000  # a short reply, not one entry per record


def test_limit_is_capped_at_one_mebibyte(_registry, monkeypatch):
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", str(10 ** 9))
    assert _registry._max_inline_bytes() == 1024 * 1024


def test_note_names_reader_settings_that_round_trip(_registry, monkeypatch, tmp_path):
    pd = pytest.importorskip("pandas")
    fa = tmp_path / "odd_ids.fa"
    names = ["NA", "null", "nan", "None", "N/A", "00012"] + [f"p{i}" for i in range(60)]
    seqs = _seqs(len(names), seed=13)
    fa.write_text("".join(f">{n}\n{s}\n" for n, s in zip(names, seqs)))
    full = _run(_registry.tfbs_predict(model_id="bounded_model", sequences_path=str(fa)))
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "0")
    r = _run(_registry.tfbs_predict(model_id="bounded_model", sequences_path=str(fa)))
    assert 'float_precision="round_trip"' in r["note"]
    df = pd.read_csv(r["results_file"]["path"], float_precision="round_trip", keep_default_na=False,
                     na_values={"scores": ["nan"]}, dtype={"ids": str})
    assert list(df["ids"]) == full["ids"] == names
    assert list(df["scores"]) == full["scores"]


def test_unwritable_results_directory_returns_an_error(_registry, monkeypatch, tmp_path):
    blocker = tmp_path / "taken"
    blocker.write_text("x")
    monkeypatch.setattr(_registry, "_results_root", lambda: blocker / "results")
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "0")
    r = _run(_registry.tfbs_predict(model_id="bounded_model", sequences=_seqs(3)))
    assert "error" in r and "fix" in r and "TFBS_PROJECT_ROOT" in r["fix"]


def test_a_reply_still_too_large_goes_to_a_reply_file(_registry, monkeypatch, tmp_path):
    monkeypatch.setenv("TFBS_MAX_INLINE_BYTES", "1000")
    result = {"num_sequences": 2, "scores": [0.5, 0.25], "warnings": ["x" * 100_000]}
    r = json.loads(_registry._bounded_reply(result, table_keys=["scores"], table_dir=lambda: tmp_path,
                                            stem="predictions"))
    assert r["inline"] is False and r["num_sequences"] == 2 and "warnings" not in r
    desc = r["reply_file"]
    saved = json.loads(open(desc["path"], encoding="utf-8").read())
    assert saved["warnings"] == ["x" * 100_000] and saved["results_file"]["rows"] == 2
    assert desc["sha256"] == _sha(desc["path"])


def test_tools_that_write_results_files_are_not_annotated_read_only(_registry):
    tools = {t.name: t for t in asyncio.run(_registry.mcp.list_tools())}
    for name in ("tfbs_predict", "tfbs_marginalize", "tfbs_classify_metrics"):
        assert tools[name].annotations.readOnlyHint is False, name


def test_hard_cap_saves_any_oversized_reply_and_keeps_the_message_form(_registry, monkeypatch):
    seqs = _seqs(50)
    direct = asyncio.run(_registry.tfbs_predict(model_id="bounded_model", sequences=seqs))
    content, structured = asyncio.run(_registry.mcp.call_tool(
        "tfbs_predict", {"model_id": "bounded_model", "sequences": seqs}))
    assert content[0].text == direct and structured == {"result": direct}  # unchanged below the cap
    monkeypatch.setattr(_registry, "REPLY_HARD_CAP_BYTES", 100)
    content, structured = asyncio.run(_registry.mcp.call_tool(
        "tfbs_predict", {"model_id": "bounded_model", "sequences": seqs}))
    r = json.loads(content[0].text)
    assert structured == {"result": content[0].text}
    assert r["inline"] is False and r["reply_file"]["bytes"] == len(direct)
    assert open(r["reply_file"]["path"], encoding="utf-8").read() == direct
