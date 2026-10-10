"""scripts/session_from_call_log.py: call log (+ Claude Code transcript) to a session log."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _converter():
    spec = importlib.util.spec_from_file_location(
        "session_from_call_log", REPO / "scripts" / "session_from_call_log.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, records: list) -> Path:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
    return path


EXTRACT_RESULT = {"num_peaks_extracted": 20, "shape": [20, 4, 200],
                  "files": {"sequences": "/w/loci/sequences.pt"}}
METRICS_RESULT = {"roc_auc": 0.9, "pr_auc": 0.8}
FIMO_HEAD = '{"total_hits": 3, "backend": "meme", "hits": [{"start": 1'
FIMO_FULL = {"total_hits": 3, "backend": "meme", "hits": [{"start": 1}, {"start": 5}, {"start": 9}]}


def _call_log(tmp_path: Path) -> Path:
    return _write_jsonl(tmp_path / "calls.jsonl", [
        {"timestamp": "2026-10-09T10:00:05.000+00:00", "tool": "tfbs_extract_loci",
         "arguments": {"bed_path": "/d/peaks.bed"}, "seconds": 1.5, "is_error": False,
         "result": EXTRACT_RESULT},
        {"timestamp": "2026-10-09T10:00:20.000+00:00", "tool": "tfbs_fimo",
         "arguments": {"sequences": ["ACGT"]}, "seconds": 2.0, "is_error": False,
         "result": {"truncated": True, "bytes": 2_000_000, "head": FIMO_HEAD}},
        {"timestamp": "2026-10-09T10:00:40.000+00:00", "tool": "tfbs_classify_metrics",
         "arguments": {"positive_scores": [0.9], "negative_scores": [0.1]}, "seconds": 0.01,
         "is_error": False, "result": METRICS_RESULT},
        {"timestamp": "2026-10-09T10:00:50.000+00:00", "tool": "tfbs_predict",
         "arguments": {"model_id": "m"}, "seconds": 0.2, "is_error": True,
         "result": {"error": "Model 'm' not loaded.", "fix": "Call tfbs_load_model."}},
    ])


def _entry(kind, ts, content, **extra):
    return {"type": kind, "timestamp": ts, "isSidechain": False, "version": "2.1.0",
            "message": {"role": kind, "content": content}, **extra}


def _transcript(tmp_path: Path) -> Path:
    fimo_text = json.dumps(FIMO_FULL)
    return _write_jsonl(tmp_path / "session.jsonl", [
        {"type": "custom-title", "customTitle": "demo"},
        _entry("user", "2026-10-09T10:00:00.000Z", "Extract the peaks in /d/peaks.bed"),
        _entry("user", "2026-10-09T10:00:00.100Z", "<local-command-caveat>x</local-command-caveat>",
               isMeta=True),
        _entry("assistant", "2026-10-09T10:00:02.000Z",
               [{"type": "thinking", "thinking": "hidden"}]),
        _entry("assistant", "2026-10-09T10:00:03.000Z",
               [{"type": "text", "text": "I will read the BED file first."},
                {"type": "tool_use", "id": "t1", "name": "Read",
                 "input": {"file_path": "/d/peaks.bed"}}]),
        _entry("user", "2026-10-09T10:00:03.250Z",
               [{"type": "tool_result", "tool_use_id": "t1", "content": "chr1\t10\t20\n"}]),
        _entry("assistant", "2026-10-09T10:00:04.900Z",
               [{"type": "tool_use", "id": "t2", "name": "mcp__tfbs__tfbs_extract_loci",
                 "input": {"bed_path": "/d/peaks.bed"}}]),
        _entry("user", "2026-10-09T10:00:06.600Z",
               [{"type": "tool_result", "tool_use_id": "t2",
                 "content": [{"type": "text", "text": json.dumps(EXTRACT_RESULT)}]}]),
        # A subagent's lines: ignored, so its call stays a call-log-only step.
        {**_entry("assistant", "2026-10-09T10:00:39.900Z",
                  [{"type": "tool_use", "id": "s1", "name": "mcp__tfbs__tfbs_classify_metrics",
                    "input": {}}]), "isSidechain": True},
        {**_entry("user", "2026-10-09T10:00:41.000Z", "subagent prompt"), "isSidechain": True},
        _entry("assistant", "2026-10-09T10:00:19.900Z",
               [{"type": "tool_use", "id": "t3", "name": "mcp__tfbs__tfbs_fimo",
                 "input": {"sequences": ["ACGT"]}}]),
        _entry("user", "2026-10-09T10:00:22.000Z",
               [{"type": "tool_result", "tool_use_id": "t3",
                 "content": [{"type": "text", "text": fimo_text}]}]),
        # A tfbs call the call log does not have: result and seconds from the transcript.
        _entry("assistant", "2026-10-09T10:00:30.000Z",
               [{"type": "tool_use", "id": "t4", "name": "mcp__tfbs__tfbs_server_info",
                 "input": {}}]),
        _entry("user", "2026-10-09T10:00:30.500Z",
               [{"type": "tool_result", "tool_use_id": "t4",
                 "content": [{"type": "text", "text": '{"default_device": "cpu"}'}]}]),
        _entry("assistant", "2026-10-09T10:01:00.000Z",
               [{"type": "text", "text": "Twenty windows were extracted."}]),
        "{not json",
    ])


def test_without_a_transcript_the_steps_are_the_logged_tool_calls(tmp_path):
    conv = _converter()
    log = conv.convert(_call_log(tmp_path))

    assert [s["role"] for s in log["steps"]] == ["tool"] * 4
    assert [s["tool"] for s in log["steps"]] == [
        "tfbs_extract_loci", "tfbs_fimo", "tfbs_classify_metrics", "tfbs_predict"]
    first = log["steps"][0]
    assert first["arguments"] == {"bed_path": "/d/peaks.bed"}
    assert first["result"] == EXTRACT_RESULT and "result_text" not in first
    assert first["seconds"] == 1.5 and first["is_error"] is False
    assert first["phase"] == "Data prep"
    assert log["steps"][2]["phase"] == "Validation"
    assert log["steps"][3]["phase"] == "Analysis" and log["steps"][3]["is_error"] is True
    # A truncated entry without a transcript: no result, the head as text.
    fimo = log["steps"][1]
    assert fimo["result"] is None and fimo["result_text"] == FIMO_HEAD
    assert fimo["result_truncated"] is True and fimo["result_bytes"] == 2_000_000
    assert log["server_call_log"] == conv.read_jsonl(tmp_path / "calls.jsonl")[0]
    assert log["recorded_at"] == "2026-10-09T10:00:05+00:00"
    assert log["server"] == "tfbs-mcp"
    assert "tfbs_mcp_commit" not in log and "Claude Code" not in log["client"]
    # Not measured for a real client's session: written as explicit nulls.
    assert log["seconds_to_initialize"] is None and log["tools_listed"] is None
    assert "seconds_to_initialize" in log and "tools_listed" in log
    assert not any("shown_args" in s for s in log["steps"])


def test_transcript_interleaves_messages_and_matches_calls(tmp_path):
    conv = _converter()
    log = conv.convert(_call_log(tmp_path), _transcript(tmp_path), commit="abc1234")
    steps = log["steps"]

    assert [(s["role"], s.get("tool")) for s in steps] == [
        ("user", None),
        ("llm", None),
        ("tool", "Read"),
        ("tool", "tfbs_extract_loci"),
        ("tool", "tfbs_fimo"),
        ("tool", "tfbs_server_info"),
        ("tool", "tfbs_classify_metrics"),
        ("tool", "tfbs_predict"),
        ("llm", None),
    ]
    assert steps[0]["text"] == "Extract the peaks in /d/peaks.bed"
    assert steps[1]["text"] == "I will read the BED file first."
    assert steps[-1]["text"] == "Twenty windows were extracted."

    extract = steps[3]
    assert extract["result"] == EXTRACT_RESULT and extract["seconds"] == 1.5
    assert "kind" not in extract
    # The truncated call-log result is replaced by the transcript's full one.
    fimo = steps[4]
    assert fimo["result"] == FIMO_FULL and "result_truncated" not in fimo
    assert fimo["seconds"] == 2.0
    # Missing from the call log: result and seconds come from the transcript.
    info = steps[5]
    assert info["result"] == {"default_device": "cpu"} and info["seconds"] == 0.5
    assert info["arguments"] == {}

    c = log["converted_from"]
    assert (c["matched"], c["transcript_only"], c["call_log_only"], c["client"]) == (2, 1, 2, 1)
    assert c["unreadable_transcript_lines"] == 1
    assert log["tfbs_mcp_commit"] == "abc1234"
    assert log["client"] == "Claude Code 2.1.0" and log["client_server_name"] == "tfbs"
    assert "server_name" not in log
    assert log["recorded_at"] == "2026-10-09T10:00:00+00:00"


def test_client_tools_are_kept_with_kind_client(tmp_path):
    conv = _converter()
    steps = conv.convert(_call_log(tmp_path), _transcript(tmp_path))["steps"]
    read = next(s for s in steps if s.get("tool") == "Read")
    assert read["kind"] == "client" and read["phase"] == "Client"
    assert read["arguments"] == {"file_path": "/d/peaks.bed"}
    assert read["result"] is None and read["result_text"] == "chr1\t10\t20\n"
    assert read["seconds"] == 0.25 and read["is_error"] is False


def test_sidechain_and_meta_lines_are_ignored(tmp_path):
    conv = _converter()
    steps = conv.convert(_call_log(tmp_path), _transcript(tmp_path))["steps"]
    texts = [s["text"] for s in steps if s["role"] in ("user", "llm")]
    assert "subagent prompt" not in texts
    assert not any("local-command-caveat" in t for t in texts)
    # The subagent's classify_metrics call was not matched: it is a call-log step.
    metrics = next(s for s in steps if s.get("tool") == "tfbs_classify_metrics")
    assert metrics["timestamp"] == "2026-10-09T10:00:40.000+00:00"
    assert metrics["result"] == METRICS_RESULT


def test_phases_file_overrides_the_default_map(tmp_path):
    conv = _converter()
    phases = tmp_path / "phases.json"
    phases.write_text(json.dumps({"tfbs_extract_loci": "Setup", "tfbs_fi*": "Motifs",
                                  "Read": "Reading"}), encoding="utf-8")
    rc = conv.main(["--call-log", str(_call_log(tmp_path)), "--transcript",
                    str(_transcript(tmp_path)), "--phases", str(phases),
                    "--out", str(tmp_path / "out" / "session_log.json")])
    assert rc == 0
    steps = json.loads((tmp_path / "out" / "session_log.json").read_text(encoding="utf-8"))["steps"]
    phase = {s["tool"]: s["phase"] for s in steps if s["role"] == "tool"}
    assert phase == {"Read": "Reading", "tfbs_extract_loci": "Setup", "tfbs_fimo": "Motifs",
                     "tfbs_server_info": "Analysis", "tfbs_classify_metrics": "Validation",
                     "tfbs_predict": "Analysis"}


def test_default_phase_map():
    conv = _converter()
    expected = {
        "tfbs_extract_loci": "Data prep", "tfbs_prepare_dataset": "Data prep",
        "tfbs_preprocess": "Data prep", "tfbs_config": "Training", "tfbs_train": "Training",
        "tfbs_sweep": "Training", "tfbs_validate": "Training", "slurm_status": "Training",
        "tfbs_remap_validation": "Validation", "tfbs_chipseq_benchmark": "Validation",
        "tfbs_disease_variant_validation": "Validation", "tfbs_classify_metrics": "Validation",
        "tfbs_fetch_structure": "Structure", "tfbs_binding_scan": "Structure",
        "tfbs_predict": "Analysis", "tfbs_mutagenesis": "Analysis",
    }
    assert {t: conv.phase_of(t, None) for t in expected} == expected
    assert conv.phase_of("Bash", None, client=True) == "Client"


def test_cli_writes_lf_utf8_and_a_missing_transcript_is_an_error(tmp_path, capsys):
    conv = _converter()
    calls = _call_log(tmp_path)
    out = tmp_path / "session_log.json"
    assert conv.main(["--call-log", str(calls), "--out", str(out), "--client", "Claude Code 9.9",
                      "--commit", "deadbee"]) == 0
    raw = out.read_bytes()
    assert b"\r\n" not in raw
    log = json.loads(raw.decode("utf-8"))
    assert log["client"] == "Claude Code 9.9" and log["tfbs_mcp_commit"] == "deadbee"
    assert capsys.readouterr().out == ""

    with pytest.raises(SystemExit, match="--transcript: no such file"):
        conv.main(["--call-log", str(calls), "--out", str(out),
                   "--transcript", str(tmp_path / "missing.jsonl")])
    with pytest.raises(SystemExit, match="--call-log: no such file"):
        conv.main(["--call-log", str(tmp_path / "none.jsonl"), "--out", str(out)])


def test_old_call_log_without_results_takes_them_from_the_transcript(tmp_path):
    conv = _converter()
    calls = _write_jsonl(tmp_path / "old.jsonl", [
        {"timestamp": "2026-10-09T10:00:05.000+00:00", "tool": "tfbs_extract_loci",
         "arguments": {"bed_path": "/d/peaks.bed"}},
    ])
    steps = conv.convert(calls, _transcript(tmp_path))["steps"]
    extract = next(s for s in steps if s.get("tool") == "tfbs_extract_loci")
    assert extract["result"] == EXTRACT_RESULT
    assert extract["seconds"] == 1.7 and extract["is_error"] is False


def test_output_renders_with_the_session_renderer_contract(tmp_path):
    """The fields TF-NN/docs/render_session.py reads are present on every step."""
    conv = _converter()
    log = conv.convert(_call_log(tmp_path), _transcript(tmp_path))
    for step in log["steps"]:
        assert step["role"] in ("user", "llm", "tool")
        if step["role"] == "tool":
            assert isinstance(step["tool"], str) and "phase" in step
            assert "result" in step and isinstance(step.get("seconds", 0), (int, float))
            assert isinstance(step["is_error"], bool)
            if step["result"] is None:
                assert isinstance(step["result_text"], str)
        else:
            assert isinstance(step["text"], str) and step["text"]
    total = sum(s.get("seconds", 0) for s in log["steps"] if s["role"] == "tool")
    assert total > 0


# ---------------------------------------------------------------------------
# Matching transcript calls to call-log entries
# ---------------------------------------------------------------------------

def _log_entry(ts, tool, arguments, result=None, seconds=0.5):
    return {"timestamp": ts, "tool": tool, "arguments": arguments, "seconds": seconds,
            "is_error": False, "result": result if result is not None else {"ok": arguments}}


def _use(uid, ts, tool, arguments):
    return _entry("assistant", ts, [{"type": "tool_use", "id": uid, "name": f"mcp__tfbs__{tool}",
                                     "input": arguments}])


def _result(uid, ts, text):
    return _entry("user", ts, [{"type": "tool_result", "tool_use_id": uid,
                                "content": [{"type": "text", "text": text}]}])


REF = {"model_id": "flank", "sequences": ["ACGTACGT"]}
ALT = {"model_id": "flank", "sequences": ["ACGGACGT"]}


def _two_predicts(tmp_path, calls):
    """A transcript that calls tfbs_predict twice (ref, then alt)."""
    log = _write_jsonl(tmp_path / "calls.jsonl", calls)
    transcript = _write_jsonl(tmp_path / "session.jsonl", [
        _entry("user", "2026-10-09T10:00:00.000Z", "Score ref and alt."),
        _use("p1", "2026-10-09T10:00:10.000Z", "tfbs_predict", REF),
        _result("p1", "2026-10-09T10:00:11.000Z", json.dumps({"from": "transcript", "which": "ref"})),
        _use("p2", "2026-10-09T10:00:20.000Z", "tfbs_predict", ALT),
        _result("p2", "2026-10-09T10:00:21.000Z", json.dumps({"from": "transcript", "which": "alt"})),
        _entry("assistant", "2026-10-09T10:00:30.000Z", [{"type": "text", "text": "Done."}]),
    ])
    return log, transcript


def _tool_steps(log):
    return [s for s in log["steps"] if s["role"] == "tool"]


def test_a_call_missing_from_the_call_log_shifts_nothing(tmp_path):
    conv = _converter()
    log, transcript = _two_predicts(tmp_path, [
        _log_entry("2026-10-09T10:00:20.100+00:00", "tfbs_predict", ALT, {"which": "alt"}),
    ])
    out = conv.convert(log, transcript)
    ref, alt = _tool_steps(out)
    assert ref["arguments"] == REF and ref["result"] == {"from": "transcript", "which": "ref"}
    assert alt["arguments"] == ALT and alt["result"] == {"which": "alt"}
    c = out["converted_from"]
    assert (c["matched"], c["transcript_only"], c["call_log_only"]) == (1, 1, 0)


def test_a_call_missing_from_the_transcript_shifts_nothing(tmp_path):
    conv = _converter()
    log = _write_jsonl(tmp_path / "calls.jsonl", [
        _log_entry("2026-10-09T10:00:10.100+00:00", "tfbs_predict", REF, {"which": "ref"}),
        _log_entry("2026-10-09T10:00:20.100+00:00", "tfbs_predict", ALT, {"which": "alt"}),
    ])
    transcript = _write_jsonl(tmp_path / "session.jsonl", [
        _entry("user", "2026-10-09T10:00:00.000Z", "Score alt."),
        _use("p2", "2026-10-09T10:00:20.000Z", "tfbs_predict", ALT),
        _result("p2", "2026-10-09T10:00:21.000Z", "{}"),
    ])
    out = conv.convert(log, transcript)
    ref, alt = _tool_steps(out)
    assert ref["arguments"] == REF and ref["result"] == {"which": "ref"}
    assert alt["arguments"] == ALT and alt["result"] == {"which": "alt"}
    c = out["converted_from"]
    assert (c["matched"], c["transcript_only"], c["call_log_only"]) == (1, 0, 1)


def test_calls_match_on_equal_arguments_not_on_log_order(tmp_path):
    """Parallel calls finish out of order: the log holds alt before ref."""
    conv = _converter()
    log, transcript = _two_predicts(tmp_path, [
        _log_entry("2026-10-09T10:00:20.100+00:00", "tfbs_predict", ALT, {"which": "alt"}),
        _log_entry("2026-10-09T10:00:10.100+00:00", "tfbs_predict", REF, {"which": "ref"}),
    ])
    out = conv.convert(log, transcript)
    ref, alt = _tool_steps(out)
    assert (ref["arguments"], ref["result"]) == (REF, {"which": "ref"})
    assert (alt["arguments"], alt["result"]) == (ALT, {"which": "alt"})
    assert ref["timestamp"] < alt["timestamp"]
    assert out["converted_from"]["matched_by_order"] == 0


def test_identical_calls_take_the_closest_entry_at_or_after_their_own_time(tmp_path):
    conv = _converter()
    log = _write_jsonl(tmp_path / "calls.jsonl", [
        _log_entry("2026-10-09T10:00:20.300+00:00", "tfbs_predict", REF, {"n": 2}),
        _log_entry("2026-10-09T10:00:10.300+00:00", "tfbs_predict", REF, {"n": 1}),
    ])
    transcript = _write_jsonl(tmp_path / "session.jsonl", [
        _use("a", "2026-10-09T10:00:10.000Z", "tfbs_predict", REF),
        _use("b", "2026-10-09T10:00:20.000Z", "tfbs_predict", REF),
    ])
    first, second = _tool_steps(conv.convert(log, transcript))
    assert first["result"] == {"n": 1} and second["result"] == {"n": 2}


def test_unequal_arguments_fall_back_to_name_and_order(tmp_path):
    conv = _converter()
    log = _write_jsonl(tmp_path / "calls.jsonl", [
        _log_entry("2026-10-09T10:00:10.100+00:00", "tfbs_predict", {"model_id": "server-side"}),
    ])
    transcript = _write_jsonl(tmp_path / "session.jsonl", [
        _use("p1", "2026-10-09T10:00:10.000Z", "tfbs_predict", {"model_id": "client-side"}),
        _result("p1", "2026-10-09T10:00:11.000Z", "{}"),
    ])
    out = conv.convert(log, transcript)
    (step,) = _tool_steps(out)
    assert step["arguments"] == {"model_id": "server-side"}
    assert out["converted_from"]["matched"] == 1
    assert out["converted_from"]["matched_by_order"] == 1


def test_entries_outside_the_transcript_time_span_are_counted_not_kept(tmp_path):
    conv = _converter()
    log, transcript = _two_predicts(tmp_path, [
        _log_entry("2026-10-08T09:00:00.000+00:00", "tfbs_predict", REF, {"stale": True}),
        _log_entry("2026-10-09T10:00:10.100+00:00", "tfbs_predict", REF, {"which": "ref"}),
        _log_entry("2026-10-09T10:00:20.100+00:00", "tfbs_predict", ALT, {"which": "alt"}),
        _log_entry("2026-10-09T18:00:00.000+00:00", "tfbs_list_models", {}),
    ])
    out = conv.convert(log, transcript)
    assert [s["result"] for s in _tool_steps(out)] == [{"which": "ref"}, {"which": "alt"}]
    c = out["converted_from"]
    assert (c["matched"], c["call_log_only"], c["calls_outside_window"]) == (2, 0, 2)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def test_an_output_too_large_stub_never_replaces_a_truncated_result(tmp_path):
    conv = _converter()
    log = _write_jsonl(tmp_path / "calls.jsonl", [
        {"timestamp": "2026-10-09T10:00:10.100+00:00", "tool": "tfbs_fimo", "arguments": {},
         "seconds": 3.0, "is_error": False,
         "result": {"truncated": True, "bytes": 2_000_000, "head": FIMO_HEAD}},
    ])
    transcript = _write_jsonl(tmp_path / "session.jsonl", [
        _use("f1", "2026-10-09T10:00:10.000Z", "tfbs_fimo", {}),
        _result("f1", "2026-10-09T10:00:13.000Z",
                "Output too large (1953.1KB). Full output saved to: C:/x/tool-results/f1.txt"),
    ])
    (step,) = _tool_steps(conv.convert(log, transcript))
    assert step["result"] is None
    assert step["result_text"] == FIMO_HEAD
    assert step["result_truncated"] is True and step["result_bytes"] == 2_000_000


PERSISTED_NOTE = ("<persisted-output>\nOutput too large (1953.1KB). Full output saved to: "
                  "C:/Users/someone/.claude/projects/x/tool-results/f1.txt\n</persisted-output>")


@pytest.mark.parametrize("note", [
    PERSISTED_NOTE,
    "Output too large (1953.1KB). Full output saved to: C:/Users/someone/tool-results/f1.txt",
])
def test_an_output_too_large_note_is_never_the_result_of_a_transcript_only_call(tmp_path, note):
    conv = _converter()
    log = _write_jsonl(tmp_path / "calls.jsonl", [])
    transcript = _write_jsonl(tmp_path / "session.jsonl", [
        _use("f1", "2026-10-09T10:00:10.000Z", "tfbs_fimo", {}),
        _result("f1", "2026-10-09T10:00:13.000Z", note),
    ])
    out = conv.convert(log, transcript)
    (step,) = _tool_steps(out)
    assert out["converted_from"]["transcript_only"] == 1
    assert step["result"] is None and step["result_text"] == ""
    assert step["result_truncated"] is True and "result_bytes" not in step
    assert step["is_error"] is False
    assert "C:/Users" not in json.dumps(out)


def test_an_output_too_large_note_is_never_the_result_of_an_entry_without_one(tmp_path):
    conv = _converter()
    log = _write_jsonl(tmp_path / "calls.jsonl", [
        {"timestamp": "2026-10-09T10:00:10.100+00:00", "tool": "tfbs_fimo", "arguments": {},
         "seconds": 3.0, "is_error": False},
    ])
    transcript = _write_jsonl(tmp_path / "session.jsonl", [
        _use("f1", "2026-10-09T10:00:10.000Z", "tfbs_fimo", {}),
        _result("f1", "2026-10-09T10:00:13.000Z", PERSISTED_NOTE),
    ])
    out = conv.convert(log, transcript)
    (step,) = _tool_steps(out)
    assert step["result"] is None and step["result_text"] == ""
    assert step["result_truncated"] is True
    assert "C:/Users" not in json.dumps(out)


def test_an_output_too_large_note_of_a_client_tool_is_not_kept(tmp_path):
    conv = _converter()
    transcript = _write_jsonl(tmp_path / "session.jsonl", [
        _entry("assistant", "2026-10-09T10:00:10.000Z",
               [{"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "cat big"}}]),
        _entry("user", "2026-10-09T10:00:11.000Z",
               [{"type": "tool_result", "tool_use_id": "b1", "content": PERSISTED_NOTE}]),
    ])
    out = conv.convert(_write_jsonl(tmp_path / "calls.jsonl", []), transcript)
    (step,) = _tool_steps(out)
    assert step["kind"] == "client"
    assert step["result"] is None and step["result_text"] == ""
    assert step["result_truncated"] is True
    assert "C:/Users" not in json.dumps(out)


def test_a_client_timeout_on_a_logged_success_is_recorded(tmp_path):
    conv = _converter()
    log = _write_jsonl(tmp_path / "calls.jsonl", [
        _log_entry("2026-10-09T10:00:10.100+00:00", "tfbs_train", {"cfg": "c.yaml"},
                   {"run_dir": "/w/run1"}, seconds=400.0),
    ])
    transcript = _write_jsonl(tmp_path / "session.jsonl", [
        _use("t1", "2026-10-09T10:00:10.000Z", "tfbs_train", {"cfg": "c.yaml"}),
        _entry("user", "2026-10-09T10:05:10.000Z",
               [{"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                 "content": [{"type": "text", "text": "MCP error -32001: Request timed out"}]}]),
    ])
    (step,) = _tool_steps(conv.convert(log, transcript))
    assert step["result"] == {"run_dir": "/w/run1"}
    assert step["is_error"] is False
    assert step["client_saw_error"] is True
    assert step["client_error_text"] == "MCP error -32001: Request timed out"


def test_no_client_error_flag_when_the_client_saw_success(tmp_path):
    conv = _converter()
    log = _write_jsonl(tmp_path / "calls.jsonl", [
        _log_entry("2026-10-09T10:00:10.100+00:00", "tfbs_list_models", {}, {"models": []}),
    ])
    transcript = _write_jsonl(tmp_path / "session.jsonl", [
        _use("t1", "2026-10-09T10:00:10.000Z", "tfbs_list_models", {}),
        _result("t1", "2026-10-09T10:00:10.500Z", '{"models": []}'),
    ])
    (step,) = _tool_steps(conv.convert(log, transcript))
    assert "client_saw_error" not in step and "client_error_text" not in step


def test_a_tfbs_result_that_is_not_json_is_kept_as_text(tmp_path):
    conv = _converter()
    log = _write_jsonl(tmp_path / "calls.jsonl", [
        {"timestamp": "2026-10-09T10:00:10.100+00:00", "tool": "tfbs_server_info",
         "arguments": {}, "seconds": 0.1, "is_error": False,
         "result": {"text": "not json at all"}},
    ])
    transcript = _write_jsonl(tmp_path / "session.jsonl", [
        _use("s1", "2026-10-09T10:00:10.000Z", "tfbs_server_info", {}),
        _result("s1", "2026-10-09T10:00:10.200Z", "not json at all"),
        _use("s2", "2026-10-09T10:00:20.000Z", "tfbs_list_models", {}),
        _result("s2", "2026-10-09T10:00:20.200Z", "Error: plain text from the client"),
    ])
    logged, transcript_only = _tool_steps(conv.convert(log, transcript))
    assert logged["result"] is None and logged["result_text"] == "not json at all"
    assert logged["is_error"] is False and "result_truncated" not in logged
    assert transcript_only["result"] is None
    assert transcript_only["result_text"] == "Error: plain text from the client"


# ---------------------------------------------------------------------------
# Lines the client injects itself
# ---------------------------------------------------------------------------

HARNESS_TEXTS = [
    "<command-name>/model</command-name>\n<command-message>model</command-message>",
    "<command-message>compact</command-message>",
    "<local-command-stdout>Set model to opus</local-command-stdout>",
    "<local-command-caveat>Caveat: generated by local commands</local-command-caveat>",
    "<task-notification>\n<task-id>b1</task-id>\n<status>completed</status>\n</task-notification>",
    "<system-reminder>Background job finished.</system-reminder>",
    "[Request interrupted by user]",
    "[Request interrupted by user for tool use]",
]


@pytest.mark.parametrize("text", HARNESS_TEXTS)
def test_harness_text_is_not_a_user_step(tmp_path, text):
    conv = _converter()
    transcript = _write_jsonl(tmp_path / "session.jsonl", [
        _entry("user", "2026-10-09T10:00:00.000Z", "Real prompt."),
        _entry("user", "2026-10-09T10:00:01.000Z", text),
        _entry("user", "2026-10-09T10:00:02.000Z", [{"type": "text", "text": text}]),
    ])
    out = conv.convert(_write_jsonl(tmp_path / "calls.jsonl", []), transcript)
    assert [s["text"] for s in out["steps"]] == ["Real prompt."]


@pytest.mark.parametrize("marker", [{"isMeta": True}, {"isCompactSummary": True},
                                    {"origin": {"kind": "task-notification"}}])
def test_meta_compaction_and_task_notification_lines_are_skipped(tmp_path, marker):
    conv = _converter()
    transcript = _write_jsonl(tmp_path / "session.jsonl", [
        _entry("user", "2026-10-09T10:00:00.000Z", "Real prompt."),
        _entry("user", "2026-10-09T10:00:01.000Z", "Looks like a user message.", **marker),
    ])
    out = conv.convert(_write_jsonl(tmp_path / "calls.jsonl", []), transcript)
    assert [s["text"] for s in out["steps"]] == ["Real prompt."]


def test_a_reminder_block_next_to_typed_text_keeps_the_typed_text(tmp_path):
    conv = _converter()
    transcript = _write_jsonl(tmp_path / "session.jsonl", [
        _entry("user", "2026-10-09T10:00:00.000Z", [
            {"type": "text", "text": "<system-reminder>Files changed.</system-reminder>"},
            {"type": "text", "text": "Now train it."}]),
    ])
    out = conv.convert(_write_jsonl(tmp_path / "calls.jsonl", []), transcript)
    assert [s["text"] for s in out["steps"]] == ["Now train it."]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_labels_by_tfbs_call_index_and_by_tool_name(tmp_path):
    conv = _converter()
    log, transcript = _two_predicts(tmp_path, [
        _log_entry("2026-10-09T10:00:10.100+00:00", "tfbs_predict", REF),
        _log_entry("2026-10-09T10:00:20.100+00:00", "tfbs_predict", ALT),
    ])
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"1": "ref", "tfbs_predict": "alt"}), encoding="utf-8")
    out_path = tmp_path / "out.json"
    assert conv.main(["--call-log", str(log), "--transcript", str(transcript),
                      "--labels", str(labels), "--out", str(out_path)]) == 0
    steps = _tool_steps(json.loads(out_path.read_text(encoding="utf-8")))
    assert [s["shown_args"] for s in steps] == ["ref", "alt"]


def test_client_steps_are_not_counted_by_label_indices(tmp_path):
    conv = _converter()
    log = conv.convert(_call_log(tmp_path), _transcript(tmp_path),
                       labels={"1": "BED", "2": "motifs"})
    labelled = {s["tool"]: s.get("shown_args") for s in log["steps"] if s["role"] == "tool"}
    assert labelled["Read"] is None
    assert labelled["tfbs_extract_loci"] == "BED" and labelled["tfbs_fimo"] == "motifs"
    assert labelled["tfbs_predict"] is None


def test_a_labels_file_that_is_not_a_string_map_is_an_error(tmp_path):
    conv = _converter()
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"1": 2}), encoding="utf-8")
    with pytest.raises(SystemExit, match="--labels"):
        conv.main(["--call-log", str(_call_log(tmp_path)), "--labels", str(labels),
                   "--out", str(tmp_path / "out.json")])
