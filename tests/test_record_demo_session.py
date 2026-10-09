"""scripts/record_demo_session.py: --set substitution and $ref resolution."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _recorder():
    spec = importlib.util.spec_from_file_location(
        "record_demo_session", REPO / "scripts" / "record_demo_session.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_set_replaces_every_string_before_refs_resolve():
    rec = _recorder()
    plan = [
        {"role": "user", "text": "Extract the peaks in @DEMO/demo_peaks.bed"},
        {"role": "tool", "tool": "tfbs_extract_loci",
         "arguments": {"bed_path": "@DEMO/demo_peaks.bed", "in_window": 24,
                       "extra": ["@DEMO_DIR/x", "@OUT"], "ref": {"$ref": "0.files.@KEY"}}},
    ]
    out = rec._substitute(plan, {"DEMO": "/d", "DEMO_DIR": "/dd", "OUT": "/o", "KEY": "sequences"})
    assert out[0]["text"] == "Extract the peaks in /d/demo_peaks.bed"
    args = out[1]["arguments"]
    assert args["bed_path"] == "/d/demo_peaks.bed" and args["in_window"] == 24
    assert args["extra"] == ["/dd/x", "/o"]
    assert args["ref"] == {"$ref": "0.files.sequences"}
    assert plan[1]["arguments"]["bed_path"] == "@DEMO/demo_peaks.bed"

    results = {0: {"files": {"sequences": "/d/loci/sequences.pt"}, "rows": [{"n": 3}]}}
    resolved = rec._resolve_refs(args, results)
    assert resolved["ref"] == "/d/loci/sequences.pt"
    assert rec._resolve_refs({"$ref": "0.rows.0.n"}, results) == 3
    assert rec._resolve_refs([{"$ref": "0"}], results) == [results[0]]


def test_set_and_env_need_key_value_pairs():
    rec = _recorder()
    assert rec._pairs(["A=1", "B=x=y"], "--set") == {"A": "1", "B": "x=y"}
    try:
        rec._pairs(["novalue"], "--set")
    except SystemExit as exc:
        assert "--set" in str(exc)
    else:
        raise AssertionError("a pair without '=' must be rejected")
