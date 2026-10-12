"""scripts/record_demo_session.py: --set substitution, $ref resolution and failed steps."""
import asyncio
import contextlib
import importlib.util
import json
import types
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


def test_a_missing_ref_key_names_the_available_keys():
    rec = _recorder()
    results = {0: {"files": {"sequences": "/s.pt"}}, 1: [{"n": 1}]}
    for ref, missing, available in (("0.files.negatives", "negatives", ["sequences"]),
                                    ("2.files", "2", [0, 1]),
                                    ("1.5", "5", [0])):
        try:
            rec._resolve_refs({"$ref": ref}, results)
        except rec.RefError as exc:
            assert (exc.ref, exc.missing, exc.available) == (ref, missing, available)
        else:
            raise AssertionError(f"{ref} must not resolve")


class _FakeSession:
    """ClientSession stand-in: a scripted reply, exception or crash per tool name."""

    def __init__(self, replies):
        self.replies = replies

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        return types.SimpleNamespace(serverInfo=types.SimpleNamespace(name="tfbs"))

    async def list_tools(self):
        return types.SimpleNamespace(tools=[object()] * 33)

    async def call_tool(self, name, arguments):
        reply = self.replies[name]
        if isinstance(reply, BaseException):
            raise reply
        text = types.SimpleNamespace(type="text", text=json.dumps(reply))
        return types.SimpleNamespace(content=[text], isError=False)


class _Crash(BaseException):
    pass


def _fake_transport(monkeypatch, rec, replies):
    @contextlib.asynccontextmanager
    async def stdio_client(params):
        yield None, None

    monkeypatch.setattr(rec, "stdio_client", stdio_client)
    monkeypatch.setattr(rec, "ClientSession", lambda read, write: _FakeSession(replies))


def test_failed_steps_become_error_entries_and_the_run_goes_on(tmp_path, monkeypatch):
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData

    rec = _recorder()
    _fake_transport(monkeypatch, rec, {
        "extract": {"files": {"sequences": "/s.pt"}},
        "broken": McpError(ErrorData(code=-32603, message="server fell over")),
        "predict": {"scores": [1.0]},
    })
    call_log = tmp_path / "calls.jsonl"
    call_log.write_text('{"tool": "extract", "note": "é"}\n', encoding="utf-8")
    plan = [
        {"role": "tool", "tool": "extract", "arguments": {}},
        {"role": "tool", "tool": "predict", "arguments": {"p": {"$ref": "0.files.negatives"}}},
        {"role": "tool", "tool": "broken", "arguments": {}},
        {"role": "tool", "tool": "predict", "arguments": {"p": {"$ref": "0.files.sequences"}}},
    ]
    out = tmp_path / "log.json"
    asyncio.run(rec.run("tfbs-mcp", plan, {"TFBS_CALL_LOG": str(call_log)}, out))
    log = json.loads(out.read_text(encoding="utf-8"))
    steps = log["steps"]
    assert [s["is_error"] for s in steps] == [False, True, True, False]
    assert steps[1]["result"]["missing_key"] == "negatives"
    assert steps[1]["result"]["available_keys"] == ["sequences"]
    assert "server fell over" in steps[2]["result"]["error"]
    assert steps[3]["arguments"] == {"p": "/s.pt"} and steps[3]["result"] == {"scores": [1.0]}
    assert log["server_call_log"] == [{"tool": "extract", "note": "é"}]


def test_a_crash_still_writes_the_steps_recorded_so_far(tmp_path, monkeypatch):
    rec = _recorder()
    _fake_transport(monkeypatch, rec, {"ok": {"x": 1}, "crash": _Crash()})
    plan = [{"role": "user", "text": "go"},
            {"role": "tool", "tool": "ok", "arguments": {}},
            {"role": "tool", "tool": "crash", "arguments": {}}]
    out = tmp_path / "log.json"
    try:
        asyncio.run(rec.run("tfbs-mcp", plan, {}, out))
    except _Crash:
        pass
    else:
        raise AssertionError("the crash must propagate")
    log = json.loads(out.read_text(encoding="utf-8"))
    assert [s.get("tool") for s in log["steps"]] == [None, "ok"]
    assert b"\r\n" not in out.read_bytes()


def test_development_docs_say_how_to_find_the_demo_and_what_figure_2_is():
    import tfbs.data.demo as demo
    from tfbs.mcp.cli import demo_dir

    text = " ".join((REPO / "docs" / "development.md").read_text(encoding="utf-8").split())
    assert 'python -c "from tfbs.mcp.cli import demo_dir; print(demo_dir())"' in text
    assert "namespace package" in text and "`__file__` is `None`" in text
    assert "Figure 2 is the session log recorded this way with the released FLANK checkpoint" in text
    assert demo.__file__ is None and (demo_dir() / "demo_peaks.bed").is_file()


def _refs(node):
    if isinstance(node, dict):
        if set(node) == {"$ref"}:
            yield node["$ref"]
        else:
            for value in node.values():
                yield from _refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _refs(value)


def test_figure_2_plan_has_twelve_tool_steps_whose_refs_point_back():
    plan = json.loads((REPO / "docs" / "demo_session_plan.json").read_text(encoding="utf-8"))
    tool_steps = [i for i, step in enumerate(plan) if step["role"] == "tool"]
    assert len(tool_steps) == 12
    refs = [(i, ref) for i in tool_steps for ref in _refs(plan[i]["arguments"])]
    assert refs
    for i, ref in refs:
        target = int(ref.partition(".")[0])
        assert target < i and target in tool_steps, (i, ref)


def test_development_docs_give_the_command_that_records_figure_2():
    import re

    plan_text = (REPO / "docs" / "demo_session_plan.json").read_text(encoding="utf-8")
    text = " ".join((REPO / "docs" / "development.md").read_text(encoding="utf-8").split())
    section = text.split("The plan behind the application note's Figure 2 is")[1]
    command = section.split("```bash")[1].split("```")[0]
    assert '--server "$(which tfbs-mcp)"' in command
    assert "--plan docs/demo_session_plan.json" in command
    assert sorted(set(re.findall(r"@([A-Z]+)", plan_text))) == ["DEMO", "RELEASE", "WORK"]
    for key in ("DEMO", "RELEASE", "WORK"):
        assert f"--set {key}=" in command, key
    for flag in ("--out", "--commit", "--env TFBS_PROJECT_ROOT=", "--env TFBS_CALL_LOG=",
                 "--timeout 900"):
        assert flag in command, flag
    assert "zenodo.21399578" in text and "nkx2-1-tfbs-variants" in text
