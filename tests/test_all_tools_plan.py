"""docs/all_tools_plan.json: the smoke-test plan that calls every tool of the server.

The plan is played by scripts/record_demo_session.py (see docs/all_tools_runbook.md).
These checks run without the cluster: every registered tool is called, every
argument fits the tool's input schema, every $ref points back to an earlier
tool step, and every "@KEY" placeholder is one the runbook documents.
"""
import asyncio
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLAN = REPO / "docs" / "all_tools_plan.json"
RUNBOOK = REPO / "docs" / "all_tools_runbook.md"

PLACEHOLDERS = {"PROJECT", "DATA", "GENOME", "MOTIF", "CLINVAR", "PDB", "PARTITION", "WORK",
                "RELEASE", "SCRIPT"}
PHASES = ["Setup", "Data prep", "Training", "Analysis", "Structure", "Validation", "SLURM"]
STEP_KEYS = {"role", "tool", "phase", "arguments", "note", "expect_error"}
JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
              "array": list, "object": dict, "null": type(None)}


@pytest.fixture(scope="module")
def plan():
    return json.loads(PLAN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schemas():
    from tfbs.mcp import server
    import tfbs.mcp.foldx_tools  # noqa: F401

    return {t.name: t.inputSchema for t in asyncio.run(server.mcp.list_tools())}


def _is_ref(value):
    return isinstance(value, dict) and set(value) == {"$ref"}


def _refs(node):
    if _is_ref(node):
        yield node["$ref"]
    elif isinstance(node, dict):
        for value in node.values():
            yield from _refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _refs(value)


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def _allowed_types(prop: dict) -> set[str]:
    if "type" in prop:
        kinds = prop["type"]
        return set(kinds if isinstance(kinds, list) else [kinds])
    return {t for option in prop.get("anyOf", []) for t in _allowed_types(option)}


def _fits(value, kinds: set[str]) -> bool:
    if not kinds:
        return True
    for kind in kinds:
        expected = JSON_TYPES.get(kind)
        if expected is None:
            return True
        # bool is an int in Python, but not a JSON integer or number.
        if isinstance(value, bool) and kind in ("integer", "number"):
            continue
        if isinstance(value, expected):
            return True
    return False


def _tool_steps(plan):
    return [(i, step) for i, step in enumerate(plan) if step["role"] == "tool"]


def test_every_registered_tool_is_called(plan, schemas):
    assert len(schemas) == 33
    called = {step["tool"] for _, step in _tool_steps(plan)}
    assert called == set(schemas), {"never called": sorted(set(schemas) - called),
                                    "not registered": sorted(called - set(schemas))}


def test_steps_have_the_documented_keys_phases_and_notes(plan):
    seen = []
    for i, step in _tool_steps(plan):
        assert set(step) <= STEP_KEYS, (i, sorted(set(step) - STEP_KEYS))
        assert step["phase"] in PHASES, (i, step["phase"])
        assert isinstance(step["arguments"], dict), i
        assert isinstance(step.get("note", ""), str) and step.get("note"), i
        assert isinstance(step.get("expect_error", False), bool), i
        if not seen or seen[-1] != step["phase"]:
            seen.append(step["phase"])
    # Grouped by phase: each phase is one block, in the documented order.
    assert seen == PHASES


def test_every_argument_is_in_the_schema_and_every_required_key_is_given(plan, schemas):
    for i, step in _tool_steps(plan):
        schema = schemas[step["tool"]]
        props = schema.get("properties", {})
        args = step["arguments"]
        unknown = set(args) - set(props)
        assert not unknown, (i, step["tool"], sorted(unknown))
        missing = set(schema.get("required", [])) - set(args)
        assert not missing, (i, step["tool"], sorted(missing))


def test_literal_argument_values_have_the_schema_type(plan, schemas):
    for i, step in _tool_steps(plan):
        props = schemas[step["tool"]].get("properties", {})
        for key, value in step["arguments"].items():
            if _is_ref(value):
                continue
            kinds = _allowed_types(props[key])
            assert _fits(value, kinds), (i, step["tool"], key, value, sorted(kinds))
            if isinstance(value, list):
                item_kinds = _allowed_types(props[key].get("items", {}))
                for option in props[key].get("anyOf", []):
                    item_kinds |= _allowed_types(option.get("items", {}))
                for item in value:
                    if not _is_ref(item):
                        assert _fits(item, item_kinds), (i, step["tool"], key, item)


def test_every_ref_points_to_an_earlier_tool_step(plan):
    tool_indices = {i for i, _ in _tool_steps(plan)}
    refs = [(i, ref) for i, step in _tool_steps(plan) for ref in _refs(step["arguments"])]
    assert refs
    for i, ref in refs:
        target, _, path = ref.partition(".")
        assert target.isdigit() and path, (i, ref)
        assert int(target) < i and int(target) in tool_indices, (i, ref)


def test_every_placeholder_is_documented(plan):
    used = {key for text in _strings(plan) for key in re.findall(r"@([A-Z][A-Z_]*)", text)}
    assert used <= PLACEHOLDERS, sorted(used - PLACEHOLDERS)
    assert {"PROJECT", "DATA", "GENOME", "MOTIF", "CLINVAR", "PDB", "PARTITION",
            "WORK", "SCRIPT"} <= used


def test_expected_errors_are_marked_and_explained(plan):
    expected = [(i, step["tool"]) for i, step in _tool_steps(plan) if step.get("expect_error")]
    assert expected == [(i, "tfbs_sweep") for i, step in _tool_steps(plan)
                        if step["tool"] == "tfbs_sweep"]
    runbook = RUNBOOK.read_text(encoding="utf-8")
    assert "tfbs_sweep" in runbook and "expect_error" in runbook


def _recorder_command(runbook: str) -> str:
    """The fenced bash block of the runbook that plays the plan, as one line."""
    blocks = re.findall(r"```bash\n(.*?)```", runbook, flags=re.S)
    commands = [b for b in blocks if "record_demo_session.py" in b and "--plan" in b]
    assert len(commands) == 1, len(commands)
    command = commands[0].split("record_demo_session.py", 1)[1]
    return " ".join(command.replace("\\\n", " ").split())


def test_runbook_command_sets_every_placeholder_and_server_variable(plan):
    command = _recorder_command(RUNBOOK.read_text(encoding="utf-8"))
    for key in sorted(PLACEHOLDERS):
        assert f"--set {key}=" in command, key
    for var in ("TFBS_PROJECT_ROOT", "TFBS_CALL_LOG", "TFBS_SLURM_PARTITION", "TFBS_GENOME_FASTA",
                "TFBS_MOTIF_FILE", "TFBS_CLINVAR_VCF", "TFBS_FOLDX_PDB"):
        assert f"--env {var}=" in command, var
    assert re.search(r"--plan \S*docs/all_tools_plan\.json", command)
    assert "--timeout" in command


def test_plan_and_runbook_use_no_long_dashes():
    for path in (PLAN, RUNBOOK, Path(__file__)):
        text = path.read_text(encoding="utf-8")
        assert "\u2013" not in text and "\u2014" not in text, path.name
