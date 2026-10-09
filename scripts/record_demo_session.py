"""Record a real tfbs-mcp session over the MCP stdio protocol.

The script launches the tfbs-mcp server exactly as an AI client would, lists
its tools, then runs a plan: user messages, the assistant's stated plan, and
tool calls with their arguments. Every call's arguments, full result and
wall time are written to a JSON log, from which the session figures of the
application note are rendered. Nothing in the log is typed in by hand.

Usage::

    python scripts/record_demo_session.py --server /path/to/tfbs-mcp \
        --plan plan.json --out session_log.json \
        [--env KEY=VALUE ...] [--set KEY=VALUE ...] [--arg ARG ...]

The plan is a JSON list of steps: {"role": "user", "text": ...},
{"role": "llm", "text": ...}, or {"role": "tool", "tool": ..., "arguments": {...},
"phase": ..., "kind": ...}. A tool step may reference an earlier result with
"$ref": "<step index>.<json path>" inside its arguments.

--set KEY=VALUE (repeatable) replaces "@KEY" by VALUE in every string of the
plan before any $ref is resolved, so one plan serves several machines, e.g.
--set DEMO=/path/to/tfbs/data/demo for "@DEMO/demo_peaks.bed".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

LOGGED_ENV_PREFIXES = ("TFBS_", "PYTHONPATH")


def _substitute(value, variables: dict[str, str]):
    """Replace "@KEY" by its value in every string inside value."""
    if isinstance(value, str):
        # Longest key first, so @DEMO_DIR is not read as @DEMO followed by _DIR.
        for key in sorted(variables, key=len, reverse=True):
            value = value.replace(f"@{key}", variables[key])
        return value
    if isinstance(value, dict):
        return {k: _substitute(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, variables) for v in value]
    return value


def _resolve_refs(value, results):
    """Replace {"$ref": "3.files.sequences"} with the value from step 3's result."""
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            idx, _, path = value["$ref"].partition(".")
            node = results[int(idx)]
            for key in path.split(".") if path else []:
                node = node[int(key)] if isinstance(node, list) else node[key]
            return node
        return {k: _resolve_refs(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_refs(v, results) for v in value]
    return value


def _parse_result(result):
    texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
    raw = "\n".join(texts)
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return None, raw


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1), encoding="utf-8", newline="\n")


async def run(server: str, plan: list, env: dict, out: Path, timeout: float = 600.0,
              server_args: list | None = None) -> None:
    # The SDK's default environment is the safe minimum a server needs; the
    # recording variables are added on top rather than inheriting everything.
    params = StdioServerParameters(command=server, args=server_args or [],
                                   env={**get_default_environment(), **env})
    log = {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "server": server,
        "server_env": {k: v for k, v in env.items() if k.startswith(LOGGED_ENV_PREFIXES)},
        "client": "scripted MCP stdio client (record_demo_session.py)",
        "steps": [],
    }
    results: dict[int, object] = {}
    t_launch = time.perf_counter()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await asyncio.wait_for(session.initialize(), 600)
            log["seconds_to_initialize"] = round(time.perf_counter() - t_launch, 3)
            tools = await asyncio.wait_for(session.list_tools(), 60)
            log["server_name"] = init.serverInfo.name
            log["tools_listed"] = len(tools.tools)
            for i, step in enumerate(plan):
                entry = dict(step)
                if step["role"] == "tool":
                    arguments = _resolve_refs(step.get("arguments", {}), results)
                    entry["arguments"] = arguments
                    t0 = time.perf_counter()
                    try:
                        result = await asyncio.wait_for(session.call_tool(step["tool"], arguments), timeout)
                        parsed, raw = _parse_result(result)
                        is_error = bool(getattr(result, "isError", False))
                    except asyncio.TimeoutError:
                        parsed, raw, is_error = {"error": f"no reply within {timeout:.0f} s"}, "", True
                    entry["seconds"] = round(time.perf_counter() - t0, 3)
                    entry["seconds_since_launch"] = round(time.perf_counter() - t_launch, 3)
                    entry["result"] = parsed
                    if parsed is None:
                        entry["result_text"] = raw
                    entry["is_error"] = is_error or (isinstance(parsed, dict) and "error" in parsed)
                    results[i] = parsed
                    status = "error" if entry["is_error"] else "ok"
                    print(f"[{i:02d}] {step['tool']} -> {status} ({entry['seconds']}s)", file=sys.stderr)
                log["steps"].append(entry)
            # Written here, before the session closes: a server that hangs on
            # shutdown must not cost the recording.
            call_log = env.get("TFBS_CALL_LOG")
            if call_log and Path(call_log).is_file():
                log["server_call_log"] = [json.loads(l) for l in Path(call_log).read_text().splitlines() if l.strip()]
            _write_json(out, log)
            print(f"wrote {out} with {len(log['steps'])} steps", file=sys.stderr)


def _pairs(items: list[str], flag: str) -> dict[str, str]:
    pairs = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"{flag} expects KEY=VALUE, got {item!r}")
        pairs[key] = value
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="path to the tfbs-mcp executable")
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="environment variable for the server (repeatable)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="variables",
                    help='replace "@KEY" by VALUE in every string of the plan (repeatable)')
    ap.add_argument("--commit", default="", help="tfbs-mcp commit to record in the log")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for one tool call")
    ap.add_argument("--arg", action="append", default=[], help="argument for the server command (repeatable)")
    args = ap.parse_args()
    env = _pairs(args.env, "--env")
    plan = _substitute(json.loads(args.plan.read_text(encoding="utf-8")),
                       _pairs(args.variables, "--set"))
    asyncio.run(run(args.server, plan, env, args.out, args.timeout, args.arg))
    if args.commit:
        log = json.loads(args.out.read_text(encoding="utf-8"))
        log["tfbs_mcp_commit"] = args.commit
        _write_json(args.out, log)


if __name__ == "__main__":
    main()
