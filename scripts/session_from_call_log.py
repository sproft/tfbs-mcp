"""Build a session log from a session played by a real MCP client.

scripts/record_demo_session.py plays a scripted plan; this script covers the
other route, a session typed into a real client such as Claude Code. The
server writes every tool call (arguments, result, wall time) to the file
named by $TFBS_CALL_LOG; this script turns that file, and optionally the
client's own transcript, into the JSON layout record_demo_session.py writes,
so the same renderer draws the figure from it. Nothing is typed in by hand.

Usage::

    python scripts/session_from_call_log.py --call-log calls.jsonl \
        --out session_log.json [--transcript <claude-code-session>.jsonl] \
        [--phases phases.json] [--labels labels.json] [--commit <hash>] \
        [--client "<name and version>"]

Without --transcript the steps are the tool calls of the call log. With it,
the user messages and the assistant's text are interleaved with the tool
calls in timestamp order. A Claude Code transcript names an MCP tool
mcp__<server>__<tool>. Each of its tool_use blocks is matched to the
call-log entry with the same tool name and equal arguments whose timestamp
is the closest at or after the tool_use time (an entry up to 5 s earlier is
accepted when none is later, and an entry that starts after the call's
tool_result came back never matches). Only a tool_use whose arguments equal
no entry falls back to the next unused entry of the same tool inside the
same time window. Call-log entries that start outside the transcript's time
span belong to another session: they are dropped and counted as
calls_outside_window; the other unmatched entries become steps of their own.

A tool_result in the transcript supplies the result when the call-log entry
has none (a log written before results were logged). When the entry holds a
truncated result, the transcript's result is used only if it parses as JSON;
otherwise the step keeps the logged head as result_text with
result_truncated true and result_bytes. Claude Code replaces outputs over
about 30 KB with a "<persisted-output> Output too large" note naming a local
file; that note is never used as a result in any case: a step with nothing
better gets an empty result_text and result_truncated true. When the
transcript's tool_result is an error but the call-log entry logged a success
(the client timed out before the server finished), the step keeps the
server's result and adds client_saw_error true and client_error_text. Calls to the
client's own tools (Bash, Read, ...) and to other MCP servers are kept as
steps with kind "client" and phase "Client". Subagent (sidechain) lines and
the lines the client injects itself are skipped: isMeta and
isCompactSummary lines, task notifications (origin.kind
"task-notification"), and user text that starts with <command-name>,
<command-message>, <local-command-stdout>, <local-command-caveat>,
<task-notification>, <system-reminder> or "[Request interrupted".

--phases is a JSON object mapping a tool name (or an fnmatch pattern such as
"tfbs_prepare_*") to a phase label; tools it does not name get the default
phase below. --labels is a JSON object mapping a 1-based index among the
tfbs-mcp tool steps (as a string, e.g. "3") or a tool name to a short label,
written as the step's shown_args (the renderer prints it inside the
brackets after the tool name); an index wins over a tool name. Without
--labels no step has shown_args.

The log records seconds_to_initialize and tools_listed as null: a session
played by a real client does not measure them.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_PHASES: list[tuple[str, tuple[str, ...]]] = [
    ("Data prep", ("tfbs_extract_loci", "tfbs_prepare_*", "tfbs_preprocess")),
    ("Training", ("tfbs_config", "tfbs_train", "tfbs_sweep", "tfbs_validate", "slurm_*")),
    ("Validation", ("tfbs_remap_validation", "tfbs_chipseq_*",
                    "tfbs_disease_variant_validation", "tfbs_classify_metrics")),
    ("Structure", ("tfbs_fetch_structure", "tfbs_binding_*")),
]
DEFAULT_PHASE = "Analysis"
CLIENT_PHASE = "Client"
TFBS_TOOL_PREFIXES = ("tfbs_", "slurm_")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
# A call-log entry may start up to this long before the transcript's tool_use
# line (the client can start a tool before it writes that line), or up to
# this long after the tool_result line, and still belong to that call.
MATCH_SLACK_SECONDS = 5.0
# User text the client injects itself; the user never typed it.
HARNESS_PREFIXES = ("<command-name>", "<command-message>", "<local-command-stdout>",
                    "<local-command-caveat>", "<task-notification>", "<system-reminder>",
                    "[Request interrupted")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> tuple[list[dict], int]:
    """The JSON objects of a JSON Lines file, and how many lines were not JSON."""
    records, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                bad += 1
    return records, bad


def parse_ts(value) -> datetime | None:
    """An ISO 8601 timestamp as an aware UTC datetime; None when unreadable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def mcp_tool_name(name: str) -> str | None:
    """The tool part of a client name mcp__<server>__<tool>, else None."""
    if not name.startswith("mcp__"):
        return None
    _, sep, tool = name[len("mcp__"):].partition("__")
    return tool if sep and tool else None


def mcp_server_name(name: str) -> str | None:
    if mcp_tool_name(name) is None:
        return None
    return name[len("mcp__"):].partition("__")[0]


def _block_text(content) -> str:
    """The text of a tool_result's content: a string or a list of text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _is_harness_text(text: str) -> bool:
    return text.lstrip().startswith(HARNESS_PREFIXES)


def _user_text(content) -> str:
    """What the user typed: text blocks the client injected itself are left out."""
    if isinstance(content, str):
        return "" if _is_harness_text(content) else content.strip()
    if isinstance(content, list):
        texts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(t for t in texts if t.strip() and not _is_harness_text(t)).strip()
    return ""


def _injected_by_client(rec: dict) -> bool:
    """A user line the client wrote itself: meta, compaction summary or task notification."""
    if rec.get("isMeta") or rec.get("isCompactSummary"):
        return True
    origin = rec.get("origin")
    return isinstance(origin, dict) and origin.get("kind") == "task-notification"


def read_transcript(records: list[dict]) -> tuple[list[dict], dict[str, dict], str | None]:
    """Events (user, llm, tool_use) in file order, tool results by id, client version."""
    events: list[dict] = []
    results: dict[str, dict] = {}
    version = None
    for order, rec in enumerate(records):
        kind = rec.get("type")
        if kind not in ("user", "assistant") or rec.get("isSidechain"):
            continue
        version = version or rec.get("version")
        ts = parse_ts(rec.get("timestamp"))
        message = rec.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if kind == "user":
            if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        results[str(b.get("tool_use_id"))] = {
                            "ts": ts, "text": _block_text(b.get("content")),
                            "is_error": bool(b.get("is_error")),
                        }
                continue
            if _injected_by_client(rec):
                continue
            text = _user_text(content)
            if text:
                events.append({"kind": "user", "ts": ts, "order": order, "text": text})
            continue
        # assistant
        if isinstance(content, str):
            if content.strip():
                events.append({"kind": "llm", "ts": ts, "order": order, "text": content.strip()})
            continue
        pending: list[str] = []

        def flush() -> None:
            text = "\n\n".join(t for t in pending if t.strip()).strip()
            if text:
                events.append({"kind": "llm", "ts": ts, "order": order, "text": text})
            pending.clear()

        for b in content if isinstance(content, list) else []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                pending.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                flush()
                events.append({"kind": "tool_use", "ts": ts, "order": order,
                               "id": str(b.get("id")), "name": str(b.get("name", "")),
                               "input": b.get("input") if isinstance(b.get("input"), dict) else {}})
        flush()
    return events, results, version


# ---------------------------------------------------------------------------
# Building steps
# ---------------------------------------------------------------------------
def phase_of(tool: str, phases: dict[str, str] | None, client: bool = False) -> str:
    for key, label in (phases or {}).items():
        if tool == key or fnmatch.fnmatchcase(tool, key):
            return label
    if client:
        return CLIENT_PHASE
    for label, patterns in DEFAULT_PHASES:
        if any(fnmatch.fnmatchcase(tool, p) for p in patterns):
            return label
    return DEFAULT_PHASE


def _parse_text(text: str):
    """(parsed JSON or None, text)."""
    try:
        return json.loads(text), text
    except (ValueError, TypeError):
        return None, text


# How Claude Code replaces a tool output it found too large: the client keeps
# the full text in a local file and hands the model this note (with the local
# path) instead. Such a note is never a result.
CLIENT_OUTPUT_NOTES = ("<persisted-output>", "Output too large")


def _is_client_output_note(text) -> bool:
    """Whether a transcript tool_result is the client's "Output too large" note."""
    return isinstance(text, str) and text.lstrip().startswith(CLIENT_OUTPUT_NOTES)


def _logged_result(entry: dict) -> tuple[str, object, str | None, int | None]:
    """(state, result, result_text, truncated size in bytes) of one call-log entry.

    state is "missing" when the entry has no result (a log written before
    results were logged), "truncated" when it holds only the head of a
    large one, else "ok".
    """
    if "result" not in entry:
        return "missing", None, None, None
    res = entry["result"]
    if isinstance(res, dict) and res.get("truncated") is True and "head" in res:
        size = res.get("bytes")
        size = size if isinstance(size, int) and not isinstance(size, bool) else None
        return "truncated", None, str(res.get("head", "")), size
    if isinstance(res, dict) and set(res) == {"text"} and isinstance(res["text"], str):
        return "ok", None, res["text"], None
    return "ok", res, None, None


def _seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return round(max((end - start).total_seconds(), 0.0), 3)


def _is_error(result, flag: bool) -> bool:
    return bool(flag) or (isinstance(result, dict) and "error" in result)


def tfbs_step(tool: str, entry: dict | None, use: dict | None, tres: dict | None,
              phases: dict[str, str] | None) -> dict:
    """One tfbs-mcp tool step from a call-log entry and/or the transcript."""
    step: dict = {"role": "tool", "tool": tool}
    if entry is not None and isinstance(entry.get("arguments"), dict):
        step["arguments"] = entry["arguments"]
    else:
        step["arguments"] = (use or {}).get("input", {})

    state, result, result_text, truncated_bytes = "missing", None, None, None
    if entry is not None:
        state, result, result_text, truncated_bytes = _logged_result(entry)
    client_note = tres is not None and _is_client_output_note(tres["text"])
    if state == "missing" and tres is not None:
        if client_note:
            # Claude Code kept the output in a local file; the note (and its
            # local path) is not the result, so the step has none.
            state, result_text = "truncated", ""
        else:
            result, result_text = _parse_text(tres["text"])
    elif state == "truncated" and tres is not None and not client_note:
        # Only a complete JSON answer replaces the logged head.
        parsed, _ = _parse_text(tres["text"])
        if parsed is not None:
            state, result, result_text = "ok", parsed, None
    step["result"] = result
    if result is None:
        step["result_text"] = result_text or ""
    if state == "truncated":
        step["result_truncated"] = True
        if truncated_bytes is not None:
            step["result_bytes"] = truncated_bytes

    seconds = entry.get("seconds") if entry is not None else None
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        seconds = _seconds((use or {}).get("ts"), (tres or {}).get("ts"))
    if seconds is not None:
        step["seconds"] = seconds

    client_error = bool((tres or {}).get("is_error"))
    if entry is not None and isinstance(entry.get("is_error"), bool):
        flag = entry["is_error"]
    else:
        flag = client_error
    step["is_error"] = _is_error(result, flag)
    if client_error and not step["is_error"]:
        # The server finished and logged a success, but the client had
        # already given up (e.g. "MCP error -32001: Request timed out"): the
        # step keeps the server's result and records what the model saw.
        step["client_saw_error"] = True
        step["client_error_text"] = "" if client_note else str(tres.get("text", ""))
    step["phase"] = phase_of(tool, phases)
    ts = parse_ts(entry.get("timestamp")) if entry is not None else None
    ts = ts or (use or {}).get("ts")
    if ts is not None:
        step["timestamp"] = ts.isoformat(timespec="milliseconds")
    return step


def client_step(use: dict, tres: dict | None, phases: dict[str, str] | None) -> dict:
    """A call to a tool of the client (Bash, Read, ...) or of another MCP server."""
    step: dict = {"role": "tool", "tool": use["name"], "kind": "client",
                  "arguments": use.get("input", {})}
    if tres is not None and _is_client_output_note(tres["text"]):
        result, text = None, ""
        step["result_truncated"] = True
    else:
        result, text = _parse_text(tres["text"]) if tres is not None else (None, "")
    step["result"] = result
    if result is None:
        step["result_text"] = text
    seconds = _seconds(use.get("ts"), (tres or {}).get("ts"))
    if seconds is not None:
        step["seconds"] = seconds
    step["is_error"] = _is_error(result, bool((tres or {}).get("is_error")))
    step["phase"] = phase_of(use["name"], phases, client=True)
    if use.get("ts") is not None:
        step["timestamp"] = use["ts"].isoformat(timespec="milliseconds")
    return step


def _in_call_window(entry_ts: datetime | None, use: dict, tres: dict | None) -> bool:
    """Whether a call-log entry can belong to this tool_use, judged by time.

    The entry must start no more than MATCH_SLACK_SECONDS before the
    tool_use line and, when the transcript holds the call's tool_result, no
    later than that line plus the same slack.  Without timestamps anything fits.
    """
    if entry_ts is None or use.get("ts") is None:
        return True
    if (entry_ts - use["ts"]).total_seconds() < -MATCH_SLACK_SECONDS:
        return False
    end = (tres or {}).get("ts")
    return end is None or (entry_ts - end).total_seconds() <= MATCH_SLACK_SECONDS


def match_calls(uses: list[dict], calls: list[dict], results: dict[str, dict]
                ) -> tuple[dict[int, int], dict[int, str]]:
    """Pair transcript tool_use events with call-log entries.

    uses are the tfbs-mcp tool_use events, each with "tool", "input", "ts"
    and "id".  Returns ({use index: call index}, {use index: "arguments" or
    "order"}).  Pairs of equal tool name and equal arguments come first,
    the smallest time gap first and an entry at or after the tool_use time
    before an earlier one, so a call missing on either side, or entries
    written out of order, shift nothing.  Only a use whose arguments equal
    no entry of its tool falls back to the earliest unused entry of that
    tool whose time fits the call (_in_call_window).
    """
    times = [parse_ts(c.get("timestamp")) for c in calls]
    candidates: list[tuple[int, float, int, int]] = []
    for u, use in enumerate(uses):
        tres = results.get(use["id"])
        for i, entry in enumerate(calls):
            if str(entry.get("tool")) != use["tool"] or entry.get("arguments") != use["input"]:
                continue
            if not _in_call_window(times[i], use, tres):
                continue
            gap = 0.0 if times[i] is None or use["ts"] is None else \
                (times[i] - use["ts"]).total_seconds()
            candidates.append((0 if gap >= 0 else 1, abs(gap), u, i))
    candidates.sort()
    pairs: dict[int, int] = {}
    how: dict[int, str] = {}
    taken: set[int] = set()
    for _, _, u, i in candidates:
        if u in pairs or i in taken:
            continue
        pairs[u], how[u] = i, "arguments"
        taken.add(i)

    by_tool: dict[str, list[int]] = defaultdict(list)
    for i, entry in enumerate(calls):
        by_tool[str(entry.get("tool"))].append(i)
    for indices in by_tool.values():
        indices.sort(key=lambda i: (times[i] or _EPOCH, i))
    rest = sorted((u for u in range(len(uses)) if u not in pairs),
                  key=lambda u: (uses[u]["ts"] or _EPOCH, u))
    for u in rest:
        use = uses[u]
        same_tool = by_tool.get(use["tool"], [])
        if any(calls[i].get("arguments") == use["input"] for i in same_tool):
            continue  # its arguments are logged, but for another call: no guessing
        tres = results.get(use["id"])
        for i in same_tool:
            if i not in taken and _in_call_window(times[i], use, tres):
                pairs[u], how[u] = i, "order"
                taken.add(i)
                break
    return pairs, how


def build_steps(calls: list[dict], events: list[dict] | None, results: dict[str, dict],
                phases: dict[str, str] | None) -> tuple[list[dict], dict]:
    """Steps in timestamp order, and counts of how the call log was matched."""
    stats = {"matched": 0, "matched_by_order": 0, "transcript_only": 0,
             "call_log_only": 0, "calls_outside_window": 0, "client": 0}
    keyed: list[tuple[datetime, int, int, dict]] = []
    logged_tools = {str(c.get("tool")) for c in calls}

    filled: list[dict] = []
    uses: list[dict] = []
    use_index: dict[tuple[int, str], int] = {}
    last = _EPOCH
    for ev in events or []:
        ts = ev["ts"] or last
        last = ts
        ev = {**ev, "ts": ts}
        filled.append(ev)
        if ev["kind"] != "tool_use":
            continue
        tool = mcp_tool_name(ev["name"])
        if tool is not None and (tool in logged_tools or tool.startswith(TFBS_TOOL_PREFIXES)):
            use_index[(ev["order"], ev["id"])] = len(uses)
            uses.append({**ev, "tool": tool})
    pairs, how = match_calls(uses, calls, results)

    for ev in filled:
        if ev["kind"] in ("user", "llm"):
            keyed.append((ev["ts"], 0, ev["order"], {
                "role": ev["kind"], "text": ev["text"],
                "timestamp": ev["ts"].isoformat(timespec="milliseconds")}))
            continue
        tres = results.get(ev["id"])
        n = use_index.get((ev["order"], ev["id"]))
        if n is None:
            stats["client"] += 1
            keyed.append((ev["ts"], 0, ev["order"], client_step(ev, tres, phases)))
            continue
        entry = calls[pairs[n]] if n in pairs else None
        if entry is None:
            stats["transcript_only"] += 1
        else:
            stats["matched"] += 1
            stats["matched_by_order"] += how[n] == "order"
        keyed.append((ev["ts"], 0, ev["order"],
                      tfbs_step(uses[n]["tool"], entry, ev, tres, phases)))

    # A call-log entry that starts outside the transcript's time span is from
    # another session (the log is append-only): left out and counted.
    stamped = [ev["ts"] for ev in events or [] if ev["ts"] is not None]
    window = None
    if events is not None and stamped:
        slack = timedelta(seconds=MATCH_SLACK_SECONDS)
        window = (min(stamped) - slack, max(stamped) + slack)
    used = set(pairs.values())
    last = _EPOCH
    for i, entry in enumerate(calls):
        stamp = parse_ts(entry.get("timestamp"))
        ts = stamp or last
        last = ts
        if i in used:
            continue
        if window is not None and stamp is not None and not window[0] <= stamp <= window[1]:
            stats["calls_outside_window"] += 1
            continue
        stats["call_log_only"] += 1
        keyed.append((ts, 1, i, tfbs_step(str(entry.get("tool")), entry, None, None, phases)))

    keyed.sort(key=lambda k: (k[0], k[1], k[2]))
    return [k[3] for k in keyed], stats


def apply_labels(steps: list[dict], labels: dict[str, str] | None) -> None:
    """Write shown_args on the tfbs-mcp tool steps that labels names.

    A key is a 1-based index among the tfbs-mcp tool steps (client steps
    are not counted) or a tool name; an index wins over a tool name.
    """
    if not labels:
        return
    n = 0
    for step in steps:
        if step.get("role") != "tool" or step.get("kind") == "client":
            continue
        n += 1
        label = labels.get(str(n), labels.get(step["tool"]))
        if label is not None:
            step["shown_args"] = label


def convert(call_log: Path, transcript: Path | None = None, phases: dict | None = None,
            commit: str = "", client: str = "", labels: dict | None = None) -> dict:
    calls, bad_calls = read_jsonl(call_log)
    events, results, version, server_name, bad_lines = None, {}, None, None, 0
    if transcript is not None:
        records, bad_lines = read_jsonl(transcript)
        events, results, version = read_transcript(records)
        tfbs_uses = [e["name"] for e in events if e["kind"] == "tool_use"
                     and (mcp_tool_name(e["name"]) or "").startswith(TFBS_TOOL_PREFIXES)]
        server_name = mcp_server_name(tfbs_uses[0]) if tfbs_uses else None
    steps, stats = build_steps(calls, events, results, phases)
    apply_labels(steps, labels)

    times = [parse_ts(s.get("timestamp")) for s in steps]
    times = [t for t in times if t is not None]
    start = min(times) if times else datetime.now(timezone.utc)
    if not client:
        client = (f"Claude Code {version}" if version else "Claude Code") if transcript \
            else "MCP client (steps from the tfbs-mcp call log)"
    log: dict = {
        "recorded_at": start.isoformat(timespec="seconds"),
        "server": "tfbs-mcp",
        "server_env": {"TFBS_CALL_LOG": str(call_log)},
        "client": client,
        # Measured by record_demo_session.py only; a real client's session has neither.
        "seconds_to_initialize": None,
        "tools_listed": None,
    }
    if server_name:
        # The name the client registered the server under (mcp__<name>__...),
        # not the server's own serverInfo name.
        log["client_server_name"] = server_name
    if commit:
        log["tfbs_mcp_commit"] = commit
    log["converted_from"] = {
        "call_log": str(call_log), "transcript": str(transcript) if transcript else None,
        "converter": "scripts/session_from_call_log.py", **stats,
        "unreadable_call_log_lines": bad_calls, "unreadable_transcript_lines": bad_lines,
    }
    log["steps"] = steps
    log["server_call_log"] = calls
    return log


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")


def _load_mapping(path: Path | None, flag: str, what: str) -> dict[str, str] | None:
    if path is None:
        return None
    if not path.is_file():
        raise SystemExit(f"{flag}: no such file {path}")
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SystemExit(f"{flag}: {path} is not JSON ({exc})") from None
    if not isinstance(mapping, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
        raise SystemExit(f"{flag}: {path} must be a JSON object mapping {what}")
    return mapping


def _load_phases(path: Path | None) -> dict[str, str] | None:
    return _load_mapping(path, "--phases", "tool name to phase label")


def _load_labels(path: Path | None) -> dict[str, str] | None:
    return _load_mapping(path, "--labels",
                         'a tfbs call index ("1", "2", ...) or a tool name to a label')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--call-log", required=True, type=Path, help="the server's $TFBS_CALL_LOG file")
    ap.add_argument("--out", required=True, type=Path, help="session log to write (JSON)")
    ap.add_argument("--transcript", type=Path, help="Claude Code session transcript (.jsonl)")
    ap.add_argument("--phases", type=Path, help="JSON object: tool name -> phase label")
    ap.add_argument("--labels", type=Path,
                    help="JSON object: 1-based tfbs call index or tool name -> shown_args label")
    ap.add_argument("--commit", default="", help="tfbs-mcp commit to record in the log")
    ap.add_argument("--client", default="", help="client name and version to record in the log")
    args = ap.parse_args(argv)
    if not args.call_log.is_file():
        raise SystemExit(f"--call-log: no such file {args.call_log}")
    if args.transcript is not None and not args.transcript.is_file():
        raise SystemExit(f"--transcript: no such file {args.transcript}")
    log = convert(args.call_log, args.transcript, _load_phases(args.phases),
                  args.commit, args.client, _load_labels(args.labels))
    _write_json(args.out, log)
    c = log["converted_from"]
    print(f"wrote {args.out} with {len(log['steps'])} steps "
          f"({c['matched']} calls matched to the transcript, {c['matched_by_order']} of them "
          f"by order only; {c['transcript_only']} from the transcript only; "
          f"{c['call_log_only']} from the call log only; {c['calls_outside_window']} call-log "
          f"entries outside the transcript's time span left out; "
          f"{c['client']} client tool calls)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
