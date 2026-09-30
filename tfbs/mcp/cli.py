"""Entry point for the ``tfbs-mcp`` command.

``tfbs-mcp``          start the server (your AI app does this, not you)
``tfbs-mcp --check``  verify the install and print the exact setup commands

This module must not import ``mcp`` at the top: its job is to report a
missing or incompatible ``mcp`` package clearly instead of with a traceback.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

MCP_REQUIREMENT = "mcp>=1.28,<2"
FIX_MCP = f'pip install "{MCP_REQUIREMENT}"'

USAGE = """\
usage: tfbs-mcp [--check]

  (no arguments)  start the MCP server. Your AI app runs this for you after
                  setup; started by hand it waits silently for input (Ctrl+C
                  to quit).
  --check         verify the installation and print the command that
                  connects the server to Claude Code or Claude Desktop.
"""


def mcp_version_problem(version: str | None) -> str | None:
    """Return a human-readable problem with the installed mcp, or None if fine."""
    if version is None:
        return f"the 'mcp' package is not installed. Fix: {FIX_MCP}"
    try:
        major = int(version.split(".")[0])
    except ValueError:
        return None
    if major >= 2:
        return (
            f"mcp {version} is installed, but this server needs mcp 1.x "
            f"(mcp 2 renamed the API it uses). Fix: {FIX_MCP}"
        )
    return None


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def find_executable() -> str | None:
    """Absolute path of the tfbs-mcp launcher in the running environment."""
    base = Path(sys.executable).parent
    # venv/conda on Linux and macOS: bin/; venv on Windows: Scripts/ (python.exe
    # lives there too); conda on Windows: python.exe at the root, launchers in
    # Scripts/.
    for folder in (base, base / "Scripts"):
        for name in ("tfbs-mcp.exe", "tfbs-mcp"):
            candidate = folder / name
            if candidate.is_file():
                return str(candidate)
    found = shutil.which("tfbs-mcp")
    return str(Path(found).resolve()) if found else None


def claude_code_command(executable: str) -> str:
    return f'claude mcp add tfbs --scope user -- "{executable}"'


def claude_desktop_config(executable: str) -> str:
    # json.dumps escapes Windows backslashes correctly, so the output can be
    # pasted as is.
    return json.dumps({"mcpServers": {"tfbs": {"command": executable, "args": []}}}, indent=2)


def _count_tools() -> int:
    import tfbs.mcp.server as server
    import tfbs.mcp.foldx_tools  # noqa: F401  registers the structure tools

    return len(asyncio.run(server.mcp.list_tools()))


def check() -> int:
    """Print a pass/fail checklist and the setup commands. Returns an exit code."""
    failed = False

    def report(ok: bool | None, text: str) -> None:
        nonlocal failed
        mark = {True: "ok  ", False: "FAIL", None: "note"}[ok]
        print(f"  [{mark}] {text}")
        if ok is False:
            failed = True

    print("tfbs-mcp installation check\n")

    py = sys.version_info
    py_text = f"Python {py.major}.{py.minor}.{py.micro}"
    if py >= (3, 14):
        report(None, f"{py_text}: the genomics and viz extras need Python 3.13 or older "
                     "(their dependency pybigtools has no 3.14 build yet)")
    else:
        report(True, py_text)

    mcp_version = _installed_version("mcp")
    problem = mcp_version_problem(mcp_version)
    report(problem is None, problem or f"mcp {mcp_version}")

    if problem is None:
        try:
            report(True, f"server loads, {_count_tools()} tools registered")
        except Exception as exc:  # report, do not crash the checker
            report(False, f"server failed to load: {type(exc).__name__}: {exc}")

    torch_version = _installed_version("torch")
    report(torch_version is not None,
           f"PyTorch {torch_version}" if torch_version else "PyTorch is not installed")

    tangermeme_version = _installed_version("tangermeme")
    report(True if tangermeme_version else None,
           f"tangermeme {tangermeme_version}" if tangermeme_version
           else "tangermeme not installed: attribution, mutagenesis and genome "
                "tools will be unavailable (install the genomics and viz extras)")

    executable = find_executable()
    report(executable is not None,
           f"executable: {executable}" if executable
           else "could not locate the tfbs-mcp executable in this environment")

    if failed:
        print("\nFix the FAIL lines above, then run `tfbs-mcp --check` again.")
        return 1

    print("\nEverything needed is installed. Connect the server with ONE of these:\n")
    print("Claude Code: run this command once, then restart Claude Code.\n")
    print(f"    {claude_code_command(executable)}\n")
    print("Claude Desktop: open Settings > Developer > Edit Config, paste the block")
    print("below (merge it into mcpServers if the file already has one), save, then")
    print("quit Claude Desktop completely and reopen it.\n")
    for line in claude_desktop_config(executable).splitlines():
        print(f"    {line}")
    print()
    return 0


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] in ("-h", "--help"):
        print(USAGE)
        return
    if args and args[0] == "--check":
        sys.exit(check())
    if args:
        sys.exit(f"tfbs-mcp: unknown argument {args[0]!r}\n\n{USAGE}")

    # Stdout belongs to the MCP protocol from here on; errors go to stderr.
    problem = mcp_version_problem(_installed_version("mcp"))
    if problem:
        sys.exit(f"tfbs-mcp: {problem}")

    from tfbs.mcp.server import main as serve

    serve()


if __name__ == "__main__":
    main()
