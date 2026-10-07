"""Entry point for the ``tfbs-mcp`` command.

``tfbs-mcp``          start the server (your AI app does this, not you)
``tfbs-mcp --check``  verify the install and print the exact setup commands
``tfbs-mcp --setup``  ask once for the optional data paths and print (or
                      register) the setup commands with them set as
                      environment variables for the server

This module must not import ``mcp`` or ``torch`` at the top: its job is to
report a missing or incompatible ``mcp`` package clearly instead of with a
traceback.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, NamedTuple

MCP_REQUIREMENT = "mcp>=1.28,<2"
FIX_MCP = f'pip install "{MCP_REQUIREMENT}"'
SERVER_NAME = "tfbs"

USAGE = """\
usage: tfbs-mcp [--check | --setup [options]]

  (no arguments)  start the MCP server. Your AI app runs this for you after
                  setup; started by hand it waits silently for input (Ctrl+C
                  to quit).
  --check         verify the installation and print the command that
                  connects the server to Claude Code or Claude Desktop.
  --setup         ask once for the optional data paths (reference genome,
                  motif file, ClinVar VCF, FoldX PDB, project root) and print
                  the Claude Code command and the Claude Desktop JSON with
                  them set as environment variables. If the claude CLI is on
                  PATH it offers to register the server with Claude Code.
                  Options: --genome PATH, --motif PATH, --clinvar PATH,
                  --foldx-pdb PATH and --project-root PATH prefill a variable;
                  --non-interactive asks nothing and uses only the flags.
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


def find_claude() -> str | None:
    """Path of the claude CLI on PATH, or None when it is not installed."""
    return shutil.which("claude")


# ---------------------------------------------------------------------------
# Setup commands. --check prints them without environment variables; --setup
# fills in the data paths the server would otherwise ask for on every call.
# ---------------------------------------------------------------------------

def _quote(arg: str) -> str:
    """Quote one argument for display when a shell would otherwise split it."""
    if arg and not any(c in arg for c in " \t\"&|<>^;()$`'*?"):
        return arg
    return '"' + arg.replace('"', '\\"') + '"'


def claude_code_argv(executable: str, env: dict[str, str] | None = None) -> list[str]:
    """The ``claude mcp add`` call as an argument list (what --setup runs)."""
    argv = ["claude", "mcp", "add", SERVER_NAME, "--scope", "user"]
    if env:
        # The CLI documents one --env taking several KEY=value pairs. It has
        # to come after the server name, or the name is read as another pair.
        argv += ["--env", *[f"{key}={value}" for key, value in env.items()]]
    argv += ["--", executable]
    return argv


def claude_code_command(executable: str, env: dict[str, str] | None = None) -> str:
    """The same call as one line to paste into a terminal."""
    pairs = ""
    if env:
        pairs = " --env " + " ".join(_quote(f"{key}={value}") for key, value in env.items())
    return f'claude mcp add {SERVER_NAME} --scope user{pairs} -- "{executable}"'


def claude_desktop_config(executable: str, env: dict[str, str] | None = None) -> str:
    # json.dumps escapes Windows backslashes correctly, so the output can be
    # pasted as is.
    server: dict[str, object] = {"command": executable, "args": []}
    if env:
        server["env"] = dict(env)
    return json.dumps({"mcpServers": {SERVER_NAME: server}}, indent=2)


def print_setup_routes(executable: str, env: dict[str, str] | None = None) -> None:
    """Print the Claude Code command and the Claude Desktop block."""
    print("Claude Code: run this command once, then restart Claude Code.\n")
    print(f"    {claude_code_command(executable, env)}\n")
    print("Claude Desktop: open Settings > Developer > Edit Config, paste the block")
    print("below (merge it into mcpServers if the file already has one), save, then")
    print("quit Claude Desktop completely and reopen it.\n")
    for line in claude_desktop_config(executable, env).splitlines():
        print(f"    {line}")
    print()


# ---------------------------------------------------------------------------
# --setup: the optional data paths, collected once outside Claude.
# ---------------------------------------------------------------------------

class SetupVariable(NamedTuple):
    env: str    # environment variable the server reads
    flag: str   # --setup option that prefills it
    label: str  # what the path points at, shown in prompts and --help
    kind: str   # "file" or "dir"

    @property
    def dest(self) -> str:
        """Attribute name argparse stores the flag under."""
        return self.flag.lstrip("-").replace("-", "_")


SETUP_VARIABLES = (
    SetupVariable("TFBS_GENOME_FASTA", "--genome", "reference genome FASTA", "file"),
    SetupVariable("TFBS_MOTIF_FILE", "--motif", "MEME-format motif file", "file"),
    SetupVariable("TFBS_CLINVAR_VCF", "--clinvar", "indexed ClinVar VCF (.vcf.gz)", "file"),
    SetupVariable("TFBS_FOLDX_PDB", "--foldx-pdb", "repaired TF-DNA PDB", "file"),
    SetupVariable("TFBS_PROJECT_ROOT", "--project-root",
                  "directory holding data/, saved_models/, results/", "dir"),
)


class PathCheck(NamedTuple):
    """Outcome of ``validate_path``: exactly one of ``path`` and ``error`` is set."""
    path: str | None     # absolute path when the value was accepted
    error: str | None    # why it was rejected; the caller asks again
    warning: str | None  # accepted, but something is worth a mention


def validate_path(variable: SetupVariable, value: str) -> PathCheck:
    """Check one user-supplied path for ``variable`` without changing anything.

    Files must exist and be files; TFBS_PROJECT_ROOT must be a directory. The
    ClinVar VCF is accepted without its ``.tbi`` index, with a warning.
    Accepted paths come back absolute, so the server finds them whatever its
    working directory is.
    """
    text = value.strip()
    if not text:
        return PathCheck(None, "no path given", None)
    path = Path(text).expanduser()
    if not path.exists():
        return PathCheck(None, f"{text} does not exist", None)
    if variable.kind == "dir":
        if not path.is_dir():
            return PathCheck(None, f"{text} is a file, but a directory is needed", None)
    elif not path.is_file():
        return PathCheck(None, f"{text} is a directory, but a file is needed", None)
    resolved = str(path.resolve())
    warning = None
    if variable.env == "TFBS_CLINVAR_VCF" and not Path(resolved + ".tbi").is_file():
        warning = (f"no index found next to it (expected {resolved}.tbi); the server "
                   "expects an indexed .vcf.gz")
    return PathCheck(resolved, None, warning)


def prompt_for_path(variable: SetupVariable,
                    ask: Callable[[str], str] = input) -> str | None:
    """Ask for one path until it validates; Enter (or a closed stdin) skips it."""
    prompt = f"{variable.env}  {variable.label} (Enter to skip): "
    while True:
        try:
            answer = ask(prompt)
        except EOFError:
            return None
        if not answer.strip():
            return None
        check = validate_path(variable, answer)
        if check.error:
            print(f"  not accepted: {check.error}")
            continue
        if check.warning:
            print(f"  warning: {check.warning}")
        return check.path


def _run_claude(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True)


def register_with_claude_code(executable: str, env: dict[str, str],
                              claude: str) -> tuple[bool, str]:
    """Replace the user-scope ``tfbs`` entry in Claude Code with one built from ``env``.

    Returns ``(ok, message)`` where ``message`` is the CLI's own output on
    success and the reason on failure.
    """
    try:
        # `claude mcp add` refuses a name that already exists at the same
        # scope, so drop any old entry first. A failure here only means there
        # was none to remove.
        _run_claude([claude, "mcp", "remove", SERVER_NAME, "--scope", "user"])
        argv = claude_code_argv(executable, env)
        result = _run_claude([claude, *argv[1:]])
    except OSError as exc:
        return False, f"could not run {claude}: {exc}"
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        detail = stderr or stdout
        return False, (f"claude mcp add exited with status {result.returncode}"
                       + (f": {detail}" if detail else ""))
    return True, stdout


def setup(prefilled: dict[str, str], interactive: bool = True,
          ask: Callable[[str], str] = input) -> int:
    """Collect the data paths, print both setup forms, offer to register.

    ``prefilled`` maps environment variable names to paths given as flags;
    those are validated and not prompted for. Returns an exit code.
    """
    executable = find_executable()
    if executable is None:
        print("tfbs-mcp: could not locate the tfbs-mcp executable in this environment; "
              "run `tfbs-mcp --check` for details.", file=sys.stderr)
        return 1

    env: dict[str, str] = {}
    for variable in SETUP_VARIABLES:
        given = prefilled.get(variable.env)
        if given is None:
            continue
        check = validate_path(variable, given)
        if check.error:
            print(f"tfbs-mcp: {variable.flag}: {check.error}", file=sys.stderr)
            return 2
        if check.warning:
            print(f"{variable.env}: warning: {check.warning}")
        env[variable.env] = check.path

    if interactive:
        print("tfbs-mcp setup\n")
        print("Enter the path for each item, or press Enter to skip it. Everything here")
        print("is optional: a skipped item can still be passed to a tool as an argument.\n")
        for variable in SETUP_VARIABLES:
            if variable.env in prefilled:
                continue
            value = prompt_for_path(variable, ask)
            if value is not None:
                env[variable.env] = value
        print()

    # Keep the variables in SETUP_VARIABLES order whatever order they came in.
    env = {v.env: env[v.env] for v in SETUP_VARIABLES if v.env in env}

    if env:
        print("Environment variables for the server:\n")
        for key, value in env.items():
            print(f"    {key}={value}")
        print()
    else:
        print("No paths were set; the commands below register the server without")
        print("environment variables.\n")
    print_setup_routes(executable, env)

    if not interactive:
        return 0
    claude = find_claude()
    if claude is None:
        print("The claude CLI is not on PATH, so nothing was registered automatically.")
        print("Run the Claude Code command above in a terminal where it is, or use the")
        print("Claude Desktop block.")
        return 0
    try:
        answer = ask("Register with Claude Code now? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        print("Not registered. Run the Claude Code command above when you are ready.")
        return 0
    ok, message = register_with_claude_code(executable, env, claude)
    if ok:
        print("Registered with Claude Code." if not message else "Registered with Claude Code:")
        for line in message.splitlines():
            print(f"    {line}")
        print("Restart Claude Code; `claude mcp get tfbs` shows the saved entry.")
        return 0
    print(f"tfbs-mcp: registration failed: {message}", file=sys.stderr)
    print("Run the Claude Code command above by hand to see the full error.",
          file=sys.stderr)
    return 1


def parse_setup_args(args: list[str]) -> argparse.Namespace:
    """Parse the options that may follow ``--setup``."""
    parser = argparse.ArgumentParser(
        prog="tfbs-mcp --setup",
        description="Collect the optional data paths once and print (or register) the "
                    "Claude setup commands with them set as environment variables. "
                    "Every path is validated; a variable given as a flag is not "
                    "prompted for.")
    for variable in SETUP_VARIABLES:
        parser.add_argument(variable.flag, dest=variable.dest, metavar="PATH",
                            help=f"{variable.env}: {variable.label}")
    parser.add_argument("--non-interactive", action="store_true",
                        help="never prompt: use only the flags above and only print "
                             "the commands")
    return parser.parse_args(args)


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
    print_setup_routes(executable)
    return 0


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] in ("-h", "--help"):
        print(USAGE)
        return
    if args and args[0] == "--check":
        sys.exit(check())
    if args and args[0] == "--setup":
        options = parse_setup_args(args[1:])
        prefilled = {v.env: getattr(options, v.dest) for v in SETUP_VARIABLES
                     if getattr(options, v.dest) is not None}
        sys.exit(setup(prefilled, interactive=not options.non_interactive))
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
