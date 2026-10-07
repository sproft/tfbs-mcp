"""Tests for the tfbs-mcp launcher and its --check and --setup commands."""
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tfbs.mcp import cli

GENOME = cli.SETUP_VARIABLES[0]
CLINVAR = cli.SETUP_VARIABLES[2]
PROJECT_ROOT = cli.SETUP_VARIABLES[4]


@pytest.mark.parametrize("version", ["1.28.1", "1.99.0"])
def test_mcp_1x_is_accepted(version):
    assert cli.mcp_version_problem(version) is None


@pytest.mark.parametrize("version", ["2.0.0", "2.2.0"])
def test_mcp_2x_is_rejected_with_the_fix(version):
    problem = cli.mcp_version_problem(version)
    assert version in problem
    assert cli.FIX_MCP in problem


def test_missing_mcp_is_rejected_with_the_fix():
    assert cli.FIX_MCP in cli.mcp_version_problem(None)


def test_desktop_config_is_valid_json_and_keeps_windows_backslashes():
    exe = r"C:\Users\you\tfbs-env\Scripts\tfbs-mcp.exe"
    parsed = json.loads(cli.claude_desktop_config(exe))
    assert parsed["mcpServers"]["tfbs"]["command"] == exe


def test_check_passes_and_prints_both_setup_routes(capsys, monkeypatch):
    # The dev checkout may not be pip-installed, so there is no launcher to find.
    monkeypatch.setattr(cli, "find_executable", lambda: "/opt/env/bin/tfbs-mcp")
    assert cli.check() == 0
    out = capsys.readouterr().out
    assert "33 tools registered" in out
    assert "claude mcp add tfbs --scope user --" in out
    assert '"mcpServers"' in out


def test_launcher_refuses_mcp_2_before_importing_the_server(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tfbs-mcp"])
    monkeypatch.setattr(cli, "_installed_version", lambda pkg: "2.2.0")
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "mcp 2.2.0" in str(exc.value.code)


def test_unknown_argument_exits_with_usage(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tfbs-mcp", "--nope"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "usage: tfbs-mcp" in str(exc.value.code)


def test_usage_lists_check_and_setup():
    assert "--check" in cli.USAGE
    assert "--setup" in cli.USAGE
    assert "--non-interactive" in cli.USAGE


def test_cli_module_imports_neither_mcp_nor_torch_at_top_level():
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert not top_level & {"mcp", "torch", "tfbs"}


# ---------------------------------------------------------------------------
# --setup: path validation (pure functions, no prompts)
# ---------------------------------------------------------------------------

def test_existing_file_is_accepted_as_an_absolute_path(tmp_path):
    fasta = tmp_path / "hg38.fa"
    fasta.write_text(">chr1\nACGT\n")
    check = cli.validate_path(GENOME, str(fasta))
    assert check.error is None
    assert check.warning is None
    assert check.path == str(fasta.resolve())
    assert Path(check.path).is_absolute()


def test_missing_file_is_reported(tmp_path):
    missing = tmp_path / "nope.fa"
    check = cli.validate_path(GENOME, str(missing))
    assert check.path is None
    assert "does not exist" in check.error
    assert "nope.fa" in check.error


def test_directory_is_rejected_where_a_file_is_needed(tmp_path):
    check = cli.validate_path(GENOME, str(tmp_path))
    assert check.path is None
    assert "directory" in check.error


def test_file_is_rejected_where_a_directory_is_needed(tmp_path):
    some_file = tmp_path / "file.txt"
    some_file.write_text("x")
    check = cli.validate_path(PROJECT_ROOT, str(some_file))
    assert check.path is None
    assert "file" in check.error
    assert cli.validate_path(PROJECT_ROOT, str(tmp_path)).path == str(tmp_path.resolve())


def test_blank_value_is_not_a_path():
    assert cli.validate_path(GENOME, "   ").error


def test_clinvar_without_index_warns_but_is_accepted(tmp_path):
    vcf = tmp_path / "clinvar.vcf.gz"
    vcf.write_bytes(b"")
    check = cli.validate_path(CLINVAR, str(vcf))
    assert check.error is None
    assert check.path == str(vcf.resolve())
    assert ".tbi" in check.warning


def test_clinvar_with_index_has_no_warning(tmp_path):
    vcf = tmp_path / "clinvar.vcf.gz"
    vcf.write_bytes(b"")
    (tmp_path / "clinvar.vcf.gz.tbi").write_bytes(b"")
    check = cli.validate_path(CLINVAR, str(vcf))
    assert check.error is None
    assert check.warning is None


def test_prompt_repeats_until_a_path_exists_and_enter_skips(tmp_path, capsys):
    fasta = tmp_path / "hg38.fa"
    fasta.write_text(">chr1\nACGT\n")
    answers = iter([str(tmp_path / "missing.fa"), str(fasta)])
    assert cli.prompt_for_path(GENOME, ask=lambda prompt: next(answers)) == str(fasta.resolve())
    assert "not accepted" in capsys.readouterr().out
    assert cli.prompt_for_path(GENOME, ask=lambda prompt: "") is None

    def closed_stdin(prompt):
        raise EOFError

    assert cli.prompt_for_path(GENOME, ask=closed_stdin) is None


# ---------------------------------------------------------------------------
# --setup: rendering of both setup forms
# ---------------------------------------------------------------------------

def test_env_is_rendered_in_both_forms():
    exe = r"C:\Users\you\tfbs-env\Scripts\tfbs-mcp.exe"
    env = {"TFBS_GENOME_FASTA": r"D:\data\hg38.fa",
           "TFBS_MOTIF_FILE": r"D:\data\motifs.meme"}
    command = cli.claude_code_command(exe, env)
    assert command == (
        'claude mcp add tfbs --scope user'
        r' --env TFBS_GENOME_FASTA=D:\data\hg38.fa'
        r' TFBS_MOTIF_FILE=D:\data\motifs.meme'
        f' -- "{exe}"'
    )
    assert cli.claude_code_argv(exe, env) == [
        "claude", "mcp", "add", "tfbs", "--scope", "user",
        "--env", r"TFBS_GENOME_FASTA=D:\data\hg38.fa",
        r"TFBS_MOTIF_FILE=D:\data\motifs.meme",
        "--", exe,
    ]
    parsed = json.loads(cli.claude_desktop_config(exe, env))
    assert parsed["mcpServers"]["tfbs"]["command"] == exe
    assert parsed["mcpServers"]["tfbs"]["args"] == []
    assert parsed["mcpServers"]["tfbs"]["env"] == env


def test_paths_with_spaces_are_quoted_in_the_command_only():
    env = {"TFBS_GENOME_FASTA": r"C:\My Data\hg38.fa"}
    assert r'--env "TFBS_GENOME_FASTA=C:\My Data\hg38.fa"' in cli.claude_code_command("x", env)
    assert cli.claude_code_argv("x", env)[7] == r"TFBS_GENOME_FASTA=C:\My Data\hg38.fa"


def test_check_output_is_unchanged_when_env_is_none():
    exe = "/opt/env/bin/tfbs-mcp"
    assert cli.claude_code_command(exe) == 'claude mcp add tfbs --scope user -- "/opt/env/bin/tfbs-mcp"'
    assert cli.claude_code_command(exe, None) == cli.claude_code_command(exe)
    assert cli.claude_code_command(exe, {}) == cli.claude_code_command(exe)
    expected = json.dumps({"mcpServers": {"tfbs": {"command": exe, "args": []}}}, indent=2)
    assert cli.claude_desktop_config(exe) == expected
    assert cli.claude_desktop_config(exe, None) == expected
    assert cli.claude_desktop_config(exe, {}) == expected
    assert "env" not in json.loads(cli.claude_desktop_config(exe))["mcpServers"]["tfbs"]


# ---------------------------------------------------------------------------
# --setup: end to end without prompts or a real claude CLI
# ---------------------------------------------------------------------------

@pytest.fixture
def data_files(tmp_path):
    genome = tmp_path / "hg38.fa"
    genome.write_text(">chr1\nACGT\n")
    motif = tmp_path / "motifs.meme"
    motif.write_text("MEME version 4\n")
    return genome, motif


def _no_prompt(prompt):
    raise AssertionError(f"prompted unexpectedly: {prompt!r}")


def test_non_interactive_setup_prints_both_forms_with_env(data_files, capsys, monkeypatch):
    genome, motif = data_files
    exe = "/opt/env/bin/tfbs-mcp"
    monkeypatch.setattr(cli, "find_executable", lambda: exe)
    monkeypatch.setattr(cli, "find_claude", lambda: "/usr/bin/claude")
    monkeypatch.setattr("builtins.input", _no_prompt)
    monkeypatch.setattr(cli, "_run_claude", lambda argv: pytest.fail("claude was run"))
    monkeypatch.setattr(sys, "argv", ["tfbs-mcp", "--setup", "--non-interactive",
                                      "--genome", str(genome), "--motif", str(motif)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    genome_abs, motif_abs = str(genome.resolve()), str(motif.resolve())
    assert cli.claude_code_command(exe, {"TFBS_GENOME_FASTA": genome_abs,
                                         "TFBS_MOTIF_FILE": motif_abs}) in out
    block = out[out.index("{"):out.rindex("}") + 1]
    parsed = json.loads("\n".join(line.strip() for line in block.splitlines()))
    assert parsed["mcpServers"]["tfbs"]["env"] == {"TFBS_GENOME_FASTA": genome_abs,
                                                   "TFBS_MOTIF_FILE": motif_abs}
    assert "TFBS_CLINVAR_VCF" not in out
    assert "Register with Claude Code" not in out


def test_setup_rejects_a_bad_prefilled_flag(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "find_executable", lambda: "/opt/env/bin/tfbs-mcp")
    code = cli.setup({"TFBS_GENOME_FASTA": str(tmp_path / "missing.fa")},
                     interactive=False, ask=_no_prompt)
    assert code == 2
    err = capsys.readouterr().err
    assert "--genome" in err
    assert "does not exist" in err


def test_setup_without_an_executable_fails(capsys, monkeypatch):
    monkeypatch.setattr(cli, "find_executable", lambda: None)
    assert cli.setup({}, interactive=False, ask=_no_prompt) == 1
    assert "could not locate" in capsys.readouterr().err


def test_interactive_setup_prompts_only_for_unfilled_variables_and_registers(
        data_files, capsys, monkeypatch):
    genome, motif = data_files
    exe = "/opt/env/bin/tfbs-mcp"
    monkeypatch.setattr(cli, "find_executable", lambda: exe)
    monkeypatch.setattr(cli, "find_claude", lambda: "/usr/bin/claude")
    prompts = []
    answers = {"TFBS_MOTIF_FILE": str(motif), "Register": "y"}

    def ask(prompt):
        prompts.append(prompt)
        key = prompt.split()[0]
        return answers.get(key, "")

    calls = []

    def fake_run(argv):
        calls.append(argv)
        if argv[2] == "remove":
            return subprocess.CompletedProcess(argv, 1, "", 'No MCP server found with name: "tfbs"')
        return subprocess.CompletedProcess(argv, 0, "Added stdio MCP server tfbs\n", "")

    monkeypatch.setattr(cli, "_run_claude", fake_run)
    code = cli.setup({"TFBS_GENOME_FASTA": str(genome)}, interactive=True, ask=ask)
    assert code == 0
    asked = [p.split()[0] for p in prompts]
    assert asked == ["TFBS_MOTIF_FILE", "TFBS_CLINVAR_VCF", "TFBS_FOLDX_PDB",
                     "TFBS_PROJECT_ROOT", "Register"]
    assert [c[:3] for c in calls] == [["/usr/bin/claude", "mcp", "remove"],
                                      ["/usr/bin/claude", "mcp", "add"]]
    assert calls[0][3:] == ["tfbs", "--scope", "user"]
    assert calls[1][3:] == ["tfbs", "--scope", "user",
                            "--env", f"TFBS_GENOME_FASTA={genome.resolve()}",
                            f"TFBS_MOTIF_FILE={motif.resolve()}",
                            "--", exe]
    out = capsys.readouterr().out
    assert "Registered with Claude Code" in out
    assert "Added stdio MCP server tfbs" in out


def test_declining_registration_only_prints(data_files, capsys, monkeypatch):
    genome, _ = data_files
    monkeypatch.setattr(cli, "find_executable", lambda: "/opt/env/bin/tfbs-mcp")
    monkeypatch.setattr(cli, "find_claude", lambda: "/usr/bin/claude")
    monkeypatch.setattr(cli, "_run_claude", lambda argv: pytest.fail("claude was run"))
    answers = iter(["", "", "", "", "n"])
    code = cli.setup({"TFBS_GENOME_FASTA": str(genome)}, interactive=True,
                     ask=lambda prompt: next(answers))
    assert code == 0
    out = capsys.readouterr().out
    assert "Not registered" in out
    assert f"TFBS_GENOME_FASTA={genome.resolve()}" in out


def test_failed_registration_reports_the_error(data_files, capsys, monkeypatch):
    genome, _ = data_files
    monkeypatch.setattr(cli, "find_executable", lambda: "/opt/env/bin/tfbs-mcp")
    monkeypatch.setattr(cli, "find_claude", lambda: "/usr/bin/claude")
    monkeypatch.setattr(cli, "_run_claude", lambda argv: subprocess.CompletedProcess(
        argv, 1, "", "MCP server tfbs already exists in user config"))
    answers = iter(["", "", "", "", "y"])
    code = cli.setup({"TFBS_GENOME_FASTA": str(genome)}, interactive=True,
                     ask=lambda prompt: next(answers))
    assert code == 1
    err = capsys.readouterr().err
    assert "registration failed" in err
    assert "already exists" in err


def test_without_claude_on_path_setup_just_prints(data_files, capsys, monkeypatch):
    genome, _ = data_files
    monkeypatch.setattr(cli, "find_executable", lambda: "/opt/env/bin/tfbs-mcp")
    monkeypatch.setattr(cli, "find_claude", lambda: None)
    monkeypatch.setattr(cli, "_run_claude", lambda argv: pytest.fail("claude was run"))
    answers = iter(["", "", "", ""])
    code = cli.setup({"TFBS_GENOME_FASTA": str(genome)}, interactive=True,
                     ask=lambda prompt: next(answers))
    assert code == 0
    out = capsys.readouterr().out
    assert "Register with Claude Code" not in out
    assert "not on PATH" in out
    assert "claude mcp add tfbs --scope user --env" in out


def test_setup_help_lists_every_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.parse_setup_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--genome", "--motif", "--clinvar", "--foldx-pdb", "--project-root",
                 "--non-interactive"):
        assert flag in out
