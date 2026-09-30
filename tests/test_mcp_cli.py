"""Tests for the tfbs-mcp launcher and its --check command."""
import json
import sys

import pytest

from tfbs.mcp import cli


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
