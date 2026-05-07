"""Tests for FoldX MCP tools."""
import asyncio
import json
import os
import shutil
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _foldx_installed() -> bool:
    return shutil.which("foldx") is not None or shutil.which("foldx5") is not None


def _make_temp_pdb(tmpdir: str) -> str:
    """Write a minimal placeholder PDB file and return its path."""
    pdb_path = os.path.join(tmpdir, "test.pdb")
    with open(pdb_path, "w") as f:
        f.write(
            "HEADER    TEST STRUCTURE\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            "END\n"
        )
    return pdb_path


# ---------------------------------------------------------------------------
# tfbs_fetch_structure
# ---------------------------------------------------------------------------

def test_fetch_structure_local_pdb():
    """Create a temp PDB and verify source='local'."""
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = _make_temp_pdb(tmpdir)
        result = json.loads(asyncio.run(tfbs_fetch_structure(local_path=pdb_path)))
        assert result["source"] == "local"
        assert os.path.isfile(result["structure_path"])


def test_fetch_structure_missing_local():
    """Verify error JSON when local file does not exist."""
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    result = json.loads(asyncio.run(tfbs_fetch_structure(local_path="/nonexistent/path.pdb")))
    assert "error" in result


def test_fetch_structure_pdb_id():
    """Download 3RKQ from RCSB and verify the file exists (needs internet)."""
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    with tempfile.TemporaryDirectory() as tmpdir:
        result = json.loads(asyncio.run(tfbs_fetch_structure(pdb_id="3RKQ", output_dir=tmpdir)))
        assert result.get("source") == "pdb", f"Unexpected result: {result}"
        assert os.path.isfile(result["structure_path"])
        assert os.path.getsize(result["structure_path"]) > 0


# ---------------------------------------------------------------------------
# tfbs_binding_energy
# ---------------------------------------------------------------------------

def test_binding_energy_no_foldx():
    """If FoldX is NOT installed, verify the error message."""
    if _foldx_installed():
        pytest.skip("FoldX is installed; skipping no-foldx test")

    from tfbs.mcp.foldx_tools import tfbs_binding_energy

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = _make_temp_pdb(tmpdir)
        result = json.loads(asyncio.run(tfbs_binding_energy(structure_path=pdb_path)))
        assert "error" in result
        assert "foldx" in result["error"].lower() or "FoldX" in result["error"]


def test_binding_energy_missing_structure():
    """Verify error JSON when structure file does not exist."""
    from tfbs.mcp.foldx_tools import tfbs_binding_energy

    result = json.loads(asyncio.run(tfbs_binding_energy(structure_path="/nonexistent/file.pdb")))
    assert "error" in result


# ---------------------------------------------------------------------------
# tfbs_binding_scan
# ---------------------------------------------------------------------------

def test_binding_scan_missing_inputs():
    """Verify error for nonexistent structure."""
    from tfbs.mcp.foldx_tools import tfbs_binding_scan

    result = json.loads(asyncio.run(tfbs_binding_scan(
        structure_path="/nonexistent/structure.pdb",
        genome_fasta="/some/genome.fa",
        region="chr1:1000-2000",
    )))
    assert "error" in result


def test_binding_scan_no_foldx():
    """If FoldX is NOT installed, verify the error message."""
    if _foldx_installed():
        pytest.skip("FoldX is installed; skipping no-foldx test")

    from tfbs.mcp.foldx_tools import tfbs_binding_scan

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = _make_temp_pdb(tmpdir)
        # Create a dummy fasta file
        fasta_path = os.path.join(tmpdir, "genome.fa")
        with open(fasta_path, "w") as f:
            f.write(">chr1\n" + "A" * 5000 + "\n")

        result = json.loads(asyncio.run(tfbs_binding_scan(
            structure_path=pdb_path,
            genome_fasta=fasta_path,
            region="chr1:100-1000",
        )))
        assert "error" in result
        assert "foldx" in result["error"].lower() or "FoldX" in result["error"]
