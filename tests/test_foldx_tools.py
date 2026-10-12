"""Tests for FoldX MCP tools."""
import asyncio
import io
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import types
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import pytest

with warnings.catch_warnings():
    # Registering the marker would need pyproject.toml, which these tests leave alone.
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    network = pytest.mark.network


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _foldx_installed() -> bool:
    return shutil.which("foldx") is not None or shutil.which("foldx5") is not None


def _run(coro) -> dict:
    return json.loads(asyncio.run(coro))


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


def _pdb_atom(serial: int, name: str, resname: str, chain: str, resnum: int) -> str:
    return (f"ATOM  {serial:5d} {name:<4} {resname:>3} {chain}{resnum:4d}    "
            f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00")


def _pdb_text(dna: str | None = None) -> str:
    """A protein residue on chain A, plus DNA on chains C and D if ``dna`` is given."""
    lines = [_pdb_atom(1, "CA", "ALA", "A", 1)]
    if dna:
        partner = dna.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        for chain, seq in (("C", dna), ("D", partner)):
            for resnum, base in enumerate(seq, start=1):
                lines.append(_pdb_atom(len(lines) + 1, "P", f"D{base}", chain, resnum))
    return "\n".join(lines) + "\nEND\n"


def _make_dna_pdb(directory: str, dna: str = "ACGTA") -> str:
    path = os.path.join(directory, "complex.pdb")
    with open(path, "w") as f:
        f.write(_pdb_text(dna))
    return path


def _make_genome(directory: str, length: int = 6000, seed: int = 7) -> tuple[str, str]:
    rng = random.Random(seed)
    seq = "".join(rng.choice("ACGT") for _ in range(length))
    path = os.path.join(directory, "genome.fa")
    with open(path, "w", newline="\n") as f:
        f.write(">chr1\n")
        for i in range(0, length, 60):
            f.write(seq[i:i + 60] + "\n")
    return path, seq


def _toy_energy(seq: str) -> float:
    return -(2 * seq.count("G") + seq.count("C")) / len(seq)


@pytest.fixture
def network_opt_in(request):
    """Live tests run only with ``-m network`` or TFBS_NETWORK_TESTS=1."""
    selected = "network" in (request.config.getoption("markexpr") or "")
    if not selected and os.environ.get("TFBS_NETWORK_TESTS") != "1":
        pytest.skip("live network test; run with -m network or TFBS_NETWORK_TESTS=1")


class _FakeBigWig:
    def __init__(self, path, registry):
        self.path, self.registry = path, registry
        self.header, self.entries = None, []

    def addHeader(self, header):
        self.header = dict(header)

    def addEntries(self, chroms, starts, ends=None, values=None):
        for chrom, start, end, value in zip(chroms, starts, ends, values):
            last_end = self.entries[-1][2] if self.entries else 0
            assert start >= last_end and end > start, "bigWig entries must be sorted and non-overlapping"
            assert end <= self.header[chrom]
            self.entries.append((chrom, start, end, value))

    def close(self):
        Path(self.path).write_bytes(b"")
        self.registry[str(Path(self.path).resolve())] = self


@pytest.fixture
def fake_bigwig(monkeypatch):
    """pyBigWig stand-in that enforces the bigWig entry rules; maps path to file."""
    written: dict = {}
    module = types.ModuleType("pyBigWig")
    module.open = lambda path, mode="r": _FakeBigWig(path, written)
    monkeypatch.setitem(sys.modules, "pyBigWig", module)
    return written


@pytest.fixture
def fake_foldx(monkeypatch):
    """FoldX stand-in: a placement's energy is _toy_energy of its sequence."""
    import tfbs.mcp.foldx_tools as fx

    placements: list[str] = []

    def run(foldx_bin, structure, working_dir, mutation_string=None):
        placements.append(mutation_string)
        return _toy_energy(mutation_string)

    monkeypatch.setattr(fx, "_find_foldx", lambda: "foldx")
    monkeypatch.setattr(fx, "_mutation_string", lambda template, seq: seq)
    monkeypatch.setattr(fx, "_run_foldx_energy", run)
    return placements


@pytest.fixture
def scan_inputs(tmp_path):
    pdb = _make_dna_pdb(str(tmp_path))
    fasta, seq = _make_genome(str(tmp_path))
    return pdb, fasta, seq


def _scan(scan_inputs, output_dir, **kwargs) -> dict:
    from tfbs.mcp.foldx_tools import tfbs_binding_scan

    pdb, fasta, _ = scan_inputs
    args = {"structure_path": pdb, "genome_fasta": fasta, "region": "chr1:100-1100",
            "output_dir": str(output_dir)}
    args.update(kwargs)
    return _run(tfbs_binding_scan(**args))


@pytest.fixture
def fake_sbatch(monkeypatch):
    """Replace slurm_submit; append the JSON replies it should give to ``replies``."""
    import tfbs.mcp.server as server

    calls: list[dict] = []
    replies: list[dict] = []

    async def submit(script_path, args=None, sbatch_flags=None, working_dir=None):
        calls.append({"script": script_path, "flags": sbatch_flags})
        return json.dumps(replies.pop(0))

    monkeypatch.setattr(server, "slurm_submit", submit)
    return calls, replies


_AF_API = "https://alphafold.ebi.ac.uk/api/prediction/P43699"
_AF_MODEL = "https://alphafold.ebi.ac.uk/files/AF-P43699-F1-model_v6.pdb"
_AF_ENTRIES = [
    # The API lists isoforms too; the isoform comes first to test the exact match.
    {"entryId": "AF-P43699-3-F1", "modelEntityId": "AF-P43699-3-F1",
     "uniprotAccession": "P43699-3", "latestVersion": 6, "globalMetricValue": 54.69,
     "entityType": "protein",
     "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P43699-3-F1-model_v6.pdb"},
    {"entryId": "AF-P43699-F1", "modelEntityId": "AF-P43699-F1",
     "uniprotAccession": "P43699", "latestVersion": 6, "globalMetricValue": 56.31,
     "entityType": "protein", "pdbUrl": _AF_MODEL},
]
_RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"


def _rcsb(pdb_id: str) -> str:
    return f"https://files.rcsb.org/download/{pdb_id}.pdb"


def _http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "error", {}, io.BytesIO(b"{}"))


class _FakeWeb:
    """urlopen stand-in serving canned bodies by URL and recording each request."""

    def __init__(self, routes):
        self.routes = routes
        self.requested: list[str] = []

    def __call__(self, req, timeout=None, **kwargs):
        url = req.full_url if isinstance(req, urllib.request.Request) else req
        self.requested.append(url)
        reply = self.routes.get(url)
        if reply is None:
            raise urllib.error.URLError(f"no route to {url}")
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(reply)


@pytest.fixture
def fake_web(monkeypatch):
    def install(routes):
        web = _FakeWeb(routes)
        monkeypatch.setattr(urllib.request, "urlopen", web)
        return web
    return install


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
        assert result["argument"] == "local_path"
        assert result["attempted"] == []
        assert result["metadata"]["has_dna"] is False
        assert os.path.isfile(result["structure_path"])


def test_fetch_structure_missing_local():
    """Verify error JSON when local file does not exist."""
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    result = json.loads(asyncio.run(tfbs_fetch_structure(local_path="/nonexistent/path.pdb")))
    assert "error" in result
    assert result["fix"]
    assert [a["argument"] for a in result["attempted"]] == ["local_path"]


def test_fetch_structure_requires_a_source():
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    for kwargs in ({}, {"pdb_id": "", "uniprot_id": "  "}):
        result = _run(tfbs_fetch_structure(**kwargs))
        assert "error" in result and "uniprot_id" in result["fix"]


def test_fetch_structure_alphafold_uses_api_model(fake_web, tmp_path, capsys):
    """The AlphaFold DB leg downloads the pdbUrl the API returns, not a fixed version."""
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    web = fake_web({_AF_API: json.dumps(_AF_ENTRIES).encode(), _AF_MODEL: _pdb_text().encode()})
    result = _run(tfbs_fetch_structure(uniprot_id="P43699", output_dir=str(tmp_path)))

    assert result["source"] == "alphafold"
    assert result["argument"] == "uniprot_id"
    assert result["attempted"] == []
    meta = result["metadata"]
    assert meta["latest_version"] == 6
    assert meta["mean_plddt"] == 56.31
    assert meta["url"] == _AF_MODEL
    assert meta["entry_id"] == "AF-P43699-F1"
    assert meta["has_dna"] is False
    assert "protein-only" in result["note"]
    assert Path(result["structure_path"]).name == "AF-P43699-F1-model_v6.pdb"
    assert Path(result["structure_path"]).is_file()
    assert web.requested == [_AF_API, _AF_MODEL]
    assert capsys.readouterr().out == ""


def test_fetch_structure_alphafold_missing_entry(fake_web, tmp_path):
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    api = "https://alphafold.ebi.ac.uk/api/prediction/Q00000"
    fake_web({api: _http_error(api, 404)})
    result = _run(tfbs_fetch_structure(uniprot_id="Q00000", output_dir=str(tmp_path)))

    assert "no entry" in result["error"] and "404" in result["error"]
    assert "UniProt accession" in result["fix"]
    assert [a["argument"] for a in result["attempted"]] == ["uniprot_id"]


def test_fetch_structure_falls_through_in_order(fake_web, tmp_path):
    """Each failing source hands over to the next, in the documented order."""
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    web = fake_web({
        _rcsb("9ZZZ"): _http_error(_rcsb("9ZZZ"), 404),
        _RCSB_SEARCH: b"",
        _AF_API: json.dumps(_AF_ENTRIES).encode(),
        _AF_MODEL: _pdb_text().encode(),
    })
    result = _run(tfbs_fetch_structure(
        local_path=str(tmp_path / "missing.pdb"),
        pdb_id="9zzz",
        gene_name="NOTAGENE",
        uniprot_id="P43699",
        output_dir=str(tmp_path),
    ))

    assert result["source"] == "alphafold"
    assert result["argument"] == "uniprot_id"
    assert [a["argument"] for a in result["attempted"]] == ["local_path", "pdb_id", "gene_name"]
    assert all(a["error"] and a["fix"] for a in result["attempted"])
    assert web.requested == [_rcsb("9ZZZ"), _RCSB_SEARCH, _AF_API, _AF_MODEL]


def test_fetch_structure_stops_at_first_success(fake_web, tmp_path):
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    web = fake_web({_rcsb("3RKQ"): _pdb_text("ACGTA").encode()})
    result = _run(tfbs_fetch_structure(
        local_path=str(tmp_path / "missing.pdb"),
        pdb_id="3RKQ",
        uniprot_id="P43699",
        output_dir=str(tmp_path),
    ))

    assert result["source"] == "pdb"
    assert result["argument"] == "pdb_id"
    assert [a["argument"] for a in result["attempted"]] == ["local_path"]
    assert result["metadata"]["has_dna"] is True
    assert "note" not in result
    assert web.requested == [_rcsb("3RKQ")]


def test_fetch_structure_gene_search(fake_web, tmp_path):
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    fake_web({
        _RCSB_SEARCH: json.dumps({"result_set": [{"identifier": "3RKQ"}]}).encode(),
        _rcsb("3RKQ"): _pdb_text("ACGTA").encode(),
    })
    result = _run(tfbs_fetch_structure(gene_name="NKX2-1", output_dir=str(tmp_path)))

    assert result["source"] == "pdb"
    assert result["argument"] == "gene_name"
    assert result["metadata"]["pdb_id"] == "3RKQ"
    assert result["metadata"]["gene_name"] == "NKX2-1"


def test_fetch_structure_all_sources_fail(fake_web, tmp_path):
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    fake_web({_AF_API: _http_error(_AF_API, 404)})
    result = _run(tfbs_fetch_structure(
        local_path=str(tmp_path / "missing.pdb"), uniprot_id="P43699", output_dir=str(tmp_path),
    ))

    assert "local_path, uniprot_id" in result["error"]
    assert result["fix"]
    assert [a["argument"] for a in result["attempted"]] == ["local_path", "uniprot_id"]


@network
def test_fetch_structure_pdb_id(network_opt_in):
    """Download 3RKQ from RCSB and verify the file exists (needs internet)."""
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    with tempfile.TemporaryDirectory() as tmpdir:
        result = json.loads(asyncio.run(tfbs_fetch_structure(pdb_id="3RKQ", output_dir=tmpdir)))
        assert result.get("source") == "pdb", f"Unexpected result: {result}"
        assert os.path.isfile(result["structure_path"])
        assert os.path.getsize(result["structure_path"]) > 0


@network
def test_fetch_structure_alphafold_live(network_opt_in, tmp_path):
    """Fetch the NKX2-1 model from the live AlphaFold DB (needs internet)."""
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    result = _run(tfbs_fetch_structure(uniprot_id="P43699", output_dir=str(tmp_path)))
    assert result.get("source") == "alphafold", f"Unexpected result: {result}"
    meta = result["metadata"]
    assert meta["latest_version"] >= 6
    assert 0 < meta["mean_plddt"] <= 100
    assert meta["has_dna"] is False
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
        assert "PATH" in result["fix"]


def test_binding_energy_missing_structure(monkeypatch):
    """Verify error JSON when structure file does not exist."""
    import tfbs.mcp.foldx_tools as fx

    monkeypatch.setattr(fx, "_find_foldx", lambda: "foldx")
    result = _run(fx.tfbs_binding_energy(structure_path="/nonexistent/file.pdb"))
    assert "not found" in result["error"]
    assert result["fix"]


def test_binding_energy_timeout_returns_error(monkeypatch, tmp_path):
    import tfbs.mcp.foldx_tools as fx

    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 600))

    monkeypatch.setattr(fx, "_find_foldx", lambda: "foldx")
    monkeypatch.setattr(subprocess, "run", timeout)
    result = _run(fx.tfbs_binding_energy(structure_path=_make_dna_pdb(str(tmp_path))))
    assert "RepairPDB timed out" in result["error"]
    assert result["fix"]


_AC_HEADER = "Pdb\tGroup1\tGroup2\tIntraclashesGroup1\tIntraclashesGroup2\tInteraction Energy\n"


@pytest.fixture
def foldx_cli(fake_foldx, monkeypatch):
    """subprocess.run stand-in for the FoldX commands tfbs_binding_energy runs.

    RepairPDB writes <stem>_Repair.pdb unless ``state["repair_writes"]`` is
    False; AnalyseComplex writes an Interaction_*_AC.fxout of -12.5 kcal/mol.
    """
    state = {"repair_writes": True, "commands": []}

    def run(cmd, cwd=None, **kwargs):
        wd = Path(cwd)
        pdb = next(a.split("=", 1)[1] for a in cmd if a.startswith("--pdb="))
        command = next(a.split("=", 1)[1] for a in cmd if a.startswith("--command="))
        state["commands"].append((command, pdb))
        if command == "RepairPDB" and state["repair_writes"]:
            (wd / f"{Path(pdb).stem}_Repair.pdb").write_text(
                (wd / pdb).read_text() + "REMARK repaired\n", newline="\n")
        elif command == "AnalyseComplex":
            (wd / f"Interaction_{Path(pdb).stem}_AC.fxout").write_text(
                _AC_HEADER + f"{pdb}\tA\tC,D\t0\t0\t-12.5\n", newline="\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="no ATOM records")

    monkeypatch.setattr(subprocess, "run", run)
    return state


def test_binding_energy_keeps_the_repaired_structure(foldx_cli, tmp_path, monkeypatch):
    from tfbs.mcp.foldx_tools import tfbs_binding_energy

    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    local = tmp_path / "inputs"
    local.mkdir()
    fetched = tmp_path / "results" / "structures" / "NKX2-1"
    fetched.mkdir(parents=True)
    shutil.copy(_make_dna_pdb(str(local)), fetched / "9U19.pdb")
    for structure, expected in (
            (local / "complex.pdb", tmp_path / "results" / "structures" / "complex"),
            (fetched / "9U19.pdb", fetched)):
        result = _run(tfbs_binding_energy(structure_path=str(structure)))
        repaired = Path(result["repaired_structure_path"])
        assert repaired == expected / f"{structure.stem}_Repair.pdb"
        assert repaired.is_file() and "REMARK repaired" in repaired.read_text()
        assert result["structure_path"] == str(structure.resolve())
        assert result["interaction_energy"] == -12.5
        assert foldx_cli["commands"][-1] == ("AnalyseComplex", repaired.name)

    out = tmp_path / "chosen"
    result = _run(tfbs_binding_energy(structure_path=str(local / "complex.pdb"),
                                      output_dir=str(out)))
    assert result["repaired_structure_path"] == str(out.resolve() / "complex_Repair.pdb")
    assert Path(result["repaired_structure_path"]).is_file()


def test_binding_energy_without_repair_names_no_repaired_file(foldx_cli, tmp_path, monkeypatch):
    from tfbs.mcp.foldx_tools import tfbs_binding_energy

    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    result = _run(tfbs_binding_energy(structure_path=_make_dna_pdb(str(tmp_path)), repair=False))
    assert result["repaired_structure_path"] is None
    assert [c for c, _ in foldx_cli["commands"]] == ["AnalyseComplex"]
    assert not (tmp_path / "results").exists()


def test_binding_energy_reports_a_repair_that_wrote_nothing(foldx_cli, tmp_path, monkeypatch):
    from tfbs.mcp.foldx_tools import tfbs_binding_energy

    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    foldx_cli["repair_writes"] = False
    result = _run(tfbs_binding_energy(structure_path=_make_dna_pdb(str(tmp_path))))
    assert result["error"] == "FoldX RepairPDB wrote no repaired structure"
    assert result["fix"] and result["stderr"] == "no ATOM records"
    assert [c for c, _ in foldx_cli["commands"]] == ["RepairPDB"]


def test_foldx_descriptions_hand_the_repaired_structure_on():
    from tfbs.mcp.foldx_tools import tfbs_binding_energy, tfbs_binding_scan

    energy = " ".join(tfbs_binding_energy.__doc__.split())
    assert "repaired_structure_path" in energy and "<results>/structures/<id>/" in energy
    for consumer in ("tfbs_binding_scan", "tfbs_chipseq_method_comparison",
                     "tfbs_remap_validation", "tfbs_disease_variant_validation",
                     "$TFBS_FOLDX_PDB"):
        assert consumer in energy, consumer
    scan = " ".join(tfbs_binding_scan.__doc__.split())
    assert "repaired_structure_path that tfbs_binding_energy returns with repair=true" in scan


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
    assert result["fix"]


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
        assert result["fix"]


def test_binding_scan_needs_dna_in_structure(fake_foldx, fake_bigwig, tmp_path):
    from tfbs.mcp.foldx_tools import tfbs_binding_scan

    fasta, _ = _make_genome(str(tmp_path))
    result = _run(tfbs_binding_scan(
        structure_path=_make_temp_pdb(str(tmp_path)), genome_fasta=fasta, region="chr1:100-1000",
    ))
    assert result["error"] == "No DNA chains found in the PDB structure."
    assert "AlphaFold" in result["fix"]


def test_binding_scan_reports_missing_pybigwig(scan_inputs, fake_foldx, monkeypatch, tmp_path):
    """The bigWig dependency is checked before any FoldX run."""
    monkeypatch.setitem(sys.modules, "pyBigWig", None)
    result = _scan(scan_inputs, tmp_path / "out")
    assert "pyBigWig" in result["error"]
    assert "foldx" in result["fix"]
    assert fake_foldx == []


def test_binding_scan_unknown_chromosome(scan_inputs, fake_foldx, fake_bigwig, tmp_path):
    result = _scan(scan_inputs, tmp_path / "out", region="chr2:100-1000")
    assert "chr2" in result["error"] and result["fix"]


def test_binding_scan_default_window_is_dna_length(scan_inputs, fake_foldx, fake_bigwig, tmp_path):
    _, _, seq = scan_inputs
    result = _scan(scan_inputs, tmp_path / "out")

    assert result["mode"] == "local"
    assert (result["window"], result["dna_length"], result["step"]) == (5, 5, 50)
    assert result["num_windows"] == (1100 - 100 - 5) // 50 + 1
    starts = [100 + 50 * i for i in range(result["num_windows"])]
    bw = fake_bigwig[result["bigwig_path"]]
    assert [(s, e) for _, s, e, _ in bw.entries] == [(s, s + 5) for s in starts]
    assert [v for *_, v in bw.entries] == pytest.approx([_toy_energy(seq[s:s + 5]) for s in starts])
    assert result["foldx_runs"] == len(starts) == result["num_scored"]


def test_binding_scan_explicit_window_scores_best_placement(scan_inputs, fake_foldx, fake_bigwig, tmp_path):
    """An explicit window is honoured: its value is the best placement inside it."""
    _, _, seq = scan_inputs
    result = _scan(scan_inputs, tmp_path / "out", region="chr1:200-260", window=8, step=2)

    assert (result["window"], result["dna_length"], result["step"]) == (8, 5, 2)
    starts = [200 + 2 * i for i in range(result["num_windows"])]
    assert starts[-1] == 252
    expected = [min(_toy_energy(seq[p:p + 5]) for p in range(s, s + 4)) for s in starts]
    bw = fake_bigwig[result["bigwig_path"]]
    assert [v for *_, v in bw.entries] == pytest.approx(expected)
    assert [(s, e) for _, s, e, _ in bw.entries] == [(s + 3, s + 5) for s in starts]
    unique_placements = starts[-1] + 3 - 200 + 1
    assert len(fake_foldx) == result["foldx_runs"] == unique_placements


def test_binding_scan_rejects_window_below_dna_length(scan_inputs, fake_foldx, fake_bigwig, tmp_path):
    result = _scan(scan_inputs, tmp_path / "out", window=3)
    assert "shorter than the DNA" in result["error"]
    assert "window >= 5" in result["fix"]
    assert fake_foldx == []


def test_binding_scan_logs_progress_without_stdout(scan_inputs, fake_foldx, fake_bigwig, tmp_path,
                                                   capsys, caplog):
    """Progress goes to logging; stdout (the MCP stdio channel) stays empty."""
    caplog.set_level(logging.INFO, logger="tfbs.mcp.foldx_tools")
    result = _scan(scan_inputs, tmp_path / "out", region="chr1:0-5500")

    assert result["num_windows"] == 110
    assert capsys.readouterr().out == ""
    assert any("100/110 windows processed" in r.getMessage() for r in caplog.records)


def test_foldx_tools_source_has_no_print():
    import tfbs.mcp.foldx_tools as fx

    assert "print(" not in Path(fx.__file__).read_text(encoding="utf-8")


def test_mutation_string_threads_both_strands(tmp_path):
    from tfbs.mcp.foldx_tools import _dna_template, _mutation_string

    template = _dna_template(_make_dna_pdb(str(tmp_path), "ACGTA"))
    assert (template.fwd_chain, template.rev_chain) == ("C", "D")
    assert _mutation_string(template, "ACGTA") is None
    assert _mutation_string(template, "ACGTT") == "aC5t,tD1a;"


def test_binding_scan_slurm_reports_missing_sbatch(scan_inputs, fake_foldx, fake_bigwig, tmp_path,
                                                   monkeypatch):
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which",
                        lambda cmd, *a, **k: None if cmd == "sbatch" else real_which(cmd, *a, **k))
    result = _scan(scan_inputs, tmp_path / "out", slurm=True)

    assert "sbatch" in result["error"]
    assert "slurm=false" in result["fix"]
    assert result["stage"] == "array"
    assert "array_job_id" not in result


def test_binding_scan_slurm_reports_failed_array_submit(scan_inputs, fake_foldx, fake_bigwig,
                                                        fake_sbatch, tmp_path):
    calls, replies = fake_sbatch
    replies.append({"error": "sbatch failed (exit 1)", "stderr": "invalid partition"})
    result = _scan(scan_inputs, tmp_path / "out", slurm=True)

    assert "sbatch failed" in result["error"] and result["fix"]
    assert result["stderr"] == "invalid partition"
    assert len(calls) == 1


def test_binding_scan_slurm_reports_missing_job_id(scan_inputs, fake_foldx, fake_bigwig,
                                                   fake_sbatch, tmp_path):
    calls, replies = fake_sbatch
    replies.append({"job_id": None, "submit_command": "sbatch foldx_scan.sh"})
    result = _scan(scan_inputs, tmp_path / "out", slurm=True)

    assert "no job ID" in result["error"] and result["fix"]
    assert len(calls) == 1


def test_binding_scan_slurm_reports_failed_merge_submit(scan_inputs, fake_foldx, fake_bigwig,
                                                        fake_sbatch, tmp_path):
    calls, replies = fake_sbatch
    replies += [{"job_id": "4242"}, {"error": "sbatch failed (exit 1)", "stderr": "bad dependency"}]
    result = _scan(scan_inputs, tmp_path / "out", slurm=True)

    assert "4242" in result["error"] and "merge" in result["error"]
    assert "--dependency=afterany:4242" in result["fix"]
    assert result["array_job_id"] == "4242"
    assert result["stage"] == "merge"
    assert calls[1]["flags"] == ["--dependency=afterany:4242"]


def test_binding_scan_slurm_jobs_match_local_scan(scan_inputs, fake_foldx, fake_bigwig, fake_sbatch,
                                                  tmp_path, monkeypatch):
    """The array and merge jobs score windows exactly as the local scan does."""
    import tfbs.mcp.foldx_tools as fx

    calls, replies = fake_sbatch
    replies += [{"job_id": "11"}, {"job_id": "12"}]
    monkeypatch.setattr(fx, "_FOLDX_RUNS_PER_TASK", 20)
    kwargs = {"region": "chr1:200-260", "window": 8, "step": 2}
    result = _scan(scan_inputs, tmp_path / "slurm", slurm=True, **kwargs)

    assert (result["array_job_id"], result["merge_job_id"]) == ("11", "12")
    assert (result["window"], result["dna_length"]) == (8, 5)
    assert (result["windows_per_task"], result["num_tasks"]) == (5, 6)
    array_sh = Path(calls[0]["script"]).read_text()
    assert "--array=0-5" in array_sh
    assert sys.executable in array_sh
    assert sys.executable in Path(calls[1]["script"]).read_text()
    for name in ("foldx_scan_worker.py", "foldx_scan_merge.py"):
        compile(Path(result["output_dir"], name).read_text(), name, "exec")

    for task in range(result["num_tasks"]):
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(task))
        fx._scan_chunk_job(result["params_path"])
    runs = len(fake_foldx)
    fx._scan_chunk_job(result["params_path"])
    assert len(fake_foldx) == runs, "a completed chunk must be skipped on resubmission"
    fx._merge_chunks_job(result["params_path"])

    local = _scan(scan_inputs, tmp_path / "local", **kwargs)
    assert fake_bigwig[result["bigwig_path"]].entries == fake_bigwig[local["bigwig_path"]].entries


def test_binding_scan_slurm_worker_script_runs_standalone(scan_inputs, fake_foldx, fake_bigwig,
                                                          fake_sbatch, tmp_path):
    """The generated worker runs as SLURM would start it, with nothing on stdout."""
    _, replies = fake_sbatch
    replies += [{"job_id": "11"}, {"job_id": "12"}]
    result = _scan(scan_inputs, tmp_path / "slurm", slurm=True, region="chr1:200-230", step=10)
    params_path = Path(result["params_path"])
    params = json.loads(params_path.read_text())
    params["foldx_bin"] = str(tmp_path / "no-foldx-here")
    params_path.write_text(json.dumps(params))

    proc = subprocess.run(
        [sys.executable, str(Path(result["output_dir"], "foldx_scan_worker.py"))],
        capture_output=True, text=True, timeout=300, cwd=str(tmp_path),
        env={**os.environ, "SLURM_ARRAY_TASK_ID": "0"},
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    rows = Path(result["chunks_dir"], "chunk_00000.csv").read_text().splitlines()
    assert rows[0] == f"# params_digest={result['params_digest']}"
    assert rows[1] == "position,energy"
    assert [r.split(",")[0] for r in rows[2:]] == ["200", "210", "220"]


# ---------------------------------------------------------------------------
# Placement warning, chunk digests and LF job files
# ---------------------------------------------------------------------------

def test_binding_scan_warns_with_the_run_count_for_a_wide_window(scan_inputs, fake_foldx,
                                                                 fake_bigwig, tmp_path):
    result = _scan(scan_inputs, tmp_path / "out", region="chr1:200-300", window=15, step=50)
    assert result["placements"] == 11
    assert result["num_windows"] == 2
    assert "11 placements x 2 windows = 22 FoldX runs" in result["warning"]
    assert result["foldx_runs_planned"] == 22 == result["foldx_runs"]

    overlapping = _scan(scan_inputs, tmp_path / "overlap", region="chr1:200-260", window=8, step=2)
    assert "4 placements x 27 windows = 108 FoldX runs" in overlapping["warning"]
    assert "(56 distinct" in overlapping["warning"]
    assert overlapping["foldx_runs_planned"] == overlapping["foldx_runs"] == 56


def test_binding_scan_default_window_has_no_warning(scan_inputs, fake_foldx, fake_bigwig, tmp_path):
    result = _scan(scan_inputs, tmp_path / "out")
    assert result["placements"] == 1
    assert "warning" not in result


def test_binding_scan_description_mentions_the_run_count_warning():
    from tfbs.mcp.foldx_tools import tfbs_binding_scan

    doc = " ".join(tfbs_binding_scan.__doc__.split())
    assert "placements x windows" in doc and '"warning"' in doc


def test_binding_scan_slurm_warns_too(scan_inputs, fake_foldx, fake_bigwig, fake_sbatch, tmp_path):
    _, replies = fake_sbatch
    replies += [{"job_id": "11"}, {"job_id": "12"}]
    result = _scan(scan_inputs, tmp_path / "slurm", slurm=True, region="chr1:200-260",
                   window=8, step=2)
    assert "FoldX runs" in result["warning"]


def test_binding_scan_job_files_use_lf(scan_inputs, fake_foldx, fake_bigwig, fake_sbatch, tmp_path):
    _, replies = fake_sbatch
    replies += [{"job_id": "11"}, {"job_id": "12"}]
    result = _scan(scan_inputs, tmp_path / "slurm", slurm=True)
    out = Path(result["output_dir"])
    for name in ("foldx_scan.sh", "foldx_merge.sh", "foldx_scan_worker.py",
                 "foldx_scan_merge.py", "foldx_scan_params.json"):
        assert b"\r\n" not in (out / name).read_bytes(), name


def test_resubmitted_scan_with_other_parameters_ignores_stale_chunks(
        scan_inputs, fake_foldx, fake_bigwig, fake_sbatch, tmp_path, monkeypatch):
    """A chunk left by a scan of another region is rewritten, never merged."""
    import tfbs.mcp.foldx_tools as fx

    _, replies = fake_sbatch
    replies += [{"job_id": "1"}, {"job_id": "2"}, {"job_id": "3"}, {"job_id": "4"}]
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    out = tmp_path / "scan"

    first = _scan(scan_inputs, out, slurm=True, region="chr1:200-230", step=10)
    fx._scan_chunk_job(first["params_path"])
    chunk = Path(first["chunks_dir"], "chunk_00000.csv")
    assert fx._chunk_digest(chunk) == first["params_digest"]

    second = _scan(scan_inputs, out, slurm=True, region="chr1:400-430", step=10)
    assert second["params_digest"] != first["params_digest"]
    assert second["params_path"] == first["params_path"]

    fx._merge_chunks_job(second["params_path"])
    assert fake_bigwig[second["bigwig_path"]].entries == [], "a stale chunk was merged"

    runs = len(fake_foldx)
    fx._scan_chunk_job(second["params_path"])
    assert len(fake_foldx) > runs, "a stale chunk must be rescored, not skipped"
    assert fx._chunk_digest(chunk) == second["params_digest"]
    fx._merge_chunks_job(second["params_path"])
    starts = [s for _, s, _, _ in fake_bigwig[second["bigwig_path"]].entries]
    assert starts == [400, 410, 420]


def test_binding_scan_description_calls_the_slurm_run_count_a_lower_bound():
    from tfbs.mcp.foldx_tools import tfbs_binding_scan

    doc = " ".join(tfbs_binding_scan.__doc__.split())
    assert "foldx_runs_planned" in doc and "with slurm=true a lower bound" in doc


_PLDDT = [90, 85, 40, 30, 45, 75, 20, 95, 49, 10]


def _af_text(plddts=_PLDDT, chain="A") -> str:
    lines = []
    for resnum, plddt in enumerate(plddts, start=1):
        for name in ("N", "CA"):
            lines.append(f"ATOM  {len(lines) + 1:5d}  {name:<3} ALA {chain}{resnum:4d}    "
                         f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00{plddt:6.2f}           C")
    return "\n".join(lines) + "\nEND\n"


def test_plddt_summary_reads_the_b_factor_column(tmp_path):
    from tfbs.mcp.foldx_tools import _plddt_summary

    pdb = tmp_path / "af.pdb"
    pdb.write_text(_af_text())
    summary = _plddt_summary(str(pdb))
    assert summary == {"fraction_confident": 0.4, "fraction_low": 0.6,
                       "low_confidence_segments": [
                           {"chain": "A", "start": 3, "end": 5},
                           {"chain": "A", "start": 7, "end": 7},
                           {"chain": "A", "start": 9, "end": 10}],
                       "low_confidence_segments_total": 3}
    pdb.write_text(_af_text([10, 90] * 30))
    many = _plddt_summary(str(pdb))
    assert len(many["low_confidence_segments"]) == 20
    assert many["low_confidence_segments_total"] == 30
    pdb.write_text("HEADER only\nEND\n")
    assert _plddt_summary(str(pdb)) is None


def test_fetch_structure_reports_plddt_for_alphafold_only(fake_web, tmp_path):
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    fake_web({_AF_API: json.dumps(_AF_ENTRIES).encode(), _AF_MODEL: _af_text().encode(),
              _rcsb("3RKQ"): _pdb_text("ACGTA").encode()})
    af = _run(tfbs_fetch_structure(uniprot_id="P43699", output_dir=str(tmp_path / "af")))
    assert af["plddt_summary"]["fraction_low"] == 0.6
    assert af["plddt_summary"]["low_confidence_segments"][0] == {"chain": "A", "start": 3, "end": 5}
    pdb = _run(tfbs_fetch_structure(pdb_id="3RKQ", output_dir=str(tmp_path / "pdb")))
    assert pdb["source"] == "pdb" and "plddt_summary" not in pdb


def test_fetch_structure_description_explains_plddt_summary():
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    doc = " ".join(tfbs_fetch_structure.__doc__.split())
    assert "plddt_summary" in doc and "flag low-confidence regions" in doc


def test_plddt_segments_do_not_run_across_chains(tmp_path):
    from tfbs.mcp.foldx_tools import _plddt_summary

    pdb = tmp_path / "af.pdb"
    two = _af_text([90, 30, 40], chain="A").replace("END\n", "") + _af_text([20, 95], chain="B")
    pdb.write_text(two)
    summary = _plddt_summary(str(pdb))
    assert summary["low_confidence_segments"] == [
        {"chain": "A", "start": 2, "end": 3}, {"chain": "B", "start": 1, "end": 1}]
    assert summary["low_confidence_segments_total"] == 2


def test_fetch_structure_downloads_into_the_results_dir_by_default(fake_web, tmp_path, monkeypatch):
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    fake_web({
        _rcsb("9ZZZ"): _http_error(_rcsb("9ZZZ"), 404),
        _AF_API: json.dumps(_AF_ENTRIES).encode(), _AF_MODEL: _af_text().encode(),
    })
    result = _run(tfbs_fetch_structure(pdb_id="9ZZZ", uniprot_id="P43699"))
    path = Path(result["structure_path"])
    assert path.parent == tmp_path / "results" / "structures" / "P43699"
    assert path.is_file()
    doc = " ".join(tfbs_fetch_structure.__doc__.split())
    assert "<results>/structures/<id>/" in doc and "temporary" not in doc


def test_fetch_structure_ids_share_a_directory_whatever_their_case(fake_web, tmp_path, monkeypatch):
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    fake_web({_rcsb("3RKQ"): _pdb_text("ACGTA").encode()})
    lower = _run(tfbs_fetch_structure(pdb_id="3rkq"))
    upper = _run(tfbs_fetch_structure(pdb_id="3RKQ"))
    assert Path(lower["structure_path"]).parent == tmp_path / "results" / "structures" / "3RKQ"
    assert lower["structure_path"] == upper["structure_path"]
    assert sorted(p.name for p in (tmp_path / "results" / "structures").iterdir()) == ["3RKQ"]

    fake_web({_AF_API: json.dumps(_AF_ENTRIES).encode(), _AF_MODEL: _af_text().encode()})
    result = _run(tfbs_fetch_structure(uniprot_id="P43699"))
    assert Path(result["structure_path"]).parent.name == "P43699"
    doc = " ".join(tfbs_fetch_structure.__doc__.split())
    assert "PDB and UniProt IDs uppercased, so 3rkq and 3RKQ share a directory" in doc


def test_binding_scan_writes_to_the_results_dir_by_default(scan_inputs, fake_foldx, fake_bigwig,
                                                           tmp_path, monkeypatch):
    from tfbs.mcp.foldx_tools import tfbs_binding_scan

    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    pdb, fasta, _ = scan_inputs
    args = {"structure_path": pdb, "genome_fasta": fasta, "region": "chr1:100-1100"}
    first = _run(tfbs_binding_scan(**args))
    out = Path(first["output_dir"])
    assert out.parent == tmp_path / "results" / "binding_scan"
    assert Path(first["bigwig_path"]).parent == out
    assert _run(tfbs_binding_scan(**args))["output_dir"] == first["output_dir"]
    for change in ({"step": 25}, {"window": 7}, {"region": "chr1:100-1200"}):
        assert _run(tfbs_binding_scan(**{**args, **change}))["output_dir"] != str(out), change
    doc = " ".join(tfbs_binding_scan.__doc__.split())
    assert "<results>/binding_scan/<digest>/" in doc and "temp" not in doc


def test_binding_scan_default_dir_tells_genomes_and_structures_apart(scan_inputs, fake_foldx,
                                                                     fake_bigwig, tmp_path,
                                                                     monkeypatch):
    from tfbs.mcp.foldx_tools import tfbs_binding_scan

    monkeypatch.setenv("TFBS_PROJECT_ROOT", str(tmp_path))
    pdb, fasta, _ = scan_inputs
    args = {"structure_path": pdb, "genome_fasta": fasta, "region": "chr1:100-1100"}
    seen = {_run(tfbs_binding_scan(**args))["output_dir"]}
    other = tmp_path / "other"
    other.mkdir()
    other_fasta, _ = _make_genome(str(other), seed=8)
    for change in ({"genome_fasta": other_fasta}, {"structure_path": _make_dna_pdb(str(other))}):
        out = _run(tfbs_binding_scan(**{**args, **change}))["output_dir"]
        assert out not in seen, change
        seen.add(out)
    for rewritten in (fasta, pdb):
        st = os.stat(rewritten)
        os.utime(rewritten, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
        out = _run(tfbs_binding_scan(**args))["output_dir"]
        assert out not in seen, rewritten
        seen.add(out)
    doc = " ".join(tfbs_binding_scan.__doc__.split())
    assert "size and modification time of both the structure and the genome FASTA" in doc


def test_fetch_structure_says_a_repeated_call_downloads_again():
    from tfbs.mcp.foldx_tools import tfbs_fetch_structure

    doc = " ".join(tfbs_fetch_structure.__doc__.split())
    assert "reuses that directory and downloads the file again" in doc
    assert "reuses its download" not in doc
