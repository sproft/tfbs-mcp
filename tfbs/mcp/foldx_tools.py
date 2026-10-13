"""
FoldX binding-energy tools for TF-DNA complex analysis.

Registers three tools on the shared ``mcp`` FastMCP instance:

* **tfbs_fetch_structure**: retrieve a structure from a local path, the PDB
  (by accession or RCSB gene-name search) or AlphaFold DB, moving on to the
  next source given when one fails.
* **tfbs_binding_energy**: run FoldX AnalyseComplex on a TF-DNA structure.
* **tfbs_binding_scan**: slide a window across a genomic region, score each
  window with FoldX and write a bigWig track, in-process or as a SLURM
  array job.

Progress goes to the module logger, never to stdout: on the stdio transport
stdout carries the MCP protocol.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple

import tfbs.mcp.server as _server
from tfbs.mcp.server import mcp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

_HTTP_TIMEOUT = 60
_RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{}.pdb"
_RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
_AFDB_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"

# FoldX runs per SLURM array task: at about 7 min per run, 250 runs take
# about 29 h, inside the 48 h default time limit.
_FOLDX_RUNS_PER_TASK = 250

_NO_DNA_NOTE = (
    "The structure has no DNA chains, so tfbs_binding_energy and "
    "tfbs_binding_scan cannot score TF-DNA binding on it. AlphaFold DB models "
    "are protein-only; for FoldX pass a TF-DNA complex (a PDB co-crystal, or "
    "an AlphaFold 3 prediction saved as PDB) as local_path."
)


def _error(error: str, fix: str, **extra: Any) -> str:
    return json.dumps({"error": error, "fix": fix, **extra})


def _find_foldx() -> str | None:
    """Return the path to the FoldX binary, or None."""
    return shutil.which("foldx") or shutil.which("foldx5")


def _parse_dna_from_pdb(pdb_path: str) -> dict[str, list[tuple[int, str]]]:
    """Parse DNA residues from a PDB file.

    Returns a dict mapping chain ID to a sorted list of (resnum, base)
    tuples, where base is a single uppercase letter (A, T, C, G).
    """
    chains: dict[str, dict[int, str]] = {}
    dna_resnames = {"DA", "DT", "DC", "DG", "A", "T", "C", "G"}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip()
            if resname not in dna_resnames:
                continue
            chain = line[21]
            resnum = int(line[22:26])
            base = resname[-1]  # DA->A, DT->T, or already single-letter
            chains.setdefault(chain, {})[resnum] = base
    return {ch: sorted(bases.items()) for ch, bases in chains.items()}


_PLDDT_CONFIDENT = 70.0
_PLDDT_LOW = 50.0
_MAX_LOW_SEGMENTS = 20


def _plddt_summary(pdb_path: str) -> dict | None:
    """Per-residue pLDDT statistics from the B-factor column of an AlphaFold PDB.

    AlphaFold writes the residue's pLDDT into the B-factor of each of its
    atoms; the first atom of every residue is read.  None when the file
    has no ATOM records.
    """
    residues: list[tuple[str, int, float]] = []
    seen: set[tuple[str, str]] = set()
    with open(pdb_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("ATOM") or len(line) < 66:
                continue
            key = (line[21], line[22:27])
            if key in seen:
                continue
            seen.add(key)
            try:
                residues.append((line[21], int(line[22:26]), float(line[60:66])))
            except ValueError:
                continue
    if not residues:
        return None
    n = len(residues)
    segments: list[dict[str, Any]] = []
    previous: tuple[str, int] | None = None
    for chain, resnum, plddt in residues:
        if plddt < _PLDDT_LOW:
            if (segments and previous is not None and previous[0] == chain
                    and resnum == previous[1] + 1 and segments[-1]["end"] == previous[1]
                    and segments[-1]["chain"] == chain.strip()):
                segments[-1]["end"] = resnum
            else:
                segments.append({"chain": chain.strip(), "start": resnum, "end": resnum})
        previous = (chain, resnum)
    return {
        "fraction_confident": round(sum(p >= _PLDDT_CONFIDENT for *_, p in residues) / n, 4),
        "fraction_low": round(sum(p < _PLDDT_LOW for *_, p in residues) / n, 4),
        "low_confidence_segments": segments[:_MAX_LOW_SEGMENTS],
        "low_confidence_segments_total": len(segments),
    }


def _has_dna(structure_path: str) -> bool | None:
    """Whether a PDB-format file has DNA chains; None if it cannot be parsed."""
    if Path(structure_path).suffix.lower() not in (".pdb", ".ent"):
        return None
    try:
        return bool(_parse_dna_from_pdb(structure_path))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _build_foldx_mutation_string(
    ref_bases: list[tuple[int, str]],
    target_seq: str,
    chain: str,
) -> str:
    """Build a FoldX mutation string to convert reference DNA to target.

    FoldX DNA mutation syntax uses lowercase: ``aC3t`` means residue A
    at chain C position 3 becomes T.

    Only generates mutations for positions that differ.  Returns empty
    string if no mutations needed.
    """
    mutations = []
    for (resnum, ref_base), tgt_base in zip(ref_bases, target_seq.upper()):
        if ref_base != tgt_base:
            mutations.append(f"{ref_base.lower()}{chain}{resnum}{tgt_base.lower()}")
    return ",".join(mutations) + ";" if mutations else ""


def _run_foldx_energy(
    foldx_bin: str,
    repaired_pdb: str,
    working_dir: str,
    mutation_string: str | None = None,
) -> float | None:
    """Run FoldX BuildModel (if mutations) + AnalyseComplex, return the energy.

    Returns the interaction energy in kcal/mol, or None on failure.
    """
    pdb_name = Path(repaired_pdb).name
    wd = Path(working_dir)

    # Copy repaired PDB to working dir if not already there
    target = wd / pdb_name
    if not target.exists():
        shutil.copy2(repaired_pdb, target)

    analyse_target = pdb_name

    # Apply mutations if needed
    if mutation_string:
        mut_file = wd / "individual_list.txt"
        mut_file.write_text(mutation_string + "\n", newline="\n")
        try:
            subprocess.run(
                [foldx_bin, "--command=BuildModel",
                 f"--pdb={pdb_name}", f"--mutant-file={mut_file.name}"],
                capture_output=True, text=True, stdin=subprocess.DEVNULL,
                timeout=1200, cwd=str(wd),
            )
        except subprocess.TimeoutExpired:
            return None
        mutated = wd / f"{Path(pdb_name).stem}_1.pdb"
        if mutated.exists():
            analyse_target = mutated.name
        else:
            return None

    # AnalyseComplex
    subprocess.run(
        [foldx_bin, "--command=AnalyseComplex",
         f"--pdb={analyse_target}", "--complexWithDNA=true"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60, cwd=str(wd),
    )

    # Parse energy from output file
    ac_files = list(wd.glob("Interaction_*_AC.fxout"))
    if not ac_files:
        return None
    lines = [l for l in ac_files[0].read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    if not lines:
        return None
    parts = lines[-1].split("\t")
    try:
        return float(parts[5]) if len(parts) > 5 else None
    except (ValueError, IndexError):
        return None


def _repaired_file(work: Path, input_name: str) -> Path | None:
    """The file FoldX RepairPDB wrote into ``work``, or None."""
    expected = work / f"{Path(input_name).stem}_Repair.pdb"
    if expected.is_file():
        return expected
    # Some FoldX versions use different naming
    others = sorted(p for p in work.glob("*Repair*.pdb") if p.name != input_name)
    return others[0] if others else None


def _repaired_dir(pdb: Path, output_dir: str | None) -> Path:
    """Where tfbs_binding_energy keeps a repaired structure."""
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    structures = (_server._results_root() / "structures").resolve()
    if pdb.parent.parent == structures:
        return pdb.parent
    return structures / _server._safe_name(pdb.stem)


_FOLDX_MISSING = "FoldX binary not found on PATH."
_FOLDX_MISSING_FIX = (
    "Install FoldX (https://foldxsuite.crg.eu/, free academic licence), put "
    "'foldx' or 'foldx5' on the PATH of the MCP server process, and restart "
    "the server."
)


def _foldx_missing_error() -> str:
    return _error(_FOLDX_MISSING, _FOLDX_MISSING_FIX)


# ---------------------------------------------------------------------------
# Structure sources for tfbs_fetch_structure
# ---------------------------------------------------------------------------

class _SourceError(Exception):
    """A structure source failed; carries the error and a fix hint."""

    def __init__(self, error: str, fix: str) -> None:
        super().__init__(error)
        self.error = error
        self.fix = fix


_RETRY_FIX = "Retry later or check the network connection; or pass local_path."


def _http_json(url: str, payload: dict | None = None) -> Any:
    """GET (or POST ``payload`` to) ``url`` and parse the JSON body; None if empty."""
    req: str | urllib.request.Request = url
    if payload is not None:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        body = resp.read().decode()
    return json.loads(body) if body.strip() else None


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``, removing a partial file on failure."""
    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except Exception:
        dest.unlink(missing_ok=True)
        raise


def _fetch_local(local_path: str, out: Path | None) -> dict:
    p = Path(local_path)
    if not p.is_file():
        raise _SourceError(
            f"Local file not found: {local_path}",
            "Check the path, or also pass pdb_id, gene_name or uniprot_id so "
            "a structure can be downloaded instead.",
        )
    return {
        "source": "local",
        "structure_path": str(p.resolve()),
        "metadata": {"original_path": str(p)},
    }


def _fetch_pdb(pdb_id: str, out: Path) -> dict:
    pdb_id = pdb_id.strip().upper()
    url = _RCSB_DOWNLOAD_URL.format(urllib.parse.quote(pdb_id))
    dest = out / f"{re.sub(r'[^A-Z0-9_]', '_', pdb_id)}.pdb"
    try:
        _download(url, dest)
    except Exception as exc:
        raise _SourceError(
            f"Failed to download PDB {pdb_id} from RCSB: {exc}",
            "Check the accession on rcsb.org (very large entries have no "
            "PDB-format file) and the network connection, or pass local_path.",
        ) from exc
    return {
        "source": "pdb",
        "structure_path": str(dest.resolve()),
        "metadata": {"pdb_id": pdb_id, "url": url},
    }


def _rcsb_protein_dna_query(gene_name: str) -> dict:
    """RCSB search for the top experimental entry with protein and DNA."""
    def polymer(kind: str) -> dict:
        return {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_entity_polymer_type",
                "operator": "exact_match",
                "value": kind,
            },
        }

    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "full_text",
                 "parameters": {"value": gene_name}},
                polymer("Protein"),
                polymer("DNA"),
            ],
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "paginate": {"start": 0, "rows": 1},
        },
    }


def _fetch_gene(gene_name: str, out: Path) -> dict:
    gene_name = gene_name.strip()
    try:
        data = _http_json(_RCSB_SEARCH_URL, _rcsb_protein_dna_query(gene_name))
    except Exception as exc:
        raise _SourceError(f"RCSB search failed for '{gene_name}': {exc}", _RETRY_FIX) from exc
    hits = (data or {}).get("result_set") or []
    if not hits:
        raise _SourceError(
            f"No PDB protein-DNA structure found for gene '{gene_name}'",
            "Try another gene symbol or alias, pass uniprot_id for an AlphaFold "
            "DB model, or pass a TF-DNA complex as local_path.",
        )
    result = _fetch_pdb(hits[0]["identifier"], out)
    result["metadata"]["gene_name"] = gene_name
    return result


def _fetch_alphafold(uniprot_id: str, out: Path) -> dict:
    acc = uniprot_id.strip()
    api_url = _AFDB_API_URL.format(urllib.parse.quote(acc))
    no_entry_fix = (
        "Pass a UniProt accession such as P43699 (not an entry name such as "
        "NKX21_HUMAN) and check it on https://alphafold.ebi.ac.uk, or pass "
        "pdb_id, gene_name or local_path."
    )
    try:
        entries = _http_json(api_url)
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 404):
            raise _SourceError(
                f"AlphaFold DB has no entry for '{acc}' (HTTP {exc.code} from {api_url})",
                no_entry_fix,
            ) from exc
        raise _SourceError(f"AlphaFold DB API request failed for '{acc}': {exc}", _RETRY_FIX) from exc
    except Exception as exc:
        raise _SourceError(f"AlphaFold DB API request failed for '{acc}': {exc}", _RETRY_FIX) from exc

    if isinstance(entries, dict):
        entries = [entries]
    models = [e for e in entries or [] if isinstance(e, dict) and e.get("pdbUrl")]
    if not models:
        raise _SourceError(f"AlphaFold DB has no PDB-format model for '{acc}' ({api_url})", no_entry_fix)

    # The API also lists isoform entries (e.g. P43699-3); prefer the exact match.
    wanted = acc.upper()
    entry = next(
        (e for e in models
         if wanted in {str(e.get(k, "")).upper() for k in ("uniprotAccession", "entryId", "modelEntityId")}),
        models[0],
    )
    model_url = entry["pdbUrl"]
    dest = out / (Path(urllib.parse.urlparse(model_url).path).name or f"AF-{wanted}.pdb")
    try:
        _download(model_url, dest)
    except Exception as exc:
        raise _SourceError(
            f"AlphaFold DB model download failed for '{acc}' ({model_url}): {exc}", _RETRY_FIX,
        ) from exc
    return {
        "source": "alphafold",
        "structure_path": str(dest.resolve()),
        "metadata": {
            "uniprot_id": entry.get("uniprotAccession", acc),
            "entry_id": entry.get("entryId"),
            "latest_version": entry.get("latestVersion"),
            "mean_plddt": entry.get("globalMetricValue"),
            "url": model_url,
            "api_url": api_url,
        },
    }


# ---------------------------------------------------------------------------
# Tool 1: tfbs_fetch_structure
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_fetch_structure",
    annotations={
        "title": "Fetch TF-DNA Structure",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tfbs_fetch_structure(
    local_path: str | None = None,
    pdb_id: str | None = None,
    gene_name: str | None = None,
    uniprot_id: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Retrieve a TF-DNA complex structure, trying each source given in turn.

    Search order, over the arguments passed: local_path, then pdb_id (RCSB
    download), then gene_name (RCSB search restricted to entries with both
    protein and DNA), then uniprot_id (AlphaFold DB; the latest model
    version is looked up through the AlphaFold DB API).  When a source
    fails, the next one given is tried, so passing several arguments makes
    a fallback chain.

    AlphaFold DB models are protein-only (``has_dna`` false), so FoldX
    cannot score TF-DNA binding on them; report their ``mean_plddt`` (0 to
    100, below 70 is low confidence) whenever you use one.  For them the
    result also has ``plddt_summary``, read from the per-residue pLDDT in
    the model's B-factor column, which lets the client flag low-confidence
    regions (often the disordered parts of a transcription factor
    outside its DNA-binding domain).

    Default structure source and why: a PDB co-crystal first (pdb_id, or
    gene_name for the top RCSB entry with protein and DNA), with
    AlphaFold DB (uniprot_id) only as the fallback when no co-crystal
    exists, because FoldX scores the protein-DNA interface and only an
    experimental complex contains that DNA with measured contacts.  Pass
    gene_name and uniprot_id together to get this order in one call.

    Args:
        local_path:  Path to a PDB file already on disk.
        pdb_id:      4-character PDB accession (e.g. ``3RKQ``).
        gene_name:   Gene name to search RCSB for (e.g. ``NKX2-1``).
        uniprot_id:  UniProt accession for AlphaFold DB (e.g. ``P43699``).
        output_dir:  Directory to save downloaded files.  Default
                     <results>/structures/<id>/, with <results> the
                     results_dir of tfbs_server_info and <id> the value of
                     the source argument being tried (e.g. 3RKQ, NKX2-1 or
                     P43699; PDB and UniProt IDs uppercased, so 3rkq and
                     3RKQ share a directory).  A repeated call reuses
                     that directory and downloads the file again,
                     overwriting an earlier copy of the same file.

    Returns:
        JSON with ``source`` (``local``, ``pdb`` or ``alphafold``),
        ``argument`` (the argument that succeeded), ``structure_path``,
        ``metadata`` and ``attempted`` (the sources that failed first, each
        with ``argument``, ``value``, ``error`` and ``fix``).  ``metadata``
        always has ``has_dna`` (null if the file is not PDB format); for
        AlphaFold DB it also has ``mean_plddt``, ``latest_version``,
        ``entry_id`` and the model ``url``.  An AlphaFold DB model also
        gets ``plddt_summary``: ``fraction_confident`` (share of residues
        with pLDDT >= 70), ``fraction_low`` (share below 50),
        ``low_confidence_segments`` (up to 20 runs of consecutive
        residues with pLDDT below 50, each as {chain, start, end} with
        residue numbers) and ``low_confidence_segments_total`` (the
        number of such runs, also beyond the first 20); PDB entries have
        none.
        If every source fails, JSON with ``error``, ``fix`` and
        ``attempted``.
    """
    legs = (
        ("local_path", local_path, _fetch_local),
        ("pdb_id", pdb_id, _fetch_pdb),
        ("gene_name", gene_name, _fetch_gene),
        ("uniprot_id", uniprot_id, _fetch_alphafold),
    )
    candidates = [(arg, value, leg) for arg, value, leg in legs if value is not None and str(value).strip()]
    if not candidates:
        return _error(
            "No structure source given.",
            "Pass at least one of local_path, pdb_id, gene_name or uniprot_id.",
        )

    attempted: list[dict] = []
    for arg, value, leg in candidates:
        out: Path | None = None
        if arg != "local_path":
            key = str(value).strip()
            if arg in ("pdb_id", "uniprot_id"):
                key = key.upper()
            out = (Path(output_dir) if output_dir
                   else _server._results_root() / "structures" / _server._safe_name(key))
            try:
                out.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                return _error(
                    f"Cannot create output directory {out}: {exc}",
                    "Pass a writable output_dir, or set TFBS_PROJECT_ROOT to a writable "
                    "directory (tfbs_server_info reports the results_dir in use).",
                    attempted=attempted,
                )
        try:
            result = leg(value, out)
        except _SourceError as exc:
            error, fix = exc.error, exc.fix
        except Exception as exc:
            error, fix = f"{type(exc).__name__}: {exc}", _RETRY_FIX
        else:
            has_dna = _has_dna(result["structure_path"])
            result["metadata"]["has_dna"] = has_dna
            result["argument"] = arg
            result["attempted"] = attempted
            if has_dna is False:
                result["note"] = _NO_DNA_NOTE
            if result["source"] == "alphafold":
                try:
                    summary = _plddt_summary(result["structure_path"])
                except OSError:
                    summary = None
                if summary is not None:
                    result["plddt_summary"] = summary
            return json.dumps(result)
        logger.info("tfbs_fetch_structure: %s=%r failed: %s", arg, value, error)
        attempted.append({"argument": arg, "value": value, "error": error, "fix": fix})

    if len(attempted) == 1:
        return _error(attempted[0]["error"], attempted[0]["fix"], attempted=attempted)
    tried = ", ".join(a["argument"] for a in attempted)
    return _error(
        f"No structure could be retrieved; every source failed ({tried}).",
        "See 'attempted' for each source's error and fix.",
        attempted=attempted,
    )


# ---------------------------------------------------------------------------
# Tool 2: tfbs_binding_energy
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_binding_energy",
    annotations={
        "title": "FoldX Binding Energy",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tfbs_binding_energy(
    structure_path: str,
    sequences: list[str] | None = None,
    repair: bool = True,
    chains_protein: str = "A",
    chains_dna: str = "C,D",
    output_dir: str | None = None,
) -> str:
    """Compute TF-DNA binding energy with FoldX AnalyseComplex.

    With repair=true (the default) FoldX RepairPDB runs first and the
    repaired complex is kept as <stem>_Repair.pdb, <stem> being the
    structure file's name without .pdb.  Its path, returned as
    ``repaired_structure_path``, is the repaired structure that
    tfbs_binding_scan (structure_path) and the FoldX legs of
    tfbs_chipseq_method_comparison and tfbs_remap_validation (foldx_pdb,
    or $TFBS_FOLDX_PDB) and tfbs_disease_variant_validation (foldx_pdb)
    expect.

    Args:
        structure_path: Path to a PDB file.
        sequences:      Optional DNA sequences (placeholder for future
                        mutation support).
        repair:         Run FoldX RepairPDB first and keep the repaired
                        file (recommended).  Pass false only for a
                        structure FoldX has already repaired.
        chains_protein: Protein chain ID(s), e.g. ``"A"``.
        chains_dna:     DNA chain ID(s), comma-separated, e.g. ``"C,D"``.
        output_dir:     Directory for the repaired file.  Default: the
                        structure's own directory when it lies in
                        <results>/structures/<id>/, where
                        tfbs_fetch_structure downloads (<results> is the
                        results_dir of tfbs_server_info), else
                        <results>/structures/<stem>/.  A repeated call
                        overwrites the earlier repaired file.

    Returns:
        JSON with ``structure_path`` (the input),
        ``repaired_structure_path`` (null with repair=false, when the
        input is scored as given) and the energy components parsed from
        the FoldX output.
    """
    foldx = _find_foldx()
    if foldx is None:
        return _foldx_missing_error()

    pdb = Path(structure_path).resolve()
    if not pdb.is_file():
        return _error(
            f"Structure file not found: {structure_path}",
            "Pass the path of a PDB file, for example the structure_path "
            "returned by tfbs_fetch_structure.",
        )

    tmpdir = tempfile.mkdtemp(prefix="tfbs_foldx_")
    work = Path(tmpdir)

    # Copy PDB into working directory
    shutil.copy2(pdb, work / pdb.name)

    pdb_name = pdb.name
    stem = pdb.stem

    stage = "RepairPDB"
    kept: Path | None = None
    try:
        # --- RepairPDB -------------------------------------------------
        if repair:
            cmd_repair = [
                foldx, "--command=RepairPDB",
                f"--pdb={pdb_name}",
                f"--output-dir={tmpdir}",
            ]
            repair_run = subprocess.run(cmd_repair, capture_output=True, text=True,
                                        stdin=subprocess.DEVNULL, cwd=tmpdir, timeout=600)
            repaired = _repaired_file(work, pdb.name)
            if repaired is None:
                return _error(
                    "FoldX RepairPDB wrote no repaired structure",
                    "Read stderr and stdout; check that the file is a PDB-format "
                    "TF-DNA complex FoldX can read.",
                    stderr=repair_run.stderr[-2000:] if repair_run.stderr else "",
                    stdout=repair_run.stdout[-2000:] if repair_run.stdout else "",
                )
            pdb_name = repaired.name
            kept = _repaired_dir(pdb, output_dir) / f"{stem}_Repair.pdb"
            try:
                kept.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repaired, kept)
            except OSError as exc:
                return _error(
                    f"Cannot save the repaired structure to {kept}: {exc}",
                    "Pass a writable output_dir, or set TFBS_PROJECT_ROOT to a writable "
                    "directory (tfbs_server_info reports the results_dir in use).",
                )

        # --- AnalyseComplex --------------------------------------------
        stage = "AnalyseComplex"
        cmd_analyse = [
            foldx, "--command=AnalyseComplex",
            f"--pdb={pdb_name}",
            f"--analyseComplexChains={chains_protein},{chains_dna}",
            "--complexWithDNA=true",
            f"--output-dir={tmpdir}",
        ]
        result = subprocess.run(cmd_analyse, capture_output=True, text=True,
                                stdin=subprocess.DEVNULL, cwd=tmpdir, timeout=600)
    except subprocess.TimeoutExpired as exc:
        return _error(
            f"FoldX {stage} timed out after {exc.timeout:.0f} s",
            "Check that the file is a single TF-DNA complex of moderate size; "
            "pass repair=false if it has already been repaired.",
        )
    except OSError as exc:
        return _error(
            f"Could not run FoldX {stage} ({foldx}): {exc}",
            "Check that the FoldX binary on PATH runs on this machine.",
        )

    if result.returncode != 0:
        return _error(
            "FoldX AnalyseComplex failed",
            "Read stderr and stdout; check that chains_protein and chains_dna "
            "match the chain IDs in the structure.",
            stderr=result.stderr[-2000:] if result.stderr else "",
            stdout=result.stdout[-2000:] if result.stdout else "",
        )

    # --- Parse output --------------------------------------------------
    interaction_file = None
    for f in work.iterdir():
        if f.name.startswith("Interaction_") and f.name.endswith("_AC.fxout"):
            interaction_file = f
            break

    if interaction_file is None:
        # Try broader match
        for f in work.iterdir():
            if f.name.startswith("Interaction_") and f.suffix == ".fxout":
                interaction_file = f
                break

    if interaction_file is None:
        return _error(
            "FoldX output file not found",
            "FoldX ran but wrote no Interaction_*.fxout; check that the "
            "structure contains the protein and DNA chains named in "
            "chains_protein and chains_dna.",
            files=[f.name for f in work.iterdir()],
        )

    energy = {}
    with open(interaction_file) as fh:
        lines = fh.readlines()
        # FoldX interaction files: header line(s) starting with "Pdb", then data
        header = None
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("Pdb"):
                header = line.split("\t")
                continue
            if header:
                vals = line.split("\t")
                for h, v in zip(header[1:], vals[1:]):
                    try:
                        energy[h.strip()] = float(v.strip())
                    except ValueError:
                        energy[h.strip()] = v.strip()
                break  # first data row only

    return json.dumps({
        "structure_path": str(pdb.resolve()),
        "repaired_structure_path": str(kept) if kept else None,
        "interaction_energy": energy.get("Interaction Energy", None),
        "backbone_hbond": energy.get("Backbone Hbond", None),
        "sidechain_hbond": energy.get("Sidechain Hbond", None),
        "van_der_waals": energy.get("Van der Waals", None),
        "electrostatics": energy.get("Electrostatics", None),
        "all_components": energy,
        "foldx_stdout": result.stdout[-1000:] if result.stdout else "",
    })


# ---------------------------------------------------------------------------
# Scan helpers shared by the local scan and the SLURM jobs
# ---------------------------------------------------------------------------

class _DnaTemplate(NamedTuple):
    """The structure's DNA: the forward chain and, if present, its partner."""

    fwd_chain: str
    fwd_bases: list[tuple[int, str]]
    rev_chain: str | None
    rev_bases: list[tuple[int, str]] | None


def _dna_template(pdb_path: str) -> _DnaTemplate | None:
    chains = _parse_dna_from_pdb(pdb_path)
    if not chains:
        return None
    ids = sorted(chains)
    rev = ids[1] if len(ids) > 1 else None
    return _DnaTemplate(ids[0], chains[ids[0]], rev, chains[rev] if rev else None)


def _mutation_string(template: _DnaTemplate, seq_fwd: str) -> str | None:
    """FoldX mutation string threading ``seq_fwd`` onto both DNA strands."""
    parts = []
    mut_fwd = _build_foldx_mutation_string(template.fwd_bases, seq_fwd, template.fwd_chain)
    if mut_fwd:
        parts.append(mut_fwd.rstrip(";"))
    if template.rev_chain and template.rev_bases:
        seq_rev = "".join(_COMPLEMENT.get(b, "N") for b in reversed(seq_fwd))
        mut_rev = _build_foldx_mutation_string(template.rev_bases, seq_rev, template.rev_chain)
        if mut_rev:
            parts.append(mut_rev.rstrip(";"))
    return ",".join(parts) + ";" if parts else None


class _WindowScorer:
    """Score scan windows with FoldX.

    A window's energy is the lowest FoldX interaction energy over every
    placement of the structure's DNA inside it (one placement when the
    window equals the DNA length).  Placement energies are cached, so
    overlapping windows share FoldX runs; windows must be scored in
    increasing order.
    """

    def __init__(self, foldx_bin: str, structure: str, template: _DnaTemplate,
                 genome: Any, chrom: str, window: int) -> None:
        self.foldx_bin = foldx_bin
        self.structure = structure
        self.template = template
        self.genome = genome
        self.chrom = chrom
        self.window = window
        self.dna_length = len(template.fwd_bases)
        self.runs = 0
        self._cache: dict[int, float] = {}

    def _placement_energy(self, pos: int) -> float:
        try:
            seq = str(self.genome[self.chrom][pos:pos + self.dna_length]).upper()
        except (KeyError, ValueError, IndexError):
            return math.nan
        if len(seq) != self.dna_length or any(c not in "ACGT" for c in seq):
            return math.nan
        self.runs += 1
        try:
            with tempfile.TemporaryDirectory(prefix="fxw_") as wdir:
                energy = _run_foldx_energy(
                    self.foldx_bin, self.structure, wdir, _mutation_string(self.template, seq),
                )
        except Exception as exc:
            logger.warning("FoldX failed at %s:%d: %s", self.chrom, pos, exc)
            return math.nan
        return math.nan if energy is None else float(energy)

    def window_energy(self, w_start: int) -> float:
        for pos in [p for p in self._cache if p < w_start]:
            del self._cache[pos]
        energies = []
        for pos in range(w_start, w_start + self.window - self.dna_length + 1):
            if pos not in self._cache:
                self._cache[pos] = self._placement_energy(pos)
            if not math.isnan(self._cache[pos]):
                energies.append(self._cache[pos])
        return min(energies) if energies else math.nan


def _bigwig_interval(w_start: int, window: int, step: int) -> tuple[int, int]:
    """Interval that carries one window's value in the bigWig.

    bigWig entries must not overlap, so when windows overlap (step < window)
    each value covers the step-wide bin centred on its window.
    """
    if window <= step:
        return w_start, w_start + window
    left = w_start + (window - step) // 2
    return left, left + step


def _write_bigwig(bw_path: Path, chrom: str, chrom_length: int,
                  scores: list[tuple[int, float]], window: int, step: int) -> int:
    """Write scored windows to ``bw_path``; return how many carried a value."""
    import pyBigWig

    entries = []
    for w_start, energy in sorted(scores):
        if math.isnan(energy):
            continue
        lo, hi = _bigwig_interval(w_start, window, step)
        hi = min(hi, chrom_length)
        if lo < hi:
            entries.append((lo, hi, float(energy)))
    bw = pyBigWig.open(str(bw_path), "w")
    try:
        bw.addHeader([(chrom, chrom_length)])
        if entries:
            bw.addEntries(
                [chrom] * len(entries),
                [e[0] for e in entries],
                ends=[e[1] for e in entries],
                values=[e[2] for e in entries],
            )
    finally:
        bw.close()
    return len(entries)


# Parameters that decide which windows a chunk holds and how they are
# scored; a chunk written under different values must not be merged.
_SCAN_KEYS = ("structure", "genome_fasta", "chrom", "start", "step", "window",
              "num_windows", "windows_per_task")
_DIGEST_PREFIX = "# params_digest="


def _params_digest(params: dict) -> str:
    blob = json.dumps({k: params.get(k) for k in _SCAN_KEYS}, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _chunk_digest(path: Path) -> str | None:
    """The params digest on a chunk CSV's first line, or None."""
    try:
        with open(path) as fh:
            first = fh.readline().strip()
    except OSError:
        return None
    return first[len(_DIGEST_PREFIX):] if first.startswith(_DIGEST_PREFIX) else None


def _chunk_rows(path: Path):
    """The data rows of a chunk CSV, without its digest line."""
    with open(path, newline="") as fh:
        yield from csv.DictReader(line for line in fh if not line.startswith("#"))


def _count_csv_rows(path: Path) -> int:
    return sum(1 for _ in _chunk_rows(path))


def _scan_chunk_job(params_path: str) -> None:
    """Score one SLURM array task's windows into ``chunk_<task>.csv``.

    Entry point of the worker script that tfbs_binding_scan(slurm=True)
    writes.  Each chunk CSV starts with the digest of the scan parameters.
    A chunk whose CSV holds all its rows under the current digest is
    skipped, so a resubmitted array redoes only unfinished chunks, and a
    chunk from a scan with another region, window or step is rewritten.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = json.loads(Path(params_path).read_text())
    digest = _params_digest(p)
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    w_from = task_id * p["windows_per_task"]
    w_to = min(w_from + p["windows_per_task"], p["num_windows"])
    csv_path = Path(p["chunks_dir"]) / f"chunk_{task_id:05d}.csv"
    if (csv_path.exists() and _chunk_digest(csv_path) == digest
            and _count_csv_rows(csv_path) >= w_to - w_from):
        logger.info("Chunk %d already complete; skipping.", task_id)
        return
    logger.info("Task %d: windows %d-%d of %d", task_id, w_from, w_to - 1, p["num_windows"])

    import pyfaidx

    template = _dna_template(p["structure"])
    genome = pyfaidx.Fasta(p["genome_fasta"])
    scorer = _WindowScorer(p["foldx_bin"], p["structure"], template, genome, p["chrom"], p["window"])
    try:
        with open(csv_path, "w", newline="") as fh:
            fh.write(f"{_DIGEST_PREFIX}{digest}\n")
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(["position", "energy"])
            for done, i in enumerate(range(w_from, w_to), start=1):
                w_start = p["start"] + i * p["step"]
                writer.writerow([w_start, scorer.window_energy(w_start)])
                fh.flush()
                if done % 10 == 0:
                    logger.info("Task %d: %d/%d windows done", task_id, done, w_to - w_from)
    finally:
        genome.close()
    logger.info("Task %d complete; wrote %s", task_id, csv_path)


def _merge_chunks_job(params_path: str) -> None:
    """Merge the chunk CSVs into the bigWig; entry point of the merge job.

    Only chunks carrying the digest of the current scan parameters are
    merged; a leftover chunk of an earlier scan in the same output_dir is
    skipped with a warning.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = json.loads(Path(params_path).read_text())
    digest = _params_digest(p)
    scores: list[tuple[int, float]] = []
    for chunk in sorted(Path(p["chunks_dir"]).glob("chunk_*.csv")):
        if _chunk_digest(chunk) != digest:
            logger.warning("Skipping %s: written by a scan with other parameters.", chunk)
            continue
        for row in _chunk_rows(chunk):
            try:
                energy = float(row["energy"])
            except (TypeError, ValueError):
                energy = math.nan
            scores.append((int(row["position"]), energy))
    n = _write_bigwig(Path(p["bigwig_path"]), p["chrom"], p["chrom_length"],
                      scores, p["window"], p["step"])
    logger.info("BigWig written to %s (%d of %d windows scored)", p["bigwig_path"], n, len(scores))


def _write_slurm_scan_scripts(out: Path, params_path: Path, n_tasks: int,
                              partition: str | None, max_time: str | None) -> tuple[Path, Path]:
    """Write the worker, merge and sbatch scripts; return the two sbatch paths."""
    root = str(Path(__file__).resolve().parents[2])
    python = shlex.quote(sys.executable)
    logs = out.resolve() / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def job_script(name: str, entry: str) -> str:
        path = out / name
        path.write_text(
            "import sys\n"
            f"sys.path.insert(0, {root!r})\n"
            f"from tfbs.mcp.foldx_tools import {entry}\n"
            f"{entry}({str(params_path.resolve())!r})\n",
            newline="\n",
        )
        return f"{python} {shlex.quote(str(path.resolve()))}"

    def sbatch_script(name: str, directives: list[str], command: str) -> Path:
        path = out / name
        lines = ["#!/bin/bash", *(f"#SBATCH {d}" for d in directives), "", "date", command, "date", ""]
        path.write_text("\n".join(lines), newline="\n")
        return path

    partition_flag = [f"--partition={partition}"] if partition else []
    array_sh = sbatch_script("foldx_scan.sh", [
        "--job-name=foldx_scan",
        f"--output={logs}/scan_%A_%a.out",
        f"--error={logs}/scan_%A_%a.err",
        f"--array=0-{n_tasks - 1}",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=8G",
        *partition_flag,
        f"--time={max_time or '48:00:00'}",
    ], job_script("foldx_scan_worker.py", "_scan_chunk_job"))
    merge_sh = sbatch_script("foldx_merge.sh", [
        "--job-name=foldx_merge",
        f"--output={logs}/merge_%j.out",
        f"--error={logs}/merge_%j.err",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=4G",
        *partition_flag,
        "--time=00:30:00",
    ], job_script("foldx_scan_merge.py", "_merge_chunks_job"))
    return array_sh, merge_sh


async def _sbatch(script: Path, sbatch_flags: list[str] | None = None) -> dict:
    """Submit through the slurm_submit tool; the dict has ``error`` on failure."""
    from tfbs.mcp import server

    try:
        result = json.loads(await server.slurm_submit(script_path=str(script), sbatch_flags=sbatch_flags))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if "error" not in result and not result.get("job_id"):
        result["error"] = "sbatch reported no job ID"
    return result


# ---------------------------------------------------------------------------
# Tool 3: tfbs_binding_scan
# ---------------------------------------------------------------------------
@mcp.tool(
    name="tfbs_binding_scan",
    annotations={
        "title": "FoldX Binding Scan",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def tfbs_binding_scan(
    structure_path: str,
    genome_fasta: str,
    region: str,
    output_dir: str | None = None,
    window: int | None = None,
    step: int = 50,
    slurm: bool = False,
    partition: str | None = None,
    max_time: str | None = None,
) -> str:
    """Slide a window across a genomic region and score TF binding with FoldX.

    Windows start every ``step`` bp.  Each window is scored by threading
    the genomic sequence onto the structure's DNA (FoldX BuildModel) and
    running AnalyseComplex; the window's value is the lowest (most
    favourable) interaction energy in kcal/mol over every placement of the
    structure's DNA inside the window.  Produces a bigWig track.

    Args:
        structure_path: Path to a PDB file of a TF-DNA complex that FoldX
                        RepairPDB has repaired: pass the
                        repaired_structure_path that tfbs_binding_energy
                        returns with repair=true, not the file
                        tfbs_fetch_structure downloaded.  It must contain
                        DNA chains (AlphaFold DB models do not).
        genome_fasta:   Path to a genome FASTA (an .fai index is built if
                        missing).
        region:         Genomic interval, e.g. ``chr1:1000-2000``.
        output_dir:     Where to write the bigWig.  Default
                        <results>/binding_scan/<digest>/, with <results>
                        the results_dir of tfbs_server_info and <digest> a
                        hash of the region, window, step and the path,
                        size and modification time of both the structure
                        and the genome FASTA, so a repeated call writes to
                        the same place, while scans against a different
                        or rewritten genome or structure never share one.
        window:         Window size in bp.  Default None means the DNA
                        length of the structure (e.g. 11 bp), one FoldX run
                        per window.  An explicit value is used as given,
                        even when it is much larger than the DNA: every
                        placement of the DNA inside the window is scored,
                        so a window costs window - DNA length + 1 FoldX
                        runs (shared between overlapping windows), and the
                        result then carries a "warning" with the run count
                        (placements x windows).  A value below the DNA
                        length is rejected.
        step:           Distance between window starts in bp (default 50).
                        With step > window the positions between windows
                        are not scored; use step <= window for gap-free
                        coverage.
        slurm:          Submit a SLURM array job (one CSV per chunk of
                        windows, completed chunks skipped on resubmission)
                        plus a dependent merge job that writes the bigWig,
                        instead of scanning in-process.
        partition:      SLURM partition name.  Default None submits
                        without --partition, so the scheduler's default
                        partition is used, because FoldX runs on CPUs
                        only and needs no GPU partition.
        max_time:       SLURM max wall time per array task (default
                        ``"48:00:00"``).

    Returns:
        JSON with ``bigwig_path``, ``num_windows``, and the ``window``,
        ``dna_length``, ``placements`` (DNA placements per window) and
        ``step`` actually used, plus ``foldx_runs_planned`` (distinct
        placements to score; with slurm=true a lower bound, because array
        tasks do not share the placements of overlapping windows at
        their chunk boundaries); with slurm=true, also ``array_job_id``
        and ``merge_job_id``.  ``warning`` reports the FoldX run count when a
        window holds more than one placement.  When windows overlap, each
        value covers the step-wide bigWig bin centred on its window.
    """
    # --- Validate inputs -----------------------------------------------
    pdb = Path(structure_path).resolve()
    if not pdb.is_file():
        return _error(
            f"Structure file not found: {structure_path}",
            "Pass a PDB file of a TF-DNA complex, for example the "
            "structure_path returned by tfbs_fetch_structure.",
        )

    fasta = Path(genome_fasta).resolve()
    if not fasta.is_file():
        return _error(
            f"Genome FASTA not found: {genome_fasta}",
            "Pass the path of the genome FASTA that the region refers to.",
        )

    foldx = _find_foldx()
    if foldx is None:
        return _foldx_missing_error()

    m = re.match(r"^([^\s:]+):(\d+)-(\d+)$", region)
    if not m:
        return _error(
            f"Invalid region format: '{region}'. Expected chrom:start-end",
            "Use chrom:start-end with integer coordinates and the contig name as it appears in the "
            "genome FASTA, e.g. chr14:36500000-36520000.",
        )
    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    if end <= start:
        return _error(f"Region end ({end}) is not after its start ({start}).",
                      "Pass the region as chrom:start-end with start < end.")
    if step < 1:
        return _error(f"step must be at least 1 bp, got {step}.", "Pass step >= 1 (default 50).")

    try:
        import pyBigWig  # noqa: F401
        import pyfaidx
    except ImportError as exc:
        return _error(
            f"Missing Python package for the scan: {exc.name}",
            'Install it with: pip install "tfbs-mcp[genomics,foldx]" '
            "(pyfaidx and pyBigWig), then restart the server.",
        )

    try:
        template = _dna_template(str(pdb))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return _error(f"Cannot parse {pdb} as PDB: {exc}", "Pass a PDB-format file (not mmCIF).")
    if template is None:
        return _error(
            "No DNA chains found in the PDB structure.",
            "The scan threads genomic sequence onto the structure's DNA, so it "
            "needs a TF-DNA complex. AlphaFold DB models are protein-only; use a "
            "PDB co-crystal or an AlphaFold 3 complex saved as PDB.",
        )
    dna_length = len(template.fwd_bases)
    if window is None:
        window = dna_length
    elif window < dna_length:
        return _error(
            f"window ({window} bp) is shorter than the DNA in the structure ({dna_length} bp).",
            f"Pass window >= {dna_length}, or omit window to use the DNA length.",
        )
    num_windows = max(1, (end - start - window) // step + 1)
    placements = window - dna_length + 1
    if step >= placements:
        runs_planned = placements * num_windows
    else:
        runs_planned = (num_windows - 1) * step + placements
    warnings: list[str] = []
    if placements > 1:
        warnings.append(
            f"window {window} bp holds {placements} placements of the {dna_length} bp "
            f"structure DNA, so the scan costs {placements} placements x {num_windows} "
            f"windows = {placements * num_windows} FoldX runs"
            + (f" ({runs_planned} distinct, shared between overlapping windows)"
               if runs_planned < placements * num_windows else "")
            + "; omit window to score one placement per window."
        )

    try:
        genome = pyfaidx.Fasta(str(fasta))
    except Exception as exc:
        return _error(f"Cannot open genome FASTA {fasta}: {exc}",
                      "Check the FASTA file, and that its folder is writable for the .fai index.")
    if chrom not in genome.keys():
        genome.close()
        return _error(f"Chromosome '{chrom}' is not in {fasta}.",
                      "Use a chromosome name exactly as it appears in the FASTA headers.")
    chrom_length = len(genome[chrom])

    out = (Path(output_dir).expanduser() if output_dir
           else _server._results_root() / "binding_scan"
           / _server._digest(region, window, step, _server._file_identity(str(pdb)),
                             _server._file_identity(str(fasta))))
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        genome.close()
        return _error(f"Cannot create output directory {out}: {exc}",
                      "Pass a writable output_dir, or set TFBS_PROJECT_ROOT to a writable "
                      "directory (tfbs_server_info reports the results_dir in use).")
    bw_path = out.resolve() / "foldx_binding_scan.bw"
    used = {
        "region": region,
        "window": window,
        "dna_length": dna_length,
        "placements": placements,
        "step": step,
        "num_windows": num_windows,
        "foldx_runs_planned": runs_planned,
        "output_dir": str(out.resolve()),
        "bigwig_path": str(bw_path),
    }
    if warnings:
        used["warning"] = " ".join(warnings)

    # --- SLURM mode (array job + dependent merge job) -------------------
    if slurm:
        genome.close()
        windows_per_task = max(1, _FOLDX_RUNS_PER_TASK // placements)
        n_tasks = (num_windows + windows_per_task - 1) // windows_per_task
        chunks_dir = out.resolve() / "chunks"
        params_path = out.resolve() / "foldx_scan_params.json"
        params = {
            "structure": str(pdb),
            "genome_fasta": str(fasta),
            "chrom": chrom,
            "chrom_length": chrom_length,
            "start": start,
            "step": step,
            "window": window,
            "num_windows": num_windows,
            "windows_per_task": windows_per_task,
            "foldx_bin": foldx,
            "chunks_dir": str(chunks_dir),
            "bigwig_path": str(bw_path),
        }
        params["params_digest"] = _params_digest(params)
        try:
            chunks_dir.mkdir(parents=True, exist_ok=True)
            params_path.write_text(json.dumps(params, indent=2), newline="\n")
            array_sh, merge_sh = _write_slurm_scan_scripts(out, params_path, n_tasks, partition, max_time)
        except OSError as exc:
            return _error(f"Cannot write the SLURM job files to {out}: {exc}",
                          "Pass a writable output_dir on a filesystem the cluster nodes can read.")

        array = await _sbatch(array_sh)
        if "error" in array:
            return _error(
                f"Submitting the FoldX scan array job failed: {array['error']}",
                "Run with slurm=false to scan in-process, or start the MCP server "
                "on a SLURM login node where sbatch is on PATH.",
                mode="slurm_array", stage="array", stderr=array.get("stderr"),
                array_script=str(array_sh), merge_script=str(merge_sh),
            )
        array_job_id = array["job_id"]

        merge = await _sbatch(merge_sh, [f"--dependency=afterany:{array_job_id}"])
        if "error" in merge:
            return _error(
                f"The array job {array_job_id} was submitted, but submitting the "
                f"merge job failed: {merge['error']}",
                f"Submit the merge yourself with: sbatch --dependency=afterany:{array_job_id} "
                f"{merge_sh}; or cancel the array job with slurm_cancel.",
                mode="slurm_array", stage="merge", array_job_id=array_job_id,
                stderr=merge.get("stderr"), merge_script=str(merge_sh),
            )

        return json.dumps({
            "mode": "slurm_array",
            "array_job_id": array_job_id,
            "merge_job_id": merge["job_id"],
            "num_tasks": n_tasks,
            "windows_per_task": windows_per_task,
            "chunks_dir": str(chunks_dir),
            "params_path": str(params_path),
            "params_digest": params["params_digest"],
            **used,
        })

    # --- Local mode ----------------------------------------------------
    scorer = _WindowScorer(foldx, str(pdb), template, genome, chrom, window)
    scores: list[tuple[int, float]] = []
    try:
        for i in range(num_windows):
            w_start = start + i * step
            scores.append((w_start, scorer.window_energy(w_start)))
            if (i + 1) % 100 == 0:
                logger.info("FoldX scan: %d/%d windows processed", i + 1, num_windows)
    finally:
        genome.close()

    try:
        num_scored = _write_bigwig(bw_path, chrom, chrom_length, scores, window, step)
    except Exception as exc:
        return _error(f"Writing the bigWig failed: {exc}",
                      "Check that output_dir is writable and pyBigWig is installed.", **used)

    result = {"mode": "local", "num_scored": num_scored, "foldx_runs": scorer.runs, **used}
    if num_scored == 0:
        warnings.append("No window could be scored: the region has no ACGT-only "
                        "sequence or FoldX failed on every placement.")
    if warnings:
        result["warning"] = " ".join(warnings)
    return json.dumps(result)
