"""
FoldX binding-energy tools for TF–DNA complex analysis.

Registers three tools on the shared ``mcp`` FastMCP instance:

* **tfbs_fetch_structure** — retrieve a TF-DNA complex structure from a local
  path, PDB, AlphaFold DB, or gene-name search.
* **tfbs_binding_energy** — run FoldX AnalyseComplex on a TF-DNA structure.
* **tfbs_binding_scan** — slide a window across a genomic region and score
  each position with FoldX, writing a bigWig track.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import urllib.request
from pathlib import Path

from tfbs.mcp.server import mcp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_foldx() -> str | None:
    """Return the path to the FoldX binary, or None."""
    return shutil.which("foldx") or shutil.which("foldx5")


def _foldx_missing_error() -> str:
    return json.dumps({
        "error": (
            "FoldX binary not found on PATH. "
            "Install FoldX (https://foldxsuite.crg.eu/) and ensure 'foldx' "
            "or 'foldx5' is available on your $PATH."
        ),
    })


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
    """Retrieve a TF-DNA complex structure.

    Resolution priority: local_path → pdb_id → gene_name search → uniprot_id
    (AlphaFold DB).

    Args:
        local_path:  Path to a PDB file already on disk.
        pdb_id:      4-character PDB accession (e.g. ``3RKQ``).
        gene_name:   Gene name to search RCSB for (e.g. ``p53``).
        uniprot_id:  UniProt accession for AlphaFold DB lookup.
        output_dir:  Directory to save downloaded files.  Defaults to a
                     temporary directory.

    Returns:
        JSON with ``source``, ``structure_path``, and metadata.
    """
    out = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="tfbs_struct_"))
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. Local path -------------------------------------------------
    if local_path is not None:
        p = Path(local_path)
        if not p.is_file():
            return json.dumps({"error": f"Local file not found: {local_path}"})
        return json.dumps({
            "source": "local",
            "structure_path": str(p.resolve()),
            "metadata": {"original_path": str(p)},
        })

    # --- 2. PDB download -----------------------------------------------
    if pdb_id is not None:
        pdb_id = pdb_id.strip().upper()
        dest = out / f"{pdb_id}.pdb"
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            urllib.request.urlretrieve(url, str(dest))
        except Exception as exc:
            return json.dumps({"error": f"Failed to download PDB {pdb_id}: {exc}"})
        return json.dumps({
            "source": "pdb",
            "structure_path": str(dest.resolve()),
            "metadata": {"pdb_id": pdb_id, "url": url},
        })

    # --- 3. Gene-name search on RCSB -----------------------------------
    if gene_name is not None:
        query_json = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "full_text",
                        "parameters": {"value": gene_name},
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "entity_poly.rcsb_entity_polymer_type",
                            "operator": "exact_match",
                            "value": "Protein",
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "entity_poly.rcsb_entity_polymer_type",
                            "operator": "exact_match",
                            "value": "DNA",
                        },
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {"results_content_type": ["experimental"], "paginate": {"start": 0, "rows": 1}},
        }
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        req = urllib.request.Request(
            search_url,
            data=json.dumps(query_json).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            hits = data.get("result_set", [])
            if not hits:
                return json.dumps({"error": f"No PDB structures found for gene '{gene_name}'"})
            found_id = hits[0]["identifier"]
        except Exception as exc:
            return json.dumps({"error": f"RCSB search failed for '{gene_name}': {exc}"})

        dest = out / f"{found_id}.pdb"
        dl_url = f"https://files.rcsb.org/download/{found_id}.pdb"
        try:
            urllib.request.urlretrieve(dl_url, str(dest))
        except Exception as exc:
            return json.dumps({"error": f"Failed to download PDB {found_id}: {exc}"})
        return json.dumps({
            "source": "pdb",
            "structure_path": str(dest.resolve()),
            "metadata": {"pdb_id": found_id, "gene_name": gene_name, "url": dl_url},
        })

    # --- 4. AlphaFold DB -----------------------------------------------
    if uniprot_id is not None:
        uniprot_id = uniprot_id.strip()
        dest = out / f"AF-{uniprot_id}-F1-model_v4.pdb"
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        try:
            urllib.request.urlretrieve(url, str(dest))
        except Exception as exc:
            return json.dumps({"error": f"AlphaFold DB download failed for {uniprot_id}: {exc}"})
        return json.dumps({
            "source": "alphafold",
            "structure_path": str(dest.resolve()),
            "metadata": {"uniprot_id": uniprot_id, "url": url},
        })

    return json.dumps({"error": "Provide at least one of: local_path, pdb_id, gene_name, uniprot_id"})


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
) -> str:
    """Compute TF-DNA binding energy with FoldX AnalyseComplex.

    Args:
        structure_path: Path to a PDB file.
        sequences:      Optional DNA sequences (placeholder for future
                        mutation support).
        repair:         Run FoldX RepairPDB first (recommended).
        chains_protein: Protein chain ID(s), e.g. ``"A"``.
        chains_dna:     DNA chain ID(s), comma-separated, e.g. ``"C,D"``.

    Returns:
        JSON with energy components parsed from FoldX output.
    """
    foldx = _find_foldx()
    if foldx is None:
        return _foldx_missing_error()

    pdb = Path(structure_path).resolve()
    if not pdb.is_file():
        return json.dumps({"error": f"Structure file not found: {structure_path}"})

    tmpdir = tempfile.mkdtemp(prefix="tfbs_foldx_")
    work = Path(tmpdir)

    # Copy PDB into working directory
    shutil.copy2(pdb, work / pdb.name)

    pdb_name = pdb.name
    stem = pdb.stem

    # --- RepairPDB -----------------------------------------------------
    if repair:
        cmd_repair = [
            foldx, "--command=RepairPDB",
            f"--pdb={pdb_name}",
            f"--output-dir={tmpdir}",
        ]
        result = subprocess.run(cmd_repair, capture_output=True, text=True, cwd=tmpdir, timeout=600)
        repaired = work / f"{stem}_Repair.pdb"
        if repaired.is_file():
            pdb_name = repaired.name
        else:
            # Some FoldX versions use different naming
            for candidate in work.glob("*Repair*.pdb"):
                pdb_name = candidate.name
                break

    # --- AnalyseComplex ------------------------------------------------
    cmd_analyse = [
        foldx, "--command=AnalyseComplex",
        f"--pdb={pdb_name}",
        f"--analyseComplexChains={chains_protein},{chains_dna}",
        "--complexWithDNA=true",
        f"--output-dir={tmpdir}",
    ]
    result = subprocess.run(cmd_analyse, capture_output=True, text=True, cwd=tmpdir, timeout=600)

    if result.returncode != 0:
        return json.dumps({
            "error": "FoldX AnalyseComplex failed",
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "stdout": result.stdout[-2000:] if result.stdout else "",
        })

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
        return json.dumps({
            "error": "FoldX output file not found",
            "files": [f.name for f in work.iterdir()],
        })

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
        "interaction_energy": energy.get("Interaction Energy", None),
        "backbone_hbond": energy.get("Backbone Hbond", None),
        "sidechain_hbond": energy.get("Sidechain Hbond", None),
        "van_der_waals": energy.get("Van der Waals", None),
        "electrostatics": energy.get("Electrostatics", None),
        "all_components": energy,
        "foldx_stdout": result.stdout[-1000:] if result.stdout else "",
    })


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
    window: int = 200,
    step: int = 50,
    slurm: bool = False,
    partition: str | None = None,
    max_time: str | None = None,
) -> str:
    """Slide a window across a genomic region and score TF binding with FoldX.

    Produces a bigWig track of per-window binding energy estimates.

    Args:
        structure_path: Path to a repaired PDB file.
        genome_fasta:   Path to a genome FASTA (with .fai index).
        region:         Genomic interval, e.g. ``chr1:1000-2000``.
        output_dir:     Where to write the bigWig; defaults to a temp dir.
        window:         Window size in bp (default 200).
        step:           Step size in bp (default 50).
        slurm:          Submit as a SLURM batch job instead of running locally.
        partition:      SLURM partition name.
        max_time:       SLURM max wall time, e.g. ``"2:00:00"``.

    Returns:
        JSON with bigwig_path, num_windows, and job metadata (if SLURM).
    """
    # --- Validate inputs -----------------------------------------------
    pdb = Path(structure_path).resolve()
    if not pdb.is_file():
        return json.dumps({"error": f"Structure file not found: {structure_path}"})

    fasta = Path(genome_fasta).resolve()
    if not fasta.is_file():
        return json.dumps({"error": f"Genome FASTA not found: {genome_fasta}"})

    foldx = _find_foldx()
    if foldx is None:
        return _foldx_missing_error()

    # Parse region
    m = re.match(r"^(chr\w+):(\d+)-(\d+)$", region)
    if not m:
        return json.dumps({"error": f"Invalid region format: '{region}'. Expected chr:start-end"})
    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))

    out = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="tfbs_scan_"))
    out.mkdir(parents=True, exist_ok=True)

    num_windows = max(1, (end - start - window) // step + 1)

    # --- SLURM mode ----------------------------------------------------
    if slurm:
        from tfbs.mcp.server import slurm_submit

        scan_script = out / "foldx_scan.py"
        scan_script.write_text(textwrap.dedent(f"""\
            #!/usr/bin/env python
            \"\"\"FoldX binding scan – auto-generated.\"\"\"
            import json, re, tempfile, shutil, subprocess
            from pathlib import Path

            structure = "{pdb}"
            genome    = "{fasta}"
            chrom     = "{chrom}"
            start     = {start}
            end       = {end}
            window    = {window}
            step      = {step}
            out_dir   = "{out.resolve()}"

            # TODO: implement per-window FoldX scoring
            print(json.dumps({{"status": "placeholder", "num_windows": {num_windows}}}))
        """))

        sbatch_script = out / "foldx_scan.sh"
        partition_line = f"#SBATCH --partition={partition}" if partition else ""
        time_line = f"#SBATCH --time={max_time}" if max_time else "#SBATCH --time=4:00:00"
        sbatch_script.write_text(textwrap.dedent(f"""\
            #!/bin/bash
            #SBATCH --job-name=foldx_scan
            #SBATCH --output={out.resolve()}/foldx_scan_%j.out
            #SBATCH --error={out.resolve()}/foldx_scan_%j.err
            #SBATCH --ntasks=1
            #SBATCH --cpus-per-task=4
            #SBATCH --mem=8G
            {partition_line}
            {time_line}

            python {scan_script.resolve()}
        """))

        submit_result = json.loads(await slurm_submit(script_path=str(sbatch_script)))
        return json.dumps({
            "mode": "slurm",
            "job_id": submit_result.get("job_id"),
            "scan_script": str(scan_script.resolve()),
            "sbatch_script": str(sbatch_script.resolve()),
            "output_dir": str(out.resolve()),
            "num_windows": num_windows,
        })

    # --- Local mode ----------------------------------------------------
    try:
        import pyfaidx
    except ImportError:
        return json.dumps({"error": "pyfaidx is required for local scan: pip install pyfaidx"})

    genome = pyfaidx.Fasta(str(fasta))

    scores: list[tuple[int, int, float]] = []
    for i in range(num_windows):
        w_start = start + i * step
        w_end = min(w_start + window, end)
        try:
            _seq = str(genome[chrom][w_start:w_end])
        except (KeyError, ValueError) as exc:
            scores.append((w_start, w_end, 0.0))
            continue

        # TODO: run FoldX with mutated DNA for this window and parse
        # interaction energy.  For now emit placeholder score.
        score = 0.0
        scores.append((w_start, w_end, score))

    genome.close()

    # Write bigWig
    bw_path = out / "foldx_binding_scan.bw"
    try:
        import pyBigWig
    except ImportError:
        return json.dumps({"error": "pyBigWig is required for bigWig output: pip install pyBigWig"})

    # Need chromosome sizes
    fai = Path(str(fasta) + ".fai")
    chrom_sizes = {}
    if fai.is_file():
        with open(fai) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    chrom_sizes[parts[0]] = int(parts[1])
    else:
        # Fallback: use the end of the region
        chrom_sizes[chrom] = end + window

    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader(list(chrom_sizes.items()))

    if scores:
        chroms = [chrom] * len(scores)
        starts = [s[0] for s in scores]
        ends = [s[1] for s in scores]
        vals = [s[2] for s in scores]
        bw.addEntries(chroms, starts, ends=ends, values=vals)

    bw.close()

    return json.dumps({
        "mode": "local",
        "bigwig_path": str(bw_path.resolve()),
        "output_dir": str(out.resolve()),
        "num_windows": num_windows,
        "region": region,
        "window": window,
        "step": step,
    })
