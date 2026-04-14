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

_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


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


def _build_foldx_mutation_string(
    ref_bases: list[tuple[int, str]],
    target_seq: str,
    chain: str,
) -> str:
    """Build a FoldX mutation string to convert reference DNA to target.

    FoldX DNA mutation syntax uses lowercase: ``aC3t`` means residue A
    at chain C position 3 → T.

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
    """Run FoldX BuildModel (if mutations) + AnalyseComplex, return ΔG.

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
        mut_file.write_text(mutation_string + "\n")
        try:
            r = subprocess.run(
                [foldx_bin, "--command=BuildModel",
                 f"--pdb={pdb_name}", f"--mutant-file={mut_file.name}"],
                capture_output=True, text=True, timeout=1200, cwd=str(wd),
            )
        except subprocess.TimeoutExpired:
            return None
        mutated = wd / f"{Path(pdb_name).stem}_1.pdb"
        if mutated.exists():
            analyse_target = mutated.name
        else:
            return None

    # AnalyseComplex
    r = subprocess.run(
        [foldx_bin, "--command=AnalyseComplex",
         f"--pdb={analyse_target}", "--complexWithDNA=true"],
        capture_output=True, text=True, timeout=60, cwd=str(wd),
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
            \"\"\"FoldX binding scan – auto-generated by tfbs_binding_scan.\"\"\"
            import json, math, tempfile, shutil, subprocess, sys
            from pathlib import Path
            import pyfaidx
            import pyBigWig

            sys.path.insert(0, "{Path(__file__).resolve().parent.parent.parent}")
            from tfbs.mcp.foldx_tools import (
                _parse_dna_from_pdb, _build_foldx_mutation_string,
                _run_foldx_energy, _COMPLEMENT,
            )

            structure = "{pdb}"
            fasta_path = "{fasta}"
            chrom     = "{chrom}"
            start     = {start}
            end       = {end}
            step      = {step}
            out_dir   = Path("{out.resolve()}")
            foldx_bin = "{foldx}"

            # Parse DNA from structure
            dna_chains = _parse_dna_from_pdb(structure)
            fwd_chain = sorted(dna_chains.keys())[0]
            rev_chain = sorted(dna_chains.keys())[1] if len(dna_chains) > 1 else None
            ref_fwd = dna_chains[fwd_chain]
            ref_rev = dna_chains[rev_chain] if rev_chain else None
            dna_len = len(ref_fwd)
            window = dna_len

            num_windows = max(1, (end - start - window) // step + 1)
            print(f"Scanning {{num_windows}} windows ({{chrom}}:{{start}}-{{end}}, DNA len={{dna_len}})")

            genome = pyfaidx.Fasta(fasta_path)
            chrom_len = len(genome[chrom])
            positions, energies = [], []

            for i in range(num_windows):
                w_start = start + i * step
                w_end = w_start + window
                try:
                    seq_fwd = str(genome[chrom][w_start:w_end]).upper()
                except (KeyError, ValueError):
                    positions.append(w_start)
                    energies.append(float("nan"))
                    continue
                if len(seq_fwd) != dna_len or any(c not in "ACGT" for c in seq_fwd):
                    positions.append(w_start)
                    energies.append(float("nan"))
                    continue

                mut_parts = []
                mut_fwd = _build_foldx_mutation_string(ref_fwd, seq_fwd, fwd_chain)
                if mut_fwd:
                    mut_parts.append(mut_fwd.rstrip(";"))
                if rev_chain and ref_rev:
                    seq_rev = "".join(_COMPLEMENT.get(b, "N") for b in reversed(seq_fwd))
                    mut_rev = _build_foldx_mutation_string(ref_rev, seq_rev, rev_chain)
                    if mut_rev:
                        mut_parts.append(mut_rev.rstrip(";"))
                mutation_str = ",".join(mut_parts) + ";" if mut_parts else None

                try:
                    with tempfile.TemporaryDirectory(prefix="fxw_") as wdir:
                        energy = _run_foldx_energy(foldx_bin, structure, wdir, mutation_str)
                        positions.append(w_start)
                        energies.append(energy if energy is not None else float("nan"))
                except Exception as exc:
                    positions.append(w_start)
                    energies.append(float("nan"))
                    print(f"  Window {{i}}: error - {{exc}}")

                if (i + 1) % 100 == 0:
                    n_valid = sum(1 for e in energies if not math.isnan(e))
                    print(f"  {{i+1}}/{{num_windows}} windows done ({{n_valid}} valid)")

            genome.close()

            # Write bigwig
            valid = [(p, e) for p, e in zip(positions, energies) if not math.isnan(e)]
            bw_path = out_dir / "foldx_binding_scan.bw"
            bw = pyBigWig.open(str(bw_path), "w")
            bw.addHeader([(chrom, chrom_len)])
            if valid:
                ps, es = zip(*valid)
                bw.addEntries([chrom]*len(ps), list(ps),
                              ends=[p+step for p in ps], values=[float(e) for e in es])
            bw.close()
            print(f"Done. Wrote {{len(valid)}} scores to {{bw_path}}")
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

    # Parse reference DNA from the structure to know what to mutate
    dna_chains = _parse_dna_from_pdb(str(pdb))
    if not dna_chains:
        return json.dumps({"error": "No DNA chains found in the PDB structure."})

    # Use the first DNA chain (forward strand) as the mutation target
    fwd_chain = sorted(dna_chains.keys())[0]
    rev_chain = sorted(dna_chains.keys())[1] if len(dna_chains) > 1 else None
    ref_bases_fwd = dna_chains[fwd_chain]
    ref_bases_rev = dna_chains[rev_chain] if rev_chain else None
    dna_len = len(ref_bases_fwd)

    # Override window to match DNA length in the structure
    if window != dna_len:
        window = dna_len
        num_windows = max(1, (end - start - window) // step + 1)

    genome = pyfaidx.Fasta(str(fasta))

    scores: list[tuple[int, int, float]] = []
    for i in range(num_windows):
        w_start = start + i * step
        w_end = w_start + window
        try:
            seq_fwd = str(genome[chrom][w_start:w_end]).upper()
        except (KeyError, ValueError):
            scores.append((w_start, w_end, float("nan")))
            continue

        if len(seq_fwd) != dna_len or any(c not in "ACGT" for c in seq_fwd):
            scores.append((w_start, w_end, float("nan")))
            continue

        # Build mutation strings for forward and reverse complement chains
        mut_fwd = _build_foldx_mutation_string(ref_bases_fwd, seq_fwd, fwd_chain)
        mut_parts = []
        if mut_fwd:
            mut_parts.append(mut_fwd.rstrip(";"))
        if rev_chain and ref_bases_rev:
            seq_rev = "".join(_COMPLEMENT.get(b, "N") for b in reversed(seq_fwd))
            mut_rev = _build_foldx_mutation_string(ref_bases_rev, seq_rev, rev_chain)
            if mut_rev:
                mut_parts.append(mut_rev.rstrip(";"))

        mutation_string = ",".join(mut_parts) + ";" if mut_parts else None

        # Run FoldX in a fresh temp dir per window to avoid file collisions
        try:
            with tempfile.TemporaryDirectory(prefix="fxw_") as wdir:
                energy = _run_foldx_energy(foldx, str(pdb), wdir, mutation_string)
                scores.append((w_start, w_end, energy if energy is not None else float("nan")))
        except Exception:
            scores.append((w_start, w_end, float("nan")))

        if (i + 1) % 100 == 0:
            print(f"  FoldX scan: {i + 1}/{num_windows} windows processed")

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

    import math
    valid_scores = [(s[0], s[1], s[2]) for s in scores
                    if not (isinstance(s[2], float) and math.isnan(s[2]))]
    if valid_scores:
        chroms_list = [chrom] * len(valid_scores)
        starts_list = [s[0] for s in valid_scores]
        ends_list = [s[1] for s in valid_scores]
        vals_list = [s[2] for s in valid_scores]
        bw.addEntries(chroms_list, starts_list, ends=ends_list, values=vals_list)

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
