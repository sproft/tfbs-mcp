"""
Disease-variant validation for TF binding prediction models.

Generic pipeline that works for any transcription factor:

1. Take a BED file of TF binding peaks (e.g. from ChIP-Atlas).
2. Pull non-coding pathogenic and benign SNVs from a ClinVar VCF that
   overlap those peaks.
3. Score each variant's reference and alternative allele with one or
   more NN models (ΔΔ NN score), with FIMO against a JASPAR PWM
   (ΔΔ FIMO log-odds score), and optionally with FoldX on a TF-DNA
   complex (ΔΔG in kcal/mol).
4. Test whether |ΔΔ| discriminates pathogenic from benign variants
   (ROC-AUC, Mann-Whitney U).
5. Generate boxplots, ROC curves, and a summary table.

Usage example:

    python scripts/disease_variant_validation.py \\
        --peaks-bed data/chipSeq/Markus/INTERSECTIONS/NKX2-1.allintersect.bed \\
        --clinvar-vcf data/clinvar/clinvar.vcf.gz \\
        --genome /path/to/hg38.fa \\
        --output-dir results/disease_validation \\
        --models all_mean=saved_models_final/all_mean/standardize/best_VCNNBpnet.ckpt \\
                 flank_mean=saved_models_final/flank_mean/standardize/best_VCNNBpnet.ckpt \\
        --motif-file data/JASPAR/MA1994.1.meme \\
        --foldx-pdb data/structures/repaired/NKX2-1_complex_CAB_Repair.pdb \\
        --peak-confidence 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfaidx
import pysam
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve

NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

PATHOGENIC_TERMS = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BENIGN_TERMS = {"Benign", "Likely_benign", "Benign/Likely_benign"}
EXCLUDE_SO = {
    "missense_variant", "synonymous_variant",
    "stop_gained", "stop_lost", "start_lost",
    "frameshift_variant", "inframe_insertion", "inframe_deletion",
    "splice_acceptor_variant", "splice_donor_variant",
    "initiator_codon_variant",
}


# ---------------------------------------------------------------------------
# Step 1: BED merging
# ---------------------------------------------------------------------------

def merge_bed_intervals(bed_path: str, min_confidence: int = 1) -> pd.DataFrame:
    """Read BED file (with optional 'num' column), merge overlapping intervals."""
    df = pd.read_csv(bed_path, sep="\t", header=None, comment="#",
                     names=["chrom", "start", "end", "num"]
                     if _bed_has_num_column(bed_path) else
                     ["chrom", "start", "end"])
    if "num" not in df.columns:
        df["num"] = 1
    df = df.sort_values(["chrom", "start"]).reset_index(drop=True)
    merged = []
    cur_chrom = cur_start = cur_end = cur_num = None
    for _, row in df.iterrows():
        if cur_chrom is None or row["chrom"] != cur_chrom or row["start"] > cur_end:
            if cur_chrom is not None:
                merged.append((cur_chrom, cur_start, cur_end, cur_num))
            cur_chrom = row["chrom"]
            cur_start = int(row["start"])
            cur_end = int(row["end"])
            cur_num = int(row["num"])
        else:
            cur_end = max(cur_end, int(row["end"]))
            cur_num = max(cur_num, int(row["num"]))
    if cur_chrom is not None:
        merged.append((cur_chrom, cur_start, cur_end, cur_num))
    out = pd.DataFrame(merged, columns=["chrom", "start", "end", "num"])
    return out[out["num"] >= min_confidence].reset_index(drop=True)


def _bed_has_num_column(path: str) -> bool:
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                return True
            return False
    return False


# ---------------------------------------------------------------------------
# Step 2: ClinVar variant extraction
# ---------------------------------------------------------------------------

def extract_clinvar_snvs(peaks: pd.DataFrame, clinvar_vcf: str) -> pd.DataFrame:
    """Find non-coding pathogenic and benign SNVs in the given peaks."""
    vcf = pysam.VariantFile(clinvar_vcf)
    hits = []
    for peak in peaks.itertuples():
        chrom_no_prefix = peak.chrom.replace("chr", "")
        try:
            for rec in vcf.fetch(chrom_no_prefix, peak.start, peak.end):
                if len(rec.ref) != 1 or any(len(a) != 1 for a in rec.alts):
                    continue
                clnsig = rec.info.get("CLNSIG", ())
                clnsig = ",".join(clnsig) if isinstance(clnsig, tuple) else str(clnsig)
                if any(p in clnsig for p in PATHOGENIC_TERMS):
                    label = "pathogenic"
                elif any(b in clnsig for b in BENIGN_TERMS):
                    label = "benign"
                else:
                    continue
                mc = rec.info.get("MC", ())
                mc_terms = "|".join(mc) if isinstance(mc, tuple) else str(mc)
                so_terms = set()
                for term in mc_terms.replace(",", "|").split("|"):
                    if "|" in term:
                        so_terms.add(term.split("|")[-1])
                    else:
                        so_terms.add(term)
                if so_terms & EXCLUDE_SO:
                    continue
                gene = rec.info.get("GENEINFO", "").split(":")[0] if rec.info.get("GENEINFO") else "?"
                hits.append({
                    "chrom": peak.chrom,
                    "pos": rec.pos,
                    "ref": rec.ref,
                    "alt": rec.alts[0],
                    "clinsig": clnsig,
                    "label": label,
                    "gene": gene,
                    "mc": mc_terms,
                    "rsid": rec.id or "",
                    "peak_num": peak.num,
                })
        except (ValueError, KeyError):
            continue
    vcf.close()
    return pd.DataFrame(hits)


# ---------------------------------------------------------------------------
# Step 3: Per-method scoring
# ---------------------------------------------------------------------------

def encode_seq(seq: str) -> torch.Tensor:
    return F.one_hot(
        torch.tensor([NUC_TO_IDX[c] for c in seq.upper()], dtype=torch.long), 4
    ).float().permute(1, 0)


def score_nn(model, sequences, batch_size=512, device="cuda"):
    out = []
    for i in range(0, len(sequences), batch_size):
        batch = torch.stack([encode_seq(s) for s in sequences[i:i + batch_size]]).to(device)
        with torch.no_grad():
            p = model(batch).cpu().numpy().squeeze()
        if p.ndim == 0:
            p = np.array([float(p)])
        out.append(p)
    return np.concatenate(out)


def extract_seq_pair(genome, chrom, pos, ref, alt, half_window):
    """Return (ref_seq, alt_seq) of length 2*half_window centered on the variant.

    Returns (None, None) if reference doesn't match the genome or sequence
    contains ambiguous bases.
    """
    s = pos - 1 - half_window
    e = s + 2 * half_window
    try:
        seq = str(genome[chrom][s:e]).upper()
    except (KeyError, ValueError):
        return None, None
    if len(seq) != 2 * half_window:
        return None, None
    if any(c not in NUC_TO_IDX for c in seq):
        return None, None
    if seq[half_window] != ref:
        return None, None
    alt_seq = seq[:half_window] + alt + seq[half_window + 1:]
    return seq, alt_seq


def score_with_nn(df: pd.DataFrame, models: dict, genome, window: int = 24, device: str = "cuda") -> pd.DataFrame:
    """Add NN ΔΔ columns for each model in `models` (name -> path)."""
    half = window // 2
    out_df = df.copy()
    keep_idx = []
    ref_seqs, alt_seqs = [], []
    for i, row in df.iterrows():
        rs, as_ = extract_seq_pair(genome, row["chrom"], int(row["pos"]),
                                   row["ref"], row["alt"], half)
        if rs is not None:
            ref_seqs.append(rs)
            alt_seqs.append(as_)
            keep_idx.append(i)
    out_df = df.iloc[keep_idx].reset_index(drop=True)

    import tfbs.nn.models as models_mod
    for name, ckpt_path in models.items():
        cls_name = "VCNNBpnet"
        # Try common architectures; default VCNNBpnet
        for candidate in ("VCNNBpnet", "RNN"):
            if hasattr(models_mod, candidate):
                cls_name = candidate
                break
        # If the user encoded the model class in the dict somehow, use it
        # (defaulting to VCNNBpnet for backwards compatibility)
        cls = getattr(models_mod, cls_name)
        m = cls.load_from_checkpoint(ckpt_path, map_location=device)
        m.eval()
        if hasattr(m, "freeze"):
            m.freeze()
        m = m.to(device)
        ref_scores = score_nn(m, ref_seqs, device=device)
        alt_scores = score_nn(m, alt_seqs, device=device)
        out_df[f"nn_{name}_ref"] = ref_scores
        out_df[f"nn_{name}_alt"] = alt_scores
        out_df[f"nn_{name}_dd"] = ref_scores - alt_scores
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    return out_df


def score_with_fimo(df: pd.DataFrame, genome, motif_file: str, window_factor: int = 2) -> pd.DataFrame:
    """Add FIMO ΔΔ column. Window length is motif_len * window_factor - 1.
    Score is the max FIMO log-odds over hits that cover the variant position.
    """
    import re as _re
    with open(motif_file) as fh:
        match = _re.search(r"w=\s*(\d+)", fh.read())
    motif_len = int(match.group(1)) if match else 13
    win = motif_len * window_factor - 1
    half = win // 2

    keep_idx = []
    ref_seqs, alt_seqs, ids = [], [], []
    for i, row in df.iterrows():
        rs, as_ = extract_seq_pair(genome, row["chrom"], int(row["pos"]),
                                   row["ref"], row["alt"], half)
        if rs is not None:
            ref_seqs.append(rs)
            alt_seqs.append(as_)
            ids.append(f"v{i}")
            keep_idx.append(i)

    out_df = df.iloc[keep_idx].reset_index(drop=True)

    with tempfile.TemporaryDirectory(prefix="fimo_dv_") as wd:
        fa = Path(wd) / "v.fasta"
        with open(fa, "w") as f:
            for vid, s in zip(ids, ref_seqs):
                f.write(f">{vid}_ref\n{s}\n")
            for vid, s in zip(ids, alt_seqs):
                f.write(f">{vid}_alt\n{s}\n")
        out_dir = Path(wd) / "out"
        cmd = ["fimo", "--thresh", "1.0", "--oc", str(out_dir),
               "--max-strand", "--max-stored-scores", "100000000",
               motif_file, str(fa)]
        subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
        fimo_df = pd.read_csv(out_dir / "fimo.tsv", sep="\t", comment="#")

    target_1based = half + 1
    ref_scores, alt_scores = [], []
    for vid in ids:
        ref_h = fimo_df[(fimo_df["sequence_name"] == f"{vid}_ref")
                        & (fimo_df["start"] <= target_1based)
                        & (fimo_df["stop"] >= target_1based)]
        alt_h = fimo_df[(fimo_df["sequence_name"] == f"{vid}_alt")
                        & (fimo_df["start"] <= target_1based)
                        & (fimo_df["stop"] >= target_1based)]
        ref_scores.append(ref_h["score"].max() if len(ref_h) else np.nan)
        alt_scores.append(alt_h["score"].max() if len(alt_h) else np.nan)
    out_df["fimo_ref"] = ref_scores
    out_df["fimo_alt"] = alt_scores
    out_df["fimo_ref_zeroed"] = out_df["fimo_ref"].fillna(0)
    out_df["fimo_alt_zeroed"] = out_df["fimo_alt"].fillna(0)
    out_df["fimo_dd"] = out_df["fimo_ref_zeroed"] - out_df["fimo_alt_zeroed"]
    return out_df


def score_with_foldx(df: pd.DataFrame, genome, foldx_pdb: str, n_workers: int = 16) -> pd.DataFrame:
    """Add FoldX ΔΔG column. Uses the DNA length encoded in the PDB."""
    from tfbs.mcp.foldx_tools import (
        _parse_dna_from_pdb, _build_foldx_mutation_string,
        _run_foldx_energy,
    )
    foldx_bin = shutil.which("foldx") or shutil.which("foldx5")
    if foldx_bin is None:
        df["foldx_ref_dG"] = np.nan
        df["foldx_alt_dG"] = np.nan
        df["foldx_ddG"] = np.nan
        return df

    dna = _parse_dna_from_pdb(foldx_pdb)
    fwd = sorted(dna.keys())[0]
    rev = sorted(dna.keys())[1] if len(dna) > 1 else None
    ref_fwd = dna[fwd]
    ref_rev = dna[rev] if rev else None
    dna_len = len(ref_fwd)
    half = dna_len // 2

    keep_idx = []
    ref_seqs, alt_seqs = [], []
    for i, row in df.iterrows():
        rs, as_ = extract_seq_pair(genome, row["chrom"], int(row["pos"]),
                                   row["ref"], row["alt"], half)
        if rs is not None and len(rs) == dna_len:
            ref_seqs.append(rs)
            alt_seqs.append(as_)
            keep_idx.append(i)
    out_df = df.iloc[keep_idx].reset_index(drop=True)

    def score_seq(seq):
        mut = []
        mf = _build_foldx_mutation_string(ref_fwd, seq, fwd)
        if mf:
            mut.append(mf.rstrip(";"))
        if rev and ref_rev:
            seq_rc = "".join(COMPLEMENT.get(b, "N") for b in reversed(seq))
            mr = _build_foldx_mutation_string(ref_rev, seq_rc, rev)
            if mr:
                mut.append(mr.rstrip(";"))
        ms = ",".join(mut) + ";" if mut else None
        try:
            with tempfile.TemporaryDirectory(prefix="fx_") as wd:
                return _run_foldx_energy(foldx_bin, foldx_pdb, wd, ms)
        except Exception:
            return None

    def process(args_tuple):
        idx, ref, alt = args_tuple
        return idx, score_seq(ref), score_seq(alt)

    from multiprocessing import Pool
    items = [(i, ref_seqs[i], alt_seqs[i]) for i in range(len(ref_seqs))]
    with Pool(n_workers) as pool:
        results = list(pool.imap_unordered(process, items, chunksize=1))

    by_idx = {idx: (er, ea) for idx, er, ea in results}
    refs, alts, dds = [], [], []
    for i in range(len(ref_seqs)):
        er, ea = by_idx.get(i, (None, None))
        refs.append(er if er is not None else np.nan)
        alts.append(ea if ea is not None else np.nan)
        dds.append(ea - er if (er is not None and ea is not None) else np.nan)
    out_df["foldx_ref_dG"] = refs
    out_df["foldx_alt_dG"] = alts
    out_df["foldx_ddG"] = dds
    return out_df


# ---------------------------------------------------------------------------
# Step 4: Statistics and plots
# ---------------------------------------------------------------------------

def evaluate(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute per-method AUC and save plots. Returns summary DataFrame."""
    methods = []
    for col in df.columns:
        if col.endswith("_dd") or col == "foldx_ddG":
            method_name = col.replace("nn_", "NN ").replace("_dd", "").replace("foldx_ddG", "FoldX")
            methods.append((method_name.strip(), col))

    is_path = (df["label"] == "pathogenic").astype(int).values
    summary = []
    for name, col in methods:
        vals = df[col].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 10 or len(set(is_path[valid])) < 2:
            continue
        abs_vals = np.abs(vals[valid])
        is_p = is_path[valid]
        auc_abs = roc_auc_score(is_p, abs_vals)
        auc_raw = roc_auc_score(is_p, vals[valid])
        u, p_mwu = mannwhitneyu(abs_vals[is_p == 1], abs_vals[is_p == 0],
                                alternative="greater")
        summary.append({
            "method": name,
            "auc_abs": auc_abs,
            "auc_raw": auc_raw,
            "mwu_p": p_mwu,
            "median_abs_path": float(np.median(abs_vals[is_p == 1])),
            "median_abs_benign": float(np.median(abs_vals[is_p == 0])),
            "n_path": int((is_p == 1).sum()),
            "n_benign": int((is_p == 0).sum()),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "summary_all_methods.csv", index=False)

    # Boxplots
    n = len([m for m, c in methods if c in df.columns])
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    ax_iter = iter(axes)
    for name, col in methods:
        if col not in df.columns:
            continue
        try:
            ax = next(ax_iter)
        except StopIteration:
            break
        vals = df[col].values.astype(float)
        valid = ~np.isnan(vals)
        p_vals = np.abs(vals[valid & (df["label"] == "pathogenic").values])
        b_vals = np.abs(vals[valid & (df["label"] == "benign").values])
        bp = ax.boxplot(
            [b_vals, p_vals],
            tick_labels=[f"benign\nn={len(b_vals)}", f"pathogenic\nn={len(p_vals)}"],
            patch_artist=True, widths=0.5, showfliers=True,
        )
        for patch, color in zip(bp["boxes"], ["#88aaff", "#ff8888"]):
            patch.set_facecolor(color)
        if len(p_vals) and len(b_vals) and len(set(is_path[valid])) > 1:
            auc_abs = roc_auc_score(is_path[valid], np.abs(vals[valid]))
            ax.set_title(f"{name}\nAUC = {auc_abs:.3f}")
        else:
            ax.set_title(name)
        ax.set_ylabel("|ΔΔ|")
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle("Pathogenic vs Benign |ΔΔ binding|", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "boxplots_all_methods.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ROC curves
    fig2, ax = plt.subplots(figsize=(9, 7))
    palette = {
        "NN all_mean": "#cc0066", "NN core_mean": "#ff8800",
        "NN flank_mean": "#009933", "FIMO": "#0066cc", "FoldX": "#9c27b0",
    }
    for name, col in methods:
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 10 or len(set(is_path[valid])) < 2:
            continue
        fpr, tpr, _ = roc_curve(is_path[valid], np.abs(vals[valid]))
        auc_abs = roc_auc_score(is_path[valid], np.abs(vals[valid]))
        ax.plot(fpr, tpr, color=palette.get(name, "#333"), linewidth=2,
                label=f"{name} (AUC={auc_abs:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    n_p = int(is_path.sum())
    n_b = len(is_path) - n_p
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"Predicting pathogenicity from |ΔΔ binding|\n"
        f"Variants in TF binding peaks (n={n_p + n_b}: {n_p} pathogenic, {n_b} benign)"
    )
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "roc_all_methods.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Disease-variant validation for TF binding models")
    parser.add_argument("--peaks-bed", required=True, help="BED file of TF binding peaks")
    parser.add_argument("--clinvar-vcf", required=True, help="Indexed ClinVar VCF (.vcf.gz with .tbi)")
    parser.add_argument("--genome", required=True, help="Reference genome FASTA")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--peak-confidence", type=int, default=1,
                        help="Minimum confidence (e.g. ChIP-Atlas overlap count)")
    parser.add_argument("--models", nargs="*", default=[],
                        help="NN models as name=path (e.g. flank=ckpt.ckpt)")
    parser.add_argument("--motif-file", help="JASPAR/MEME motif file for FIMO")
    parser.add_argument("--foldx-pdb", help="Repaired TF-DNA complex PDB for FoldX")
    parser.add_argument("--nn-window", type=int, default=24,
                        help="Window size around variant for NN scoring")
    parser.add_argument("--foldx-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: merge peaks ===")
    peaks = merge_bed_intervals(args.peaks_bed, min_confidence=args.peak_confidence)
    print(f"Peaks at confidence >= {args.peak_confidence}: {len(peaks):,}")

    print("\n=== Step 2: extract ClinVar SNVs ===")
    df = extract_clinvar_snvs(peaks, args.clinvar_vcf)
    print(f"Variants: {len(df)} ({(df['label']=='pathogenic').sum()} pathogenic, "
          f"{(df['label']=='benign').sum()} benign)")
    df.to_csv(output / "clinvar_snvs.csv", index=False)

    if len(df) == 0:
        print("No variants found. Exiting.")
        return

    genome = pyfaidx.Fasta(args.genome)

    if args.models:
        print("\n=== Step 3a: NN scoring ===")
        models = {}
        for spec in args.models:
            if "=" not in spec:
                print(f"Skipping malformed --models entry: {spec}")
                continue
            name, path = spec.split("=", 1)
            models[name] = path
        df = score_with_nn(df, models, genome, window=args.nn_window, device=args.device)
        print(f"After NN scoring: {len(df)} variants")

    if args.motif_file:
        print("\n=== Step 3b: FIMO scoring ===")
        df = score_with_fimo(df, genome, args.motif_file)
        print(f"After FIMO scoring: {len(df)} variants")

    if args.foldx_pdb:
        print("\n=== Step 3c: FoldX scoring ===")
        df = score_with_foldx(df, genome, args.foldx_pdb,
                              n_workers=args.foldx_workers)
        print(f"After FoldX scoring: {len(df)} variants")

    df.to_csv(output / "combined_results.csv", index=False)
    genome.close()

    print("\n=== Step 4: evaluation ===")
    summary = evaluate(df, output)
    print(summary.to_string(index=False))

    print(f"\nDone. Results in {output}")


if __name__ == "__main__":
    main()
