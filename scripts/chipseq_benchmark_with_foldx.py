"""
ChIP-seq classification benchmark with NN + FIMO + FoldX side by side.

For each (positive_TF, negative_TF) pair, samples N sequences from each TF's
ChIP-Atlas merged peaks, scores them with three NN models, FIMO (JASPAR),
and FoldX (TF-DNA crystal complex), and computes ROC-AUC for each method.

Uses the same subsample across methods so AUCs are directly comparable.

For NKX2.1 with the 11 bp C+A+B complex:
  ~3 min per FoldX call × 500 seq × 5 TFs / 32 cores ≈ 80 min total.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfaidx
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve

NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
IDX_TO_NUC = ["A", "C", "G", "T"]
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def merge_intervals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["chrom", "start"]).reset_index(drop=True)
    out = []
    cc = cs = ce = cn = None
    for _, row in df.iterrows():
        if cc is None or row["chrom"] != cc or row["start"] > ce:
            if cc is not None:
                out.append((cc, cs, ce, cn))
            cc = row["chrom"]
            cs = int(row["start"])
            ce = int(row["end"])
            cn = int(row["num"])
        else:
            ce = max(ce, int(row["end"]))
            cn = max(cn, int(row["num"]))
    if cc is not None:
        out.append((cc, cs, ce, cn))
    return pd.DataFrame(out, columns=["chrom", "start", "end", "num"])


def extract_sequences(peaks: pd.DataFrame, genome_path: str, window: int, n: int, seed: int):
    """Pick n random peaks and extract a `window`-bp sequence centered on each peak midpoint."""
    rng = np.random.RandomState(seed)
    if len(peaks) > n:
        idx = rng.choice(len(peaks), n, replace=False)
        peaks = peaks.iloc[idx].reset_index(drop=True)
    genome = pyfaidx.Fasta(genome_path)
    seqs = []
    for _, row in peaks.iterrows():
        mid = row["start"] + (row["end"] - row["start"]) // 2
        s = mid - window // 2
        e = s + window
        try:
            seq = str(genome[row["chrom"]][s:e]).upper()
        except (KeyError, ValueError):
            continue
        if len(seq) != window or any(c not in NUC_TO_IDX for c in seq):
            continue
        seqs.append(seq)
    genome.close()
    return seqs


# ---------------------------------------------------------------------------
# NN scoring
# ---------------------------------------------------------------------------

def encode(seq: str) -> torch.Tensor:
    return F.one_hot(
        torch.tensor([NUC_TO_IDX[c] for c in seq], dtype=torch.long), 4
    ).float().permute(1, 0)


def score_nn(model, sequences, batch_size=512, device="cuda"):
    out = []
    for i in range(0, len(sequences), batch_size):
        batch = torch.stack([encode(s) for s in sequences[i:i + batch_size]]).to(device)
        with torch.no_grad():
            p = model(batch).cpu().numpy().squeeze()
        if p.ndim == 0:
            p = np.array([float(p)])
        out.append(p)
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# FIMO scoring (one batch FASTA)
# ---------------------------------------------------------------------------

def score_fimo(sequences: list[str], motif_file: str) -> np.ndarray:
    """Returns max FIMO score per sequence (0 if no hit)."""
    with tempfile.TemporaryDirectory(prefix="fimo_chip_") as wd:
        fa = Path(wd) / "seqs.fasta"
        with open(fa, "w") as f:
            for i, s in enumerate(sequences):
                f.write(f">s{i}\n{s}\n")
        out_dir = Path(wd) / "out"
        cmd = ["fimo", "--thresh", "1.0", "--oc", str(out_dir),
               "--max-strand", "--max-stored-scores", "100000000",
               motif_file, str(fa)]
        subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
        try:
            df = pd.read_csv(out_dir / "fimo.tsv", sep="\t", comment="#")
        except FileNotFoundError:
            return np.zeros(len(sequences))
    scores = np.zeros(len(sequences))
    if "sequence_name" in df.columns and "score" in df.columns:
        for sn, sub in df.groupby("sequence_name"):
            try:
                idx = int(str(sn).lstrip("s"))
            except ValueError:
                continue
            if 0 <= idx < len(sequences):
                scores[idx] = sub["score"].max()
    return scores


# ---------------------------------------------------------------------------
# FoldX scoring (parallel)
# ---------------------------------------------------------------------------

def score_foldx_one(args_tuple):
    """Score one sequence — picklable for multiprocessing."""
    seq, structure, foldx_bin = args_tuple
    sys.path.insert(0, "/sc-projects/sc-proj-cc17-P09_TFBS")
    from tfbs.mcp.foldx_tools import (
        _parse_dna_from_pdb, _build_foldx_mutation_string, _run_foldx_energy,
    )
    dna = _parse_dna_from_pdb(structure)
    fwd = sorted(dna.keys())[0]
    rev = sorted(dna.keys())[1] if len(dna) > 1 else None
    ref_fwd = dna[fwd]
    ref_rev = dna[rev] if rev else None
    dna_len = len(ref_fwd)

    if len(seq) > dna_len:
        # Center-trim the sequence to match the structure's DNA length
        offset = (len(seq) - dna_len) // 2
        seq = seq[offset:offset + dna_len]
    elif len(seq) < dna_len:
        return None

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
            return _run_foldx_energy(foldx_bin, structure, wd, ms)
    except Exception:
        return None


def score_foldx_batch(sequences: list[str], structure: str, foldx_bin: str,
                      n_workers: int = 32) -> np.ndarray:
    items = [(s, structure, foldx_bin) for s in sequences]
    with Pool(n_workers) as pool:
        results = list(pool.imap(score_foldx_one, items, chunksize=1))
    out = np.array(
        [(-r if r is not None else float("nan")) for r in results],
        dtype=float,
    )  # invert so higher = stronger binding (consistent with NN/FIMO)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersect-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--motif-file", required=True)
    parser.add_argument("--foldx-pdb", required=True)
    parser.add_argument("--pos-tf", default="NKX2-1")
    parser.add_argument("--neg-tfs", nargs="+",
                        default=["GATA1", "MYOD1", "NKX2-5", "RXRA"])
    parser.add_argument("--datasets", nargs="+",
                        default=["all_mean", "core_mean", "flank_mean"])
    parser.add_argument("--models-dir", default="saved_models_final")
    parser.add_argument("--model-type", default="VCNNBpnet")
    parser.add_argument("--n-per-tf", type=int, default=500,
                        help="Sequences sampled per TF (FoldX is ~3 min/seq)")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--foldx-workers", type=int, default=32)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intersect_dir = Path(args.intersect_dir)

    # 1. Read and merge intersection BEDs for each TF
    tfs = [args.pos_tf] + args.neg_tfs
    tf_seqs = {}
    for tf in tfs:
        bed = intersect_dir / f"{tf}.allintersect.bed"
        if not bed.exists():
            print(f"Skipping {tf}: BED not found")
            continue
        df = pd.read_csv(bed, sep="\t", usecols=[0, 1, 2, 3],
                         names=["chrom", "start", "end", "num"], header=0)
        merged = merge_intervals(df)
        seqs = extract_sequences(merged, args.genome, args.window,
                                 args.n_per_tf, args.seed)
        tf_seqs[tf] = seqs
        print(f"{tf}: {len(merged):,} merged regions → {len(seqs)} sequences")

    # 2. Load NN models
    sys.path.insert(0, "/sc-projects/sc-proj-cc17-P09_TFBS")
    import tfbs.nn.models as models_mod
    cls = getattr(models_mod, args.model_type)
    loaded = {}
    for name in args.datasets:
        ckpt = Path(args.models_dir) / name / "standardize" / f"best_{args.model_type}.ckpt"
        if not ckpt.exists():
            print(f"Skipping NN {name}: checkpoint not found")
            continue
        m = cls.load_from_checkpoint(str(ckpt), map_location=args.device)
        m.eval()
        if hasattr(m, "freeze"):
            m.freeze()
        m = m.to(args.device)
        loaded[name] = m

    # 3. Score sequences with all methods
    foldx_bin = shutil.which("foldx") or shutil.which("foldx5")
    all_scores = {}  # (tf, method) -> np.ndarray
    for tf, seqs in tf_seqs.items():
        print(f"\n=== Scoring {tf} ({len(seqs)} sequences) ===")
        for name, model in loaded.items():
            print(f"  NN {name}...")
            all_scores[(tf, f"NN_{name}")] = score_nn(model, seqs, device=args.device)
        print(f"  FIMO...")
        all_scores[(tf, "FIMO")] = score_fimo(seqs, args.motif_file)
        if foldx_bin and args.foldx_pdb:
            print(f"  FoldX (parallel, {args.foldx_workers} workers)...")
            all_scores[(tf, "FoldX")] = score_foldx_batch(
                seqs, args.foldx_pdb, foldx_bin, n_workers=args.foldx_workers,
            )

    # 4. Save raw scores
    rows = []
    for tf, seqs in tf_seqs.items():
        for i, seq in enumerate(seqs):
            row = {"tf": tf, "seq_idx": i, "sequence": seq}
            for method in ["NN_" + n for n in loaded] + ["FIMO", "FoldX"]:
                key = (tf, method)
                if key in all_scores:
                    row[method] = all_scores[key][i]
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "raw_scores.csv", index=False)

    # 5. Compute AUC per (pos vs neg) pair per method
    methods = ["NN_" + n for n in loaded] + ["FIMO"]
    if foldx_bin and args.foldx_pdb:
        methods.append("FoldX")

    summary = []
    for neg in args.neg_tfs:
        if neg not in tf_seqs:
            continue
        for method in methods:
            pos_key = (args.pos_tf, method)
            neg_key = (neg, method)
            if pos_key not in all_scores or neg_key not in all_scores:
                continue
            ps = np.asarray(all_scores[pos_key])
            ns = np.asarray(all_scores[neg_key])

            # Drop NaNs (e.g. failed FoldX)
            ps = ps[~np.isnan(ps)]
            ns = ns[~np.isnan(ns)]
            if len(ps) < 10 or len(ns) < 10:
                continue

            # Balance
            n = min(len(ps), len(ns))
            rng = np.random.RandomState(args.seed)
            if len(ps) > n:
                ps = rng.choice(ps, n, replace=False)
            if len(ns) > n:
                ns = rng.choice(ns, n, replace=False)

            y_true = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
            y_score = np.concatenate([ps, ns])
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            summary.append({
                "pos_tf": args.pos_tf, "neg_tf": neg,
                "method": method, "auc": roc_auc,
                "n_pos": len(ps), "n_neg": len(ns),
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "auc_summary.csv", index=False)
    print("\n=== AUC summary ===")
    pivot = summary_df.pivot_table(
        index="method", columns="neg_tf", values="auc",
    )
    print(pivot.to_string(float_format="{:.3f}".format))
    pivot.to_csv(output_dir / "auc_pivot.csv")

    # 6. ROC plots
    palette = {
        "NN_all_mean": "#cc0066", "NN_core_mean": "#ff8800",
        "NN_flank_mean": "#009933", "FIMO": "#0066cc", "FoldX": "#9c27b0",
    }

    # One panel per negative TF
    n_neg = sum(1 for tf in args.neg_tfs if tf in tf_seqs)
    fig, axes = plt.subplots(1, n_neg, figsize=(6 * n_neg, 6), squeeze=False)
    axes = axes.flatten()

    for ax, neg in zip(axes, [t for t in args.neg_tfs if t in tf_seqs]):
        for method in methods:
            pos_key = (args.pos_tf, method)
            neg_key = (neg, method)
            if pos_key not in all_scores or neg_key not in all_scores:
                continue
            ps = np.asarray(all_scores[pos_key])
            ns = np.asarray(all_scores[neg_key])
            ps = ps[~np.isnan(ps)]
            ns = ns[~np.isnan(ns)]
            if len(ps) < 10 or len(ns) < 10:
                continue
            n = min(len(ps), len(ns))
            rng = np.random.RandomState(args.seed)
            ps_s = rng.choice(ps, n, replace=False) if len(ps) > n else ps
            ns_s = rng.choice(ns, n, replace=False) if len(ns) > n else ns
            y_true = np.concatenate([np.ones(len(ps_s)), np.zeros(len(ns_s))])
            y_score = np.concatenate([ps_s, ns_s])
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=palette.get(method, "#333"), linewidth=2,
                    label=f"{method} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{args.pos_tf} vs {neg}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle(
        f"NN + FIMO + FoldX classification of {args.pos_tf} vs negative TFs "
        f"(n={args.n_per_tf} sequences per TF, balanced)",
        fontsize=14, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "roc_by_neg_tf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Bar chart: AUC per (method, neg_tf)
    fig2, ax = plt.subplots(figsize=(12, 6))
    method_list = sorted(summary_df["method"].unique(),
                         key=lambda m: ["NN_all_mean","NN_core_mean","NN_flank_mean","FIMO","FoldX"].index(m)
                         if m in ["NN_all_mean","NN_core_mean","NN_flank_mean","FIMO","FoldX"] else 99)
    neg_list = [t for t in args.neg_tfs if t in tf_seqs]
    width = 0.8 / len(method_list)
    x = np.arange(len(neg_list))
    for i, method in enumerate(method_list):
        sub = summary_df[summary_df["method"] == method]
        aucs = [sub[sub["neg_tf"] == n]["auc"].values
                for n in neg_list]
        aucs = [v[0] if len(v) else 0 for v in aucs]
        ax.bar(x + i * width, aucs, width,
               color=palette.get(method, "#333"), label=method)
    ax.set_xticks(x + width * (len(method_list) - 1) / 2)
    ax.set_xticklabels(neg_list)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"{args.pos_tf} vs each negative TF — ROC-AUC across methods")
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.5, label="random")
    ax.legend(loc="lower right")
    ax.set_ylim([0.4, 1.0])
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "auc_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
