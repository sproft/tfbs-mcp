"""
External validation of NKX2-1 binding-site models against ReMap 2022.

Positives: ReMap 2022 hg38 NKX2-1 non-redundant peaks. Column 5 (score)
counts how many of the 4 ChIP-seq experiments support the peak.

Negatives: random length-matched, GC-matched genomic regions, excluding
any ReMap NKX2-1 peak +/- 1 kb.

Methods compared:
  - VCNNBpnet (all_mean, core_mean, flank_mean): tile 24 bp windows over
    each region, take the maximum score per region.
  - FIMO (JASPAR MA1994.1): scan the full region with FIMO, take the max
    log-likelihood score (0 if no hit at default threshold).
  - FoldX (NKX2-1 C+A+B crystal): score the center 11 bp window of each
    region. Single FoldX call per region for tractability.

Outputs:
  - positive_scores.csv: one row per peak with method scores and metadata
  - negative_scores.csv: same for matched negatives
  - roc_all_methods.png: ROC curves per method
  - summary.csv: AUCs
"""

import argparse
import sys
import tempfile
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfaidx
import torch
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, "/sc-projects/sc-proj-cc17-P09_TFBS/scripts")
from chipseq_benchmark_with_foldx import (
    NUC_TO_IDX,
    score_nn,
    score_foldx_batch,
)

import tfbs.nn.models as models


def load_model(checkpoint_path, model_type, device="cuda"):
    model_cls = getattr(models, model_type)
    model = model_cls.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    return model.to(device)


def gc_content(seq):
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0


def dinucleotide_shuffle(seq, rng):
    """Altschul-Erickson dinucleotide shuffle: preserves exact dinucleotide
    counts of `seq` while randomizing the order. Returns a string of the
    same length over A/C/G/T (unknown bases passed through unchanged but
    excluded from the shuffle).
    """
    seq = seq.upper()
    if len(seq) < 2:
        return seq
    # Build adjacency: for each base b, list of bases that follow it in seq
    edges = {b: [] for b in "ACGT"}
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if a in edges and b in NUC_TO_IDX:
            edges[a].append(b)
    if not any(edges.values()):
        return seq
    # Shuffle edge lists, but keep one edge per base reserved as the "last"
    # outgoing edge so a Eulerian path from start exists.
    last_base = seq[-1]
    last_edge = {b: None for b in edges}
    for a, lst in edges.items():
        rng.shuffle(lst)
    # Identify a valid last edge per non-final base: an edge to a vertex from
    # which last_base is reachable. For a typical genomic sequence with all
    # 4 bases present the Eulerian path always exists; we just guarantee the
    # last edge of each list goes to a vertex that has outgoing edges except
    # for the terminal base.
    for a in "ACGT":
        if a == last_base or not edges[a]:
            continue
        # Find an edge that does not lead to a dead end
        for k in range(len(edges[a]) - 1, -1, -1):
            b = edges[a][k]
            if b == last_base or len(edges[b]) > 0:
                # Move this edge to the end so it is consumed last
                edges[a].append(edges[a].pop(k))
                break
    # Walk the Eulerian path from the original start
    start = seq[0]
    if start not in NUC_TO_IDX:
        return seq
    out = [start]
    cur = start
    # Position counters per base
    idx = {b: 0 for b in edges}
    while True:
        if idx[cur] >= len(edges[cur]):
            break
        nxt = edges[cur][idx[cur]]
        idx[cur] += 1
        out.append(nxt)
        cur = nxt
    if len(out) != len(seq):
        # Fall back to mononucleotide shuffle if Eulerian failed (rare)
        chars = list(seq)
        rng.shuffle(chars)
        return "".join(chars)
    return "".join(out)


def extract_full_region(row, genome, max_len=2000):
    s, e = int(row["start"]), int(row["end"])
    if e - s > max_len:
        # Cap very wide peaks so the FIMO sweep stays bounded
        mid = (s + e) // 2
        s, e = mid - max_len // 2, mid + max_len // 2
    try:
        seq = str(genome[row["chrom"]][s:e]).upper()
    except (KeyError, ValueError):
        return ""
    if any(c not in NUC_TO_IDX and c != "N" for c in seq):
        return ""
    return seq


def extract_centered_short(row, genome, window, center_col=None):
    if center_col and center_col in row and not pd.isna(row[center_col]):
        centre = int(row[center_col])
    else:
        centre = int(row["start"]) + (int(row["end"]) - int(row["start"])) // 2
    s, e = centre - window // 2, centre - window // 2 + window
    try:
        seq = str(genome[row["chrom"]][s:e]).upper()
    except (KeyError, ValueError):
        return ""
    if len(seq) != window or any(c not in NUC_TO_IDX for c in seq):
        return ""
    return seq


def tile_windows(seq, window):
    """Return all `window`-length windows in seq with N-positions skipped."""
    out = []
    for i in range(len(seq) - window + 1):
        sub = seq[i:i + window]
        if any(c not in NUC_TO_IDX for c in sub):
            continue
        out.append(sub)
    return out


def nn_max_per_region(model, region_seqs, window, device, batch_size=512):
    """Tile each region into `window`-bp pieces, score, take max per region."""
    all_windows = []
    counts = []
    for seq in region_seqs:
        ws = tile_windows(seq, window)
        all_windows.extend(ws)
        counts.append(len(ws))
    if not all_windows:
        return np.full(len(region_seqs), np.nan)
    flat_scores = score_nn(model, all_windows, batch_size=batch_size, device=device)
    out = np.empty(len(region_seqs), dtype=float)
    cur = 0
    for j, n in enumerate(counts):
        if n == 0:
            out[j] = np.nan
        else:
            out[j] = float(np.max(flat_scores[cur:cur + n]))
        cur += n
    return out


def fimo_max_per_region(region_seqs, motif_file):
    """Run FIMO on the full regions and report max score per region (0 if none)."""
    if not region_seqs:
        return np.zeros(0)
    with tempfile.TemporaryDirectory(prefix="fimo_remap_") as wd:
        fa = Path(wd) / "regions.fasta"
        with open(fa, "w") as f:
            for i, s in enumerate(region_seqs):
                f.write(f">r{i}\n{s}\n")
        out_dir = Path(wd) / "out"
        cmd = ["fimo", "--thresh", "1e-3", "--oc", str(out_dir),
               "--max-strand", "--max-stored-scores", "100000000",
               motif_file, str(fa)]
        subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
        try:
            df = pd.read_csv(out_dir / "fimo.tsv", sep="\t", comment="#")
        except FileNotFoundError:
            return np.zeros(len(region_seqs))
    scores = np.zeros(len(region_seqs))
    if "sequence_name" in df.columns and "score" in df.columns:
        for sn, sub in df.groupby("sequence_name"):
            try:
                idx = int(str(sn).lstrip("r"))
            except ValueError:
                continue
            if 0 <= idx < len(region_seqs):
                scores[idx] = sub["score"].max()
    return scores


def sample_matched_negative(rng, chr_lens, excl, peak_len, target_gc, genome,
                            gc_tol=0.05, max_tries=200):
    """One random region matching peak_len and target_gc, not in excl regions."""
    chr_list = list(chr_lens.keys())
    for _ in range(max_tries):
        chrom = chr_list[rng.randint(0, len(chr_list))]
        chr_len = chr_lens[chrom]
        start = rng.randint(0, max(1, chr_len - peak_len))
        end = start + peak_len
        # Reject if overlaps any exclude interval on this chromosome
        ok = True
        for s2, e2 in excl.get(chrom, []):
            if start < e2 and s2 < end:
                ok = False
                break
            if s2 >= end:
                break
        if not ok:
            continue
        try:
            seq = str(genome[chrom][start:end]).upper()
        except (KeyError, ValueError):
            continue
        # Allow up to a few Ns; only reject if they dominate
        if seq.count("N") > 0.1 * len(seq):
            continue
        if abs(gc_content(seq) - target_gc) > gc_tol:
            continue
        return chrom, start, end, seq
    # Relaxed fallback: any non-overlapping region of the right length
    for _ in range(max_tries):
        chrom = chr_list[rng.randint(0, len(chr_list))]
        chr_len = chr_lens[chrom]
        start = rng.randint(0, max(1, chr_len - peak_len))
        end = start + peak_len
        ok = True
        for s2, e2 in excl.get(chrom, []):
            if start < e2 and s2 < end:
                ok = False
                break
        if not ok:
            continue
        try:
            seq = str(genome[chrom][start:end]).upper()
        except (KeyError, ValueError):
            continue
        if seq.count("N") > 0.1 * len(seq):
            continue
        return chrom, start, end, seq
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remap-bed", required=True)
    parser.add_argument("--score-min", type=int, default=4)
    parser.add_argument("--max-positives", type=int, default=500)
    parser.add_argument("--genome",
                        default="/sc-projects/sc-proj-btg/P09/data/genomes/hg38/hg38.fa")
    parser.add_argument("--motif-file",
                        default="/sc-projects/sc-proj-cc17-P09_TFBS/data/JASPAR/MA1994.1.meme")
    parser.add_argument("--foldx-pdb",
                        default="/sc-projects/sc-proj-cc17-P09_TFBS/data/structures/repaired/NKX2-1_complex_CAB_Repair.pdb")
    parser.add_argument("--foldx-bin", default="/home/profts/.local/bin/foldx")
    parser.add_argument("--datasets", nargs="+",
                        default=["all_mean", "core_mean", "flank_mean"])
    parser.add_argument("--models-dir", default="saved_models_final")
    parser.add_argument("--model-type", default="VCNNBpnet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--foldx-workers", type=int, default=32)
    parser.add_argument("--foldx-window", type=int, default=11)
    parser.add_argument("--nn-window", type=int, default=24)
    parser.add_argument("--max-region-len", type=int, default=500,
                        help="Cap peak/negative region length for tiling")
    parser.add_argument("--negative-mode", default="genomic",
                        choices=["genomic", "shuffle"],
                        help="genomic = length+GC matched random regions; "
                             "shuffle = dinucleotide-shuffled positives")
    parser.add_argument("--skip-foldx", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Load NN models ----------------
    nn_models = {}
    for ds in args.datasets:
        ckpt = Path(args.models_dir) / ds / "standardize" / f"best_{args.model_type}.ckpt"
        if not ckpt.exists():
            print(f"[skip] {ckpt} not found")
            continue
        print(f"Loading NN model {ds}")
        nn_models[ds] = load_model(str(ckpt), args.model_type, args.device)

    # ---------------- Read ReMap peaks ----------------
    remap_df = pd.read_csv(
        args.remap_bed, sep="\t", header=None, comment="#",
        names=["chrom", "start", "end", "name", "score", "strand",
               "thickStart", "thickEnd", "itemRgb"],
    )
    print(f"\nReMap NKX2-1 peaks: {len(remap_df):,}")
    print("Score (#experiments) distribution:")
    print(remap_df["score"].value_counts().sort_index())

    high_conf = remap_df[remap_df["score"] >= args.score_min].reset_index(drop=True)
    print(f"\nFiltered to score >= {args.score_min}: {len(high_conf)} peaks")

    if len(high_conf) > args.max_positives:
        rng_pos = np.random.RandomState(args.seed)
        idx = rng_pos.choice(len(high_conf), args.max_positives, replace=False)
        high_conf = high_conf.iloc[idx].reset_index(drop=True)
        print(f"Subsampled to {args.max_positives} positives")

    # ---------------- Extract sequences (positives) ----------------
    genome = pyfaidx.Fasta(args.genome)
    pos_full = []           # full peak sequences for NN tiling and FIMO
    pos_short = []          # 11 bp center windows for FoldX
    pos_lengths = []
    pos_keep = []
    pos_meta = []
    for i, row in high_conf.iterrows():
        full = extract_full_region(row, genome, max_len=args.max_region_len)
        if not full:
            continue
        short = extract_centered_short(row, genome, window=args.foldx_window,
                                        center_col="thickStart")
        pos_full.append(full)
        pos_short.append(short)
        pos_lengths.append(len(full))
        pos_keep.append(i)
        pos_meta.append((row["chrom"], int(row["start"]), int(row["end"]),
                         row["name"], row["score"]))
    print(f"\nPositives extracted: {len(pos_full)} regions, "
          f"median length {int(np.median(pos_lengths))} bp")

    # ---------------- Sample length+GC matched negatives ----------------
    # Build exclusion set: ALL ReMap peaks (any score) +/- 1 kb pad
    excl = {}
    pad = 1000
    for _, r in remap_df.iterrows():
        excl.setdefault(r["chrom"], []).append((max(0, int(r["start"]) - pad),
                                                int(r["end"]) + pad))
    for c, lst in excl.items():
        lst.sort()
        merged = [lst[0]]
        for s, e in lst[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        excl[c] = merged

    chr_lens = {c: len(genome[c]) for c in excl.keys() if c.startswith("chr") and "_" not in c}

    rng_neg = np.random.RandomState(args.seed + 1)
    neg_full = []
    neg_short = []
    neg_meta = []
    if args.negative_mode == "shuffle":
        print("\nGenerating dinucleotide-shuffled negatives from positives...")
        for j, full in enumerate(pos_full):
            shuf = dinucleotide_shuffle(full, rng_neg)
            neg_full.append(shuf)
            mid = len(shuf) // 2
            s11 = mid - args.foldx_window // 2
            ns_short = shuf[s11:s11 + args.foldx_window]
            if len(ns_short) != args.foldx_window or any(c not in NUC_TO_IDX for c in ns_short):
                ns_short = ""
            neg_short.append(ns_short)
            neg_meta.append((f"shuffle_of_{pos_meta[j][0]}", pos_meta[j][1], pos_meta[j][2]))
    else:
        print("\nSampling length+GC matched negatives...")
        for j, (full, length) in enumerate(zip(pos_full, pos_lengths)):
            target_gc = gc_content(full)
            sample = sample_matched_negative(
                rng_neg, chr_lens, excl, length, target_gc, genome,
            )
            if sample is None:
                continue
            chrom, ns, ne, nseq = sample
            if "N" in nseq:
                nseq = "".join(c if c in NUC_TO_IDX else "ACGT"[rng_neg.randint(0, 4)]
                               for c in nseq)
            neg_full.append(nseq)
            mid = (ns + ne) // 2
            s11, e11 = mid - args.foldx_window // 2, mid - args.foldx_window // 2 + args.foldx_window
            try:
                ns_short = str(genome[chrom][s11:e11]).upper()
                if "N" in ns_short:
                    ns_short = "".join(c if c in NUC_TO_IDX else "ACGT"[rng_neg.randint(0, 4)]
                                        for c in ns_short)
                if len(ns_short) != args.foldx_window:
                    ns_short = ""
            except (KeyError, ValueError):
                ns_short = ""
            neg_short.append(ns_short)
            neg_meta.append((chrom, ns, ne))
    genome.close()
    print(f"Negatives generated: {len(neg_full)} regions")

    pos_gc = np.mean([gc_content(s) for s in pos_full])
    neg_gc = np.mean([gc_content(s) for s in neg_full])
    print(f"Mean GC content: positives {pos_gc:.3f}, negatives {neg_gc:.3f}")

    # ---------------- Score with NN models (max over tile) ----------------
    print("\nScoring NN models (max over 24-bp windows in each region)...")
    pos_nn_scores = {}
    neg_nn_scores = {}
    for ds, m in nn_models.items():
        print(f"  {ds}")
        pos_nn_scores[ds] = nn_max_per_region(m, pos_full, args.nn_window, args.device)
        neg_nn_scores[ds] = nn_max_per_region(m, neg_full, args.nn_window, args.device)

    # ---------------- Score with FIMO ----------------
    print("Scoring FIMO on full regions...")
    pos_fimo = fimo_max_per_region(pos_full, args.motif_file)
    neg_fimo = fimo_max_per_region(neg_full, args.motif_file)
    print(f"  FIMO hits (score > 0): pos {(pos_fimo > 0).sum()}/{len(pos_fimo)}, "
          f"neg {(neg_fimo > 0).sum()}/{len(neg_fimo)}")

    # ---------------- Score with FoldX (center 11 bp only) ----------------
    if args.skip_foldx:
        pos_foldx = np.full(len(pos_short), np.nan)
        neg_foldx = np.full(len(neg_short), np.nan)
        print("FoldX skipped (--skip-foldx).")
    else:
        # Filter to non-empty 11-bp seqs
        pos_short_idx = [i for i, s in enumerate(pos_short) if s]
        neg_short_idx = [i for i, s in enumerate(neg_short) if s]
        pos_short_seqs = [pos_short[i] for i in pos_short_idx]
        neg_short_seqs = [neg_short[i] for i in neg_short_idx]
        print(f"Scoring FoldX on {len(pos_short_seqs)} pos + {len(neg_short_seqs)} neg "
              f"({args.foldx_window} bp center, {args.foldx_workers} workers)...")
        pos_fx_arr = score_foldx_batch(pos_short_seqs, args.foldx_pdb, args.foldx_bin,
                                        n_workers=args.foldx_workers)
        neg_fx_arr = score_foldx_batch(neg_short_seqs, args.foldx_pdb, args.foldx_bin,
                                        n_workers=args.foldx_workers)
        pos_foldx = np.full(len(pos_short), np.nan)
        neg_foldx = np.full(len(neg_short), np.nan)
        for j, src_i in enumerate(pos_short_idx):
            pos_foldx[src_i] = pos_fx_arr[j]
        for j, src_i in enumerate(neg_short_idx):
            neg_foldx[src_i] = neg_fx_arr[j]
        print(f"  FoldX OK: pos {np.sum(~np.isnan(pos_foldx))}/{len(pos_foldx)}, "
              f"neg {np.sum(~np.isnan(neg_foldx))}/{len(neg_foldx)}")

    # ---------------- Save tables ----------------
    pos_table = pd.DataFrame({
        "chrom":  [m[0] for m in pos_meta],
        "start":  [m[1] for m in pos_meta],
        "end":    [m[2] for m in pos_meta],
        "remap_name":  [m[3] for m in pos_meta],
        "remap_score": [m[4] for m in pos_meta],
        "region_len":  pos_lengths,
        **{f"nn_{ds}_max": pos_nn_scores[ds] for ds in nn_models},
        "fimo_max":  pos_fimo,
        "foldx_neg_dG_centered": pos_foldx,
    })
    pos_table.to_csv(output_dir / "positive_scores.csv", index=False)

    neg_table = pd.DataFrame({
        "chrom": [m[0] for m in neg_meta],
        "start": [m[1] for m in neg_meta],
        "end":   [m[2] for m in neg_meta],
        "region_len":  [len(s) for s in neg_full],
        **{f"nn_{ds}_max": neg_nn_scores[ds] for ds in nn_models},
        "fimo_max":  neg_fimo,
        "foldx_neg_dG_centered": neg_foldx,
    })
    neg_table.to_csv(output_dir / "negative_scores.csv", index=False)

    # ---------------- ROC curves ----------------
    methods = {f"NN {ds}": (pos_nn_scores[ds], neg_nn_scores[ds]) for ds in nn_models}
    methods["FIMO"]  = (pos_fimo, neg_fimo)
    methods["FoldX"] = (pos_foldx, neg_foldx)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    colors = {"NN all_mean": "#2196F3", "NN core_mean": "#FF9800",
              "NN flank_mean": "#4CAF50", "FIMO": "#9C27B0", "FoldX": "#E91E63"}
    summary_rows = []
    for name, (ps, ns) in methods.items():
        valid_p = ~np.isnan(ps)
        valid_n = ~np.isnan(ns)
        ps_, ns_ = ps[valid_p], ns[valid_n]
        if len(ps_) < 5 or len(ns_) < 5:
            continue
        n = min(len(ps_), len(ns_))
        rng2 = np.random.RandomState(args.seed)
        ps_bal = rng2.choice(ps_, n, replace=False) if len(ps_) > n else ps_
        ns_bal = rng2.choice(ns_, n, replace=False) if len(ns_) > n else ns_
        y_true = np.concatenate([np.ones(n), np.zeros(n)])
        y_score = np.concatenate([ps_bal, ns_bal])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors.get(name, "#333"), linewidth=2,
                label=f"{name} (AUC={roc_auc:.3f})")
        summary_rows.append({
            "method": name, "auc": roc_auc, "n_pos": int(n), "n_neg": int(n),
        })
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("FPR", fontsize=12)
    ax.set_ylabel("TPR", fontsize=12)
    neg_label = ("dinucleotide-shuffled positives" if args.negative_mode == "shuffle"
                 else "length+GC matched random regions")
    ax.set_title(
        f"ReMap 2022 NKX2-1 (score>={args.score_min}, n={len(pos_full)}) vs {neg_label}\n"
        f"NN/FIMO: max over the full peak. FoldX: center {args.foldx_window} bp.",
        fontsize=12,
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_all_methods.png", dpi=150, bbox_inches="tight")
    plt.close()

    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)

    print(f"\nDone. Outputs in {output_dir}")
    print(f"  positive_scores.csv     ({len(pos_table)} positives)")
    print(f"  negative_scores.csv     ({len(neg_table)} negatives)")
    print(f"  roc_all_methods.png")
    print(f"  summary.csv")


if __name__ == "__main__":
    main()
