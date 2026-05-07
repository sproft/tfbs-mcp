"""
ChIP-seq Benchmark Analysis

Evaluates TFBS neural network models against ChIP-Atlas intersection BED files.
Two analysis modes:

1. Classification: Can the model distinguish NKX2-1 binding sites from other TFs?
   Tests at different experiment-count cutoffs (1, 3, 5, 8).

2. Regression: Does the model score correlate with the number of ChIP-seq
   experiments supporting each binding site?

Input: ChIP-Atlas intersection BED files (allintersect.bed format) with columns:
  chrom, start, end, num, list, GSM1.bed, GSM2.bed, ...

Output: ROC curve plots, scatter/binned regression plots, CSV summary tables.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfaidx
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc, roc_curve

import tfbs.nn.models as models


NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
IDX_TO_NUC = ["A", "C", "G", "T"]
MODEL_COLORS = {"all_mean": "#2196F3", "core_mean": "#FF9800", "flank_mean": "#4CAF50"}
NEG_TF_COLORS = {"GATA1": "#E91E63", "MYOD1": "#9C27B0", "NKX2-5": "#FF5722", "RXRA": "#009688"}


def load_model(checkpoint_path, model_type, device="cuda"):
    """Load a trained model checkpoint."""
    model_cls = getattr(models, model_type)
    model = model_cls.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    return model.to(device)


def merge_intervals(df):
    """Merge overlapping/adjacent intervals, keeping max num per region."""
    df = df.sort_values(["chrom", "start"]).reset_index(drop=True)
    merged = []
    cur_chrom, cur_start, cur_end, cur_num = None, None, None, 0
    for _, row in df.iterrows():
        if cur_chrom is None or row["chrom"] != cur_chrom or row["start"] > cur_end:
            if cur_chrom is not None:
                merged.append((cur_chrom, cur_start, cur_end, cur_num))
            cur_chrom = row["chrom"]
            cur_start = row["start"]
            cur_end = row["end"]
            cur_num = row["num"]
        else:
            cur_end = max(cur_end, row["end"])
            cur_num = max(cur_num, row["num"])
    if cur_chrom is not None:
        merged.append((cur_chrom, cur_start, cur_end, cur_num))
    return pd.DataFrame(merged, columns=["chrom", "start", "end", "num"])


def extract_sequences(peaks_df, genome_path, window=24, max_sequences=None):
    """Extract centered DNA sequences from peak regions."""
    genome = pyfaidx.Fasta(genome_path)
    sequences, nums = [], []

    sample = peaks_df
    if max_sequences and len(peaks_df) > max_sequences:
        sample = peaks_df.sample(n=max_sequences, random_state=42).reset_index(drop=True)

    for _, row in sample.iterrows():
        mid = row["start"] + (row["end"] - row["start"]) // 2
        s, e = mid - window // 2, mid + window // 2
        chrom = row["chrom"]
        try:
            chrom_len = len(genome[chrom])
        except KeyError:
            continue
        if s < 0 or e >= chrom_len:
            continue
        seq = str(genome[chrom][s:e]).upper()
        if any(c not in NUC_TO_IDX for c in seq) or len(seq) != window:
            continue
        sequences.append(seq)
        nums.append(row["num"])

    genome.close()
    return sequences, np.array(nums)


def score_sequences(model, sequences, device="cuda", batch_size=1024):
    """Score DNA sequences with a model, return numpy array of scores."""
    all_scores = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        indices = torch.tensor(
            [[NUC_TO_IDX[c] for c in seq] for seq in batch], dtype=torch.long
        )
        ohe = torch.nn.functional.one_hot(indices, num_classes=4).float().permute(0, 2, 1)
        ohe = ohe.to(device)
        with torch.no_grad():
            preds = model(ohe).cpu().numpy().squeeze()
        if preds.ndim == 0:
            preds = np.array([float(preds)])
        all_scores.append(preds)
    return np.concatenate(all_scores)


def run_classification_vs_negatives(pos_scores, neg_scores_dict, model_name, output_dir):
    """ROC curves: positive TF vs each negative TF."""
    fig, axes = plt.subplots(1, len(neg_scores_dict), figsize=(6 * len(neg_scores_dict), 6))
    if len(neg_scores_dict) == 1:
        axes = [axes]

    results = []
    for ax, (neg_tf, neg_scores) in zip(axes, neg_scores_dict.items()):
        n = min(len(pos_scores), len(neg_scores))
        rng = np.random.RandomState(42)
        ps = rng.choice(pos_scores, n, replace=False) if len(pos_scores) > n else pos_scores
        ns = rng.choice(neg_scores, n, replace=False) if len(neg_scores) > n else neg_scores

        y_true = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
        y_score = np.concatenate([ps, ns])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=NEG_TF_COLORS.get(neg_tf, "#333"), linewidth=2,
                label=f"vs {neg_tf} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_title(f"{model_name}: NKX2-1 vs {neg_tf}", fontsize=14)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.legend(fontsize=11, loc="lower right")
        ax.grid(True, linestyle='--', alpha=0.3)

        results.append({"model": model_name, "neg_tf": neg_tf,
                        "roc_auc": roc_auc, "n_pos": len(ps), "n_neg": len(ns)})

    fig.tight_layout()
    fig.savefig(output_dir / f"roc_{model_name}_vs_negatives.png", dpi=150, bbox_inches="tight")
    plt.close()
    return results


def run_cutoff_classification(model_scores_dict, nums, cutoffs, output_dir):
    """ROC curves at different experiment-count cutoffs."""
    fig, axes = plt.subplots(1, len(cutoffs), figsize=(6 * len(cutoffs), 6))
    if len(cutoffs) == 1:
        axes = [axes]

    results = []
    for ax, cutoff in zip(axes, cutoffs):
        labels = (nums >= cutoff).astype(int)
        n_pos, n_neg = labels.sum(), len(labels) - labels.sum()

        if n_pos == 0 or n_neg == 0:
            ax.set_title(f"Cutoff ≥ {cutoff}\n(skipped: {n_pos} pos / {n_neg} neg)", fontsize=14)
            continue

        for model_name, scores in model_scores_dict.items():
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=MODEL_COLORS.get(model_name, "#333"), linewidth=2,
                    label=f"{model_name} (AUC={roc_auc:.3f})")
            results.append({"cutoff": cutoff, "model": model_name,
                            "roc_auc": roc_auc, "n_pos": int(n_pos), "n_neg": int(n_neg)})

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_title(f"Cutoff: num ≥ {cutoff}\n({n_pos:,} pos / {n_neg:,} neg)", fontsize=14)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle("NKX2-1 Binding Classification at Experiment-Count Cutoffs", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves_cutoffs.png", dpi=150, bbox_inches="tight")
    plt.close()
    return results


def run_regression(model_scores_dict, nums, output_dir):
    """Scatter + binned plots: NN score vs experiment count."""
    # Scatter plots
    fig, axes = plt.subplots(1, len(model_scores_dict), figsize=(7 * len(model_scores_dict), 6))
    if len(model_scores_dict) == 1:
        axes = [axes]

    results = []
    for ax, (model_name, scores) in zip(axes, model_scores_dict.items()):
        ax.scatter(scores, nums, alpha=0.1, s=5, color=MODEL_COLORS.get(model_name, "#333"))
        r_p, _ = pearsonr(scores, nums)
        r_s, _ = spearmanr(scores, nums)
        z = np.polyfit(scores, nums, 1)
        p = np.poly1d(z)
        x_range = np.linspace(scores.min(), scores.max(), 100)
        ax.plot(x_range, p(x_range), 'r-', linewidth=2, alpha=0.8)
        ax.set_title(f"{model_name}\nPearson r={r_p:.3f}, Spearman ρ={r_s:.3f}", fontsize=14)
        ax.set_xlabel("NN Prediction Score", fontsize=12)
        ax.set_ylabel("Number of Experiments", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        results.append({"model": model_name, "pearson_r": r_p, "spearman_rho": r_s})

    fig.suptitle("NN Score vs Experiment Count (NKX2-1)", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "regression_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Binned plots
    fig2, axes2 = plt.subplots(1, len(model_scores_dict), figsize=(7 * len(model_scores_dict), 6))
    if len(model_scores_dict) == 1:
        axes2 = [axes2]

    for ax, (model_name, scores) in zip(axes2, model_scores_dict.items()):
        max_num = min(nums.max(), 20)
        bin_edges = list(range(1, max_num + 1))
        bin_means, bin_stds, bin_x = [], [], []
        for n in bin_edges:
            mask = nums == n
            if mask.sum() > 5:
                bin_means.append(scores[mask].mean())
                bin_stds.append(scores[mask].std())
                bin_x.append(n)
        ax.errorbar(bin_x, bin_means, yerr=bin_stds, fmt='o-',
                    color=MODEL_COLORS.get(model_name, "#333"),
                    linewidth=2, markersize=8, capsize=4, capthick=2)
        ax.set_title(f"{model_name}", fontsize=14)
        ax.set_xlabel("Number of Experiments", fontsize=12)
        ax.set_ylabel("Mean NN Score (± std)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

    fig2.suptitle("Mean NN Score by Experiment Count (NKX2-1)", fontsize=16, y=1.02)
    fig2.tight_layout()
    fig2.savefig(output_dir / "regression_binned.png", dpi=150, bbox_inches="tight")
    plt.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="ChIP-seq benchmark analysis for TFBS models")
    parser.add_argument("--intersect-dir", type=str, required=True,
                        help="Directory with *.allintersect.bed files")
    parser.add_argument("--genome", type=str,
                        default="/sc-projects/sc-proj-btg/P09/data/genomes/hg38/hg38.fa",
                        help="Reference genome FASTA")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for plots and CSVs")
    parser.add_argument("--pos-tf", type=str, default="NKX2-1",
                        help="Positive transcription factor name")
    parser.add_argument("--neg-tfs", type=str, nargs="+",
                        default=["GATA1", "MYOD1", "NKX2-5", "RXRA"],
                        help="Negative transcription factor names")
    parser.add_argument("--models-dir", type=str,
                        default="saved_models_final",
                        help="Base directory for model checkpoints")
    parser.add_argument("--model-type", type=str, default="VCNNBpnet",
                        help="Model architecture class name")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["all_mean", "core_mean", "flank_mean"],
                        help="Dataset names (subdirectories of models-dir)")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 3, 5, 8],
                        help="Experiment-count cutoffs for classification")
    parser.add_argument("--max-sequences", type=int, default=15000,
                        help="Max sequences per TF (for speed)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (cuda or cpu)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intersect_dir = Path(args.intersect_dir)

    # Load models
    loaded_models = {}
    for dataset in args.datasets:
        ckpt = Path(args.models_dir) / dataset / "standardize" / f"best_{args.model_type}.ckpt"
        if not ckpt.exists():
            print(f"Warning: checkpoint not found: {ckpt}")
            continue
        print(f"Loading {dataset}...")
        loaded_models[dataset] = load_model(str(ckpt), args.model_type, args.device)

    # Read and merge positive TF intersection file
    pos_file = intersect_dir / f"{args.pos_tf}.allintersect.bed"
    print(f"\nReading {args.pos_tf} from {pos_file}...")
    pos_df = pd.read_csv(pos_file, sep="\t", usecols=[0, 1, 2, 3],
                         names=["chrom", "start", "end", "num"], header=0)
    pos_merged = merge_intervals(pos_df)
    print(f"  {len(pos_df):,} intervals → {len(pos_merged):,} merged regions")

    # Extract positive sequences with nums
    pos_seqs, pos_nums = extract_sequences(
        pos_merged, args.genome, window=loaded_models[args.datasets[0]].input_length,
        max_sequences=args.max_sequences,
    )
    print(f"  Extracted {len(pos_seqs):,} sequences")

    # Score positive sequences with all models
    pos_model_scores = {}
    for name, model in loaded_models.items():
        print(f"  Scoring {args.pos_tf} with {name}...")
        pos_model_scores[name] = score_sequences(model, pos_seqs, args.device)

    # === Cutoff classification ===
    print("\n=== Cutoff Classification ===")
    cutoff_results = run_cutoff_classification(pos_model_scores, pos_nums, args.cutoffs, output_dir)
    pd.DataFrame(cutoff_results).to_csv(output_dir / "classification_cutoff_results.csv", index=False)

    # === Regression ===
    print("\n=== Regression Analysis ===")
    reg_results = run_regression(pos_model_scores, pos_nums, output_dir)
    pd.DataFrame(reg_results).to_csv(output_dir / "regression_results.csv", index=False)

    # === Classification vs negative TFs ===
    print("\n=== Classification vs Negative TFs ===")
    all_neg_results = []
    for neg_tf in args.neg_tfs:
        neg_file = intersect_dir / f"{neg_tf}.allintersect.bed"
        if not neg_file.exists():
            print(f"  Skipping {neg_tf} — file not found")
            continue
        neg_df = pd.read_csv(neg_file, sep="\t", usecols=[0, 1, 2, 3],
                             names=["chrom", "start", "end", "num"], header=0)
        neg_merged = merge_intervals(neg_df)
        neg_seqs, _ = extract_sequences(
            neg_merged, args.genome, window=loaded_models[args.datasets[0]].input_length,
            max_sequences=args.max_sequences,
        )
        print(f"  {neg_tf}: {len(neg_seqs):,} sequences")

        for name, model in loaded_models.items():
            neg_scores = score_sequences(model, neg_seqs, args.device)
            neg_dict = {neg_tf: neg_scores}
            results = run_classification_vs_negatives(
                pos_model_scores[name], neg_dict, name, output_dir,
            )
            all_neg_results.extend(results)

    pd.DataFrame(all_neg_results).to_csv(output_dir / "classification_vs_negatives.csv", index=False)

    # Combined ROC plot: all models × all negatives
    print("\n=== Combined ROC Plots ===")
    # Reload neg sequences for combined plot
    neg_model_scores = {}
    for neg_tf in args.neg_tfs:
        neg_file = intersect_dir / f"{neg_tf}.allintersect.bed"
        if not neg_file.exists():
            continue
        neg_df = pd.read_csv(neg_file, sep="\t", usecols=[0, 1, 2, 3],
                             names=["chrom", "start", "end", "num"], header=0)
        neg_merged = merge_intervals(neg_df)
        neg_seqs, _ = extract_sequences(
            neg_merged, args.genome, window=loaded_models[args.datasets[0]].input_length,
            max_sequences=args.max_sequences,
        )
        for name, model in loaded_models.items():
            neg_model_scores[(neg_tf, name)] = score_sequences(model, neg_seqs, args.device)

    # Plot by model
    fig, axes = plt.subplots(1, len(loaded_models), figsize=(7 * len(loaded_models), 6))
    if len(loaded_models) == 1:
        axes = [axes]
    for ax, model_name in zip(axes, loaded_models):
        for neg_tf in args.neg_tfs:
            key = (neg_tf, model_name)
            if key not in neg_model_scores:
                continue
            ps = pos_model_scores[model_name]
            ns = neg_model_scores[key]
            n = min(len(ps), len(ns))
            rng = np.random.RandomState(42)
            ps_ = rng.choice(ps, n, replace=False) if len(ps) > n else ps
            ns_ = rng.choice(ns, n, replace=False) if len(ns) > n else ns
            y_true = np.concatenate([np.ones(len(ps_)), np.zeros(len(ns_))])
            y_score = np.concatenate([ps_, ns_])
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=NEG_TF_COLORS.get(neg_tf, "#333"), linewidth=2,
                    label=f"vs {neg_tf} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_title(f"{model_name}", fontsize=14)
        ax.set_xlabel("FPR", fontsize=12)
        ax.set_ylabel("TPR", fontsize=12)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, linestyle='--', alpha=0.3)
    fig.suptitle(f"ROC: {args.pos_tf} vs All Negatives — By Model", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves_by_model.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot by negative TF
    fig2, axes2 = plt.subplots(1, len(args.neg_tfs), figsize=(6 * len(args.neg_tfs), 6))
    if len(args.neg_tfs) == 1:
        axes2 = [axes2]
    for ax, neg_tf in zip(axes2, args.neg_tfs):
        for model_name in loaded_models:
            key = (neg_tf, model_name)
            if key not in neg_model_scores:
                continue
            ps = pos_model_scores[model_name]
            ns = neg_model_scores[key]
            n = min(len(ps), len(ns))
            rng = np.random.RandomState(42)
            ps_ = rng.choice(ps, n, replace=False) if len(ps) > n else ps
            ns_ = rng.choice(ns, n, replace=False) if len(ns) > n else ns
            y_true = np.concatenate([np.ones(len(ps_)), np.zeros(len(ns_))])
            y_score = np.concatenate([ps_, ns_])
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=MODEL_COLORS.get(model_name, "#333"), linewidth=2,
                    label=f"{model_name} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax.set_title(f"{args.pos_tf} vs {neg_tf}", fontsize=14)
        ax.set_xlabel("FPR", fontsize=12)
        ax.set_ylabel("TPR", fontsize=12)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, linestyle='--', alpha=0.3)
    fig2.suptitle(f"ROC: {args.pos_tf} vs Negatives — By TF", fontsize=16, y=1.02)
    fig2.tight_layout()
    fig2.savefig(output_dir / "roc_curves_by_negative_tf.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
