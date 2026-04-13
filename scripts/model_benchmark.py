import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_and_calculate_auc(
    tf_pos,
    tfs_neg,
    models,
    pr=False,
    mean=False,
    norm="none",
    model_type="SimpleCNN",
    base_results_path="/sc-projects/sc-proj-cc17-P09_TFBS/results",
    kmers_len=24,
):
    """
    Generates and saves ROC or PR curves and calculates AUC scores.

    Args:
        tf_pos (str): The name of the positive transcription factor.
        tfs_neg (list): A list of names for the negative transcription factors.
        models (list): A list of model identifiers to evaluate.
        pr (bool): If True, plots Precision-Recall curves. Otherwise, plots ROC curves.
        mean (bool): If True, uses the mean score from FIMO. Otherwise, uses the max score.
        norm (str): The type of normalization used, e.g., 'normalize', 'standardize'.
        model_type (str): The type of neural network model, e.g., 'SimpleCNN'.
        base_results_path (str): The base directory for results and predictions.
        kmers_len (int): The length of k-mers used in the model.

    Returns:
        list: A list of dictionaries, where each dictionary contains the AUC
              and the parameters used for a single model against a negative TF.
    """
    run_results = []
    n = len(tfs_neg)

    # Plot styling: larger figure and fonts for readability
    title_fs = 20
    label_fs = 18
    legend_fs = 18
    tick_fs = 14
    line_w = 2.0
    plt.rcParams.update({'font.size': tick_fs})
    # Create a subplot for each negative TF
    fig, axes = plt.subplots(
        nrows=n, figsize=(12, 10 * max(1, n)), squeeze=False
    )
    axes = axes.flatten()

    # --- Pre-load positive FIMO scores ---
    try:
        fimo_pos_path = os.path.join(base_results_path, "FIMO", "500", tf_pos, "fimo.tsv")
        hits_pos_df = pd.read_csv(fimo_pos_path, sep="\t", header=0, comment="#")
        
        # Using a more robust way to filter and aggregate
        pos_fimo_scores = hits_pos_df[hits_pos_df["motif_id"] == "MA1994.1"].groupby("sequence_name")["score"]
        pos_fimo = pos_fimo_scores.mean() if mean else pos_fimo_scores.max()
    except FileNotFoundError:
        print(f"Warning: FIMO file not found for positive TF '{tf_pos}'. Skipping FIMO curves.")
        pos_fimo = None

    curve_type_str = "PR" if pr else "ROC"
    score_type_str = "mean" if mean else "max"

    for ax, tf_neg in zip(axes, tfs_neg):
        # --- Plot curves for each NN model ---
        for model in models:
            try:
                # Construct file paths cleanly
                pred_path_pos = os.path.join(base_results_path, "predictions", "500", f"{kmers_len}", f"{model}", norm, f"sequences_{tf_pos}_{model_type}.csv")
                pred_path_neg = os.path.join(base_results_path, "predictions", "500", f"{kmers_len}", f"{model}", norm, f"sequences_{tf_neg}_{model_type}.csv")

                df_pos = pd.read_csv(pred_path_pos, index_col=0)
                df_neg = pd.read_csv(pred_path_neg, index_col=0)

                score_col = 'mean_score' if mean else 'max_score'
                pos_scores = df_pos[score_col]
                neg_scores = df_neg[score_col]

                # Subsample negative set for ROC curve to balance classes
                if len(neg_scores) > len(pos_scores) and not pr:
                    neg_scores = neg_scores.sample(n=len(pos_scores), random_state=42)

                y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
                y_score = np.concatenate([pos_scores, neg_scores])

                if pr:
                    precision, recall, _ = precision_recall_curve(y_true, y_score)
                    curve_auc = auc(recall, precision)
                    ax.plot(recall, precision, label=f"{model.replace('_mean', '').upper()} (AUC={curve_auc:.3f})", linewidth=line_w)
                else:
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    curve_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f"{model.replace('_mean', '').upper()} (AUC={curve_auc:.3f})", linewidth=line_w)
                    # Store result for this specific run
                    run_results.append({
                        'model': model,
                        'tf_neg': tf_neg,
                        'auc': curve_auc
                    })

            except FileNotFoundError as e:
                print(f"Could not find prediction file, skipping model '{model}' for TF '{tf_neg}'. Details: {e}")
                continue

        # --- Plot FIMO curve for comparison ---
        neg_fimo = None
        if pos_fimo is not None:
            try:
                fimo_neg_path = os.path.join(base_results_path, "FIMO", "500", tf_neg, "fimo.tsv")
                hits_neg_df = pd.read_csv(fimo_neg_path, sep="\t", header=0, comment="#")

                neg_fimo_scores = hits_neg_df[hits_neg_df["motif_id"] == "MA1994.1"].groupby("sequence_name")["score"]
                neg_fimo = neg_fimo_scores.mean() if mean else neg_fimo_scores.max()

                if len(neg_fimo) > len(pos_fimo) and not pr:
                    neg_fimo = neg_fimo.sample(n=len(pos_fimo), random_state=42)

                y_true_fimo = [1] * len(pos_fimo) + [0] * len(neg_fimo)
                y_score_fimo = list(pos_fimo) + list(neg_fimo)

                if pr:
                    precision, recall, _ = precision_recall_curve(y_true_fimo, y_score_fimo)
                    fimo_auc = auc(recall, precision)
                    ax.plot(recall, precision, label=f'PWM (AUC={fimo_auc:.3f})', linestyle='--', color='blue', linewidth=line_w)
                else:
                    fpr, tpr, _ = roc_curve(y_true_fimo, y_score_fimo)
                    fimo_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'PWM (AUC={fimo_auc:.3f})', linestyle='--', color='blue', linewidth=line_w)

            except FileNotFoundError:
                print(f"Warning: FIMO file not found for negative TF '{tf_neg}'. Skipping FIMO curve.")

        # --- Finalize subplot appearance ---
        title = f"{curve_type_str}-curve: {tf_pos.replace('-','.')} vs {tf_neg.replace('-','.')}"

        if pr:
            ratio = len(pos_fimo) / (len(pos_fimo) + len(neg_fimo)) if pos_fimo is not None and neg_fimo is not None and (len(pos_fimo) + len(neg_fimo)) > 0 else 0
            ax.plot([0, 1], [ratio, ratio], 'k--', linewidth=1.0, label=f'Baseline ({ratio:.2f})')
            ax.set_xlabel("Recall", fontsize=label_fs)
            ax.set_ylabel("Precision", fontsize=label_fs)
        else:
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0)
            ax.set_xlabel("False Positive Rate", fontsize=label_fs)
            ax.set_ylabel("True Positive Rate", fontsize=label_fs)

        ax.set_title(title, fontsize=title_fs)
        ax.legend(loc='best', fontsize=legend_fs)
        ax.tick_params(axis='both', labelsize=tick_fs)
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout(pad=3.0)
    
    # --- Save the figure ---
    plot_dir = os.path.join(base_results_path, "plots_final")
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"{tf_pos}_vs_{'_'.join(tfs_neg)}_{curve_type_str}_{score_type_str}_{norm}_{model_type}_{kmers_len}.png"
    plt.savefig(os.path.join(plot_dir, filename), dpi=150)
    plt.close()
    
    return run_results


if __name__ == "__main__":
    # --- Configuration ---
    BASE_RESULTS_PATH = "/sc-projects/sc-proj-cc17-P09_TFBS/results"
    TF_POS = "NKX2-1"
    TFS_NEG = ["GATA1", "MYOD1", "NKX2-5", "RXRA"]

    # Define parameter sets to iterate over
    MODEL_TYPES = ["VCNNBpnet"]
    NORMALIZATION_METHODS = ["standardize"]
    MODELS_LIST = [
        "core_mean",
        "flank_mean",
        "all_mean",
    ]
    SCORE_AGGREGATION = [False]  # False=max, True=mean
    CURVE_TYPE = [False]  # False=ROC, True=PR. Only doing ROC for the final table.
    KMERS_LEN = [24, 500]

    # --- Main Execution ---
    master_results_list = []

    # Use itertools.product to cleanly iterate through all parameter combinations
    param_combinations = itertools.product(
        MODEL_TYPES, NORMALIZATION_METHODS, SCORE_AGGREGATION, CURVE_TYPE, KMERS_LEN
    )

    print("Starting analysis. This may take a while...")
    for i, (model_type, norm, use_mean, use_pr, kmers_len) in enumerate(param_combinations):
        print(f"Running combination {i+1}: model_type={model_type}, norm={norm}, mean={use_mean}, pr={use_pr}, kmers_len={str(kmers_len)}")

        # Call the function to generate plots and get AUCs for the current combination
        run_aucs = plot_and_calculate_auc(
            tf_pos=TF_POS,
            tfs_neg=TFS_NEG,
            models=MODELS_LIST,
            pr=use_pr,
            mean=use_mean,
            norm=norm,
            model_type=model_type,
            base_results_path=BASE_RESULTS_PATH,
            kmers_len=kmers_len,
        )

        # Add the parameters of the current run to each result dictionary
        for result in run_aucs:
            result['tf_pos'] = TF_POS
            result['model_type'] = model_type
            result['normalisation'] = norm
            result['score_type'] = 'mean' if use_mean else 'max'
            result['curve_type'] = 'PR' if use_pr else 'ROC'
            result['kmers_len'] = kmers_len
            master_results_list.append(result)

    # --- Final Output ---
    # Convert the master list of results into a single DataFrame
    results_df = pd.DataFrame(master_results_list)

    # Reorder columns for clarity
    column_order = [
        'tf_pos', 'tf_neg', 'model_type', 'model', 'normalisation',
        'score_type', 'curve_type', 'kmers_len', 'auc'
    ]
    results_df = results_df[column_order]

    # Save the comprehensive DataFrame to a single CSV file
    output_filename = os.path.join(BASE_RESULTS_PATH, "plots_final", "comprehensive_auc_results.csv")
    results_df.to_csv(output_filename, index=False)

    print(f"\nAnalysis complete. All ROC AUC results saved to:\n{output_filename}")
    print("\nFinal DataFrame head:")
    print(results_df.head())
