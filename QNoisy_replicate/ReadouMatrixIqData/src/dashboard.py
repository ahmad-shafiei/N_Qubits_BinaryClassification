import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

METRICS = ["FidelityMean", "L1Mean", "KLMean"]

NOISE_TYPES = ["Synthetic", "Experimental Single", "Experimental Correlated"]
TRAINING_ORDER = ["No Training", "Synthetic",
                  "Experimental Single", "Experimental Correlated"]

DATASET_LABELS = {
    "synthetic": "Synthetic",
    "experimental_single": "Experimental Single",
    "experimental_correlated": "Experimental Correlated",
}

METRIC_META = {
    "FidelityMean": {
        "cmap": "viridis",
        "title": "Fidelity (higher = better)",
        "higher_is_better": True,
        "ylabel": "Fidelity",
    },
    "L1Mean": {
        "cmap": "viridis_r",
        "title": "L1 Error (lower = better)",
        "higher_is_better": False,
        "ylabel": "L1",
    },
    "KLMean": {
        "cmap": "viridis_r",
        "title": "KL Divergence (lower = better)",
        "higher_is_better": False,
        "ylabel": "KL",
    },
}


def _results_by_noise(results_list):
    return {
        DATASET_LABELS.get(r["Dataset"], r["Dataset"]): r
        for r in results_list
    }


def build_matched_comparison_df(experiments):
    """Baseline vs matched-trained metrics for each (circuit, noise) pair."""
    rows = []
    circuits = sorted({circuit for circuit, _ in experiments.keys()})

    for circuit in circuits:
        baseline = _results_by_noise(experiments.get((circuit, "No Training"), []))

        for noise_label in NOISE_TYPES:
            row = {"Circuit": circuit, "Noise": noise_label}

            if noise_label in baseline:
                for metric in METRICS:
                    row[f"Baseline_{metric}"] = baseline[noise_label].get(metric)

            trained = _results_by_noise(experiments.get((circuit, noise_label), []))
            if noise_label in trained:
                for metric in METRICS:
                    row[f"Trained_{metric}"] = trained[noise_label].get(metric)

            if "Baseline_FidelityMean" in row and "Trained_FidelityMean" in row:
                row["Delta_FidelityMean"] = (
                    row["Trained_FidelityMean"] - row["Baseline_FidelityMean"]
                )
            if "Baseline_L1Mean" in row and "Trained_L1Mean" in row:
                row["Delta_L1Mean"] = row["Baseline_L1Mean"] - row["Trained_L1Mean"]
            if "Baseline_KLMean" in row and "Trained_KLMean" in row:
                row["Delta_KLMean"] = row["Baseline_KLMean"] - row["Trained_KLMean"]

            rows.append(row)

    return pd.DataFrame(rows)


def show_matched_comparison_table(comparison_df, metric="FidelityMean"):
    meta = METRIC_META.get(metric, {"title": metric})
    cols = ["Circuit", "Noise",
            f"Baseline_{metric}", f"Trained_{metric}", f"Delta_{metric}"]
    cols = [c for c in cols if c in comparison_df.columns]
    view = comparison_df[cols].copy()
    view.columns = ["Circuit", "Noise", "Baseline", "Trained", "Delta"]
    print(f"\n================ {metric} | {meta['title']} (matched train/test) ================")
    print(view.round(3).to_string(index=False))
    return view


def show_all_matched_tables(comparison_df, round_digits=3):
    for metric in METRICS:
        show_matched_comparison_table(comparison_df, metric)


def matched_metrics_dashboard(comparison_df, metrics=None):
    """Grouped bar charts: baseline vs trained for each noise type."""
    metrics = metrics or METRICS
    circuits = comparison_df["Circuit"].unique()

    for circuit in circuits:
        df_c = comparison_df[comparison_df["Circuit"] == circuit]
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]

        x = range(len(NOISE_TYPES))
        width = 0.35

        for ax, metric in zip(axes, metrics):
            meta = METRIC_META.get(metric, {"title": metric, "ylabel": metric})
            baseline_col = f"Baseline_{metric}"
            trained_col = f"Trained_{metric}"

            baseline_vals = [
                df_c.loc[df_c["Noise"] == n, baseline_col].values[0]
                if n in df_c["Noise"].values and baseline_col in df_c.columns
                else float("nan")
                for n in NOISE_TYPES
            ]
            trained_vals = [
                df_c.loc[df_c["Noise"] == n, trained_col].values[0]
                if n in df_c["Noise"].values and trained_col in df_c.columns
                else float("nan")
                for n in NOISE_TYPES
            ]

            ax.bar([i - width / 2 for i in x], baseline_vals, width,
                   label="No Training", alpha=0.85)
            ax.bar([i + width / 2 for i in x], trained_vals, width,
                   label="Matched Training", alpha=0.85)
            ax.set_xticks(list(x))
            ax.set_xticklabels(NOISE_TYPES, rotation=15, ha="right")
            ax.set_ylabel(meta["ylabel"])
            ax.set_title(meta["title"])
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"{circuit} | matched train/test evaluation", fontsize=14)
        plt.tight_layout()
        plt.show()


# --- legacy helpers (kept for compatibility) ---

def build_summary_df(experiments):
    rows = []
    for (circuit, training_noise), results_list in experiments.items():
        row = {"Circuit": circuit, "Training Noise": training_noise}
        for r in results_list:
            dataset = DATASET_LABELS.get(r["Dataset"], r["Dataset"])
            for m in METRICS:
                if m in r:
                    row[f"{dataset}_{m}"] = r[m]
        rows.append(row)

    df = pd.DataFrame(rows).set_index(["Circuit", "Training Noise"])
    ordered_cols = [f"{ds}_{m}" for ds in NOISE_TYPES for m in METRICS]
    df = df[[c for c in ordered_cols if c in df.columns]]
    return df


def metric_view(summary_df, metric):
    cols = [f"{ds}_{metric}" for ds in NOISE_TYPES]
    cols = [c for c in cols if c in summary_df.columns]
    view = summary_df[cols].copy()
    view.columns = [c.rsplit("_", 1)[0] for c in cols]
    return view


def show_summary_table(summary_df, metric="FidelityMean"):
    view = metric_view(summary_df, metric)
    circuits = summary_df.index.get_level_values(0).unique()
    order = [(c, t) for c in circuits for t in TRAINING_ORDER]
    order = [k for k in order if k in view.index]
    view = view.reindex(order)
    print(view.round(3).to_string())


def full_dashboard(summary_df, metric="FidelityMean"):
    meta = METRIC_META.get(metric, {})
    circuits = summary_df.index.get_level_values(0).unique()
    for circuit in circuits:
        df_c = metric_view(summary_df.xs(circuit), metric)
        df_c = df_c.reindex([t for t in TRAINING_ORDER if t in df_c.index])
        plt.figure(figsize=(8, 5))
        sns.heatmap(df_c, annot=True, fmt=".3f", cmap=meta.get("cmap", "viridis"))
        plt.title(f"{circuit} | {metric}")
        plt.xlabel("Test Noise (matched only for trained rows)")
        plt.ylabel("Training Condition")
        plt.tight_layout()
        plt.show()
