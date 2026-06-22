import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

METRICS = ["FidelityMean", "L1Mean", "KLMean"]

NOISE_TYPES = ["Synthetic", "Experimental Single", "Experimental Correlated"]
CIRCUIT_ORDER = ["Independent", "ZZ FeatureMap"]
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
    """Baseline (No Training) vs matched-trained for each (circuit, noise).

    Trained metrics come from key (circuit, training_noise_label) where
    training_noise_label equals the display noise name (e.g. 'Synthetic').
    """
    rows = []
    circuits = [c for c in CIRCUIT_ORDER
                  if any(k[0] == c for k in experiments.keys())]
    if not circuits:
        circuits = sorted({circuit for circuit, _ in experiments.keys()})

    for circuit in circuits:
        baseline = _results_by_noise(
            experiments.get((circuit, "No Training"), []))

        for noise_label in NOISE_TYPES:
            row = {"Circuit": circuit, "Noise": noise_label}

            if noise_label in baseline:
                for metric in METRICS:
                    row[f"Baseline_{metric}"] = baseline[noise_label].get(metric)

            # trained phase stored under training_noise == noise_label
            trained = _results_by_noise(
                experiments.get((circuit, noise_label), []))
            if noise_label in trained:
                for metric in METRICS:
                    row[f"Trained_{metric}"] = trained[noise_label].get(metric)

            for metric in METRICS:
                b_key, t_key = f"Baseline_{metric}", f"Trained_{metric}"
                if b_key in row and t_key in row:
                    if metric == "FidelityMean":
                        row[f"Delta_{metric}"] = row[t_key] - row[b_key]
                    else:
                        # L1/KL: improvement = baseline - trained (lower is better)
                        row[f"Delta_{metric}"] = row[b_key] - row[t_key]

            rows.append(row)

    return pd.DataFrame(rows)


def show_matched_comparison_table(comparison_df, metric="FidelityMean"):
    meta = METRIC_META.get(metric, {"title": metric})
    cols = ["Circuit", "Noise",
            f"Baseline_{metric}", f"Trained_{metric}", f"Delta_{metric}"]
    cols = [c for c in cols if c in comparison_df.columns]
    view = comparison_df[cols].copy()
    view.columns = ["Circuit", "Noise", "Baseline", "Trained", "Delta"]
    # only rows where a matched trained model exists
    view = view.dropna(subset=["Trained"])
    print(f"\n================ {metric} | {meta['title']} (matched train/test) ================")
    print(view.round(3).to_string(index=False))
    return view


def show_all_matched_tables(comparison_df):
    for metric in METRICS:
        show_matched_comparison_table(comparison_df, metric)


def plot_fidelity_bars(comparison_df):
    """Grouped bar chart: baseline vs trained fidelity for each circuit."""
    metric = "FidelityMean"
    baseline_col = f"Baseline_{metric}"
    trained_col = f"Trained_{metric}"

    for circuit in comparison_df["Circuit"].unique():
        df_c = comparison_df[comparison_df["Circuit"] == circuit].copy()
        df_c = df_c.dropna(subset=[trained_col])

        if df_c.empty:
            print(f"No matched trained results for {circuit}")
            continue

        noises = df_c["Noise"].tolist()
        x = np.arange(len(noises))
        width = 0.35

        baseline_vals = df_c[baseline_col].values
        trained_vals = df_c[trained_col].values

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width / 2, baseline_vals, width,
               label="No Training (baseline)", color="#c44e52", alpha=0.75)
        ax.bar(x + width / 2, trained_vals, width,
               label="After Training", color="#4c72b0", alpha=0.75)

        for i, (b, t) in enumerate(zip(baseline_vals, trained_vals)):
            ax.annotate(f"{(t - b)*100:+.3f}", (x[i] + width / 2, t),  ##gfggg
                        ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(noises, rotation=12, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean Fidelity", fontsize=10)
        ax.set_title(f"{circuit} | Fidelity: baseline vs training", fontsize=10)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()


def matched_metrics_dashboard(comparison_df, metrics=None):
    """Grouped bar charts for all metrics, per circuit."""
    metrics = metrics or METRICS

    for circuit in comparison_df["Circuit"].unique():
        df_c = comparison_df[comparison_df["Circuit"] == circuit].copy()
        df_c = df_c.dropna(subset=[f"Trained_{metrics[0]}"])
        if df_c.empty:
            continue

        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]

        noises = df_c["Noise"].tolist()
        x = np.arange(len(noises))
        width = 0.35

        for ax, metric in zip(axes, metrics):
            meta = METRIC_META.get(metric, {"title": metric, "ylabel": metric})
            baseline_col = f"Baseline_{metric}"
            trained_col = f"Trained_{metric}"

            baseline_vals = df_c[baseline_col].values
            trained_vals = df_c[trained_col].values

            ax.bar(x - width / 2, baseline_vals, width,
                   label="No Training", alpha=0.85)
            ax.bar(x + width / 2, trained_vals, width,
                   label="Training", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(noises, rotation=12, ha="right")
            ax.set_ylabel(meta["ylabel"])
            ax.set_title(meta["title"])
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"{circuit} |train/test", fontsize=14)
        plt.tight_layout()
        plt.show()


# alias used in notebook
metrics_dashboard = matched_metrics_dashboard


# --- legacy wide-format helpers ---

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
    return df[[c for c in ordered_cols if c in df.columns]]


def metric_view(summary_df, metric):
    cols = [f"{ds}_{metric}" for ds in NOISE_TYPES if f"{ds}_{metric}" in summary_df.columns]
    view = summary_df[cols].copy()
    view.columns = [c.rsplit("_", 1)[0] for c in cols]
    return view


def show_summary_table(summary_df, metric="FidelityMean"):
    view = metric_view(summary_df, metric)
    circuits = summary_df.index.get_level_values(0).unique()
    order = [(c, t) for c in circuits for t in TRAINING_ORDER]
    order = [k for k in order if k in view.index]
    print(view.reindex(order).round(3).to_string())


def full_dashboard(summary_df, metric="FidelityMean"):
    meta = METRIC_META.get(metric, {})
    for circuit in summary_df.index.get_level_values(0).unique():
        df_c = metric_view(summary_df.xs(circuit), metric)
        df_c = df_c.reindex([t for t in TRAINING_ORDER if t in df_c.index])
        plt.figure(figsize=(8, 5))
        sns.heatmap(df_c, annot=True, fmt=".3f", cmap=meta.get("cmap", "viridis"))
        plt.title(f"{circuit} | {metric}")
        plt.tight_layout()
        plt.show()
