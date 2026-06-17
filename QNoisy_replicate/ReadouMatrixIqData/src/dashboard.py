import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

METRICS = ["FidelityMean", "L1Mean", "KLMean"]

DATASET_ORDER = ["Synthetic", "Experimental Single", "Experimental Correlated"]
TRAINING_ORDER = ["No Training", "Synthetic",
                  "Experimental Single", "Experimental Correlated"]

DATASET_LABELS = {
    "synthetic": "Synthetic",
    "experimental_single": "Experimental Single",
    "experimental_correlated": "Experimental Correlated",
}

METRIC_META = {
    "FidelityMean": {"cmap": "viridis", "title": "Fidelity"},
    "L1Mean": {"cmap": "viridis_r", "title": "L1 Error"},
    "KLMean": {"cmap": "viridis_r", "title": "KL Divergence"},
}


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

    ordered_cols = [f"{ds}_{m}" for ds in DATASET_ORDER for m in METRICS]
    df = df[[c for c in ordered_cols if c in df.columns]]

    return df


def metric_view(summary_df, metric):
    cols = [f"{ds}_{metric}" for ds in DATASET_ORDER]
    cols = [c for c in cols if c in summary_df.columns]

    view = summary_df[cols].copy()
    view.columns = [c.split("_")[0] for c in view.columns]
    return view


def full_dashboard(summary_df, metric="FidelityMean"):
    meta = METRIC_META.get(metric, {})

    circuits = summary_df.index.get_level_values(0).unique()

    for circuit in circuits:
        df_c = metric_view(summary_df.xs(circuit), metric)
        df_c = df_c.reindex(TRAINING_ORDER)

        plt.figure(figsize=(8, 5))
        sns.heatmap(df_c, annot=True, cmap=meta.get("cmap", "viridis"))
        plt.title(f"{circuit} | {metric}")
        plt.show()


def show_summary_table(summary_df, metric="FidelityMean"):
    view = metric_view(summary_df, metric)

    circuits = summary_df.index.get_level_values(0).unique()
    order = [(c, t) for c in circuits for t in TRAINING_ORDER]
    order = [k for k in order if k in view.index]

    view = view.reindex(order)
    print(view.round(3).to_string())