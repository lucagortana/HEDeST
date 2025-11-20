from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from openpyxl import load_workbook


def box_plot_perf(
    file_infos: list[tuple[str, str, str]], level: str = "cells", title: str = "", savefig: str = None
) -> None:
    """
    Creates a box plot comparing performance metrics across multiple models.

    Args:
    - file_infos: List of tuples in the form (file_path, sheet_name, model_name)
    - level: 'cells' or 'slide', determines which metrics to plot
    - title: Plot title
    - savefig: Path to save figure, if desired
    """

    if level == "cells":
        metrics = ["Global Accuracy", "Balanced Accuracy", "Weighted F1 Score", "Weighted Precision", "Weighted Recall"]
    elif level == "slide":
        metrics = [
            "Pearson Correlation global",
            "Spearman Correlation global",
        ]
    else:
        raise ValueError("Level must be either 'cells' or 'slide'.")

    df_list = []
    for file_path, sheet_name, model_name in file_infos:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df["Model"] = model_name
        df_list.append(df[metrics + ["Model"]])

    combined_df = pd.concat(df_list, ignore_index=True)
    melted_df = pd.melt(combined_df, id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Value")

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted_df, x="Metric", y="Value", hue="Model")
    plt.title(title)
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.grid(True)

    if savefig:
        plt.savefig(savefig, dpi=300)
        print(f"Figure saved to {savefig}")

    plt.show()


def bar_plot_perf(
    file_infos: list[tuple[str, str, str, str]],
    level: str = "cells",
    title: str = "",
    figsize: tuple = (12, 6),
    savefig: str = None,
    context: str = "talk",
) -> None:
    """
    Creates a Seaborn bar plot with error bars for performance metrics across models,
    supporting custom colors.

    Args:
    - file_infos: List of tuples (file_path, sheet_name, model_name, color)
    - level: 'cells' or 'spots'
    - title: Plot title
    - figsize: Tuple specifying figure size
    - savefig: Path to save figure, if desired
    - context: Seaborn context ('paper', 'notebook', 'talk', or 'poster')
    """

    # Set style and context
    sns.set_context(context)
    sns.set_style("whitegrid")

    # Determine metrics to extract
    if level == "cells":
        metrics = {
            "Global Accuracy": "Global Acc.",
            "Balanced Accuracy": "Balanced Acc.",
            "Weighted F1 Score": "Weighted F1",
            "Weighted Precision": "Weighted Pre.",
            "Weighted Recall": "Weighted Rec.",
        }
    elif level == "spots":
        metrics = {"Pearson Correlation global": "Pearson Corr.", "Spearman Correlation global": "Spearman Corr."}
    else:
        raise ValueError("Level must be either 'cells' or 'spots'.")

    # Collect data and model colors
    df_rows = []
    model_colors = {}

    for file_info in file_infos:
        if len(file_info) == 4:
            file_path, sheet_name, model_name, color = file_info
            model_colors[model_name] = color
        else:
            file_path, sheet_name, model_name = file_info
            model_colors[model_name] = None  # fallback

        df = pd.read_excel(file_path, sheet_name=sheet_name).iloc[0]
        for long_name, short_name in metrics.items():
            df_rows.append(
                {
                    "Model": model_name,
                    "Metric": short_name,  # what will appear on x-axis
                    "Value": df[long_name],  # read from file using full name
                    "CI": df.get(f"{long_name} ci", 0),
                }
            )

    plot_df = pd.DataFrame(df_rows)

    # Create palette mapping
    palette = {model: color for model, color in model_colors.items() if color is not None}

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=plot_df, x="Metric", y="Value", hue="Model", palette=palette, errorbar=None, order=list(metrics.values())
    )

    # Add manual confidence intervals
    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        x = patch.get_x() + patch.get_width() / 2
        y = row["Value"]
        ci = row["CI"]
        ax.errorbar(x=x, y=y, yerr=ci, fmt="none", c="black", capsize=5, lw=1.3)

    # Aesthetic adjustments
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.legend(title="", loc="lower right", frameon=True)
    plt.grid(True)
    plt.tight_layout()

    # Save if needed
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {savefig}")

    plt.show()


def compute_statistics(metrics_list: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    """
    Computes mean and confidence intervals for a list of metrics.

    Args:
        metrics_list: List of dictionaries containing metrics from each run.

    Returns:
        Tuple containing the mean and confidence intervals for each metric.
    """

    df_metrics = pd.DataFrame(metrics_list)
    mean_values = df_metrics.mean().to_dict()
    std_values = df_metrics.std()
    count_values = df_metrics.count()
    se_values = std_values / np.sqrt(count_values)
    ci_values = {f"{key} ci": 1.96 * se for key, se in se_values.items()}

    return mean_values, ci_values


def save_metrics_to_excel(metrics_dict: dict[str, float], seed: int, excel_path: str) -> None:
    """
    Saves per-run metrics to an Excel file and maintain an aggregated summary sheet.

    Args:
        metrics_dict:
            Dictionary containing metrics for the current run.
            Keys are metric names and values are numeric results.
        seed:
            Random seed identifier for the run. This will be added as a separate
            column in the per-run sheet.
        excel_path:
            Path to the Excel file where results should be stored. If the file
            exists, it will be overwritten with updated sheets.
    """

    new_row = {"seed": seed, **metrics_dict}

    if os.path.exists(excel_path):
        # Load existing Excel file
        _ = load_workbook(excel_path)
        per_run_df = pd.read_excel(excel_path, sheet_name="per_run")
        per_run_df = pd.concat([per_run_df, pd.DataFrame([new_row])], ignore_index=True)

        # Compute summary
        metric_keys = [k for k in new_row.keys() if k != "seed"]
        metric_dicts = per_run_df[metric_keys].to_dict(orient="records")
        mean_vals, ci_vals = compute_statistics(metric_dicts)

        summary_row = {**mean_vals, **ci_vals}
        summary_df = pd.DataFrame([summary_row])

        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
            # Ensure summary is first sheet
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            per_run_df.to_excel(writer, sheet_name="per_run", index=False)
    else:
        # Create new Excel file
        per_run_df = pd.DataFrame([new_row])
        metric_keys = [k for k in new_row.keys() if k != "seed"]
        metric_dicts = per_run_df[metric_keys].to_dict(orient="records")
        mean_vals, ci_vals = compute_statistics(metric_dicts)
        summary_row = {**mean_vals, **ci_vals}
        summary_df = pd.DataFrame([summary_row])

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            per_run_df.to_excel(writer, sheet_name="per_run", index=False)


def plot_probability_histograms_with_uncertainty(preds: list[pd.DataFrame], ct_list: list[str]) -> None:
    """
    Plots histogram distributions of predicted probabilities with uncertainty bands.

    Args:
        preds:
            A list where each element is a DataFrame containing predicted
            probabilities. Each DataFrame must have a column for every cell type
            listed in `ct_list`.
        ct_list:
            List of cell-type (or class) names to visualize. Each name must match
            a column in each DataFrame in `preds`.
    """

    bin_edges = np.linspace(0, 1, 21)  # 20 bins: 0.0–0.05, ..., 0.95–1.0
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(18, 15))

    global_max = 0
    binned_per_ctype = []

    for ctype in ct_list:
        binned_counts = [np.histogram(df[ctype].values, bins=bin_edges)[0] for df in preds]
        binned_counts = np.array(binned_counts)
        binned_per_ctype.append(binned_counts)
        # global_max = max(global_max, (binned_counts.mean(axis=0) + binned_counts.std(axis=0)).max())
        global_max = 2000

    for i, ctype in enumerate(ct_list):
        binned_counts = binned_per_ctype[i]
        mean_counts = np.array(binned_counts).mean(axis=0)
        std_counts = np.array(binned_counts).std(axis=0)

        ax = plt.subplot(3, 3, i + 1)
        ax.bar(
            bin_centers,
            mean_counts,
            width=0.051,
            align="center",
            alpha=0.7,
            color="steelblue",
            yerr=std_counts,
            capsize=4,
        )
        ax.set_title(f"{ctype}", fontsize=14)
        ax.set_xlabel("Predicted Probability")
        if i == 0:
            ax.set_ylabel("Count")
        ax.grid(True)
        ax.set_ylim(0, global_max * 1.1)

    plt.suptitle("Probability Distributions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
