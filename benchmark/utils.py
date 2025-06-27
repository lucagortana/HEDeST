from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from openpyxl import load_workbook


def box_plot_perf(file_infos, level="cells", title="", savefig=None):
    """
    Creates a box plot comparing performance metrics across multiple models.

    Parameters:
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


def bar_plot_perf(file_infos, level="cells", title="", figsize=(12, 6), savefig=None):
    """
    Creates a bar plot with error bars for performance metrics across models.

    Parameters:
    - file_infos: List of tuples in the form (file_path, sheet_name, model_name)
    - level: 'cells' or 'slide', determines which metrics to plot
    - title: Plot title
    - savefig: Path to save figure, if desired
    """
    if level == "cells":
        metrics = ["Global Accuracy", "Balanced Accuracy", "Weighted F1 Score", "Weighted Precision", "Weighted Recall"]
    elif level == "slide":
        metrics = ["Pearson Correlation global", "Spearman Correlation global"]
    else:
        raise ValueError("Level must be either 'cells' or 'slide'.")

    df_rows = []

    for file_path, sheet_name, model_name in file_infos:
        df = pd.read_excel(file_path, sheet_name=sheet_name).iloc[0]
        row_data = {"Model": model_name}
        for metric in metrics:
            row_data["Metric"] = metric
            row_data["Value"] = df[metric]
            row_data["CI"] = df.get(f"{metric} ci", 0)
            df_rows.append(row_data.copy())

    plot_df = pd.DataFrame(df_rows)

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=plot_df,
        x="Metric",
        y="Value",
        hue="Model",
        errorbar=None,
        capsize=0.1,
        err_kws={"linewidth": 1.5},
        palette="Paired",
    )

    # Add error bars manually
    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        x = patch.get_x() + patch.get_width() / 2
        y = row["Value"]
        ci = row["CI"]
        ax.errorbar(x=x, y=y, yerr=ci, fmt="none", c="black", capsize=5, lw=1.5)

    plt.title(title)
    plt.legend(loc="lower right", frameon=True, title="")
    plt.ylabel("Performance")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.grid(False)
    plt.tight_layout()

    if savefig:
        plt.savefig(savefig, dpi=300)
        print(f"Figure saved to {savefig}")

    plt.show()


def compute_statistics(metrics_list):
    """
    Compute mean and confidence intervals for a list of metrics.

    Args:
        metrics_list (List[Dict[str, float]]): List of dictionaries containing metrics from each run.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: Tuple containing the mean and confidence intervals for each metric.
    """

    df_metrics = pd.DataFrame(metrics_list)
    mean_values = df_metrics.mean().to_dict()
    std_values = df_metrics.std()
    count_values = df_metrics.count()
    se_values = std_values / np.sqrt(count_values)
    ci_values = {f"{key} ci": 1.96 * se for key, se in se_values.items()}

    return mean_values, ci_values


def save_metrics_to_excel(metrics_dict, seed, excel_path):
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
