from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def generate_X_perm(X_sparse):
    X_sparse_perm = []
    for row_X in X_sparse:
        indices = np.where(row_X > 0)[0]
        subrow_X = row_X[indices[0] : indices[-1] + 1]
        np.random.shuffle(subrow_X)
        row_X[indices[0] : indices[-1] + 1] = subrow_X
        X_sparse_perm.append(row_X)
    X_sparse_perm = np.stack(X_sparse_perm)
    X_perm = X_sparse_perm.sum(axis=0)
    return X_perm, X_sparse_perm


def compute_stats(true_labels, predicted_labels, per_class=False):
    global_accuracy = accuracy_score(true_labels, predicted_labels)

    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)

    # Weighted metrics
    weighted_f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    weighted_precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
    weighted_recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)

    metrics = {
        "Global Accuracy": global_accuracy,
        "Balanced Accuracy": balanced_acc,
        "Weighted F1 Score": weighted_f1,
        "Weighted Precision": weighted_precision,
        "Weighted Recall": weighted_recall,
    }

    if per_class:
        unique_classes = np.unique(true_labels)
        f1_per_class = f1_score(true_labels, predicted_labels, average=None, zero_division=0)
        precision_per_class = precision_score(true_labels, predicted_labels, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
        cm = confusion_matrix(true_labels, predicted_labels)

        metrics.update(
            {
                "F1 Score (Per Class)": dict(zip(unique_classes, f1_per_class)),
                "Precision (Per Class)": dict(zip(unique_classes, precision_per_class)),
                "Recall (Per Class)": dict(zip(unique_classes, recall_per_class)),
                "Confusion Matrix": pd.DataFrame(cm, columns=list(unique_classes), index=list(unique_classes)),
            }
        )

    return metrics


def aggregate_stats(metrics_list):
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
