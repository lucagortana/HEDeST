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


def prepare_data(ground_truth, spot_dict, embeddings, batch_size=30):
    # 1. Unique sorted cell IDs across both spot_dict and ground_truth
    cell_ids = sorted(set(ground_truth.index.astype(str)) & set(sum(spot_dict.values(), [])))
    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
    N = len(cell_ids)
    print("Cell IDs ok")

    # 2. Sorted spot IDs
    spot_ids = sorted(spot_dict.keys())
    spot_id_to_idx = {sid: i for i, sid in enumerate(spot_ids)}
    M = len(spot_ids)
    print("Spot IDs ok")

    # 3. Map one-hot labels to class indices
    cell_type_names = ground_truth.columns.tolist()
    # L = len(cell_type_names)
    gt_labels = ground_truth.loc[ground_truth.index.astype(str).isin(cell_ids)]
    label_array = gt_labels.to_numpy()
    type_indices = label_array.argmax(axis=1) + 1  # get integer labels: 1, 2, ..., L

    # Map cell_id → class index
    cell_id_to_label = {str(cid): type_idx for cid, type_idx in zip(gt_labels.index.astype(str), type_indices)}
    print("Cell type labels ok")

    # 4. B: Features per cell (N x K)
    K = len(next(iter(embeddings.values())))  # 2048
    B = np.zeros((N, K))

    for i, cid in enumerate(cell_ids):
        if cid in embeddings:
            B[i] = embeddings[cid].detach().cpu().numpy()
        else:
            raise ValueError(f"Cell ID {cid} found in ground_truth or spot_dict but not in embeddings.")
    print("Embeddings ok")

    # 5. A and X_sparse
    A = np.zeros((M, N), dtype=int)
    X_sparse = np.zeros((M, N), dtype=int)

    for sid, cell_list in spot_dict.items():
        m = spot_id_to_idx[sid]
        for cid in cell_list:
            if cid in cell_id_to_idx and cid in cell_id_to_label:
                n = cell_id_to_idx[cid]
                A[m, n] = 1
                X_sparse[m, n] = cell_id_to_label[cid]
    print("Adjacency matrix A and sparse matrix X_sparse ok")

    # 6. X: flat label vector (N,)
    X = np.zeros(N, dtype=int)
    for cid, n in cell_id_to_idx.items():
        if cid in cell_id_to_label:
            X[n] = cell_id_to_label[cid]

    print("Flat label vector X ok")

    # 7. New organization of matrices
    new_order = []
    seen = set()

    for row in X_sparse:
        cell_indices = np.where(row > 0)[0]
        for idx in cell_indices:
            if idx not in seen:
                new_order.append(idx)
                seen.add(idx)

    X_sparse = X_sparse[:, new_order]
    A = A[:, new_order]
    B = B[new_order]
    X = X[new_order]

    X_perm, _ = generate_X_perm(X_sparse.copy())
    print("New organization of matrices completed.")

    # 8. Divide matrices into batches
    A_batches = []
    B_batches = []
    X_perm_batches = []

    num_spots = A.shape[0]

    for start in range(0, num_spots, batch_size):
        end = min(start + batch_size, num_spots)

        A_batch = A[start:end]
        used_cell_indices = np.where(A_batch.sum(axis=0) > 0)[0]

        A_batch = A_batch[:, used_cell_indices]
        B_batch = B[used_cell_indices]
        X_perm_batch = X_perm[used_cell_indices]

        A_batches.append(A_batch)
        B_batches.append(B_batch)
        X_perm_batches.append(X_perm_batch)

    print("->Data preparation completed.")

    return (
        A_batches[:20],
        B_batches[:20],
        X_perm_batches[:20],
        X_perm[: sum(A_batches[i].shape[1] for i in range(20))],
        X[: sum(A_batches[i].shape[1] for i in range(20))],
    )
