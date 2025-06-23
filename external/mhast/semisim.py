from __future__ import annotations

import argparse
import json
import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
from celltype_permutation import hierarchical_permutations
from joblib import delayed
from joblib import Parallel
from utils import aggregate_stats
from utils import compute_stats
from utils import generate_X_perm


def run_batch_with_timeout(A_batch, X_perm_batch, B_batch, timeout=300):
    def target_fn(queue):
        result = hierarchical_permutations(A_batch, X_perm_batch, B_batch)
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target_fn, args=(queue,))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        print("-> Skipping batch (timeout >5min)")
        return None

    return queue.get()


def run_single_iteration(ground_truth, spot_dict, embeddings):
    A_batches, B_batches, X_perm_batches, X_perm, X = prepare_data(ground_truth, spot_dict, embeddings)

    X_global_batches = []

    for batch_idx, (A_batch, X_perm_batch, B_batch) in enumerate(zip(A_batches, X_perm_batches, B_batches)):
        print(f"Processing batch {batch_idx + 1}/{len(A_batches)}")

        X_global_batch = run_batch_with_timeout(A_batch, X_perm_batch, B_batch)

        if X_global_batch is not None:
            X_global_batches.append(X_global_batch)
        else:
            print(f"-> Padding skipped batch {batch_idx} with zeros")
            X_global_batches.append(np.zeros_like(X_perm_batch))

    X_global = np.concatenate(X_global_batches)
    return X_global, X_perm, X


def prepare_data(ground_truth, spot_dict, embeddings, batch_size=30):
    # 1. Unique sorted cell IDs across both spot_dict and ground_truth
    cell_ids = sorted(set(ground_truth.index.astype(str)) & set(sum(spot_dict.values(), [])))
    cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
    N = len(cell_ids)

    # 2. Sorted spot IDs
    spot_ids = sorted(spot_dict.keys())
    spot_id_to_idx = {sid: i for i, sid in enumerate(spot_ids)}
    M = len(spot_ids)

    # 3. Map one-hot labels to class indices
    # cell_type_names = ground_truth.columns.tolist()
    # L = len(cell_type_names)
    gt_labels = ground_truth.loc[ground_truth.index.astype(str).isin(cell_ids)]
    label_array = gt_labels.to_numpy()
    type_indices = label_array.argmax(axis=1) + 1  # get integer labels: 1, 2, ..., L

    # Map cell_id → class index
    cell_id_to_label = {str(cid): type_idx for cid, type_idx in zip(gt_labels.index.astype(str), type_indices)}

    # 4. B: Features per cell (N x K)
    K = len(next(iter(embeddings.values())))  # 2048
    B = np.zeros((N, K))

    for i, cid in enumerate(cell_ids):
        if cid in embeddings:
            B[i] = embeddings[cid].detach().cpu().numpy()
        else:
            raise ValueError(f"Cell ID {cid} found in ground_truth or spot_dict but not in embeddings.")

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

    # 6. X: flat label vector (N,)
    X = np.zeros(N, dtype=int)
    for cid, n in cell_id_to_idx.items():
        if cid in cell_id_to_label:
            X[n] = cell_id_to_label[cid]

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

    return A_batches, B_batches, X_perm_batches, X_perm, X


def main(data_path, gt_filename, spot_dict_filename, embeddings_filename, n_iter, output_csv):

    ground_truth_path = os.path.join(data_path, gt_filename)
    spot_dict_path = os.path.join(data_path, spot_dict_filename)
    embeddings_path = os.path.join(data_path, embeddings_filename)
    print(f"-> Loading data from {data_path}")

    ground_truth = pd.read_csv(ground_truth_path, index_col=0)
    embeddings = torch.load(embeddings_path)

    with open(spot_dict_path, "r") as file:
        spot_dict = json.load(file)
    print("-> Data loaded successfully")

    cell_ids = sorted(set(ground_truth.index.astype(str)) & set(sum(spot_dict.values(), [])))

    num_workers = min(5, multiprocessing.cpu_count())
    results = Parallel(n_jobs=num_workers)(
        delayed(run_single_iteration)(ground_truth, spot_dict, embeddings) for _ in range(n_iter)
    )

    print("-> All iterations completed")

    X_globals_all, X_perms_all, Xs_all = zip(*results)

    X_globals_all = [np.array(x) for x in X_globals_all]
    X_perms_all = [np.array(x) for x in X_perms_all]
    Xs_all = [np.array(x) for x in Xs_all]

    X_globals_stack = np.stack(X_globals_all)  # shape: (n_iter, n_cells)
    valid_mask = ~np.any(X_globals_stack == 0, axis=0)

    X_globals_all = [Xg[valid_mask] for Xg in X_globals_all]
    X_perms_all = [Xp[valid_mask] for Xp in X_perms_all]
    Xs_all = [X[valid_mask] for X in Xs_all]

    remaining_indices = list(np.where(valid_mask)[0])
    remaining_cell_ids = [cell_ids[i] for i in remaining_indices]
    pd.Series(remaining_cell_ids).to_csv("remaining_cell_ids.csv", index=False, header=False)

    metrics_before_list = []
    metrics_after_list = []
    for X, X_perm, X_global in zip(Xs_all, X_perms_all, X_globals_all):
        metrics_before = compute_stats(X, X_perm, per_class=False)
        metrics_after = compute_stats(X, X_global, per_class=False)
        metrics_before_list.append(metrics_before)
        metrics_after_list.append(metrics_after)

    mean_values_before, ci_values_before = aggregate_stats(metrics_before_list)
    mean_values_after, ci_values_after = aggregate_stats(metrics_after_list)

    result_df = pd.DataFrame(
        [
            {"state": "before", **mean_values_before, **ci_values_before},
            {"state": "after", **mean_values_after, **ci_values_after},
        ]
    )
    result_df.to_csv(output_csv, index=False)
    print(f"-> Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hierarchical permutation evaluation on spatial transcriptomics data."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Base directory containing data.")
    parser.add_argument("--gt_filename", type=str, required=True, help="CSV file containing ground truth labels.")
    parser.add_argument("--spot_dict_filename", type=str, required=True, help="JSON file with spot dictionary.")
    parser.add_argument("--embeddings_filename", type=str, required=True, help="PT file with embeddings.")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of repetitions")
    parser.add_argument("--output_csv", type=str, default="results.csv", help="Path to output CSV")

    args = parser.parse_args()
    main(
        args.data_path,
        args.gt_filename,
        args.spot_dict_filename,
        args.embeddings_filename,
        args.n_iter,
        args.output_csv,
    )
