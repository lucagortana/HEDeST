from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import threading
import time

import numpy as np
import pandas as pd
import torch
from celltype_permutation import hierarchical_permutations
from joblib import delayed
from joblib import Parallel
from semisim_utils import aggregate_stats
from semisim_utils import compute_stats
from semisim_utils import prepare_data


def run_single_iteration(ground_truth, spot_dict, embeddings):
    A_batches, B_batches, X_perm_batches, X_perm, X = prepare_data(ground_truth, spot_dict, embeddings)

    X_global_batches = []

    for batch_idx, (A_batch, X_perm_batch, B_batch) in enumerate(zip(A_batches, X_perm_batches, B_batches)):
        print(f"Processing batch {batch_idx + 1}/{len(A_batches)}")
        result = [None]
        start_time = time.time()

        def run_permutation():
            result[0] = hierarchical_permutations(A_batch, X_perm_batch, B_batch)

        thread = threading.Thread(target=run_permutation)
        thread.start()

        while thread.is_alive():
            thread.join(timeout=5)
            elapsed = time.time() - start_time
            if elapsed > 300:
                print(f"Skipping batch {batch_idx} (timeout >5min)")
                break

        if result[0] is not None:
            X_global_batches.append(result[0])
        else:
            print(f"-> Padding skipped batch {batch_idx} with zeros")
            X_global_batches.append(np.zeros_like(X_perm_batch))

    X_global = np.concatenate(X_global_batches)
    return X_global, X_perm, X


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

    np.save(os.path.join("X_globals_all.npy"), np.array(X_globals_all, dtype=object))
    np.save(os.path.join("X_perms_all.npy"), np.array(X_perms_all, dtype=object))
    np.save(os.path.join("Xs_all.npy"), np.array(Xs_all, dtype=object))

    print("Global results saved.")

    X_globals_all = [np.array(x) for x in X_globals_all]
    X_perms_all = [np.array(x) for x in X_perms_all]
    Xs_all = [np.array(x) for x in Xs_all]

    X_globals_stack = np.stack(X_globals_all)  # shape: (n_iter, n_cells)
    valid_mask = ~np.any(X_globals_stack == 0, axis=0)

    X_globals_all = [Xg[valid_mask] for Xg in X_globals_all]
    X_perms_all = [Xp[valid_mask] for Xp in X_perms_all]
    Xs_all = [X[valid_mask] for X in Xs_all]

    np.save(os.path.join("X_globals_filtered.npy"), np.array(X_globals_all))
    np.save(os.path.join("X_perms_filtered.npy"), np.array(X_perms_all))
    np.save(os.path.join("Xs_filtered.npy"), np.array(Xs_all))

    print("Filtered results saved.")

    remaining_indices = list(np.where(valid_mask)[0])
    remaining_cell_ids = [cell_ids[i] for i in remaining_indices]
    pd.Series(remaining_cell_ids).to_csv("remaining_cell_ids.csv", index=False, header=False)
    print("Remaining cell IDs saved to 'remaining_cell_ids.csv'.")

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
