from __future__ import annotations

import time
from itertools import product

import numpy as np
from more_itertools import distinct_permutations
from sklearn.metrics import calinski_harabasz_score

# from tqdm import tqdm


def local_permutations(A, X_perm, B):
    """
    Rearrange spots according to their best permutation per bag

    Args:
        A: one-hot encoded matrix (N cells x M spots) indicating the belonging of each cell to a spot
        B: matrix (N cells x K features) indicating morphological features per cell
        X_perm: one-hot encoded matrix (N cells x öL types) indicating the initial assigned cell type per cell

    Returns:
        X_local: rearranged cell type per cell according to local optimization
        multiple_option_permutations: permutations for spots that cannot be optimized locally
        multiple_option_indices: indices for permutations for spots that cannot be optimized locally
    """

    X_sparse_perm = X_perm * A

    best_permutations = []
    indices_order = []

    print("--------------------------------------------------------------")
    print("M_spot\t Perm_1\t \t Perm_2\t CHI")
    print("--------------------------------------------------------------")
    for Xm_i, Xm in enumerate(X_sparse_perm):
        values = Xm[Xm > 0]
        indices = np.where(Xm > 0)[0]
        perms = list(distinct_permutations(values))
        highest_score = float("-inf")
        best_permutation = []
        if len(perms) == 1:
            best_permutation = np.array(perms)
        else:
            for perm in perms:
                B_subset = B[indices]
                X_subset = np.array(perm)
                if len(np.unique(X_subset)) == len(X_subset):
                    best_permutation = np.array(perms)
                    break
                else:
                    score = calinski_harabasz_score(B_subset, X_subset)
                    if score == highest_score:
                        best_permutation.append(perm)
                    # Update highest score and corresponding permutation if a higher score is found
                    elif score > highest_score:
                        highest_score = score
                        best_permutation = [perm]
        best_permutations.append(np.stack(best_permutation))
        indices_order.append(indices)
        print(Xm_i, "\t", len(perms), "\t" "\t", np.stack(best_permutation).shape[0], "\t", highest_score)
    print("--------------------------------------------------------------")

    X_local = X_perm.copy()
    multiple_option_permutations = []
    multiple_option_indices = []

    for permutations, indices in zip(best_permutations, indices_order):
        if len(permutations) == 1:
            permutation = permutations[0]
            X_local[indices] = permutation
        if len(permutations) > 1:
            multiple_option_permutations.append(permutations)
            multiple_option_indices.append(indices)

    return X_local, multiple_option_permutations, multiple_option_indices


def global_permutations(B, X_local, multiple_option_permutations, multiple_option_indices):
    """
    Rearrange spots according to their best global permutation

    Args:
        X_local, multiple_option_permutations, multiple_option_indices: results from local permutations
        B: matrix (N cells x K features) indicating morphological features per cell

    Returns:
        X_global: rearranged cell type per cell according to global optimization
    """

    # total_tests = np.prod(np.array([len(perm) for perm in multiple_option_permutations]))
    X_global = X_local.copy()
    max_score = float("-inf")
    best_permutation = None
    new_inds = np.hstack(multiple_option_indices)
    # Iterate through the Cartesian product of multiple_option_permutations
    for new_perms in product(*multiple_option_permutations):
        new_perms = np.hstack(new_perms)
        # Apply the new permutation to X_next
        X_global[new_inds] = new_perms
        # Calculate Calinski-Harabasz score for the updated X_next
        score = calinski_harabasz_score(B, X_global)
        # Update minimum score and corresponding permutation if a lower score is found
        if score > max_score:
            max_score = score
            best_permutation = new_perms
    X_global[new_inds] = best_permutation
    print("Maximum Score:", max_score)
    return X_global


def hierarchical_permutations(A, X_perm, B):
    t = time.time()
    X_local, multiple_option_permutations, multiple_option_indices = local_permutations(A, X_perm, B)
    X_global = global_permutations(B, X_local, multiple_option_permutations, multiple_option_indices)
    elapsed = time.time() - t
    print("Total time:", elapsed)
    return X_global
