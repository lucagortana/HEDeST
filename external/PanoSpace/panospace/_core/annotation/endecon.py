"""panospace._core.annotation.endecon
===================================
EnDecon-based back-end for **single-cell cell-type annotation**.

References
----------
.. [1] Tu, H. *et al.* "**EnDecon**: ensemble deconvolution of spatial
       transcriptomics." *Nature Biotechnology* (2025).

Overview
~~~~~~~~
The back-end follows the original EnDecon paper but adapts the final step from
*spot* to *cell* level by redistributing the consensus cell-type probabilities
according to the spatial positions of detected nuclei.

Steps
-----
1. **Spot-level ensemble deconvolution**:  Run multiple deconvolution methods
   (RCTD, DWLS, Cell2location, …) on the spot counts contained in
   ``sdata.tables['spots']``; aggregate their outputs with EnDecon to obtain a
   spot x cell-type probability matrix ``P``.
2. **Probability propagation**:  For each spot, assign its probabilities to
   the nuclei whose centroids fall inside the spot circle (up-sample).  If a
   nucleus overlaps multiple spots (rare for Visium), normalise by overlap
   fraction.
3. **Optional smoothing**:  Apply a cell-type-specific spatial kernel to
   regularise local fluctuations (disabled by default).

The heavy lifting for step 1 is delegated to the external *EnDecon* Python
package.  We import it lazily; if it is not available, a simple fallback that
assigns the global reference cell-type frequencies is used so that example
pipelines still run.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("panospace._core.annotation.endecon")

# -----------------------------------------------------------------------------
# Optional dependency check ----------------------------------------------------
# -----------------------------------------------------------------------------
# _ENDECON_AVAILABLE = importlib.util.find_spec("endecon") is not None
# if _ENDECON_AVAILABLE:
#     from endecon import EnDecon  # type: ignore
# else:
#     logger.debug("EnDecon not available – falling back to dummy annotator.")

# -----------------------------------------------------------------------------
# Core function ----------------------------------------------------------------
# -----------------------------------------------------------------------------
import logging
import numpy as np

logger = logging.getLogger(__name__)


def l1_norm(X, Y):
    """Compute L1 norm between two matrices"""
    return np.sum(np.abs(X - Y))


def weighted_median(values, weights):
    """Weighted median similar to spatstat.geom::weighted.median"""
    # Sort by values
    sorted_indices = np.argsort(values)
    values_sorted = values[sorted_indices]
    weights_sorted = weights[sorted_indices]
    # Cumulative weight
    cumulative_weight = np.cumsum(weights_sorted) / np.sum(weights_sorted)
    # Find first index where cumulative weight >= 0.5
    median_idx = np.searchsorted(cumulative_weight, 0.5)
    return values_sorted[median_idx]


def endecon_core(results_deconv, lambda_=None, prob_quantile=0.5, niter=100, epsilon=1e-5, verbose=True):
    """
    Ensemble deconvolution results using weighted median method.

    Parameters
    ----------
    results_deconv : list of np.ndarray
        List of matrices (spots x cell-types) from individual methods.
    lambda_ : float or None
        Regularization parameter. If None, auto-determined.
    prob_quantile : float
        Quantile for lambda estimation.
    niter : int
        Max number of iterations.
    epsilon : float
        Convergence threshold.
    verbose : bool
        Whether to print iteration logs.

    Returns
    -------
    dict with keys:
        - 'H_norm': ensemble result (spots x cell-types) normalized by row sum
        - 'w': weights assigned to each method
    """
    num_methods = len(results_deconv)
    num_spots, num_cell_types = results_deconv[0].shape

    # Initialize weights equally
    w = np.ones(num_methods) / num_methods

    # Initialize H as weighted average
    # print([res.shape for res in results_deconv])
    H = np.sum([res * w[i] for i, res in enumerate(results_deconv)], axis=0)

    k = 1
    loss_all_temp = 0.0

    while k <= niter:
        # Compute L1 distances
        l1_distances = np.array([l1_norm(res, H) for res in results_deconv])

        # Initialize lambda if not provided
        if k == 1 and lambda_ is None:
            lambda_ = np.quantile(l1_distances, prob_quantile)
            if verbose:
                logger.info(f"Auto-selected lambda = {lambda_}")

        # Update weights
        exp_term = np.exp(-l1_distances / lambda_)
        w = exp_term / np.sum(exp_term)

        # Update H using weighted median for each element
        stacked = np.stack(results_deconv, axis=2)  # shape: (spots, cell_types, methods)
        H = np.apply_along_axis(lambda x: weighted_median(x, w), 2, stacked)

        # Compute losses
        loss_main = np.sum(l1_distances * w)
        loss_entropy = np.sum(w * np.log(w + 1e-12))
        loss_all = loss_main + lambda_ * loss_entropy

        if verbose:
            logger.info(
                f"iter: {k}, loss_main: {loss_main:.4f}, loss_entropy: {loss_entropy:.4f}, "
                f"loss_all: {loss_all:.4f}, lambda: {lambda_:.4f}"
            )

        # Check convergence
        if abs(loss_all - loss_all_temp) < epsilon:
            break

        loss_all_temp = loss_all
        k += 1

    # Normalize rows
    row_sums = H.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    H_norm = H / row_sums

    return {"H_norm": H_norm, "w": w}
