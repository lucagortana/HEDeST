from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import norm
from qpsolvers import solve_qp


def intersect(a, b):
    return list(set(a) & set(b))


def row_max_indices(matrix):
    """Return column index of maximum value for each row."""
    return np.argmax(matrix, axis=1)


def solve_OLS_internal(S, B):
    D = S.T @ S
    d = S.T @ B
    n = S.shape[1]

    # Convert to QP standard form
    # minimize 0.5 x^T D x - d^T x  s.t. x >= 0
    G = -np.eye(n)  # -x <= 0  → x >= 0
    h = np.zeros(n)

    x = solve_qp(D, -d, G, h, solver="osqp")  # or "cvxopt"
    return np.clip(x, 0, None)


def find_dampening_constant(S, B, gold_standard, n_iter=50):
    ws = (1 / (S @ gold_standard)) ** 2

    ws[np.isinf(ws)] = np.nan
    ws[np.isnan(ws)] = np.nanmax(ws[~np.isnan(ws)])
    ws = ws.astype(np.float64)
    ws = np.clip(ws, 0, 1e6)

    ws_scaled = ws / np.min(ws)
    ws_scaled[np.isinf(ws_scaled)] = np.nanmax(ws_scaled[~np.isinf(ws_scaled)])
    max_val = np.nanmax(ws_scaled)

    best_j = 1
    best_score = np.inf

    for j in range(1, int(np.ceil(np.log2(max_val))) + 1):
        multiplier = 2 ** (j - 1)
        ws_dampened = np.clip(ws_scaled, None, multiplier)
        # Cross-validation
        scores = []
        for _ in range(n_iter):
            subset = np.random.choice(len(ws), len(ws) // 2, replace=False)
            W = ws_dampened[subset]
            # print('S:', S, 'B:', B, 'W:', W)
            fit_coef, *_ = np.linalg.lstsq(S[subset, :], B[subset] * W, rcond=None)
            fit_coef[fit_coef < 0] = 0
            scores.append(np.std(fit_coef))
        mean_score = np.mean(scores)
        if mean_score < best_score:
            best_score = mean_score
            best_j = j
    return best_j


def solve_dampened_WLSj(S, B, gold_standard, j):
    multiplier = 2 ** (j - 1)
    ws = (1 / (S @ gold_standard)) ** 2
    ws_scaled = ws / np.min(ws)
    ws_dampened = np.clip(ws_scaled, None, multiplier)

    W = np.diag(ws_dampened)
    D = S.T @ W @ S
    d = S.T @ W @ B
    n = S.shape[1]

    G = -np.eye(n)
    h = np.zeros(n)

    # Normalize (similar to R's /sc)
    sc = np.linalg.norm(D, 2)
    x = solve_qp(D / sc, -d / sc, G, h, solver="osqp")
    return np.clip(x, 0, None)


def optimize_solveDampenedWLS(S, B, constant_J, tol=0.01, max_iter=1000):
    solution = solve_OLS_internal(S, B)
    j = constant_J
    change = 1
    iterations = 0

    while change > tol and iterations < max_iter:
        new_solution = solve_dampened_WLSj(S, B, solution, j)
        solution_avg = (4 * solution + new_solution) / 5
        change = norm(solution_avg - solution)
        solution = solution_avg
        iterations += 1

    return solution / np.sum(solution) if np.sum(solution) > 0 else solution


def optimize_deconvolute_dwls(exp_df, signature_df):
    genes = intersect(signature_df.index, exp_df.index)
    S = signature_df.loc[genes].values
    subBulk = exp_df.loc[genes].values

    all_exp = exp_df.mean(axis=1).loc[genes].values
    solution_all_exp = solve_OLS_internal(S, all_exp)
    constant_J = find_dampening_constant(S, all_exp, solution_all_exp)

    results = []
    for j in range(subBulk.shape[1]):
        B = subBulk[:, j]
        if np.sum(B) > 0:
            solDWLS = optimize_solveDampenedWLS(S, B, constant_J)
        else:
            solDWLS = np.zeros(S.shape[1])
        results.append(solDWLS)

    results = np.array(results).T
    return pd.DataFrame(results, index=signature_df.columns, columns=exp_df.columns)


def enrich_analysis(expr_df, sign_matrix_df):
    inter_genes = intersect(sign_matrix_df.index, expr_df.index)
    filter_sig = sign_matrix_df.loc[inter_genes]

    # log2(rowMeans(2^expr - 1) + 1)
    mean_gene_expr = np.log2(np.mean(np.expm1(expr_df.loc[inter_genes].values), axis=1) + 1)
    gene_fold = expr_df.loc[inter_genes].values - mean_gene_expr[:, None]

    cell_col_mean = gene_fold.mean(axis=0)
    cell_col_sd = gene_fold.std(axis=0)

    enrichment = []
    for i in range(filter_sig.shape[1]):
        signames = filter_sig.index[filter_sig.iloc[:, i] == 1]
        sig_idx = [inter_genes.index(g) for g in signames]
        sig_col_mean = gene_fold[sig_idx].mean(axis=0)
        m = len(signames)
        zscore = (sig_col_mean - cell_col_mean) * np.sqrt(m) / cell_col_sd
        enrichment.append(zscore)

    enrichment = np.array(enrichment)
    return pd.DataFrame(enrichment, index=filter_sig.columns, columns=expr_df.columns)


def enrich_deconvolution(expr_df, log_expr_df, cluster_info, ct_exp_df, cutoff=1.0):
    # Build 0/1 enrichment matrix
    enrich_matrix = pd.DataFrame(0, index=ct_exp_df.index, columns=ct_exp_df.columns)
    max_indices = row_max_indices(ct_exp_df.values)
    for i, idx in enumerate(max_indices):
        enrich_matrix.iat[i, idx] = 1

    # PAGE enrichment
    enrich_result = enrich_analysis(log_expr_df, enrich_matrix)

    # Initialize results
    dwls_results = pd.DataFrame(0, index=ct_exp_df.columns, columns=expr_df.columns)

    clusters = sorted(set(cluster_info))
    for cluster in clusters:
        cluster_cells = [i for i, c in enumerate(cluster_info) if c == cluster]
        cluster_enrich = enrich_result.iloc[:, cluster_cells]
        row_i_max = cluster_enrich.max(axis=1)

        ct = row_i_max[row_i_max > cutoff].index.tolist()
        if len(ct) < 2:  # Need at least 2
            top2 = row_i_max.sort_values(ascending=False).head(2).index.tolist()
            ct = top2

        ct_gene = []
        for ctype in ct:
            ct_gene += enrich_matrix.index[enrich_matrix[ctype] == 1].tolist()
        uniq_ct_gene = list(set(ct_gene) & set(expr_df.index))

        select_sig_exp = ct_exp_df.loc[uniq_ct_gene, ct]
        cluster_cell_exp = expr_df.loc[uniq_ct_gene, expr_df.columns[cluster_cells]]

        cluster_dwls = optimize_deconvolute_dwls(cluster_cell_exp, select_sig_exp)
        dwls_results.loc[ct, cluster_cell_exp.columns] = cluster_dwls

    dwls_results[dwls_results < 0] = 0
    return dwls_results.T


def runDWLSDeconv(expr_df, log_expr_df, cluster_info, ct_exp_df, cutoff=1.0):
    """
    Equivalent to Giotto::runDWLSDeconv

    Parameters
    ----------
    expr_df : pd.DataFrame
        Spatial expression matrix (genes x spots)
    log_expr_df : pd.DataFrame
        Log-transformed expression matrix (genes x spots)
    cluster_info : list or array
        Cluster/region information for each spot
    ct_exp_df : pd.DataFrame
        Cell type signature matrix (genes x cell types)
    cutoff : float
        Enrichment score threshold
    """
    return enrich_deconvolution(expr_df, log_expr_df, cluster_info, ct_exp_df, cutoff)
