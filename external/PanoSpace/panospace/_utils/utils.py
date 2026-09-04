"""
PanoSpace Common Utilities
==========================

"""
from __future__ import annotations

from typing import Literal
from typing import Optional
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import KDTree


def radius_membership_sparse(
    base_points: np.ndarray,
    query_points: np.ndarray,
    r: Union[float, np.ndarray],
    metric: Literal["euclidean", "chebyshev"] = "euclidean",
    leaf_size: int = 40,
    chunk_size: Optional[int] = None,
    dtype=np.int8,
    sort_results: bool = False,
) -> csr_matrix:
    """Construct a sparse membership matrix based on radius-neighborhood queries.

    The function builds a :class:`scipy.sparse.csr_matrix` whose rows correspond to
    ``query_points`` and whose columns correspond to ``base_points``. An entry is
    set to one when the distance between the associated points is within the
    specified radius, and zero otherwise.

    Parameters
    ----------
    base_points : numpy.ndarray
        Array with shape ``(n_base, n_features)`` that stores the reference points
        used to build the KD-tree.
    query_points : numpy.ndarray
        Array with shape ``(n_query, n_features)`` containing the points whose
        neighbors are queried.
    r : float or numpy.ndarray
        Distance threshold(s) used to determine neighborhood membership. A scalar
        applies the same radius to every query point, while a one-dimensional array
        of shape ``(n_query,)`` allows specifying a custom radius per query.
    metric : {"euclidean", "chebyshev"}, default="euclidean"
        Distance metric used by :class:`sklearn.neighbors.KDTree` when performing
        the queries.
    leaf_size : int, default=40
        Leaf size passed to :class:`sklearn.neighbors.KDTree`, controlling the
        trade-off between query speed and tree construction cost.
    chunk_size : int or None, default=None
        When provided, queries are processed in batches of ``chunk_size`` rows to
        limit memory usage. ``None`` processes all query points at once.
    dtype : data-type, default=numpy.int8
        Data type assigned to the non-zero entries of the resulting sparse matrix.
    sort_results : bool, default=False
        Whether the neighbor indices in each row should be sorted. Passed through
        to :meth:`sklearn.neighbors.KDTree.query_radius`.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse membership matrix of shape ``(n_query, n_base)`` where non-zero
        entries indicate that a ``base_point`` lies within the radius of a
        ``query_point``.

    Raises
    ------
    ValueError
        If an unsupported metric is requested or if ``r`` is an array with an
        incompatible shape.

    Examples
    --------
    >>> import numpy as np
    >>> from panospace._utils.utils import radius_membership_sparse
    >>> base = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    >>> query = np.array([[0.1, 0.1], [1.8, 1.9]])
    >>> membership = radius_membership_sparse(base, query, r=0.5)
    >>> membership.toarray()
    array([[1, 0, 0],
           [0, 0, 1]], dtype=int8)

    The radius may also be provided per query point and combined with chunked
    queries when memory is limited:

    >>> radii = np.array([0.5, 0.2])
    >>> radius_membership_sparse(base, query, r=radii, chunk_size=1).toarray()
    array([[1, 0, 0],
           [0, 0, 0]], dtype=int8)
    """
    if metric not in ("euclidean", "chebyshev"):
        raise ValueError("metric must be 'euclidean' or 'chebyshev'")

    base_points = np.ascontiguousarray(base_points, dtype=np.float64)
    query_points = np.ascontiguousarray(query_points, dtype=np.float64)

    n_query = query_points.shape[0]
    n_base = base_points.shape[0]

    tree = KDTree(base_points, leaf_size=leaf_size, metric=metric)

    # Normalize r to array format (for chunking)
    if np.isscalar(r):
        r_arr = None
        r_scalar = float(r)
    else:
        r = np.asarray(r, dtype=np.float64)
        if r.shape != (n_query,):
            raise ValueError("when r is an array, its shape must be (n_query,)")
        r_arr = r
        r_scalar = None

    if chunk_size is None:
        counts = tree.query_radius(query_points, r=r_scalar if r_arr is None else r_arr, count_only=True)
        total_nnz = int(np.sum(counts))
        indptr = np.empty(n_query + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        indices = np.empty(total_nnz, dtype=np.int32)
        data = np.ones(total_nnz, dtype=dtype)

        neighbor_lists = tree.query_radius(
            query_points,
            r=r_scalar if r_arr is None else r_arr,
            return_distance=False,
            sort_results=sort_results,
        )
        pos = 0
        for i in range(n_query):
            inds = neighbor_lists[i].astype(np.int32, copy=False)
            end = pos + inds.size
            indices[pos:end] = inds
            pos = end
    else:
        indptr = np.zeros(n_query + 1, dtype=np.int64)
        starts = list(range(0, n_query, chunk_size))
        for start in starts:
            end = min(start + chunk_size, n_query)
            qp = query_points[start:end]
            if r_arr is None:
                counts_blk = tree.query_radius(qp, r=r_scalar, count_only=True)
            else:
                counts_blk = tree.query_radius(qp, r=r_arr[start:end], count_only=True)
            indptr[start + 1 : end + 1] = counts_blk

        np.cumsum(indptr, out=indptr)
        total_nnz = int(indptr[-1])
        indices = np.empty(total_nnz, dtype=np.int32)
        data = np.ones(total_nnz, dtype=dtype)

        for start in starts:
            end = min(start + chunk_size, n_query)
            qp = query_points[start:end]
            if r_arr is None:
                lists_blk = tree.query_radius(qp, r=r_scalar, return_distance=False, sort_results=sort_results)
            else:
                lists_blk = tree.query_radius(qp, r=r_arr[start:end], return_distance=False, sort_results=sort_results)
            write_pos = indptr[start]
            for inds in lists_blk:
                inds = inds.astype(np.int32, copy=False)
                next_pos = write_pos + inds.size
                indices[write_pos:next_pos] = inds
                write_pos = next_pos

    M = csr_matrix((data, indices, indptr), shape=(n_query, n_base), dtype=dtype)
    return M
