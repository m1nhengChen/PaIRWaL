from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def _upper_tri(n: int):
    return np.triu_indices(n, k=1)


def add_mst_edges(W: np.ndarray, sim: np.ndarray) -> np.ndarray:
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    mst_sparse = minimum_spanning_tree(dist)
    mst = mst_sparse.toarray()
    mst_und = (mst > 0) | (mst.T > 0)
    W[mst_und] = np.maximum(W[mst_und], sim[mst_und])
    return W


def sparsify_weighted(sim: np.ndarray, method: str, density: float, keep_mst: bool = True) -> np.ndarray:
    sim = np.asarray(sim, dtype=np.float64)
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"matrix must be square, got {sim.shape}")
    n = sim.shape[0]

    if method == "full":
        W = sim.copy()
        np.fill_diagonal(W, 0.0)
        return W

    if method != "density":
        raise ValueError(f"Unknown sparsify method: {method}")

    iu, ju = _upper_tri(n)
    w = sim[iu, ju]
    n_edges = w.size
    k = int(np.floor(density * n_edges))
    k = max(k, n - 1)
    k = min(k, n_edges)

    idx = np.argpartition(-w, kth=max(0, k - 1))[:k]
    W = np.zeros((n, n), dtype=np.float64)
    W[iu[idx], ju[idx]] = sim[iu[idx], ju[idx]]
    W[ju[idx], iu[idx]] = sim[iu[idx], ju[idx]]
    np.fill_diagonal(W, 0.0)

    if keep_mst:
        W = add_mst_edges(W, sim)
        np.fill_diagonal(W, 0.0)

    return W


def weighted_to_unweighted_neighbors(W: np.ndarray, eps: float = 0.0):
    A = (W > eps).astype(np.uint8)
    np.fill_diagonal(A, 0)
    return [np.flatnonzero(A[u]).astype(np.int64) for u in range(A.shape[0])]
