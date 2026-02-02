from __future__ import annotations

from pathlib import Path
import numpy as np


def load_roi_onehot(roi_path: Path, n_nodes: int, roi_dim: int = 150) -> np.ndarray:
    A = np.loadtxt(roi_path, dtype=np.float32)
    if A.ndim == 1:
        A = A[None, :]
    if A.shape[1] != roi_dim:
        raise ValueError(f"{roi_path.name}: roi dim mismatch {A.shape[1]} != {roi_dim}")

    if A.shape[0] != n_nodes:
        if A.shape[0] > n_nodes:
            print(f"[WARN] {roi_path.name}: roi rows={A.shape[0]} > N={n_nodes} -> truncate")
            A = A[:n_nodes, :]
        else:
            print(f"[WARN] {roi_path.name}: roi rows={A.shape[0]} < N={n_nodes} -> pad zeros")
            pad = np.zeros((n_nodes - A.shape[0], roi_dim), dtype=np.float32)
            A = np.vstack([A, pad])
    return A


def roi_onehot_to_id(roi_onehot: np.ndarray) -> np.ndarray:
    sums = roi_onehot.sum(axis=1)
    roi_id = np.argmax(roi_onehot, axis=1).astype(np.int64)
    roi_id[sums <= 0] = -1
    return roi_id
