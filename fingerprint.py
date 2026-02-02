from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd


DROP_COLS = {
    "subject_id", "SubjectID", "sid",
    "label", "raw_label",
    "index", "idx", "node_idx", "node_index",
    "hemi", "hemisphere"
}


def _feature_cols_from_df(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.select_dtypes(include=[np.number]).columns.tolist():
        if c in DROP_COLS:
            continue
        if c.lower() in DROP_COLS:
            continue
        cols.append(c)
    return cols


def scan_feature_columns(
    fp_dir: Path,
    fp_suffix: str,
    subject_ids: List[str],
    mode: str = "union",  # union / intersection / first
) -> List[str]:
    """
    Determine a global feature column list so every graph has same feature dimension.
    """
    mode = mode.lower()
    if mode not in {"union", "intersection", "first"}:
        raise ValueError("--fp-cols-mode must be one of: union, intersection, first")

    sets: List[Set[str]] = []
    first_cols: Optional[List[str]] = None

    for sid in subject_ids:
        fp_path = fp_dir / f"{sid}{fp_suffix}"
        if not fp_path.exists():
            raise FileNotFoundError(f"Fingerprint not found: {fp_path}")
        df = pd.read_csv(fp_path, sep="\t")
        cols = _feature_cols_from_df(df)
        if len(cols) == 0:
            raise RuntimeError(f"{fp_path.name}: no numeric feature columns after dropping meta cols.")
        if first_cols is None:
            first_cols = cols
        sets.append(set(cols))

    if first_cols is None:
        raise RuntimeError("No fingerprints found to determine feature columns.")

    if mode == "first":
        return first_cols

    if mode == "union":
        return sorted(set().union(*sets))

    inter = set.intersection(*sets) if sets else set()
    if len(inter) == 0:
        raise RuntimeError("Intersection of feature columns is empty. Try --fp-cols-mode union.")
    return sorted(inter)


def load_fingerprint_tsv_aligned(
    fp_path: Path,
    n_nodes: int,
    feature_cols: List[str],
    sort_by_node_idx: bool = False,
    zscore_per_graph: bool = False,
) -> np.ndarray:
    df = pd.read_csv(fp_path, sep="\t")

    if sort_by_node_idx:
        for key in ["node_idx", "node_index", "idx", "index"]:
            if key in df.columns:
                df = df.sort_values(by=key, kind="mergesort").reset_index(drop=True)
                break

    num_df = df.reindex(columns=feature_cols)
    num_df = num_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = num_df.to_numpy(dtype=np.float32)

    if X.shape[0] != n_nodes:
        if X.shape[0] > n_nodes:
            print(f"[WARN] {fp_path.name}: fp rows={X.shape[0]} > N={n_nodes} -> truncate")
            X = X[:n_nodes, :]
        else:
            print(f"[WARN] {fp_path.name}: fp rows={X.shape[0]} < N={n_nodes} -> pad zeros")
            pad = np.zeros((n_nodes - X.shape[0], X.shape[1]), dtype=np.float32)
            X = np.vstack([X, pad])

    if zscore_per_graph:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd

    return X
