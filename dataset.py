# dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from graph import sparsify_weighted, weighted_to_unweighted_neighbors
from walk import compute_transition_probs_mdlr, sample_random_walk
from record import EventSeq, record_walk_events
from fingerprint import load_fingerprint_tsv_aligned
from roi import load_roi_onehot, roi_onehot_to_id


@dataclass
class GraphItem:
    subject_id: str
    site: str
    raw_label: int
    y: int
    neighbors: List[np.ndarray]
    probs: List[np.ndarray]
    X: np.ndarray                 # (N,F) fingerprint node features
    roi_id: Optional[np.ndarray]  # (N,) or None


def build_graph_items(
    entries_f: List[dict],
    y_encoded: np.ndarray,
    sparsify: str,
    density: float,
    keep_mst: bool,
    fp_dir: Path,
    fp_suffix: str,
    feature_cols: List[str],
    fp_sort_by_idx: bool,
    fp_zscore: bool,
    roi_dir: Optional[Path],
    roi_suffix: str,
    roi_dim: int,
    record_attr: str,
) -> List[GraphItem]:
    out: List[GraphItem] = []

    for e, y in zip(entries_f, y_encoded):
        sid = str(e.get("subject_id", ""))
        sim = np.asarray(e["matrix"], dtype=np.float64)
        n = sim.shape[0]

        W = sparsify_weighted(sim, method=sparsify, density=density, keep_mst=keep_mst)
        neighbors = weighted_to_unweighted_neighbors(W, eps=0.0)
        probs = compute_transition_probs_mdlr(neighbors)

        fp_path = fp_dir / f"{sid}{fp_suffix}"
        if not fp_path.exists():
            raise FileNotFoundError(f"Fingerprint not found: {fp_path}")
        X = load_fingerprint_tsv_aligned(
            fp_path=fp_path,
            n_nodes=n,
            feature_cols=feature_cols,
            sort_by_node_idx=fp_sort_by_idx,
            zscore_per_graph=fp_zscore,
        )

        roi_id = None
        if record_attr == "roi":
            if roi_dir is None:
                raise ValueError("--record-attr roi requires --roi-dir")
            roi_path = roi_dir / f"{sid}{roi_suffix}"
            if not roi_path.exists():
                raise FileNotFoundError(f"ROI onehot not found: {roi_path}")
            roi_onehot = load_roi_onehot(roi_path, n_nodes=n, roi_dim=roi_dim)
            roi_id = roi_onehot_to_id(roi_onehot)

        out.append(GraphItem(
            subject_id=sid,
            site=str(e.get("site", "")),
            raw_label=int(e.get("label", -1)),
            y=int(y),
            neighbors=neighbors,
            probs=probs,
            X=X,
            roi_id=roi_id,
        ))

    return out


def build_eventseq_with_features(
    ev: EventSeq,
    X: np.ndarray,
) -> Tuple[List[int], np.ndarray, List[float], List[float]]:
    """
    Convert an EventSeq carrying node_index to:
      token_ids: List[int]
      node_feat: np.ndarray (L,F) (0 rows for non-node positions)
      node_mask: List[float]
      attn_mask: List[float]
    """
    L = len(ev.token_ids)
    Fd = X.shape[1]
    nf = np.zeros((L, Fd), dtype=np.float32)
    nm = np.zeros((L,), dtype=np.float32)
    for t, v in enumerate(ev.node_index):
        if v >= 0:
            nf[t] = X[int(v)]
            nm[t] = 1.0
    return ev.token_ids, nf, nm.tolist(), ev.attn_mask


def pad_event_batch_with_features(
    token_lists: List[List[int]],
    node_feats: List[np.ndarray],
    node_masks: List[List[float]],
    attn_masks: List[List[float]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = len(token_lists)
    L = max(len(x) for x in token_lists)
    Fd = int(node_feats[0].shape[1])

    tok = torch.full((B, L), 0, dtype=torch.long)
    nf = torch.zeros((B, L, Fd), dtype=torch.float32)
    nm = torch.zeros((B, L), dtype=torch.float32)
    am = torch.zeros((B, L), dtype=torch.float32)

    for i in range(B):
        li = len(token_lists[i])
        tok[i, :li] = torch.tensor(token_lists[i], dtype=torch.long)
        nf[i, :li] = torch.tensor(node_feats[i], dtype=torch.float32)
        nm[i, :li] = torch.tensor(node_masks[i], dtype=torch.float32)
        am[i, :li] = torch.tensor(attn_masks[i], dtype=torch.float32)

    return tok, nf, nm, am


def sample_walk_eventseqs(
    items: List[GraphItem],
    indices: np.ndarray,
    n_walks_per_graph: int,
    walk_len: int,
    rng: np.random.RandomState,
    restart_alpha: Optional[float],
    restart_period_k: Optional[int],
    non_backtracking: bool,
    use_named_neighbors: bool,
    record_attr: str,
    attr_on_hash: bool,
):
    """
    Returns:
      token_lists: list of token_id lists
      node_feats:  list of (L,F) arrays
      node_masks:  list of node_mask lists
      attn_masks:  list of attn_mask lists
      labels:      y labels per walk
      gids:        graph indices per walk
    """
    token_lists: List[List[int]] = []
    node_feats: List[np.ndarray] = []
    node_masks: List[List[float]] = []
    attn_masks: List[List[float]] = []
    labels: List[int] = []
    gids: List[int] = []

    for gi in indices.tolist():
        item = items[int(gi)]
        n = len(item.neighbors)
        for _ in range(n_walks_per_graph):
            v0 = int(rng.randint(0, n)) if n > 0 else 0
            wr = sample_random_walk(
                neighbors=item.neighbors,
                probs=item.probs,
                l=walk_len,
                rng=rng,
                start_v=v0,
                restart_alpha=restart_alpha,
                restart_period_k=restart_period_k,
                non_backtracking=non_backtracking,
            )
            ev = record_walk_events(
                walk_res=wr,
                neighbors=item.neighbors,
                roi_id=item.roi_id,
                use_named_neighbors=use_named_neighbors,
                record_attr=record_attr,
                attr_on_hash=attr_on_hash,
            )
            tl, nf, nm, am = build_eventseq_with_features(ev, item.X)
            token_lists.append(tl)
            node_feats.append(nf)
            node_masks.append(nm)
            attn_masks.append(am)
            labels.append(int(item.y))
            gids.append(int(gi))

    return token_lists, node_feats, node_masks, attn_masks, labels, gids
