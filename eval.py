from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import GraphItem, sample_walk_eventseqs, pad_event_batch_with_features


@torch.no_grad()
def eval_graph_level(
    model: nn.Module,
    items: List[GraphItem],
    eval_idx: np.ndarray,
    rng: np.random.RandomState,
    device: torch.device,
    walk_len: int,
    n_walks_per_graph: int,
    batch_size: int,
    restart_alpha: Optional[float],
    restart_period_k: Optional[int],
    non_backtracking: bool,
    use_named_neighbors: bool,
    record_attr: str,
    attr_on_hash: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, str, int, int]]]:
    """
    Graph-level evaluation:
    - sample K walks per graph
    - reader -> per-walk probs
    - average probs per graph
    """
    model.eval()

    token_lists, node_feats, node_masks, attn_masks, labels, gids = sample_walk_eventseqs(
        items=items,
        indices=eval_idx,
        n_walks_per_graph=n_walks_per_graph,
        walk_len=walk_len,
        rng=rng,
        restart_alpha=restart_alpha,
        restart_period_k=restart_period_k,
        non_backtracking=non_backtracking,
        use_named_neighbors=use_named_neighbors,
        record_attr=record_attr,
        attr_on_hash=attr_on_hash,
    )

    all_probs = []
    all_gids = []

    for i in range(0, len(labels), batch_size):
        tok, nf, nm, am = pad_event_batch_with_features(
            token_lists[i:i + batch_size],
            node_feats[i:i + batch_size],
            node_masks[i:i + batch_size],
            attn_masks[i:i + batch_size],
        )
        tok = tok.to(device)
        nf = nf.to(device)
        nm = nm.to(device)
        am = am.to(device)

        logits = model(tok, nf, nm, am)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_gids.extend(gids[i:i + batch_size])

    all_probs = np.concatenate(all_probs, axis=0)
    all_gids = np.array(all_gids, dtype=np.int64)

    graph_probs = []
    graph_true = []
    meta = []
    for gi in eval_idx.tolist():
        m = (all_gids == int(gi))
        p = all_probs[m].mean(axis=0)
        graph_probs.append(p)
        graph_true.append(items[int(gi)].y)
        meta.append((
            items[int(gi)].subject_id,
            items[int(gi)].site,
            items[int(gi)].raw_label,
            len(items[int(gi)].neighbors),
        ))

    graph_probs = np.stack(graph_probs, axis=0)
    y_true = np.array(graph_true, dtype=np.int64)
    y_pred = np.argmax(graph_probs, axis=1)
    return y_true, y_pred, graph_probs, meta
