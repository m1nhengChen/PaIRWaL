# train.py
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from dataset import GraphItem, sample_walk_eventseqs, pad_event_batch_with_features


def class_weights_from_y(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float)


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    items: List[GraphItem],
    train_idx: np.ndarray,
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
) -> float:
    model.train()

    token_lists, node_feats, node_masks, attn_masks, labels, _ = sample_walk_eventseqs(
        items=items,
        indices=train_idx,
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

    perm = rng.permutation(len(labels))
    token_lists = [token_lists[i] for i in perm]
    node_feats = [node_feats[i] for i in perm]
    node_masks = [node_masks[i] for i in perm]
    attn_masks = [attn_masks[i] for i in perm]
    labels = [labels[i] for i in perm]

    total_loss = 0.0
    total_n = 0

    for i in range(0, len(labels), batch_size):
        yb = torch.tensor(labels[i:i + batch_size], dtype=torch.long, device=device)
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

        optimizer.zero_grad(set_to_none=True)
        logits = model(tok, nf, nm, am)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * yb.size(0)
        total_n += yb.size(0)

    return total_loss / max(1, total_n)
