from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class WalkResult:
    walk: np.ndarray        # (l+1,)
    restart: np.ndarray     # (l,) flags


def compute_degrees(neighbors: List[np.ndarray]) -> np.ndarray:
    return np.array([len(nbrs) for nbrs in neighbors], dtype=np.int64)


def compute_transition_probs_mdlr(neighbors: List[np.ndarray]) -> List[np.ndarray]:
    """
    Transition p(u->v) ∝ c(u,v), MDLR conductance:
    c(u,v)=1/min(deg(u),deg(v))
    """
    deg = compute_degrees(neighbors)
    probs = []
    for u, nbrs in enumerate(neighbors):
        if nbrs.size == 0:
            probs.append(np.array([], dtype=np.float64))
            continue
        c = 1.0 / np.minimum(deg[u], deg[nbrs]).astype(np.float64)
        s = c.sum()
        if s <= 0:
            p = np.full_like(c, 1.0 / c.size, dtype=np.float64)
        else:
            p = c / s
        probs.append(p)
    return probs


def categorical_sample(items: np.ndarray, p: np.ndarray, rng: np.random.RandomState) -> int:
    if items.size == 1:
        return int(items[0])
    return int(items[rng.choice(items.size, p=p)])


def sample_random_walk(
    neighbors: List[np.ndarray],
    probs: List[np.ndarray],
    l: int,
    rng: np.random.RandomState,
    start_v: Optional[int] = None,
    restart_alpha: Optional[float] = None,
    restart_period_k: Optional[int] = None,
    non_backtracking: bool = True,
) -> WalkResult:
    n = len(neighbors)
    if n == 0:
        raise ValueError("Empty graph")

    v0 = int(rng.randint(0, n)) if start_v is None else int(start_v)

    walk = np.empty((l + 1,), dtype=np.int64)
    walk[0] = v0
    restart = np.zeros((l,), dtype=np.int64)

    nbrs0 = neighbors[v0]
    if nbrs0.size == 0:
        walk[1:] = v0
        return WalkResult(walk=walk, restart=restart)

    v1 = categorical_sample(nbrs0, probs[v0], rng)
    walk[1] = v1
    restart[0] = 0

    for t in range(2, l + 1):
        rt = 0
        if restart_alpha is not None:
            if restart[t - 2] == 0:
                rt = int(rng.rand() < restart_alpha)
        elif restart_period_k is not None:
            if (t % restart_period_k) == 0:
                rt = 1
        restart[t - 1] = rt

        if rt == 1:
            walk[t] = v0
            continue

        prev = int(walk[t - 1])
        prevprev = int(walk[t - 2])

        nbrs = neighbors[prev]
        if nbrs.size == 0:
            walk[t] = prev
            continue

        if non_backtracking:
            if nbrs.size == 1 and int(nbrs[0]) == prevprev:
                walk[t] = prevprev
                continue
            mask = (nbrs != prevprev)
            S = nbrs[mask]
            if S.size == 0:
                walk[t] = prevprev
                continue
            p_full = probs[prev]
            pS = p_full[mask]
            s = pS.sum()
            if s <= 0:
                pS = np.full_like(pS, 1.0 / pS.size, dtype=np.float64)
            else:
                pS = pS / s
            walk[t] = categorical_sample(S, pS, rng)
        else:
            walk[t] = categorical_sample(nbrs, probs[prev], rng)

    return WalkResult(walk=walk, restart=restart)
