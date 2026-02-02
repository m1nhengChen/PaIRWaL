# record.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from walk import WalkResult

# Token IDs (small fixed vocab for structure tokens)
PAD = 0
BOS = 1
EOS = 2
DASH = 3   # "-"
SEMI = 4   # ";"
HASH = 5   # "#"
ATTR = 6   # "@"

# ROI attribute token base (large to separate from small tokens)
ATTR_BASE = 4096


def roi_token(roi_id: int) -> int:
    return ATTR_BASE + int(roi_id)


@dataclass
class EventSeq:
    token_ids: List[int]            # token id at each position (0 if not a token position)
    node_index: List[int]           # node index at each position (-1 if not a node position)
    attn_mask: List[int]            # 1 valid, 0 padding (before padding it's all 1)


def record_walk_events(
    walk_res: WalkResult,
    neighbors: List[np.ndarray],
    roi_id: Optional[np.ndarray],
    use_named_neighbors: bool,
    record_attr: str,               # "none" or "roi"
    attr_on_hash: bool,             # if True, also attach ROI attr after HASH-node events
) -> EventSeq:
    """
    Produces an event sequence:
      - token positions: BOS, DASH, SEMI, HASH, roi_token, EOS
      - node positions: carry node_index (we will fetch its fingerprint vector)
    The "named neighbors" part follows the anonymized ordering via discovery ids (invariant).
    """
    walk = walk_res.walk
    restart = walk_res.restart

    v0 = int(walk[0])
    S = {v0}
    id_map = {v0: 1}
    next_id = 2
    T = set()  # directed edges for Algorithm-3 style named neighbor recording

    token_ids: List[int] = []
    node_index: List[int] = []
    attn_mask: List[int] = []

    def push_token(tid: int):
        token_ids.append(int(tid))
        node_index.append(-1)
        attn_mask.append(1)

    def push_node(v: int):
        token_ids.append(0)
        node_index.append(int(v))
        attn_mask.append(1)

    def push_attr_for_node(v: int):
        if record_attr != "roi":
            return
        if roi_id is None:
            return
        rid = int(roi_id[v])
        if rid < 0:
            return
        push_token(roi_token(rid))

    # BOS
    push_token(BOS)
    # node v0
    push_node(v0)
    push_attr_for_node(v0)

    for t in range(1, walk.size):
        vt = int(walk[t])
        rt = int(restart[t - 1])

        if vt not in S:
            S.add(vt)
            id_map[vt] = next_id
            next_id += 1

        if rt == 0:
            push_token(DASH)
            push_node(vt)
            push_attr_for_node(vt)

            if use_named_neighbors:
                vprev = int(walk[t - 1])
                T.add((vprev, vt))
                T.add((vt, vprev))

                nbrs = neighbors[vt]
                if nbrs.size > 0:
                    U = [int(u) for u in nbrs if int(u) in S]
                    if len(U) > 0:
                        U.sort(key=lambda u: id_map[u])
                        for u in U:
                            if (vt, u) not in T:
                                push_token(HASH)
                                push_node(u)
                                if attr_on_hash:
                                    push_attr_for_node(u)
                                T.add((vt, u))
                                T.add((u, vt))
        else:
            push_token(SEMI)
            push_node(vt)
            push_attr_for_node(vt)

    push_token(EOS)
    return EventSeq(token_ids=token_ids, node_index=node_index, attn_mask=attn_mask)
