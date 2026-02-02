from __future__ import annotations

from typing import List
import numpy as np


def filter_task(entries: List[dict], mode: str, classes: List[int], pos: int, neg: List[int]):
    keep = []
    y = []
    for e in entries:
        lab = int(e["label"])
        if mode == "multiclass":
            if lab in classes:
                keep.append(e)
                y.append(lab)
        else:
            if lab == pos:
                keep.append(e)
                y.append(1)
            elif lab in neg:
                keep.append(e)
                y.append(0)
    return keep, np.array(y, dtype=int)
