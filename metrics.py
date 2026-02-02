# metrics.py
from __future__ import annotations

from typing import List, Dict
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, matthews_corrcoef, confusion_matrix


def parse_list_of_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def macro_specificity(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    specs = []
    for c in range(n_classes):
        yt = (y_true == c).astype(int)
        yp = (y_pred == c).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        denom = tn + fp
        specs.append((tn / denom) if denom > 0 else np.nan)
    return float(np.nanmean(specs))


def compute_fold_metrics(y_true, y_pred, proba_or_score, mode: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["ACC"] = float(accuracy_score(y_true, y_pred))
    out["MCC"] = float(matthews_corrcoef(y_true, y_pred))

    if mode == "binary":
        out["SEN"] = float(recall_score(y_true, y_pred, pos_label=1))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out["SPE"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        out["AUROC"] = float("nan") if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, proba_or_score))
    else:
        C = int(proba_or_score.shape[1])
        out["SEN"] = float(recall_score(y_true, y_pred, average="macro"))
        out["SPE"] = float(macro_specificity(y_true, y_pred, n_classes=C))
        out["AUROC"] = float("nan") if len(np.unique(y_true)) < 2 else float(
            roc_auc_score(y_true, proba_or_score, multi_class="ovr", average="macro")
        )
    return out
