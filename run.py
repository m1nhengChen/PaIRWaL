#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Probability-invariant RWNN baseline (Graph Classification)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from metrics import parse_list_of_ints, compute_fold_metrics
from task import filter_task
from fingerprint import scan_feature_columns
from dataset import build_graph_items
from models_readers import build_reader, estimate_vocab_size
from train import train_one_epoch, class_weights_from_y
from eval import eval_graph_level


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-npy", required=True)
    ap.add_argument("--out-dir", required=True)

    # task
    ap.add_argument("--mode", choices=["multiclass", "binary"], default="multiclass")
    ap.add_argument("--classes", default="0,2,3")
    ap.add_argument("--pos", type=int, default=2)
    ap.add_argument("--neg", default="0")

    # graph build
    ap.add_argument("--sparsify", choices=["density", "full"], default="density")
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--keep-mst", action="store_true")

    # fingerprints
    ap.add_argument("--fp-dir", required=True, help="Directory containing {sid}_fp.tsv")
    ap.add_argument("--fp-suffix", default="_fp.tsv")
    ap.add_argument("--fp-sort-by-idx", action="store_true", help="If TSV has node_idx/index, sort by it.")
    ap.add_argument("--fp-zscore", action="store_true", help="Z-score features per-graph.")
    ap.add_argument("--fp-cols-mode", choices=["union", "intersection", "first"], default="union",
                    help="How to align feature columns across subjects.")

    # RWNN walk / record
    ap.add_argument("--walk-len", type=int, default=64)
    ap.add_argument("--non-backtracking", action="store_true")
    ap.add_argument("--restart-alpha", type=float, default=None)
    ap.add_argument("--restart-k", type=int, default=None)
    ap.add_argument("--named-neighbors", action="store_true")
    ap.add_argument("--n-walks-train", type=int, default=8)
    ap.add_argument("--n-walks-eval", type=int, default=64)

    ap.add_argument("--record-attr", choices=["none", "roi"], default="none")
    ap.add_argument("--attr-on-hash", action="store_true",
                    help="If record-attr=roi, also attach ROI tokens after HASH neighbor node events.")

    # ROI (optional)
    ap.add_argument("--roi-dir", default=None)
    ap.add_argument("--roi-suffix", default="_3hinge_0_hop_feature_lr_a2009s.txt")
    ap.add_argument("--roi-dim", type=int, default=150)

    # reader
    ap.add_argument("--reader", choices=["bow", "cnn", "gru", "transformer"], default="bow")
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--tr-layers", type=int, default=2)
    ap.add_argument("--tr-heads", type=int, default=4)

    # optimization
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=30)

    # cv
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_dir = Path(args.fp_dir).expanduser().resolve()
    if not fp_dir.exists():
        raise FileNotFoundError(f"--fp-dir not found: {fp_dir}")

    roi_dir = Path(args.roi_dir).expanduser().resolve() if args.roi_dir is not None else None
    if args.record_attr == "roi":
        if roi_dir is None or not roi_dir.exists():
            raise FileNotFoundError("--record-attr roi requires a valid --roi-dir")

    rng = np.random.RandomState(args.seed)

    arr = np.load(args.in_npy, allow_pickle=True)
    entries = [x.item() if isinstance(x, np.ndarray) else x for x in arr.tolist()]

    classes = parse_list_of_ints(args.classes)
    neg = parse_list_of_ints(args.neg)

    entries_f, y_raw = filter_task(entries, args.mode, classes=classes, pos=args.pos, neg=neg)
    if len(entries_f) == 0:
        raise RuntimeError("No subjects after filtering. Check --mode/--classes/--pos/--neg.")

    # Encode labels
    if args.mode == "multiclass":
        class_list = sorted(list(set(y_raw.tolist())))
        class_to_idx = {c: i for i, c in enumerate(class_list)}
        y = np.array([class_to_idx[int(v)] for v in y_raw], dtype=int)
        n_classes = len(class_list)
        used_raw = class_list
    else:
        y = y_raw.copy()
        n_classes = 2
        used_raw = {"pos": args.pos, "neg": neg}

    vals, cnts = np.unique(y, return_counts=True)
    if cnts.min() < args.kfold:
        raise RuntimeError(f"Not enough samples for kfold={args.kfold}. counts={dict(zip(vals.tolist(), cnts.tolist()))}")

    # Determine global fingerprint feature columns
    subject_ids = [str(e.get("subject_id", "")) for e in entries_f]
    feature_cols = scan_feature_columns(
        fp_dir=fp_dir,
        fp_suffix=args.fp_suffix,
        subject_ids=subject_ids,
        mode=args.fp_cols_mode,
    )
    feat_dim = len(feature_cols)
    print(f"[INFO] Fingerprint feature dim = {feat_dim} (cols_mode={args.fp_cols_mode})")

    # Build graph items
    items = build_graph_items(
        entries_f=entries_f,
        y_encoded=y,
        sparsify=args.sparsify,
        density=float(args.density),
        keep_mst=bool(args.keep_mst),
        fp_dir=fp_dir,
        fp_suffix=args.fp_suffix,
        feature_cols=feature_cols,
        fp_sort_by_idx=bool(args.fp_sort_by_idx),
        fp_zscore=bool(args.fp_zscore),
        roi_dir=roi_dir,
        roi_suffix=args.roi_suffix,
        roi_dim=int(args.roi_dim),
        record_attr=args.record_attr,
    )

    vocab_size = estimate_vocab_size(args.record_attr, int(args.roi_dim))
    device = torch.device(args.device)

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)

    pred_rows = []
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(y)), y), start=1):
        train_idx = np.array(train_idx, dtype=int)
        test_idx = np.array(test_idx, dtype=int)

        # Train/val split
        tr = train_idx.copy()
        rng.shuffle(tr)
        n_val = max(1, int(np.floor(0.15 * tr.size)))
        val_idx = tr[:n_val]
        tr_idx = tr[n_val:]

        model = build_reader(
            reader=args.reader,
            vocab_size=vocab_size,
            emb_dim=int(args.emb_dim),
            feat_dim=feat_dim,
            hidden=int(args.hidden),
            n_classes=n_classes,
            dropout=float(args.dropout),
            tr_layers=int(args.tr_layers),
            tr_heads=int(args.tr_heads),
        ).to(device)

        cw = class_weights_from_y(y[tr_idx], n_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

        best_val = float("inf")
        best_state = None
        bad = 0

        fold_rng = np.random.RandomState(args.seed + 1000 * fold)

        for epoch in range(1, int(args.epochs) + 1):
            _ = train_one_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                items=items,
                train_idx=tr_idx,
                rng=fold_rng,
                device=device,
                walk_len=int(args.walk_len),
                n_walks_per_graph=int(args.n_walks_train),
                batch_size=int(args.batch_size),
                restart_alpha=args.restart_alpha,
                restart_period_k=args.restart_k,
                non_backtracking=bool(args.non_backtracking),
                use_named_neighbors=bool(args.named_neighbors),
                record_attr=args.record_attr,
                attr_on_hash=bool(args.attr_on_hash),
            )

            # Val: graph-level aggregation
            yv_true, yv_pred, yv_prob, _ = eval_graph_level(
                model=model,
                items=items,
                eval_idx=val_idx,
                rng=fold_rng,
                device=device,
                walk_len=int(args.walk_len),
                n_walks_per_graph=int(args.n_walks_eval),
                batch_size=int(args.batch_size),
                restart_alpha=args.restart_alpha,
                restart_period_k=args.restart_k,
                non_backtracking=bool(args.non_backtracking),
                use_named_neighbors=bool(args.named_neighbors),
                record_attr=args.record_attr,
                attr_on_hash=bool(args.attr_on_hash),
            )
            vloss = float(-np.log(np.clip(yv_prob[np.arange(yv_true.size), yv_true], 1e-9, 1.0)).mean())

            if vloss < best_val - 1e-6:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= int(args.patience):
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Test
        yt_true, yt_pred, yt_prob, meta = eval_graph_level(
            model=model,
            items=items,
            eval_idx=test_idx,
            rng=np.random.RandomState(args.seed + 9999 + fold),
            device=device,
            walk_len=int(args.walk_len),
            n_walks_per_graph=int(args.n_walks_eval),
            batch_size=int(args.batch_size),
            restart_alpha=args.restart_alpha,
            restart_period_k=args.restart_k,
            non_backtracking=bool(args.non_backtracking),
            use_named_neighbors=bool(args.named_neighbors),
            record_attr=args.record_attr,
            attr_on_hash=bool(args.attr_on_hash),
        )

        if args.mode == "binary":
            metrics = compute_fold_metrics(yt_true, yt_pred, yt_prob[:, 1], mode="binary")
        else:
            metrics = compute_fold_metrics(yt_true, yt_pred, yt_prob, mode="multiclass")

        metrics["fold"] = fold
        fold_metrics.append(metrics)

        for j, gi in enumerate(test_idx.tolist()):
            sid, site, raw_lab, n_nodes = meta[j]
            row = {
                "fold": fold,
                "subject_id": sid,
                "site": site,
                "raw_label": int(raw_lab),
                "y_true": int(yt_true[j]),
                "y_pred": int(yt_pred[j]),
                "n_nodes": int(n_nodes),

                "model": "prob_invariant_rwnn_schemeA_flat",
                "reader": args.reader,
                "walk_len": int(args.walk_len),
                "named_neighbors": int(bool(args.named_neighbors)),
                "non_backtracking": int(bool(args.non_backtracking)),
                "restart_alpha": float(args.restart_alpha) if args.restart_alpha is not None else "None",
                "restart_k": int(args.restart_k) if args.restart_k is not None else "None",
                "n_walks_eval": int(args.n_walks_eval),
                "record_attr": args.record_attr,
                "attr_on_hash": int(bool(args.attr_on_hash)),

                "sparsify": args.sparsify,
                "density": float(args.density),
                "keep_mst": int(bool(args.keep_mst)),

                "fp_dir": str(fp_dir),
                "fp_suffix": args.fp_suffix,
                "fp_cols_mode": args.fp_cols_mode,
                "feat_dim": int(feat_dim),
            }
            for c in range(yt_prob.shape[1]):
                row[f"proba_c{c}"] = float(yt_prob[j, c])
            pred_rows.append(row)

        print(f"[FOLD {fold}] ACC={metrics['ACC']:.4f} SEN={metrics['SEN']:.4f} "
              f"SPE={metrics['SPE']:.4f} AUROC={metrics['AUROC']:.4f} MCC={metrics['MCC']:.4f}")

    mdf = pd.DataFrame(fold_metrics)
    summary = {k: {"mean": float(mdf[k].mean()), "std": float(mdf[k].std(ddof=1))}
               for k in ["ACC", "SEN", "SPE", "AUROC", "MCC"]}

    pred_path = out_dir / "predictions.tsv"
    pd.DataFrame(pred_rows).to_csv(pred_path, sep="\t", index=False)

    metrics_path = out_dir / "metrics.json"
    payload = {
        "args": vars(args),
        "n_subjects": int(len(y)),
        "mode": args.mode,
        "classes_used_raw": used_raw,
        "fold_metrics": fold_metrics,
        "summary": summary,
        "vocab_size": int(vocab_size),
        "feat_dim": int(feat_dim),
        "feature_cols": feature_cols,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n[SUMMARY] mean ± std")
    for k in ["ACC", "SEN", "SPE", "AUROC", "MCC"]:
        print(f"  {k}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")
    print(f"\n[OK] Saved predictions -> {pred_path}")
    print(f"[OK] Saved metrics     -> {metrics_path}")


if __name__ == "__main__":
    main()
