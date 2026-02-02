#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landmark neighborhood fingerprints from left/right cortical VTK meshes.

Inputs (same as your MIND scripts):
  --lh-vtk, --rh-vtk: VTK/VTP surface with PointData scalar arrays
  --lh-landmarks, --rh-landmarks: txt files of 0-based vertex ids, order preserved
  --feature: one or more PointData array names
  --ring OR --radius-mm: define neighborhood
  --include-center: include seed vertex or not
  --min-verts: drop landmarks with too few valid vertices
  --no-drop-zero: keep vertices with any zero feature
  --no-zscore: disable z-scoring across whole cortex (lh+rh)

Output (ONLY ONE FILE):
  --out-fp: a TSV (or txt) with fingerprints.
    Default format: one row per landmark node.
    You can also output one-row-per-subject via --aggregate subject.

Fingerprints per landmark include (for each feature):
  mean, std, min, max, median, iqr, mad, q05,q10,q25,q75,q90,q95, skew, kurtosis
Plus:
  n_verts, hemi (lh/rh), landmark_index
  cross-feature: cov_trace, cov_logdet (if F>1), feature_energy (mean squared norm)

Example:
python landmark_fingerprint_from_vtk.py \
  --lh-vtk subj_lh.vtk --rh-vtk subj_rh.vtk \
  --lh-landmarks lh_3hinge_ids.txt --rh-landmarks rh_3hinge_ids.txt \
  --feature curv sulc thickness \
  --ring 5 --min-verts 10 \
  --out-fp subj_landmark_fp.tsv

Author: you :)
"""

import argparse
import os
import heapq
import numpy as np
import pandas as pd
import vtk
from vtk.util import numpy_support as vtknp
from scipy.stats import skew, kurtosis


# --------------------------- VTK I/O ---------------------------

def _read_polydata(path: str) -> vtk.vtkPolyData:
    ext = os.path.splitext(path)[1].lower()
    reader = vtk.vtkXMLPolyDataReader() if ext == ".vtp" else vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read VTK polydata or empty: {path}")
    return poly

def _get_point_array(poly: vtk.vtkPolyData, key: str) -> np.ndarray:
    arr = poly.GetPointData().GetArray(key)
    if arr is None and poly.GetPointData().GetScalars() is not None:
        if poly.GetPointData().GetScalars().GetName() == key:
            arr = poly.GetPointData().GetScalars()
    if arr is None:
        raise KeyError(f"PointData array '{key}' not found")
    np_arr = vtknp.vtk_to_numpy(arr).reshape(-1)
    if np_arr.ndim != 1:
        raise ValueError(f"Array '{key}' is not 1D per-vertex scalar.")
    return np_arr


# --------------------------- Landmark reading ---------------------------

def read_landmark_txt(path: str) -> list[int]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(int(s))
    return out


# --------------------------- Mesh neighborhood ---------------------------

def build_adjacency(poly: vtk.vtkPolyData) -> list[list[int]]:
    n = poly.GetNumberOfPoints()
    adj = [set() for _ in range(n)]

    polys = poly.GetPolys()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()

    while polys.GetNextCell(id_list):
        ids = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]

        for i in range(len(ids)):
            a = ids[i]
            b = ids[(i + 1) % len(ids)]
            if a != b:
                adj[a].add(b)
                adj[b].add(a)

        if len(ids) > 3:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    if a != b:
                        adj[a].add(b)
                        adj[b].add(a)

    return [sorted(list(s)) for s in adj]

def k_ring_neighbors(adj: list[list[int]], seed: int, k: int, include_center: bool = True) -> np.ndarray:
    if k < 0:
        raise ValueError("k (ring) must be >= 0")
    visited = set([seed])
    frontier = [seed]
    for _ in range(k):
        nxt = []
        for v in frontier:
            for nb in adj[v]:
                if nb not in visited:
                    visited.add(nb)
                    nxt.append(nb)
        frontier = nxt
        if not frontier:
            break
    if not include_center and seed in visited:
        visited.remove(seed)
    return np.array(sorted(list(visited)), dtype=np.int64)

def geodesic_radius_neighbors_mm(adj: list[list[int]],
                                points_xyz: np.ndarray,
                                seed: int,
                                radius_mm: float,
                                include_center: bool = True) -> np.ndarray:
    if radius_mm <= 0:
        raise ValueError("radius_mm must be > 0")

    dist = {seed: 0.0}
    pq = [(0.0, seed)]
    visited = set()

    while pq:
        d, v = heapq.heappop(pq)
        if v in visited:
            continue
        visited.add(v)
        if d > radius_mm:
            break
        for nb in adj[v]:
            w = float(np.linalg.norm(points_xyz[v] - points_xyz[nb]))
            nd = d + w
            if nd <= radius_mm and (nb not in dist or nd < dist[nb]):
                dist[nb] = nd
                heapq.heappush(pq, (nd, nb))

    if not include_center and seed in visited:
        visited.remove(seed)
    return np.array(sorted(list(visited)), dtype=np.int64)


# --------------------------- Feature loading & cleaning ---------------------------

def build_vertex_features(lh_path: str,
                          rh_path: str,
                          feature_keys: list[str],
                          drop_zero: bool = True,
                          drop_nonfinite: bool = True,
                          zscore: bool = True):
    lh_poly = _read_polydata(lh_path)
    rh_poly = _read_polydata(rh_path)

    lh_feats = np.stack([_get_point_array(lh_poly, k) for k in feature_keys], axis=1)
    rh_feats = np.stack([_get_point_array(rh_poly, k) for k in feature_keys], axis=1)

    lh_mask = np.ones(lh_feats.shape[0], dtype=bool)
    rh_mask = np.ones(rh_feats.shape[0], dtype=bool)

    if drop_nonfinite:
        lh_mask &= np.isfinite(lh_feats).all(axis=1)
        rh_mask &= np.isfinite(rh_feats).all(axis=1)

    if drop_zero:
        lh_mask &= ~(lh_feats == 0).any(axis=1)
        rh_mask &= ~(rh_feats == 0).any(axis=1)

    if zscore:
        all_feats = np.vstack([lh_feats[lh_mask], rh_feats[rh_mask]])
        mu = all_feats.mean(axis=0)
        sd = all_feats.std(axis=0, ddof=0)
        sd = np.where(sd > 0, sd, 1.0)
        lh_feats = (lh_feats - mu) / sd
        rh_feats = (rh_feats - mu) / sd

    return lh_poly, rh_poly, lh_feats, rh_feats, lh_mask, rh_mask


# --------------------------- Fingerprint stats ---------------------------

def _nan_safe_quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))

def _mad(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def feature_stats_1d(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
            "median": np.nan, "iqr": np.nan, "mad": np.nan,
            "q05": np.nan, "q10": np.nan, "q25": np.nan, "q75": np.nan, "q90": np.nan, "q95": np.nan,
            "skew": np.nan, "kurt": np.nan,
        }

    q25 = _nan_safe_quantile(x, 0.25)
    q75 = _nan_safe_quantile(x, 0.75)
    iqr = q75 - q25

    # bias=False is usually better for small samples; if too small, scipy returns nan sometimes
    sk = float(skew(x, bias=False)) if x.size >= 3 else np.nan
    ku = float(kurtosis(x, bias=False)) if x.size >= 4 else np.nan

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "iqr": float(iqr),
        "mad": _mad(x),
        "q05": _nan_safe_quantile(x, 0.05),
        "q10": _nan_safe_quantile(x, 0.10),
        "q25": float(q25),
        "q75": float(q75),
        "q90": _nan_safe_quantile(x, 0.90),
        "q95": _nan_safe_quantile(x, 0.95),
        "skew": sk,
        "kurt": ku,
    }

def cross_feature_stats(X: np.ndarray) -> dict:
    """
    X: (n, F) samples for one landmark.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    n, F = X.shape if X.ndim == 2 else (0, 0)
    if n == 0 or F == 0:
        return {"cov_trace": np.nan, "cov_logdet": np.nan, "energy_mean_sqnorm": np.nan}

    # mean squared L2 norm (energy)
    energy = float(np.mean(np.sum(X * X, axis=1)))

    if F == 1 or n < 2:
        return {"cov_trace": np.nan, "cov_logdet": np.nan, "energy_mean_sqnorm": energy}

    C = np.cov(X, rowvar=False, bias=True)  # (F,F)
    tr = float(np.trace(C))

    # logdet may fail if singular; add small ridge
    ridge = 1e-6 * np.eye(F)
    sign, logdet = np.linalg.slogdet(C + ridge)
    cov_logdet = float(logdet) if sign > 0 else np.nan

    return {"cov_trace": tr, "cov_logdet": cov_logdet, "energy_mean_sqnorm": energy}


# --------------------------- Main extraction ---------------------------

def compute_fingerprints(
    lh_vtk: str,
    rh_vtk: str,
    lh_landmarks_txt: str,
    rh_landmarks_txt: str,
    feature_keys: list[str],
    mode: str,
    ring: int | None,
    radius_mm: float | None,
    include_center: bool,
    drop_zero: bool,
    zscore: bool,
    min_verts: int,
    subject_id: str,
    aggregate: str,  # "landmark" or "subject"
) -> pd.DataFrame:

    lh_poly, rh_poly, lh_feats, rh_feats, lh_valid, rh_valid = build_vertex_features(
        lh_vtk, rh_vtk, feature_keys, drop_zero=drop_zero, drop_nonfinite=True, zscore=zscore
    )
    lh_seeds = read_landmark_txt(lh_landmarks_txt)
    rh_seeds = read_landmark_txt(rh_landmarks_txt)

    lh_adj = build_adjacency(lh_poly)
    rh_adj = build_adjacency(rh_poly)
    lh_xyz = vtknp.vtk_to_numpy(lh_poly.GetPoints().GetData()).astype(np.float64)
    rh_xyz = vtknp.vtk_to_numpy(rh_poly.GetPoints().GetData()).astype(np.float64)

    rows = []

    def _one_hemi(hemi: str, seeds, adj, xyz, feats, valid_mask):
        nV = feats.shape[0]
        for li, seed_vid in enumerate(seeds):
            if seed_vid < 0 or seed_vid >= nV:
                raise ValueError(f"[{hemi}] landmark line {li+1}: vertex id {seed_vid} out of range 0..{nV-1}")

            if mode == "ring":
                verts = k_ring_neighbors(adj, seed_vid, k=ring, include_center=include_center)
            else:
                verts = geodesic_radius_neighbors_mm(adj, xyz, seed_vid, radius_mm=radius_mm, include_center=include_center)

            verts = verts[valid_mask[verts]]
            node = f"{hemi}_{li:04d}"

            if verts.size < min_verts:
                # keep a row with NaNs so your downstream knows it exists, or skip
                # here: keep row with NaNs but mark dropped
                base = {
                    "subject_id": subject_id,
                    "node": node,
                    "hemi": hemi,
                    "landmark_index": int(li),
                    "n_verts": int(verts.size),
                    "dropped": 1,
                }
                # fill per-feature NaNs
                for fk in feature_keys:
                    for stat in ["mean","std","min","max","median","iqr","mad","q05","q10","q25","q75","q90","q95","skew","kurt"]:
                        base[f"{fk}__{stat}"] = np.nan
                for k in ["cov_trace","cov_logdet","energy_mean_sqnorm"]:
                    base[k] = np.nan
                rows.append(base)
                continue

            X = feats[verts]  # (n,F)
            base = {
                "subject_id": subject_id,
                "node": node,
                "hemi": hemi,
                "landmark_index": int(li),
                "n_verts": int(verts.size),
                "dropped": 0,
            }

            # per-feature stats
            for fi, fk in enumerate(feature_keys):
                st = feature_stats_1d(X[:, fi])
                for k, v in st.items():
                    base[f"{fk}__{k}"] = v

            # cross-feature stats
            cst = cross_feature_stats(X)
            base.update(cst)

            rows.append(base)

    _one_hemi("lh", lh_seeds, lh_adj, lh_xyz, lh_feats, lh_valid)
    _one_hemi("rh", rh_seeds, rh_adj, rh_xyz, rh_feats, rh_valid)

    df = pd.DataFrame(rows)

    if aggregate == "landmark":
        return df

    # aggregate to subject-level: take (nan)mean over landmarks for each numeric column (excluding identifiers)
    num_cols = [c for c in df.columns if c not in ["subject_id","node","hemi","landmark_index"]]
    # keep dropped rate + avg n_verts too
    out = {"subject_id": subject_id}
    for c in num_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        out[c] = float(np.nanmean(x.to_numpy(dtype=float))) if np.isfinite(x.to_numpy(dtype=float)).any() else np.nan
    return pd.DataFrame([out])


# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract landmark neighborhood fingerprint stats from VTK meshes.")
    p.add_argument("--lh-vtk", required=True)
    p.add_argument("--rh-vtk", required=True)
    p.add_argument("--lh-landmarks", required=True)
    p.add_argument("--rh-landmarks", required=True)
    p.add_argument("--feature", required=True, nargs="+")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ring", type=int, default=None)
    g.add_argument("--radius-mm", type=float, default=None)
    p.add_argument("--include-center", action="store_true")
    p.add_argument("--min-verts", type=int, default=2)
    p.add_argument("--no-drop-zero", action="store_true")
    p.add_argument("--no-zscore", action="store_true")
    p.add_argument("--subject-id", default="subject", help="Write into output column subject_id.")
    p.add_argument("--aggregate", choices=["landmark","subject"], default="landmark",
                   help="Output one row per landmark or aggregate to one row per subject.")
    p.add_argument("--out-fp", required=True, help="Output fingerprint TSV path.")
    return p.parse_args()

def main():
    args = parse_args()
    mode = "ring" if args.ring is not None else "mm"
    if mode == "ring" and args.ring < 0:
        raise ValueError("--ring must be >= 0")

    df = compute_fingerprints(
        lh_vtk=args.lh_vtk,
        rh_vtk=args.rh_vtk,
        lh_landmarks_txt=args.lh_landmarks,
        rh_landmarks_txt=args.rh_landmarks,
        feature_keys=args.feature,
        mode=mode,
        ring=args.ring,
        radius_mm=args.radius_mm,
        include_center=bool(args.include_center),
        drop_zero=not args.no_drop_zero,
        zscore=not args.no_zscore,
        min_verts=int(args.min_verts),
        subject_id=str(args.subject_id),
        aggregate=str(args.aggregate),
    )

    out_path = os.path.abspath(args.out_fp)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Saved fingerprint -> {out_path}")
    print(f"[OK] Rows={len(df)} Cols={df.shape[1]} (aggregate={args.aggregate})")

if __name__ == "__main__":
    main()
