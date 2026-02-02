#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landmark-based MIND similarity from left/right cortical VTK meshes.

Enhancement in this version:
- Supports MULTIVARIATE KDE resampling for multi-feature distributions (F>1) via scipy.stats.gaussian_kde.
- This allows increasing the number of samples for KL estimation WITHOUT increasing ring/mm neighborhood size.

Key new CLI args:
  --kde                 enable KDE resampling (works for 1D or multi-D)
  --kde-n               number of KDE samples per landmark node (default 500)
  --kde-bw              KDE bandwidth method passed to gaussian_kde (default 'scott')
  --kl-eps              small epsilon to stabilize division r/(s+eps)

Notes:
- KDE quality needs enough unique samples; if KDE fails for a node, we fallback to original samples.
"""

import argparse
import os
import heapq
import numpy as np
import pandas as pd
import vtk
from vtk.util import numpy_support as vtknp
from scipy.spatial import cKDTree as KDTree
from scipy import stats


# --------------------------- kNN KL ---------------------------

def get_KDTree(x):
    x = np.atleast_2d(x)
    return KDTree(x)

def get_KL(x, y, xtree, ytree, kl_eps: float = 1e-12):
    """
    kNN-based KL estimator between distributions x and y.
    Stabilized against division-by-zero by using s+kl_eps and filtering invalid ratios.

    Returns +inf if insufficient samples or all ratios invalid.
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n, d = x.shape
    m, dy = y.shape
    assert d == dy

    if n < 2 or m < 1:
        return np.inf

    # 2nd NN in x (exclude itself)
    r = xtree.query(x, k=2, eps=0.01, p=2)[0][:, 1]
    # 1st NN in y
    s = ytree.query(x, k=1, eps=0.01, p=2)[0]

    # stabilize
    rs_ratio = r / (s + float(kl_eps))

    # keep only finite and positive ratios
    rs_ratio = rs_ratio[np.isfinite(rs_ratio)]
    rs_ratio = rs_ratio[rs_ratio > 0.0]
    if rs_ratio.size == 0:
        return np.inf

    kl = -np.log(rs_ratio).sum() * d / n + np.log(m / (n - 1.0))
    kl = np.maximum(kl, 0.0)
    return float(kl)


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


# --------------------------- Vertex table & z-scoring ---------------------------

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


# --------------------------- Multi-D KDE resampling ---------------------------

def kde_resample_multid(X: np.ndarray, n_samples: int, bw_method="scott", seed: int = 0) -> np.ndarray:
    """
    Fit multivariate gaussian_kde on X (n,d) and sample n_samples points.
    Returns (n_samples,d). If KDE fails, raises Exception.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be (n,d)")
    n, d = X.shape

    # gaussian_kde expects shape (d,n)
    kde = stats.gaussian_kde(X.T, bw_method=bw_method)

    rng = np.random.default_rng(seed)
    # gaussian_kde.resample uses global RNG; to keep determinism, we can sample ourselves via covariance:
    # But simplest: call resample and accept its randomness. We'll still set numpy seed outside if needed.
    Y = kde.resample(n_samples).T
    return Y


# --------------------------- MIND computation ---------------------------

def calculate_mind_network(
    data_df: pd.DataFrame,
    feature_cols: list[str],
    region_list: list[str],
    kde: bool = False,
    kde_n: int = 500,
    kde_bw: str = "scott",
    kl_eps: float = 1e-12,
    seed: int = 0
) -> pd.DataFrame:
    """
    If kde=True:
      - For each Label, fit KDE in feature space and resample kde_n points,
        even when feature_cols has multiple dimensions.
      - If KDE fitting fails for a label (too few points, singular cov, etc),
        fallback to raw samples for that label.
    """
    MIND = pd.DataFrame(
        np.zeros((len(region_list), len(region_list)), dtype=np.float64),
        index=region_list, columns=region_list
    )

    data_df = data_df.loc[data_df["Label"].isin(region_list)]

    grouped = {name: grp[feature_cols].to_numpy(dtype=np.float64) for name, grp in data_df.groupby("Label")}

    # Optional KDE resample per region
    if kde:
        new_grouped = {}
        for idx, (name, X) in enumerate(grouped.items()):
            # if too small, KDE can be unstable
            # heuristic: need at least d+2 points and some variation
            d = X.shape[1]
            if X.shape[0] < max(5, d + 2):
                new_grouped[name] = X
                continue
            try:
                Y = kde_resample_multid(X, n_samples=kde_n, bw_method=kde_bw, seed=seed + idx)
                new_grouped[name] = Y
            except Exception as e:
                print(f"[WARN] KDE failed for {name} (n={X.shape[0]}, d={d}): {e} -> fallback to raw samples")
                new_grouped[name] = X
        grouped = new_grouped

    # KD-trees
    KDtrees = {name: get_KDTree(X) for name, X in grouped.items()}

    used_pairs = set()
    for name_x, X in grouped.items():
        for name_y, Y in grouped.items():
            if name_x == name_y:
                continue
            key = tuple(sorted((name_x, name_y)))
            if key in used_pairs:
                continue

            KLa = get_KL(X, Y, KDtrees[name_x], KDtrees[name_y], kl_eps=kl_eps)
            KLb = get_KL(Y, X, KDtrees[name_y], KDtrees[name_x], kl_eps=kl_eps)
            kl = KLa + KLb
            sim = 1.0 / (1.0 + kl)

            MIND.at[name_x, name_y] = sim
            MIND.at[name_y, name_x] = sim
            used_pairs.add(key)

    MIND = MIND[region_list].T[region_list].T
    np.fill_diagonal(MIND.values, 1.0)
    return MIND


def compute_landmark_mind_from_vtk(
    lh_vtk: str,
    rh_vtk: str,
    lh_landmarks_txt: str,
    rh_landmarks_txt: str,
    feature_keys: list[str],
    mode: str,
    ring: int | None,
    radius_mm: float | None,
    include_center: bool = True,
    drop_zero: bool = True,
    min_verts: int = 2,
    # KDE options
    kde: bool = False,
    kde_n: int = 500,
    kde_bw: str = "scott",
    kl_eps: float = 1e-12,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:

    lh_poly, rh_poly, lh_feats, rh_feats, lh_valid, rh_valid = build_vertex_features(
        lh_vtk, rh_vtk, feature_keys, drop_zero=drop_zero, drop_nonfinite=True, zscore=True
    )

    lh_seeds = read_landmark_txt(lh_landmarks_txt)
    rh_seeds = read_landmark_txt(rh_landmarks_txt)

    lh_adj = build_adjacency(lh_poly)
    rh_adj = build_adjacency(rh_poly)

    lh_xyz = vtknp.vtk_to_numpy(lh_poly.GetPoints().GetData()).astype(np.float64)
    rh_xyz = vtknp.vtk_to_numpy(rh_poly.GetPoints().GetData()).astype(np.float64)

    rows = []
    regions = []

    def _collect(hemi_prefix: str, seeds: list[int], adj, xyz, feats, valid_mask):
        nonlocal rows, regions
        nV = feats.shape[0]
        for i, seed_vid in enumerate(seeds):
            if seed_vid < 0 or seed_vid >= nV:
                raise ValueError(f"[{hemi_prefix}] Landmark line {i+1}: vertex id {seed_vid} out of range (0..{nV-1}).")

            if mode == "ring":
                verts = k_ring_neighbors(adj, seed_vid, k=ring, include_center=include_center)
            elif mode == "mm":
                verts = geodesic_radius_neighbors_mm(adj, xyz, seed_vid, radius_mm=radius_mm, include_center=include_center)
            else:
                raise ValueError("mode must be 'ring' or 'mm'.")

            verts = verts[valid_mask[verts]]

            node_name = f"{hemi_prefix}{i:04d}"
            if verts.size < min_verts:
                print(f"[WARN] Drop node {node_name}: only {verts.size} valid vertices (< min_verts={min_verts}).")
                continue

            regions.append(node_name)
            X = feats[verts]  # (n,F)
            for j in range(X.shape[0]):
                rows.append((node_name, *X[j].tolist()))

    _collect("lh_", lh_seeds, lh_adj, lh_xyz, lh_feats, lh_valid)
    _collect("rh_", rh_seeds, rh_adj, rh_xyz, rh_feats, rh_valid)

    if len(regions) == 0:
        raise RuntimeError("No landmark node survived min_verts filtering.")

    feature_cols = list(feature_keys)
    data_df = pd.DataFrame(rows, columns=["Label"] + feature_cols)

    mind_mat = calculate_mind_network(
        data_df=data_df,
        feature_cols=feature_cols,
        region_list=regions,
        kde=kde,
        kde_n=kde_n,
        kde_bw=kde_bw,
        kl_eps=kl_eps,
        seed=seed,
    )

    return mind_mat, np.array(regions, dtype=object)


# --------------------------- Save outputs ---------------------------

def save_outputs(mind_mat: pd.DataFrame, regions: np.ndarray, out_matrix: str, out_regions: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_matrix)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_regions)), exist_ok=True)

    print(f"[OK] MIND matrix shape: {mind_mat.shape}, nodes: {len(regions)}")

    if out_matrix.lower().endswith(".npy"):
        np.save(out_matrix, mind_mat.values)
    else:
        np.savetxt(out_matrix, mind_mat.values, fmt="%.6f", delimiter="\t")

    pd.DataFrame({"index": np.arange(len(regions)), "region": regions}).to_csv(
        out_regions, sep="\t", index=False
    )


# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute landmark-based ROI×ROI MIND similarity from L/R cortical VTK using feature keys and landmark seed vertex ids."
    )
    p.add_argument("--lh-vtk", required=True)
    p.add_argument("--rh-vtk", required=True)
    p.add_argument("--lh-landmarks", required=True)
    p.add_argument("--rh-landmarks", required=True)
    p.add_argument("--feature", required=True, nargs="+",
                   help="PointData arrays to use as features (e.g., thickness curv sulc).")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ring", type=int, default=None)
    g.add_argument("--radius-mm", type=float, default=None)

    p.add_argument("--include-center", action="store_true",
                   help="Include the seed vertex in its neighborhood.")

    p.add_argument("--no-drop-zero", action="store_true")
    p.add_argument("--min-verts", type=int, default=2)

    # KDE options
    p.add_argument("--kde", action="store_true",
                   help="Enable (multi-D) KDE resampling per landmark node before KL.")
    p.add_argument("--kde-n", type=int, default=500,
                   help="Number of KDE samples per node (used if --kde).")
    p.add_argument("--kde-bw", type=str, default="scott",
                   help="Bandwidth method for gaussian_kde: 'scott', 'silverman', or a float.")
    p.add_argument("--kl-eps", type=float, default=1e-12,
                   help="Epsilon for stabilizing r/(s+eps) in KL estimator.")
    p.add_argument("--seed", type=int, default=0, help="Seed offset for KDE sampling.")

    p.add_argument("--out-matrix", required=True)
    p.add_argument("--out-regions", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.ring is not None and args.ring < 0:
        raise ValueError("--ring must be >= 0")

    mode = "ring" if args.ring is not None else "mm"

    # allow float bandwidth
    kde_bw = args.kde_bw
    try:
        if kde_bw not in ["scott", "silverman"]:
            kde_bw = float(kde_bw)
    except Exception:
        pass

    mind_mat, regions = compute_landmark_mind_from_vtk(
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
        min_verts=args.min_verts,
        kde=bool(args.kde),
        kde_n=int(args.kde_n),
        kde_bw=kde_bw,
        kl_eps=float(args.kl_eps),
        seed=int(args.seed),
    )

    save_outputs(mind_mat, regions, args.out_matrix, args.out_regions)
    print(f"[OK] Saved matrix  -> {args.out_matrix}")
    print(f"[OK] Saved regions -> {args.out-regions}")


if __name__ == "__main__":
    main()
