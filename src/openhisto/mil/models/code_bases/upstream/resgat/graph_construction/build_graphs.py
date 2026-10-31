"""
Hybrid spatial-feature kNN graph construction for WSI patch embeddings.

Algorithm
---------
For each node *i*:

1. Find the ``d_neighbors`` nearest spatial neighbours (Euclidean on patch
   coordinates).
2. Find the ``f_neighbors`` nearest feature neighbours (cosine similarity on
   patch embeddings).
3. Take the **intersection** of these two sets, ordered by feature similarity.
4. Keep the top ``final_k`` edges.  If the intersection is empty, fall back to
   the top-3 feature neighbours.

The resulting directed edges are symmetrised, coalesced, and augmented with
self-loops before saving as a PyG ``Data`` object.
"""

import argparse
import os
from typing import Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce, add_self_loops


# ------------------------------------------------------------------
# Core edge-building logic
# ------------------------------------------------------------------

def _top_k_intersection(primary: np.ndarray,
                        secondary: np.ndarray) -> np.ndarray:
    """Ordered intersection of two neighbour arrays (primary ordering)."""
    secondary_set = set(secondary.tolist())
    return np.array([int(x) for x in primary if int(x) in secondary_set],
                    dtype=np.int64)


def build_edges_knn(coords: np.ndarray,
                    feats: np.ndarray,
                    f_neighbors: int = 50,
                    d_neighbors: int = 15,
                    final_k: int = 6,
                    no_intersection_mode: str = "feature",
                    coords_algo: str = "auto",
                    feats_algo: str = "brute") -> np.ndarray:
    """Build edge list via hybrid spatial-feature kNN intersection.

    Returns:
        edges: ``(E, 2)`` int64 array of directed ``(src, dst)`` pairs.
    """
    n = coords.shape[0]
    if n <= 1 or feats.shape[0] <= 1:
        raise ValueError("Need at least 2 nodes to build edges.")

    coord_nn = NearestNeighbors(n_neighbors=d_neighbors + 1,
                                metric="euclidean", algorithm=coords_algo)
    coord_nn.fit(coords)
    _, coord_inds = coord_nn.kneighbors(coords)
    coord_inds = coord_inds[:, 1:]

    feat_nn = NearestNeighbors(n_neighbors=f_neighbors + 1,
                               metric="cosine", algorithm=feats_algo)
    feat_nn.fit(feats)
    _, feat_inds = feat_nn.kneighbors(feats)
    feat_inds = feat_inds[:, 1:]

    edges = []
    for i in range(n):
        common = _top_k_intersection(feat_inds[i], coord_inds[i])
        if common.size == 0:
            if no_intersection_mode == "feature":
                k_nearest = list(feat_inds[i][:min(3, feat_inds.shape[1])])
            else:  # spatial_random
                cands = coord_inds[i][:min(8, coord_inds.shape[1])]
                k_nearest = np.random.choice(
                    cands, size=min(3, len(cands)), replace=False
                ).astype(int).tolist()
        else:
            k_nearest = list(common[:final_k])
            if len(k_nearest) < 3:
                k_nearest.extend(
                    feat_inds[i][:min(3 - len(k_nearest), feat_inds.shape[1])]
                )
        for j in k_nearest:
            edges.append([i, int(j)])
    return np.array(edges, dtype=np.int64)


# ------------------------------------------------------------------
# Save graph
# ------------------------------------------------------------------

def save_graph(x: torch.Tensor, edges_np: np.ndarray, label: int,
               out_path: str) -> None:
    """Save a PyG ``Data`` graph with undirected edges and self-loops."""
    if edges_np.size == 0:
        raise ValueError("No edges to build graph.")
    edge_index = torch.from_numpy(edges_np).long().t().contiguous()
    edge_index = coalesce(edge_index, num_nodes=x.size(0))
    edge_index = to_undirected(edge_index, num_nodes=x.size(0))
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    data = Data(x=x, edge_index=edge_index,
                y=torch.tensor(int(label), dtype=torch.long))
    torch.save(data, out_path)


# ------------------------------------------------------------------
# Coordinate helpers
# ------------------------------------------------------------------

def load_coords_from_h5(h5_path: str) -> Optional[np.ndarray]:
    """Load ``(N, 2)`` patch coordinates from an h5 file."""
    if not h5_path or not os.path.exists(h5_path):
        return None
    try:
        with h5py.File(h5_path, "r") as f:
            if "coords" in f:
                return np.asarray(f["coords"][:], dtype=np.float32)
    except Exception as e:
        print(f"Warning: could not load coords from {h5_path}: {e}")
    return None


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build hybrid spatial-feature kNN graphs from patch "
                    "embeddings (.pt) and coordinates (.h5).")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing per-slide .pt feature files.")
    parser.add_argument("--h5_dir", type=str, default=None,
                        help="Directory containing per-slide .h5 coordinate "
                             "files. Defaults to <input_dir>/../h5_files.")
    parser.add_argument("--label_csv", type=str, default=None,
                        help="CSV with columns [filename, label] mapping "
                             ".pt stems to integer labels.")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "graphs"),
                        help="Output directory for graph .pt files "
                             "(default: graph_construction/graphs/).")
    parser.add_argument("--f_neighbors", type=int, default=50)
    parser.add_argument("--d_neighbors", type=int, default=15)
    parser.add_argument("--final_k", type=int, default=6)
    parser.add_argument("--no_intersection_mode", type=str, default="feature",
                        choices=["feature", "spatial_random"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.h5_dir is None:
        args.h5_dir = os.path.join(os.path.dirname(args.input_dir), "h5_files")

    label_map = {}
    if args.label_csv and os.path.isfile(args.label_csv):
        df = pd.read_csv(args.label_csv)
        for _, row in df.iterrows():
            label_map[str(row.iloc[0])] = int(row.iloc[1])

    pt_files = sorted(f for f in os.listdir(args.input_dir)
                      if f.endswith(".pt"))
    print(f"Found {len(pt_files)} .pt files. Building graphs "
          f"(f={args.f_neighbors}, d={args.d_neighbors}, k={args.final_k}).")

    for idx, pt_name in enumerate(pt_files):
        stem = os.path.splitext(pt_name)[0]
        label = label_map.get(stem, 0)
        pt_path = os.path.join(args.input_dir, pt_name)
        h5_path = os.path.join(args.h5_dir, stem + ".h5")

        x = torch.load(pt_path, map_location="cpu")
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.float()

        coords = load_coords_from_h5(h5_path)
        if coords is None:
            print(f"[{idx + 1}/{len(pt_files)}] Skipping {stem}: "
                  "no coordinates found.")
            continue

        edges_np = build_edges_knn(
            coords, x.numpy(),
            f_neighbors=args.f_neighbors,
            d_neighbors=args.d_neighbors,
            final_k=args.final_k,
            no_intersection_mode=args.no_intersection_mode,
        )
        out_path = os.path.join(args.output_dir,
                                f"Weighted_Graph_{stem}-{label}.pt")
        save_graph(x, edges_np, label, out_path)

        avg_deg = edges_np.shape[0] / x.size(0) if x.size(0) > 0 else 0
        print(f"[{idx + 1}/{len(pt_files)}] {stem} | label={label} | "
              f"nodes={x.size(0)} | avg_deg={avg_deg:.2f}")


if __name__ == "__main__":
    main()
