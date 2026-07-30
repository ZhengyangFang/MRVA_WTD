from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_undirected

from .utils import ensure_dir, load_active_grid, load_or_build_aem_cache


def _regular_grid_candidates(active: pd.DataFrame, connectivity: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if connectivity not in {4, 8}:
        raise ValueError("regular_connectivity must be 4 or 8")
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    rc_to_index = {
        (int(row), int(col)): idx for idx, (row, col) in enumerate(active[["row", "col"]].to_numpy())
    }
    src, dst, dist = [], [], []
    for idx, (row, col, x, y) in enumerate(active[["row", "col", "x", "y"]].to_numpy()):
        for drow, dcol in offsets:
            nbr = rc_to_index.get((int(row + drow), int(col + dcol)))
            if nbr is None:
                continue
            src.append(idx)
            dst.append(nbr)
            if drow == 0 or dcol == 0:
                dist.append(1000.0)
            else:
                dist.append(float(np.sqrt(2.0) * 1000.0))
    return np.asarray(src), np.asarray(dst), np.asarray(dist, dtype=np.float32)


def _knn_candidates(active: pd.DataFrame, k_neighbors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = active[["x", "y"]].to_numpy(dtype=np.float32)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="auto", metric="euclidean")
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    src = np.repeat(np.arange(coords.shape[0], dtype=np.int64), k_neighbors)
    dst = indices[:, 1:].reshape(-1).astype(np.int64)
    dist = distances[:, 1:].reshape(-1).astype(np.float32)
    return src, dst, dist


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    max_value = float(weights.max()) if weights.size else 1.0
    if max_value <= 0.0:
        return np.ones_like(weights, dtype=np.float32)
    weights = weights / max_value
    weights = np.clip(weights, 1e-6, None)
    return weights.astype(np.float32)


def _base_candidate_edges(config: dict, active: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    graph_cfg = config["graph"]
    mode = str(graph_cfg.get("distance_mode", "knn")).lower()
    if mode == "regular":
        return _regular_grid_candidates(active, int(graph_cfg.get("regular_connectivity", 8)))
    return _knn_candidates(active, int(graph_cfg["k_neighbors"]))


def _graph_dir(config: dict) -> Path:
    return ensure_dir(Path(config["paths"]["output_dir"]) / "graphs")


def build_distance_graph(config: dict) -> Path:
    active = load_active_grid(config)
    src, dst, dist = _base_candidate_edges(config, active)

    length_scale = float(config["graph"]["length_scale_m"])
    raw_weight = np.exp(-dist / length_scale).astype(np.float32)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(raw_weight, dtype=torch.float32)

    if bool(config["graph"].get("make_undirected", True)):
        edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="mean")

    edge_weight = torch.tensor(_normalize_weights(edge_weight.cpu().numpy()), dtype=torch.float32)
    graph = {
        "graph_type": "distance",
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "grid_ids": torch.tensor(active["grid_id"].to_numpy(dtype=np.int64), dtype=torch.long),
        "x": torch.tensor(active["x"].to_numpy(dtype=np.float32)),
        "y": torch.tensor(active["y"].to_numpy(dtype=np.float32)),
        "row": torch.tensor(active["row"].to_numpy(dtype=np.int64), dtype=torch.long),
        "col": torch.tensor(active["col"].to_numpy(dtype=np.int64), dtype=torch.long),
        "length_scale_m": length_scale,
        "distance_mode": str(config["graph"].get("distance_mode", "knn")),
        "k_neighbors": int(config["graph"]["k_neighbors"]),
        "regular_connectivity": int(config["graph"].get("regular_connectivity", 8)),
        "grid_id_to_index": {
            int(grid_id): idx for idx, grid_id in enumerate(active["grid_id"].to_numpy(dtype=np.int64))
        },
    }
    out_path = _graph_dir(config) / "distance_graph.pt"
    torch.save(graph, out_path)
    return out_path


def build_aem_graph(config: dict) -> Path:
    active = load_active_grid(config)
    aem_cache = load_or_build_aem_cache(config)
    profiles = np.asarray(aem_cache["profiles"], dtype=np.float32)
    src, dst, dist = _base_candidate_edges(config, active)

    pair_diff = np.abs(profiles[src] - profiles[dst])
    rdist = np.mean(pair_diff, axis=1).astype(np.float32)

    length_scale = float(config["graph"]["length_scale_m"])
    rho_scale = float(config["graph"]["rho_scale"])
    raw_weight = np.exp(-dist / length_scale) * np.exp(-rdist / rho_scale)
    raw_weight = raw_weight.astype(np.float32)

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(raw_weight, dtype=torch.float32)
    edge_rdist = torch.tensor(rdist, dtype=torch.float32)
    edge_dist = torch.tensor(dist, dtype=torch.float32)

    if bool(config["graph"].get("make_undirected", True)):
        edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce="mean")
        _, edge_rdist = to_undirected(torch.tensor(np.vstack([src, dst]), dtype=torch.long), edge_rdist, reduce="mean")
        _, edge_dist = to_undirected(torch.tensor(np.vstack([src, dst]), dtype=torch.long), edge_dist, reduce="mean")

    edge_weight = torch.tensor(_normalize_weights(edge_weight.cpu().numpy()), dtype=torch.float32)
    graph = {
        "graph_type": "aem",
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "edge_distance_m": edge_dist,
        "edge_profile_distance": edge_rdist,
        "grid_ids": torch.tensor(active["grid_id"].to_numpy(dtype=np.int64), dtype=torch.long),
        "x": torch.tensor(active["x"].to_numpy(dtype=np.float32)),
        "y": torch.tensor(active["y"].to_numpy(dtype=np.float32)),
        "row": torch.tensor(active["row"].to_numpy(dtype=np.int64), dtype=torch.long),
        "col": torch.tensor(active["col"].to_numpy(dtype=np.int64), dtype=torch.long),
        "length_scale_m": length_scale,
        "rho_scale": rho_scale,
        "distance_mode": str(config["graph"].get("distance_mode", "knn")),
        "k_neighbors": int(config["graph"]["k_neighbors"]),
        "regular_connectivity": int(config["graph"].get("regular_connectivity", 8)),
        "grid_id_to_index": {
            int(grid_id): idx for idx, grid_id in enumerate(active["grid_id"].to_numpy(dtype=np.int64))
        },
    }
    out_path = _graph_dir(config) / "aem_graph.pt"
    torch.save(graph, out_path)
    return out_path


def build_all_graphs(config: dict) -> dict[str, Path]:
    return {
        "distance": build_distance_graph(config),
        "aem": build_aem_graph(config),
    }
