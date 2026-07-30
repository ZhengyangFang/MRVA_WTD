from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from .utils import (
    ensure_dir,
    load_or_build_aem_cache,
    load_or_build_dynamic_cache,
    load_topography_cache,
    pumping_log_dynamic_feature_name,
    pumping_raw_dynamic_feature_name,
    processed_paths,
    safe_standardize,
    time_unit_from_config,
)


def load_sample_table(config: dict) -> pd.DataFrame:
    path = processed_paths(config)["sample_table_csv"]
    frame = pd.read_csv(path)
    for col in ["start_month", "target_month", "start_time", "target_time"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col])
    target_col = target_time_column(config, frame.columns)
    horizon_col = horizon_column(config, frame.columns)
    return frame.sort_values(["split", target_col, "grid_id", horizon_col]).reset_index(drop=True)


def split_sample_table(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {split: sub.reset_index(drop=True) for split, sub in frame.groupby("split", sort=False)}


def horizon_column(config: dict, columns: list[str] | pd.Index | None = None) -> str:
    time_unit = time_unit_from_config(config)
    preferred = "h_steps" if time_unit in {"day", "week"} else "h_months"
    if columns is None:
        return preferred
    cols = set(columns)
    if preferred in cols:
        return preferred
    if "h_steps" in cols:
        return "h_steps"
    if "h_months" in cols:
        return "h_months"
    raise KeyError("Could not find horizon column in sample table.")


def start_time_column(config: dict, columns: list[str] | pd.Index | None = None) -> str:
    time_unit = time_unit_from_config(config)
    preferred = "start_time" if time_unit in {"day", "week"} else "start_month"
    if columns is None:
        return preferred
    cols = set(columns)
    if preferred in cols:
        return preferred
    if "start_time" in cols:
        return "start_time"
    if "start_month" in cols:
        return "start_month"
    raise KeyError("Could not find start time column in sample table.")


def target_time_column(config: dict, columns: list[str] | pd.Index | None = None) -> str:
    time_unit = time_unit_from_config(config)
    preferred = "target_time" if time_unit in {"day", "week"} else "target_month"
    if columns is None:
        return preferred
    cols = set(columns)
    if preferred in cols:
        return preferred
    if "target_time" in cols:
        return "target_time"
    if "target_month" in cols:
        return "target_month"
    raise KeyError("Could not find target time column in sample table.")


def start_time_idx_column(config: dict, columns: list[str] | pd.Index | None = None) -> str:
    time_unit = time_unit_from_config(config)
    preferred = "start_time_idx" if time_unit in {"day", "week"} else "start_month_idx"
    if columns is None:
        return preferred
    cols = set(columns)
    if preferred in cols:
        return preferred
    if "start_time_idx" in cols:
        return "start_time_idx"
    if "start_month_idx" in cols:
        return "start_month_idx"
    raise KeyError("Could not find start time index column in sample table.")


def target_time_idx_column(config: dict, columns: list[str] | pd.Index | None = None) -> str:
    time_unit = time_unit_from_config(config)
    preferred = "target_time_idx" if time_unit in {"day", "week"} else "target_month_idx"
    if columns is None:
        return preferred
    cols = set(columns)
    if preferred in cols:
        return preferred
    if "target_time_idx" in cols:
        return "target_time_idx"
    if "target_month_idx" in cols:
        return "target_month_idx"
    raise KeyError("Could not find target time index column in sample table.")


def forcing_feature_columns(config: dict) -> list[str]:
    if bool(config["features"].get("use_log1p_pumping", False)):
        return [
            "sum_net_infiltration_t_to_tH",
            "log1p_sum_pumping_t_to_tH",
        ]
    return [
        "sum_net_infiltration_t_to_tH",
        "sum_pumping_t_to_tH",
    ]


def forcing_sequence_feature_columns(config: dict, dynamic_feature_names: list[str] | None = None) -> list[str]:
    explicit_cols = config.get("features", {}).get("forcing_sequence_columns")
    if explicit_cols is not None:
        if not isinstance(explicit_cols, (list, tuple)):
            raise TypeError(
                "features.forcing_sequence_columns must be a list/tuple of dynamic feature column names."
            )
        cols = []
        seen: set[str] = set()
        for value in explicit_cols:
            col = str(value).strip()
            if not col or col in seen:
                continue
            cols.append(col)
            seen.add(col)
        if not cols:
            raise ValueError("features.forcing_sequence_columns is empty after normalization.")
        if dynamic_feature_names is not None:
            missing = [col for col in cols if col not in dynamic_feature_names]
            if missing:
                raise KeyError(f"Missing forcing sequence features in dynamic cache: {missing}")
        return cols

    time_unit = time_unit_from_config(config)
    if time_unit == "day":
        cols = ["daily_net_infiltration"]
        cols.append(
            pumping_log_dynamic_feature_name(config, time_unit=time_unit)
            if bool(config["features"].get("use_log1p_pumping", False))
            else pumping_raw_dynamic_feature_name(config, time_unit=time_unit, for_log_input=False)
        )
    elif time_unit == "week":
        cols = ["weekly_sum_net_infiltration"]
        cols.append(
            pumping_log_dynamic_feature_name(config, time_unit=time_unit)
            if bool(config["features"].get("use_log1p_pumping", False))
            else pumping_raw_dynamic_feature_name(config, time_unit=time_unit, for_log_input=False)
        )
    else:
        cols = ["monthly_sum_net_infiltration"]
        cols.append(
            pumping_log_dynamic_feature_name(config, time_unit=time_unit)
            if bool(config["features"].get("use_log1p_pumping", False))
            else pumping_raw_dynamic_feature_name(config, time_unit=time_unit, for_log_input=False)
        )
    if dynamic_feature_names is not None:
        missing = [col for col in cols if col not in dynamic_feature_names]
        if missing:
            raise KeyError(f"Missing forcing sequence features in dynamic cache: {missing}")
    return cols


def horizon_feature_columns(config: dict, columns: list[str] | pd.Index | None = None) -> list[str]:
    time_unit = time_unit_from_config(config)
    if time_unit in {"day", "week"}:
        preferred = ["h_steps", "start_phase_sin", "start_phase_cos"]
        fallback = ["h_months", "start_month_sin", "start_month_cos"]
    else:
        preferred = ["h_months", "start_month_sin", "start_month_cos"]
        fallback = ["h_steps", "start_phase_sin", "start_phase_cos"]
    if columns is None:
        return preferred
    cols = set(columns)
    if all(col in cols for col in preferred):
        return preferred
    if all(col in cols for col in fallback):
        return fallback
    raise KeyError("Could not find horizon feature columns in sample table.")


@dataclass
class CacheBundle:
    sample_table: pd.DataFrame
    dynamic_cache: dict[str, Any]
    aem_cache: dict[str, Any]
    topo_cache: dict[str, Any]


def _neighborhood_cache_path(
    config: dict,
    sample_table_path: Path,
    graph_bundle: dict[str, Any],
    root_hops: int,
) -> Path:
    graph_type = str(graph_bundle.get("graph_type", "graph"))
    distance_mode = str(graph_bundle.get("distance_mode", config["graph"].get("distance_mode", "knn"))).lower()
    if distance_mode == "regular":
        graph_tag = f"{graph_type}_regular{int(graph_bundle.get('regular_connectivity', config['graph'].get('regular_connectivity', 8)))}"
    else:
        graph_tag = f"{graph_type}_knn{int(graph_bundle.get('k_neighbors', config['graph'].get('k_neighbors', 8)))}"
    root = ensure_dir(processed_paths(config)["neighborhood_cache_dir"])
    return root / f"{sample_table_path.stem}_{graph_tag}_h{int(root_hops)}.pt"


def load_or_build_neighborhood_cache(
    config: dict,
    sample_table: pd.DataFrame,
    graph_bundle: dict[str, Any],
    graph_path: Path,
    root_hops: int,
) -> dict[int, dict[str, torch.Tensor | int]]:
    sample_table_path = processed_paths(config)["sample_table_csv"]
    cache_path = _neighborhood_cache_path(config, sample_table_path, graph_bundle, root_hops)
    latest_source_time = max(
        graph_path.stat().st_mtime,
        sample_table_path.stat().st_mtime,
    )
    if cache_path.exists() and cache_path.stat().st_mtime >= latest_source_time:
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    edge_index = graph_bundle["edge_index"]
    edge_weight = graph_bundle["edge_weight"]
    root_grid_ids = np.sort(sample_table["grid_id"].unique())
    neighborhoods: dict[int, dict[str, torch.Tensor | int]] = {}
    for grid_id in root_grid_ids:
        root_idx = int(graph_bundle["grid_id_to_index"][int(grid_id)])
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=root_idx,
            num_hops=int(root_hops),
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=int(graph_bundle["grid_ids"].shape[0]),
        )
        neighborhoods[int(grid_id)] = {
            "subset": subset.clone().cpu(),
            "edge_index": sub_edge_index.clone().cpu(),
            "edge_weight": edge_weight[edge_mask].clone().cpu(),
            "root_local_idx": int(mapping.item()),
        }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(neighborhoods, cache_path)
    return neighborhoods


def load_cache_bundle(config: dict) -> CacheBundle:
    dynamic_cache = load_or_build_dynamic_cache(config)
    return CacheBundle(
        sample_table=load_sample_table(config),
        dynamic_cache=dynamic_cache,
        aem_cache=load_or_build_aem_cache(config),
        topo_cache=load_topography_cache(config),
    )


def _collect_train_forcing_sequence_values(
    config: dict,
    train_sample_table: pd.DataFrame,
    dynamic_cache: dict,
) -> tuple[list[str], np.ndarray]:
    dynamic = np.asarray(dynamic_cache["data"], dtype=np.float32)
    dynamic_feature_names = list(dynamic_cache["feature_names"])
    seq_cols = forcing_sequence_feature_columns(config, dynamic_feature_names)
    seq_feat_idx = [dynamic_feature_names.index(col) for col in seq_cols]
    grid_id_to_index = dynamic_cache["grid_id_to_index"]

    chunks: list[np.ndarray] = []
    start_idx_col = start_time_idx_column(config, train_sample_table.columns)
    horizon_col = horizon_column(config, train_sample_table.columns)
    for row in train_sample_table.itertuples(index=False):
        grid_idx = int(grid_id_to_index[int(row.grid_id)])
        start_idx = int(getattr(row, start_idx_col))
        horizon = int(getattr(row, horizon_col))
        future = dynamic[grid_idx, start_idx + 1 : start_idx + 1 + horizon, :]
        if future.size == 0:
            continue
        chunks.append(future[:, seq_feat_idx].astype(np.float32))

    if not chunks:
        return seq_cols, np.zeros((1, len(seq_cols)), dtype=np.float32)
    return seq_cols, np.concatenate(chunks, axis=0)


def fit_train_scalers(
    config: dict,
    sample_table: pd.DataFrame,
    dynamic_cache: dict,
    aem_cache: dict,
    topo_cache: dict,
) -> dict[str, np.ndarray]:
    train = sample_table[sample_table["split"] == "train"].reset_index(drop=True)
    if train.empty:
        raise ValueError("Training split is empty; cannot fit scalers.")

    unique_train_grid_ids = np.sort(train["grid_id"].unique())
    aem_grid_map = {int(grid_id): idx for idx, grid_id in enumerate(aem_cache["grid_ids"])}
    aem_indices = np.array([aem_grid_map[int(grid_id)] for grid_id in unique_train_grid_ids], dtype=np.int64)
    aem_profiles = np.asarray(aem_cache["profiles"], dtype=np.float32)[aem_indices]
    topo_grid_map = {int(grid_id): idx for idx, grid_id in enumerate(topo_cache["grid_ids"])}
    topo_indices = np.array([topo_grid_map[int(grid_id)] for grid_id in unique_train_grid_ids], dtype=np.int64)
    topo_features = np.asarray(topo_cache["features"], dtype=np.float32)[topo_indices]

    horizon_cols = horizon_feature_columns(config, train.columns)
    forcing_cols = forcing_feature_columns(config)
    forcing_sequence_cols, forcing_sequence_values = _collect_train_forcing_sequence_values(
        config,
        train,
        dynamic_cache,
    )
    horizon_values = train[horizon_cols].to_numpy(dtype=np.float32)
    forcing_values = train[forcing_cols].to_numpy(dtype=np.float32)
    root_wtd_values = train[["wtd_t_m_bls"]].to_numpy(dtype=np.float32)
    target_values = train[["delta_h_m"]].to_numpy(dtype=np.float32)
    horizon_col = horizon_column(config, train.columns)
    horizon_targets = []
    horizon_target_means = []
    horizon_target_stds = []
    for horizon in sorted(int(h) for h in train[horizon_col].unique()):
        sub = train.loc[train[horizon_col] == horizon, "delta_h_m"].to_numpy(dtype=np.float32)
        horizon_targets.append(horizon)
        horizon_target_means.append(float(sub.mean()))
        horizon_target_stds.append(float(sub.std() + 1e-6))

    return {
        "aem_profile_mean": aem_profiles.mean(axis=0).astype(np.float32),
        "aem_profile_std": aem_profiles.std(axis=0).astype(np.float32) + 1e-6,
        "topography_feature_names": list(topo_cache.get("feature_names", [])),
        "topography_mean": topo_features.mean(axis=0).astype(np.float32),
        "topography_std": topo_features.std(axis=0).astype(np.float32) + 1e-6,
        "horizon_feature_names": horizon_cols,
        "horizon_mean": horizon_values.mean(axis=0).astype(np.float32),
        "horizon_std": horizon_values.std(axis=0).astype(np.float32) + 1e-6,
        "forcing_feature_names": forcing_cols,
        "forcing_mean": forcing_values.mean(axis=0).astype(np.float32),
        "forcing_std": forcing_values.std(axis=0).astype(np.float32) + 1e-6,
        "forcing_sequence_feature_names": forcing_sequence_cols,
        "forcing_sequence_mean": forcing_sequence_values.mean(axis=0).astype(np.float32),
        "forcing_sequence_std": forcing_sequence_values.std(axis=0).astype(np.float32) + 1e-6,
        "root_wtd_mean": root_wtd_values.mean(axis=0).astype(np.float32),
        "root_wtd_std": root_wtd_values.std(axis=0).astype(np.float32) + 1e-6,
        "target_mean": target_values.mean(axis=0).astype(np.float32),
        "target_std": target_values.std(axis=0).astype(np.float32) + 1e-6,
        "target_horizons": np.asarray(horizon_targets, dtype=np.int64),
        "target_mean_by_horizon": np.asarray(horizon_target_means, dtype=np.float32),
        "target_std_by_horizon": np.asarray(horizon_target_stds, dtype=np.float32),
    }


def _standardize(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return safe_standardize(array.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)).astype(np.float32)


def _target_scaling_mode(config: dict) -> str:
    return str(config.get("training", {}).get("target_scaling", "global")).lower()


def _target_mean_std_for_horizon(scalers: dict[str, np.ndarray], horizon: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode != "per_horizon":
        return scalers["target_mean"], scalers["target_std"]
    horizons = np.asarray(scalers.get("target_horizons", []), dtype=np.int64)
    means = np.asarray(scalers.get("target_mean_by_horizon", []), dtype=np.float32)
    stds = np.asarray(scalers.get("target_std_by_horizon", []), dtype=np.float32)
    match = np.where(horizons == int(horizon))[0]
    if match.size == 0:
        return scalers["target_mean"], scalers["target_std"]
    idx = int(match[0])
    return np.asarray([means[idx]], dtype=np.float32), np.asarray([stds[idx]], dtype=np.float32)


class TemporalGNNDataset(Dataset):
    def __init__(
        self,
        config: dict,
        sample_table: pd.DataFrame,
        dynamic_cache: dict,
        aem_cache: dict,
        topo_cache: dict,
        graph_bundle: dict[str, Any],
        scalers: dict[str, np.ndarray],
        neighborhood_cache: dict[int, dict[str, torch.Tensor | int]] | None = None,
    ) -> None:
        self.config = config
        self.sample_table = sample_table.reset_index(drop=True)
        self.dynamic_cache = dynamic_cache
        self.aem_cache = aem_cache
        self.topo_cache = topo_cache
        self.graph = graph_bundle
        self.scalers = scalers
        self.aem_grid_map = {int(grid_id): idx for idx, grid_id in enumerate(np.asarray(aem_cache["grid_ids"], dtype=np.int64))}
        self.topo_grid_map = {int(grid_id): idx for idx, grid_id in enumerate(np.asarray(topo_cache["grid_ids"], dtype=np.int64))}
        self.root_hops = int(config["graph"]["graph_hops"])
        self.dynamic_feature_names = list(dynamic_cache["feature_names"])
        self.dynamic_sequence_cols = forcing_sequence_feature_columns(config, self.dynamic_feature_names)
        self.dynamic_feature_indices = [self.dynamic_feature_names.index(col) for col in self.dynamic_sequence_cols]
        self.horizon_col = horizon_column(config, self.sample_table.columns)
        self.start_idx_col = start_time_idx_column(config, self.sample_table.columns)
        self.target_idx_col = target_time_idx_column(config, self.sample_table.columns)
        self.horizon_feature_cols = horizon_feature_columns(config, self.sample_table.columns)
        self.max_horizon = max(int(v) for v in config["data"]["horizons"])
        self.neighborhoods = neighborhood_cache if neighborhood_cache is not None else self._build_neighborhoods()
        self.target_scaling_mode = _target_scaling_mode(config)

    def _build_neighborhoods(self) -> dict[int, dict[str, torch.Tensor | int]]:
        edge_index = self.graph["edge_index"]
        edge_weight = self.graph["edge_weight"]
        root_grid_ids = np.sort(self.sample_table["grid_id"].unique())
        neighborhoods: dict[int, dict[str, torch.Tensor | int]] = {}
        for grid_id in root_grid_ids:
            root_idx = int(self.graph["grid_id_to_index"][int(grid_id)])
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx=root_idx,
                num_hops=self.root_hops,
                edge_index=edge_index,
                relabel_nodes=True,
                num_nodes=int(self.graph["grid_ids"].shape[0]),
            )
            neighborhoods[int(grid_id)] = {
                "subset": subset.clone().cpu(),
                "edge_index": sub_edge_index.clone().cpu(),
                "edge_weight": edge_weight[edge_mask].clone().cpu(),
                "root_local_idx": int(mapping.item()),
            }
        return neighborhoods

    def __len__(self) -> int:
        return len(self.sample_table)

    def _node_forcing_summary(self, node_indices: np.ndarray, start_idx: int, horizon: int) -> np.ndarray:
        if not bool(self.config["data"].get("scenario_mode", True)):
            return np.zeros((len(node_indices), 2), dtype=np.float32)
        dynamic = np.asarray(self.dynamic_cache["data"], dtype=np.float32)
        feat_names = list(self.dynamic_cache["feature_names"])
        time_unit = time_unit_from_config(self.config)
        if time_unit == "day":
            swb_name = "daily_net_infiltration"
            pump_name = pumping_raw_dynamic_feature_name(
                self.config,
                time_unit=time_unit,
                for_log_input=bool(self.config["features"].get("use_log1p_pumping", False)),
            )
        elif time_unit == "week":
            swb_name = "weekly_sum_net_infiltration"
            pump_name = pumping_raw_dynamic_feature_name(
                self.config,
                time_unit=time_unit,
                for_log_input=bool(self.config["features"].get("use_log1p_pumping", False)),
            )
        else:
            swb_name = "monthly_sum_net_infiltration"
            pump_name = pumping_raw_dynamic_feature_name(
                self.config,
                time_unit=time_unit,
                for_log_input=bool(self.config["features"].get("use_log1p_pumping", False)),
            )
        swb_idx = feat_names.index(swb_name)
        pump_idx = feat_names.index(pump_name)
        future = dynamic[node_indices, start_idx + 1 : start_idx + 1 + horizon, :]
        sum_net = np.sum(future[:, :, swb_idx], axis=1)
        sum_pump = np.sum(future[:, :, pump_idx], axis=1)
        if bool(self.config["features"].get("use_log1p_pumping", False)):
            pump_sum_feat = np.log1p(np.clip(sum_pump, a_min=0.0, a_max=None))
        else:
            pump_sum_feat = sum_pump
        return np.column_stack([sum_net, pump_sum_feat]).astype(np.float32)

    def _node_forcing_sequence(self, node_indices: np.ndarray, start_idx: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        seq = np.zeros((len(node_indices), self.max_horizon, len(self.dynamic_feature_indices)), dtype=np.float32)
        lengths = np.ones(len(node_indices), dtype=np.int64)
        if not bool(self.config["features"].get("use_horizon_forcing_sequence", False)):
            return seq, lengths

        dynamic = np.asarray(self.dynamic_cache["data"], dtype=np.float32)
        future = dynamic[node_indices, start_idx + 1 : start_idx + 1 + horizon, :]
        if future.size == 0:
            return seq, lengths

        seq_values = future[:, :, self.dynamic_feature_indices].astype(np.float32)
        seq_values = _standardize(
            seq_values,
            self.scalers["forcing_sequence_mean"],
            self.scalers["forcing_sequence_std"],
        )
        seq[:, : seq_values.shape[1], :] = seq_values
        lengths[:] = seq_values.shape[1]
        return seq, lengths

    def __getitem__(self, idx: int) -> Data:
        row = self.sample_table.iloc[idx]
        neighborhood = self.neighborhoods[int(row.grid_id)]
        subset = neighborhood["subset"].numpy().astype(np.int64)
        edge_index = neighborhood["edge_index"]
        edge_weight = neighborhood["edge_weight"]
        root_local_idx = int(neighborhood["root_local_idx"])
        start_idx = int(row[self.start_idx_col])
        horizon = int(row[self.horizon_col])

        aem_idx = np.array(
            [self.aem_grid_map[int(self.dynamic_cache["grid_ids"][node_idx])] for node_idx in subset],
            dtype=np.int64,
        )
        aem_profile = _standardize(
            np.asarray(self.aem_cache["profiles"], dtype=np.float32)[aem_idx],
            self.scalers["aem_profile_mean"],
            self.scalers["aem_profile_std"],
        )
        topo_idx = np.array(
            [self.topo_grid_map[int(self.dynamic_cache["grid_ids"][node_idx])] for node_idx in subset],
            dtype=np.int64,
        )
        topo_features = _standardize(
            np.asarray(self.topo_cache["features"], dtype=np.float32)[topo_idx],
            self.scalers["topography_mean"],
            self.scalers["topography_std"],
        )
        horizon_vec = _standardize(
            row[self.horizon_feature_cols].to_numpy(dtype=np.float32),
            self.scalers["horizon_mean"],
            self.scalers["horizon_std"],
        )
        horizon_node = np.repeat(horizon_vec[None, :], repeats=len(subset), axis=0).astype(np.float32)
        forcing_node = self._node_forcing_summary(subset, start_idx, horizon)
        forcing_node = _standardize(forcing_node, self.scalers["forcing_mean"], self.scalers["forcing_std"])
        forcing_sequence, forcing_sequence_length = self._node_forcing_sequence(subset, start_idx, horizon)
        root_wtd = _standardize(
            np.asarray([row.wtd_t_m_bls], dtype=np.float32),
            self.scalers["root_wtd_mean"],
            self.scalers["root_wtd_std"],
        )
        target = _standardize(
            np.asarray([row.delta_h_m], dtype=np.float32),
            *_target_mean_std_for_horizon(self.scalers, int(row[self.horizon_col]), self.target_scaling_mode),
        )
        root_mask = np.zeros(len(subset), dtype=bool)
        root_mask[root_local_idx] = True

        return Data(
            aem_profile=torch.tensor(aem_profile, dtype=torch.float32),
            topography_features=torch.tensor(topo_features, dtype=torch.float32),
            horizon_features=torch.tensor(horizon_node, dtype=torch.float32),
            forcing_summary=torch.tensor(forcing_node, dtype=torch.float32),
            forcing_sequence=torch.tensor(forcing_sequence, dtype=torch.float32),
            forcing_sequence_length=torch.tensor(forcing_sequence_length, dtype=torch.long),
            edge_index=edge_index.clone(),
            edge_weight=edge_weight.clone(),
            root_mask=torch.tensor(root_mask, dtype=torch.bool),
            root_wtd=torch.tensor(root_wtd, dtype=torch.float32),
            y=torch.tensor(target, dtype=torch.float32),
            sample_id=torch.tensor([int(row.sample_id)], dtype=torch.long),
            grid_id=torch.tensor([int(row.grid_id)], dtype=torch.long),
            start_time_idx=torch.tensor([int(row[self.start_idx_col])], dtype=torch.long),
            target_time_idx=torch.tensor([int(row[self.target_idx_col])], dtype=torch.long),
            h_steps=torch.tensor([int(row[self.horizon_col])], dtype=torch.long),
            x_coord=torch.tensor([float(row.x)], dtype=torch.float32),
            y_coord=torch.tensor([float(row.y)], dtype=torch.float32),
            delta_h_true=torch.tensor([float(row.delta_h_m)], dtype=torch.float32),
            n_days_observed_t=torch.tensor([int(row.n_days_observed_t)], dtype=torch.long),
            n_days_observed_target=torch.tensor([int(row.n_days_observed_target)], dtype=torch.long),
            n_unique_sites_t=torch.tensor(
                [int(row.n_unique_sites_t) if "n_unique_sites_t" in self.sample_table.columns else 1],
                dtype=torch.long,
            ),
            n_unique_sites_target=torch.tensor(
                [int(row.n_unique_sites_target) if "n_unique_sites_target" in self.sample_table.columns else 1],
                dtype=torch.long,
            ),
            num_nodes=len(subset),
        )
