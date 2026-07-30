from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
import rasterio


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["__config_path__"] = str(Path(path).resolve())
    return config


def time_unit_from_config(config: dict[str, Any]) -> str:
    return str(config.get("data", {}).get("time_unit", "month")).lower()


def pumping_log_input_mode(config: dict[str, Any]) -> str:
    raw = str(config.get("features", {}).get("pumping_log_input", "raw")).strip().lower()
    if raw in {"mm", "depth_mm", "equivalent_mm", "water_depth_mm"}:
        return "depth_mm"
    return "raw"


def pumping_raw_dynamic_feature_name(
    config: dict[str, Any],
    time_unit: str | None = None,
    *,
    for_log_input: bool = False,
) -> str:
    time_unit = (time_unit or time_unit_from_config(config)).lower()
    if time_unit == "day":
        base = "daily_pumping"
    elif time_unit == "week":
        base = "weekly_pumping"
    else:
        base = "monthly_pumping"
    if for_log_input and pumping_log_input_mode(config) == "depth_mm":
        return f"{base}_mm"
    return base


def pumping_log_dynamic_feature_name(
    config: dict[str, Any],
    time_unit: str | None = None,
) -> str:
    raw_name = pumping_raw_dynamic_feature_name(config, time_unit=time_unit, for_log_input=True)
    return f"{raw_name}_log1p"


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timestamp_to_month_ord(ts: pd.Timestamp) -> int:
    return int(ts.year) * 12 + int(ts.month)


def month_ord_to_timestamp(month_ord: int) -> pd.Timestamp:
    year = month_ord // 12
    month = month_ord % 12
    if month == 0:
        year -= 1
        month = 12
    return pd.Timestamp(year=year, month=month, day=1)


def month_range(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS")


def add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return (pd.Timestamp(ts).to_period("M") + n).to_timestamp()


def month_sin_cos(ts: pd.Timestamp) -> tuple[float, float]:
    month = int(pd.Timestamp(ts).month)
    angle = 2.0 * math.pi * (month - 1) / 12.0
    return float(math.sin(angle)), float(math.cos(angle))



def safe_standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std == 0.0, 1.0, std)
    return (values - mean) / std


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if y_true.size <= 1 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom == 0.0:
        r2 = float("nan")
        nse = float("nan")
    else:
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
        nse = r2
    bias = float(np.mean(y_pred - y_true))
    return {
        "pearson_r": pearson_r,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nse": nse,
        "bias": bias,
    }


def split_name_from_date(date: pd.Timestamp, config: dict[str, Any]) -> str | None:
    data_cfg = config["data"]
    train_end = pd.Timestamp(data_cfg["train_end"])
    val_start = pd.Timestamp(data_cfg["val_start"])
    val_end = pd.Timestamp(data_cfg["val_end"])
    test_start = pd.Timestamp(data_cfg["test_start"])
    test_end = pd.Timestamp(data_cfg["test_end"])

    date = pd.Timestamp(date)
    if date <= train_end:
        return "train"
    if val_start <= date <= val_end:
        return "validation"
    if test_start <= date <= test_end:
        return "test"
    return None


def assign_split(
    start_month: pd.Timestamp,
    target_month: pd.Timestamp,
    config: dict[str, Any],
) -> str | None:
    split_by = str(config["data"]["split_by"]).lower()
    if split_by == "start_date":
        return split_name_from_date(start_month, config)
    return split_name_from_date(target_month, config)


def output_paths(config: dict[str, Any], experiment_name: str) -> dict[str, Path]:
    root = ensure_dir(config["paths"]["output_dir"])
    exp_dir = ensure_dir(root / "checkpoints" / experiment_name)
    pred_dir = ensure_dir(root / "predictions")
    metric_dir = ensure_dir(root / "metrics")
    fig_dir = ensure_dir(root / "figures")
    graph_dir = ensure_dir(root / "graphs")
    return {
        "output_root": root,
        "experiment_dir": exp_dir,
        "prediction_dir": pred_dir,
        "metric_dir": metric_dir,
        "figure_dir": fig_dir,
        "graph_dir": graph_dir,
    }


def device_from_config() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def processed_dir(config: dict[str, Any]) -> Path:
    return ensure_dir(config["paths"]["data_processed_dir"])


def processed_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = processed_dir(config)
    time_unit = time_unit_from_config(config)
    if time_unit == "day":
        dynamic_csv = root / "day_forcing.csv"
        dynamic_cache = root / "dynamic_day_cache.pt"
        wtd_csv = root / "wtd_daily_clean.csv"
        wtd_meta = root / "wtd_daily_metadata.json"
    elif time_unit == "week":
        dynamic_csv = root / "week_forcing.csv"
        dynamic_cache = root / "dynamic_week_cache.pt"
        wtd_csv = root / "wtd_weekly_clean.csv"
        wtd_meta = root / "wtd_weekly_metadata.json"
    else:
        dynamic_csv = root / "swb_monthly.csv"
        dynamic_cache = root / "dynamic_monthly_cache.pt"
        wtd_csv = root / "wtd_monthly_clean.csv"
        wtd_meta = root / "wtd_monthly_metadata.json"
    return {
        "root": root,
        "swb_monthly_csv": root / "swb_monthly.csv",
        "swb_metadata_json": root / "swb_monthly_metadata.json",
        "aiwum_monthly_csv": root / "aiwum_monthly.csv",
        "aiwum_metadata_json": root / "aiwum_monthly_metadata.json",
        "wtd_monthly_csv": root / "wtd_monthly_clean.csv",
        "wtd_metadata_json": root / "wtd_monthly_metadata.json",
        "dynamic_csv": dynamic_csv,
        "dynamic_cache_pt": dynamic_cache,
        "wtd_csv": wtd_csv,
        "wtd_metadata_json_generic": wtd_meta,
        "aem_cache_pt": root / "aem_node_features.pt",
        "neighborhood_cache_dir": root / "neighborhood_cache",
        "sample_table_csv": Path(config["paths"]["sample_table_path"]),
        "sample_metadata_json": root / "sample_table_metadata.json",
        "month_index_csv": root / "month_index.csv",
        "time_index_csv": root / "time_index.csv",
        "scalers_pt": root / "train_scalers.pt",
    }


def load_active_grid(config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(config["paths"]["active_grid_path"]).copy()
    frame = frame.rename(
        columns={
            "node_id": "grid_id",
            "x_center": "x",
            "y_center": "y",
        }
    )
    frame = frame.sort_values("grid_id").reset_index(drop=True)
    return frame


def month_index_frame(config: dict[str, Any]) -> pd.DataFrame:
    starts = month_range(config["data"]["start_date"], config["data"]["end_date"])
    out = pd.DataFrame({"month_start": starts})
    out["month_label"] = out["month_start"].dt.strftime("%Y-%m")
    out["month_ord"] = out["month_start"].apply(timestamp_to_month_ord)
    out["month_idx"] = np.arange(len(out), dtype=int)
    return out


def load_or_build_aem_cache(config: dict[str, Any]) -> dict[str, Any]:
    def _normalize_aem_depth_bands(features_cfg: dict[str, Any]) -> list[tuple[float, float]] | None:
        raw_bands = features_cfg.get("aem_depth_bands_m")
        if raw_bands is None:
            return None
        if not isinstance(raw_bands, (list, tuple)) or len(raw_bands) == 0:
            raise ValueError("features.aem_depth_bands_m must be a non-empty list of [zmin, zmax] pairs.")

        bands: list[tuple[float, float]] = []
        prev_zmax = -float("inf")
        for idx, item in enumerate(raw_bands):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(
                    f"features.aem_depth_bands_m[{idx}] must be a two-value [zmin, zmax] pair; got {item!r}."
                )
            zmin = float(item[0])
            zmax = float(item[1])
            if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
                raise ValueError(
                    f"Invalid AEM depth band {item!r}; expected finite bounds with zmax > zmin."
                )
            if zmin < prev_zmax:
                raise ValueError(
                    "features.aem_depth_bands_m must be ordered from shallow to deep without overlap."
                )
            bands.append((zmin, zmax))
            prev_zmax = zmax
        return bands

    def _aggregate_aem_profile_to_bands(
        profiles: np.ndarray,
        depths: np.ndarray,
        bands: list[tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        agg = np.full((profiles.shape[0], len(bands)), np.nan, dtype=np.float32)
        band_midpoints: list[float] = []
        for band_idx, (zmin, zmax) in enumerate(bands):
            band_mask = (depths >= zmin) & (depths < zmax)
            if not np.any(band_mask):
                raise ValueError(
                    f"AEM depth band [{zmin}, {zmax}) m keeps zero layers. "
                    f"Available depth centers span {float(depths.min())} to {float(depths.max())} m."
                )
            agg[:, band_idx] = np.nanmean(profiles[:, band_mask], axis=1).astype(np.float32)
            band_midpoints.append(0.5 * (zmin + zmax))

        inds = np.where(~np.isfinite(agg))
        if inds[0].size:
            band_means = np.nanmean(agg, axis=0)
            agg[inds] = np.take(band_means, inds[1])
        return agg.astype(np.float32), np.asarray(band_midpoints, dtype=np.float32)

    def _apply_aem_profile_view(cache: dict[str, Any]) -> dict[str, Any]:
        features_cfg = config.get("features", {})
        profiles = np.asarray(cache["profiles"], dtype=np.float32)
        depths = np.asarray(cache.get("depths_m", []), dtype=np.float32)
        if profiles.ndim != 2 or depths.ndim != 1 or profiles.shape[1] != depths.shape[0]:
            return cache

        bands = _normalize_aem_depth_bands(features_cfg)
        if bands is not None:
            band_profiles, band_midpoints = _aggregate_aem_profile_to_bands(profiles, depths, bands)
            view = dict(cache)
            view["profiles"] = band_profiles
            view["depths_m"] = band_midpoints
            view["aem_profile_layer_count"] = int(len(bands))
            view["aem_profile_max_depth_m"] = float(max(zmax for _, zmax in bands))
            view["aem_profile_depth_bands_m"] = [[float(zmin), float(zmax)] for zmin, zmax in bands]
            return view

        max_depth_value = features_cfg.get("aem_max_depth_m")
        if max_depth_value is not None:
            max_depth_m = float(max_depth_value)
            keep_mask = depths <= max_depth_m
            if not np.any(keep_mask):
                raise ValueError(
                    f"AEM max depth {max_depth_m} m keeps zero layers. "
                    f"Available depth range is {float(depths.min())} to {float(depths.max())} m."
                )
        else:
            requested_layers = int(features_cfg.get("aem_num_layers", profiles.shape[1]))
            requested_layers = max(1, min(requested_layers, profiles.shape[1]))
            keep_mask = np.zeros_like(depths, dtype=bool)
            keep_mask[:requested_layers] = True

        if bool(np.all(keep_mask)):
            return cache

        view = dict(cache)
        view["profiles"] = profiles[:, keep_mask].astype(np.float32)
        view["depths_m"] = depths[keep_mask].astype(np.float32)
        view["aem_profile_layer_count"] = int(keep_mask.sum())
        view["aem_profile_max_depth_m"] = float(view["depths_m"].max())
        return view

    cache_path = processed_paths(config)["aem_cache_pt"]
    aem_path = Path(config["paths"]["aem_profile_path"])
    depth_path = Path(config["paths"]["aem_depth_path"])
    active_grid_path = Path(config["paths"]["active_grid_path"])
    if cache_path.exists() and cache_path.stat().st_mtime >= max(
        aem_path.stat().st_mtime,
        depth_path.stat().st_mtime,
        active_grid_path.stat().st_mtime,
    ):
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        return _apply_aem_profile_view(cache)

    active = load_active_grid(config)
    with depth_path.open("r", encoding="utf-8") as f:
        depths = np.asarray(json.load(f), dtype=np.float32)

    with rasterio.open(aem_path) as ds:
        stack = ds.read().astype(np.float32)
        nodata = ds.nodata
    if nodata is not None:
        stack[stack == nodata] = np.nan

    src_rows = (stack.shape[1] - 1) - active["row"].to_numpy(dtype=int)
    cols = active["col"].to_numpy(dtype=int)
    profiles = np.transpose(stack[:, src_rows, cols], (1, 0)).astype(np.float32)

    layer_means = np.nanmean(profiles, axis=0)
    inds = np.where(~np.isfinite(profiles))
    profiles[inds] = np.take(layer_means, inds[1])

    cache = {
        "grid_ids": active["grid_id"].to_numpy(dtype=np.int64),
        "profiles": profiles.astype(np.float32),
        "depths_m": depths.astype(np.float32),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return _apply_aem_profile_view(cache)


def load_topography_cache(config: dict[str, Any]) -> dict[str, Any]:
    active = load_active_grid(config)
    topo_path_value = str(config.get("paths", {}).get("topography_1km_path", "")).strip()
    if not topo_path_value:
        return {
            "grid_ids": active["grid_id"].to_numpy(dtype=np.int64),
            "features": np.zeros((len(active), 0), dtype=np.float32),
            "feature_names": [],
        }

    topo_path = Path(topo_path_value)
    if not topo_path.is_absolute():
        config_path = config.get("__config_path__")
        if config_path:
            topo_path = Path(config_path).resolve().parent.parent / topo_path
        else:
            topo_path = topo_path.resolve()
    if not topo_path.exists():
        raise FileNotFoundError(f"Topography feature file not found: {topo_path}")

    topo = pd.read_csv(topo_path).copy()
    if "grid_id" not in topo.columns:
        if "node_id" in topo.columns:
            topo = topo.rename(columns={"node_id": "grid_id"})
        else:
            raise KeyError("Topography CSV must contain 'grid_id' or 'node_id'.")

    feature_names = list(config.get("features", {}).get("topography_columns", ["mean_elevation_m", "mean_slope_deg"]))
    missing = [col for col in feature_names if col not in topo.columns]
    if missing:
        raise KeyError(f"Missing topography columns in {topo_path}: {missing}")

    merged = active[["grid_id"]].merge(topo[["grid_id", *feature_names]], on="grid_id", how="left")
    features = merged[feature_names].to_numpy(dtype=np.float32)
    if np.isnan(features).any():
        col_means = np.nanmean(features, axis=0)
        inds = np.where(~np.isfinite(features))
        features[inds] = np.take(col_means, inds[1])

    return {
        "grid_ids": active["grid_id"].to_numpy(dtype=np.int64),
        "features": features.astype(np.float32),
        "feature_names": feature_names,
        "source_path": str(topo_path),
    }


def save_month_index(config: dict[str, Any]) -> pd.DataFrame:
    frame = month_index_frame(config)
    frame.to_csv(processed_paths(config)["month_index_csv"], index=False, lineterminator="\n")
    return frame


def load_month_index(config: dict[str, Any]) -> pd.DataFrame:
    path = processed_paths(config)["month_index_csv"]
    if path.exists():
        return pd.read_csv(path, parse_dates=["month_start"])
    return save_month_index(config)


def time_index_frame(config: dict[str, Any]) -> pd.DataFrame:
    time_unit = time_unit_from_config(config)
    if time_unit != "month":
        raise ValueError(f"time_index_frame can only synthesize month indices, got time_unit={time_unit!r}")
    month_frame = month_index_frame(config)
    out = pd.DataFrame(
        {
            "time_index": month_frame["month_idx"].to_numpy(dtype=int),
            "time_start": month_frame["month_start"],
            "time_label": month_frame["month_label"],
            "year": month_frame["month_start"].dt.year.to_numpy(dtype=int),
            "month": month_frame["month_start"].dt.month.to_numpy(dtype=int),
        }
    )
    return out


def save_time_index(config: dict[str, Any]) -> pd.DataFrame:
    frame = time_index_frame(config)
    frame.to_csv(processed_paths(config)["time_index_csv"], index=False, lineterminator="\n")
    return frame


def load_time_index(config: dict[str, Any]) -> pd.DataFrame:
    path = processed_paths(config)["time_index_csv"]
    if path.exists():
        return pd.read_csv(path, parse_dates=["time_start"])
    return save_time_index(config)


def _sanitize_dynamic_cache_inplace(
    cache: dict[str, Any],
    pumping_missing_as_zero: bool,
) -> bool:
    data = np.asarray(cache.get("data"), dtype=np.float32)
    if data.ndim != 3:
        return False
    feature_names = list(cache.get("feature_names", []))
    if len(feature_names) != data.shape[2]:
        return False

    changed = False
    for feat_idx, feat_name in enumerate(feature_names):
        values = data[:, :, feat_idx]
        missing_mask = ~np.isfinite(values)
        if not np.any(missing_mask):
            continue

        if pumping_missing_as_zero and feat_name in {
            "monthly_pumping",
            "monthly_pumping_mm",
            "monthly_pumping_log1p",
            "monthly_pumping_mm_log1p",
        }:
            fill_value = 0.0
        else:
            finite_vals = values[np.isfinite(values)]
            fill_value = float(np.nanmedian(finite_vals)) if finite_vals.size else 0.0
        values[missing_mask] = fill_value
        data[:, :, feat_idx] = values
        changed = True

    if changed:
        cache["data"] = data.astype(np.float32)
    return changed


def load_or_build_dynamic_cache(config: dict[str, Any]) -> dict[str, Any]:
    cache_path = processed_paths(config)["dynamic_cache_pt"]
    if cache_path.exists():
        try:
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception:
            cache_path.unlink(missing_ok=True)
            cache = None
        if cache is None:
            pass
        else:
            expected_months = month_index_frame(config)["month_label"].to_numpy()
            cached_months = np.asarray(cache.get("month_labels", cache.get("time_labels", [])))
            if cached_months.shape[0] != expected_months.shape[0] or not np.array_equal(cached_months, expected_months):
                cache_path.unlink(missing_ok=True)
                cache = None
            else:
                cached_features = set(cache.get("feature_names", []))
                expected_features = {"monthly_sum_net_infiltration", "monthly_pumping"}
                if pumping_log_input_mode(config) == "depth_mm":
                    expected_features.update({"monthly_pumping_mm", "monthly_pumping_mm_log1p"})
                if not expected_features.issubset(cached_features):
                    cache_path.unlink(missing_ok=True)
                    cache = None
                else:
                    changed = _sanitize_dynamic_cache_inplace(
                        cache,
                        pumping_missing_as_zero=bool(config["features"].get("pumping_missing_as_zero", True)),
                    )
                    if changed:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(cache, cache_path)
                    return cache

    time_unit = time_unit_from_config(config)
    if time_unit != "month":
        raise FileNotFoundError(
            f"Prebuilt dynamic cache not found for time_unit={time_unit!r}: {cache_path}. "
            "Daily/weekly training expects the processed cache to already exist."
        )

    paths = processed_paths(config)
    swb_path = paths["swb_monthly_csv"]
    aiwum_path = paths["aiwum_monthly_csv"]
    if not swb_path.exists():
        raise FileNotFoundError(f"Monthly SWB file not found: {swb_path}")
    if not aiwum_path.exists():
        raise FileNotFoundError(f"Monthly AIWUM file not found: {aiwum_path}")
    month_path = paths["month_index_csv"]
    active_grid_path = Path(config["paths"]["active_grid_path"])
    latest_source_time = max(
        swb_path.stat().st_mtime,
        aiwum_path.stat().st_mtime,
        month_path.stat().st_mtime if month_path.exists() else 0.0,
        active_grid_path.stat().st_mtime,
    )
    if cache_path.exists() and cache_path.stat().st_mtime >= latest_source_time:
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    active = load_active_grid(config)
    month_index = load_month_index(config)
    swb = pd.read_csv(swb_path, parse_dates=["month_start"])
    aiwum = pd.read_csv(aiwum_path, parse_dates=["month_start"])

    swb_features = [
        column
        for column in [
            "monthly_sum_net_infiltration",
            "monthly_mean_net_infiltration",
            "monthly_max_7day_net_infiltration",
        ]
        if column in swb.columns
    ]
    aiwum_features = [
        column
        for column in [
            "monthly_pumping",
            "monthly_pumping_mm",
            "monthly_pumping_log1p",
            "monthly_pumping_mm_log1p",
        ]
        if column in aiwum.columns
    ]
    feature_names = swb_features + aiwum_features

    grid_ids = active["grid_id"].to_numpy(dtype=np.int64)
    month_labels = month_index["month_label"].to_numpy()
    month_ords = month_index["month_ord"].to_numpy(dtype=np.int64)
    n_grid = len(grid_ids)
    n_month = len(month_labels)
    n_feat = len(feature_names)

    grid_to_idx = {int(grid_id): idx for idx, grid_id in enumerate(grid_ids)}
    month_to_idx = {label: idx for idx, label in enumerate(month_labels)}

    data = np.full((n_grid, n_month, n_feat), np.nan, dtype=np.float32)

    def _assign(frame: pd.DataFrame, cols: list[str]) -> None:
        if not cols:
            return
        row_idx = frame["grid_id"].map(grid_to_idx).to_numpy(dtype=np.int64)
        month_idx = frame["month_label"].map(month_to_idx).to_numpy(dtype=np.int64)
        for col in cols:
            feat_idx = feature_names.index(col)
            data[row_idx, month_idx, feat_idx] = frame[col].to_numpy(dtype=np.float32)

    _assign(swb, swb_features)
    _assign(aiwum, aiwum_features)

    cache = {
        "grid_ids": grid_ids,
        "time_labels": month_labels,
        "time_starts": month_index["month_start"].to_numpy(),
        "month_labels": month_labels,
        "month_ords": month_ords,
        "feature_names": feature_names,
        "data": data,
        "grid_id_to_index": grid_to_idx,
        "time_label_to_index": month_to_idx,
        "month_label_to_index": month_to_idx,
        "month_ord_to_index": {int(v): idx for idx, v in enumerate(month_ords)},
    }
    _sanitize_dynamic_cache_inplace(
        cache,
        pumping_missing_as_zero=bool(config["features"].get("pumping_missing_as_zero", True)),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache
