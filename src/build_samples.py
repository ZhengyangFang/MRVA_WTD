from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .preprocess_aiwum import preprocess_aiwum
from .preprocess_swb import preprocess_swb
from .preprocess_wtd import preprocess_wtd
from .utils import (
    assign_split,
    load_or_build_dynamic_cache,
    load_month_index,
    processed_paths,
    pumping_log_input_mode,
    pumping_log_dynamic_feature_name,
    pumping_raw_dynamic_feature_name,
    save_json,
)


@dataclass
class SampleBuildResult:
    sample_table: pd.DataFrame
    metadata: dict


def _ensure_preprocessed(config: dict) -> None:
    paths = processed_paths(config)
    if not paths["swb_monthly_csv"].exists():
        preprocess_swb(config)
    aiwum_needs_refresh = not paths["aiwum_monthly_csv"].exists()
    if not aiwum_needs_refresh:
        try:
            aiwum_cols = set(pd.read_csv(paths["aiwum_monthly_csv"], nrows=1).columns)
        except Exception:
            aiwum_cols = set()
        expected_cols = {"monthly_pumping", "monthly_pumping_log1p"}
        if pumping_log_input_mode(config) == "depth_mm":
            expected_cols.update({"monthly_pumping_mm", "monthly_pumping_mm_log1p"})
        aiwum_needs_refresh = not expected_cols.issubset(aiwum_cols)
    if aiwum_needs_refresh:
        preprocess_aiwum(config)
    if not paths["wtd_monthly_csv"].exists():
        preprocess_wtd(config)
    load_month_index(config)


def _has_finite_window(data: np.ndarray) -> bool:
    return bool(np.isfinite(data).all())


def _uses_spatial_grid_split(config: dict) -> bool:
    split_by = str(config.get("data", {}).get("split_by", "")).lower()
    return split_by in {
        "spatial_random_grid",
        "spatial_grid",
        "random_grid",
        "grid_random",
        "spatiotemporal_grid_time",
        "space_time_grid",
    }


def _uses_spatiotemporal_split(config: dict) -> bool:
    split_by = str(config.get("data", {}).get("split_by", "")).lower()
    return split_by in {"spatiotemporal_grid_time", "space_time_grid"}


def _spatial_split_fractions(config: dict) -> list[tuple[str, float]]:
    data_cfg = config.get("data", {})
    fractions = data_cfg.get(
        "spatial_split_fractions",
        {"train": 0.70, "validation": 0.15, "test": 0.15},
    )
    if _uses_spatiotemporal_split(config):
        ordered_splits = ["train", "test"]
    else:
        ordered_splits = ["train", "validation", "test"]
    parsed: list[tuple[str, float]] = []
    for split in ordered_splits:
        value = float(fractions.get(split, 0.0))
        if value > 0.0:
            parsed.append((split, value))
    if not parsed:
        raise ValueError("spatial_split_fractions must contain at least one positive split fraction.")

    total = float(sum(value for _, value in parsed))
    return [(split, value / total) for split, value in parsed]


def _allocate_split_counts(n_items: int, fractions: list[tuple[str, float]]) -> np.ndarray:
    raw_counts = np.asarray([n_items * fraction for _, fraction in fractions], dtype=float)
    counts = np.floor(raw_counts).astype(int)
    remainder = int(n_items - counts.sum())
    if remainder > 0:
        fractional_order = np.argsort(-(raw_counts - counts))
        counts[fractional_order[:remainder]] += 1
    return counts


def _build_spatial_grid_split(config: dict, wtd: pd.DataFrame, paths: dict) -> tuple[dict[int, str], dict]:
    grid_ids = np.sort(wtd["grid_id"].dropna().astype(int).unique())
    if len(grid_ids) == 0:
        raise ValueError("Cannot build a spatial split because the WTD table has no grid_id values.")

    fractions = _spatial_split_fractions(config)
    seed = int(config.get("data", {}).get("spatial_split_seed", config.get("training", {}).get("seed", 42)))
    rng = np.random.default_rng(seed)
    shuffled_grid_ids = rng.permutation(grid_ids)
    counts = _allocate_split_counts(len(shuffled_grid_ids), fractions)

    labels = np.empty(len(shuffled_grid_ids), dtype=object)
    start = 0
    for (split, _), count in zip(fractions, counts):
        labels[start : start + int(count)] = split
        start += int(count)

    lookup = pd.DataFrame({"grid_id": shuffled_grid_ids.astype(np.int64), "split": labels})
    lookup = lookup.sort_values("grid_id").reset_index(drop=True)

    split_counts = lookup["split"].value_counts().sort_index()
    metadata = {
        "mode": "spatiotemporal_grid_time" if _uses_spatiotemporal_split(config) else "spatial_random_grid",
        "seed": seed,
        "grid_count": int(len(lookup)),
        "split_fractions": {split: float(fraction) for split, fraction in fractions},
        "grid_split_counts": {split: int(count) for split, count in split_counts.items()},
    }
    return dict(zip(lookup["grid_id"].astype(int), lookup["split"].astype(str))), metadata


def _time_split_label(start_month: pd.Timestamp, target_month: pd.Timestamp, config: dict) -> str | None:
    split_by = str(config["data"]["split_by"]).lower()
    if split_by == "start_date":
        date = start_month
    else:
        date = target_month

    split = assign_split(start_month, target_month, config)
    if split == "train":
        return "train_years"
    if split == "validation":
        return "validation_years"
    if split == "test":
        return "test_years"
    return None


def _assign_spatiotemporal_split(
    grid_id: int,
    start_month: pd.Timestamp,
    target_month: pd.Timestamp,
    config: dict,
    grid_split_by_id: dict[int, str],
) -> tuple[str | None, str | None, str | None, str | None]:
    raw_grid_split = grid_split_by_id.get(int(grid_id))
    if raw_grid_split == "train":
        grid_split = "train_grid"
    elif raw_grid_split == "test":
        grid_split = "test_grid"
    else:
        grid_split = None

    time_split = _time_split_label(start_month, target_month, config)
    if grid_split is None or time_split is None:
        return None, grid_split, time_split, None

    if grid_split == "train_grid" and time_split == "train_years":
        return "train", grid_split, time_split, "train"
    if grid_split == "train_grid" and time_split == "validation_years":
        return "validation", grid_split, time_split, "validation"
    if grid_split == "train_grid" and time_split == "test_years":
        return "test_temporal", grid_split, time_split, "temporal_generalization"
    if grid_split == "test_grid" and time_split == "train_years":
        return "test_spatial", grid_split, time_split, "spatial_generalization"

    # Deliberately leave test grids in validation/test years out of the main split.
    # This keeps the requested total test set as test_temporal + test_spatial.
    return None, grid_split, time_split, "held_out_unused"


def _forcing_summary(
    dynamic: np.ndarray,
    feature_names: list[str],
    config: dict,
    start_idx: int,
    horizon: int,
    use_log1p_pumping: bool,
) -> dict[str, float]:
    future = dynamic[start_idx + 1 : start_idx + 1 + horizon]
    swb = future[:, feature_names.index("monthly_sum_net_infiltration")]
    pump_raw_name = pumping_raw_dynamic_feature_name(config, for_log_input=False)
    pumping_raw = future[:, feature_names.index(pump_raw_name)]

    out = {
        "sum_net_infiltration_t_to_tH": float(np.sum(swb)),
        "sum_pumping_t_to_tH": float(np.sum(pumping_raw)),
    }
    if use_log1p_pumping:
        pump_log_source_name = pumping_raw_dynamic_feature_name(config, for_log_input=True)
        pumping_for_log = future[:, feature_names.index(pump_log_source_name)]
        out["log1p_sum_pumping_t_to_tH"] = float(np.log1p(max(float(np.sum(pumping_for_log)), 0.0)))
    else:
        out["log1p_sum_pumping_t_to_tH"] = float(out["sum_pumping_t_to_tH"])
    return out


def _forcing_sequence_feature_names(config: dict) -> list[str]:
    explicit_cols = config.get("features", {}).get("forcing_sequence_columns")
    if explicit_cols is not None:
        cols = []
        seen: set[str] = set()
        for value in explicit_cols:
            col = str(value).strip()
            if not col or col in seen:
                continue
            cols.append(col)
            seen.add(col)
        if cols:
            return cols
    return [
        "monthly_sum_net_infiltration",
        pumping_log_dynamic_feature_name(config)
        if bool(config["features"].get("use_log1p_pumping", False))
        else pumping_raw_dynamic_feature_name(config, for_log_input=False),
    ]


def _filter_eval_quality(sample_table: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    quality_cfg = config.get("data", {}).get("split_quality_filter", {})
    if not bool(quality_cfg.get("enabled", False)):
        return sample_table, {}

    min_days = int(quality_cfg.get("min_observation_days", 0))
    splits = {str(split) for split in quality_cfg.get("apply_to_splits", [])}
    if min_days <= 0 or not splits:
        return sample_table, {}

    if "n_days_observed_t" not in sample_table.columns or "n_days_observed_target" not in sample_table.columns:
        return sample_table, {}

    frame = sample_table.copy()
    target_split = frame["split"].astype(str).isin(splits)
    low_quality = (
        target_split
        & (
            (frame["n_days_observed_t"].astype(float) < float(min_days))
            | (frame["n_days_observed_target"].astype(float) < float(min_days))
        )
    )
    removed = frame.loc[low_quality].copy()
    filtered = frame.loc[~low_quality].reset_index(drop=True)

    metadata = {
        "enabled": True,
        "min_observation_days": min_days,
        "apply_to_splits": sorted(splits),
        "removed_count": int(len(removed)),
        "kept_count": int(len(filtered)),
        "removed_by_split": {
            split: int(count) for split, count in removed["split"].value_counts().sort_index().items()
        },
    }
    return filtered, metadata


def build_sample_table(config: dict) -> SampleBuildResult:
    _ensure_preprocessed(config)
    paths = processed_paths(config)
    wtd = pd.read_csv(paths["wtd_monthly_csv"], parse_dates=["month_start"])
    month_index = load_month_index(config)
    dynamic_cache = load_or_build_dynamic_cache(config)
    feature_names = list(dynamic_cache["feature_names"])
    dynamic = np.asarray(dynamic_cache["data"], dtype=np.float32)
    grid_to_idx = dynamic_cache["grid_id_to_index"]
    month_to_idx = dynamic_cache["month_label_to_index"]

    lookback = int(config["data"]["lookback_months"])
    horizons = [int(v) for v in config["data"]["horizons"]]
    scenario_mode = bool(config["data"].get("scenario_mode", True))
    use_log1p_pumping = bool(config["features"].get("use_log1p_pumping", False))
    use_month_sin_cos = bool(config["features"].get("use_month_sin_cos", True))
    use_spatial_split = _uses_spatial_grid_split(config)
    use_spatiotemporal_split = _uses_spatiotemporal_split(config)
    grid_split_by_id: dict[int, str] = {}
    spatial_split_metadata: dict | None = None
    if use_spatial_split:
        grid_split_by_id, spatial_split_metadata = _build_spatial_grid_split(config, wtd, paths)

    month_index = month_index.set_index("month_label")
    wtd = wtd.copy()
    wtd["month_label"] = wtd["month_start"].dt.strftime("%Y-%m")
    wtd["month_idx"] = wtd["month_label"].map(month_to_idx)
    wtd["month_ord"] = wtd["month_start"].dt.year * 12 + wtd["month_start"].dt.month

    records: list[dict] = []
    omitted_spatiotemporal_counts: dict[str, int] = {}
    for grid_id, group in wtd.groupby("grid_id", sort=False):
        group = group.sort_values("month_start").reset_index(drop=True)
        month_row = {int(row.month_ord): row for row in group.itertuples(index=False)}
        grid_idx = int(grid_to_idx[int(grid_id)])
        grid_dynamic = dynamic[grid_idx]

        for row in group.itertuples(index=False):
            start_month = pd.Timestamp(row.month_start)
            start_label = str(row.month_label)
            start_idx = int(row.month_idx)

            start_sin, start_cos = np.sin(0.0), np.cos(0.0)
            if use_month_sin_cos:
                angle = 2.0 * np.pi * (start_month.month - 1) / 12.0
                start_sin = float(np.sin(angle))
                start_cos = float(np.cos(angle))

            for horizon in horizons:
                target_month = (start_month.to_period("M") + horizon).to_timestamp()
                target_ord = int(target_month.year) * 12 + int(target_month.month)
                target_row = month_row.get(target_ord)
                if target_row is None:
                    continue

                target_idx = month_to_idx.get(target_month.strftime("%Y-%m"))
                if target_idx is None:
                    continue

                if scenario_mode:
                    if start_idx + horizon >= grid_dynamic.shape[0]:
                        continue
                    future_window = grid_dynamic[start_idx + 1 : start_idx + 1 + horizon]
                    pump_required_name = pumping_raw_dynamic_feature_name(
                        config,
                        for_log_input=use_log1p_pumping,
                    )
                    required_future_cols = [
                        feature_names.index("monthly_sum_net_infiltration"),
                        feature_names.index(pump_required_name),
                    ]
                    if not _has_finite_window(future_window[:, required_future_cols]):
                        continue
                    forcing_summary = _forcing_summary(
                        grid_dynamic,
                        feature_names,
                        config,
                        start_idx,
                        horizon,
                        use_log1p_pumping=use_log1p_pumping,
                    )
                else:
                    forcing_summary = {
                        "sum_net_infiltration_t_to_tH": 0.0,
                        "sum_pumping_t_to_tH": 0.0,
                        "log1p_sum_pumping_t_to_tH": 0.0,
                    }

                delta_wtd = float(target_row.wtd_m_bls - row.wtd_m_bls)
                delta_h = float(-delta_wtd)
                if use_spatiotemporal_split:
                    split, grid_split, time_split, eval_case = _assign_spatiotemporal_split(
                        int(grid_id),
                        start_month,
                        target_month,
                        config,
                        grid_split_by_id,
                    )
                else:
                    split = assign_split(start_month, target_month, config)
                    grid_split = grid_split_by_id.get(int(grid_id)) if use_spatial_split else None
                    time_split = _time_split_label(start_month, target_month, config)
                    eval_case = split
                if split is None:
                    if use_spatiotemporal_split:
                        key = f"{grid_split or 'unknown_grid'}__{time_split or 'unknown_time'}__{eval_case or 'unassigned'}"
                        omitted_spatiotemporal_counts[key] = omitted_spatiotemporal_counts.get(key, 0) + 1
                    continue

                record = {
                    "grid_id": int(grid_id),
                    "row": int(row.row),
                    "col": int(row.col),
                    "cell_id": int(row.cell_id),
                    "x": float(row.x),
                    "y": float(row.y),
                    "start_month": start_month,
                    "start_month_label": start_label,
                    "start_month_idx": start_idx,
                    "target_month": target_month,
                    "target_month_label": target_month.strftime("%Y-%m"),
                    "target_month_idx": int(target_idx),
                    "start_year": int(start_month.year),
                    "target_year": int(target_month.year),
                    "h_months": int(horizon),
                    "wtd_t_m_bls": float(row.wtd_m_bls),
                    "wtd_t_plus_h_m_bls": float(target_row.wtd_m_bls),
                    "delta_wtd_m": delta_wtd,
                    "delta_h_m": delta_h,
                    "n_days_observed_t": int(row.n_days_observed),
                    "n_days_observed_target": int(target_row.n_days_observed),
                    "n_unique_sites_t": int(row.n_unique_sites),
                    "n_unique_sites_target": int(target_row.n_unique_sites),
                    "n_site_days_t": int(row.n_site_days),
                    "n_site_days_target": int(target_row.n_site_days),
                    "mean_n_wells_per_day_t": float(row.mean_n_wells_per_day),
                    "mean_n_wells_per_day_target": float(target_row.mean_n_wells_per_day),
                    "start_month_sin": float(start_sin),
                    "start_month_cos": float(start_cos),
                    "split": split,
                    "grid_split": grid_split,
                    "time_split": time_split,
                    "eval_case": eval_case,
                }
                record.update(forcing_summary)
                records.append(record)

    sample_table = pd.DataFrame(records).sort_values(
        ["split", "target_month", "grid_id", "h_months"]
    ).reset_index(drop=True)
    sample_table, quality_filter_metadata = _filter_eval_quality(sample_table, config)
    sample_table.insert(0, "sample_id", np.arange(len(sample_table), dtype=np.int64))
    sample_table.to_csv(paths["sample_table_csv"], index=False, lineterminator="\n")
    for old_split_path in paths["root"].glob("sample_table_*.csv"):
        old_split_path.unlink(missing_ok=True)
    for split, split_frame in sample_table.groupby("split", sort=True):
        split_frame.to_csv(paths["root"] / f"sample_table_{split}.csv", index=False, lineterminator="\n")
    if use_spatiotemporal_split:
        test_parts = sample_table[sample_table["split"].isin(["test_temporal", "test_spatial"])].copy()
        if not test_parts.empty:
            test_parts["split"] = "test"
            test_parts.to_csv(paths["root"] / "sample_table_test.csv", index=False, lineterminator="\n")

    metadata = {
        "target_type": config["data"]["target_type"],
        "lookback_months": lookback,
        "horizons": horizons,
        "scenario_mode": scenario_mode,
        "split_by": config["data"]["split_by"],
        "dynamic_sequence_features": [],
        "forcing_summary_features": [
            "sum_net_infiltration_t_to_tH",
            "sum_pumping_t_to_tH",
            "log1p_sum_pumping_t_to_tH",
        ],
        "pumping_log_input_mode": pumping_log_input_mode(config),
        "forcing_sequence_features": _forcing_sequence_feature_names(config),
        "sample_count": int(len(sample_table)),
        "split_counts": {
            split: int(count) for split, count in sample_table["split"].value_counts().sort_index().items()
        },
        "notes": {
            "delta_wtd_m": "WTD(t+H) - WTD(t), where WTD is depth to water below land surface.",
            "delta_h_m": "-delta_wtd_m; positive values indicate groundwater-level rise.",
            "scenario_mode": "When true, future monthly forcing summaries from t+1 to t+H are provided as scenario-condition inputs.",
            "forcing_sequence": "Future monthly forcing sequences from t+1 to t+H are derived on the fly from the dynamic cache during model training/evaluation.",
            "pumping_log_input_mode": "Controls whether the log-transformed pumping feature is computed from raw m^3/km^2/month or equivalent mm/month.",
            "lookback_months": "Route B uses no past forcing history in the model input. This field is kept only for bookkeeping and should be 0.",
        },
    }
    if spatial_split_metadata is not None:
        metadata["spatial_split"] = spatial_split_metadata
        if use_spatiotemporal_split:
            metadata["omitted_spatiotemporal_counts"] = {
                key: int(value) for key, value in sorted(omitted_spatiotemporal_counts.items())
            }
            metadata["notes"][
                "spatiotemporal_grid_time"
            ] = (
                "train=train grids + train years; validation=train grids + validation years; "
                "test_temporal=train grids + test years; test_spatial=test grids + train years. "
                "sample_table_test.csv is the union of test_temporal and test_spatial."
            )
        else:
            metadata["notes"][
                "spatial_random_grid"
            ] = "Samples are split by grid_id, so every observed grid belongs to exactly one split across all months."
    if quality_filter_metadata:
        metadata["split_quality_filter"] = quality_filter_metadata
        filtered_splits = {str(split) for split in quality_filter_metadata.get("apply_to_splits", [])}
        filtered_split_list = ", ".join(sorted(filtered_splits))
        if "train" in filtered_splits:
            metadata["notes"][
                "split_quality_filter"
            ] = (
                "The configured quality filter removes low-quality samples from the selected splits "
                f"({filtered_split_list}), including training samples when listed in apply_to_splits."
            )
        else:
            metadata["notes"][
                "split_quality_filter"
            ] = (
                "The configured quality filter removes low-quality samples only from the selected evaluation splits "
                f"({filtered_split_list}). Training samples are left intact so low-density observations can still "
                "contribute with loss weighting."
            )
    save_json(metadata, paths["sample_metadata_json"])
    return SampleBuildResult(sample_table=sample_table, metadata=metadata)
