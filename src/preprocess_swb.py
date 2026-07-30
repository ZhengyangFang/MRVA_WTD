from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .utils import (
    load_active_grid,
    month_index_frame,
    processed_paths,
    save_json,
)


INCH_TO_MM = 25.4


def _rolling_max7(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 7:
        return np.nansum(values, axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=7, axis=0)
    rolling_sum = np.nansum(windows, axis=-1)
    return np.nanmax(rolling_sum, axis=0)


def _safe_nanmean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.nansum(values, axis=axis)
    out = np.full(total.shape, np.nan, dtype=np.float32)
    mask = count > 0
    out[mask] = (total[mask] / count[mask]).astype(np.float32)
    return out


def preprocess_swb(config: dict) -> tuple[pd.DataFrame, dict]:
    paths = processed_paths(config)
    active = load_active_grid(config)
    src_rows = None
    cols = active["col"].to_numpy(dtype=int)
    rows = active["row"].to_numpy(dtype=int)

    month_index = month_index_frame(config)
    records = []
    unit_in = str(config["features"].get("swb_input_unit", "in"))
    unit_out = str(config["features"].get("swb_output_unit", unit_in))
    convert_to_mm = unit_in == "in" and unit_out == "mm"
    scale = INCH_TO_MM if convert_to_mm else 1.0
    use_month_mean = bool(config["features"].get("generate_monthly_mean_net_infiltration", True))
    use_max7 = bool(config["features"].get("generate_monthly_max_7day_net_infiltration", True))

    swb_dir = Path(config["paths"]["swb_dir"])
    for tif in sorted(swb_dir.glob("swb_net_infiltration_*_daily_1km_masked.tif")):
        with rasterio.open(tif) as ds:
            if src_rows is None:
                src_rows = (ds.height - 1) - rows
            nodata = ds.nodata
            stack = ds.read().astype(np.float32)
            daily = stack[:, src_rows, cols]
            if nodata is not None:
                daily[daily == nodata] = np.nan

            descs = [pd.Timestamp(desc) for desc in ds.descriptions]
            year_month = pd.Series(descs).dt.strftime("%Y-%m")
            unique_months = pd.unique(year_month)

            for month_label in unique_months:
                mask = year_month == month_label
                values = daily[mask.to_numpy()]
                row = pd.DataFrame(
                    {
                        "grid_id": active["grid_id"].to_numpy(dtype=np.int64),
                        "month_label": month_label,
                        "monthly_sum_net_infiltration": np.nansum(values, axis=0) * scale,
                    }
                )
                if use_month_mean:
                    row["monthly_mean_net_infiltration"] = _safe_nanmean(values, axis=0) * scale
                if use_max7:
                    row["monthly_max_7day_net_infiltration"] = _rolling_max7(values) * scale
                records.append(row)

    out = pd.concat(records, ignore_index=True)
    out["month_start"] = pd.to_datetime(out["month_label"] + "-01")
    out = out.merge(month_index[["month_label", "month_ord", "month_idx"]], on="month_label", how="left")
    out = out.sort_values(["grid_id", "month_start"]).reset_index(drop=True)
    out.to_csv(paths["swb_monthly_csv"], index=False, lineterminator="\n")

    meta = {
        "source_dir": str(swb_dir),
        "input_unit": unit_in,
        "output_unit": unit_out,
        "columns": [c for c in out.columns if c not in {"grid_id", "month_label", "month_start", "month_ord", "month_idx"}],
        "notes": {
            "monthly_sum_net_infiltration": "SWB-derived recharge-related forcing aggregated from daily net infiltration.",
            "monthly_mean_net_infiltration": "Calendar-month mean of daily net infiltration.",
            "monthly_max_7day_net_infiltration": "Maximum 7-day rolling sum within each calendar month.",
        },
    }
    save_json(meta, paths["swb_metadata_json"])
    return out, meta
