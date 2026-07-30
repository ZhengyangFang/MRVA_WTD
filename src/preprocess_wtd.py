from __future__ import annotations

import pandas as pd

from .utils import processed_paths, save_json


def preprocess_wtd(config: dict) -> tuple[pd.DataFrame, dict]:
    src = config["paths"]["wtd_monthly_path"]
    paths = processed_paths(config)
    frame = pd.read_csv(src, parse_dates=["month_start"]).copy()
    frame = frame.rename(
        columns={
            "node_id": "grid_id",
            "x_center": "x",
            "y_center": "y",
            "mean_wtd_m": "wtd_m_bls",
        }
    )
    keep_cols = [
        "grid_id",
        "row",
        "col",
        "cell_id",
        "x",
        "y",
        "month_start",
        "month_label",
        "wtd_m_bls",
        "std_daily_wtd_m",
        "min_daily_wtd_m",
        "max_daily_wtd_m",
        "n_days_observed",
        "mean_n_wells_per_day",
        "max_n_wells_per_day",
        "source_names",
        "n_unique_sites",
        "n_site_days",
    ]
    frame = frame[keep_cols].sort_values(["grid_id", "month_start"]).reset_index(drop=True)
    frame["month_ord"] = frame["month_start"].dt.year * 12 + frame["month_start"].dt.month
    frame.to_csv(paths["wtd_monthly_csv"], index=False, lineterminator="\n")

    meta = {
        "source_path": src,
        "unit": "m_below_land_surface",
        "notes": "Monthly WTD is based on field measurements + DV + IV daily mean and remains an observed label product, not a simulated head product.",
    }
    save_json(meta, paths["wtd_metadata_json"])
    return frame, meta
