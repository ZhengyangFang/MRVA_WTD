from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .utils import load_active_grid, month_index_frame, processed_paths, save_json


MONTH_NAME_TO_NUM = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def preprocess_aiwum(config: dict) -> tuple[pd.DataFrame, dict]:
    paths = processed_paths(config)
    active = load_active_grid(config)
    rows = active["row"].to_numpy(dtype=int)
    cols = active["col"].to_numpy(dtype=int)
    aiwum_dir = Path(config["paths"]["aiwum_dir"])
    month_index = month_index_frame(config)
    pattern = re.compile(r"AIWUM[12]_1km_m3_(\d{4})_([A-Za-z]{3})\.tif$")
    missing_as_zero = bool(config["features"].get("pumping_missing_as_zero", True))

    records = []
    src_rows = None
    for year_dir in sorted(aiwum_dir.glob("20*")):
        if not year_dir.is_dir():
            continue
        for tif in sorted(year_dir.glob("*.tif")):
            match = pattern.match(tif.name)
            if not match:
                continue
            year = int(match.group(1))
            month = MONTH_NAME_TO_NUM[match.group(2)]
            month_label = f"{year:04d}-{month:02d}"
            with rasterio.open(tif) as ds:
                if src_rows is None:
                    src_rows = (ds.height - 1) - rows
                nodata = ds.nodata
                arr = ds.read(1).astype(np.float32)
                values = arr[src_rows, cols]
                if nodata is not None:
                    values[values == nodata] = np.nan
            if missing_as_zero:
                values = np.nan_to_num(values, nan=0.0)

            records.append(
                pd.DataFrame(
                    {
                        "grid_id": active["grid_id"].to_numpy(dtype=np.int64),
                        "month_label": month_label,
                        "monthly_pumping": values.astype(np.float32),
                    }
                )
            )

    out = pd.concat(records, ignore_index=True)
    out["month_start"] = pd.to_datetime(out["month_label"] + "-01")
    out = out.merge(month_index[["month_label", "month_ord", "month_idx"]], on="month_label", how="left")
    out["monthly_pumping_mm"] = out["monthly_pumping"].to_numpy(dtype=np.float32) / 1000.0
    out["monthly_pumping_log1p"] = np.log1p(np.clip(out["monthly_pumping"].to_numpy(dtype=np.float32), a_min=0.0, a_max=None))
    out["monthly_pumping_mm_log1p"] = np.log1p(
        np.clip(out["monthly_pumping_mm"].to_numpy(dtype=np.float32), a_min=0.0, a_max=None)
    )
    out = out.sort_values(["grid_id", "month_start"]).reset_index(drop=True)
    out.to_csv(paths["aiwum_monthly_csv"], index=False, lineterminator="\n")

    meta = {
        "source_dir": str(aiwum_dir),
        "unit": "m3_per_km2_per_month",
        "equivalent_depth_unit": "mm_per_month",
        "columns": [
            "monthly_pumping",
            "monthly_pumping_mm",
            "monthly_pumping_log1p",
            "monthly_pumping_mm_log1p",
        ],
        "notes": {
            "monthly_pumping": "AIWUM monthly groundwater-use estimate aligned to the MRVA 1 km grid.",
            "monthly_pumping_mm": "Equivalent pumping depth on a 1 km^2 cell (1 mm = 1000 m^3 per km^2).",
            "monthly_pumping_log1p": "log1p-transformed monthly_pumping in original m^3/km^2/month units.",
            "monthly_pumping_mm_log1p": "log1p-transformed monthly_pumping_mm in equivalent mm/month units.",
        },
        "missing_as_zero": missing_as_zero,
    }
    save_json(meta, paths["aiwum_metadata_json"])
    return out, meta
