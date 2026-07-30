from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
GRACE_DIR = REPO_ROOT / "data" / "7 GRACE"
RAW_DIR = REPO_ROOT / "data_raw" / "7 GRACE"

SOURCE_URL = "https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"
SOURCE_FILE = RAW_DIR / "CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"
WEIGHTS_PATH = GRACE_DIR / "grace_mrva_gridcell_weights.csv"

START_MONTH = "2011-01"
END_MONTH = "2023-12"

OUT_PREFIX = "grace_mrva_2011_2023"
OUT_REGIONAL_VALID = GRACE_DIR / f"{OUT_PREFIX}_timeseries.csv"
OUT_REGIONAL_FULL = GRACE_DIR / f"{OUT_PREFIX}_timeseries_full156.csv"
OUT_CELL_MONTHLY = GRACE_DIR / f"{OUT_PREFIX}_cell_monthly.csv"
OUT_SUBSET_NC = GRACE_DIR / "grace_csr_rl0603_mrva_2011_2023_subset.nc"
OUT_SUMMARY = GRACE_DIR / f"{OUT_PREFIX}_summary.json"


def _download_source() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    GRACE_DIR.mkdir(parents=True, exist_ok=True)

    if SOURCE_FILE.exists() and SOURCE_FILE.stat().st_size > 1_000_000:
        print(f"Source already exists: {SOURCE_FILE}")
        return

    print(f"Downloading CSR GRACE/GRACE-FO RL06.3 Mascon file:\n{SOURCE_URL}")
    try:
        head = requests.head(SOURCE_URL, timeout=30, allow_redirects=True)
        head.raise_for_status()
        verify = True
    except requests.exceptions.SSLError:
        verify = False
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
        head = requests.head(SOURCE_URL, timeout=30, allow_redirects=True, verify=False)
        head.raise_for_status()

    expected_size = int(head.headers.get("content-length", "0"))
    tmp_path = SOURCE_FILE.with_suffix(".nc.part")
    with requests.get(SOURCE_URL, stream=True, timeout=120, verify=verify) as response:
        response.raise_for_status()
        downloaded = 0
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if expected_size:
                    print(
                        f"\rDownloaded {downloaded / 1024**2:.1f} / {expected_size / 1024**2:.1f} MB",
                        end="",
                        flush=True,
                    )
        print()

    tmp_path.replace(SOURCE_FILE)
    print(f"Saved source: {SOURCE_FILE}")


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    weighted = np.where(valid, values, 0.0) * weights[None, :]
    denom = np.where(valid, weights[None, :], 0.0).sum(axis=1)
    out = weighted.sum(axis=1) / denom
    out[denom == 0] = np.nan
    return out


def _decode_grace_time(time_da: xr.DataArray) -> pd.DatetimeIndex:
    values = time_da.to_numpy()
    if np.issubdtype(values.dtype, np.datetime64):
        return pd.to_datetime(values)

    units = time_da.attrs.get("units") or time_da.attrs.get("Units")
    if not units or "since" not in units:
        raise ValueError(f"Cannot decode GRACE time coordinate with attrs: {time_da.attrs}")

    base_text = units.split("since", 1)[1].strip().replace("Z", "")
    base = pd.Timestamp(base_text).tz_localize(None)
    return pd.DatetimeIndex(base + pd.to_timedelta(values.astype(float), unit="D"))


def main() -> None:
    _download_source()

    weights = pd.read_csv(WEIGHTS_PATH).copy()
    weights["cell_key"] = weights["lat_idx"].astype(int).astype(str) + "_" + weights["lon_idx"].astype(int).astype(str)
    weights = weights.sort_values(["lat_idx", "lon_idx"]).reset_index(drop=True)

    start_ts = pd.Timestamp(f"{START_MONTH}-01")
    end_ts = pd.Timestamp(f"{END_MONTH}-28") + pd.offsets.MonthEnd(0)
    expected_months = pd.period_range(START_MONTH, END_MONTH, freq="M").astype(str)

    lat_min = int(weights["lat_idx"].min())
    lat_max = int(weights["lat_idx"].max())
    lon_min = int(weights["lon_idx"].min())
    lon_max = int(weights["lon_idx"].max())

    with xr.open_dataset(SOURCE_FILE, engine="netcdf4", decode_times=False) as ds:
        time = _decode_grace_time(ds["time"])
        time_mask = (time >= start_ts) & (time <= end_ts)
        if not np.any(time_mask):
            raise ValueError("No GRACE records found for requested time range.")

        subset = ds[["lwe_thickness"]].isel(
            time=np.where(time_mask)[0],
            lat=slice(lat_min, lat_max + 1),
            lon=slice(lon_min, lon_max + 1),
        )
        subset.to_netcdf(OUT_SUBSET_NC, engine="netcdf4")

        selected_time = time[time_mask]
        values_rect = subset["lwe_thickness"].to_numpy().astype(np.float32, copy=False)
        lat_local = weights["lat_idx"].to_numpy(dtype=np.int64) - lat_min
        lon_local = weights["lon_idx"].to_numpy(dtype=np.int64) - lon_min
        cell_values = values_rect[:, lat_local, lon_local]

        source_attrs = dict(ds.attrs)
        units = getattr(ds["lwe_thickness"], "units", "cm")

    cell_df = pd.DataFrame(cell_values, columns=weights["cell_key"].tolist())
    cell_df.insert(0, "month", selected_time.strftime("%Y-%m"))
    cell_month = cell_df.groupby("month", sort=True).mean(numeric_only=True)

    regional_values = _weighted_mean(cell_month.to_numpy(dtype=np.float64), weights["weight"].to_numpy(dtype=np.float64))
    regional_valid = pd.DataFrame(
        {
            "month": cell_month.index,
            "month_start": pd.to_datetime(cell_month.index + "-01").strftime("%Y-%m-%d"),
            "grace_lwe_thickness_cm": regional_values,
        }
    )
    regional_valid["grace_lwe_thickness_mm"] = regional_valid["grace_lwe_thickness_cm"] * 10.0
    regional_valid["grace_lwe_thickness_m"] = regional_valid["grace_lwe_thickness_cm"] / 100.0

    full = pd.DataFrame({"month": expected_months})
    full["month_start"] = pd.to_datetime(full["month"] + "-01").dt.strftime("%Y-%m-%d")
    full = full.merge(
        regional_valid[["month", "grace_lwe_thickness_cm", "grace_lwe_thickness_mm", "grace_lwe_thickness_m"]],
        on="month",
        how="left",
    )
    full["has_grace_solution"] = full["grace_lwe_thickness_cm"].notna()
    full = full[
        [
            "month",
            "month_start",
            "has_grace_solution",
            "grace_lwe_thickness_cm",
            "grace_lwe_thickness_mm",
            "grace_lwe_thickness_m",
        ]
    ]

    cell_long = (
        cell_month.reset_index()
        .melt(id_vars="month", var_name="cell_key", value_name="grace_cm")
        .rename(columns={"month": "month_label"})
        .merge(
            weights[["cell_key", "lat_idx", "lon_idx", "lat_deg", "lon_deg_0_360", "lon_deg_-180_180"]],
            on="cell_key",
            how="left",
        )
    )

    regional_valid.to_csv(OUT_REGIONAL_VALID, index=False)
    full.to_csv(OUT_REGIONAL_FULL, index=False)
    cell_long.to_csv(OUT_CELL_MONTHLY, index=False)

    missing_months = full.loc[~full["has_grace_solution"], "month"].tolist()
    summary = {
        "source_url": SOURCE_URL,
        "source_file": str(SOURCE_FILE),
        "source_doi": source_attrs.get("id"),
        "source_title": source_attrs.get("title"),
        "source_product_version": source_attrs.get("product_version"),
        "time_range_requested": {"start": f"{START_MONTH}-01", "end": full["month_start"].iloc[-1]},
        "time_records_selected": int(len(selected_time)),
        "calendar_month_records_after_merge": int(len(regional_valid)),
        "months_expected_in_range": int(len(full)),
        "months_missing_in_grace": int(len(missing_months)),
        "missing_months": missing_months,
        "time_selected_first": selected_time.min().strftime("%Y-%m-%d"),
        "time_selected_last": selected_time.max().strftime("%Y-%m-%d"),
        "grace_units": units,
        "mrva_unique_grace_cells": int(len(weights)),
        "subset_lat_index_range": [lat_min, lat_max],
        "subset_lon_index_range": [lon_min, lon_max],
        "subset_lat_deg_range": [float(weights["lat_deg"].min()), float(weights["lat_deg"].max())],
        "subset_lon_deg_0_360_range": [float(weights["lon_deg_0_360"].min()), float(weights["lon_deg_0_360"].max())],
        "subset_lon_deg_-180_180_range": [
            float(weights["lon_deg_-180_180"].min()),
            float(weights["lon_deg_-180_180"].max()),
        ],
        "outputs": {
            "regional_timeseries_csv": str(OUT_REGIONAL_VALID),
            "regional_timeseries_full156_csv": str(OUT_REGIONAL_FULL),
            "cell_monthly_csv": str(OUT_CELL_MONTHLY),
            "gridcell_weights_csv": str(WEIGHTS_PATH),
            "subset_nc": str(OUT_SUBSET_NC),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Saved: {OUT_REGIONAL_VALID}")
    print(f"Saved: {OUT_REGIONAL_FULL}")
    print(f"Saved: {OUT_CELL_MONTHLY}")
    print(f"Saved: {OUT_SUBSET_NC}")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"GRACE valid months: {int(full['has_grace_solution'].sum())} / {len(full)}")
    print("Missing months:", ", ".join(missing_months))


if __name__ == "__main__":
    main()
