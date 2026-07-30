from __future__ import annotations

import json
from pathlib import Path

import earthaccess
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_CELL_PATH = REPO_ROOT / "data" / "2 well WTD" / "active_cell_lookup.csv"
OUT_DIR = REPO_ROOT / "data" / "10 precipitation" / "daymet_prcp_monthly_v4r1_mrva"
RAW_NC_DIR = OUT_DIR / "_raw_nc_tmp"

YEARS = list(range(2011, 2024))  # 2011-2023 inclusive

# Daymet V4R1 monthly dataset in Earthdata CMR
SHORT_NAME = "Daymet_Monthly_V4R1_2131"
VERSION = "4.1"
TARGET_FILE_TEMPLATE = "daymet_v4_prcp_monttl_na_{year}.nc"

# Daymet Lambert Conformal Conic projection from official user guide.
DAYMET_LCC = CRS.from_proj4(
    "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)


def _nearest_index_ascending(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(axis, values)
    idx = np.clip(idx, 1, len(axis) - 1)
    left = axis[idx - 1]
    right = axis[idx]
    take_left = np.abs(values - left) <= np.abs(values - right)
    return idx - take_left.astype(np.int64)


def _build_index_map(daymet_x: np.ndarray, daymet_y: np.ndarray, cells: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:5070", DAYMET_LCC, always_xy=True)
    x_daymet, y_daymet = transformer.transform(
        cells["x_center"].to_numpy(dtype=float),
        cells["y_center"].to_numpy(dtype=float),
    )

    ix = _nearest_index_ascending(daymet_x, x_daymet)
    # y axis in Daymet files is descending (north to south), so reverse for search.
    y_rev = daymet_y[::-1]
    iy_rev = _nearest_index_ascending(y_rev, y_daymet)
    iy = (len(daymet_y) - 1) - iy_rev
    return iy.astype(np.int64), ix.astype(np.int64)


def _pick_target_granule(granules: list, year: int):
    target_name = TARGET_FILE_TEMPLATE.format(year=year)
    for g in granules:
        granule_ur = g["umm"].get("GranuleUR", "")
        if target_name in granule_ur:
            return g
    raise FileNotFoundError(f"Could not find granule for {target_name}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_NC_DIR.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(
        ACTIVE_CELL_PATH,
        usecols=["node_id", "row", "col", "x_center", "y_center"],
    ).sort_values("node_id")
    n_grid = len(cells)
    if cells["node_id"].iloc[0] != 0 or cells["node_id"].iloc[-1] != n_grid - 1:
        raise ValueError("node_id is expected to be contiguous from 0 to n_grid-1.")

    print(f"MRVA active cells: {n_grid}")
    earthaccess.login(strategy="netrc")

    all_month_labels: list[str] = []
    all_blocks: list[np.ndarray] = []
    iy: np.ndarray | None = None
    ix: np.ndarray | None = None

    for year in YEARS:
        print(f"\n=== Year {year} ===")
        granules = earthaccess.search_data(
            short_name=SHORT_NAME,
            version=VERSION,
            temporal=(f"{year}-01-01", f"{year}-12-31"),
        )
        granule = _pick_target_granule(granules, year)

        downloaded = earthaccess.download([granule], RAW_NC_DIR)
        if not downloaded:
            raise RuntimeError(f"Download failed for year {year}")
        nc_path = Path(downloaded[0])
        print(f"Downloaded: {nc_path.name}")

        with xr.open_dataset(nc_path, engine="netcdf4") as ds:
            if iy is None or ix is None:
                iy, ix = _build_index_map(
                    ds["x"].to_numpy().astype(np.float64, copy=False),
                    ds["y"].to_numpy().astype(np.float64, copy=False),
                    cells,
                )
                print("Built MRVA->Daymet nearest-neighbor index map.")

            point_prcp = ds["prcp"].isel(
                y=xr.DataArray(iy, dims="points"),
                x=xr.DataArray(ix, dims="points"),
            ).to_numpy().astype(np.float32, copy=False)

            if point_prcp.shape[0] != 12:
                raise ValueError(f"Expected 12 months for {year}, got {point_prcp.shape[0]}")

            month_labels = pd.to_datetime(ds["time"].to_numpy()).strftime("%Y-%m").tolist()
            all_month_labels.extend(month_labels)
            all_blocks.append(point_prcp)

        # keep disk usage small: remove annual CONUS file after extracting MRVA points.
        nc_path.unlink(missing_ok=True)
        print(f"Extracted and removed raw file for {year}.")

    matrix = np.vstack(all_blocks).astype(np.float32, copy=False)
    if matrix.shape != (len(all_month_labels), n_grid):
        raise ValueError(f"Unexpected matrix shape: {matrix.shape}")

    out_matrix = OUT_DIR / "daymet_prcp_monthly_mrva_1km.npy"
    out_month_idx = OUT_DIR / "month_index.csv"
    out_grid = OUT_DIR / "grid_lookup.csv"
    out_summary = OUT_DIR / "summary.json"

    np.save(out_matrix, matrix)
    pd.DataFrame(
        {
            "month_idx": np.arange(len(all_month_labels), dtype=np.int32),
            "month_label": all_month_labels,
        }
    ).to_csv(out_month_idx, index=False)
    cells.rename(columns={"node_id": "grid_id"}).to_csv(out_grid, index=False)

    summary = {
        "source_short_name": SHORT_NAME,
        "source_version": VERSION,
        "years": YEARS,
        "variable": "prcp",
        "unit": "mm",
        "description": "Daymet monthly total precipitation on MRVA 1 km active cells (nearest-neighbor from Daymet NA grid).",
        "n_months": int(matrix.shape[0]),
        "n_grids": int(matrix.shape[1]),
        "matrix_min_mm": float(np.nanmin(matrix)),
        "matrix_mean_mm": float(np.nanmean(matrix)),
        "matrix_max_mm": float(np.nanmax(matrix)),
        "outputs": {
            "matrix_npy": str(out_matrix),
            "month_index_csv": str(out_month_idx),
            "grid_lookup_csv": str(out_grid),
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Saved: {out_matrix}")
    print(f"Saved: {out_month_idx}")
    print(f"Saved: {out_grid}")
    print(f"Saved: {out_summary}")


if __name__ == "__main__":
    main()
