from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from mainline_reconstruction import MAINLINE_RECON_ROOT


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID_LOOKUP = MAINLINE_RECON_ROOT / "metadata" / "grid_lookup.csv"
DEFAULT_TOPO_DIR = REPO_ROOT / "data" / "5 topography"
TNM_ELEVATION_EXPORT_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)


def download_3dep_dem(bounds: tuple[float, float, float, float], width: int, height: int) -> tuple[np.ndarray, dict]:
    """Download a north-up 3DEP DEM image in EPSG:5070 for the requested grid extent."""
    left, bottom, right, top = bounds
    params = {
        "bbox": f"{left},{bottom},{right},{top}",
        "bboxSR": "5070",
        "imageSR": "5070",
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "F32",
        "noData": "-9999",
        "f": "image",
    }
    url = TNM_ELEVATION_EXPORT_URL + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "Codex-coordinate-audit/1.0"})
    with urlopen(request, timeout=180) as response:
        payload = response.read()

    with MemoryFile(payload) as memfile:
        with memfile.open() as dataset:
            dem = dataset.read(1).astype(np.float32)
            nodata = dataset.nodata
            if nodata is not None:
                dem = np.where(dem == nodata, np.nan, dem).astype(np.float32)
            metadata = {
                "content_type": response.headers.get("Content-Type"),
                "source_url": url,
                "downloaded_shape": list(dem.shape),
                "downloaded_crs": str(dataset.crs),
                "downloaded_bounds": [float(v) for v in dataset.bounds],
                "downloaded_transform": list(dataset.transform)[:6],
            }
    return dem, metadata


def slope_degrees_from_dem(dem: np.ndarray, cell_size_m: float = 1000.0) -> np.ndarray:
    """Compute slope in degrees from a north-up DEM."""
    filled = dem.astype(np.float64).copy()
    if np.isnan(filled).any():
        median = np.nanmedian(filled)
        filled = np.where(np.isfinite(filled), filled, median)
    grad_y, grad_x = np.gradient(filled, cell_size_m, cell_size_m)
    slope = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
    return slope.astype(np.float32)


def write_geotiff(path: Path, array: np.ndarray, bounds: tuple[float, float, float, float], nodata: float = -9999.0) -> None:
    left, bottom, right, top = bounds
    height, width = array.shape
    transform = from_origin(left, top, (right - left) / width, (top - bottom) / height)
    out = np.where(np.isfinite(array), array, nodata).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:5070",
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 3,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def backup_existing(topo_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = topo_dir / f"_backup_before_3dep_coordinate_fix_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name in ["topo_1km.csv", "elev_1km.tif", "slope_1km.tif", "topo_summary.json"]:
        src = topo_dir / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)
    return backup_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild MRVA 1 km topography from USGS 3DEP using grid coordinates.")
    parser.add_argument("--grid-lookup", type=Path, default=DEFAULT_GRID_LOOKUP)
    parser.add_argument("--topo-dir", type=Path, default=DEFAULT_TOPO_DIR)
    parser.add_argument("--no-backup", action="store_true", help="Do not create a timestamped backup of existing topo files.")
    args = parser.parse_args()

    grid_lookup = args.grid_lookup.resolve()
    topo_dir = args.topo_dir.resolve()
    topo_dir.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(grid_lookup)
    required = {"grid_id", "row", "col", "x", "y"}
    missing = sorted(required - set(grid.columns))
    if missing:
        raise ValueError(f"{grid_lookup} missing required columns: {missing}")

    grid = grid.sort_values("grid_id").reset_index(drop=True)
    n_rows = int(grid["row"].max()) + 1
    n_cols = int(grid["col"].max()) + 1
    cell_size = 1000.0
    bounds = (
        float(grid["x"].min() - cell_size / 2),
        float(grid["y"].min() - cell_size / 2),
        float(grid["x"].max() + cell_size / 2),
        float(grid["y"].max() + cell_size / 2),
    )

    if not args.no_backup:
        backup_dir = backup_existing(topo_dir)
        print(f"Backed up previous topography files to: {backup_dir}")
    else:
        backup_dir = None

    dem, source_meta = download_3dep_dem(bounds, width=n_cols, height=n_rows)
    if dem.shape != (n_rows, n_cols):
        raise ValueError(f"Downloaded DEM shape {dem.shape} != expected {(n_rows, n_cols)}")
    slope = slope_degrees_from_dem(dem, cell_size_m=cell_size)

    raster_row = (n_rows - 1 - grid["row"].to_numpy(dtype=np.int64))
    raster_col = grid["col"].to_numpy(dtype=np.int64)
    elev_values = dem[raster_row, raster_col]
    slope_values = slope[raster_row, raster_col]

    if not np.isfinite(elev_values).all():
        bad = int((~np.isfinite(elev_values)).sum())
        raise ValueError(f"Downloaded DEM has {bad} non-finite values at active grid cells.")

    topo_table = pd.DataFrame(
        {
            "grid_id": grid["grid_id"].to_numpy(dtype=np.int64),
            "mean_elevation_m": elev_values.astype(np.float32),
            "mean_slope_deg": slope_values.astype(np.float32),
        }
    )
    topo_csv = topo_dir / "topo_1km.csv"
    topo_table.to_csv(topo_csv, index=False)

    elev_masked = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    slope_masked = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    elev_masked[raster_row, raster_col] = elev_values.astype(np.float32)
    slope_masked[raster_row, raster_col] = slope_values.astype(np.float32)
    write_geotiff(topo_dir / "elev_1km.tif", elev_masked, bounds)
    write_geotiff(topo_dir / "slope_1km.tif", slope_masked, bounds)

    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    max_idx = int(np.nanargmax(elev_values))
    max_row = grid.iloc[max_idx]
    max_lon, max_lat = transformer.transform(float(max_row["x"]), float(max_row["y"]))
    min_idx = int(np.nanargmin(elev_values))
    min_row = grid.iloc[min_idx]
    min_lon, min_lat = transformer.transform(float(min_row["x"]), float(min_row["y"]))

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "USGS 3DEP ImageServer exportImage in EPSG:5070, sampled by grid_lookup x/y",
        "grid_lookup_path": str(grid_lookup.relative_to(REPO_ROOT) if grid_lookup.is_relative_to(REPO_ROOT) else grid_lookup),
        "backup_dir": str(backup_dir.relative_to(REPO_ROOT) if backup_dir and backup_dir.is_relative_to(REPO_ROOT) else backup_dir),
        "bounds_epsg5070": list(bounds),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_active_grid_cells": int(len(grid)),
        "source": source_meta,
        "mean_elevation_stats_m": {
            "min": float(np.nanmin(elev_values)),
            "p05": float(np.nanpercentile(elev_values, 5)),
            "median": float(np.nanmedian(elev_values)),
            "p95": float(np.nanpercentile(elev_values, 95)),
            "max": float(np.nanmax(elev_values)),
        },
        "mean_slope_stats_deg": {
            "min": float(np.nanmin(slope_values)),
            "p05": float(np.nanpercentile(slope_values, 5)),
            "median": float(np.nanmedian(slope_values)),
            "p95": float(np.nanpercentile(slope_values, 95)),
            "max": float(np.nanmax(slope_values)),
        },
        "max_elevation_cell": {
            "grid_id": int(max_row["grid_id"]),
            "row": int(max_row["row"]),
            "col": int(max_row["col"]),
            "x": float(max_row["x"]),
            "y": float(max_row["y"]),
            "lon": float(max_lon),
            "lat": float(max_lat),
            "mean_elevation_m": float(elev_values[max_idx]),
        },
        "min_elevation_cell": {
            "grid_id": int(min_row["grid_id"]),
            "row": int(min_row["row"]),
            "col": int(min_row["col"]),
            "x": float(min_row["x"]),
            "y": float(min_row["y"]),
            "lon": float(min_lon),
            "lat": float(min_lat),
            "mean_elevation_m": float(elev_values[min_idx]),
        },
    }
    (topo_dir / "topo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {topo_csv}")
    print(f"Wrote: {topo_dir / 'elev_1km.tif'}")
    print(f"Wrote: {topo_dir / 'slope_1km.tif'}")
    print(f"Wrote: {topo_dir / 'topo_summary.json'}")
    print("Elevation stats (m):", summary["mean_elevation_stats_m"])
    print("Max elevation cell:", summary["max_elevation_cell"])


if __name__ == "__main__":
    main()
