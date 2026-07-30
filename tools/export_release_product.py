from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECON_ROOT = ROOT / "outputs" / "RECON_MAIN_2011_2023"
DEFAULT_OUTPUT = ROOT / "release_data" / "mrva_monthly_wtd_2011_2023.nc"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the canonical monthly MRVA WTD reconstruction as NetCDF."
    )
    parser.add_argument(
        "--recon-root",
        type=Path,
        default=DEFAULT_RECON_ROOT,
        help="Canonical reconstruction root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output NetCDF path.",
    )
    return parser.parse_args()


def load_inputs(recon_root: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    wtd_path = recon_root / "reconstruction" / "wtd_reconstructed_matrix.npy"
    uncertainty_path = (
        recon_root
        / "model_uncertainty"
        / "monthly_model_uncertainty_radius_matrix.npy"
    )
    grid_path = recon_root / "metadata" / "grid_lookup.csv"
    month_path = recon_root / "metadata" / "month_index.csv"

    for path in (wtd_path, uncertainty_path, grid_path, month_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {display_path(path)}")

    wtd = np.load(wtd_path, mmap_mode="r")
    uncertainty = np.load(uncertainty_path, mmap_mode="r")
    grid = pd.read_csv(grid_path)
    months = pd.read_csv(month_path)

    if wtd.shape != uncertainty.shape:
        raise ValueError(
            f"WTD shape {wtd.shape} does not match uncertainty shape {uncertainty.shape}."
        )
    if wtd.shape != (len(months), len(grid)):
        raise ValueError(
            f"Matrix shape {wtd.shape} does not match "
            f"{len(months)} months x {len(grid)} grid cells."
        )
    if not np.isfinite(wtd).all():
        raise ValueError("The canonical WTD matrix contains non-finite values.")
    if not np.isfinite(uncertainty).all():
        raise ValueError("The PI75 uncertainty radius contains non-finite values.")
    if np.any(np.asarray(uncertainty) < 0):
        raise ValueError("The PI75 uncertainty radius contains negative values.")

    required_grid = {"grid_id", "row", "col", "cell_id", "x", "y"}
    required_month = {
        "month_idx",
        "month_label",
        "is_anchor_month",
        "interval_id",
        "direct_observation_count",
    }
    if missing := required_grid.difference(grid.columns):
        raise ValueError(f"Grid lookup is missing columns: {sorted(missing)}")
    if missing := required_month.difference(months.columns):
        raise ValueError(f"Month index is missing columns: {sorted(missing)}")

    return wtd, uncertainty, grid, months


def build_dataset(
    wtd: np.ndarray,
    uncertainty: np.ndarray,
    grid: pd.DataFrame,
    months: pd.DataFrame,
) -> xr.Dataset:
    projected_crs = CRS.from_epsg(5070)
    geographic_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(
        projected_crs,
        geographic_crs,
        always_xy=True,
    )
    longitude, latitude = transformer.transform(
        grid["x"].to_numpy(dtype=np.float64),
        grid["y"].to_numpy(dtype=np.float64),
    )
    time = pd.to_datetime(months["month_label"].astype(str) + "-01").to_numpy()

    dataset = xr.Dataset(
        data_vars={
            "water_table_depth": (
                ("time", "cell"),
                wtd,
                {
                    "long_name": "monthly water-table depth below land surface",
                    "units": "m",
                    "positive": "down",
                    "comment": (
                        "Positive values are below land surface; negative values "
                        "indicate reconstructed water levels above land surface."
                    ),
                    "grid_mapping": "crs",
                },
            ),
            "uncertainty_radius_pi75": (
                ("time", "cell"),
                uncertainty,
                {
                    "long_name": "nominal 75 percent prediction-interval radius",
                    "units": "m",
                    "interval_level": 0.75,
                    "grid_mapping": "crs",
                },
            ),
            "is_anchor_month": (
                "time",
                months["is_anchor_month"].to_numpy(dtype=np.int8),
                {"long_name": "anchor-month flag"},
            ),
            "reconstruction_interval_id": (
                "time",
                months["interval_id"].to_numpy(dtype=np.int16),
                {"long_name": "anchor-to-anchor reconstruction interval identifier"},
            ),
            "direct_observation_count": (
                "time",
                months["direct_observation_count"].to_numpy(dtype=np.int32),
                {"long_name": "number of directly observed grid cells in month"},
            ),
        },
        coords={
            "time": ("time", time, {"long_name": "month"}),
            "cell": ("cell", np.arange(len(grid), dtype=np.int32)),
            "grid_id": ("cell", grid["grid_id"].to_numpy(dtype=np.int32)),
            "cell_id": ("cell", grid["cell_id"].to_numpy(dtype=np.int32)),
            "row": ("cell", grid["row"].to_numpy(dtype=np.int32)),
            "col": ("cell", grid["col"].to_numpy(dtype=np.int32)),
            "x": (
                "cell",
                grid["x"].to_numpy(dtype=np.float64),
                {
                    "long_name": "Albers easting",
                    "standard_name": "projection_x_coordinate",
                    "units": "m",
                },
            ),
            "y": (
                "cell",
                grid["y"].to_numpy(dtype=np.float64),
                {
                    "long_name": "Albers northing",
                    "standard_name": "projection_y_coordinate",
                    "units": "m",
                },
            ),
            "longitude": (
                "cell",
                np.asarray(longitude, dtype=np.float64),
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "latitude": (
                "cell",
                np.asarray(latitude, dtype=np.float64),
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
        },
        attrs={
            "Conventions": "CF-1.10, ACDD-1.3",
            "title": "Monthly water-table depth reconstruction for the MRVA, 2011-2023",
            "summary": (
                "Geophysics-informed GNN reconstruction of monthly water-table "
                "depth on the active 1-km Mississippi River Valley alluvial aquifer grid."
            ),
            "spatial_reference": "EPSG:5070",
            "horizontal_grid_spacing": "1 km",
            "temporal_resolution": "monthly",
            "time_coverage_start": "2011-01-01",
            "time_coverage_end": "2023-12-01",
            "geospatial_lon_min": float(np.min(longitude)),
            "geospatial_lon_max": float(np.max(longitude)),
            "geospatial_lat_min": float(np.min(latitude)),
            "geospatial_lat_max": float(np.max(latitude)),
            "wtd_sign_convention": "positive downward below land surface",
            "uncertainty_definition": "nominal 75 percent prediction-interval radius",
            "source_code": "tools/export_release_product.py",
        },
    )

    crs_attrs = projected_crs.to_cf()
    crs_attrs["spatial_ref"] = projected_crs.to_wkt()
    dataset["crs"] = xr.DataArray(np.int8(0), attrs=crs_attrs)
    return dataset


def write_dataset(dataset: xr.Dataset, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.unlink(missing_ok=True)

    matrix_encoding = {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
        "chunksizes": (12, 4096),
    }
    encoding = {
        "water_table_depth": matrix_encoding.copy(),
        "uncertainty_radius_pi75": matrix_encoding.copy(),
        "time": {"dtype": "int32"},
    }
    dataset.to_netcdf(
        tmp,
        engine="netcdf4",
        format="NETCDF4",
        encoding=encoding,
    )
    tmp.replace(output)


def write_checksum(output: Path) -> Path:
    checksum_path = output.parent / "SHA256SUMS.txt"
    checksum_path.write_text(
        f"{sha256(output)}  {output.name}\n",
        encoding="ascii",
    )
    return checksum_path


def main() -> None:
    args = parse_args()
    recon_root = args.recon_root.resolve()
    output = args.output.resolve()

    wtd, uncertainty, grid, months = load_inputs(recon_root)
    dataset = build_dataset(wtd, uncertainty, grid, months)
    write_dataset(dataset, output)
    checksum_path = write_checksum(output)

    print(f"NetCDF: {display_path(output)}")
    print(f"Shape: {wtd.shape[0]} months x {wtd.shape[1]} grid cells")
    print(f"Size: {output.stat().st_size / (1024 ** 2):.1f} MB")
    print(f"Checksums: {display_path(checksum_path)}")


if __name__ == "__main__":
    main()
