from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree
from shapely import points
from shapely.geometry import LineString, MultiLineString


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TOPO_CSV = REPO_ROOT / "data" / "5 topography" / "topo_1km.csv"
DEFAULT_TOPO_SUMMARY_JSON = REPO_ROOT / "data" / "5 topography" / "topo_summary.json"
DEFAULT_ELEV_TIF = REPO_ROOT / "data" / "5 topography" / "elev_1km.tif"
DEFAULT_ACTIVE_GRID_CSV = REPO_ROOT / "data" / "2 well WTD" / "active_cell_lookup.csv"
DEFAULT_BOUNDARY_SHP = REPO_ROOT / "data" / "0 mask MRVA" / "outerboundary.shp"
DEFAULT_HYDRORIVERS_SHP = (
    REPO_ROOT
    / "data_raw"
    / "11 river_network"
    / "HydroRIVERS_NorthAmerica"
    / "HydroRIVERS_v10_na_shp"
    / "HydroRIVERS_v10_na.shp"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add HydroRIVERS order-1/2 distance features to MRVA topo_1km.csv."
    )
    parser.add_argument("--topo-csv", type=Path, default=DEFAULT_TOPO_CSV)
    parser.add_argument("--topo-summary-json", type=Path, default=DEFAULT_TOPO_SUMMARY_JSON)
    parser.add_argument("--elevation-tif", type=Path, default=DEFAULT_ELEV_TIF)
    parser.add_argument("--active-grid-csv", type=Path, default=DEFAULT_ACTIVE_GRID_CSV)
    parser.add_argument("--boundary-shp", type=Path, default=DEFAULT_BOUNDARY_SHP)
    parser.add_argument("--hydrorivers-shp", type=Path, default=DEFAULT_HYDRORIVERS_SHP)
    parser.add_argument("--order-field", type=str, default="ORD_CLAS")
    parser.add_argument("--orders", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--densify-spacing-m", type=float, default=250.0)
    return parser.parse_args()


def load_active_grid(active_grid_csv: Path) -> pd.DataFrame:
    active = pd.read_csv(active_grid_csv).copy()
    rename_map = {
        "node_id": "grid_id",
        "cell_id": "cell_id",
        "x_center": "x",
        "y_center": "y",
    }
    active = active.rename(columns=rename_map)
    required = {"grid_id", "x", "y"}
    missing = required - set(active.columns)
    if missing:
        raise KeyError(f"Active grid CSV missing required columns: {sorted(missing)}")
    return active


def load_order12_rivers(hydrorivers_shp: Path, boundary_shp: Path, order_field: str, orders: list[int]) -> gpd.GeoDataFrame:
    boundary = gpd.read_file(boundary_shp)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:5070")
    boundary_5070 = boundary.to_crs("EPSG:5070")
    boundary_wgs84 = boundary_5070.to_crs("EPSG:4326")

    rivers = gpd.read_file(hydrorivers_shp, bbox=tuple(boundary_wgs84.total_bounds))
    if rivers.empty:
        raise ValueError("No HydroRIVERS features found inside the MRVA bounding box.")
    if order_field not in rivers.columns:
        raise KeyError(f"HydroRIVERS field not found: {order_field}")

    rivers = rivers.to_crs("EPSG:5070")
    rivers = gpd.clip(rivers, boundary_5070)
    rivers = rivers[rivers.geometry.notna()].copy()
    rivers = rivers.explode(index_parts=False, ignore_index=True)
    rivers = rivers[rivers[order_field].isin(orders)].copy()
    if rivers.empty:
        raise ValueError(f"No HydroRIVERS features remain after filtering {order_field} in {orders}.")
    return rivers


def read_elevation_raster(elevation_tif: Path) -> tuple[rasterio.io.DatasetReader, np.ndarray, float]:
    src = rasterio.open(elevation_tif)
    arr = src.read(1).astype(np.float32)
    nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    fill = float(np.nanmedian(arr))
    return src, arr, fill


def sample_raster_nearest(xy: np.ndarray, src, arr: np.ndarray, fill: float) -> np.ndarray:
    inv = ~src.transform
    cols, rows = inv * (xy[:, 0], xy[:, 1])
    cols = np.floor(cols).astype(np.int64)
    rows = np.floor(rows).astype(np.int64)
    ok = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
    out = np.full(xy.shape[0], fill, dtype=np.float32)
    vals = np.full(xy.shape[0], np.nan, dtype=np.float32)
    vals[ok] = arr[rows[ok], cols[ok]]
    good = np.isfinite(vals)
    out[good] = vals[good]
    return out


def densify_line_xy(line: LineString, spacing_m: float) -> np.ndarray:
    if line.length <= 0:
        return np.empty((0, 2), dtype=np.float64)
    if line.length <= spacing_m:
        coords = np.asarray(line.coords, dtype=np.float64)
        return coords[:, :2]
    distances = np.arange(0.0, float(line.length), spacing_m, dtype=np.float64)
    if distances.size == 0 or distances[-1] < line.length:
        distances = np.concatenate([distances, [float(line.length)]])
    pts = [line.interpolate(float(d)) for d in distances]
    return np.asarray([(p.x, p.y) for p in pts], dtype=np.float64)


def build_river_support_points(
    rivers: gpd.GeoDataFrame,
    src,
    elev_arr: np.ndarray,
    elev_fill: float,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    xy_chunks: list[np.ndarray] = []
    for geom in rivers.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            chunk = densify_line_xy(geom, spacing_m)
            if chunk.size:
                xy_chunks.append(chunk)
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                chunk = densify_line_xy(part, spacing_m)
                if chunk.size:
                    xy_chunks.append(chunk)
    if not xy_chunks:
        raise ValueError("No river support points were generated for relief sampling.")
    river_xy = np.vstack(xy_chunks).astype(np.float64)
    river_elev = sample_raster_nearest(river_xy, src, elev_arr, elev_fill)
    return river_xy, river_elev


def compute_distance_features(
    active: pd.DataFrame,
    rivers: gpd.GeoDataFrame,
    elevation_tif: Path,
    densify_spacing_m: float,
) -> pd.DataFrame:
    union = rivers.geometry.union_all() if hasattr(rivers.geometry, "union_all") else rivers.geometry.unary_union
    point_geom = gpd.GeoSeries(points(active["x"].to_numpy(), active["y"].to_numpy()), crs="EPSG:5070")
    distance_m = point_geom.distance(union).to_numpy(dtype=np.float32)
    cell_xy = active[["x", "y"]].to_numpy(dtype=np.float64)

    elev_src, elev_arr, elev_fill = read_elevation_raster(elevation_tif)
    try:
        cell_elev = sample_raster_nearest(cell_xy, elev_src, elev_arr, elev_fill)
        river_xy, river_elev = build_river_support_points(rivers, elev_src, elev_arr, elev_fill, densify_spacing_m)
    finally:
        elev_src.close()

    tree = cKDTree(river_xy)
    _, nearest_idx = tree.query(cell_xy, k=1)
    nearest_river_elev = river_elev[np.asarray(nearest_idx, dtype=np.int64)]
    relief_m = (cell_elev - nearest_river_elev).astype(np.float32)

    out = active[["grid_id"]].copy()
    out["distance_to_order12_river_m"] = distance_m
    out["distance_to_order12_river_km"] = (distance_m / 1000.0).astype(np.float32)
    out["log1p_distance_to_order12_river_m"] = np.log1p(distance_m).astype(np.float32)
    out["nearest_order12_stream_elevation_m"] = nearest_river_elev.astype(np.float32)
    out["relief_to_order12_stream_m"] = relief_m
    return out


def merge_into_topography(topo_csv: Path, river_features: pd.DataFrame) -> pd.DataFrame:
    topo = pd.read_csv(topo_csv).copy()
    if "grid_id" not in topo.columns:
        if "node_id" in topo.columns:
            topo = topo.rename(columns={"node_id": "grid_id"})
        else:
            raise KeyError("Topography CSV must contain 'grid_id' or 'node_id'.")

    drop_cols = [
        "distance_to_order12_river_m",
        "distance_to_order12_river_km",
        "log1p_distance_to_order12_river_m",
        "nearest_order12_stream_elevation_m",
        "relief_to_order12_stream_m",
    ]
    topo = topo.drop(columns=[c for c in drop_cols if c in topo.columns], errors="ignore")
    merged = topo.merge(river_features, on="grid_id", how="left")
    if merged[drop_cols].isna().any().any():
        raise ValueError("River-distance merge left missing values in topo table.")
    return merged


def update_summary(summary_path: Path, river_features: pd.DataFrame, order_field: str, orders: list[int]) -> None:
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}

    vals = river_features["distance_to_order12_river_m"].to_numpy(dtype=np.float64)
    relief = river_features["relief_to_order12_stream_m"].to_numpy(dtype=np.float64)
    summary["river_distance_order12"] = {
        "order_field": order_field,
        "orders_included": [int(v) for v in orders],
        "n_active_grid_cells": int(len(vals)),
        "distance_stats_m": {
            "min": float(np.nanmin(vals)),
            "p05": float(np.nanpercentile(vals, 5)),
            "median": float(np.nanmedian(vals)),
            "p95": float(np.nanpercentile(vals, 95)),
            "max": float(np.nanmax(vals)),
        },
        "log1p_distance_stats": {
            "min": float(np.nanmin(np.log1p(vals))),
            "p05": float(np.nanpercentile(np.log1p(vals), 5)),
            "median": float(np.nanmedian(np.log1p(vals))),
            "p95": float(np.nanpercentile(np.log1p(vals), 95)),
            "max": float(np.nanmax(np.log1p(vals))),
        },
        "relief_stats_m": {
            "min": float(np.nanmin(relief)),
            "p05": float(np.nanpercentile(relief, 5)),
            "median": float(np.nanmedian(relief)),
            "p95": float(np.nanpercentile(relief, 95)),
            "max": float(np.nanmax(relief)),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    active = load_active_grid(args.active_grid_csv)
    rivers = load_order12_rivers(args.hydrorivers_shp, args.boundary_shp, args.order_field, list(args.orders))
    river_features = compute_distance_features(
        active,
        rivers,
        args.elevation_tif,
        float(args.densify_spacing_m),
    )
    topo_updated = merge_into_topography(args.topo_csv, river_features)

    args.topo_csv.parent.mkdir(parents=True, exist_ok=True)
    topo_updated.to_csv(args.topo_csv, index=False)
    update_summary(args.topo_summary_json, river_features, args.order_field, list(args.orders))

    lengths_km = rivers.geometry.length.to_numpy(dtype=np.float64) / 1000.0
    dist = river_features["distance_to_order12_river_m"].to_numpy(dtype=np.float64)
    relief = river_features["relief_to_order12_stream_m"].to_numpy(dtype=np.float64)
    print(f"Wrote: {args.topo_csv}")
    print(f"Updated: {args.topo_summary_json}")
    print(
        "HydroRIVERS subset:",
        {
            "order_field": args.order_field,
            "orders": list(args.orders),
            "segment_count": int(len(rivers)),
            "total_length_km": float(lengths_km.sum()),
        },
    )
    print(
        "Distance stats (m):",
        {
            "min": float(np.nanmin(dist)),
            "p05": float(np.nanpercentile(dist, 5)),
            "median": float(np.nanmedian(dist)),
            "p95": float(np.nanpercentile(dist, 95)),
            "max": float(np.nanmax(dist)),
        },
    )
    print(
        "Relief stats (m):",
        {
            "min": float(np.nanmin(relief)),
            "p05": float(np.nanpercentile(relief, 5)),
            "median": float(np.nanmedian(relief)),
            "p95": float(np.nanpercentile(relief, 95)),
            "max": float(np.nanmax(relief)),
        },
    )


if __name__ == "__main__":
    main()
