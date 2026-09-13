from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from pyproj import CRS, Geod, Transformer
from rasterio.plot import plotting_extent
from rasterio.transform import rowcol
from rasterio.warp import Resampling, calculate_default_transform, reproject
import shapefile  # pyshp


CLASS_ORDER = [3, 2, 1, -1, -2, -3]
CLASS_LABELS = [
    "High connectivity",
    "Intermediate connectivity",
    "Some connectivity",
    "Thin confining",
    "Intermediate confining",
    "Thick confining",
]
CLASS_COLORS = [
    "#8b0000",  # dark red
    "#ef0000",  # red
    "#f49a9a",  # light red
    "#8f88e8",  # light purple
    "#1313db",  # blue
    "#0a0a97",  # dark blue
]
DISPLAY_EXTENT = (-92.8, -88.4, 31.2, 37.4)  # lon_min, lon_max, lat_min, lat_max


def _format_lon_w(lon: float) -> str:
    value = abs(float(lon))
    deg = int(value)
    minutes = int(round((value - deg) * 60))
    if minutes == 60:
        deg += 1
        minutes = 0
    return f"{deg}\N{DEGREE SIGN}{minutes:02d}\N{PRIME}W"


def _format_lat_n(lat: float) -> str:
    value = abs(float(lat))
    deg = int(value)
    minutes = int(round((value - deg) * 60))
    if minutes == 60:
        deg += 1
        minutes = 0
    return f"{deg}\N{DEGREE SIGN}{minutes:02d}\N{PRIME}N"


def _build_class_index(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    class_idx = np.full(arr.shape, np.nan, dtype=np.float32)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    for idx, cls in enumerate(CLASS_ORDER):
        class_idx[arr == cls] = idx
    return class_idx


def _build_active_cell_mask(
    ds: rasterio.io.DatasetReader,
    active_cell_csv: Path,
    cell_crs: CRS = CRS.from_epsg(5070),
) -> np.ndarray:
    cells = pd.read_csv(active_cell_csv, usecols=["x_center", "y_center"])
    xs = cells["x_center"].to_numpy(dtype=np.float64)
    ys = cells["y_center"].to_numpy(dtype=np.float64)
    ds_crs = CRS.from_user_input(ds.crs)
    if not ds_crs.equals(cell_crs):
        transformer = Transformer.from_crs(cell_crs, ds_crs, always_xy=True)
        xs, ys = transformer.transform(xs, ys)
    rows, cols = rowcol(ds.transform, xs, ys)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    valid = (
        (rows >= 0)
        & (rows < ds.height)
        & (cols >= 0)
        & (cols < ds.width)
    )
    mask = np.zeros((ds.height, ds.width), dtype=bool)
    mask[rows[valid], cols[valid]] = True
    return mask


def _reproject_to_wgs84(src_arr: np.ndarray, src_transform, src_crs, src_nodata):
    dst_crs = CRS.from_epsg(4326)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        dst_crs,
        src_arr.shape[1],
        src_arr.shape[0],
        left=src_transform.c,
        bottom=src_transform.f + src_transform.e * src_arr.shape[0],
        right=src_transform.c + src_transform.a * src_arr.shape[1],
        top=src_transform.f,
    )
    dst_arr = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return dst_arr, dst_transform


def _plot_boundary(
    ax: plt.Axes,
    shp_path: Path,
    src_crs: CRS,
    line_color: str = "#cfcfcf",
    line_width: float = 0.9,
) -> None:
    if not shp_path.exists():
        return
    reader = shapefile.Reader(str(shp_path))
    transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
    for shape_rec in reader.shapes():
        pts = np.asarray(shape_rec.points, dtype=np.float64)
        if pts.size == 0:
            continue
        lon, lat = transformer.transform(pts[:, 0], pts[:, 1])
        parts = list(shape_rec.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            sl = slice(parts[i], parts[i + 1])
            ax.plot(lon[sl], lat[sl], color=line_color, linewidth=line_width, zorder=4)


def _add_scale_bar(ax: plt.Axes, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> None:
    geod = Geod(ellps="WGS84")
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min

    bar_lat = lat_min + 0.075 * lat_span
    bar_h = 0.023 * lat_span
    lon_after_200km, _, _ = geod.fwd(0.0, bar_lat, 90.0, 200_000.0)
    bar_lon0 = lon_max - 0.08 * lon_span - lon_after_200km

    lon1, _, _ = geod.fwd(bar_lon0, bar_lat, 90.0, 100_000.0)
    lon2, _, _ = geod.fwd(bar_lon1 := lon1, bar_lat, 90.0, 100_000.0)

    ax.add_patch(
        Rectangle(
            (bar_lon0, bar_lat),
            bar_lon1 - bar_lon0,
            bar_h,
            facecolor="black",
            edgecolor="black",
            linewidth=1.0,
            zorder=8,
        )
    )
    ax.add_patch(
        Rectangle(
            (bar_lon1, bar_lat),
            lon2 - bar_lon1,
            bar_h,
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            zorder=8,
        )
    )

    y_text = bar_lat + bar_h + 0.02 * lat_span
    ax.text(bar_lon0, y_text, "0", ha="center", va="bottom", fontsize=11, zorder=9)
    ax.text(bar_lon1, y_text, "100", ha="center", va="bottom", fontsize=11, zorder=9)
    ax.text(lon2, y_text, "200 km", ha="center", va="bottom", fontsize=11, zorder=9)


def make_connectivity_figure(
    connectivity_tif: Path,
    active_cell_csv: Path,
    boundary_shp: Path,
    boundary_prj: Path,
    output_png: Path,
    dpi: int = 300,
) -> None:
    with rasterio.open(connectivity_tif) as ds:
        src_arr = ds.read(1).astype(np.float32)
        src_transform = ds.transform
        src_crs = CRS.from_user_input(ds.crs)
        src_nodata = ds.nodata
        active_mask = _build_active_cell_mask(ds, active_cell_csv)

    # Keep only MRVA active cells.
    src_arr = np.where(active_mask, src_arr, np.nan).astype(np.float32, copy=False)

    class_idx_src = _build_class_index(src_arr, src_nodata)
    class_idx_wgs84, dst_transform = _reproject_to_wgs84(
        class_idx_src,
        src_transform,
        src_crs,
        np.nan,
    )

    native_extent = plotting_extent(class_idx_wgs84, dst_transform)
    lon_min, lon_max, lat_min, lat_max = DISPLAY_EXTENT
    extent = native_extent
    origin = "upper" if dst_transform.e < 0 else "lower"

    cmap = ListedColormap(CLASS_COLORS, name="connectivity_classes")
    norm = BoundaryNorm(np.arange(-0.5, len(CLASS_ORDER) + 0.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(8.6, 7.2), facecolor="#e6e6e6")
    ax.set_facecolor("#e6e6e6")

    ax.imshow(
        class_idx_wgs84,
        cmap=cmap,
        norm=norm,
        origin=origin,
        extent=extent,
        interpolation="nearest",
        zorder=2,
    )

    # Light outer boundary line.
    if boundary_prj.exists():
        boundary_crs = CRS.from_wkt(boundary_prj.read_text(encoding="utf-8"))
    else:
        boundary_crs = src_crs
    _plot_boundary(ax, boundary_shp, boundary_crs, line_color="#d0d0d0", line_width=0.9)

    # Dark outline around classified mask.
    valid = np.isfinite(class_idx_wgs84)
    if np.any(valid):
        left, right, bottom, top = native_extent
        x = np.linspace(left, right, class_idx_wgs84.shape[1])
        y = (
            np.linspace(top, bottom, class_idx_wgs84.shape[0])
            if origin == "upper"
            else np.linspace(bottom, top, class_idx_wgs84.shape[0])
        )
        ax.contour(x, y, valid.astype(np.float32), levels=[0.5], colors="black", linewidths=0.8, zorder=5)

    # Ticks/labels styled to match target map.
    x_ticks = [-92.5, -90.0]
    y_ticks = [32.5, 35.0, 37.0]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([_format_lon_w(x) for x in x_ticks], fontsize=12)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([_format_lat_n(yv) for yv in y_ticks], fontsize=12, rotation=90, va="center")
    ax.tick_params(
        axis="both",
        direction="in",
        length=7,
        width=1.1,
        top=True,
        bottom=True,
        left=True,
        right=True,
        labeltop=True,
        labelbottom=False,
        labelright=False,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")

    # Legend.
    handles = [
        Patch(facecolor=color, edgecolor=color, label=label)
        for color, label in zip(CLASS_COLORS, CLASS_LABELS)
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.53, 0.24),
        frameon=False,
        fontsize=7.9,
        labelspacing=0.45,
        handlelength=1.8,
        handleheight=1.2,
    )

    _add_scale_bar(ax, lon_min, lon_max, lat_min, lat_max)

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("")
    ax.set_ylabel("")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    return argparse.Namespace(
        connectivity_tif=repo_root / "data" / "8 connectivity" / "ConfiningLayer_SurfaceConnectivity.tif",
        active_cell_csv=repo_root / "data" / "2 well WTD" / "active_cell_lookup.csv",
        boundary_shp=repo_root / "data" / "0 mask MRVA" / "outerboundary.shp",
        boundary_prj=repo_root / "data" / "0 mask MRVA" / "outerboundary.prj",
        output_png=repo_root / "outputs" / "figures" / "connectivity_classes_reference.png",
        dpi=300,
    )


def main() -> None:
    args = parse_args()
    make_connectivity_figure(
        connectivity_tif=args.connectivity_tif,
        active_cell_csv=args.active_cell_csv,
        boundary_shp=args.boundary_shp,
        boundary_prj=args.boundary_prj,
        output_png=args.output_png,
        dpi=args.dpi,
    )
    print(f"Saved: {args.output_png}")


if __name__ == "__main__":
    main()
