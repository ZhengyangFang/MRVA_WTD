from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen, urlretrieve

import numpy as np
import pandas as pd
import rasterio
import shapefile  # pyshp
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Polygon
from shapely.prepared import prep


DEFAULT_REGION = (-92.8, -88.4, 31.15, 37.55)  # lon_min, lon_max, lat_min, lat_max


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value in (None, "") else str(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def read_region() -> tuple[float, float, float, float]:
    explicit = os.environ.get("FIG1A_REGION")
    if explicit:
        parts = explicit.replace(",", "/").split("/")
        if len(parts) != 4:
            raise ValueError("FIG1A_REGION must be lon_min/lon_max/lat_min/lat_max")
        return tuple(float(part) for part in parts)  # type: ignore[return-value]

    lon_min, lon_max, lat_min, lat_max = DEFAULT_REGION
    return (
        lon_min - env_float("FIG1A_PAD_W", 0.0),
        lon_max + env_float("FIG1A_PAD_E", 0.0),
        lat_min - env_float("FIG1A_PAD_S", 0.0),
        lat_max + env_float("FIG1A_PAD_N", 0.0),
    )


REGION = read_region()
MAP_WIDTH_CM = env_float("FIG1A_MAP_WIDTH_CM", 10.2)
PROJECTION = f"M{MAP_WIDTH_CM:.2f}c"
FONT_SCALE = env_float("FIG1A_FONT_SCALE", 1.0)
FONT_ADD_PT = env_float("FIG1A_FONT_ADD_PT", 0.0)
STATE_LABEL_FONT_SCALE = env_float("FIG1A_STATE_LABEL_FONT_SCALE", FONT_SCALE)
STATE_LABEL_FONT_ADD_PT = env_float("FIG1A_STATE_LABEL_FONT_ADD_PT", FONT_ADD_PT)
STATE_LABEL_COLOR = env_str("FIG1A_STATE_LABEL_COLOR", "#9a9a9a")
DEM_SIZE = env_str("FIG1A_DEM_SIZE", "900,1300")
SHADE_INTENSITY = env_float("FIG1A_SHADE_INTENSITY", 0.65)
TERRAIN_TRANSPARENCY = min(max(env_float("FIG1A_TERRAIN_TRANSPARENCY", 0.0), 0.0), 100.0)
AEM_SEGMENT_STRIDE = max(env_int("FIG1A_AEM_SEGMENT_STRIDE", 1), 1)
AEM_VERTEX_STRIDE = max(env_int("FIG1A_AEM_VERTEX_STRIDE", 1), 1)
AEM_LINE_PEN = env_str("FIG1A_AEM_LINE_PEN", "1.0p,white")
WELL_SYMBOL_SIZE = env_str("FIG1A_WELL_SYMBOL_SIZE", "0.022c")
BOUNDARY_PEN = env_str("FIG1A_BOUNDARY_PEN", "0.85p,white")
STATE_BOUNDARY_PEN = env_str("FIG1A_STATE_BOUNDARY_PEN", "0.55p,black")
DRAW_WELLS = env_bool("FIG1A_DRAW_WELLS", True)
DRAW_STATE_LABELS = env_bool("FIG1A_DRAW_STATE_LABELS", False)
DRAW_LEGEND = env_bool("FIG1A_DRAW_LEGEND", False)
MRVA_FILL_CYAN = env_str("FIG1A_MRVA_FILL_CYAN", "#d8f3ee@45")
MRVA_FILL_YELLOW = env_str("FIG1A_MRVA_FILL_YELLOW", "#fff3bf@72")
TRIBUTARY_ORDER1_PEN = env_str("FIG1A_TRIBUTARY_ORDER1_PEN", "0.55p,#72b7d2@48")
TRIBUTARY_ORDER2_PEN = env_str("FIG1A_TRIBUTARY_ORDER2_PEN", "0.28p,#9ed4e8@62")
MISSISSIPPI_HALO_PEN = env_str("FIG1A_MISSISSIPPI_HALO_PEN", "2.4p,white@12")
MISSISSIPPI_PEN = env_str("FIG1A_MISSISSIPPI_PEN", "1.45p,#045a8d")
LEGEND_WIDTH_CM = env_float("FIG1A_LEGEND_WIDTH_CM", 3.65)
LEGEND_OFFSET = env_str("FIG1A_LEGEND_OFFSET", "0.12c/1.15c")
SCALEBAR_OFFSET = env_str("FIG1A_SCALEBAR_OFFSET", "0.55c/0.35c")
NORTH_ARROW_SIZE_CM = env_float("FIG1A_NORTH_ARROW_SIZE_CM", 1.0)
NORTH_ARROW_OFFSET = env_str("FIG1A_NORTH_ARROW_OFFSET", "0.22c/0.2c")
MAP_FRAME_TYPE = env_str("FIG1A_MAP_FRAME_TYPE", "plain")
MAP_FRAME_WIDTH = env_str("FIG1A_MAP_FRAME_WIDTH", "5p")
MAP_FRAME_PEN = env_str("FIG1A_MAP_FRAME_PEN", "thicker,black")
MAP_FRAME_PERCENT = env_str("FIG1A_MAP_FRAME_PERCENT", "100")
MAP_TICK_LENGTH_PRIMARY = env_str("FIG1A_MAP_TICK_LENGTH_PRIMARY", "3p")
FIGURE_CACHE_SUBDIR = env_str("FIG1A_CACHE_SUBDIR", "figure_cache")
HYDRORIVERS_URL = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_na_shp.zip"
TNM_ELEVATION_EXPORT_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
RAW_AEM_NC = (
    "data_raw/1 MAP Electrical Resistivity & Facies Classification/"
    "MAP_RegionalAEM_2022_ResistivityFacies_DepthGrids_log10res_masked_latestmask.nc"
)
AEM_FLIGHTLINES_DIR = (
    "data_raw/1 MAP Electrical Resistivity & Facies Classification/original_flightlines"
)
AEM_FLIGHTLINE_RELEASES = (
    ("5f35b17382cee144fb35a733", "MAP_RegionalTempest_2020_FlightLines"),
)
FONT_REGULAR = "ArialMT"
FONT_BOLD = "Arial-BoldMT"
FONT_ITALIC = "Arial-ItalicMT"
FONT_BOLD_ITALIC = "Arial-BoldItalicMT"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def gmt_executable(root: Path) -> Path:
    local = root / ".gmt_env" / "Library" / "bin" / "gmt.exe"
    return local if local.exists() else Path("gmt")


def run_gmt(gmt: Path, args: list[str], cwd: Path) -> None:
    subprocess.run([str(gmt), *args], cwd=cwd, check=True)


def region_string(region: tuple[float, float, float, float] = REGION) -> str:
    return f"{region[0]}/{region[1]}/{region[2]}/{region[3]}"


def pt(size: float) -> str:
    return f"{size * FONT_SCALE + FONT_ADD_PT:.1f}p"


def state_pt(size: float) -> str:
    return f"{size * STATE_LABEL_FONT_SCALE + STATE_LABEL_FONT_ADD_PT:.1f}p"


def gmt_fill_enabled(fill: str) -> bool:
    return fill.strip().lower() not in {"", "none", "off", "false", "0", "transparent"}


def write_gmt_custom_fonts(work: Path) -> Path:
    """Register Windows Arial faces for GMT/PostScript output."""
    out = work / "PSL_custom_fonts.txt"
    out.write_text(
        "\n".join(
            [
                f"{FONT_REGULAR} 0.700 0",
                f"{FONT_BOLD} 0.700 0",
                f"{FONT_ITALIC} 0.700 0",
                f"{FONT_BOLD_ITALIC} 0.700 0",
                "",
            ]
        ),
        encoding="ascii",
    )
    return out


def safe_region_token(region: tuple[float, float, float, float] = REGION) -> str:
    def one(value: float) -> str:
        text = f"{value:.2f}".replace("-", "m").replace(".", "p")
        return text

    return "_".join(one(value) for value in region)


def figure_cache_dir(work: Path, name: str) -> Path:
    """Dedicated Fig. 1a derived-data cache, separate from training/MRVA inputs."""
    path = work / FIGURE_CACHE_SUBDIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def thin_segment(seg: np.ndarray, vertex_stride: int = AEM_VERTEX_STRIDE) -> np.ndarray:
    if vertex_stride <= 1 or len(seg) <= 2:
        return seg
    sampled = seg[::vertex_stride]
    if not np.array_equal(sampled[-1], seg[-1]):
        sampled = np.vstack([sampled, seg[-1]])
    return sampled


def write_gmt_lines(path: Path, segments: list[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            if len(seg) < 2:
                continue
            f.write(f"> segment_{i}\n")
            np.savetxt(f, seg[:, :2], fmt="%.7f %.7f")


def shape_segments_wgs84(shp_path: Path, extent: tuple[float, float, float, float] | None = None) -> list[np.ndarray]:
    reader = shapefile.Reader(str(shp_path))
    segments: list[np.ndarray] = []
    for shape_record in reader.shapeRecords():
        shape = shape_record.shape
        if not getattr(shape, "points", None) or not hasattr(shape, "bbox"):
            continue
        if extent is not None:
            xmin, ymin, xmax, ymax = shape.bbox
            lon_min, lon_max, lat_min, lat_max = extent
            if xmax < lon_min - 0.4 or xmin > lon_max + 0.4 or ymax < lat_min - 0.4 or ymin > lat_max + 0.4:
                continue
        pts = np.asarray(shape.points, dtype=np.float64)
        parts = list(shape.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            seg = pts[parts[i] : parts[i + 1]]
            if len(seg) >= 2:
                segments.append(seg)
    return segments


def ensure_aem_flightlines(root: Path, item_id: str, stem: str) -> Path:
    """Download one original USGS AEM flight-line shapefile if it is absent."""
    out_dir = root / AEM_FLIGHTLINES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    shp = out_dir / f"{stem}.shp"
    required_exts = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx")
    required_paths = [out_dir / f"{stem}{ext}" for ext in required_exts]
    if all(path.exists() and path.stat().st_size > 0 for path in required_paths):
        return shp

    for ext in required_exts:
        path = out_dir / f"{stem}{ext}"
        url = f"https://www.sciencebase.gov/catalog/file/get/{item_id}?name={stem}{ext}"
        expected = 0
        try:
            with urlopen(Request(url, method="HEAD"), timeout=60) as response:
                expected = int(response.headers.get("Content-Length") or 0)
            if path.exists() and (expected == 0 or path.stat().st_size == expected):
                continue
            with urlopen(url, timeout=60) as response:
                tmp = path.with_suffix(path.suffix + ".tmp")
                with tmp.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp.replace(path)
        except Exception as exc:
            raise RuntimeError(f"Could not download AEM flight-line file {path.name}") from exc

    return shp


def write_aem_flightline_segments(root: Path, work: Path) -> Path:
    """Write real USGS flight-line geometries as GMT segments in WGS84."""
    out = figure_cache_dir(work, "aem") / "aem_original_flightlines.gmt"
    lon_min, lon_max, lat_min, lat_max = REGION
    margin = 0.02

    segments: list[np.ndarray] = []
    release_counts: list[str] = []
    for item_id, stem in AEM_FLIGHTLINE_RELEASES:
        shp = ensure_aem_flightlines(root, item_id, stem)
        prj = shp.with_suffix(".prj")
        src_crs = CRS.from_wkt(prj.read_text(encoding="utf-8")) if prj.exists() else CRS.from_epsg(5070)
        transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)

        reader = shapefile.Reader(str(shp))
        n_before = len(segments)
        for shape in reader.shapes():
            pts = np.asarray(shape.points, dtype=np.float64)
            if pts.size == 0:
                continue
            parts = list(shape.parts) + [len(pts)]
            for i in range(len(parts) - 1):
                seg_xy = pts[parts[i] : parts[i + 1]]
                if len(seg_xy) < 2:
                    continue
                lon, lat = transformer.transform(seg_xy[:, 0], seg_xy[:, 1])
                coords = np.column_stack([lon, lat])
                keep = (
                    np.isfinite(coords[:, 0])
                    & np.isfinite(coords[:, 1])
                    & (coords[:, 0] >= lon_min - margin)
                    & (coords[:, 0] <= lon_max + margin)
                    & (coords[:, 1] >= lat_min - margin)
                    & (coords[:, 1] <= lat_max + margin)
                )
                if int(keep.sum()) < 2:
                    continue
                breaks = np.where(np.diff(np.flatnonzero(keep)) > 1)[0] + 1
                for run_idx in np.split(np.flatnonzero(keep), breaks):
                    if run_idx.size >= 2:
                        segments.append(thin_segment(coords[run_idx]))
        release_counts.append(f"{stem}: {len(segments) - n_before:,}")

    n_unthinned = len(segments)
    if AEM_SEGMENT_STRIDE > 1:
        segments = segments[::AEM_SEGMENT_STRIDE]

    write_gmt_lines(out, segments)
    print(
        "AEM original flight-line segments in map extent: "
        f"{len(segments):,} shown from {n_unthinned:,} "
        f"(segment stride={AEM_SEGMENT_STRIDE}, vertex stride={AEM_VERTEX_STRIDE})"
    )
    print("  " + "; ".join(release_counts))
    return out


def write_aem_lines(root: Path, work: Path) -> Path:
    """Prefer original flight lines; fall back to gridded coverage if unavailable."""
    try:
        return write_aem_flightline_segments(root, work)
    except Exception as exc:
        print(f"Original AEM flight lines unavailable; falling back to gridded traces. Reason: {exc}")
        return write_raw_aem_coverage_lines(
            root,
            work,
            row_step=env_int("FIG1A_AEM_RAW_ROW_STEP", 4),
            col_step=env_int("FIG1A_AEM_RAW_COL_STEP", 2),
        )


def river_segments(shp_path: Path, extent: tuple[float, float, float, float]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    reader = shapefile.Reader(str(shp_path))
    all_segments: list[np.ndarray] = []
    mississippi_segments: list[np.ndarray] = []
    for shape_record in reader.shapeRecords():
        shape = shape_record.shape
        if not getattr(shape, "points", None) or not hasattr(shape, "bbox"):
            continue
        xmin, ymin, xmax, ymax = shape.bbox
        lon_min, lon_max, lat_min, lat_max = extent
        if xmax < lon_min - 0.4 or xmin > lon_max + 0.4 or ymax < lat_min - 0.4 or ymin > lat_max + 0.4:
            continue
        record = shape_record.record.as_dict()
        text = " ".join(str(record.get(k, "")) for k in ["name", "name_en", "label", "name_alt"]).lower()
        is_mississippi = "mississippi" in text
        pts = np.asarray(shape.points, dtype=np.float64)
        parts = list(shape.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            seg = pts[parts[i] : parts[i + 1]]
            if len(seg) < 2:
                continue
            if is_mississippi:
                mississippi_segments.append(seg)
            else:
                all_segments.append(seg)
    return all_segments, mississippi_segments


def ensure_hydrorivers(root: Path, work: Path | None = None) -> Path:
    if work is None:
        river_dir = root / "data_raw" / "11 river_network" / "HydroRIVERS_NorthAmerica"
    else:
        river_dir = figure_cache_dir(work, "hydrography") / "raw" / "HydroRIVERS_NorthAmerica"
    shp = river_dir / "HydroRIVERS_v10_na_shp" / "HydroRIVERS_v10_na.shp"
    if shp.exists():
        return shp

    river_dir.mkdir(parents=True, exist_ok=True)
    zip_path = river_dir / "HydroRIVERS_v10_na_shp.zip"
    if not zip_path.exists():
        existing_zip = root / "data_raw" / "11 river_network" / "HydroRIVERS_NorthAmerica" / "HydroRIVERS_v10_na_shp.zip"
        if existing_zip.exists() and existing_zip.resolve() != zip_path.resolve():
            print("Copying HydroRIVERS North America into the Fig1a hydrography cache...")
            shutil.copy2(existing_zip, zip_path)
        else:
            print("Downloading HydroRIVERS North America into the Fig1a hydrography cache...")
            urlretrieve(HYDRORIVERS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(river_dir)
    if not shp.exists():
        raise FileNotFoundError(f"HydroRIVERS shapefile was not found after extraction: {shp}")
    return shp


def hydrorivers_order_segments(
    shp_path: Path,
    extent: tuple[float, float, float, float],
    keep_near: Polygon | None = None,
    orders: tuple[int, ...] = (1, 2, 3),
) -> dict[int, list[np.ndarray]]:
    reader = shapefile.Reader(str(shp_path))
    fields = [field[0] for field in reader.fields[1:]]
    ord_idx = fields.index("ORD_CLAS")
    segments: dict[int, list[np.ndarray]] = {order: [] for order in orders}
    lon_min, lon_max, lat_min, lat_max = extent
    prepared_keep = prep(keep_near) if keep_near is not None else None

    for shape_record in reader.iterShapeRecords():
        shape = shape_record.shape
        if not getattr(shape, "points", None) or not hasattr(shape, "bbox"):
            continue
        rec = shape_record.record
        try:
            order = int(rec[ord_idx])
        except Exception:
            continue
        if order not in segments:
            continue
        xmin, ymin, xmax, ymax = shape.bbox
        if xmax < lon_min - 0.15 or xmin > lon_max + 0.15 or ymax < lat_min - 0.15 or ymin > lat_max + 0.15:
            continue

        pts = np.asarray(shape.points, dtype=np.float64)
        parts = list(shape.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            seg = pts[parts[i] : parts[i + 1]]
            if len(seg) >= 2:
                if prepared_keep is not None and not prepared_keep.intersects(LineString(seg)):
                    continue
                segments[order].append(seg)
    return segments


def projected_boundary_segments(shp_path: Path, prj_path: Path) -> list[np.ndarray]:
    src_crs = CRS.from_wkt(prj_path.read_text(encoding="utf-8")) if prj_path.exists() else CRS.from_epsg(5070)
    transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
    reader = shapefile.Reader(str(shp_path))
    segments: list[np.ndarray] = []
    for shape in reader.shapes():
        pts = np.asarray(shape.points, dtype=np.float64)
        if pts.size == 0:
            continue
        parts = list(shape.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            seg = pts[parts[i] : parts[i + 1]]
            lon, lat = transformer.transform(seg[:, 0], seg[:, 1])
            segments.append(np.column_stack([lon, lat]))
    return segments


def projected_boundary_polygon_wgs84(shp_path: Path, prj_path: Path) -> Polygon:
    src_crs = CRS.from_wkt(prj_path.read_text(encoding="utf-8")) if prj_path.exists() else CRS.from_epsg(5070)
    transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
    reader = shapefile.Reader(str(shp_path))
    shape = reader.shapes()[0]
    pts = np.asarray(shape.points, dtype=np.float64)
    lon, lat = transformer.transform(pts[:, 0], pts[:, 1])
    return Polygon(np.column_stack([lon, lat]))


def write_text(path: Path, rows: list[tuple[float, float, str, float, str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for lon, lat, font, angle, justify, text in rows:
            f.write(f"{lon:.5f} {lat:.5f} {font} {angle:.1f} {justify} {text}\n")


def write_labels(work: Path) -> None:
    state_rows = [
        (-93.18, 36.55, f"{state_pt(8)},{FONT_ITALIC},{STATE_LABEL_COLOR}", 0, "CM", "Missouri"),
        (-93.16, 34.05, f"{state_pt(8)},{FONT_ITALIC},{STATE_LABEL_COLOR}", 0, "CM", "Arkansas"),
        (-93.58, 31.55, f"{state_pt(5)},{FONT_ITALIC},{STATE_LABEL_COLOR}", 0, "CM", "Louisiana"),
        (-88.92, 33.08, f"{state_pt(7)},{FONT_ITALIC},{STATE_LABEL_COLOR}", 0, "CM", "Mississippi"),
        (-87.42, 33.35, f"{state_pt(5)},{FONT_ITALIC},{STATE_LABEL_COLOR}", 0, "CM", "Alabama"),
        (-87.82, 35.45, f"{state_pt(7)},{FONT_ITALIC},{STATE_LABEL_COLOR}", 0, "CM", "Tennessee"),
        (-88.20, 36.86, f"{state_pt(7)},{FONT_ITALIC},{STATE_LABEL_COLOR}", 0, "CM", "Kentucky"),
        (-89.50, 38.20, f"{state_pt(7)},{FONT_ITALIC},{STATE_LABEL_COLOR}", 0, "CM", "Illinois"),
    ]
    lon_min, lon_max, lat_min, lat_max = REGION
    state_rows = [
        row
        for row in state_rows
        if lon_min <= row[0] <= lon_max and lat_min <= row[1] <= lat_max
    ]
    write_text(work / "state_labels.txt", state_rows)


def legend_pen_parts(pen: str, fallback_width: str, fallback_color: str) -> tuple[str, str]:
    parts = [part.strip() for part in pen.split(",") if part.strip()]
    width = parts[0] if parts else fallback_width
    color = parts[1] if len(parts) > 1 else fallback_color
    if color.lower() in {"none", "off", "false", "0", "transparent"}:
        color = fallback_color
    return width, color


def prepare_full_region_dem(gmt: Path, work: Path) -> Path:
    """Cut a full-frame GMT relief grid instead of rebuilding DEM only inside the analysis mask."""
    topo_cache = figure_cache_dir(work, "topography")
    token = f"{safe_region_token()}_{DEM_SIZE.replace(',', 'x')}"
    dem_tif = topo_cache / f"fig1a_3dep_dem_wgs84_{token}.tif"
    dem_grid = topo_cache / f"fig1a_3dep_dem_gmt_{token}.nc"
    if dem_grid.exists():
        return dem_grid

    params = {
        "f": "json",
        "bbox": f"{REGION[0]},{REGION[2]},{REGION[1]},{REGION[3]}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": DEM_SIZE,
        "format": "tiff",
        "pixelType": "F32",
        "noData": "-9999",
        "interpolation": "RSP_BilinearInterpolation",
    }
    if not dem_tif.exists():
        request_url = f"{TNM_ELEVATION_EXPORT_URL}?{urlencode(params)}"
        with urlopen(request_url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        href = payload.get("href")
        if not href:
            raise RuntimeError(f"3DEP export did not return a GeoTIFF URL: {payload}")

        tmp = dem_tif.with_suffix(".download.tif")
        urlretrieve(href, tmp)
        tmp.replace(dem_tif)

    xyz = topo_cache / f"fig1a_3dep_dem_{token}.xyz"
    with rasterio.open(dem_tif) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        transform = src.transform
        x = transform.c + (np.arange(src.width) + 0.5) * transform.a
        y = transform.f + (np.arange(src.height) + 0.5) * transform.e
        xx, yy = np.meshgrid(x, y)
        valid = np.isfinite(arr)
        np.savetxt(xyz, np.column_stack([xx[valid], yy[valid], arr[valid]]), fmt="%.8f %.8f %.3f")
        region = f"{src.bounds.left}/{src.bounds.right}/{src.bounds.bottom}/{src.bounds.top}"
        inc = f"{abs(transform.a)}/{abs(transform.e)}"

    try:
        run_gmt(gmt, ["xyz2grd", str(xyz), f"-G{dem_grid}", f"-R{region}", f"-I{inc}", "-r"], work)
    finally:
        if xyz.exists():
            xyz.unlink()

    return dem_grid


def _aem_crs_from_dataset(ds) -> CRS:
    spatial_ref = ds.get("spatial_ref")
    if spatial_ref is not None:
        attrs = spatial_ref.attrs
        for key in ("crs_wkt", "spatial_ref"):
            if attrs.get(key):
                try:
                    return CRS.from_wkt(attrs[key])
                except Exception:
                    pass

    # Define the AEM grid projection.
    return CRS.from_proj4(
        "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 "
        "+lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
    )


def write_raw_aem_coverage_lines(root: Path, work: Path, row_step: int = 4, col_step: int = 2) -> Path:
    """Approximate AEM survey traces from continuous valid rows in the raw AEM grid."""
    import xarray as xr

    raw_path = root / RAW_AEM_NC
    out = figure_cache_dir(work, "aem") / "aem_raw_coverage_lines.gmt"
    if not raw_path.exists():
        out.write_text("", encoding="utf-8")
        print(f"Raw AEM grid not found, skipping AEM trace lines: {raw_path}")
        return out

    with xr.open_dataset(raw_path, mask_and_scale=True) as ds:
        valid = np.isfinite(np.asarray(ds["log10res"].isel(z=0).values))
        x = np.asarray(ds["x"].values, dtype=np.float64)
        y = np.asarray(ds["y"].values, dtype=np.float64)
        src_crs = _aem_crs_from_dataset(ds)

    transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
    lon_min, lon_max, lat_min, lat_max = REGION
    row_step = max(int(row_step), 1)
    col_step = max(int(col_step), 1)

    segments: list[np.ndarray] = []
    for row in range(0, valid.shape[0], row_step):
        cols = np.flatnonzero(valid[row])
        if cols.size < 2:
            continue
        breaks = np.where(np.diff(cols) > 1)[0] + 1
        for run in np.split(cols, breaks):
            if run.size < 3:
                continue
            sampled_cols = run[::col_step]
            if sampled_cols[-1] != run[-1]:
                sampled_cols = np.r_[sampled_cols, run[-1]]
            xx = x[sampled_cols]
            yy = np.full_like(xx, y[row], dtype=np.float64)
            lon, lat = transformer.transform(xx, yy)
            in_region = (
                np.isfinite(lon)
                & np.isfinite(lat)
                & (lon >= lon_min)
                & (lon <= lon_max)
                & (lat >= lat_min)
                & (lat <= lat_max)
            )
            if int(in_region.sum()) >= 2:
                segments.append(np.column_stack([lon[in_region], lat[in_region]]))

    write_gmt_lines(out, segments)
    print(f"AEM raw gridded trace-like segments in map extent: {len(segments):,} (row step={row_step})")
    return out


def write_well_points(root: Path, work: Path) -> Path:
    candidates = [
        root / "data" / "2 well WTD" / "2011_2023" / "well_lookup.csv",
        root / "data" / "2 well WTD" / "shallow_only_2011_2023" / "well_lookup.csv",
    ]
    well_lookup = next((path for path in candidates if path.exists()), None)
    if well_lookup is None:
        raise FileNotFoundError("Could not find a well_lookup.csv in the expected WTD input folders.")
    wells = pd.read_csv(well_lookup)
    wells = wells[wells["target_filter_keep"].astype(bool) & wells["is_representative_site"].astype(bool)].copy()
    wells = wells.dropna(subset=["lon", "lat"])
    out = work / "wells.txt"
    wells[["lon", "lat"]].to_csv(out, sep=" ", index=False, header=False, float_format="%.7f")
    return out


def write_map_layers(root: Path, work: Path) -> dict[str, Path]:
    states_shp = root / "data_raw" / "natural_earth" / "states" / "ne_10m_admin_1_states_provinces_lines.shp"
    rivers_shp = root / "data_raw" / "natural_earth" / "rivers" / "ne_10m_rivers_lake_centerlines.shp"
    boundary_shp = root / "data" / "0 mask MRVA" / "outerboundary.shp"
    boundary_prj = root / "data" / "0 mask MRVA" / "outerboundary.prj"

    hydro_cache = figure_cache_dir(work, "hydrography")
    context_cache = figure_cache_dir(work, "map_context")
    paths = {
        "states": context_cache / "states.gmt",
        "river_order1": hydro_cache / "fig1a_hydrorivers_order1.gmt",
        "river_order2": hydro_cache / "fig1a_hydrorivers_order2.gmt",
        "mississippi": hydro_cache / "fig1a_mississippi_river_highlight.gmt",
        "boundary": context_cache / "mrva_boundary.gmt",
    }
    write_gmt_lines(paths["states"], shape_segments_wgs84(states_shp, REGION))
    _, mississippi = river_segments(rivers_shp, REGION)
    write_gmt_lines(paths["mississippi"], mississippi)
    print(f"Mississippi River highlight segments in map extent: {len(mississippi):,}")

    # Load major rivers across the map extent.
    hydrorivers = hydrorivers_order_segments(ensure_hydrorivers(root, work), REGION, keep_near=None, orders=(1, 2))
    for order, segs in hydrorivers.items():
        write_gmt_lines(paths[f"river_order{order}"], segs)
        print(f"HydroRIVERS ORD_CLAS={order}: {len(segs):,} reaches in full map extent")

    boundary = projected_boundary_segments(boundary_shp, boundary_prj)
    write_gmt_lines(paths["boundary"], boundary)
    return paths


def write_cpt(work: Path) -> Path:
    cpt = work / "terrain_arcgis_like.cpt"
    cpt.write_text(
        "\n".join(
            [
                "# COLOR_MODEL = RGB",
                "0 82/147/169 10 100/164/161",
                "10 100/164/161 25 127/178/150",
                "25 127/178/150 55 156/194/139",
                "55 156/194/139 95 195/211/151",
                "95 195/211/151 150 216/201/148",
                "150 216/201/148 240 203/168/128",
                "240 203/168/128 360 165/132/109",
                "360 165/132/109 600 236/236/232",
                "B 82/147/169",
                "F 236/236/232",
                "N 255/255/255",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return cpt


def write_legend(work: Path) -> Path:
    legend = work / "legend.txt"
    boundary_width, boundary_color = legend_pen_parts(BOUNDARY_PEN, "0.9p", "black")
    aem_width, aem_color = legend_pen_parts(AEM_LINE_PEN, "0.35p", "#111111")
    legend.write_text(
        "\n".join(
            [
                f"S 0.12c - 0.45c {boundary_color} {boundary_width} 0.42c Study region",
                f"S 0.12c - 0.45c {aem_color} {aem_width} 0.42c AEM flight lines",
                "S 0.12c - 0.45c #045a8d 0.65p 0.42c River",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return legend


def write_context_cache_manifest(
    root: Path,
    work: Path,
    topo_grid: Path,
    aem_lines: Path,
    layers: dict[str, Path],
) -> Path:
    """Record Fig. 1a-specific derived inputs without mixing them into model data."""
    cache_root = work / FIGURE_CACHE_SUBDIR

    def rel(path: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return str(path)

    manifest = {
        "purpose": "Derived context layers used only for Fig. 1a map rendering.",
        "region_lonlat": {
            "west": REGION[0],
            "east": REGION[1],
            "south": REGION[2],
            "north": REGION[3],
        },
        "dem_size_pixels": DEM_SIZE,
        "cache_root": rel(cache_root),
        "topography": {
            "dem_grid_gmt": rel(topo_grid),
            "source": "USGS 3DEP elevation export, clipped to the Fig. 1a map extent",
        },
        "hydrography": {
            "order1_gmt": rel(layers["river_order1"]),
            "order2_gmt": rel(layers["river_order2"]),
            "source": "HydroRIVERS North America, ORD_CLAS 1-2, subset across the full Fig. 1a map extent",
        },
        "mississippi_highlight": {
            "gmt": rel(layers["mississippi"]),
            "source": "Named Mississippi River segments from the regional map-context river layer",
        },
        "aem": {
            "flightlines_gmt": rel(aem_lines),
        },
        "map_context": {
            "states_gmt": rel(layers["states"]),
            "mrva_boundary_gmt": rel(layers["boundary"]),
        },
    }
    out = cache_root / "fig1a_context_cache_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def make_figure(output_prefix: Path, dpi: int = 600) -> dict[str, Path]:
    root = repo_root()
    work = root / "outputs" / "figures" / "Fig1a_MRVA_study_system_GMT"
    work.mkdir(parents=True, exist_ok=True)

    print(
        "Fig1a config: "
        f"region={region_string()}, projection={PROJECTION}, DEM_SIZE={DEM_SIZE}, "
        f"font_scale={FONT_SCALE}, font_add_pt={FONT_ADD_PT}, "
        f"terrain_transparency={TERRAIN_TRANSPARENCY}, shade_intensity={SHADE_INTENSITY}, "
        f"AEM segment stride={AEM_SEGMENT_STRIDE}, "
        f"AEM vertex stride={AEM_VERTEX_STRIDE}, AEM pen={AEM_LINE_PEN}, "
        f"draw_wells={DRAW_WELLS}, draw_state_labels={DRAW_STATE_LABELS}, "
        f"draw_legend={DRAW_LEGEND}, export_dpi={int(dpi)}, cache={work / FIGURE_CACHE_SUBDIR}"
    )

    gmt = gmt_executable(root)
    env_path = os.environ.get("PATH", "")
    if gmt.is_absolute():
        os.environ["PATH"] = f"{gmt.parent};{env_path}"

    write_gmt_custom_fonts(work)
    topo_grid = prepare_full_region_dem(gmt, work)
    aem_lines = write_aem_lines(root, work)
    wells = write_well_points(root, work) if DRAW_WELLS else None
    if wells is None:
        stale_wells = work / "wells.txt"
        if stale_wells.exists():
            stale_wells.unlink()
    layers = write_map_layers(root, work)
    manifest = write_context_cache_manifest(root, work, topo_grid, aem_lines, layers)
    if DRAW_STATE_LABELS:
        write_labels(work)
    else:
        stale_labels = work / "state_labels.txt"
        if stale_labels.exists():
            stale_labels.unlink()
    cpt = write_cpt(work)
    legend = write_legend(work) if DRAW_LEGEND else None
    if legend is None:
        stale_legend = work / "legend.txt"
        if stale_legend.exists():
            stale_legend.unlink()

    shade_token = f"{safe_region_token()}_{DEM_SIZE.replace(',', 'x')}"
    shade = figure_cache_dir(work, "topography") / f"fig1a_topography_shade_{shade_token}.nc"
    out = output_prefix
    region = region_string()
    print(f"Fig1a context-cache manifest: {manifest}")

    run_gmt(gmt, ["begin", str(out), "png,pdf", f"E{int(dpi)}"], work)
    try:
        run_gmt(
            gmt,
            [
                "set",
                "MAP_FRAME_TYPE",
                MAP_FRAME_TYPE,
                "MAP_FRAME_WIDTH",
                MAP_FRAME_WIDTH,
                "MAP_FRAME_PEN",
                MAP_FRAME_PEN,
                "MAP_FRAME_PERCENT",
                MAP_FRAME_PERCENT,
                "FORMAT_GEO_MAP",
                "ddd:mmF",
                "FONT",
                FONT_REGULAR,
                "FONT_ANNOT_PRIMARY",
                f"{pt(7)},{FONT_REGULAR}",
                "FONT_LABEL",
                f"{pt(8)},{FONT_REGULAR}",
                "FONT_TITLE",
                f"{pt(9)},{FONT_BOLD}",
                "MAP_TICK_LENGTH_PRIMARY",
                MAP_TICK_LENGTH_PRIMARY,
            ],
            work,
        )
        run_gmt(gmt, ["grdgradient", str(topo_grid), "-A315", f"-Ne{SHADE_INTENSITY}", f"-G{shade}"], work)
        run_gmt(gmt, ["coast", f"-R{region}", f"-J{PROJECTION}", "-G#f7f3e7", "-S#eaf3f7", "-A1000", "-W0.2p,#cfcfcf", f"-N2/{STATE_BOUNDARY_PEN}"], work)
        grdimage_args = ["grdimage", str(topo_grid), f"-R{region}", f"-J{PROJECTION}", f"-C{cpt}", f"-I{shade}"]
        if TERRAIN_TRANSPARENCY > 0:
            grdimage_args.append(f"-t{TERRAIN_TRANSPARENCY:.0f}")
        run_gmt(gmt, grdimage_args, work)
        if gmt_fill_enabled(MRVA_FILL_CYAN):
            run_gmt(gmt, ["plot", str(layers["boundary"]), f"-R{region}", f"-J{PROJECTION}", f"-G{MRVA_FILL_CYAN}", "-W0"], work)
        if gmt_fill_enabled(MRVA_FILL_YELLOW):
            run_gmt(gmt, ["plot", str(layers["boundary"]), f"-R{region}", f"-J{PROJECTION}", f"-G{MRVA_FILL_YELLOW}", "-W0"], work)
        run_gmt(gmt, ["plot", str(aem_lines), f"-R{region}", f"-J{PROJECTION}", f"-W{AEM_LINE_PEN}"], work)
        run_gmt(gmt, ["plot", str(layers["river_order2"]), f"-R{region}", f"-J{PROJECTION}", f"-W{TRIBUTARY_ORDER2_PEN}"], work)
        run_gmt(gmt, ["plot", str(layers["river_order1"]), f"-R{region}", f"-J{PROJECTION}", f"-W{TRIBUTARY_ORDER1_PEN}"], work)
        if gmt_fill_enabled(MISSISSIPPI_HALO_PEN):
            run_gmt(gmt, ["plot", str(layers["mississippi"]), f"-R{region}", f"-J{PROJECTION}", f"-W{MISSISSIPPI_HALO_PEN}"], work)
        run_gmt(gmt, ["plot", str(layers["mississippi"]), f"-R{region}", f"-J{PROJECTION}", f"-W{MISSISSIPPI_PEN}"], work)
        run_gmt(gmt, ["plot", str(layers["states"]), f"-R{region}", f"-J{PROJECTION}", f"-W{STATE_BOUNDARY_PEN}"], work)
        if wells is not None:
            run_gmt(gmt, ["plot", str(wells), f"-R{region}", f"-J{PROJECTION}", f"-Sc{WELL_SYMBOL_SIZE}", "-G#2b2b2b@55", "-W0"], work)
        run_gmt(gmt, ["plot", str(layers["boundary"]), f"-R{region}", f"-J{PROJECTION}", f"-W{BOUNDARY_PEN}"], work)
        if DRAW_STATE_LABELS:
            run_gmt(gmt, ["text", str(work / "state_labels.txt"), f"-R{region}", f"-J{PROJECTION}", "-F+f+a+j", "-N"], work)
        run_gmt(
            gmt,
            [
                "basemap",
                f"-R{region}",
                f"-J{PROJECTION}",
                "-Bxa1.5f0.5",
                "-Bya2.5f0.5",
                "-BWSne",
                f"-LjBR+c32.0+w100k+f+lkm+ar+o{SCALEBAR_OFFSET}",
                f"-TdjTL+w{NORTH_ARROW_SIZE_CM:.2f}c+f2+l+o{NORTH_ARROW_OFFSET}",
            ],
            work,
        )
        if legend is not None:
            run_gmt(
                gmt,
                [
                    "legend",
                    str(legend),
                    f"-R{region}",
                    f"-J{PROJECTION}",
                    f"-DjBR+w{LEGEND_WIDTH_CM:.2f}c+o{LEGEND_OFFSET}",
                    "-C0.08c/0.07c",
                ],
                work,
            )
    finally:
        run_gmt(gmt, ["end"], work)

    png = out.with_suffix(".png")
    pdf = out.with_suffix(".pdf")
    return {"png": png, "pdf": pdf, "work_dir": work}


def parse_args() -> argparse.Namespace:
    root = repo_root()
    return argparse.Namespace(
        output_prefix=root / "outputs" / "figures" / "Fig1a_MRVA_study_system_GMT" / "Fig1a_MRVA_study_system_GMT",
        dpi=env_int("FIG1A_DPI", 600),
    )


def main() -> None:
    args = parse_args()
    outputs = make_figure(Path(args.output_prefix), dpi=int(args.dpi))
    for key, path in outputs.items():
        print(f"{key}: {path}")
    # Remove the unused PyGMT exit hook.
    try:
        import atexit
        import sys

        pygmt_session = sys.modules.get("pygmt.session_management")
        pygmt_end = getattr(pygmt_session, "end", None)
        if pygmt_end is not None:
            atexit.unregister(pygmt_end)
    except Exception:
        pass


if __name__ == "__main__":
    main()
