from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import shapefile  # pyshp
from pyproj import CRS, Transformer


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value in (None, "") else str(value)


def parse_region(value: str) -> tuple[float, float, float, float]:
    parts = value.replace(",", "/").split("/")
    if len(parts) != 4:
        raise ValueError("Region strings must be lon_min/lon_max/lat_min/lat_max")
    lon_min, lon_max, lat_min, lat_max = (float(part) for part in parts)
    return lon_min, lon_max, lat_min, lat_max


CONUS_REGION = env_str("FIG1A_POS_REGION", "-125/-66/24/50")
STUDY_REGION = env_str("FIG1A_POS_STUDY_REGION", "-94/-87/30.5/38.5")
MAP_WIDTH_CM = env_float("FIG1A_POS_MAP_WIDTH_CM", 7.2)
EXPORT_DPI = env_int("FIG1A_POS_DPI", 600)
PROJECTION = env_str("FIG1A_POS_PROJECTION", f"B-96/37/29.5/45.5/{MAP_WIDTH_CM:.2f}c")
LAND_FILL = env_str("FIG1A_POS_LAND_FILL", "#fbfbfb")
WATER_FILL = env_str("FIG1A_POS_WATER_FILL", "white")
STATE_PEN = env_str("FIG1A_POS_STATE_PEN", "0.28p,#d8d8d8")
USA_OUTLINE_PEN = env_str("FIG1A_POS_USA_OUTLINE_PEN", "0.45p,#cfcfcf")
BOX_FILL = env_str("FIG1A_POS_BOX_FILL", "#efb190@36")
BOX_PEN = env_str("FIG1A_POS_BOX_PEN", "0.55p,#ba6a52")
MRVA_FILL = env_str("FIG1A_POS_MRVA_FILL", "#c97154@30")
MRVA_PEN = env_str("FIG1A_POS_MRVA_PEN", "0.65p,#3b2018")
FONT_REGULAR = "ArialMT"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def gmt_executable(root: Path) -> Path:
    local = root / ".gmt_env" / "Library" / "bin" / "gmt.exe"
    if local.exists():
        return local
    found = shutil_which("gmt")
    if found:
        return Path(found)
    raise FileNotFoundError("GMT executable not found. Install GMT or add gmt.exe to PATH.")


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


def run_gmt(gmt: Path, args: list[str], cwd: Path) -> None:
    subprocess.run([str(gmt), *args], cwd=cwd, check=True)


def write_gmt_custom_fonts(work: Path) -> None:
    (work / "PSL_custom_fonts.txt").write_text("ArialMT 0.700 0\n", encoding="utf-8")


def write_gmt_segments(path: Path, segments: list[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            if len(seg) < 2:
                continue
            f.write(f"> segment_{i}\n")
            for lon, lat in seg:
                f.write(f"{lon:.7f} {lat:.7f}\n")


def natural_earth_usa_segments(
    shp_path: Path,
    extent: tuple[float, float, float, float],
) -> list[np.ndarray]:
    reader = shapefile.Reader(str(shp_path))
    lon_min, lon_max, lat_min, lat_max = extent
    state_segments: list[np.ndarray] = []
    for shape_record in reader.iterShapeRecords():
        record = shape_record.record.as_dict()
        if record.get("ADM0_A3") != "USA":
            continue
        shape = shape_record.shape
        if not getattr(shape, "points", None) or not hasattr(shape, "bbox"):
            continue
        xmin, ymin, xmax, ymax = shape.bbox
        if xmax < lon_min or xmin > lon_max or ymax < lat_min or ymin > lat_max:
            continue
        points = np.asarray(shape.points, dtype=np.float64)
        parts = list(shape.parts) + [len(points)]
        if not (record.get("NAME_L") and record.get("NAME_R")):
            continue
        for i in range(len(parts) - 1):
            seg = points[parts[i] : parts[i + 1]]
            if len(seg) >= 2:
                state_segments.append(seg)
    return state_segments


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


def write_study_box(path: Path, region: tuple[float, float, float, float]) -> None:
    lon_min, lon_max, lat_min, lat_max = region
    points = [
        (lon_min, lat_min),
        (lon_max, lat_min),
        (lon_max, lat_max),
        (lon_min, lat_max),
        (lon_min, lat_min),
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("> study_region_box\n")
        for lon, lat in points:
            f.write(f"{lon:.7f} {lat:.7f}\n")


def make_figure(output_prefix: Path, dpi: int = EXPORT_DPI) -> dict[str, Path]:
    root = repo_root()
    work = output_prefix.parent
    work.mkdir(parents=True, exist_ok=True)

    gmt = gmt_executable(root)
    env_path = os.environ.get("PATH", "")
    if gmt.is_absolute():
        os.environ["PATH"] = f"{gmt.parent};{env_path}"

    write_gmt_custom_fonts(work)
    study_box = work / "study_region_box.gmt"
    mrva_boundary = work / "mrva_boundary_wgs84.gmt"
    usa_states = work / "usa_state_boundaries.gmt"
    stale_outline = work / "usa_outline.gmt"
    if stale_outline.exists():
        stale_outline.unlink()
    write_study_box(study_box, parse_region(STUDY_REGION))
    states_shp = root / "data_raw" / "natural_earth" / "states" / "ne_10m_admin_1_states_provinces_lines.shp"
    state_segments = natural_earth_usa_segments(states_shp, parse_region(CONUS_REGION))
    write_gmt_segments(usa_states, state_segments)
    boundary_shp = root / "data" / "0 mask MRVA" / "outerboundary.shp"
    boundary_prj = root / "data" / "0 mask MRVA" / "outerboundary.prj"
    write_gmt_segments(mrva_boundary, projected_boundary_segments(boundary_shp, boundary_prj))

    print(
        "Fig1a position config: "
        f"region={CONUS_REGION}, projection={PROJECTION}, study_region={STUDY_REGION}, "
        f"usa_state_segments={len(state_segments)}, "
        f"box_fill={BOX_FILL}, mrva_pen={MRVA_PEN}, export_dpi={int(dpi)}"
    )

    run_gmt(gmt, ["begin", str(output_prefix), "png,pdf", f"E{int(dpi)}"], work)
    try:
        run_gmt(
            gmt,
            [
                "set",
                "MAP_FRAME_TYPE",
                "plain",
                "FONT",
                FONT_REGULAR,
                "MAP_FRAME_PEN",
                "0p,white",
            ],
            work,
        )
        run_gmt(gmt, ["basemap", f"-R{CONUS_REGION}", f"-J{PROJECTION}", f"-B+g{WATER_FILL}"], work)
        run_gmt(
            gmt,
            [
                "coast",
                f"-R{CONUS_REGION}",
                f"-J{PROJECTION}",
                "-A5000",
                f"-EUS+g{LAND_FILL}+p{USA_OUTLINE_PEN}",
            ],
            work,
        )
        run_gmt(gmt, ["plot", str(usa_states), f"-R{CONUS_REGION}", f"-J{PROJECTION}", f"-W{STATE_PEN}"], work)
        run_gmt(
            gmt,
            [
                "plot",
                str(study_box),
                f"-R{CONUS_REGION}",
                f"-J{PROJECTION}",
                f"-G{BOX_FILL}",
                f"-W{BOX_PEN}",
            ],
            work,
        )
        run_gmt(
            gmt,
            [
                "plot",
                str(mrva_boundary),
                f"-R{CONUS_REGION}",
                f"-J{PROJECTION}",
                f"-G{MRVA_FILL}",
                f"-W{MRVA_PEN}",
            ],
            work,
        )
    finally:
        run_gmt(gmt, ["end"], work)

    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    return {"png": png, "pdf": pdf, "work_dir": work}


def parse_args() -> argparse.Namespace:
    root = repo_root()
    return argparse.Namespace(
        output_prefix=root / "outputs" / "figures" / "Fig1a_Position_GMT" / "Fig1a_Position_GMT"
    )


def main() -> None:
    args = parse_args()
    outputs = make_figure(Path(args.output_prefix))
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
