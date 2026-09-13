from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from mainline_reconstruction import MAINLINE_RECON_ROOT


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID_LOOKUP = MAINLINE_RECON_ROOT / "metadata" / "grid_lookup.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "coordinate_audit"


@dataclass
class AuditRecord:
    path: str
    kind: str
    status: str
    issue: str
    crs: str = ""
    shape: str = ""
    bounds: str = ""
    n_rows: int | None = None
    n_cols: int | None = None
    valid_fraction: float | None = None
    direct_mask_match: float | None = None
    northup_mask_match: float | None = None
    max_abs_dx_m: float | None = None
    max_abs_dy_m: float | None = None
    notes: str = ""


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def load_grid(grid_lookup: Path) -> pd.DataFrame:
    grid = pd.read_csv(grid_lookup)
    required = {"grid_id", "row", "col", "x", "y"}
    missing = sorted(required - set(grid.columns))
    if missing:
        raise ValueError(f"{grid_lookup} missing required columns: {missing}")
    return grid.sort_values("grid_id").reset_index(drop=True)


def grid_masks(grid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n_rows = int(grid["row"].max()) + 1
    n_cols = int(grid["col"].max()) + 1
    direct = np.zeros((n_rows, n_cols), dtype=bool)
    northup = np.zeros((n_rows, n_cols), dtype=bool)
    rr = grid["row"].to_numpy(dtype=np.int64)
    cc = grid["col"].to_numpy(dtype=np.int64)
    direct[rr, cc] = True
    northup[n_rows - 1 - rr, cc] = True
    return direct, northup


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / union)


def audit_grid_lookup(grid_lookup: Path, grid: pd.DataFrame) -> list[AuditRecord]:
    row_y_corr = float(grid[["row", "y"]].corr().loc["row", "y"])
    col_x_corr = float(grid[["col", "x"]].corr().loc["col", "x"])
    status = "PASS" if row_y_corr > 0.999 and col_x_corr > 0.999 else "WARN"
    issue = "grid row increases northward; GeoTIFF exports must invert row index" if status == "PASS" else "row/col are not monotonic with x/y"
    return [
        AuditRecord(
            path=rel(grid_lookup),
            kind="grid_lookup",
            status=status,
            issue=issue,
            n_rows=int(grid["row"].max()) + 1,
            n_cols=int(grid["col"].max()) + 1,
            notes=f"row-y corr={row_y_corr:.6f}; col-x corr={col_x_corr:.6f}",
        )
    ]


def audit_csv_coordinates(path: Path, grid: pd.DataFrame) -> AuditRecord | None:
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return None
    cols = set(header.columns)
    grid_id_col = "grid_id" if "grid_id" in cols else ("node_id" if "node_id" in cols else None)
    x_col = "x" if "x" in cols else ("x_center" if "x_center" in cols else None)
    y_col = "y" if "y" in cols else ("y_center" if "y_center" in cols else None)
    if grid_id_col is None:
        return None
    if {"distance_km", "profile_id"}.issubset(cols):
        return AuditRecord(
            rel(path),
            "csv_profile_coordinates",
            "PASS",
            "transect/profile coordinates are sampled points; grid_id is nearest-cell reference, not exact x/y",
        )
    has_xy = x_col is not None and y_col is not None
    has_rowcol = {"row", "col"}.issubset(cols)
    if not (has_xy or has_rowcol):
        return None

    usecols = [grid_id_col] + [c for c in [x_col, y_col, "row", "col"] if c is not None and c in cols]
    try:
        df = pd.read_csv(path, usecols=usecols)
    except Exception as exc:
        return AuditRecord(rel(path), "csv_grid_coordinates", "WARN", f"could not read coordinate columns: {exc}")
    rename = {grid_id_col: "grid_id"}
    if x_col:
        rename[x_col] = "x"
    if y_col:
        rename[y_col] = "y"
    df = df.rename(columns=rename)
    if df.empty:
        return AuditRecord(rel(path), "csv_grid_coordinates", "WARN", "empty file")

    g = grid[["grid_id", "x", "y", "row", "col"]]
    merged = df.merge(g, on="grid_id", how="inner", suffixes=("", "_grid"))
    if merged.empty:
        return AuditRecord(rel(path), "csv_grid_coordinates", "WARN", "no grid_id overlap with active grid")

    max_abs_dx = None
    max_abs_dy = None
    notes = [f"grid_id overlap {len(merged):,}/{len(df):,} rows"]
    status = "PASS"
    issues: list[str] = []
    if {"x", "y"}.issubset(df.columns):
        dx = np.abs(merged["x"].to_numpy(dtype=float) - merged["x_grid"].to_numpy(dtype=float))
        dy = np.abs(merged["y"].to_numpy(dtype=float) - merged["y_grid"].to_numpy(dtype=float))
        max_abs_dx = float(np.nanmax(dx))
        max_abs_dy = float(np.nanmax(dy))
        notes.append(f"max |dx|={max_abs_dx:.3f} m; max |dy|={max_abs_dy:.3f} m")
        if max_abs_dx > 1e-4 or max_abs_dy > 1e-4:
            status = "FAIL"
            issues.append("x/y do not match grid_lookup for same grid_id")
    if {"row", "col"}.issubset(df.columns):
        dr = np.abs(merged["row"].to_numpy(dtype=float) - merged["row_grid"].to_numpy(dtype=float))
        dc = np.abs(merged["col"].to_numpy(dtype=float) - merged["col_grid"].to_numpy(dtype=float))
        notes.append(f"max |drow|={float(np.nanmax(dr)):.3f}; max |dcol|={float(np.nanmax(dc)):.3f}")
        if np.nanmax(dr) > 0 or np.nanmax(dc) > 0:
            status = "FAIL"
            issues.append("row/col do not match grid_lookup for same grid_id")
    return AuditRecord(
        path=rel(path),
        kind="csv_grid_coordinates",
        status=status,
        issue="; ".join(issues) if issues else "grid_id coordinate columns agree with active grid",
        max_abs_dx_m=max_abs_dx,
        max_abs_dy_m=max_abs_dy,
        notes=" | ".join(notes),
    )


def audit_raster(path: Path, grid_direct: np.ndarray, grid_northup: np.ndarray) -> AuditRecord:
    try:
        with rasterio.open(path) as ds:
            arr = ds.read(1, masked=True)
            valid = ~np.asarray(arr.mask, dtype=bool)
            if arr.mask is np.ma.nomask:
                valid = np.ones(arr.shape, dtype=bool)
            if ds.nodata is not None:
                data = np.asarray(arr.filled(ds.nodata))
                valid &= data != ds.nodata
            shape = f"{ds.height}x{ds.width}"
            bounds = ",".join(f"{v:.3f}" for v in ds.bounds)
            crs = str(ds.crs)
            valid_fraction = float(valid.mean())
            direct_match = jaccard(valid, grid_direct)
            northup_match = jaccard(valid, grid_northup)
    except Exception as exc:
        return AuditRecord(path=rel(path), kind="raster", status="WARN", issue=f"could not read raster: {exc}")

    status = "PASS"
    issue = "raster readable"
    notes = ""
    if crs and "5070" in crs and valid.shape == grid_direct.shape:
        if math.isfinite(northup_match) and northup_match > 0.98:
            issue = "valid mask aligns with north-up grid orientation"
        elif math.isfinite(direct_match) and direct_match > 0.98 and direct_match > northup_match + 0.05:
            status = "FAIL"
            issue = "valid mask aligns with grid row order but GeoTIFF is north-up; likely north/south georeference mismatch"
        elif valid_fraction > 0.98:
            issue = "full-rectangle raster; mask orientation not diagnostic"
        else:
            status = "WARN"
            issue = "valid mask does not clearly match active grid"
        notes = f"direct J={direct_match:.4f}; north-up J={northup_match:.4f}"
    elif crs and "4326" in crs:
        issue = "lon/lat raster; not compared to EPSG:5070 mask"
    else:
        if crs and "5070" in crs and (valid.shape[1] == grid_direct.shape[1]) and abs(valid.shape[0] - grid_direct.shape[0]) <= 2:
            status = "WARN"
            issue = "EPSG:5070 raster has near-matching 1 km grid shape but different row count; inspect export transform"
        notes = "shape or CRS does not match active 1 km EPSG:5070 grid"

    return AuditRecord(
        path=rel(path),
        kind="raster",
        status=status,
        issue=issue,
        crs=crs,
        shape=shape,
        bounds=bounds,
        valid_fraction=valid_fraction,
        direct_mask_match=direct_match,
        northup_mask_match=northup_match,
        notes=notes,
    )


def audit_topography_semantics(topo_csv: Path, grid: pd.DataFrame) -> AuditRecord:
    if not topo_csv.exists():
        return AuditRecord(rel(topo_csv), "topography_semantic", "WARN", "topo_1km.csv missing")
    topo = pd.read_csv(topo_csv)
    if not {"grid_id", "mean_elevation_m"}.issubset(topo.columns):
        return AuditRecord(rel(topo_csv), "topography_semantic", "WARN", "topo_1km.csv missing expected columns")
    merged = grid.merge(topo[["grid_id", "mean_elevation_m"]], on="grid_id", how="left")
    transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    imax = int(merged["mean_elevation_m"].idxmax())
    row = merged.loc[imax]
    lon, lat = transformer.transform(float(row["x"]), float(row["y"]))
    grid0 = merged.iloc[0]
    status = "PASS"
    issues = []
    if float(row["mean_elevation_m"]) > 250 and lat < 34.5:
        status = "FAIL"
        issues.append("highest elevation is attached too far south")
    if float(grid0["mean_elevation_m"]) > 100:
        status = "FAIL"
        issues.append("southern grid_id 0 has implausibly high elevation")
    if not issues:
        issues.append("topography max/min placement is plausible against MRVA regional relief")
    return AuditRecord(
        path=rel(topo_csv),
        kind="topography_semantic",
        status=status,
        issue="; ".join(issues),
        notes=(
            f"max elevation {float(row['mean_elevation_m']):.2f} m at "
            f"grid_id={int(row['grid_id'])}, lat={lat:.4f}, lon={lon:.4f}; "
            f"grid_id0 elevation={float(grid0['mean_elevation_m']):.2f} m"
        ),
    )


def candidate_csvs() -> Iterable[Path]:
    for base in [REPO_ROOT / "data", REPO_ROOT / "outputs"]:
        if not base.exists():
            continue
        yield from base.rglob("*.csv")


def candidate_rasters() -> Iterable[Path]:
    for base in [REPO_ROOT / "data", REPO_ROOT / "outputs"]:
        if not base.exists():
            continue
        for pattern in ["*.tif", "*.tiff"]:
            yield from base.rglob(pattern)


def write_reports(records: list[AuditRecord], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in records]
    csv_path = out_dir / "coordinate_audit_records.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = pd.DataFrame(rows)
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return ""
        cols = list(frame.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in frame.iterrows():
            vals = [str(row[col]).replace("\n", " ") for col in cols]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    md = ["# Coordinate Integrity Audit", ""]
    md.append("## Status Counts")
    counts = summary["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count")
    md.append(markdown_table(counts))
    md.append("")
    md.append("## Failures And Warnings")
    subset = summary[summary["status"].isin(["FAIL", "WARN"])][["status", "kind", "path", "issue", "notes"]]
    if subset.empty:
        md.append("No failures or warnings.")
    else:
        md.append(markdown_table(subset))
    md.append("")
    md.append("## All Records")
    md.append(markdown_table(summary[["status", "kind", "path", "issue", "notes"]]))
    (out_dir / "coordinate_audit_report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {out_dir / 'coordinate_audit_report.md'}")
    print(summary["status"].value_counts(dropna=False).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit coordinate/grid orientation integrity for MRVA project files.")
    parser.add_argument("--grid-lookup", type=Path, default=DEFAULT_GRID_LOOKUP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rasters", type=int, default=500, help="Maximum number of rasters to audit.")
    args = parser.parse_args()

    grid_lookup = args.grid_lookup.resolve()
    out_dir = args.out_dir.resolve()
    grid = load_grid(grid_lookup)
    grid_direct, grid_northup = grid_masks(grid)

    records: list[AuditRecord] = []
    records.extend(audit_grid_lookup(grid_lookup, grid))
    records.append(audit_topography_semantics(REPO_ROOT / "data" / "5 topography" / "topo_1km.csv", grid))

    seen: set[Path] = set()
    for path in candidate_csvs():
        if path in seen:
            continue
        seen.add(path)
        rec = audit_csv_coordinates(path, grid)
        if rec is not None:
            records.append(rec)

    rasters = list(candidate_rasters())
    priority = []
    for path in rasters:
        p = str(path).replace("\\", "/")
        if any(
            key in p
            for key in [
                "data/5 topography",
                "data/1 resistivity",
                "data/3 SWB",
                "data/8 connectivity",
                "data/10 precipitation",
                "map_exports/geotiff",
                "response_metrics",
            ]
        ):
            priority.append(path)
    for path in priority[: args.max_rasters]:
        records.append(audit_raster(path, grid_direct, grid_northup))

    write_reports(records, out_dir)


if __name__ == "__main__":
    main()
