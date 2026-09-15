from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None


ROOT = Path(__file__).resolve().parents[1]
PUMPING_DIR = ROOT / "data" / "4 pumping out"
REPRO_ROOT = ROOT / "data_raw" / "4__AIWUM1.0_reproduce"
AIWUM1_REPO = REPRO_ROOT / "source" / "aiwum1"
REFERENCE_TIF = PUMPING_DIR / "2014" / "AIWUM2_1km_m3_2014_Jan.tif"
IDOMAIN_PATH = PUMPING_DIR / "idomain_1km_target.dat"

TARGET_YEARS = [2011, 2012, 2013]
OVERLAP_YEARS = [2014, 2015, 2016, 2017]
AIWUM2_CLIM_YEARS = list(range(2014, 2024))
GROWING_MONTHS = list(range(4, 11))
NONGROW_MONTHS = [1, 2, 3, 11]

MONTH_NUM_TO_NAME = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge-correct AIWUM1 2011-2013 pumping to the AIWUM2 2014-2023 style "
            "using AIWUM1-vs-AIWUM2 overlap years."
        )
    )
    parser.add_argument("--pumping-dir", type=Path, default=PUMPING_DIR)
    parser.add_argument("--aiwum1-repo", type=Path, default=AIWUM1_REPO)
    parser.add_argument("--reference-tif", type=Path, default=REFERENCE_TIF)
    parser.add_argument("--idomain-path", type=Path, default=IDOMAIN_PATH)
    parser.add_argument("--overlap-folder", default="MRVA_2014_2017_km")
    parser.add_argument("--source-folder", default="MRVA_2011_2013_km")
    parser.add_argument("--sigma-pixels", type=float, default=8.0)
    parser.add_argument("--factor-min", type=float, default=0.2)
    parser.add_argument("--factor-max", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def target_profile_and_mask(reference_tif: Path, idomain_path: Path) -> tuple[dict, np.ndarray]:
    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()
    profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate", interleave="pixel")

    idomain = np.loadtxt(idomain_path)
    mask = np.flipud(idomain) != 0
    if mask.shape != (profile["height"], profile["width"]):
        raise ValueError(f"Mask shape {mask.shape} does not match reference raster.")
    return profile, mask


def clean_array(arr: np.ndarray, nodata: float | int | None, mask: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    if nodata is not None:
        out[out == nodata] = np.nan
    out[~mask] = np.nan
    return out


def read_target_grid(path: Path, profile: dict, mask: np.ndarray) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        if (
            src.width == profile["width"]
            and src.height == profile["height"]
            and src.transform == profile["transform"]
            and str(src.crs) == str(profile["crs"])
        ):
            return clean_array(src.read(1), src.nodata, mask)

        nodata = float(profile["nodata"])
        dst = np.full((profile["height"], profile["width"]), nodata, dtype=np.float32)
        reproject(
            source=src.read(1).astype(np.float32),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            dst_nodata=nodata,
            resampling=Resampling.average,
        )
    return clean_array(dst, nodata, mask)


def aiwum1_source_path(repo: Path, folder: str, year: int, month: int) -> Path:
    return repo / "Output" / folder / "rasters" / "km" / "Annual_totals" / f"y{year}_m{month}_tot.tif"


def aiwum1_local_path(pumping_dir: Path, year: int, month: int) -> Path:
    return pumping_dir / str(year) / f"AIWUM1_1km_m3_{year}_{MONTH_NUM_TO_NAME[month]}.tif"


def aiwum2_local_path(pumping_dir: Path, year: int, month: int) -> Path:
    return pumping_dir / str(year) / f"AIWUM2_1km_m3_{year}_{MONTH_NUM_TO_NAME[month]}.tif"


def masked_sum(arr: np.ndarray) -> float:
    return float(np.nansum(arr))


def masked_gaussian(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    if gaussian_filter is None or sigma <= 0:
        return np.where(valid, values, np.nan).astype(np.float32)
    weights = valid.astype(np.float32)
    numer = gaussian_filter(np.where(valid, values, 0.0).astype(np.float32), sigma=sigma)
    denom = gaussian_filter(weights, sigma=sigma)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    np.divide(numer, denom, out=out, where=denom > 1.0e-6)
    return out


def write_raster(path: Path, arr: np.ndarray, profile: dict, mask: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Pass --overwrite to replace it.")
    out = arr.astype(np.float32, copy=True)
    out[~mask] = float(profile["nodata"])
    out[~np.isfinite(out)] = float(profile["nodata"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def backup_raw_files(pumping_dir: Path, years: list[int], overwrite: bool) -> Path:
    backup_root = pumping_dir / "_aiwum1_raw_2011_2013_before_bridge_calibration"
    backup_root.mkdir(parents=True, exist_ok=True)
    for year in years:
        for month in range(1, 13):
            src = aiwum1_local_path(pumping_dir, year, month)
            if not src.exists():
                continue
            dst = backup_root / str(year) / src.name
            if dst.exists() and not overwrite:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return backup_root


def main() -> None:
    args = parse_args()
    profile, mask = target_profile_and_mask(args.reference_tif, args.idomain_path)
    nodata = float(profile["nodata"])

    diagnostics_dir = args.pumping_dir / "_calibration_metadata"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    month_factors: dict[int, np.ndarray] = {}
    month_rows: list[dict] = []

    for month in GROWING_MONTHS:
        a1_stack = []
        a2_stack = []
        for year in OVERLAP_YEARS:
            a1_stack.append(read_target_grid(aiwum1_source_path(args.aiwum1_repo, args.overlap_folder, year, month), profile, mask))
            a2_stack.append(read_target_grid(aiwum2_local_path(args.pumping_dir, year, month), profile, mask))
        a1 = np.nanmean(np.stack(a1_stack, axis=0), axis=0)
        a2 = np.nanmean(np.stack(a2_stack, axis=0), axis=0)

        total_a1 = masked_sum(a1)
        total_a2 = masked_sum(a2)
        total_scale = total_a2 / total_a1 if total_a1 > 0 else 1.0

        positive = mask & np.isfinite(a1) & np.isfinite(a2) & (a1 > 1.0)
        ratio = np.full(a1.shape, np.nan, dtype=np.float32)
        ratio[positive] = a2[positive] / np.maximum(a1[positive], 1.0e-6)
        smoothed = masked_gaussian(ratio, positive, args.sigma_pixels)
        factor = np.where(np.isfinite(smoothed), smoothed, total_scale).astype(np.float32)
        factor = np.clip(factor, args.factor_min, args.factor_max)
        factor[~mask] = np.nan
        month_factors[month] = factor

        month_rows.append(
            {
                "month": month,
                "month_name": MONTH_NUM_TO_NAME[month],
                "overlap_aiwum1_mean_total_m3": total_a1,
                "overlap_aiwum2_mean_total_m3": total_a2,
                "monthly_total_scale_aiwum2_over_aiwum1": total_scale,
                "factor_p05": float(np.nanpercentile(factor[mask], 5)),
                "factor_p50": float(np.nanpercentile(factor[mask], 50)),
                "factor_p95": float(np.nanpercentile(factor[mask], 95)),
            }
        )

    # Build the monthly climatology from corrected annual totals.
    nongrow_templates: dict[int, np.ndarray] = {}
    nongrow_ratio_to_growing: dict[int, float] = {}
    aiwum2_growing_total_all = 0.0
    aiwum2_nongrow_total_by_month = {month: 0.0 for month in NONGROW_MONTHS}

    for year in AIWUM2_CLIM_YEARS:
        for month in GROWING_MONTHS:
            aiwum2_growing_total_all += masked_sum(read_target_grid(aiwum2_local_path(args.pumping_dir, year, month), profile, mask))
        for month in NONGROW_MONTHS:
            arr = read_target_grid(aiwum2_local_path(args.pumping_dir, year, month), profile, mask)
            aiwum2_nongrow_total_by_month[month] += masked_sum(arr)

    for month in NONGROW_MONTHS:
        clim_stack = [
            read_target_grid(aiwum2_local_path(args.pumping_dir, year, month), profile, mask)
            for year in AIWUM2_CLIM_YEARS
        ]
        template = np.nanmean(np.stack(clim_stack, axis=0), axis=0)
        template[~mask] = np.nan
        template_sum = masked_sum(template)
        if template_sum <= 0:
            template = np.where(mask, 0.0, np.nan).astype(np.float32)
        else:
            template = (template / template_sum).astype(np.float32)
        nongrow_templates[month] = template
        nongrow_ratio_to_growing[month] = aiwum2_nongrow_total_by_month[month] / aiwum2_growing_total_all

    backup_root = backup_raw_files(args.pumping_dir, TARGET_YEARS, args.overwrite)

    year_rows: list[dict] = []
    for year in TARGET_YEARS:
        raw_growing_total = 0.0
        corrected_growing_total = 0.0
        corrected_arrays: dict[int, np.ndarray] = {}

        for month in GROWING_MONTHS:
            raw = read_target_grid(aiwum1_source_path(args.aiwum1_repo, args.source_folder, year, month), profile, mask)
            raw_total = masked_sum(raw)
            factor = month_factors[month]
            corrected = raw * factor
            corrected[~np.isfinite(corrected)] = 0.0

            target_total = raw_total * month_rows[month - 4]["monthly_total_scale_aiwum2_over_aiwum1"]
            corrected_total = masked_sum(corrected)
            if corrected_total > 0 and target_total > 0:
                corrected *= target_total / corrected_total

            corrected_arrays[month] = corrected.astype(np.float32)
            raw_growing_total += raw_total
            corrected_growing_total += masked_sum(corrected)

        corrected_nongrowing_total = 0.0
        for month in range(1, 13):
            out_path = aiwum1_local_path(args.pumping_dir, year, month)
            if month in GROWING_MONTHS:
                arr = corrected_arrays[month]
            elif month in NONGROW_MONTHS:
                target_total = corrected_growing_total * nongrow_ratio_to_growing[month]
                arr = nongrow_templates[month] * target_total
                corrected_nongrowing_total += masked_sum(arr)
            else:
                arr = np.where(mask, 0.0, np.nan).astype(np.float32)
            write_raster(out_path, arr, profile, mask, args.overwrite)

        year_rows.append(
            {
                "year": year,
                "raw_growing_total_m3": raw_growing_total,
                "corrected_growing_total_m3": corrected_growing_total,
                "corrected_nongrowing_total_m3": corrected_nongrowing_total,
                "corrected_annual_total_m3": corrected_growing_total + corrected_nongrowing_total,
                "nongrowing_share": corrected_nongrowing_total / (corrected_growing_total + corrected_nongrowing_total),
            }
        )

    summary = {
        "created_at_epoch": int(time.time()),
        "method": "AIWUM1 2011-2013 bridge-corrected to AIWUM2 style using AIWUM1-vs-AIWUM2 overlap.",
        "target_years": TARGET_YEARS,
        "overlap_years": OVERLAP_YEARS,
        "aiwum2_climatology_years_for_nongrowing": AIWUM2_CLIM_YEARS,
        "growing_months": GROWING_MONTHS,
        "nongrowing_months_from_aiwum2_climatology": NONGROW_MONTHS,
        "december_policy": "zero, matching AIWUM2 near-zero/nodata December pattern",
        "sigma_pixels": args.sigma_pixels,
        "factor_clip": [args.factor_min, args.factor_max],
        "raw_backup_root": str(backup_root),
        "monthly_bridge": month_rows,
        "nongrowing_ratio_to_corrected_growing": {
            MONTH_NUM_TO_NAME[m]: nongrow_ratio_to_growing[m] for m in NONGROW_MONTHS
        },
        "year_summary": year_rows,
    }

    summary_path = diagnostics_dir / "aiwum1_2011_2013_bridge_to_aiwum2_summary.json"
    csv_path = diagnostics_dir / "aiwum1_2011_2013_bridge_to_aiwum2_year_summary.csv"
    monthly_csv_path = diagnostics_dir / "aiwum1_2011_2013_bridge_to_aiwum2_monthly_bridge.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(year_rows[0].keys()) + "\n")
        for row in year_rows:
            handle.write(",".join(str(row[key]) for key in row.keys()) + "\n")

    with monthly_csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(month_rows[0].keys()) + "\n")
        for row in month_rows:
            handle.write(",".join(str(row[key]) for key in row.keys()) + "\n")

    print(f"Saved: {summary_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {monthly_csv_path}")
    for row in year_rows:
        print(
            f"{row['year']}: raw growing={row['raw_growing_total_m3']/1e9:.3f} km3, "
            f"corrected annual={row['corrected_annual_total_m3']/1e9:.3f} km3, "
            f"nongrowing share={row['nongrowing_share']:.3%}"
        )


if __name__ == "__main__":
    main()
