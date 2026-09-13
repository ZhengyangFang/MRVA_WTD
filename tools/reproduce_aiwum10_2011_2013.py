from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import requests
import rasterio
from rasterio.warp import Resampling, reproject


ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = ROOT / "data_raw" / "4__AIWUM1.0_reproduce"
SOURCE_ROOT = REPRO_ROOT / "source"
REPO_DIR = SOURCE_ROOT / "aiwum1"
RUNNER_PATH = REPRO_ROOT / "run_aiwum10_2011_2013.py"
MANIFEST_PATH = REPRO_ROOT / "aiwum10_reproduction_manifest.json"
LOCAL_PUMPING_DIR = ROOT / "data" / "4 pumping out"
REFERENCE_TIF = LOCAL_PUMPING_DIR / "2014" / "AIWUM2_1km_m3_2014_Jan.tif"
IDOMAIN_PATH = LOCAL_PUMPING_DIR / "idomain_1km_target.dat"
DOWNLOAD_CACHE = REPRO_ROOT / "download_cache"

GIT_URL = "https://code.usgs.gov/map/wu/aiwum1.git"
ARCHIVE_URL = "https://code.usgs.gov/map/wu/aiwum1/-/archive/master/aiwum1-master.zip"
GITLAB_API = "https://code.usgs.gov/api/v4"
PROJECT_PATH = "map/wu/aiwum1"
MINIMAL_TARGET_YEARS = {"2011", "2012", "2013"}
AIWUM_PRISM_YEARS = range(1999, 2019)
PRISM_SERVICE_URL = "https://services.nacse.org/prism/data/get/us/4km/ppt/{year}{month:02d}"
CDL_SERVICE_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"

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
GROWING_MONTHS = list(range(4, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set up and ingest an AIWUM 1.0-style pumping reproduction for 2011-2013. "
            "The original AIWUM 1.0 model requires CDL and PRISM inputs; this script checks "
            "those requirements and can ingest the generated rasters onto the local MRVA 1 km grid."
        )
    )
    parser.add_argument(
        "action",
        choices=[
            "setup",
            "check",
            "write-runner",
            "download-prism",
            "download-cdl",
            "download-inputs",
            "ingest",
            "all",
        ],
        help="Action to run. 'all' runs setup, check, and write-runner, but not the original AIWUM model.",
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2011, 2012, 2013])
    parser.add_argument("--repo-dir", type=Path, default=REPO_DIR)
    parser.add_argument("--repro-root", type=Path, default=REPRO_ROOT)
    parser.add_argument("--pumping-dir", type=Path, default=LOCAL_PUMPING_DIR)
    parser.add_argument("--reference-tif", type=Path, default=REFERENCE_TIF)
    parser.add_argument("--idomain-path", type=Path, default=IDOMAIN_PATH)
    parser.add_argument("--output-folder", default="MRVA_2011_2013_km")
    parser.add_argument("--prism-res", type=int, choices=[4000, 800], default=4000)
    parser.add_argument("--irr-source", choices=["mirad", "NCA"], default="mirad")
    parser.add_argument("--fill-nongrowing", choices=["zero", "skip"], default="zero")
    parser.add_argument(
        "--download-mode",
        choices=["api-minimal", "full"],
        default="api-minimal",
        help="Use api-minimal to avoid slow full-repository clones; use full to clone/archive the whole repo.",
    )
    parser.add_argument("--overwrite-source", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--overwrite-inputs", action="store_true")
    parser.add_argument("--run-original", action="store_true", help="Run the generated AIWUM 1.0 runner.")
    return parser.parse_args()


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_command(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def gitlab_project_id() -> int:
    url = f"{GITLAB_API}/projects/{requests.utils.quote(PROJECT_PATH, safe='')}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return int(response.json()["id"])


def gitlab_tree(project_id: int) -> list[dict]:
    records: list[dict] = []
    page = 1
    while True:
        response = requests.get(
            f"{GITLAB_API}/projects/{project_id}/repository/tree",
            params={"ref": "master", "recursive": "true", "per_page": 100, "page": page},
            timeout=90,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        records.extend(batch)
        if response.headers.get("X-Next-Page"):
            page += 1
        else:
            break
    return records


def include_minimal_path(path: str) -> bool:
    always = {
        "README.md",
        "LICENSE.md",
        "environment.yml",
        "setup.py",
        "aiwum/AIWUM.py",
        "examples/meras_example.py",
        "data/inputs/Crop_lookup.csv",
        "data/inputs/Delta_extent.tif",
    }
    if path in always:
        return True
    prefixes = (
        "docs/",
        "data/inputs/flowmeters/",
        "data/inputs/County_boundaries/",
        "data/inputs/MERAS_inputs/",
    )
    if path.startswith(prefixes):
        return True
    if path.startswith("data/inputs/irrigated_acres_estimates/YMD_boundaries/permitted_rasters/"):
        return any(path.endswith(f"irr_pct_{year}.tif") for year in MINIMAL_TARGET_YEARS)
    if path.startswith("data/inputs/irrigated_acres_estimates/YMD_boundaries/shapefiles/"):
        return any(f"YMD_{year}_poly_5070" in path for year in MINIMAL_TARGET_YEARS)
    if path.startswith("data/inputs/irrigated_acres_estimates/mirad/"):
        return "2012" in path
    if path.startswith("data/inputs/irrigated_acres_estimates/census_of_ag/"):
        return "2012" in path
    if path.startswith("data/inputs/NHD/") and "NHD_MAP_merged_nofish" in path:
        return True
    return False


def download_gitlab_file(project_id: int, path: str, dest: Path) -> None:
    url = f"{GITLAB_API}/projects/{project_id}/repository/files/{requests.utils.quote(path, safe='')}/raw"
    response = requests.get(url, params={"ref": "master"}, timeout=120)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)


def stream_download(url: str, dest: Path, retries: int = 5, params: dict | None = None) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, params=params, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            tmp.replace(dest)
            return
        except Exception as exc:
            print(f"Download failed {attempt}/{retries}: {url} -> {exc}")
            if tmp.exists():
                tmp.unlink()
            time.sleep(min(45, 5 * attempt))
    raise RuntimeError(f"Could not download after {retries} attempts: {url}")


def copy_raster_as(src_path: Path, dst_path: Path, driver: str) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(driver=driver)
        data = src.read()
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)


def download_api_minimal(repo_dir: Path, overwrite: bool) -> None:
    marker = repo_dir / "aiwum" / "AIWUM.py"
    if marker.exists() and not overwrite:
        print(f"Using existing API-minimal AIWUM1 source: {repo_dir}")
        return
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)

    project_id = gitlab_project_id()
    tree = gitlab_tree(project_id)
    blob_paths = [item["path"] for item in tree if item.get("type") == "blob"]
    selected = sorted(path for path in blob_paths if include_minimal_path(path))
    print(f"Downloading {len(selected)} AIWUM1 minimal files via GitLab API")

    failures: list[dict] = []
    for idx, path in enumerate(selected, start=1):
        dest = repo_dir / Path(path)
        if dest.exists() and not overwrite:
            continue
        try:
            download_gitlab_file(project_id, path, dest)
        except Exception as exc:
            failures.append({"path": path, "error": str(exc)})
        if idx % 25 == 0 or idx == len(selected):
            print(f"  {idx}/{len(selected)} files processed")

    manifest = {
        "mode": "api-minimal",
        "project": PROJECT_PATH,
        "project_id": project_id,
        "selected_count": len(selected),
        "failures": failures,
        "omitted_large_external_inputs": [
            "USDA CDL rasters for target years and 2014",
            "PRISM monthly precipitation rasters for 1999-2017",
        ],
    }
    (repo_dir / "api_minimal_download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failures:
        print(f"Warning: {len(failures)} files failed to download. See api_minimal_download_manifest.json")
    print(f"API-minimal source prepared at: {repo_dir}")


def expected_prism_bil(repo_dir: Path, year: int, month: int) -> Path:
    return (
        repo_dir
        / "data"
        / "inputs"
        / "PRISM"
        / "4km"
        / "ppt"
        / f"PRISM_ppt_stable_4kmM3_{year}{month:02d}_bil.bil"
    )


def download_prism_4km(args: argparse.Namespace) -> None:
    cache_dir = args.repro_root / "download_cache" / "prism_4km_monthly_ppt"
    created = []
    for year in AIWUM_PRISM_YEARS:
        for month in range(1, 13):
            out_bil = expected_prism_bil(args.repo_dir, year, month)
            if out_bil.exists() and not args.overwrite_inputs:
                continue
            url = PRISM_SERVICE_URL.format(year=year, month=month)
            zip_path = cache_dir / f"prism_ppt_us_25m_{year}{month:02d}.zip"
            print(f"PRISM {year}-{month:02d}")
            stream_download(url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                tif_names = [name for name in zf.namelist() if name.lower().endswith(".tif")]
                if not tif_names:
                    raise RuntimeError(f"No tif found inside {zip_path}")
                tmp_dir = cache_dir / "extracted"
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                zf.extract(tif_names[0], tmp_dir)
                tmp_tif = tmp_dir / tif_names[0]
                copy_raster_as(tmp_tif, out_bil, driver="EHdr")
                shutil.rmtree(tmp_dir)
            created.append(str(out_bil))
            if len(created) % 24 == 0:
                print(f"  PRISM rasters prepared: {len(created)}")
    print(f"Prepared {len(created)} PRISM rasters")


def cdl_expected_img(repo_dir: Path, year: int) -> Path:
    return repo_dir / "data" / "inputs" / "CDL" / f"{year}_30m_cdls" / f"{year}_30m_cdls.img"


def cdl_bbox_from_reference(reference_tif: Path) -> str:
    with rasterio.open(reference_tif) as ds:
        bounds = ds.bounds
    return f"{bounds.left},{bounds.bottom},{bounds.right},{bounds.top}"


def parse_cdl_return_url(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text)
        for elem in root.iter():
            if elem.tag.endswith("returnURL") and elem.text:
                return elem.text.strip()
    except ET.ParseError:
        pass
    match = re.search(r"<returnURL>(.*?)</returnURL>", xml_text)
    if match:
        return match.group(1).strip()
    raise RuntimeError(f"Could not parse CDL returnURL from response: {xml_text[:500]}")


def download_cdl(args: argparse.Namespace) -> None:
    cache_dir = args.repro_root / "download_cache" / "cdl_bbox"
    bbox = cdl_bbox_from_reference(args.reference_tif)
    years = sorted(set(args.years + [2014]))
    for year in years:
        out_img = cdl_expected_img(args.repo_dir, year)
        if out_img.exists() and not args.overwrite_inputs:
            continue
        print(f"CDL {year} bbox={bbox}")
        response = requests.get(CDL_SERVICE_URL, params={"year": str(year), "bbox": bbox}, timeout=(30, 180))
        response.raise_for_status()
        tif_url = parse_cdl_return_url(response.text)
        tif_path = cache_dir / f"CDL_{year}_MRVA_bbox.tif"
        stream_download(tif_url, tif_path)
        copy_raster_as(tif_path, out_img, driver="HFA")
        print(f"  saved {out_img}")


def download_archive(repo_dir: Path, overwrite: bool) -> None:
    if repo_dir.exists() and not overwrite:
        print(f"Using existing AIWUM1 source: {repo_dir}")
        return
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    git_exe = shutil.which("git")
    if git_exe:
        try:
            run_command([git_exe, "clone", "--depth", "1", GIT_URL, str(repo_dir)])
            return
        except Exception as exc:
            print(f"git clone failed, falling back to archive download: {exc}")

    zip_path = repo_dir.parent / "aiwum1-master.zip"
    tmp_path = zip_path.with_suffix(".zip.part")
    print(f"Downloading AIWUM1 archive: {ARCHIVE_URL}")
    with requests.get(ARCHIVE_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(zip_path)

    extract_dir = repo_dir.parent / "archive_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    candidates = [p for p in extract_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError(f"No source directory found inside {zip_path}")
    shutil.move(str(candidates[0]), str(repo_dir))
    shutil.rmtree(extract_dir)
    print(f"Extracted AIWUM1 source to: {repo_dir}")


def setup_source(args: argparse.Namespace) -> None:
    if args.download_mode == "api-minimal":
        download_api_minimal(args.repo_dir, args.overwrite_source)
    else:
        download_archive(args.repo_dir, args.overwrite_source)


def expected_prism_paths(repo_dir: Path, prism_res: int) -> list[Path]:
    paths = []
    for year in AIWUM_PRISM_YEARS:
        for month in range(1, 13):
            if prism_res == 4000:
                name = f"PRISM_ppt_stable_4kmM3_{year}{month:02d}_bil.bil"
                paths.append(repo_dir / "data" / "inputs" / "PRISM" / "4km" / "ppt" / name)
            else:
                name = f"prism_ppt_us_30s_{year}{month:02d}.bil"
                paths.append(repo_dir / "data" / "inputs" / "PRISM" / "800m" / "ppt" / name)
    return paths


def expected_cdl_paths(repo_dir: Path, years: list[int]) -> list[Path]:
    needed_years = sorted(set(years + [2014]))
    return [
        repo_dir / "data" / "inputs" / "CDL" / f"{year}_30m_cdls" / f"{year}_30m_cdls.img"
        for year in needed_years
    ]


def check_requirements(args: argparse.Namespace) -> dict:
    repo_dir = args.repo_dir
    required_modules = ["geopandas", "georasters", "osgeo", "statsmodels", "xarray", "openpyxl", "xlrd"]
    module_status = {name: has_module(name) for name in required_modules}

    source_files = {
        "AIWUM.py": repo_dir / "aiwum" / "AIWUM.py",
        "README.md": repo_dir / "README.md",
        "environment.yml": repo_dir / "environment.yml",
        "rate_table": repo_dir / "data" / "inputs" / "flowmeters" / "rate_tables" / "Irrigation_rates_in_IWUM.xlsx",
        "mdeq_2014": repo_dir / "data" / "inputs" / "flowmeters" / "MDEQ" / "2014 Water Use Data.xlsx",
        "mdeq_2015": repo_dir / "data" / "inputs" / "flowmeters" / "MDEQ" / "2015 Water Use Data.xlsx",
        "mdeq_2016": repo_dir / "data" / "inputs" / "flowmeters" / "MDEQ" / "2016 Water Use Data.xlsx",
        "mdeq_2017": repo_dir / "data" / "inputs" / "flowmeters" / "MDEQ" / "2017_Water Use Analysis_Final.xlsx",
        "nhd_map_shp": repo_dir / "data" / "inputs" / "NHD" / "NHD_MAP_merged_nofish.shp",
        "nhd_map_dbf": repo_dir / "data" / "inputs" / "NHD" / "NHD_MAP_merged_nofish.dbf",
        "nhd_map_shx": repo_dir / "data" / "inputs" / "NHD" / "NHD_MAP_merged_nofish.shx",
        "nhd_map_prj": repo_dir / "data" / "inputs" / "NHD" / "NHD_MAP_merged_nofish.prj",
        "mirad_2012": repo_dir / "data" / "inputs" / "irrigated_acres_estimates" / "mirad" / "mirad_250_2012.tif",
        "target_reference": args.reference_tif,
        "target_idomain": args.idomain_path,
    }
    source_status = {label: path.exists() for label, path in source_files.items()}

    cdl_paths = expected_cdl_paths(repo_dir, args.years)
    prism_paths = expected_prism_paths(repo_dir, args.prism_res)
    ymd_paths = []
    for year in args.years:
        ymd_paths.extend(
            [
                repo_dir
                / "data"
                / "inputs"
                / "irrigated_acres_estimates"
                / "YMD_boundaries"
                / "permitted_rasters"
                / f"irr_pct_{year}.tif",
                repo_dir
                / "data"
                / "inputs"
                / "irrigated_acres_estimates"
                / "YMD_boundaries"
                / "shapefiles"
                / f"YMD_{year}_poly_5070.shp",
                repo_dir
                / "data"
                / "inputs"
                / "irrigated_acres_estimates"
                / "YMD_boundaries"
                / "shapefiles"
                / f"YMD_{year}_poly_5070.dbf",
                repo_dir
                / "data"
                / "inputs"
                / "irrigated_acres_estimates"
                / "YMD_boundaries"
                / "shapefiles"
                / f"YMD_{year}_poly_5070.shx",
                repo_dir
                / "data"
                / "inputs"
                / "irrigated_acres_estimates"
                / "YMD_boundaries"
                / "shapefiles"
                / f"YMD_{year}_poly_5070.prj",
            ]
        )

    missing = {
        "modules": [name for name, ok in module_status.items() if not ok],
        "source_files": [str(path) for label, path in source_files.items() if not source_status[label]],
        "cdl": [str(path) for path in cdl_paths if not path.exists()],
        "prism": [str(path) for path in prism_paths if not path.exists()],
        "ymd": [str(path) for path in ymd_paths if not path.exists()],
    }
    ready = not any(missing.values())

    report = {
        "ready_to_run_original_aiwum": ready,
        "repo_dir": str(repo_dir),
        "years": args.years,
        "prism_res": args.prism_res,
        "module_status": module_status,
        "source_status": source_status,
        "missing_counts": {key: len(value) for key, value in missing.items()},
        "missing_examples": {key: value[:10] for key, value in missing.items()},
        "notes": [
            "AIWUM 1.0 only supports April-October pumping months in the original code.",
            "For 2011-2013, AIWUM 1.0 uses the YMD annual-rate branch and monthly allocation weights.",
            "Strict reproduction requires PRISM precipitation; the 4 km PRISM option is public, while 800 m is proprietary.",
        ],
    }
    args.repro_root.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:6000])
    print(f"Saved manifest: {MANIFEST_PATH}")
    return report


def write_runner(args: argparse.Namespace) -> None:
    args.repro_root.mkdir(parents=True, exist_ok=True)
    years_literal = repr(args.years)
    month_literal = repr(GROWING_MONTHS)
    runner = f"""
    from __future__ import annotations

    import os
    import sys
    from pathlib import Path

    repo_dir = Path(r"{args.repo_dir}")
    sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)

    for rel in [
        "data/inter/flowmeters/MDEQ",
        "data/inter/PRISM/4km/ppt/MAP",
        "data/inter/PRISM/4km/ppt",
        "data/inter/PRISM/800m/ppt/MAP",
        "data/inter/PRISM/800m/ppt",
        "data/inter/irrigated_acres_estimates/mirad",
        "data/inter/irrigated_acres_estimates/census_of_ag",
        "data/inter/irrigated_acres_estimates/YMD_boundaries/permitted_rasters",
        "output/Model_files",
    ]:
        (repo_dir / rel).mkdir(parents=True, exist_ok=True)

    import aiwum.AIWUM as aiwum

    years = {years_literal}
    month_range = {month_literal}
    model_domain = Path(r"{args.reference_tif}")

    aiwum.aiwum(
        years,
        month_range,
        interval="km",
        output_units="m3",
        netcdf_out=True,
        model_domain=str(model_domain),
        output_folder="{args.output_folder}",
        delta_refine=True,
        resolution="low",
        noa_binary=False,
        keep_inter=False,
        nhd_extent="MAP",
        prism_res={args.prism_res},
        clip_to_domain=True,
        irr_source="{args.irr_source}",
    )
    """
    RUNNER_PATH.write_text(textwrap.dedent(runner).strip() + "\n", encoding="utf-8")
    print(f"Wrote runner: {RUNNER_PATH}")
    print(f"Run after check passes: python -u {RUNNER_PATH}")


def load_target_profile(reference_tif: Path, idomain_path: Path) -> tuple[dict, np.ndarray]:
    with rasterio.open(reference_tif) as ref:
        profile = ref.profile.copy()
    profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate", interleave="pixel")

    idomain = np.loadtxt(idomain_path)
    mask = np.flipud(idomain) != 0
    if mask.shape != (profile["height"], profile["width"]):
        raise ValueError(f"Mask shape {mask.shape} does not match reference raster shape.")
    return profile, mask


def find_aiwum_month_raster(repo_dir: Path, output_folder: str, year: int, month: int) -> Path | None:
    candidates = [
        repo_dir / "Output" / output_folder / "rasters" / "km" / "Annual_totals" / f"y{year}_m{month}_tot.tif",
        repo_dir / "Output" / "rasters" / "km" / "Annual_totals" / f"y{year}_m{month}_tot.tif",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted((repo_dir / "Output").rglob(f"y{year}_m{month}_tot.tif")) if (repo_dir / "Output").exists() else []
    return matches[0] if matches else None


def resample_to_target(source_path: Path, profile: dict, target_mask: np.ndarray) -> np.ndarray:
    nodata = float(profile["nodata"])
    dst = np.full((profile["height"], profile["width"]), nodata, dtype=np.float32)
    with rasterio.open(source_path) as src:
        source = src.read(1).astype(np.float32)
        src_nodata = src.nodata
        reproject(
            source=source,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            dst_nodata=nodata,
            resampling=Resampling.average,
        )
    dst[~target_mask] = nodata
    return dst


def write_raster(path: Path, arr: np.ndarray, profile: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"Exists, skipping: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def ingest_outputs(args: argparse.Namespace) -> None:
    profile, target_mask = load_target_profile(args.reference_tif, args.idomain_path)
    nodata = float(profile["nodata"])
    records = []
    for year in args.years:
        for month in range(1, 13):
            out_path = (
                args.pumping_dir
                / str(year)
                / f"AIWUM1_1km_m3_{year}_{MONTH_NUM_TO_NAME[month]}.tif"
            )
            source_path = find_aiwum_month_raster(args.repo_dir, args.output_folder, year, month)
            if source_path is None:
                if month not in GROWING_MONTHS and args.fill_nongrowing == "zero":
                    arr = np.zeros((profile["height"], profile["width"]), dtype=np.float32)
                    arr[~target_mask] = nodata
                    write_raster(out_path, arr, profile, args.overwrite_output)
                    records.append(
                        {
                            "year": year,
                            "month": month,
                            "source": "filled_zero_nongrowing_month",
                            "output": str(out_path),
                            "sum_m3": 0.0,
                        }
                    )
                else:
                    records.append(
                        {
                            "year": year,
                            "month": month,
                            "source": None,
                            "output": None,
                            "status": "missing_source",
                        }
                    )
                continue

            arr = resample_to_target(source_path, profile, target_mask)
            write_raster(out_path, arr, profile, args.overwrite_output)
            valid = arr != nodata
            records.append(
                {
                    "year": year,
                    "month": month,
                    "source": str(source_path),
                    "output": str(out_path),
                    "sum_m3": float(np.nansum(arr[valid])),
                    "valid_cells": int(valid.sum()),
                }
            )

    summary = {
        "created_at_epoch": int(time.time()),
        "method": "AIWUM1.0 original-code output ingested to local MRVA 1 km grid",
        "repo_dir": str(args.repo_dir),
        "output_folder": args.output_folder,
        "fill_nongrowing": args.fill_nongrowing,
        "records": records,
    }
    summary_path = args.pumping_dir / "aiwum10_reproduced_2011_2013_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")


def main() -> None:
    args = parse_args()
    args.repro_root.mkdir(parents=True, exist_ok=True)

    if args.action in {"setup", "all"}:
        setup_source(args)

    if args.action in {"check", "all"}:
        check_requirements(args)

    if args.action in {"write-runner", "all"}:
        write_runner(args)

    if args.action in {"download-prism", "download-inputs"}:
        download_prism_4km(args)

    if args.action in {"download-cdl", "download-inputs"}:
        download_cdl(args)

    if args.run_original:
        run_command([sys.executable, "-u", str(RUNNER_PATH)], cwd=args.repro_root)

    if args.action == "ingest":
        ingest_outputs(args)


if __name__ == "__main__":
    main()
