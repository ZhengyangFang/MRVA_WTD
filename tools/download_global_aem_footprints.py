from __future__ import annotations

import csv
import argparse
import html
import json
import math
import os
import re
import struct
import time
import zipfile
import zlib
from pathlib import Path
from urllib.parse import urljoin, urlparse

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rasterio.features
from affine import Affine
from shapely.geometry import LineString, box, mapping, shape
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "13 Global AEM"
MANIFEST_JSON = DATA_DIR / "aem_public_datasets_manifest.json"
DOWNLOAD_DIR = DATA_DIR / "downloads"
DERIVED_DIR = DATA_DIR / "derived"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

MAX_DOWNLOAD_BYTES = 150 * 1024 * 1024
MAX_GRID_DOWNLOAD_BYTES = 120 * 1024 * 1024
REQUEST_TIMEOUT = 45
USER_AGENT = "Mozilla/5.0 (compatible; Codex Global AEM footprint collector)"
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")

VECTOR_EXTS = {
    ".zip",
    ".shp",
    ".shx",
    ".dbf",
    ".prj",
    ".cpg",
    ".geojson",
    ".json",
    ".kml",
    ".kmz",
    ".gml",
    ".gpkg",
    ".dxf",
}
GRID_EXTS = {
    ".gxf",
    ".nc",
    ".ncml",
    ".tif",
    ".tiff",
    ".asc",
    ".grd",
}
KEEP_KEYWORDS = (
    "flight",
    "flightline",
    "flight_line",
    "flight-line",
    "flown",
    "line_path",
    "line_paths",
    "surveyarea",
    "survey_area",
    "survey area",
    "footprint",
    "outline",
    "boundary",
    "extent",
    "inventory",
    "polygon",
    "shp",
    "shape",
    "kml",
    "geojson",
    "gml",
    "gpkg",
)
BAD_VECTOR_KEYWORDS = (
    "bedrock_fault",
    "faults",
    "fault",
    "lineament",
    "lineaments",
    "contour",
    "contours",
    "drafts",
)
GRID_KEYWORDS = (
    "res400",
    "res1800",
    "resistivity",
    "conductivity",
    "apparent",
    "depthgrid",
    "depthgrids",
)
SKIP_KEYWORDS = (
    "voxel",
    "inversion",
    "aemdata",
    "raw_data",
    "processed",
    "database",
    "slpk",
    "report",
    "appendix",
    "appendices",
    "pdf",
)

# Approximate survey bounds.
APPROX_BBOX = {
    "AEM03": (-125.0, 32.0, -114.0, 42.5),
    "AEM04": (-122.4, 39.3, -121.3, 40.2),
    "AEM06": (-94.2, 30.5, -88.0, 37.9),
    "AEM08": (-91.1, 33.3, -90.2, 34.4),
    "AEM09": (-94.5, 29.0, -88.0, 38.2),
    "AEM11": (-90.6, 38.5, -86.8, 42.0),
    "AEM19": (-107.3, 38.4, -106.6, 39.3),
    "AEM28": (-124.5, 35.0, -113.0, 44.5),
    "AEM33": (-170.0, 51.0, -130.0, 72.0),
    "AEM34": (-146.0, 62.0, -140.0, 64.5),
    "AEM35": (-104.2, 39.7, -95.0, 43.1),
    "AEM36": (-140.5, 60.4, -137.0, 61.9),
    "AEM37": (-141.0, 42.0, -52.0, 84.0),
    "AEM38": (-83.5, 45.0, -80.0, 48.5),
    "AEM39": (-80.0, 44.5, -57.0, 62.5),
    "AEM40": (-115.0, 51.0, -112.0, 54.0),
    "AEM41": (-125.5, 52.0, -120.0, 55.5),
    "AEM42": (-128.5, 52.0, -123.0, 55.5),
    "AEM43": (-123.5, 55.0, -119.0, 58.5),
    "AEM44": (112.0, -44.0, 154.0, -10.0),
    "AEM45": (130.0, -25.5, 145.5, -12.0),
    "AEM46": (137.0, -29.0, 146.0, -16.0),
    "AEM47": (115.0, -36.0, 119.0, -32.0),
    "AEM48": (117.0, -33.0, 135.0, -24.0),
    "AEM49": (119.0, -28.5, 125.5, -24.0),
    "AEM50": (139.0, -22.5, 142.5, -18.0),
    "AEM51": (141.0, -33.0, 143.0, -30.5),
    "AEM52": (146.0, -33.0, 149.0, -29.5),
    "AEM53": (112.0, -36.0, 129.0, -14.0),
    "AEM54": (129.0, -38.5, 141.0, -25.0),
    "AEM55": (8.0, 54.5, 15.5, 58.2),
    "AEM56": (-10.8, 51.0, -5.2, 55.5),
    "AEM57": (-8.4, 54.0, -5.3, 55.4),
    "AEM58": (10.5, 55.0, 24.5, 69.5),
    "AEM59": (4.0, 58.0, 31.0, 71.5),
    "AEM60": (19.0, 59.0, 32.0, 70.5),
    "AEM61": (5.5, 47.0, 15.5, 55.5),
    "AEM62": (3.1, 50.9, 4.4, 51.8),
    "AEM63": (2.5, 50.6, 5.9, 51.6),
    "AEM64": (-32.0, 71.0, -22.0, 73.5),
    "AEM65": (-52.5, 63.5, -48.0, 65.5),
    "AEM66": (176.0, -40.2, 177.5, -39.0),
    "AEM67": (176.0, -40.5, 177.2, -39.3),
    "AEM68": (175.0, -41.5, 176.5, -40.4),
    "AEM69": (166.0, -47.5, 179.5, -34.0),
    "AEM70": (108.5, -2.3, 114.8, 1.7),
}

# Survey files with generic download names.
MANUAL_DOWNLOADS = {
    "AEM36": [
        "https://open.yukon.ca/information/6c4f1356-6c72-42c4-9d9f-145b6d9458d4/resource/ee69c298-93e3-44fa-bb10-fd840c8553be/download/kluane-lake-west-electromagnetic-survey-parts-of-nts-115g-5-6-11-and-12-rxd271cc.zip",
    ],
    "AEM34": [
        "https://dggs.alaska.gov/webpubs/data/gpr2020_015_akhwy_2005_images_registered.zip",
    ],
    "AEM40": [
        "https://static.ags.aer.ca/files/document/DIG/DIG_2003_0013.zip",
    ],
    "AEM46": [
        "https://d28rz98at9flks.cloudfront.net/145120/145120_00_1.zip",
    ],
    "AEM25": [
        "https://www.sciencebase.gov/catalog/file/get/6197f026d34eb622f692eed1?f=__disk__fd%2F2e%2Ff3%2Ffd2ef36ab96be91c34e4b324124e2951809c5e8b",
    ],
    "AEM52": [
        "https://d28rz98at9flks.cloudfront.net/74137/aem_data_survey_plan.zip",
    ],
    "AEM61": [
        "https://download.bgr.de/bgr/aerogeophysik/D-AERO-INSPIRE/gml/D-AERO-INSPIRE.zip",
    ],
}

GEOJSON_DOWNLOADS = {
    "AEM57": [
        (
            "tellus_survey_flight_lines.geojson",
            "https://map.bgs.ac.uk/arcgis/rest/services/GeoIndex_GSNI/GSNI_Geophysics/MapServer/1/query?where=1%3D1&outFields=%2A&returnGeometry=true&f=geojson&outSR=4326",
        ),
    ],
    "AEM58": [
        (
            "sgu_slingram_flightarea.geojson",
            "https://api.sgu.se/oppnadata/geofysik-flyg-em-slingram-detaljerad/ogc/features/v1/collections/aero-flightarea/items?f=json&limit=10000",
        ),
        (
            "sgu_vlf_flightarea.geojson",
            "https://api.sgu.se/oppnadata/geofysik-flyg-em-vlf/ogc/features/v1/collections/aero-flightarea/items?f=json&limit=10000",
        ),
        (
            "sgu_tem_flightarea.geojson",
            "https://api.sgu.se/oppnadata/geofysik-flyg-em-tem-resistivitet/ogc/features/v1/collections/aero-tem-flygomrade/items?f=json&limit=10000",
        ),
    ],
    "AEM62": [
        (
            "freshem_vlieglijnen.geojson",
            "https://projectgeodata.zeeland.nl/geoserver/freshem/wfs?service=WFS&version=2.0.0&request=GetFeature&typeNames=freshem%3Avlieglijnen&outputFormat=application%2Fjson",
        ),
    ],
}

REMOTE_ZIP_MEMBER_DOWNLOADS = {
    "AEM38": [
        {
            "url": "https://prd-0420-geoontario-0000-blob-cge0eud7azhvfsf7.z01.azurefd.net/lrc-geology-documents/publication/GDS1087/GDS1087.zip",
            "members": [
                "GDS1087/vector_files/BAPATH83.dxf",
            ],
        },
    ],
    # Extract the survey-line members from the archive.
    "AEM51": [
        {
            "url": "https://d28rz98at9flks.cloudfront.net/130349/130349_data.zip",
            "members": [
                "shape_files/BHMAR_HM_survey_lines.dbf",
                "shape_files/BHMAR_HM_survey_lines.prj",
                "shape_files/BHMAR_HM_survey_lines.shp",
                "shape_files/BHMAR_HM_survey_lines.shx",
                "shape_files/BHMAR_LM_survey_lines.dbf",
                "shape_files/BHMAR_LM_survey_lines.prj",
                "shape_files/BHMAR_LM_survey_lines.shp",
                "shape_files/BHMAR_LM_survey_lines.shx",
            ],
        },
    ],
}

EXTRA_SCIENCEBASE_IDS = {
    "AEM06": ["5f35b17382cee144fb35a733"],
    "AEM07": ["5d76b872e4b0c4f70d01ff75"],
    "AEM08": ["5c9e6c42e4b0b8a7f62f5da6"],
    "AEM09": ["62069b13d34ec05caca511f0"],
    "AEM19": ["5f4d802d82ce4c3d123191ff"],
}

CSV_DEFAULT_CRS = {
    # Coordinate systems for projected survey tables.
    "AEM17": 32615,  # Cedar Rapids, Iowa, UTM zone 15N
    "AEM18": 32612,  # Yellowstone processed files, E_UTM12N / N_UTM12N
    "AEM20": 32612,  # Upper San Pedro Basin, Arizona, UTM zone 12N
}

VECTOR_DEFAULT_CRS = {
    "AEM38": 26917,  # Ontario GDS1087 vector DXF files use NAD83 UTM-style metre coordinates.
}

XYZ_COLUMNS = [
    "Line",
    "Easting",
    "Northing",
    "Fiducial",
    "DTM",
    "Depth_lower",
    "Depth_upper",
    "Resistivity",
]


def safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return (text[:max_len] or "file").strip("_")


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def is_candidate_name(name: str) -> bool:
    lower = name.lower()
    ext = Path(urlparse(lower).path).suffix
    if ext not in VECTOR_EXTS:
        return False
    if any(k in lower for k in BAD_VECTOR_KEYWORDS):
        return False
    if any(k in lower for k in SKIP_KEYWORDS) and not any(k in lower for k in ("flight", "line", "footprint", "survey_area", "surveyarea")):
        return False
    return any(k in lower for k in KEEP_KEYWORDS)


def is_candidate_grid_name(name: str) -> bool:
    lower = name.lower()
    ext = Path(urlparse(lower).path).suffix
    if ext not in GRID_EXTS:
        return False
    if any(k in lower for k in ("raw", "processeddata", "contractor", "package", "report", "xml", "ncml")):
        return False
    return any(k in lower for k in GRID_KEYWORDS)


def is_candidate_csv_name(name: str) -> bool:
    lower = name.lower()
    ext = Path(urlparse(lower).path).suffix
    if ext != ".csv":
        return False
    if any(k in lower for k in ("dictionary", "metadata", "readme")):
        return False
    return any(
        k in lower
        for k in (
            "measureddata",
            "measured_data",
            "processeddata",
            "processed_data",
            "cdi",
            "flightline",
            "flight_line",
            "navigation",
            "nav",
        )
    )


def is_candidate_facet(name: str) -> bool:
    lower = name.lower()
    if any(k in lower for k in BAD_VECTOR_KEYWORDS):
        return False
    return any(k in lower for k in ("flight", "line", "survey_area", "surveyarea", "footprint", "inventory", "polygon", "boundary", "extent"))


def content_length(s: requests.Session, url: str) -> int | None:
    try:
        r = s.head(url, allow_redirects=True, timeout=REQUEST_TIMEOUT)
        if r.status_code < 400 and r.headers.get("content-length"):
            return int(r.headers["content-length"])
    except Exception:
        return None
    return None


def download_url(
    s: requests.Session,
    url: str,
    out_path: Path,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    expected_size: int | None = None,
) -> tuple[str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        size = out_path.stat().st_size
        if expected_size and expected_size > 0 and size < max(1024, expected_size * 0.5):
            out_path.unlink(missing_ok=True)
        else:
            return "exists", size
    length = content_length(s, url)
    if length is not None and length > max_bytes:
        return f"skipped_size_{length}", 0
    try:
        with s.get(url, stream=True, allow_redirects=True, timeout=REQUEST_TIMEOUT) as r:
            if r.status_code >= 400:
                return f"http_{r.status_code}", 0
            total = 0
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 512):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        f.close()
                        tmp.unlink(missing_ok=True)
                        return f"skipped_stream_size_gt_{max_bytes}", 0
                    f.write(chunk)
            tmp.replace(out_path)
            return "downloaded", total
    except Exception as exc:
        return f"error_{type(exc).__name__}", 0


def remote_file_size(s: requests.Session, url: str) -> int:
    r = s.head(url, allow_redirects=True, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    if not r.headers.get("content-length"):
        raise ValueError("Missing Content-Length for remote ZIP")
    return int(r.headers["content-length"])


def remote_range(s: requests.Session, url: str, start: int, end: int) -> bytes:
    r = s.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    if r.status_code != 206:
        raise ValueError(f"Remote server did not honor Range request: HTTP {r.status_code}")
    return r.content


def remote_zip_central_directory(s: requests.Session, url: str) -> tuple[int, dict[str, dict[str, int]]]:
    size = remote_file_size(s, url)
    tail_len = min(size, 4 * 1024 * 1024)
    tail_start = size - tail_len
    tail = remote_range(s, url, tail_start, size - 1)
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0:
        raise ValueError("ZIP end-of-central-directory record not found")
    fields = struct.unpack_from("<4s4H2LH", tail, pos)
    n_total = fields[4]
    cd_size = fields[5]
    cd_offset = fields[6]
    cd_start_in_tail = cd_offset - tail_start
    if cd_start_in_tail >= 0 and cd_start_in_tail + cd_size <= len(tail):
        cd = tail[cd_start_in_tail:cd_start_in_tail + cd_size]
    else:
        cd = remote_range(s, url, cd_offset, cd_offset + cd_size - 1)

    entries: dict[str, dict[str, int]] = {}
    i = 0
    while i + 46 <= len(cd) and cd[i:i + 4] == b"PK\x01\x02":
        vals = struct.unpack_from("<4s6H3L5H2L", cd, i)
        comp_method = vals[4]
        comp_size = vals[8]
        uncomp_size = vals[9]
        fn_len = vals[10]
        extra_len = vals[11]
        comment_len = vals[12]
        local_off = vals[16]
        name = cd[i + 46:i + 46 + fn_len].decode("utf-8", errors="replace")
        entries[name] = {
            "method": comp_method,
            "compressed_size": comp_size,
            "uncompressed_size": uncomp_size,
            "local_offset": local_off,
        }
        i += 46 + fn_len + extra_len + comment_len
    if n_total and len(entries) != n_total:
        raise ValueError(f"Parsed {len(entries)} ZIP entries, expected {n_total}")
    return size, entries


def download_remote_zip_members(s: requests.Session, rec: dict, downloads: list[dict]) -> None:
    entries = REMOTE_ZIP_MEMBER_DOWNLOADS.get(rec["source_id"], [])
    if not entries:
        return
    folder = DOWNLOAD_DIR / rec["source_id"] / "remote_zip_members"
    folder.mkdir(parents=True, exist_ok=True)
    for spec in entries:
        url = spec["url"]
        try:
            _, directory = remote_zip_central_directory(s, url)
        except Exception as exc:
            downloads.append({"source_id": rec["source_id"], "kind": "remote_zip_members", "url": url, "status": f"directory_error_{type(exc).__name__}"})
            continue
        for member in spec["members"]:
            info = directory.get(member)
            if info is None:
                downloads.append({"source_id": rec["source_id"], "kind": "remote_zip_member", "name": member, "url": url, "status": "missing_member"})
                continue
            out_path = folder / Path(member)
            if out_path.exists() and out_path.stat().st_size > 0:
                downloads.append({"source_id": rec["source_id"], "kind": "remote_zip_member", "name": member, "url": url, "status": "exists", "bytes": out_path.stat().st_size})
                continue
            try:
                local_header = remote_range(s, url, info["local_offset"], info["local_offset"] + 30 - 1)
                vals = struct.unpack_from("<4s5H3L2H", local_header, 0)
                if vals[0] != b"PK\x03\x04":
                    raise ValueError("Invalid local file header")
                method = vals[3]
                fn_len = vals[9]
                extra_len = vals[10]
                data_start = info["local_offset"] + 30 + fn_len + extra_len
                compressed = remote_range(s, url, data_start, data_start + info["compressed_size"] - 1)
                if method == 0:
                    payload = compressed
                elif method == 8:
                    payload = zlib.decompress(compressed, -15)
                else:
                    raise ValueError(f"Unsupported ZIP compression method {method}")
                if len(payload) != info["uncompressed_size"]:
                    raise ValueError("Unexpected uncompressed member size")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(payload)
                downloads.append({"source_id": rec["source_id"], "kind": "remote_zip_member", "name": member, "url": url, "status": "downloaded", "bytes": len(payload)})
            except Exception as exc:
                downloads.append({"source_id": rec["source_id"], "kind": "remote_zip_member", "name": member, "url": url, "status": f"extract_error_{type(exc).__name__}"})


def resolve_doi_to_sciencebase_id(s: requests.Session, doi: str) -> str | None:
    if not doi.startswith("10.5066/"):
        return None
    try:
        r = s.get(f"https://doi.org/{doi}", allow_redirects=True, timeout=REQUEST_TIMEOUT)
    except Exception:
        return None
    m = re.search(r"sciencebase\.gov/catalog/item/([A-Za-z0-9]+)", r.url)
    if m:
        return m.group(1)
    m = re.search(r"sciencebase\.gov/catalog/item/([A-Za-z0-9]+)", r.text)
    return m.group(1) if m else None


def sciencebase_ids_for_record(s: requests.Session, rec: dict) -> list[str]:
    ids: list[str] = []
    url = rec["url"]
    for pattern in (r"sciencebase\.gov/catalog/item/([A-Za-z0-9]+)", r"USGS%3A([A-Za-z0-9]+)", r"USGS:([A-Za-z0-9]+)"):
        for match in re.findall(pattern, url):
            ids.append(match)
    for doi in re.findall(r"10\.5066/[A-Za-z0-9]+", rec.get("reference", "") + " " + url):
        sbid = resolve_doi_to_sciencebase_id(s, doi)
        if sbid:
            ids.append(sbid)
    seen = set()
    return [x for x in ids if not (x in seen or seen.add(x))]


def bbox_feature(rec: dict, bbox: tuple[float, float, float, float], source: str, item_id: str | None = None) -> dict:
    minx, miny, maxx, maxy = bbox
    return {
        "type": "Feature",
        "properties": {
            "source_id": rec["source_id"],
            "seq": rec["seq"],
            "region_label": rec["region_label"],
            "dataset_label": rec["dataset_label"],
            "source_url": rec["url"],
            "footprint_source": source,
            "precision_rank": 3 if source.endswith("bbox") or source == "arcgis_extent" else 4,
            "sciencebase_id": item_id or "",
        },
        "geometry": mapping(box(minx, miny, maxx, maxy)),
    }


def sciencebase_file_url(file_rec: dict) -> str | None:
    # Select the downloadable file URL.
    return file_rec.get("publishedS3Uri") or file_rec.get("downloadUri") or file_rec.get("url")


def process_sciencebase(s: requests.Session, rec: dict, item_id: str, downloads: list[dict], bboxes: list[dict]) -> None:
    folder = DOWNLOAD_DIR / rec["source_id"] / f"sciencebase_{item_id}"
    folder.mkdir(parents=True, exist_ok=True)
    url = f"https://www.sciencebase.gov/catalog/item/{item_id}?format=json"
    try:
        data = s.get(url, timeout=REQUEST_TIMEOUT).json()
    except Exception as exc:
        downloads.append({"source_id": rec["source_id"], "kind": "sciencebase", "url": url, "status": f"metadata_error_{type(exc).__name__}"})
        return
    (folder / "sciencebase_item.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    bb = data.get("spatial", {}).get("boundingBox")
    if bb:
        bboxes.append(
            bbox_feature(
                rec,
                (float(bb["minX"]), float(bb["minY"]), float(bb["maxX"]), float(bb["maxY"])),
                "sciencebase_bbox",
                item_id,
            )
        )

    for f in data.get("files", []) or []:
        name = f.get("name") or "sciencebase_file"
        size = int(f.get("size") or 0)
        uri = sciencebase_file_url(f)
        is_vector = is_candidate_name(name)
        is_grid = is_candidate_grid_name(name)
        is_csv = is_candidate_csv_name(name)
        if not uri or not (is_vector or is_grid or is_csv):
            continue
        max_bytes = MAX_GRID_DOWNLOAD_BYTES if is_grid else MAX_DOWNLOAD_BYTES
        if is_vector:
            kind = "sciencebase_file"
        elif is_grid:
            kind = "sciencebase_grid_file"
        else:
            kind = "sciencebase_csv_file"
        status, bytes_written = download_url(s, uri, folder / safe_name(name), max_bytes=max_bytes, expected_size=size)
        downloads.append({"source_id": rec["source_id"], "kind": kind, "name": name, "url": uri, "size": size, "status": status, "bytes": bytes_written})

    for facet in data.get("facets", []) or []:
        facet_name = facet.get("name", "")
        files = facet.get("files", []) or []
        total_size = sum(int(f.get("size") or 0) for f in files)
        has_candidate_file = any(
            is_candidate_name(f.get("name") or "")
            or is_candidate_grid_name(f.get("name") or "")
            or is_candidate_csv_name(f.get("name") or "")
            for f in files
        )
        if not files or (not is_candidate_facet(facet_name) and not has_candidate_file) or total_size > MAX_DOWNLOAD_BYTES:
            continue
        facet_folder = folder / safe_name(facet_name)
        for f in files:
            name = f.get("name") or "facet_file"
            uri = sciencebase_file_url(f)
            is_vector = is_candidate_name(name)
            is_grid = is_candidate_grid_name(name)
            is_csv = is_candidate_csv_name(name)
            if not uri or not (is_vector or is_grid or is_csv):
                continue
            size = int(f.get("size") or 0)
            max_bytes = MAX_GRID_DOWNLOAD_BYTES if is_grid else MAX_DOWNLOAD_BYTES
            if is_vector:
                kind = "sciencebase_facet"
            elif is_grid:
                kind = "sciencebase_grid_facet"
            else:
                kind = "sciencebase_csv_facet"
            status, bytes_written = download_url(s, uri, facet_folder / safe_name(name), max_bytes=max_bytes, expected_size=size)
            downloads.append({"source_id": rec["source_id"], "kind": kind, "name": name, "facet": facet_name, "url": uri, "size": size, "status": status, "bytes": bytes_written})


def sciencebase_child_ids(s: requests.Session, rec: dict, item_id: str, downloads: list[dict]) -> list[str]:
    """Return ScienceBase child item identifiers."""
    folder = DOWNLOAD_DIR / rec["source_id"] / f"sciencebase_{item_id}"
    folder.mkdir(parents=True, exist_ok=True)
    url = f"https://www.sciencebase.gov/catalog/items?parentId={item_id}&format=json&max=1000"
    try:
        r = s.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        downloads.append({"source_id": rec["source_id"], "kind": "sciencebase_children", "url": url, "status": f"metadata_error_{type(exc).__name__}"})
        return []
    (folder / "sciencebase_children.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    items = data.get("items") or data.get("children") or []
    child_ids = [item.get("id") for item in items if item.get("id")]
    downloads.append({"source_id": rec["source_id"], "kind": "sciencebase_children", "url": url, "status": "metadata_saved", "children": len(child_ids)})
    return child_ids


def process_cnra_aem(s: requests.Session, rec: dict, downloads: list[dict]) -> None:
    api = "https://data.cnra.ca.gov/api/3/action/package_show?id=aem"
    api_folder = DOWNLOAD_DIR / rec["source_id"] / "cnra_api"
    try:
        data = s.get(api, timeout=REQUEST_TIMEOUT).json()
        for resource in data.get("result", {}).get("resources", []) or []:
            name = resource.get("name") or ""
            url = resource.get("url") or ""
            lower = f"{name} {url}".lower()
            if not url.lower().endswith(".zip"):
                continue
            if not ("flown" in lower or "flight line" in lower or "flight_line" in lower or "flightlines" in lower):
                continue
            out_name = safe_name(Path(urlparse(url).path).name or name)
            status, bytes_written = download_url(s, url, api_folder / out_name, max_bytes=MAX_DOWNLOAD_BYTES)
            downloads.append({"source_id": rec["source_id"], "kind": "cnra_api_flightlines", "name": name, "url": url, "status": status, "bytes": bytes_written})
        return
    except Exception as exc:
        downloads.append({"source_id": rec["source_id"], "kind": "cnra_api", "url": api, "status": f"metadata_error_{type(exc).__name__}"})

    try:
        text = s.get(rec["url"], timeout=REQUEST_TIMEOUT).text
    except Exception as exc:
        downloads.append({"source_id": rec["source_id"], "kind": "cnra", "url": rec["url"], "status": f"metadata_error_{type(exc).__name__}"})
        return
    folder = DOWNLOAD_DIR / rec["source_id"] / "cnra"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "dataset_page.html").write_text(text, encoding="utf-8", errors="ignore")
    links = [html.unescape(x) for x in re.findall(r'href=["\']([^"\']+)["\']', text)]
    candidates = []
    for link in links:
        abs_url = urljoin(rec["url"], link)
        lower = abs_url.lower()
        if (lower.endswith(".zip") or ".shp.zip" in lower) and any(k in lower for k in ("flight", "flown", "flightline", "flight_line", "flown_lines")):
            candidates.append(abs_url)
    for idx, url in enumerate(sorted(set(candidates)), start=1):
        name = safe_name(Path(urlparse(url).path).name or f"cnra_flightlines_{idx}.zip")
        status, bytes_written = download_url(s, url, folder / name, max_bytes=MAX_DOWNLOAD_BYTES)
        downloads.append({"source_id": rec["source_id"], "kind": "cnra_flightlines", "name": name, "url": url, "status": status, "bytes": bytes_written})


def process_arcgis_item(s: requests.Session, rec: dict, bboxes: list[dict], downloads: list[dict]) -> None:
    m = re.search(r"[?&]id=([A-Za-z0-9]+)", rec["url"])
    if not m:
        return
    item_id = m.group(1)
    api = f"https://www.arcgis.com/sharing/rest/content/items/{item_id}?f=json"
    folder = DOWNLOAD_DIR / rec["source_id"] / f"arcgis_{item_id}"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        data = s.get(api, timeout=REQUEST_TIMEOUT).json()
    except Exception as exc:
        downloads.append({"source_id": rec["source_id"], "kind": "arcgis", "url": api, "status": f"metadata_error_{type(exc).__name__}"})
        return
    (folder / "arcgis_item.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    extent = data.get("extent")
    if extent and len(extent) == 2:
        bboxes.append(bbox_feature(rec, (float(extent[0][0]), float(extent[0][1]), float(extent[1][0]), float(extent[1][1])), "arcgis_extent", item_id))
    downloads.append({"source_id": rec["source_id"], "kind": "arcgis", "url": api, "status": "metadata_saved"})


def process_arcgis_webmap_layers(s: requests.Session, rec: dict, downloads: list[dict]) -> None:
    m = re.search(r"[?&]id=([A-Za-z0-9]+)", rec["url"])
    if not m:
        return
    item_id = m.group(1)
    folder = DOWNLOAD_DIR / rec["source_id"] / f"arcgis_{item_id}"
    folder.mkdir(parents=True, exist_ok=True)
    api = f"https://www.arcgis.com/sharing/rest/content/items/{item_id}/data?f=json"
    try:
        data = s.get(api, timeout=REQUEST_TIMEOUT).json()
    except Exception as exc:
        downloads.append({"source_id": rec["source_id"], "kind": "arcgis_webmap", "url": api, "status": f"metadata_error_{type(exc).__name__}"})
        return
    (folder / "arcgis_webmap_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    for layer in data.get("operationalLayers", []) or []:
        title = layer.get("title") or layer.get("id") or "arcgis_layer"
        url = (layer.get("url") or "").rstrip("/")
        lower_title = title.lower()
        if "featureserver" not in url.lower():
            continue
        if not ("aem" in lower_title and ("survey line" in lower_title or "flight" in lower_title)):
            continue
        out_path = folder / f"{safe_name(title)}.geojson"
        if out_path.exists() and out_path.stat().st_size > 0:
            downloads.append({"source_id": rec["source_id"], "kind": "arcgis_feature_layer_geojson", "name": title, "url": url, "status": "exists", "bytes": out_path.stat().st_size})
            continue
        features = []
        offset = 0
        page_size = 2000
        while True:
            try:
                resp = s.get(
                    f"{url}/query",
                    params={
                        "where": "1=1",
                        "outFields": "*",
                        "returnGeometry": "true",
                        "outSR": "4326",
                        "f": "geojson",
                        "resultOffset": offset,
                        "resultRecordCount": page_size,
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                page = resp.json()
            except Exception as exc:
                downloads.append({"source_id": rec["source_id"], "kind": "arcgis_feature_layer_geojson", "name": title, "url": url, "status": f"query_error_{type(exc).__name__}"})
                features = []
                break
            page_features = page.get("features", []) or []
            features.extend(page_features)
            if len(page_features) < page_size and not page.get("exceededTransferLimit"):
                break
            offset += len(page_features) or page_size
            if offset > 200000:
                break
        if not features:
            continue
        collection = {"type": "FeatureCollection", "features": features}
        out_path.write_text(json.dumps(collection, ensure_ascii=False), encoding="utf-8")
        downloads.append({"source_id": rec["source_id"], "kind": "arcgis_feature_layer_geojson", "name": title, "url": url, "status": "downloaded", "bytes": out_path.stat().st_size, "features": len(features)})


def process_zenodo(s: requests.Session, rec: dict, downloads: list[dict]) -> None:
    m = re.search(r"zenodo\.org/records/(\d+)", rec["url"])
    if not m:
        return
    rid = m.group(1)
    api = f"https://zenodo.org/api/records/{rid}"
    folder = DOWNLOAD_DIR / rec["source_id"] / f"zenodo_{rid}"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        data = s.get(api, timeout=REQUEST_TIMEOUT).json()
    except Exception as exc:
        downloads.append({"source_id": rec["source_id"], "kind": "zenodo", "url": api, "status": f"metadata_error_{type(exc).__name__}"})
        return
    (folder / "zenodo_record.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for f in data.get("files", []) or []:
        name = f.get("key") or f.get("filename") or "zenodo_file"
        url = f.get("links", {}).get("self")
        size = int(f.get("size") or 0)
        lower = name.lower()
        if not url or not (is_candidate_name(name) or is_candidate_grid_name(name) or any(k in lower for k in ("nav", "gps", "flight", "line"))):
            continue
        max_bytes = MAX_DOWNLOAD_BYTES if not is_candidate_grid_name(name) else MAX_GRID_DOWNLOAD_BYTES
        status, bytes_written = download_url(s, url, folder / safe_name(name), max_bytes=max_bytes, expected_size=size)
        downloads.append({"source_id": rec["source_id"], "kind": "zenodo_file", "name": name, "url": url, "size": size, "status": status, "bytes": bytes_written})


def process_generic_page(s: requests.Session, rec: dict, downloads: list[dict]) -> None:
    try:
        r = s.get(rec["url"], timeout=REQUEST_TIMEOUT)
    except Exception as exc:
        downloads.append({"source_id": rec["source_id"], "kind": "generic_page", "url": rec["url"], "status": f"metadata_error_{type(exc).__name__}"})
        return
    if r.status_code >= 400:
        downloads.append({"source_id": rec["source_id"], "kind": "generic_page", "url": rec["url"], "status": f"http_{r.status_code}"})
        return
    text = r.text
    folder = DOWNLOAD_DIR / rec["source_id"] / "generic"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "source_page.html").write_text(text, encoding="utf-8", errors="ignore")
    links = [html.unescape(x) for x in re.findall(r'href=["\']([^"\']+)["\']', text)]
    downloaded = 0
    for link in sorted(set(links)):
        abs_url = urljoin(rec["url"], link)
        name = Path(urlparse(abs_url).path).name
        if not name or not (is_candidate_name(name) or is_candidate_grid_name(name)):
            continue
        is_grid = is_candidate_grid_name(name)
        status, bytes_written = download_url(s, abs_url, folder / safe_name(name), max_bytes=MAX_GRID_DOWNLOAD_BYTES if is_grid else MAX_DOWNLOAD_BYTES)
        downloads.append({"source_id": rec["source_id"], "kind": "generic_grid_link" if is_grid else "generic_link", "name": name, "url": abs_url, "status": status, "bytes": bytes_written})
        if status in ("downloaded", "exists"):
            downloaded += 1
        if downloaded >= 8:
            break


def process_manual_downloads(s: requests.Session, rec: dict, downloads: list[dict]) -> None:
    urls = MANUAL_DOWNLOADS.get(rec["source_id"], [])
    if not urls:
        return
    folder = DOWNLOAD_DIR / rec["source_id"] / "manual"
    folder.mkdir(parents=True, exist_ok=True)
    for url in urls:
        name = Path(urlparse(url).path).name or f"{rec['source_id']}_manual_download"
        if "catalog/file/get" in url:
            name = f"{rec['source_id']}_manual_flightlines.zip"
        status, bytes_written = download_url(s, url, folder / safe_name(name), max_bytes=MAX_DOWNLOAD_BYTES)
        downloads.append({"source_id": rec["source_id"], "kind": "manual_download", "name": name, "url": url, "status": status, "bytes": bytes_written})


def process_geojson_downloads(s: requests.Session, rec: dict, downloads: list[dict]) -> None:
    entries = GEOJSON_DOWNLOADS.get(rec["source_id"], [])
    if not entries:
        return
    folder = DOWNLOAD_DIR / rec["source_id"] / "source_specific_geojson"
    folder.mkdir(parents=True, exist_ok=True)
    for name, url in entries:
        status, bytes_written = download_url(s, url, folder / name, max_bytes=MAX_DOWNLOAD_BYTES)
        downloads.append({"source_id": rec["source_id"], "kind": "source_specific_geojson", "name": name, "url": url, "status": status, "bytes": bytes_written})


def extract_zips() -> list[dict]:
    rows = []
    for zpath in DOWNLOAD_DIR.rglob("*.zip"):
        out_dir = zpath.with_suffix("")
        if out_dir.exists() and any(out_dir.iterdir()):
            rows.append({"zip": str(zpath.relative_to(ROOT)), "status": "exists"})
            continue
        try:
            with zipfile.ZipFile(zpath) as z:
                names = z.namelist()
                keep = [
                    n for n in names
                    if Path(n).suffix.lower() in VECTOR_EXTS
                    or Path(n).suffix.lower() in GRID_EXTS
                    or Path(n).suffix.lower() in {".csv", ".xyz"}
                    or any(k in n.lower() for k in ("flight", "line", "survey", "footprint", "outline", "boundary"))
                ]
                if keep:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    z.extractall(out_dir)
                    rows.append({"zip": str(zpath.relative_to(ROOT)), "status": "extracted", "members": len(names)})
                else:
                    rows.append({"zip": str(zpath.relative_to(ROOT)), "status": "no_vector_like_members", "members": len(names)})
        except Exception as exc:
            rows.append({"zip": str(zpath.relative_to(ROOT)), "status": f"error_{type(exc).__name__}"})
    return rows


def vector_files() -> list[Path]:
    exts = {".shp", ".geojson", ".gpkg", ".gml", ".kml", ".dxf"}
    files = [p for p in DOWNLOAD_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    for p in DOWNLOAD_DIR.rglob("*.gdb"):
        if p.is_dir() and any(child.name.lower().endswith(".gdbtable") for child in p.iterdir() if child.is_file()):
            files.append(p)
    return files


def grid_files() -> list[Path]:
    exts = {".gxf", ".nc", ".tif", ".tiff", ".asc", ".grd"}
    candidates = [
        p for p in DOWNLOAD_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in exts
        and "colorbar" not in p.name.lower()
        and "legend" not in p.name.lower()
    ]
    by_source: dict[str, list[Path]] = {}
    for p in candidates:
        by_source.setdefault(source_id_from_path(p), []).append(p)
    selected = []
    for paths in by_source.values():
        def score(path: Path) -> tuple[int, int]:
            lower = path.name.lower()
            priority = 50
            if "res400" in lower:
                priority = 0
            elif "2022" in lower and "depth" in lower:
                priority = 1
            elif "depth" in lower:
                priority = 2
            elif "res1800" in lower:
                priority = 3
            elif "elevation" in lower:
                priority = 4
            return priority, path.stat().st_size
        selected.append(sorted(paths, key=score)[0])
    return selected


def csv_files() -> list[Path]:
    return [p for p in DOWNLOAD_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".xyz"}]


def source_id_from_path(path: Path) -> str:
    return next((part for part in path.parts if re.fullmatch(r"AEM\d{2}", part)), "")


def read_table_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".xyz":
        return XYZ_COLUMNS
    return list(pd.read_csv(path, nrows=0).columns)


def read_table_columns_only(path: Path, usecols: list[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".xyz":
        return pd.read_csv(
            path,
            comment="/",
            sep=r"\s+",
            names=XYZ_COLUMNS,
            usecols=usecols,
            low_memory=False,
        )
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def vector_path_is_relevant(path: Path) -> bool:
    lower = str(path).lower()
    if any(k in lower for k in BAD_VECTOR_KEYWORDS):
        return False
    return True


def footprint_feature(source_id: str, rec: dict, geom, source: str, path: Path, feature_count: int, precision_rank: int) -> dict:
    return {
        "type": "Feature",
        "properties": {
            "source_id": source_id,
            "seq": rec.get("seq"),
            "region_label": rec.get("region_label", ""),
            "dataset_label": rec.get("dataset_label", ""),
            "source_url": rec.get("url", ""),
            "footprint_source": source,
            "precision_rank": precision_rank,
            "vector_path": str(path.relative_to(ROOT)),
            "feature_count": int(feature_count),
        },
        "geometry": mapping(geom),
    }


def geometry_union(gdf: gpd.GeoDataFrame):
    geom = gdf.geometry.dropna()
    if geom.empty:
        return None
    try:
        out = geom.union_all()
    except AttributeError:
        out = unary_union(list(geom))
    if out.is_empty:
        return None
    return out.simplify(0.02, preserve_topology=True)


def read_vector_layers(path: Path) -> list[tuple[str | None, gpd.GeoDataFrame]]:
    if path.suffix.lower() == ".gdb":
        try:
            import pyogrio

            layers = pyogrio.list_layers(path)[:, 0]
        except Exception:
            layers = []
        out = []
        for layer in layers:
            try:
                out.append((str(layer), gpd.read_file(path, layer=layer)))
            except Exception:
                continue
        return out
    if path.suffix.lower() == ".gml":
        try:
            import pyogrio

            layers = [str(x) for x in pyogrio.list_layers(path)[:, 0]]
            if "Campaign" in layers:
                return [("Campaign", gpd.read_file(path, layer="Campaign"))]
        except Exception:
            pass
    return [(None, gpd.read_file(path))]


def footprint_from_vectors() -> tuple[list[dict], list[dict]]:
    features = []
    rows = []
    for path in vector_files():
        if not vector_path_is_relevant(path):
            rows.append({"path": str(path.relative_to(ROOT)), "status": "skipped_non_aem_vector"})
            continue
        try:
            source_id = source_id_from_path(path)
            rec = RECORD_BY_ID.get(source_id, {})
            layer_rows = 0
            layer_features = 0
            for layer, gdf in read_vector_layers(path):
                if gdf.empty:
                    continue
                if source_id == "AEM02" and "EM" in gdf.columns:
                    gdf = gdf[pd.to_numeric(gdf["EM"], errors="coerce").fillna(0).astype(int) == 1].copy()
                if source_id == "AEM02" and gdf.empty:
                    continue
                if gdf.crs is None:
                    gdf = gdf.set_crs(VECTOR_DEFAULT_CRS.get(source_id, 4326), allow_override=True)
                gdf = gdf.to_crs(4326)
                geom = geometry_union(gdf)
                if geom is None:
                    continue
                features.append(footprint_feature(source_id, rec, geom, "downloaded_vector_geometry", path, len(gdf), 1))
                layer_rows += 1
                layer_features += len(gdf)
            if layer_rows == 0:
                rows.append({"path": str(path.relative_to(ROOT)), "source_id": source_id, "status": "empty"})
            else:
                rows.append({"path": str(path.relative_to(ROOT)), "source_id": source_id, "status": "read", "layers": layer_rows, "features": layer_features})
        except Exception as exc:
            rows.append({"path": str(path.relative_to(ROOT)), "status": f"error_{type(exc).__name__}"})
    return features, rows


def parse_gxf_grid(path: Path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header: dict[str, str] = {}
    grid_start = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.upper() == "#GRID":
            grid_start = i + 1
            break
        if line.startswith("#") and i + 1 < len(lines):
            key = line[1:].strip().upper()
            val = lines[i + 1].strip()
            header[key] = val
            i += 2
        else:
            i += 1
    if grid_start is None:
        raise ValueError("No #GRID block found")
    ncols = int(float(header["POINTS"].split()[0]))
    nrows = int(float(header["ROWS"].split()[0]))
    x0 = float(header["XORIGIN"].split()[0])
    y0 = float(header["YORIGIN"].split()[0])
    dx = float(header["PTSEPARATION"].split()[0])
    dy = float(header["RWSEPARATION"].split()[0])
    dummy = float(header.get("DUMMY", "-1e32").split()[0])
    values = np.fromstring(" ".join(lines[grid_start:]), sep=" ", dtype=float)
    if values.size < nrows * ncols:
        raise ValueError(f"Grid has {values.size} values, expected {nrows * ncols}")
    arr = values[: nrows * ncols].reshape((nrows, ncols))
    mask = np.isfinite(arr) & (np.abs(arr - dummy) > 1e-20) & (arr > dummy * 0.1)
    transform = Affine.translation(x0, y0) * Affine.scale(dx, dy)
    return mask.astype("uint8"), transform


def footprint_from_mask(mask: np.ndarray, transform: Affine):
    geoms = []
    for geom, value in rasterio.features.shapes(mask, mask=mask.astype(bool), transform=transform):
        if value == 1:
            geoms.append(shape(geom))
    if not geoms:
        return None
    geom = unary_union(geoms)
    if geom.is_empty:
        return None
    return geom.simplify(0.03, preserve_topology=True)


def footprint_from_netcdf(path: Path):
    import xarray as xr

    ds = xr.open_dataset(path)
    candidates = []
    for name, da in ds.data_vars.items():
        lower = name.lower()
        if lower.endswith("_bnds") or lower in {"spatial_ref", "albers_conical_equal_area"}:
            continue
        if da.ndim >= 2 and np.issubdtype(da.dtype, np.number):
            candidates.append((name, da))
    if not candidates:
        return None
    name, da = candidates[0]
    while da.ndim > 2:
        da = da.isel({da.dims[0]: 0})
    arr = da.values
    mask = np.isfinite(arr)
    x_name = next((c for c in ("lon", "longitude", "x") if c in ds.coords or c in ds.variables), None)
    y_name = next((c for c in ("lat", "latitude", "y") if c in ds.coords or c in ds.variables), None)
    if x_name is None or y_name is None:
        return None
    x = np.asarray(ds[x_name].values)
    y = np.asarray(ds[y_name].values)
    if x.ndim != 1 or y.ndim != 1:
        return None
    dx = float(np.nanmedian(np.diff(x)))
    dy = float(np.nanmedian(np.diff(y)))
    transform = Affine.translation(float(x.min()), float(y.min())) * Affine.scale(abs(dx), abs(dy))
    geom = footprint_from_mask(mask.astype("uint8"), transform)
    if geom is None:
        return None
    crs_attrs = {}
    for crs_name in ("spatial_ref", "albers_conical_equal_area"):
        if crs_name in ds.variables:
            crs_attrs = dict(ds[crs_name].attrs)
            break
    crs = crs_attrs.get("crs_wkt") or crs_attrs.get("proj_string") or 4326
    return gpd.GeoSeries([geom], crs=crs).to_crs(4326).iloc[0]


def footprint_from_raster(path: Path):
    import rasterio

    with rasterio.open(path) as ds:
        out_h = min(ds.height, 1600)
        out_w = max(1, int(round(ds.width * out_h / ds.height)))
        if out_w > 1600:
            out_w = 1600
            out_h = max(1, int(round(ds.height * out_w / ds.width)))
        arr = ds.read(out_shape=(ds.count, out_h, out_w))
        transform = ds.transform * Affine.scale(ds.width / out_w, ds.height / out_h)

        if ds.nodata is not None:
            mask = np.any(arr != ds.nodata, axis=0)
        elif ds.count >= 3 and np.issubdtype(arr.dtype, np.integer):
            rgb = np.moveaxis(arr[:3], 0, -1)
            # Mask the white area outside the survey.
            mask = ~np.all(rgb >= 245, axis=-1)
        else:
            mask = np.isfinite(arr[0])

        if mask.mean() > 0.98:
            return None
        geom = footprint_from_mask(mask.astype("uint8"), transform)
        if geom is None:
            return None
        crs = ds.crs or 4326
        return gpd.GeoSeries([geom], crs=crs).to_crs(4326).iloc[0]


def footprint_from_grids() -> tuple[list[dict], list[dict]]:
    features = []
    rows = []
    for path in grid_files():
        source_id = source_id_from_path(path)
        rec = RECORD_BY_ID.get(source_id, {})
        try:
            if path.suffix.lower() == ".gxf":
                mask, transform = parse_gxf_grid(path)
                geom = footprint_from_mask(mask, transform)
            elif path.suffix.lower() == ".nc":
                geom = footprint_from_netcdf(path)
            elif path.suffix.lower() in {".tif", ".tiff"}:
                geom = footprint_from_raster(path)
            else:
                rows.append({"path": str(path.relative_to(ROOT)), "source_id": source_id, "status": "skipped_grid_format"})
                continue
            if geom is None:
                rows.append({"path": str(path.relative_to(ROOT)), "source_id": source_id, "status": "empty_grid"})
                continue
            features.append(footprint_feature(source_id, rec, geom, "surface_grid_valid_data_footprint", path, 1, 2))
            rows.append({"path": str(path.relative_to(ROOT)), "source_id": source_id, "status": "read_grid"})
        except Exception as exc:
            rows.append({"path": str(path.relative_to(ROOT)), "source_id": source_id, "status": f"error_{type(exc).__name__}"})
    return features, rows


def build_csv_lines(
    pts: pd.DataFrame,
    x_col: str,
    y_col: str,
    line_col: str | None,
    vertex_col: str | None,
    crs,
) -> gpd.GeoDataFrame | None:
    geoms = []
    if line_col and line_col in pts.columns:
        groups = pts.groupby(line_col)
    else:
        groups = [(None, pts)]
    for _, sub in groups:
        if vertex_col and vertex_col in sub.columns:
            sub = sub.sort_values(vertex_col)
        xy = sub[[x_col, y_col]].to_numpy(dtype=float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) >= 2:
            geoms.append(LineString(xy))
    if not geoms:
        return None
    return gpd.GeoDataFrame(geometry=geoms, crs=crs)


def build_csv_point_hull(
    pts: pd.DataFrame,
    x_col: str,
    y_col: str,
    epsg_col: str | None,
    crs,
    max_points: int = 50000,
):
    hulls = []
    if epsg_col and epsg_col in pts.columns:
        groups = pts.groupby(epsg_col)
    else:
        groups = [(None, pts)]
    for epsg_value, sub in groups:
        if len(sub) > max_points:
            sub = sub.iloc[:: max(1, math.ceil(len(sub) / max_points))]
        xy = sub[[x_col, y_col]].dropna().to_numpy(dtype=float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < 3:
            continue
        if epsg_col and epsg_col in pts.columns:
            epsg_digits = re.search(r"(\d+)", str(epsg_value))
            if not epsg_digits:
                continue
            group_crs = int(epsg_digits.group(1))
        else:
            group_crs = crs
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=group_crs)
        geom_series = gdf.to_crs(4326).geometry
        unioned = geom_series.union_all() if hasattr(geom_series, "union_all") else geom_series.unary_union
        hulls.append(unioned.convex_hull)
    if not hulls:
        return None
    return unary_union(hulls).buffer(0.03)


def footprint_from_csvs() -> tuple[list[dict], list[dict]]:
    features = []
    rows = []
    for path in csv_files():
        source_id = source_id_from_path(path)
        rec = RECORD_BY_ID.get(source_id, {})
        try:
            cols = {c.lower(): c for c in read_table_columns(path)}
            lon_col = next((cols[c] for c in cols if c in ("lon", "long", "longitude")), None)
            lat_col = next((cols[c] for c in cols if c in ("lat", "latitude")), None)
            east_col = next(
                (
                    cols[c]
                    for c in cols
                    if c in ("easting", "x")
                    or c.startswith("e_utm")
                    or c.startswith("utm_e")
                    or c.endswith("_easting")
                ),
                None,
            )
            north_col = next(
                (
                    cols[c]
                    for c in cols
                    if c in ("northing", "y")
                    or c.startswith("n_utm")
                    or c.startswith("utm_n")
                    or c.endswith("_northing")
                ),
                None,
            )
            epsg_col = next((cols[c] for c in cols if c == "epsg"), None)
            line_col = next((cols[c] for c in cols if c in ("survey_line", "line", "line_no", "lineno", "flightline")), None)
            vertex_col = next((cols[c] for c in cols if c in ("vertex", "fid", "point_id", "sequence")), None)

            if lon_col and lat_col:
                usecols = [lon_col, lat_col] + ([line_col] if line_col else []) + ([vertex_col] if vertex_col else [])
                df = read_table_columns_only(path, usecols)
                pts = df.dropna(subset=[lon_col, lat_col])
                crs = 4326
                x_col, y_col = lon_col, lat_col
            elif east_col and north_col and (epsg_col or source_id in CSV_DEFAULT_CRS):
                usecols = [east_col, north_col] + ([epsg_col] if epsg_col else []) + ([line_col] if line_col else []) + ([vertex_col] if vertex_col else [])
                df = read_table_columns_only(path, usecols)
                drop_cols = [east_col, north_col] + ([epsg_col] if epsg_col else [])
                pts = df.dropna(subset=drop_cols)
                x_col, y_col = east_col, north_col
                crs = None if epsg_col else CSV_DEFAULT_CRS[source_id]
            else:
                rows.append({"path": str(path.relative_to(ROOT)), "source_id": source_id, "status": "no_coordinate_columns"})
                continue

            if len(pts) < 2:
                rows.append({"path": str(path.relative_to(ROOT)), "source_id": source_id, "status": "too_few_points"})
                continue

            if len(pts) > 50000:
                geom = build_csv_point_hull(pts, x_col, y_col, epsg_col if epsg_col and epsg_col in pts.columns else None, crs)
                if geom is None:
                    rows.append({"path": str(path.relative_to(ROOT)), "source_id": source_id, "status": "too_few_valid_points"})
                    continue
                features.append(footprint_feature(source_id, rec, geom, "downloaded_csv_survey_point_hull", path, len(pts), 1))
                rows.append({"path": str(path.relative_to(ROOT)), "source_id": source_id, "status": "read_csv_point_hull", "features": len(pts)})
                continue

            line_geoms = []
            if epsg_col and epsg_col in pts.columns:
                for epsg_value, sub in pts.groupby(epsg_col):
                    epsg_digits = re.search(r"(\d+)", str(epsg_value))
                    if not epsg_digits:
                        continue
                    sub_gdf = build_csv_lines(sub, x_col, y_col, line_col, vertex_col, int(epsg_digits.group(1)))
                    if sub_gdf is not None:
                        line_geoms.extend(sub_gdf.to_crs(4326).geometry.tolist())
            else:
                sub_gdf = build_csv_lines(pts, x_col, y_col, line_col, vertex_col, crs)
                if sub_gdf is not None:
                    line_geoms.extend(sub_gdf.geometry.tolist())
            if not line_geoms:
                rows.append({"path": str(path.relative_to(ROOT)), "source_id": source_id, "status": "too_few_valid_lines"})
                continue
            geom = unary_union(line_geoms).buffer(0.03)
            features.append(footprint_feature(source_id, rec, geom, "downloaded_csv_flightline_geometry", path, len(pts), 1))
            rows.append({"path": str(path.relative_to(ROOT)), "source_id": source_id, "status": "read_csv_flightline", "features": len(pts)})
        except Exception as exc:
            rows.append({"path": str(path.relative_to(ROOT)), "source_id": source_id, "status": f"error_{type(exc).__name__}"})
    return features, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_bbox_features() -> list[dict]:
    path = DERIVED_DIR / "global_aem_metadata_bboxes.geojson"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("features", []))
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and rebuild global AEM footprint products.")
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Rebuild footprints from files already under data/13 Global AEM/downloads without querying remote sites.",
    )
    args = parser.parse_args()

    global RECORD_BY_ID
    records = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    RECORD_BY_ID = {r["source_id"]: r for r in records}
    s = session()
    downloads: list[dict] = []
    bbox_features: list[dict] = load_existing_bbox_features() if args.skip_downloads else []

    if args.skip_downloads:
        print("Skipping remote downloads; rebuilding footprints from local files.")
    else:
        for idx, rec in enumerate(records, start=1):
            label = rec["region_label"][:60].encode("ascii", "backslashreplace").decode("ascii")
            print(f"[{idx:02d}/{len(records)}] {rec['source_id']} {label}")
            parent_ids = sciencebase_ids_for_record(s, rec) + EXTRA_SCIENCEBASE_IDS.get(rec["source_id"], [])
            parent_ids = list(dict.fromkeys(parent_ids))
            ids = []
            for parent_id in parent_ids:
                ids.append(parent_id)
                ids.extend(sciencebase_child_ids(s, rec, parent_id, downloads))
                time.sleep(0.05)
            ids = list(dict.fromkeys(ids))
            for item_id in ids:
                process_sciencebase(s, rec, item_id, downloads, bbox_features)
                time.sleep(0.15)
            if rec["source_id"] == "AEM03":
                process_cnra_aem(s, rec, downloads)
            if "arcgis.com/home/item" in rec["url"]:
                process_arcgis_item(s, rec, bbox_features, downloads)
                process_arcgis_webmap_layers(s, rec, downloads)
            if "zenodo.org/records" in rec["url"]:
                process_zenodo(s, rec, downloads)
            if not ids and rec["source_id"] not in {"AEM03"} and "arcgis.com/home/item" not in rec["url"] and "zenodo.org/records" not in rec["url"]:
                process_generic_page(s, rec, downloads)
            process_manual_downloads(s, rec, downloads)
            process_geojson_downloads(s, rec, downloads)
            download_remote_zip_members(s, rec, downloads)

    for rec in records:
        if rec["source_id"] in APPROX_BBOX:
            # Add missing approximate bounds.
            if not any(f["properties"]["source_id"] == rec["source_id"] for f in bbox_features):
                bbox_features.append(bbox_feature(rec, APPROX_BBOX[rec["source_id"]], "approx_from_description"))

    if not args.skip_downloads:
        write_csv(DATA_DIR / "download_status.csv", downloads)
        (DATA_DIR / "download_status.json").write_text(json.dumps(downloads, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_rows = extract_zips()
    write_csv(DATA_DIR / "zip_extract_status.csv", zip_rows)

    vector_features, vector_rows = footprint_from_vectors()
    write_csv(DATA_DIR / "vector_read_status.csv", vector_rows)
    grid_features, grid_rows = footprint_from_grids()
    write_csv(DATA_DIR / "grid_read_status.csv", grid_rows)
    csv_features, csv_rows = footprint_from_csvs()
    write_csv(DATA_DIR / "csv_flightline_read_status.csv", csv_rows)

    # Select the most detailed footprint for each survey.
    exact_features = vector_features + csv_features
    have_exact = {f["properties"]["source_id"] for f in exact_features}
    have_grid = {f["properties"]["source_id"] for f in grid_features}
    final_features = (
        exact_features
        + [f for f in grid_features if f["properties"]["source_id"] not in have_exact]
        + [f for f in bbox_features if f["properties"]["source_id"] not in have_exact and f["properties"]["source_id"] not in have_grid]
    )
    geojson = {"type": "FeatureCollection", "features": final_features}
    out_geojson = DERIVED_DIR / "global_aem_footprints.geojson"
    out_geojson.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
    bbox_geojson = DERIVED_DIR / "global_aem_metadata_bboxes.geojson"
    bbox_geojson.write_text(json.dumps({"type": "FeatureCollection", "features": bbox_features}, ensure_ascii=False), encoding="utf-8")

    summary = {
        "n_manifest_records": len(records),
        "n_download_rows": len(downloads),
        "download_mode": "local_rebuild_only" if args.skip_downloads else "remote_download_and_rebuild",
        "n_vector_files_read": sum(1 for r in vector_rows if r.get("status") == "read"),
        "n_grid_files_read": sum(1 for r in grid_rows if r.get("status") == "read_grid"),
        "n_csv_footprints_read": sum(1 for r in csv_rows if r.get("status") in {"read_csv_flightline", "read_csv_point_hull"}),
        "n_sources_with_exact_geometry": len(have_exact),
        "n_sources_with_grid_footprint": len(have_grid - have_exact),
        "n_bbox_features": len(bbox_features),
        "n_final_footprints": len(final_features),
        "outputs": {
            "footprints_geojson": str(out_geojson.relative_to(ROOT)),
            "metadata_bbox_geojson": str(bbox_geojson.relative_to(ROOT)),
            "download_status_csv": str((DATA_DIR / "download_status.csv").relative_to(ROOT)),
            "vector_status_csv": str((DATA_DIR / "vector_read_status.csv").relative_to(ROOT)),
            "grid_status_csv": str((DATA_DIR / "grid_read_status.csv").relative_to(ROOT)),
            "csv_flightline_status_csv": str((DATA_DIR / "csv_flightline_read_status.csv").relative_to(ROOT)),
        },
    }
    (DATA_DIR / "global_aem_download_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
