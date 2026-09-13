from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pygmt
import xarray as xr
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures" / "Fig1b_conceptual_framework"
ACTIVE_LOOKUP_PATH = ROOT / "data" / "2 well WTD" / "active_cell_lookup.csv"
TOPO_PATH = ROOT / "data" / "5 topography" / "topo_1km.csv"
AEM_NC_PATH = (
    ROOT
    / "data_raw"
    / "1 MAP Electrical Resistivity & Facies Classification"
    / "MAP_RegionalAEM_2022_ResistivityFacies_DepthGrids_log10res_masked_latestmask.nc"
)

SLICE_TOP_M = 80.0
SLICE_BOTTOM_M = 85.0
SLICE_DEPTH_M = 0.5 * (SLICE_TOP_M + SLICE_BOTTOM_M)
PERSPECTIVE = [238, 24]
PROJECTION = "X15.2c/8.2c"
ZSIZE = "3.4c"


def output_paths() -> dict[str, Path]:
    prefix = OUT_DIR / "Fig1b_3D_AEM_perspective"
    return {
        "png": prefix.with_suffix(".png"),
        "pdf": prefix.with_suffix(".pdf"),
        "topo_nc": OUT_DIR / "mrva_topography_1km.nc",
        "slice_nc": OUT_DIR / "mrva_log10res_80_85m_slice.nc",
        "slice_elev_nc": OUT_DIR / "mrva_slice_elevation_80_85m.nc",
        "terrain_cpt": OUT_DIR / "terrain_gray_soft.cpt",
        "res_cpt": OUT_DIR / "aem_resistivity_80_85m.cpt",
    }


def build_topography_grid() -> xr.DataArray:
    active = pd.read_csv(ACTIVE_LOOKUP_PATH)
    topo = pd.read_csv(TOPO_PATH)
    merged = active.merge(topo, left_on="node_id", right_on="grid_id", how="left")

    x_vals_m = np.sort(merged["x_center"].unique().astype(np.float64))
    y_vals_m = np.sort(merged["y_center"].unique().astype(np.float64))
    x_to_idx = {float(x): i for i, x in enumerate(x_vals_m)}
    y_to_idx = {float(y): i for i, y in enumerate(y_vals_m)}

    arr = np.full((len(y_vals_m), len(x_vals_m)), np.nan, dtype=np.float32)
    for row in merged.itertuples(index=False):
        iy = y_to_idx[float(row.y_center)]
        ix = x_to_idx[float(row.x_center)]
        arr[iy, ix] = np.float32(row.mean_elevation_m)

    return xr.DataArray(
        arr,
        coords={"y": y_vals_m / 1000.0, "x": x_vals_m / 1000.0},
        dims=("y", "x"),
        name="mean_elevation_m",
        attrs={"long_name": "MRVA mean elevation", "units": "m NAVD88"},
    )


def build_resistivity_slice(topo_grid: xr.DataArray) -> xr.DataArray:
    x_vals_m = np.asarray(topo_grid["x"].values, dtype=np.float64) * 1000.0
    y_vals_m = np.asarray(topo_grid["y"].values, dtype=np.float64) * 1000.0
    with xr.open_dataset(AEM_NC_PATH, engine="netcdf4") as ds:
        slice_da = (
            ds["log10res"]
            .sel(z=SLICE_DEPTH_M, method="nearest")
            .interp(x=x_vals_m, y=y_vals_m, method="linear")
            .transpose("y", "x")
            .astype(np.float32)
        )

    out = xr.DataArray(
        slice_da.to_numpy(),
        coords=topo_grid.coords,
        dims=topo_grid.dims,
        name="log10res_80_85m_bls",
        attrs={
            "long_name": "AEM log10 resistivity",
            "units": "ohm m",
            "depth_interval_bls_m": f"{SLICE_TOP_M:.0f}-{SLICE_BOTTOM_M:.0f}",
        },
    )
    return out.where(np.isfinite(topo_grid))


def write_custom_cpts(paths: dict[str, Path], terrain_min: float, terrain_max: float) -> None:
    terrain_lines = [
        f"{terrain_min:.3f} 228/234/214 {terrain_min + 0.38 * (terrain_max - terrain_min):.3f} 154/183/101",
        f"{terrain_min + 0.38 * (terrain_max - terrain_min):.3f} 154/183/101 {terrain_min + 0.72 * (terrain_max - terrain_min):.3f} 79/138/61",
        f"{terrain_min + 0.72 * (terrain_max - terrain_min):.3f} 79/138/61 {terrain_max:.3f} 238/238/238",
        "B 245/245/245",
        "F 70/70/70",
        "N 255/255/255",
    ]
    paths["terrain_cpt"].write_text("\n".join(terrain_lines) + "\n", encoding="utf-8")

    resistivity_lines = [
        "0.0 52/33/110 0.5 55/86/175",
        "0.5 55/86/175 1.0 49/160/122",
        "1.0 49/160/122 1.5 177/216/86",
        "1.5 177/216/86 2.0 248/198/56",
        "2.0 248/198/56 2.5 239/133/182",
        "2.5 239/133/182 3.0 248/248/248",
        "B 35/24/86",
        "F 255/255/255",
        "N 230/230/230",
    ]
    paths["res_cpt"].write_text("\n".join(resistivity_lines) + "\n", encoding="utf-8")


def plot_scene(
    topo_grid: xr.DataArray,
    resistivity_grid: xr.DataArray,
    slice_elev_grid: xr.DataArray,
    paths: dict[str, Path],
) -> None:
    valid_topo = topo_grid.to_numpy()
    valid_slice = slice_elev_grid.to_numpy()

    x_min = float(topo_grid["x"].min())
    x_max = float(topo_grid["x"].max())
    y_min = float(topo_grid["y"].min())
    y_max = float(topo_grid["y"].max())
    z_top = float(np.nanmax(valid_topo) + 35.0)
    z_bottom = float(np.nanmin(valid_slice) - 40.0)
    region = [x_min, x_max, y_min, y_max, z_bottom, z_top]

    pygmt.config(
        MAP_FRAME_TYPE="plain",
        FONT="9p,Helvetica,black",
        FONT_LABEL="9p,Helvetica,black",
        MAP_ANNOT_OFFSET_PRIMARY="2p",
    )

    fig = pygmt.Figure()
    fig.grdview(
        grid=topo_grid,
        region=region,
        projection=PROJECTION,
        zsize=ZSIZE,
        perspective=PERSPECTIVE,
        surftype="s",
        cmap=str(paths["terrain_cpt"]),
        shading="+a235+nt0.9+m0.15",
        transparency=16,
        frame=[
            "xaf+lEasting (km)",
            "yaf+lNorthing (km)",
            "zaf+lElevation (m)",
            "WSnEZ",
        ],
    )
    fig.grdview(
        grid=slice_elev_grid,
        drape_grid=resistivity_grid,
        region=region,
        projection=PROJECTION,
        zsize=ZSIZE,
        perspective=PERSPECTIVE,
        surftype="s",
        cmap=str(paths["res_cpt"]),
        shading=0.18,
        transparency=3,
        plane=z_bottom,
        facade_fill="235/235/235",
        facade_pen="0.18p,gray35",
        frame=False,
    )
    fig.colorbar(
        cmap=str(paths["res_cpt"]),
        position="JBC+w8.8c/0.32c+o0c/0.5c+h",
        frame=['xaf0.5+lLog10 Resistivity (ohm-m)'],
        box="+gwhite@75+p0.35p,gray40+r0.04c",
    )
    fig.text(
        position="BC",
        text=f"AEM resistivity sheet: {SLICE_TOP_M:.0f}-{SLICE_BOTTOM_M:.0f} m b.l.s.",
        font="8.0p,Helvetica,black",
        justify="BC",
        offset="0c/1.55c",
        fill="white@70",
        pen="0.25p,gray50",
    )
    fig.savefig(paths["png"], dpi=500)
    fig.savefig(paths["pdf"])


def crop_white_png(path: Path, pad: int = 18) -> None:
    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()
    if bbox is None:
        return
    left = max(bbox[0] - pad, 0)
    upper = max(bbox[1] - pad, 0)
    right = min(bbox[2] + pad, image.size[0])
    lower = min(bbox[3] + pad, image.size[1])
    image.crop((left, upper, right, lower)).save(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = output_paths()
    topo_grid = build_topography_grid()
    resistivity_grid = build_resistivity_slice(topo_grid)
    slice_elev_grid = (topo_grid - SLICE_DEPTH_M).where(np.isfinite(resistivity_grid))

    terrain_min = float(np.nanmin(topo_grid.to_numpy()))
    terrain_max = float(np.nanmax(topo_grid.to_numpy()))
    write_custom_cpts(paths, terrain_min=terrain_min, terrain_max=terrain_max)

    topo_grid.to_netcdf(paths["topo_nc"])
    resistivity_grid.to_netcdf(paths["slice_nc"])
    slice_elev_grid.to_netcdf(paths["slice_elev_nc"])
    plot_scene(topo_grid, resistivity_grid, slice_elev_grid, paths)
    crop_white_png(paths["png"])

    print(f"png: {paths['png']}")
    print(f"pdf: {paths['pdf']}")
    print(f"topo_nc: {paths['topo_nc']}")
    print(f"slice_nc: {paths['slice_nc']}")
    print(f"slice_elev_nc: {paths['slice_elev_nc']}")
    print(f"slice_depth_bls_m: {SLICE_DEPTH_M:.1f}")


if __name__ == "__main__":
    main()
