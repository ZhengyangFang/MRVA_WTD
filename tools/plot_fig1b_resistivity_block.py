from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import xarray as xr
from scipy.ndimage import binary_erosion


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

HSTEP = 4
ZSTEP = 10.0
MAX_DEPTH = 340.0
WIDTH_STRETCH = 1.8

RESISTIVITY_CMAP = [
    "#2c1e7f",
    "#3155a6",
    "#2ca7c9",
    "#97d35f",
    "#ffd34a",
    "#f3a0c5",
    "#f4f1f6",
]
TERRAIN_CMAP = [
    "#d5c79c",
    "#d8dd8e",
    "#8eb965",
    "#60854e",
    "#40603f",
]
def output_paths() -> dict[str, Path]:
    prefix = OUT_DIR / "Fig1b_resistivity_block3d"
    return {
        "png": prefix.with_suffix(".png"),
        "pdf": prefix.with_suffix(".pdf"),
    }


def save_png_as_pdf(png_path: Path, pdf_path: Path) -> None:
    img = plt.imread(png_path)
    fig, ax = plt.subplots(figsize=(img.shape[1] / 200.0, img.shape[0] / 200.0), dpi=200)
    ax.imshow(img)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(pdf_path, dpi=400, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def load_coarse_topography() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = pd.read_csv(ACTIVE_LOOKUP_PATH)
    topo = pd.read_csv(TOPO_PATH)
    df = active.merge(topo, left_on="node_id", right_on="grid_id", how="left")
    coarse = (
        df.assign(crow=df["row"] // HSTEP, ccol=df["col"] // HSTEP)
        .groupby(["crow", "ccol"], as_index=False)
        .agg(topo=("mean_elevation_m", "mean"))
    )

    nr = int(coarse["crow"].max()) + 1
    nc = int(coarse["ccol"].max()) + 1
    ztop = np.full((nr, nc), np.nan, dtype=np.float32)
    rr = coarse["crow"].to_numpy(dtype=np.int64)
    cc = coarse["ccol"].to_numpy(dtype=np.int64)
    ztop[rr, cc] = coarse["topo"].to_numpy(dtype=np.float32)

    return df, ztop, rr, cc, np.isfinite(ztop)


def interpolate_resistivity(df: pd.DataFrame, nr: int, nc: int, target_depths: np.ndarray) -> np.ndarray:
    xvals = np.array(
        [df.loc[df["col"] // HSTEP == j, "x_center"].median() for j in range(nc)],
        dtype=np.float64,
    )
    yvals = np.array(
        [df.loc[df["row"] // HSTEP == i, "y_center"].median() for i in range(nr)],
        dtype=np.float64,
    )

    with xr.open_dataset(AEM_NC_PATH, engine="netcdf4") as ds:
        return (
            ds["log10res"]
            .interp(x=xvals, y=yvals, z=target_depths, method="nearest")
            .transpose("z", "y", "x")
            .to_numpy()
            .astype(np.float32)
        )


def build_voxel_curtain(ztop: np.ndarray, mask: np.ndarray, resistivity: np.ndarray) -> pv.DataSet:
    perim2d = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)

    zmin = float(np.floor(np.nanmin(ztop) - MAX_DEPTH - 10.0))
    zmax = float(np.ceil(np.nanmax(ztop) + 10.0))
    z_edges = np.arange(zmin, zmax + ZSTEP, ZSTEP, dtype=np.float64)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    target_depths = np.arange(ZSTEP / 2.0, MAX_DEPTH + ZSTEP / 2.0, ZSTEP, dtype=np.float64)

    nr, nc = ztop.shape
    nz = len(z_centers)
    occupied = np.zeros((nr, nc, nz), dtype=np.uint8)
    resvals = np.full((nr, nc, nz), np.nan, dtype=np.float32)

    for r, c in zip(*np.where(perim2d)):
        depth_bls = ztop[r, c] - z_centers
        inside = (depth_bls >= 0.0) & (depth_bls <= MAX_DEPTH)
        occupied[r, c, inside] = 1
        if inside.any():
            d_idx = np.rint((depth_bls[inside] - target_depths[0]) / ZSTEP).astype(np.int64)
            d_idx = np.clip(d_idx, 0, len(target_depths) - 1)
            resvals[r, c, inside] = resistivity[d_idx, r, c]

    img = pv.ImageData()
    img.dimensions = (nr + 1, nc + 1, nz + 1)
    img.origin = (0.0, 0.0, zmin)
    img.spacing = (HSTEP, HSTEP * WIDTH_STRETCH, ZSTEP)
    img.cell_data["occupied"] = occupied.flatten(order="F")
    img.cell_data["log10res"] = np.nan_to_num(resvals, nan=-9999.0).flatten(order="F")
    return img.threshold(value=0.5, scalars="occupied")


def build_top_skin(ztop: np.ndarray, mask: np.ndarray) -> pv.PolyData:
    points: list[list[float]] = []
    faces: list[int] = []
    elev: list[float] = []
    pid = 0

    for r, c in zip(*np.where(mask)):
        x0 = r * HSTEP
        x1 = (r + 1) * HSTEP
        y0 = c * HSTEP * WIDTH_STRETCH
        y1 = (c + 1) * HSTEP * WIDTH_STRETCH
        z = float(ztop[r, c])

        points.extend(
            [
                [x0, y0, z],
                [x1, y0, z],
                [x1, y1, z],
                [x0, y1, z],
            ]
        )
        faces.extend([4, pid, pid + 1, pid + 2, pid + 3])
        elev.append(z)
        pid += 4

    top = pv.PolyData(np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int64))
    top.cell_data["elev"] = np.asarray(elev, dtype=np.float32)
    return top


def plot_block() -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = output_paths()

    df, ztop, rr, cc, mask = load_coarse_topography()
    target_depths = np.arange(ZSTEP / 2.0, MAX_DEPTH + ZSTEP / 2.0, ZSTEP, dtype=np.float64)
    resistivity = interpolate_resistivity(df, ztop.shape[0], ztop.shape[1], target_depths)

    curtain = build_voxel_curtain(ztop, mask, resistivity)
    top_skin = build_top_skin(ztop, mask)
    outline = top_skin.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )

    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1200))
    plotter.set_background("white")
    plotter.add_mesh(
        curtain,
        scalars="log10res",
        cmap=RESISTIVITY_CMAP,
        clim=[0.0, 3.0],
        show_edges=False,
        lighting=True,
        ambient=0.28,
        diffuse=0.86,
        specular=0.03,
        scalar_bar_args={
            "title": "Log10 Resistivity (ohm-m)",
            "n_labels": 4,
            "width": 0.07,
            "height": 0.55,
            "position_x": 0.88,
            "position_y": 0.18,
            "color": "black",
            "fmt": "%.1f",
        },
    )
    plotter.add_mesh(
        top_skin,
        scalars="elev",
        cmap=TERRAIN_CMAP,
        show_edges=False,
        lighting=True,
        ambient=0.40,
        diffuse=0.82,
        specular=0.02,
        show_scalar_bar=False,
    )
    plotter.add_mesh(outline, color="black", line_width=1.2)

    plotter.enable_parallel_projection()
    plotter.view_isometric()
    plotter.reset_camera()
    plotter.camera.azimuth += 35
    plotter.camera.elevation += 12
    plotter.camera.zoom(1.5)
    plotter.show(screenshot=str(outputs["png"]))

    save_png_as_pdf(outputs["png"], outputs["pdf"])
    return outputs


def main() -> None:
    outputs = plot_block()
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
