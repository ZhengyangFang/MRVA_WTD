from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def surface_y(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=float)
    y = (
        11.2
        + 6.5 * np.exp(-((x_arr - 10.0) / 12.0) ** 2)
        + 2.0 * np.sin((x_arr - 8.0) / 10.0)
        - 2.6 * np.exp(-((x_arr - 47.0) / 10.0) ** 2)
        + 2.8 * np.exp(-((x_arr - 77.0) / 11.0) ** 2)
    )
    if np.isscalar(x):
        return float(y)
    return y


def lens_polygon(
    x0: float,
    x1: float,
    center_y: float,
    thickness: float,
    *,
    n: int = 160,
    top_wiggle: float = 0.8,
    bottom_wiggle: float = 0.6,
    phase: float = 0.0,
) -> np.ndarray:
    x = np.linspace(x0, x1, n)
    t = np.sin(np.pi * (x - x0) / (x1 - x0))
    top = center_y + 0.5 * thickness * t + top_wiggle * np.sin(x / 5.2 + phase)
    bottom = center_y - 0.5 * thickness * t + bottom_wiggle * np.sin(x / 6.7 + phase + 1.6)
    return np.vstack([np.column_stack([x, top]), np.column_stack([x[::-1], bottom[::-1]])])


def add_lens(ax, x0, x1, center_y, thickness, face, edge, alpha=0.92, hatch=None, z=2):
    poly = lens_polygon(x0, x1, center_y, thickness)
    patch = Polygon(poly, closed=True, facecolor=face, edgecolor=edge, linewidth=0.8, alpha=alpha, hatch=hatch, zorder=z)
    ax.add_patch(patch)
    return patch


def add_arrow(ax, xy0, xy1, color, lw=1.3, ms=10, alpha=1.0, z=8, style="-|>"):
    arr = FancyArrowPatch(
        xy0,
        xy1,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        alpha=alpha,
        zorder=z,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arr)
    return arr


def add_text(ax, x, y, text, *, color="#222222", size=7.0, weight="normal", ha="center", va="center", z=20, box=False):
    bbox = None
    if box:
        bbox = {
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.72,
        }
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=size,
        fontweight=weight,
        ha=ha,
        va=va,
        zorder=z,
        bbox=bbox,
    )


def add_cloud(ax, x, y, scale=1.0, color="#f8fbff", edge="#6aaed6"):
    circles = [
        (x - 2.2 * scale, y, 1.7 * scale),
        (x, y + 0.9 * scale, 2.2 * scale),
        (x + 2.5 * scale, y, 1.8 * scale),
        (x + 0.9 * scale, y - 0.5 * scale, 2.0 * scale),
    ]
    for cx, cy, r in circles:
        ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor=edge, linewidth=0.7, zorder=12))
    ax.add_patch(Rectangle((x - 4.1 * scale, y - 1.4 * scale), 8.4 * scale, 1.7 * scale, facecolor=color, edgecolor="none", zorder=13))
    ax.plot([x - 4.0 * scale, x + 4.1 * scale], [y - 0.45 * scale, y - 0.45 * scale], color=edge, linewidth=0.7, zorder=14)


def add_sun(ax, x, y, r=2.0):
    ax.add_patch(Circle((x, y), r, facecolor="#f5bf4f", edgecolor="#b97822", linewidth=0.8, zorder=12))
    for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        x0 = x + np.cos(ang) * (r + 0.5)
        y0 = y + np.sin(ang) * (r + 0.5)
        x1 = x + np.cos(ang) * (r + 1.5)
        y1 = y + np.sin(ang) * (r + 1.5)
        ax.plot([x0, x1], [y0, y1], color="#b97822", linewidth=0.7, zorder=12)


def add_pump_well(ax, x=59.0):
    sy = surface_y(x)
    ax.plot([x, x], [sy + 1.0, -58], color="#4a3b34", linewidth=1.4, zorder=10)
    ax.add_patch(Rectangle((x - 0.85, sy + 0.2), 1.7, 1.1, facecolor="#5b5048", edgecolor="#2e2925", linewidth=0.7, zorder=11))
    for yy in np.linspace(-38, -55, 4):
        ax.plot([x - 0.9, x + 0.9], [yy, yy], color="#4a3b34", linewidth=0.7, zorder=11)
    # Pumping and irrigation arrows.
    add_arrow(ax, (x, -14), (x, sy + 7.5), "#9a5a24", lw=1.6, ms=12, z=12)
    add_arrow(ax, (x + 2.2, sy + 7.0), (x + 8.4, sy + 7.0), "#9a5a24", lw=1.2, ms=9, z=12)
    add_arrow(ax, (x + 7.2, sy + 6.7), (x + 7.2, sy + 1.0), "#9a5a24", lw=0.9, ms=8, alpha=0.85, z=12)
    add_text(ax, x + 9.0, sy + 3.8, "Irrigation\npumping", color="#8a4d1f", size=6.5, ha="left")


def add_farmland(ax):
    for i, x0 in enumerate([16, 23.5, 31, 38.5, 46, 64]):
        sy0 = surface_y(x0)
        w = 5.7
        poly = np.array(
            [
                [x0, sy0 + 0.1],
                [x0 + w, surface_y(x0 + w) + 0.1],
                [x0 + w - 0.7, surface_y(x0 + w) + 2.3],
                [x0 + 0.5, sy0 + 2.3],
            ]
        )
        color = "#b7d17b" if i % 2 == 0 else "#d6c36f"
        ax.add_patch(Polygon(poly, facecolor=color, edgecolor="#9aa75b", linewidth=0.45, alpha=0.9, zorder=8))


def make_figure(output_prefix: Path, dpi: int = 600) -> dict[str, Path]:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.8,
        }
    )

    colors = {
        "sky": "#fbfdff",
        "soil": "#b78f5d",
        "fine": "#8ea4b2",
        "fine_light": "#b8c8cf",
        "sand": "#f2c166",
        "sand_edge": "#bc812e",
        "mixed": "#8fbf8c",
        "wet": "#1f78b4",
        "dry": "#d27c2c",
        "flow": "#2878b8",
        "pump": "#9a5a24",
    }

    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=170)
    ax.set_xlim(0, 100)
    ax.set_ylim(-78, 34)
    ax.set_facecolor(colors["sky"])
    ax.axis("off")

    x = np.linspace(0, 100, 420)
    sy = surface_y(x)

    # Sky and land/subsurface domains.
    ax.fill_between(x, sy, 34, color=colors["sky"], zorder=0)
    ax.fill_between(x, -78, sy, color="#d8e0df", zorder=0)
    ax.fill_between(x, sy - 5.5, sy, color="#8d6a46", zorder=5)
    ax.plot(x, sy, color="#2d2d2d", linewidth=1.0, zorder=15)

    # Mississippi River.
    river_x = np.array([86.5, 91.5, 96.5])
    river_y = surface_y(river_x) + np.array([0.5, 0.1, 0.4])
    river_poly = np.array([[85.5, surface_y(85.5) + 0.1], [97.5, surface_y(97.5) + 0.1], [96.3, -0.5], [88.3, -0.9]])
    ax.add_patch(Polygon(river_poly, facecolor="#5da9d6", edgecolor="#2a6f9b", linewidth=0.8, alpha=0.9, zorder=9))
    add_text(ax, 93.0, 7.0, "Mississippi\nRiver", color="#1f5f8b", size=6.0, ha="left", box=True)

    add_farmland(ax)

    # Depositional architecture.
    ax.add_patch(Rectangle((0, -78), 100, 32, facecolor="#91aab5", alpha=0.42, edgecolor="none", zorder=1))
    add_lens(ax, 5, 45, -17, 16, colors["sand"], colors["sand_edge"], alpha=0.95, z=3)
    add_lens(ax, 18, 72, -47, 18, colors["sand"], colors["sand_edge"], alpha=0.88, z=3)
    add_lens(ax, 68, 98, -27, 20, "#efc45e", colors["sand_edge"], alpha=0.92, z=3)
    add_lens(ax, 28, 70, -20, 19, colors["fine_light"], "#738a96", alpha=0.78, hatch="///", z=4)
    add_lens(ax, 52, 92, -55, 16, colors["fine"], "#667d88", alpha=0.72, hatch="///", z=4)
    add_lens(ax, 10, 35, -37, 11, colors["mixed"], "#648b61", alpha=0.72, z=5)
    add_lens(ax, 72, 92, -12, 9, colors["mixed"], "#648b61", alpha=0.65, z=5)

    # Subtle depth bands.
    for yy in [-20, -40, -60]:
        ax.plot([0, 100], [yy, yy], color="white", linewidth=0.35, alpha=0.25, zorder=2)

    # Water-table states.
    wet = surface_y(x) - (9.5 + 1.8 * np.sin((x - 5) / 16) - 2.0 * np.exp(-((x - 90) / 14) ** 2))
    drawdown = 5.0 + 11.0 * np.exp(-((x - 59) / 11) ** 2) + 6.0 * np.exp(-((x - 42) / 14) ** 2)
    drought = wet - drawdown
    ax.plot(x, wet, color=colors["wet"], linewidth=2.2, zorder=13)
    ax.plot(x, drought, color=colors["dry"], linewidth=1.8, linestyle=(0, (5, 3)), zorder=13)

    # Shade drawdown magnitude.
    ax.fill_between(x, drought, wet, where=(x > 34) & (x < 73), color="#d27c2c", alpha=0.12, zorder=12)

    # Pumping well.
    add_pump_well(ax, 59)

    # Climate and recharge.
    add_cloud(ax, 18, 28, scale=0.95)
    for xx in [13.5, 17, 20.5, 24]:
        add_arrow(ax, (xx, 24.3), (xx, surface_y(xx) + 2.2), colors["wet"], lw=0.9, ms=7, alpha=0.85, z=12)
    add_text(ax, 19, 31.6, "Climate input", color="#2f77a8", size=7.4, weight="bold")
    add_text(ax, 11.2, 20.5, "Precipitation /\nfloodwater", color="#2f77a8", size=6.2, ha="left", box=True)

    add_sun(ax, 75, 28, r=1.8)
    add_text(ax, 80.0, 24.2, "Drought", color="#9a5a24", size=6.5, ha="left", box=True)

    for xx in [20, 31, 82]:
        add_arrow(ax, (xx, surface_y(xx) - 1.5), (xx, wet[np.argmin(np.abs(x - xx))] + 1.0), colors["wet"], lw=1.0, ms=8, alpha=0.9)
    add_text(ax, 27.5, surface_y(27) - 5.8, "Net\ninfiltration", color="#2f77a8", size=6.0, ha="right", box=True)

    # Blocked recharge over fine-grained unit.
    add_arrow(ax, (47, surface_y(47) - 2), (47, -10.5), colors["wet"], lw=0.9, ms=7, alpha=0.75)
    ax.plot([43.8, 50.2], [-11.7, -11.7], color="#596d75", linewidth=1.6, zorder=14)
    add_text(ax, 52.2, -11.2, "limited\nvertical flow", color="#596d75", size=5.8, ha="left", box=True)

    # Lateral flow arrows inside connected sands.
    add_arrow(ax, (17, -18), (35, -17), colors["flow"], lw=1.2, ms=9, alpha=0.85)
    add_arrow(ax, (33, -47), (55, -46), colors["flow"], lw=1.2, ms=9, alpha=0.85)
    add_arrow(ax, (77, -29), (92, -26), colors["flow"], lw=1.1, ms=8, alpha=0.85)
    add_text(ax, 38, -53, "Lateral flow through\nconnected sand bodies", color="#1f5f8b", size=6.4)

    # Key labels.
    ax.text(11, -5, "Connected sand", color="#875a1c", fontsize=6.5, ha="left", va="center", zorder=18)
    ax.text(51, -24, "Fine-grained\nconfining unit", color="#4f6470", fontsize=6.3, ha="center", va="center", zorder=18)
    ax.text(70, -64, "AEM-resolved sediment architecture", color="#4f6470", fontsize=7.2, ha="center", va="center", zorder=18)
    ax.text(5, wet[np.argmin(np.abs(x - 5))] + 2.2, "Wet/recharged\nwater table", color=colors["wet"], fontsize=6.2, ha="left", va="bottom", zorder=18)
    ax.text(63, drought[np.argmin(np.abs(x - 63))] - 3.0, "Drought drawdown", color=colors["dry"], fontsize=6.4, ha="left", va="top", zorder=18)

    # Small bracket for response outcome.
    ax.plot([4, 96], [-72, -72], color="#727272", linewidth=0.7, zorder=18)
    add_arrow(ax, (42, -72), (42, -66), "#727272", lw=0.7, ms=6)
    add_arrow(ax, (60, -72), (60, -66), "#727272", lw=0.7, ms=6)
    add_text(ax, 50, -75.0, "Spatially heterogeneous groundwater-table response", color="#3b3b3b", size=7.2)

    add_text(ax, 1.0, 31.3, "b", color="#111111", size=10.5, weight="bold", ha="left")
    add_text(ax, 98.5, -76.0, "conceptual; not to scale", color="#666666", size=5.6, ha="right")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": output_prefix.with_suffix(".png"),
        "pdf": output_prefix.with_suffix(".pdf"),
        "svg": output_prefix.with_suffix(".svg"),
    }
    fig.savefig(outputs["png"], dpi=dpi, bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    root = repo_root()
    return argparse.Namespace(
        output_prefix=root / "outputs" / "figures" / "Fig1b_conceptual_framework" / "Fig1b_conceptual_framework",
        dpi=600,
    )


def main() -> None:
    args = parse_args()
    outputs = make_figure(Path(args.output_prefix), dpi=int(args.dpi))
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
