from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mainline_reconstruction import MAINLINE_RECON_ROOT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECON_ROOT = MAINLINE_RECON_ROOT
DEFAULT_GRACE_PATH = ROOT / "data" / "7 GRACE" / "grace_mrva_2011_2023_timeseries_full156.csv"

# Storage-layer settings.
DEFAULT_LAYER_THICKNESS_M = [200.0, 100.0, 50.0, 25.0, 10.0, 5.0, 1.0, 0.6, 0.3, 0.1]
DEFAULT_TSD_M = float(sum(DEFAULT_LAYER_THICKNESS_M))  # 392.0
DEFAULT_POROSITY = 0.16  # Default porosity.


def _parse_layers(text: str | None) -> list[float]:
    if text is None or str(text).strip() == "":
        return [float(v) for v in DEFAULT_LAYER_THICKNESS_M]
    vals = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not vals:
        raise ValueError("Parsed empty layer thickness list.")
    return vals


def _infer_cell_size_m(grid_lookup: pd.DataFrame) -> float:
    x = np.sort(grid_lookup["x"].dropna().unique())
    y = np.sort(grid_lookup["y"].dropna().unique())
    dx = np.median(np.diff(x)) if len(x) > 1 else 1000.0
    dy = np.median(np.diff(y)) if len(y) > 1 else 1000.0
    return float(np.mean([abs(dx), abs(dy)]))


def _load_porosity_layers(
    *,
    n_grid: int,
    n_layers: int,
    porosity_path: Path | None,
    fallback_porosity: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load porosity values as a layer-by-grid array."""
    if porosity_path is None:
        arr = np.full((n_layers, n_grid), float(fallback_porosity), dtype=np.float32)
        meta = {
            "porosity_source": "fallback_constant",
            "fallback_porosity": float(fallback_porosity),
            "note": "No porosity file provided. Replace with 10-layer porosity to fully match paper setup.",
        }
        return arr, meta

    if not porosity_path.exists():
        raise FileNotFoundError(f"Porosity file not found: {porosity_path}")

    if porosity_path.suffix.lower() == ".npy":
        raw = np.load(porosity_path)
        if raw.ndim == 1:
            if raw.shape[0] == n_grid:
                arr = np.tile(raw.reshape(1, n_grid), (n_layers, 1))
            elif raw.shape[0] == n_layers:
                arr = np.tile(raw.reshape(n_layers, 1), (1, n_grid))
            else:
                raise ValueError(f"Unsupported 1D porosity shape {raw.shape}, expected n_grid or n_layers.")
        elif raw.ndim == 2:
            if raw.shape == (n_layers, n_grid):
                arr = raw
            elif raw.shape == (n_grid, n_layers):
                arr = raw.T
            else:
                raise ValueError(
                    f"Unsupported 2D porosity shape {raw.shape}, expected {(n_layers, n_grid)} or {(n_grid, n_layers)}."
                )
        else:
            raise ValueError(f"Unsupported porosity ndim={raw.ndim}, expected 1D/2D.")
        arr = np.asarray(arr, dtype=np.float32)
        meta = {
            "porosity_source": str(porosity_path),
            "porosity_format": "npy",
            "shape_loaded": list(raw.shape),
            "shape_used": [int(arr.shape[0]), int(arr.shape[1])],
        }
        return arr, meta

    if porosity_path.suffix.lower() == ".csv":
        frame = pd.read_csv(porosity_path)
        if "grid_id" not in frame.columns:
            raise ValueError("Porosity CSV must contain 'grid_id' column.")
        frame["grid_id"] = frame["grid_id"].astype(int)
        cols_multi = [c for c in frame.columns if c.lower().startswith("porosity_l")]
        cols_multi = sorted(cols_multi, key=lambda c: int("".join(ch for ch in c if ch.isdigit()) or "0"))
        if len(cols_multi) >= n_layers:
            cols_use = cols_multi[:n_layers]
            frame2 = frame.set_index("grid_id")[cols_use].sort_index()
            if len(frame2) < n_grid:
                raise ValueError(f"Porosity CSV has {len(frame2)} rows, expected at least n_grid={n_grid}.")
            # Align porosity by grid identifier.
            frame2 = frame2.reindex(np.arange(n_grid))
            if frame2.isna().any().any():
                raise ValueError("Porosity CSV has missing grid_id rows or missing porosity values.")
            arr = frame2.to_numpy(dtype=np.float32).T
            meta = {
                "porosity_source": str(porosity_path),
                "porosity_format": "csv_multi_layer",
                "porosity_columns_used": cols_use,
                "shape_used": [int(arr.shape[0]), int(arr.shape[1])],
            }
            return arr, meta

        if "porosity" in frame.columns:
            v = (
                frame.set_index("grid_id")["porosity"]
                .sort_index()
                .reindex(np.arange(n_grid))
                .to_numpy(dtype=np.float32)
            )
            if np.isnan(v).any():
                raise ValueError("Porosity CSV has missing values after reindex by grid_id.")
            arr = np.tile(v.reshape(1, n_grid), (n_layers, 1))
            meta = {
                "porosity_source": str(porosity_path),
                "porosity_format": "csv_single_layer",
                "shape_used": [int(arr.shape[0]), int(arr.shape[1])],
            }
            return arr, meta

        raise ValueError(
            "Porosity CSV must contain either 'porosity' or multi-layer columns like porosity_l1..porosity_l10."
        )

    raise ValueError(f"Unsupported porosity file suffix: {porosity_path.suffix}")


def _compute_storage_equiv_depth_matrix(
    *,
    wtd_matrix_m_bls: np.ndarray,
    porosity_layers: np.ndarray,
    layer_thickness_m: list[float],
    tsd_m: float,
) -> np.ndarray:
    """Compute equivalent saturated depth for each month and grid cell."""
    n_month, n_grid = wtd_matrix_m_bls.shape
    n_layers = len(layer_thickness_m)
    if porosity_layers.shape != (n_layers, n_grid):
        raise ValueError(
            f"porosity_layers shape {porosity_layers.shape} does not match expected {(n_layers, n_grid)}."
        )

    d_sat = np.clip(float(tsd_m) - np.asarray(wtd_matrix_m_bls, dtype=np.float32), 0.0, float(tsd_m))
    out = np.zeros_like(d_sat, dtype=np.float32)
    lower = 0.0
    for li, thick in enumerate(layer_thickness_m):
        h = np.clip(d_sat - lower, 0.0, float(thick))
        out += h * porosity_layers[li].reshape(1, n_grid)
        lower += float(thick)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute groundwater storage using paper-style layered formulation.")
    parser.add_argument("recon_root", nargs="?", default=str(DEFAULT_RECON_ROOT))
    parser.add_argument("--grace-path", default=str(DEFAULT_GRACE_PATH))
    parser.add_argument("--porosity-path", default="", help="Optional porosity file (.npy/.csv).")
    parser.add_argument("--fallback-porosity", type=float, default=DEFAULT_POROSITY)
    parser.add_argument("--layers", default="", help="Comma-separated layer thickness list in meters.")
    parser.add_argument("--tsd", type=float, default=DEFAULT_TSD_M)
    args = parser.parse_args()

    recon_root = Path(args.recon_root).resolve()
    grace_path = Path(args.grace_path).resolve()
    porosity_path = Path(args.porosity_path).resolve() if str(args.porosity_path).strip() else None
    layer_thickness_m = _parse_layers(args.layers)

    matrix_path = recon_root / "reconstruction" / "wtd_reconstructed_matrix.npy"
    month_path = recon_root / "metadata" / "month_index.csv"
    grid_path = recon_root / "metadata" / "grid_lookup.csv"
    unc_path = recon_root / "model_uncertainty" / "monthly_model_uncertainty_radius_matrix.npy"

    wtd = np.asarray(np.load(matrix_path, mmap_mode="r"), dtype=np.float32)
    month_index = pd.read_csv(month_path)
    month_col = "month_label" if "month_label" in month_index.columns else "month"
    months = month_index[month_col].astype(str).tolist()
    grid_lookup = pd.read_csv(grid_path, usecols=["grid_id", "x", "y"])
    n_month, n_grid = wtd.shape

    if len(months) != n_month:
        raise ValueError(f"Month index length {len(months)} does not match matrix month dimension {n_month}.")

    cell_size_m = _infer_cell_size_m(grid_lookup)
    cell_area_m2 = float(cell_size_m) * float(cell_size_m)
    total_area_m2 = float(cell_area_m2) * float(n_grid)

    porosity_layers, porosity_meta = _load_porosity_layers(
        n_grid=n_grid,
        n_layers=len(layer_thickness_m),
        porosity_path=porosity_path,
        fallback_porosity=float(args.fallback_porosity),
    )

    eq_depth = _compute_storage_equiv_depth_matrix(
        wtd_matrix_m_bls=wtd,
        porosity_layers=porosity_layers,
        layer_thickness_m=layer_thickness_m,
        tsd_m=float(args.tsd),
    )
    storage_m3 = eq_depth.sum(axis=1, dtype=np.float64) * cell_area_m2
    storage_equiv_m = storage_m3 / total_area_m2

    out_dir = recon_root / "diagnostics" / "paper_storage"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame({
        "month_label": months,
        "storage_m3": storage_m3.astype(np.float64),
        "storage_equiv_m": storage_equiv_m.astype(np.float64),
    })

    out["month_date"] = pd.to_datetime(out["month_label"] + "-01")

    grace_found = grace_path.exists()
    if grace_found:
        grace = pd.read_csv(grace_path)
        grace = grace.rename(columns={"month": "month_label"})
        keep = ["month_label", "has_grace_solution", "grace_lwe_thickness_cm"]
        missing = [c for c in keep if c not in grace.columns]
        if missing:
            raise ValueError(f"GRACE file missing columns: {missing}")
        out = out.merge(grace[keep], on="month_label", how="left")
        grace_mask = out["has_grace_solution"].fillna(False).to_numpy(dtype=bool)
    else:
        out["has_grace_solution"] = False
        out["grace_lwe_thickness_cm"] = np.nan
        grace_mask = np.zeros(len(out), dtype=bool)

    if grace_mask.any():
        base_m = float(out.loc[grace_mask, "storage_equiv_m"].mean())
    else:
        base_m = float(out["storage_equiv_m"].mean())

    out["storage_anom_m"] = out["storage_equiv_m"] - base_m
    out["storage_anom_cm"] = 100.0 * out["storage_anom_m"]

    if grace_mask.any():
        out["grace_anom_cm"] = out["grace_lwe_thickness_cm"] - float(out.loc[grace_mask, "grace_lwe_thickness_cm"].mean())
    else:
        out["grace_anom_cm"] = np.nan

    if unc_path.exists():
        unc = np.asarray(np.load(unc_path, mmap_mode="r"), dtype=np.float32)
        if unc.shape == wtd.shape:
            # Propagate WTD uncertainty to storage.
            wtd_shallow = np.maximum(wtd - unc, 0.0)  # shallower water table -> larger storage
            wtd_deeper = wtd + unc                    # deeper water table -> smaller storage
            eq_hi = _compute_storage_equiv_depth_matrix(
                wtd_matrix_m_bls=wtd_shallow,
                porosity_layers=porosity_layers,
                layer_thickness_m=layer_thickness_m,
                tsd_m=float(args.tsd),
            )
            eq_lo = _compute_storage_equiv_depth_matrix(
                wtd_matrix_m_bls=wtd_deeper,
                porosity_layers=porosity_layers,
                layer_thickness_m=layer_thickness_m,
                tsd_m=float(args.tsd),
            )
            s_hi = eq_hi.sum(axis=1, dtype=np.float64) * cell_area_m2
            s_lo = eq_lo.sum(axis=1, dtype=np.float64) * cell_area_m2
            out["storage_upper_m3_approx"] = s_hi
            out["storage_lower_m3_approx"] = s_lo
            out["storage_upper_equiv_m_approx"] = s_hi / total_area_m2
            out["storage_lower_equiv_m_approx"] = s_lo / total_area_m2
            out["storage_pi95_radius_cm_approx"] = 50.0 * (
                out["storage_upper_equiv_m_approx"] - out["storage_lower_equiv_m_approx"]
            )

    out_csv = out_dir / "paper_storage_monthly.csv"
    out.to_csv(out_csv, index=False)

    # Draw the regional comparison.
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.5), sharex=False)

    ax = axes[0]
    ax.plot(out["month_date"], out["storage_equiv_m"], color="#1f77b4", linewidth=1.7, label="GW storage equivalent depth (m)")
    if "storage_upper_equiv_m_approx" in out.columns and "storage_lower_equiv_m_approx" in out.columns:
        ax.fill_between(
            out["month_date"],
            out["storage_lower_equiv_m_approx"],
            out["storage_upper_equiv_m_approx"],
            color="#1f77b4",
            alpha=0.18,
            linewidth=0.0,
            label="Approx. uncertainty envelope",
        )
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylabel("Storage Eq. Depth (m)")
    ax.legend(loc="upper right", frameon=False)

    ax2 = axes[1]
    ax2.plot(out["month_date"], out["storage_anom_cm"], color="#1f77b4", linewidth=1.7, label="GW storage anomaly (cm)")
    if grace_mask.any():
        ax2.plot(
            out.loc[grace_mask, "month_date"],
            out.loc[grace_mask, "grace_anom_cm"],
            color="#d62728",
            linewidth=1.3,
            marker="o",
            markersize=2.3,
            label="GRACE LWE anomaly (cm)",
        )
        r = float(out.loc[grace_mask, "storage_anom_cm"].corr(out.loc[grace_mask, "grace_anom_cm"]))
        rs, pval = spearmanr(
            out.loc[grace_mask, "storage_anom_cm"].to_numpy(dtype=float),
            out.loc[grace_mask, "grace_anom_cm"].to_numpy(dtype=float),
            nan_policy="omit",
        )
        txt = f"Pearson r={r:.2f}, Spearman r={float(rs):.2f} (p={float(pval):.2g}), N={int(grace_mask.sum())}"
        ax2.text(
            0.01,
            0.98,
            txt,
            transform=ax2.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
        )
    ax2.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.set_ylabel("Anomaly (cm)")
    ax2.set_xlabel("Time")
    ax2.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig_path = out_dir / "paper_storage_vs_grace.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "method": "paper_S1.3_layered_storage",
        "reference_paper": "Ma et al. 2026 supplementary Eq.S1-S2",
        "recon_root": str(recon_root),
        "n_month": int(n_month),
        "n_grid": int(n_grid),
        "cell_size_m_inferred": float(cell_size_m),
        "cell_area_m2_assumed": float(cell_area_m2),
        "total_area_m2": float(total_area_m2),
        "layer_thickness_m": [float(v) for v in layer_thickness_m],
        "tsd_m": float(args.tsd),
        "porosity_meta": porosity_meta,
        "grace_path": str(grace_path),
        "grace_overlap_months": int(grace_mask.sum()),
        "outputs": {
            "monthly_csv": str(out_csv),
            "figure_png": str(fig_path),
        },
    }
    meta_path = out_dir / "paper_storage_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Paper-style storage calculation done.")
    print(f"monthly_csv: {out_csv}")
    print(f"figure_png: {fig_path}")
    print(f"metadata_json: {meta_path}")
    if porosity_meta.get("porosity_source") == "fallback_constant":
        print("WARNING: using fallback constant porosity. Replace with 10-layer porosity to fully match paper setup.")


if __name__ == "__main__":
    main()
