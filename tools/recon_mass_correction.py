from __future__ import annotations

import shutil
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]

WTD_MONTHLY_PATH = ROOT / "data" / "2 well WTD" / "2011_2023" / "wtd_monthly.csv"
GRACE_PATH = ROOT / "data" / "7 GRACE" / "grace_mrva_2011_2023_timeseries_full156.csv"
CONNECTIVITY_TIF_PATH = ROOT / "data" / "8 connectivity" / "ConfiningLayer_SurfaceConnectivity.tif"

SY_CLASS_MAP = {-3: 0.08, -2: 0.11, -1: 0.14, 1: 0.18, 2: 0.22, 3: 0.26}
SY_DEFAULT = 0.18


def display_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def replace_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        tmp_path.unlink(missing_ok=True)
        np.save(tmp_path, array)
        tmp_path.replace(path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise OSError(
            f"Could not replace {display_path(path)}. Close any notebook, viewer, GIS, or Python process "
            "that may be using this file, then rerun the reconstruction."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply well-derived regional storage-mass correction to a reconstruction."
    )
    parser.add_argument(
        "source_recon_root",
        help="Source reconstruction root produced by tools/run_recon.py.",
    )
    parser.add_argument(
        "out_recon_root",
        help="Output reconstruction root after regional correction.",
    )
    return parser.parse_args()


def load_month_index(recon_root: Path) -> pd.DataFrame:
    month_index = pd.read_csv(recon_root / "metadata" / "month_index.csv")
    month_index["month_label"] = month_index["month_label"].astype(str)
    month_index["date"] = pd.to_datetime(month_index["month_label"] + "-01")
    return month_index


def sample_specific_yield(grid_lookup: pd.DataFrame) -> np.ndarray:
    coords = list(zip(grid_lookup["x"].to_numpy(), grid_lookup["y"].to_numpy()))
    with rasterio.open(CONNECTIVITY_TIF_PATH) as src:
        classes = np.array([value[0] for value in src.sample(coords)], dtype=np.float32)
        if src.nodata is not None:
            classes[classes == src.nodata] = np.nan

    sy = np.full(len(grid_lookup), SY_DEFAULT, dtype=np.float32)
    for cls, value in SY_CLASS_MAP.items():
        sy[classes == cls] = value
    return sy


def load_grace(month_index: pd.DataFrame) -> pd.DataFrame:
    grace = pd.read_csv(GRACE_PATH)
    grace = grace.rename(columns={"grace_lwe_thickness_cm": "grace_lwe_cm"})
    grace["month_label"] = grace["month"].astype(str) if "month" in grace.columns else grace["month_label"].astype(str)
    grace = month_index[["month_label"]].merge(
        grace[["month_label", "has_grace_solution", "grace_lwe_cm"]],
        on="month_label",
        how="left",
    )
    valid = grace["has_grace_solution"].astype(str).str.lower().isin(["true", "1", "yes"])
    valid = valid & grace["grace_lwe_cm"].notna()
    grace["has_valid_grace"] = valid.to_numpy()
    grace["grace_anom_cm"] = grace["grace_lwe_cm"] - float(grace.loc[valid, "grace_lwe_cm"].mean())
    return grace


def compute_fig2_storage_proxy_cm(
    matrix_m_bls: np.ndarray,
    sy_by_grid: np.ndarray,
    valid_reference_months: np.ndarray,
) -> np.ndarray:
    reference_wtd = np.nanmean(matrix_m_bls[valid_reference_months, :], axis=0)
    delta_wtd = matrix_m_bls - reference_wtd[None, :]
    return (-100.0 * np.nanmean(delta_wtd * sy_by_grid[None, :], axis=1)).astype(np.float32)


def build_observation_mask(
    *,
    month_index: pd.DataFrame,
    grid_lookup: pd.DataFrame,
) -> np.ndarray:
    n_month = len(month_index)
    n_grid = len(grid_lookup)
    label_to_idx = {label: idx for idx, label in enumerate(month_index["month_label"].astype(str))}
    cell_to_idx = {int(cell_id): idx for idx, cell_id in enumerate(grid_lookup["cell_id"].astype(int))}

    obs = pd.read_csv(WTD_MONTHLY_PATH, usecols=["cell_id", "month_label"])
    obs = obs[obs["month_label"].astype(str).isin(label_to_idx)]
    obs = obs[obs["cell_id"].astype(int).isin(cell_to_idx)]
    obs["month_idx"] = obs["month_label"].astype(str).map(label_to_idx).astype(int)
    obs["grid_idx"] = obs["cell_id"].astype(int).map(cell_to_idx).astype(int)

    mask = np.zeros((n_month, n_grid), dtype=bool)
    mask[obs["month_idx"].to_numpy(dtype=int), obs["grid_idx"].to_numpy(dtype=int)] = True
    return mask


def build_well_storage_target_cm(
    *,
    month_index: pd.DataFrame,
    grid_lookup: pd.DataFrame,
    baseline_wtd_m: np.ndarray,
    sy_by_grid: np.ndarray,
    valid_reference_months: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    label_to_idx = {label: idx for idx, label in enumerate(month_index["month_label"].astype(str))}
    cell_to_idx = {int(cell_id): idx for idx, cell_id in enumerate(grid_lookup["cell_id"].astype(int))}

    obs = pd.read_csv(WTD_MONTHLY_PATH, usecols=["cell_id", "month_label", "mean_wtd_m"])
    obs = obs[obs["month_label"].astype(str).isin(label_to_idx)]
    obs = obs[obs["cell_id"].astype(int).isin(cell_to_idx)]
    obs["month_idx"] = obs["month_label"].astype(str).map(label_to_idx).astype(int)
    obs["grid_idx"] = obs["cell_id"].astype(int).map(cell_to_idx).astype(int)

    grid_idx = obs["grid_idx"].to_numpy(dtype=int)
    obs["anomaly_m"] = obs["mean_wtd_m"].to_numpy(dtype=np.float32) - baseline_wtd_m[grid_idx]
    grid_median_anom = obs.groupby("grid_idx")["anomaly_m"].median()
    obs["demeaned_anomaly_m"] = obs["anomaly_m"] - obs["grid_idx"].map(grid_median_anom)
    obs["sy_weighted_demeaned_anomaly_m"] = obs["demeaned_anomaly_m"] * sy_by_grid[grid_idx]

    records: list[dict[str, object]] = []
    for month_idx, month_label in enumerate(month_index["month_label"].astype(str)):
        sub = obs[obs["month_idx"] == month_idx]
        if len(sub) == 0:
            records.append(
                {
                    "month_idx": month_idx,
                    "month_label": month_label,
                    "well_observation_count": 0,
                    "well_sy_weighted_demeaned_mean_m": np.nan,
                }
            )
        else:
            records.append(
                {
                    "month_idx": month_idx,
                    "month_label": month_label,
                    "well_observation_count": int(len(sub)),
                    "well_sy_weighted_demeaned_mean_m": float(sub["sy_weighted_demeaned_anomaly_m"].mean()),
                }
            )

    target = pd.DataFrame(records)
    filled = (
        pd.Series(target["well_sy_weighted_demeaned_mean_m"].to_numpy(dtype=np.float64))
        .interpolate(method="linear", limit_direction="both")
        .rolling(window=3, center=True, min_periods=1)
        .mean()
    )
    raw_storage_cm = -100.0 * filled.to_numpy(dtype=np.float64)
    raw_storage_cm = raw_storage_cm - float(np.nanmean(raw_storage_cm[valid_reference_months]))
    target["well_storage_proxy_target_cm"] = raw_storage_cm.astype(np.float32)
    return target["well_storage_proxy_target_cm"].to_numpy(dtype=np.float32), target


def base_correction_weights(recon_root: Path, n_grid: int) -> np.ndarray:
    summary_path = recon_root / "metadata" / "selective_anchor_grid_summary.csv"
    if not summary_path.exists():
        return np.ones(n_grid, dtype=np.float32)

    summary = pd.read_csv(summary_path)
    if "local_support_score" not in summary.columns:
        return np.ones(n_grid, dtype=np.float32)

    support = summary["local_support_score"].to_numpy(dtype=np.float32)
    if support.shape != (n_grid,):
        return np.ones(n_grid, dtype=np.float32)
    return np.clip(1.0 - support, 0.15, 1.0).astype(np.float32)


def prepare_output_tree(source_root: Path, out_root: Path) -> None:
    (out_root / "metadata").mkdir(parents=True, exist_ok=True)
    (out_root / "reconstruction").mkdir(parents=True, exist_ok=True)
    (out_root / "diagnostics" / "regional_mass_correction").mkdir(parents=True, exist_ok=True)

    for item in (source_root / "metadata").iterdir():
        if item.is_file():
            target = out_root / "metadata" / item.name
            if item.resolve() != target.resolve():
                shutil.copy2(item, target)


def correlation_row(name: str, values: np.ndarray, grace_anom_cm: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    mask = valid & np.isfinite(values) & np.isfinite(grace_anom_cm)
    if int(mask.sum()) < 3:
        return {"series": name, "n": int(mask.sum()), "pearson_r": np.nan, "spearman_rho": np.nan}
    return {
        "series": name,
        "n": int(mask.sum()),
        "pearson_r": float(pearsonr(values[mask], grace_anom_cm[mask]).statistic),
        "spearman_rho": float(spearmanr(values[mask], grace_anom_cm[mask]).statistic),
    }


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_recon_root).resolve()
    out_root = Path(args.out_recon_root).resolve()

    month_index = load_month_index(source_root)
    grid_lookup = pd.read_csv(source_root / "metadata" / "grid_lookup.csv")
    baseline_wtd_m = pd.read_csv(source_root / "metadata" / "longterm_baseline_on_grid.csv")[
        "longterm_baseline_wtd_m_bls"
    ].to_numpy(dtype=np.float32)
    matrix = np.asarray(np.load(source_root / "reconstruction" / "wtd_reconstructed_matrix.npy"), dtype=np.float32)

    grace = load_grace(month_index)
    valid_grace = grace["has_valid_grace"].to_numpy(dtype=bool)
    sy_by_grid = sample_specific_yield(grid_lookup)
    source_storage_cm = compute_fig2_storage_proxy_cm(matrix, sy_by_grid, valid_grace)

    target_storage_cm, target_table = build_well_storage_target_cm(
        month_index=month_index,
        grid_lookup=grid_lookup,
        baseline_wtd_m=baseline_wtd_m,
        sy_by_grid=sy_by_grid,
        valid_reference_months=valid_grace,
    )

    obs_mask = build_observation_mask(month_index=month_index, grid_lookup=grid_lookup)
    base_weights = base_correction_weights(source_root, len(grid_lookup))

    corrected = matrix.copy()
    correction_rows: list[dict[str, object]] = []
    for month_idx, month_label in enumerate(month_index["month_label"].astype(str)):
        diff_cm = float(target_storage_cm[month_idx] - source_storage_cm[month_idx])
        weights = base_weights.copy()
        weights[obs_mask[month_idx]] = 0.0
        weighted_mean_sy = float(np.nanmean(weights * sy_by_grid))
        if weighted_mean_sy <= 1.0e-6:
            applied = np.zeros_like(weights, dtype=np.float32)
            uniform_shift_m = 0.0
        else:
            uniform_shift_m = -diff_cm / (100.0 * weighted_mean_sy)
            applied = (uniform_shift_m * weights).astype(np.float32)
            corrected[month_idx] = corrected[month_idx] + applied

        correction_rows.append(
            {
                "month_idx": int(month_idx),
                "month_label": str(month_label),
                "source_storage_proxy_cm": float(source_storage_cm[month_idx]),
                "well_storage_proxy_target_cm": float(target_storage_cm[month_idx]),
                "target_minus_source_cm": diff_cm,
                "weighted_mean_sy": weighted_mean_sy,
                "uniform_equivalent_shift_m": float(uniform_shift_m),
                "applied_shift_min_m": float(np.nanmin(applied)),
                "applied_shift_p50_m": float(np.nanmedian(applied)),
                "applied_shift_max_m": float(np.nanmax(applied)),
                "observed_cells_preserved": int(obs_mask[month_idx].sum()),
            }
        )

    corrected_storage_cm = compute_fig2_storage_proxy_cm(corrected, sy_by_grid, valid_grace)

    prepare_output_tree(source_root, out_root)
    replace_npy(out_root / "reconstruction" / "wtd_reconstructed_matrix.npy", corrected.astype(np.float32))

    diag_dir = out_root / "diagnostics" / "regional_mass_correction"
    correction_table = pd.DataFrame(correction_rows)
    correction_table["corrected_storage_proxy_cm"] = corrected_storage_cm
    correction_table["grace_anom_cm"] = grace["grace_anom_cm"].to_numpy(dtype=np.float32)
    correction_table["has_valid_grace"] = valid_grace
    correction_table = correction_table.merge(
        target_table[["month_label", "well_observation_count", "well_sy_weighted_demeaned_mean_m"]],
        on="month_label",
        how="left",
    )
    correction_table.to_csv(diag_dir / "regional_mass_correction_monthly.csv", index=False)

    metrics = pd.DataFrame(
        [
            correlation_row("raw_reconstruction", source_storage_cm, grace["grace_anom_cm"].to_numpy(dtype=np.float32), valid_grace),
            correlation_row("well_target", target_storage_cm, grace["grace_anom_cm"].to_numpy(dtype=np.float32), valid_grace),
            correlation_row(
                "corrected_reconstruction",
                corrected_storage_cm,
                grace["grace_anom_cm"].to_numpy(dtype=np.float32),
                valid_grace,
            ),
        ]
    )
    metrics.to_csv(diag_dir / "regional_mass_correction_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.0, 3.4), dpi=300)
    ax.plot(month_index["date"], source_storage_cm, color="#999999", lw=1.1, label="Raw reconstruction")
    ax.plot(month_index["date"], corrected_storage_cm, color="#1b1b1b", lw=1.3, label="Corrected reconstruction")
    ax.plot(month_index["date"], target_storage_cm, color="#2a9d8f", lw=1.0, label="well-derived target")
    ax.scatter(
        month_index.loc[valid_grace, "date"],
        grace.loc[valid_grace, "grace_anom_cm"],
        s=12,
        facecolor="white",
        edgecolor="#4c78a8",
        lw=0.7,
        label="GRACE",
        zorder=3,
    )
    ax.axhline(0.0, color="#333333", lw=0.5)
    ax.set_ylabel("Storage anomaly (cm)")
    ax.set_xlabel("Year")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", lw=0.5)
    fig.tight_layout()
    fig.savefig(diag_dir / "regional_mass_correction_vs_grace.png", dpi=300)
    plt.close(fig)

    print("Regional mass-corrected reconstruction written.")
    print(f"source_root: {display_path(source_root)}")
    print(f"out_root: {display_path(out_root)}")
    print(metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
