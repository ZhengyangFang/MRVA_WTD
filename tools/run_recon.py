from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reconstruct import (  # noqa: E402
    BaselineAdjustedPredictor,
    SpecialistPredictor,
    build_observation_matrices,
    load_longterm_baseline_field,
    reconstruct_monthly_wtd_adaptive,
    save_reconstruction_outputs,
)


DEFAULT_OUT_DIR = ROOT / "outputs" / "RECON_MAIN_2011_2023"
WTD_MONTHLY_PATH = ROOT / "data" / "2 well WTD" / "2011_2023" / "wtd_monthly.csv"
LONGTERM_PATH = ROOT / "data" / "6 longterm_mean_WTD" / "longterm_wtd_1km.csv"
EXPECTED_SEEDS = ("seed11", "seed22", "seed33", "seed44", "seed55")
EXPECTED_JANUARY_ANCHOR_COUNT = 13
FINAL_ENDPOINT_LABEL = "2023-12"

ENSEMBLE_ROOTS = {
    1: ROOT / "outputs" / "GNN_H1",
    3: ROOT / "outputs" / "GNN_H3",
    6: ROOT / "outputs" / "GNN_H6",
}


def display_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mainline groundwater reconstruction.")
    parser.add_argument(
        "--out-dir",
        default="",
        help=f"Optional output directory. Default: {display_path(DEFAULT_OUT_DIR)}.",
    )
    return parser.parse_args()


def _tensor_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _validation_record_from_metrics(metrics_path: Path) -> dict[str, float]:
    frame = pd.read_csv(metrics_path)
    validation_rows = frame.loc[frame["split"] == "validation"]
    if validation_rows.empty:
        raise ValueError(f"Validation split not found in metrics file: {metrics_path}")
    val_row = validation_rows.iloc[0]
    return {
        "val_pearson_r": float(val_row["pearson_r"]) if "pearson_r" in frame.columns else float("nan"),
        "val_nse": float(val_row["nse"]) if "nse" in frame.columns else float(val_row["r2"]),
        "val_rmse": float(val_row["rmse"]),
    }


def all_seed_checkpoints(horizon: int, ensemble_root: Path) -> list[tuple[Path, dict[str, float]]]:
    results = []
    missing: list[str] = []
    for seed_name in EXPECTED_SEEDS:
        seed_dir = ensemble_root / seed_name
        metrics_path = seed_dir / "metrics" / "aem_gnn_overall_metrics.csv"
        checkpoint_path = seed_dir / "checkpoints" / "aem_gnn" / "best_checkpoint.pt"
        if not metrics_path.exists():
            missing.append(f"{seed_name}: missing metrics file at {metrics_path}")
            continue
        if not checkpoint_path.exists():
            missing.append(f"{seed_name}: missing checkpoint at {checkpoint_path}")
            continue
        record = _validation_record_from_metrics(metrics_path)
        results.append((checkpoint_path, record))
    if missing:
        bullet_list = "\n".join(f" - {msg}" for msg in missing)
        raise FileNotFoundError(f"Missing required ensemble members under {ensemble_root}.\n{bullet_list}")
    return results


class EnsembleSpecialistPredictor:
    """Averages delta_h predictions from multiple SpecialistPredictor members."""

    def __init__(self, members: list[SpecialistPredictor]) -> None:
        if not members:
            raise ValueError("At least one member predictor is required.")
        ref = members[0]
        ref_grid_ids = np.asarray(ref.grid_ids)
        ref_month_labels = np.asarray(ref.month_labels).astype(str)
        ref_edge_index = _tensor_to_numpy(ref.edge_index)
        ref_edge_weight = _tensor_to_numpy(ref.edge_weight)
        ref_active_grid = ref.active_grid.reset_index(drop=True)
        ref_active_cols = [
            col for col in ("grid_id", "row", "col", "cell_id", "x", "y")
            if col in ref_active_grid.columns
        ]

        for member_idx, member in enumerate(members[1:], start=2):
            if member.horizon != ref.horizon:
                raise ValueError(f"Ensemble member {member_idx} has horizon {member.horizon}, expected {ref.horizon}.")
            if not np.array_equal(np.asarray(member.grid_ids), ref_grid_ids):
                raise ValueError(f"Ensemble member {member_idx} has inconsistent grid_ids.")
            if not np.array_equal(np.asarray(member.month_labels).astype(str), ref_month_labels):
                raise ValueError(f"Ensemble member {member_idx} has inconsistent month_labels.")
            if not np.array_equal(_tensor_to_numpy(member.edge_index), ref_edge_index):
                raise ValueError(f"Ensemble member {member_idx} has inconsistent graph edges.")
            member_edge_weight = _tensor_to_numpy(member.edge_weight)
            if member_edge_weight.shape != ref_edge_weight.shape or not np.allclose(
                member_edge_weight,
                ref_edge_weight,
                atol=1.0e-8,
                rtol=0.0,
            ):
                raise ValueError(f"Ensemble member {member_idx} has inconsistent graph weights.")
            member_active_grid = member.active_grid.reset_index(drop=True)
            if ref_active_cols and not member_active_grid[ref_active_cols].equals(ref_active_grid[ref_active_cols]):
                raise ValueError(f"Ensemble member {member_idx} has inconsistent active-grid metadata.")

        self._members = members
        self.horizon = members[0].horizon
        self.grid_ids = members[0].grid_ids
        self.active_grid = members[0].active_grid
        self.month_labels = members[0].month_labels
        self.edge_index = members[0].edge_index
        self.edge_weight = members[0].edge_weight

    def predict_delta_full(self, current_wtd_m_bls: np.ndarray, start_month_idx: int) -> np.ndarray:
        preds = [member.predict_delta_full(current_wtd_m_bls, start_month_idx) for member in self._members]
        return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)

    def validation_metrics(self) -> dict[str, float]:
        all_metrics = [member.validation_metrics() for member in self._members]
        keys = list(all_metrics[0].keys())
        return {key: float(np.mean([metrics[key] for metrics in all_metrics])) for key in keys}

    def conformal_qhat(self, level: float = 0.75) -> float | None:
        valid = [q for q in (member.conformal_qhat(level=level) for member in self._members) if q is not None]
        return float(np.mean(valid)) if valid else None


def build_anomaly_observations(observed_absolute: dict[str, object], baseline_wtd: np.ndarray) -> dict[str, object]:
    baseline_wtd = np.asarray(baseline_wtd, dtype=np.float32)
    out = {
        "values": np.asarray(observed_absolute["values"], dtype=np.float32).copy(),
        "quality": np.asarray(observed_absolute["quality"], dtype=np.float32).copy(),
        "mask": np.asarray(observed_absolute["mask"], dtype=bool).copy(),
        "monthly_observation_counts": np.asarray(
            observed_absolute["monthly_observation_counts"],
            dtype=np.int64,
        ).copy(),
    }
    mask = out["mask"]
    out["values"][mask] = out["values"][mask] - np.broadcast_to(baseline_wtd[None, :], out["values"].shape)[mask]
    return out


def _robust_spread(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    q75, q25 = np.percentile(values, [75.0, 25.0])
    return float((q75 - q25) / 1.349) if q75 > q25 else float(np.std(values))


def _well_regional_signal(obs_values: np.ndarray, obs_mask: np.ndarray) -> tuple[pd.Series, pd.Series]:
    n_month, n_grid = obs_values.shape
    grid_median = np.zeros(n_grid, dtype=np.float32)
    grid_has_obs = obs_mask.sum(axis=0) > 0
    for grid_idx in np.where(grid_has_obs)[0]:
        idx = np.where(obs_mask[:, grid_idx])[0]
        grid_median[grid_idx] = float(np.nanmedian(obs_values[idx, grid_idx]))

    regional_anomaly = np.full(n_month, np.nan, dtype=np.float32)
    for month_idx in range(n_month):
        month_mask = obs_mask[month_idx]
        if np.any(month_mask):
            month_vals = obs_values[month_idx, month_mask] - grid_median[month_mask]
            regional_anomaly[month_idx] = float(np.nanmedian(month_vals))

    regional_series = pd.Series(regional_anomaly, index=np.arange(n_month, dtype=np.int64), dtype=np.float32)
    if regional_series.notna().any():
        regional_filled = regional_series.interpolate(method="linear", limit_direction="both")
        regional_smoothed = regional_filled.rolling(window=3, center=True, min_periods=1).mean()
        regional_centered = regional_smoothed - float(regional_smoothed.mean())
    else:
        regional_centered = pd.Series(np.zeros(n_month, dtype=np.float32), index=regional_series.index, dtype=np.float32)

    valid_abs = np.abs(regional_centered.to_numpy(dtype=np.float32))
    valid_abs = valid_abs[np.isfinite(valid_abs)]
    if valid_abs.size > 0:
        cap = max(float(np.quantile(valid_abs, 0.95)), 0.20)
        regional_centered = regional_centered.clip(lower=-cap, upper=cap)
    return regional_series, regional_centered.astype(np.float32)


def build_selective_anchor_baseline(
    observed_anomaly: dict[str, object],
    baseline_quality: np.ndarray,
    anchor_month_indices: list[int],
    month_labels: np.ndarray,
    grid_ids: np.ndarray,
    baseline_wtd: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], pd.DataFrame, pd.DataFrame]:
    obs_values = np.asarray(observed_anomaly["values"], dtype=np.float32)
    obs_mask = np.asarray(observed_anomaly["mask"], dtype=bool)
    monthly_observation_counts = np.asarray(observed_anomaly["monthly_observation_counts"], dtype=np.int64)
    n_month, n_grid = obs_values.shape
    grid_ids = np.asarray(grid_ids, dtype=np.int64)
    if grid_ids.shape != (n_grid,):
        raise ValueError(f"grid_ids shape {grid_ids.shape} does not match n_grid={n_grid}")

    n_obs = np.sum(obs_mask, axis=0).astype(np.int32)
    median_anom = np.zeros(n_grid, dtype=np.float32)
    spread = np.zeros(n_grid, dtype=np.float32)
    support = np.zeros(n_grid, dtype=np.float32)
    anchor_quality = np.asarray(baseline_quality, dtype=np.float32).copy()

    for grid_idx in range(n_grid):
        idx = np.where(obs_mask[:, grid_idx])[0]
        if idx.size == 0:
            support[grid_idx] = 0.0
            continue
        vals = obs_values[idx, grid_idx].astype(np.float32)
        median_anom[grid_idx] = float(np.median(vals))
        spread_val = _robust_spread(vals)
        spread[grid_idx] = spread_val

        count_score = float(np.clip((idx.size - 1) / 5.0, 0.0, 1.0))
        stability_score = 1.0 / (1.0 + spread_val / 2.0)
        support_score = count_score * stability_score
        support[grid_idx] = float(np.clip(support_score, 0.0, 1.0))

        if idx.size == 1:
            anchor_quality[grid_idx] = float(np.clip(anchor_quality[grid_idx] * 0.65, 0.05, 0.45))
        else:
            baseline_factor = 0.25 + 0.45 * (1.0 - support[grid_idx])
            local_quality = 0.10 + 0.55 * support[grid_idx]
            anchor_quality[grid_idx] = float(
                np.clip(max(anchor_quality[grid_idx] * baseline_factor, local_quality), 0.08, 0.80)
            )

    selective_offset = median_anom * support
    regional_raw, regional_centered = _well_regional_signal(obs_values, obs_mask)

    regional_sensitivity = (0.65 + 0.35 * (1.0 - support)).astype(np.float32)
    unobs_reg = support == 0.0
    obs_reg_fallback = support > 0.1

    fallback_a, fallback_b = 0.0, 0.0
    if obs_reg_fallback.sum() >= 20:
        xf = baseline_wtd[obs_reg_fallback].astype(np.float64)
        yf = median_anom[obs_reg_fallback].astype(np.float64)
        wf = support[obs_reg_fallback].astype(np.float64)
        xfm = float(np.average(xf, weights=wf))
        yfm = float(np.average(yf, weights=wf))
        fallback_a = float(np.sum(wf * (xf - xfm) * (yf - yfm)) / max(np.sum(wf * (xf - xfm) ** 2), 1.0e-6))
        fallback_b = yfm - fallback_a * xfm

    anchor_base_fields_by_month: dict[int, np.ndarray] = {}
    regression_rows: list[dict[str, object]] = []
    for month_idx in anchor_month_indices:
        midx = int(month_idx)
        region_value = float(regional_centered.iloc[midx])
        month_spatial_offset = selective_offset.copy()

        if unobs_reg.sum() > 0:
            month_obs_mask = obs_mask[midx]
            has_obs_this_month = month_obs_mask & (support > 0.0)
            n_this = int(has_obs_this_month.sum())
            if n_this >= 10:
                x = baseline_wtd[has_obs_this_month].astype(np.float64)
                y = (obs_values[midx, has_obs_this_month] - region_value).astype(np.float64)
                w = support[has_obs_this_month].astype(np.float64)
                xm = float(np.average(x, weights=w))
                ym = float(np.average(y, weights=w))
                a = float(np.sum(w * (x - xm) * (y - ym)) / max(np.sum(w * (x - xm) ** 2), 1.0e-6))
                b = ym - a * xm
                predicted = (a * baseline_wtd[unobs_reg].astype(np.float64) + b).astype(np.float32)

                q05, q95 = np.nanquantile(y, [0.05, 0.95])
                margin = max(0.50, 0.25 * float(q95 - q05))
                predicted = np.clip(predicted, float(q05 - margin), float(q95 + margin)).astype(np.float32)
                month_spatial_offset[unobs_reg] = predicted
                regression_rows.append(
                    {
                        "anchor_month_idx": midx,
                        "anchor_month_label": str(month_labels[midx]),
                        "n_obs_regression": n_this,
                        "slope": a,
                        "intercept": b,
                        "clip_low_m": float(q05 - margin),
                        "clip_high_m": float(q95 + margin),
                        "pred_min_m": float(predicted.min()),
                        "pred_max_m": float(predicted.max()),
                    }
                )
            else:
                predicted = (fallback_a * baseline_wtd[unobs_reg].astype(np.float64) + fallback_b).astype(np.float32)
                month_spatial_offset[unobs_reg] = predicted

        anchor_base_fields_by_month[midx] = (
            month_spatial_offset + regional_sensitivity * region_value
        ).astype(np.float32)

    anchor_base_field_quality_by_month = {
        int(month_idx): np.clip(
            anchor_quality * (0.75 + 0.45 * min(float(monthly_observation_counts[int(month_idx)]) / 120.0, 1.0)),
            0.08,
            0.90,
        ).astype(np.float32)
        for month_idx in anchor_month_indices
    }

    anchor_signal_summary = pd.DataFrame(
        {
            "anchor_month_idx": [int(v) for v in anchor_month_indices],
            "anchor_month_label": [str(month_labels[int(v)]) for v in anchor_month_indices],
            "monthly_observation_count": [int(monthly_observation_counts[int(v)]) for v in anchor_month_indices],
            "regional_anomaly_raw_m": [float(regional_raw.iloc[int(v)]) for v in anchor_month_indices],
            "regional_anomaly_smoothed_centered_m": [
                float(regional_centered.iloc[int(v)]) for v in anchor_month_indices
            ],
        }
    )
    if regression_rows:
        regression_summary = pd.DataFrame(regression_rows)
        anchor_signal_summary = anchor_signal_summary.merge(
            regression_summary,
            on=["anchor_month_idx", "anchor_month_label"],
            how="left",
        )

    summary = pd.DataFrame(
        {
            "grid_id": grid_ids,
            "n_observed_months": n_obs,
            "local_median_anomaly_m": median_anom,
            "local_spread_m": spread,
            "local_support_score": support,
            "regional_sensitivity": regional_sensitivity,
            "baseline_quality_in": baseline_quality,
            "selective_anchor_quality": anchor_quality,
            "selective_anchor_anomaly_m": selective_offset,
        }
    )
    return anchor_base_fields_by_month, anchor_base_field_quality_by_month, summary, anchor_signal_summary


def build_anchor_indices(month_labels: np.ndarray) -> list[int]:
    anchors = [idx for idx, label in enumerate(map(str, month_labels)) if label.endswith("-01")]
    if len(anchors) != EXPECTED_JANUARY_ANCHOR_COUNT:
        labels = [str(month_labels[idx]) for idx in anchors]
        raise ValueError(
            f"Expected {EXPECTED_JANUARY_ANCHOR_COUNT} January anchors for 2011-2023, "
            f"found {len(anchors)}: {labels}"
        )
    labels = [str(value) for value in month_labels]
    if FINAL_ENDPOINT_LABEL not in labels:
        raise ValueError(f"Final endpoint anchor {FINAL_ENDPOINT_LABEL} is not in month_labels.")
    final_idx = labels.index(FINAL_ENDPOINT_LABEL)
    if final_idx not in anchors:
        anchors.append(final_idx)
    return anchors


def _windows_exclusive_open_error(path: Path) -> str | None:
    if sys.platform != "win32" or not path.exists():
        return None

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    generic_read = 0x80000000
    generic_write = 0x40000000
    open_existing = 3
    file_attribute_normal = 0x80
    invalid_handle_value = ctypes.c_void_p(-1).value

    handle = kernel32.CreateFileW(
        str(path),
        generic_read | generic_write,
        0,
        None,
        open_existing,
        file_attribute_normal,
        None,
    )
    if handle == invalid_handle_value:
        err = ctypes.get_last_error()
        return f"Windows error {err}"
    kernel32.CloseHandle(handle)
    return None


def assert_output_targets_available(out_dir: Path) -> None:
    targets = [
        out_dir / "reconstruction" / "wtd_reconstructed_matrix.npy",
    ]
    locked = [(path, err) for path in targets if (err := _windows_exclusive_open_error(path))]
    if locked:
        details = "\n".join(f" - {display_path(path)} ({err})" for path, err in locked)
        raise PermissionError(
            "Cannot rebuild because existing reconstruction output files are open or locked:\n"
            f"{details}\n"
            "Close notebooks/kernels that loaded these files with mmap_mode='r' "
            "(for example drought metrics, Fig2/Fig3/SFIG2 notebooks), plus any viewer/GIS using them, then rerun."
        )


def main() -> None:
    args = parse_args()
    out_dir = (
        Path(args.out_dir).resolve()
        if str(args.out_dir).strip()
        else DEFAULT_OUT_DIR
    )
    assert_output_targets_available(out_dir)

    base_predictors: dict[int, EnsembleSpecialistPredictor] = {}
    for horizon, ensemble_root in ENSEMBLE_ROOTS.items():
        seed_checkpoints = all_seed_checkpoints(horizon, ensemble_root)
        print(f"H{horizon}: loading {len(seed_checkpoints)} seed checkpoints...")
        members = [SpecialistPredictor.from_checkpoint(ckpt) for ckpt, _ in seed_checkpoints]
        base_predictors[horizon] = EnsembleSpecialistPredictor(members)

    master = base_predictors[6]
    longterm = load_longterm_baseline_field(
        LONGTERM_PATH,
        master.active_grid,
        value_col="wtd_model_mean_m",
    )
    baseline_wtd = np.asarray(longterm["values"], dtype=np.float32)
    baseline_quality = np.asarray(longterm["quality"], dtype=np.float32)

    predictors = {
        horizon: BaselineAdjustedPredictor(base_predictor=predictor, baseline_wtd_m=baseline_wtd)
        for horizon, predictor in base_predictors.items()
    }

    observed_absolute = build_observation_matrices(
        wtd_monthly_path=WTD_MONTHLY_PATH,
        active_grid=master.active_grid,
        month_labels=master.month_labels,
    )
    observed_anomaly = build_anomaly_observations(observed_absolute, baseline_wtd)

    anchor_month_indices = build_anchor_indices(master.month_labels)
    (
        anchor_base_fields_by_month,
        anchor_base_field_quality_by_month,
        selective_summary,
        _anchor_signal_summary,
    ) = build_selective_anchor_baseline(
        observed_anomaly,
        baseline_quality,
        anchor_month_indices,
        master.month_labels,
        grid_ids=master.grid_ids,
        baseline_wtd=baseline_wtd,
    )

    reconstruction = reconstruct_monthly_wtd_adaptive(
        predictors=predictors,
        observed=observed_anomaly,
        anchor_month_indices=anchor_month_indices,
        anchor_base_fields_by_month=anchor_base_fields_by_month,
        anchor_base_field_quality=0.15,
        anchor_base_field_quality_by_month=anchor_base_field_quality_by_month,
        smooth_lambda=1.0,
        split_rmse_threshold_m=2.0,
        split_p95_threshold_m=3.0,
        soft_anchor=True,
        soft_anchor_obs_ref_count=100.0,
        soft_anchor_alpha_min=0.10,
        soft_anchor_alpha_max=0.80,
        soft_anchor_mismatch_scale_m=3.0,
        direct_obs_exact_min_count=2,
        direct_obs_single_alpha_max=0.05,
        direct_obs_single_mismatch_scale_m=8.0,
        use_local_anchor_priors=True,
        local_anchor_prior_multi_obs_min_count=2,
        local_anchor_prior_extrap_window_months=18,
        local_anchor_prior_single_obs_window_months=3,
        local_anchor_prior_interp_span_scale_months=60.0,
        smooth_local_anchor_priors_into_background=True,
        local_anchor_prior_graph_quality_scale=5.0,
        local_anchor_prior_graph_quality_floor=0.10,
        use_local_observation_anchors=True,
        local_anchor_multi_obs_min_count=2,
        local_anchor_bridge_alpha_min=0.70,
        local_anchor_bridge_alpha_max=0.80,
        local_anchor_bridge_gap_scale_months=48.0,
        local_anchor_edge_window_months=12,
        local_anchor_edge_alpha_max=0.50,
        local_anchor_single_obs_alpha_max=0.08,
        local_anchor_single_obs_mismatch_scale_m=8.0,
        use_monthly_residual_diffusion=True,
        monthly_residual_smooth_lambda=1.25,
        monthly_residual_gain=0.70,
        monthly_residual_cap_m=2.50,
        monthly_residual_min_obs=20,
        monthly_residual_exact_at_obs=True,
        allow_final_extrapolation=False,
    )

    reconstruction["longterm_baseline_wtd_m_bls"] = baseline_wtd
    reconstruction["longterm_baseline_quality"] = baseline_quality
    reconstruction["reconstructed_absolute_matrix"] = reconstruction["reconstructed_matrix"] + baseline_wtd[None, :]

    outputs = save_reconstruction_outputs(reconstruction, observed_absolute, out_dir)

    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary = pd.DataFrame(
        {
            "grid_id": master.grid_ids,
            "longterm_baseline_wtd_m_bls": baseline_wtd,
            "longterm_baseline_quality": baseline_quality,
            "longterm_fill_source": np.asarray(longterm["fill_source"], dtype=object),
            "longterm_direct_flag": np.asarray(longterm["has_direct_footprint_cell"], dtype=bool),
            "longterm_n_footprint_points": np.asarray(longterm["n_footprint_points"], dtype=np.float32),
        }
    )
    baseline_summary.to_csv(meta_dir / "longterm_baseline_on_grid.csv", index=False, lineterminator="\n")
    selective_summary.to_csv(meta_dir / "selective_anchor_grid_summary.csv", index=False, lineterminator="\n")

    print("Mainline reconstruction finished.")
    for key in [
        "reconstructed_matrix_npy",
        "grid_lookup_csv",
        "month_index_csv",
        "interval_summary_csv",
        "gap_paths_json",
    ]:
        if key in outputs:
            print(f"{key}: {display_path(outputs[key])}")


if __name__ == "__main__":
    main()
