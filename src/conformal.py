from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _level_suffix(level: float) -> str:
    return str(int(round(level * 100)))


def conformal_qhat(abs_residuals: Iterable[float], alpha: float) -> float:
    scores = np.asarray(list(abs_residuals), dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("No finite residuals available for conformal calibration.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}")

    scores = np.sort(scores)
    n = scores.size
    q_level = math.ceil((n + 1) * (1.0 - alpha)) / n
    q_level = min(max(q_level, 0.0), 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def fit_split_conformal(
    predictions: pd.DataFrame,
    fit_split: str = "validation",
    pred_col: str = "y_pred_delta_h_m",
    true_col: str = "y_true_delta_h_m",
    levels: Iterable[float] = (0.90, 0.95),
) -> pd.DataFrame:
    frame = predictions[predictions["split"] == fit_split].copy()
    if frame.empty:
        raise ValueError(f"No rows found for split={fit_split!r}.")

    abs_residuals = (frame[true_col].astype(float) - frame[pred_col].astype(float)).abs().to_numpy()
    rows: list[dict[str, Any]] = []
    for level in levels:
        alpha = 1.0 - float(level)
        qhat = conformal_qhat(abs_residuals, alpha)
        rows.append(
            {
                "level": float(level),
                "alpha": alpha,
                "qhat_m": qhat,
                "n_calibration": int(len(frame)),
                "fit_split": fit_split,
                "pred_col": pred_col,
                "true_col": true_col,
            }
        )
    return pd.DataFrame(rows)


def apply_split_conformal(
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    pred_col: str = "y_pred_delta_h_m",
) -> pd.DataFrame:
    frame = predictions.copy()
    for row in calibration.to_dict(orient="records"):
        level = float(row["level"])
        qhat = float(row["qhat_m"])
        suffix = _level_suffix(level)
        frame[f"pi_{suffix}_radius_m"] = qhat
        frame[f"pi_{suffix}_lower_m"] = frame[pred_col].astype(float) - qhat
        frame[f"pi_{suffix}_upper_m"] = frame[pred_col].astype(float) + qhat
    return frame


def summarize_prediction_intervals(
    predictions_with_intervals: pd.DataFrame,
    true_col: str = "y_true_delta_h_m",
    levels: Iterable[float] = (0.90, 0.95),
    calibration: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    calibration_lookup: dict[float, float] = {}
    if calibration is not None and not calibration.empty:
        qhat_col = "qhat_m" if "qhat_m" in calibration.columns else "qhat"
        calibration_lookup = {
            float(row["level"]): float(row[qhat_col])
            for row in calibration.to_dict(orient="records")
        }
    for split, sub in predictions_with_intervals.groupby("split", sort=False):
        split_true = sub[true_col].astype(float)
        for level in levels:
            suffix = _level_suffix(float(level))
            lower_col = f"pi_{suffix}_lower_m"
            upper_col = f"pi_{suffix}_upper_m"
            radius_col = f"pi_{suffix}_radius_m"
            radius = sub[radius_col].astype(float)
            covered = ((split_true >= sub[lower_col].astype(float)) & (split_true <= sub[upper_col].astype(float))).to_numpy()
            row = {
                "split": split,
                "level": float(level),
                "nominal_coverage": float(level),
                "empirical_coverage": float(np.mean(covered)),
                # Kept for backward compatibility with older notebooks/CSVs.
                "qhat_m": float(radius.iloc[0]),
                "radius_reference_m": float(radius.iloc[0]),
                "mean_radius_m": float(radius.mean()),
                "median_radius_m": float(radius.median()),
                "radius_is_constant": bool(np.allclose(radius.to_numpy(), radius.iloc[0])),
                "mean_width_m": float((sub[upper_col] - sub[lower_col]).mean()),
                "median_width_m": float((sub[upper_col] - sub[lower_col]).median()),
                "n_samples": int(len(sub)),
            }
            if float(level) in calibration_lookup:
                row["calibration_qhat"] = calibration_lookup[float(level)]
            rows.append(row)
    return pd.DataFrame(rows)


def fit_normalized_conformal(
    predictions: pd.DataFrame,
    fit_split: str = "validation",
    pred_col: str = "y_pred_delta_h_m",
    true_col: str = "y_true_delta_h_m",
    scale_col: str = "ensemble_std_m",
    levels: Iterable[float] = (0.90, 0.95),
    eps: float = 1.0e-6,
) -> pd.DataFrame:
    frame = predictions[predictions["split"] == fit_split].copy()
    if frame.empty:
        raise ValueError(f"No rows found for split={fit_split!r}.")

    scale = frame[scale_col].astype(float).to_numpy()
    scale = np.maximum(scale, float(eps))
    abs_residuals = (frame[true_col].astype(float) - frame[pred_col].astype(float)).abs().to_numpy()
    normalized_scores = abs_residuals / scale

    rows: list[dict[str, Any]] = []
    for level in levels:
        alpha = 1.0 - float(level)
        qhat = conformal_qhat(normalized_scores, alpha)
        rows.append(
            {
                "level": float(level),
                "alpha": alpha,
                "qhat": qhat,
                "n_calibration": int(len(frame)),
                "fit_split": fit_split,
                "pred_col": pred_col,
                "true_col": true_col,
                "scale_col": scale_col,
                "eps": float(eps),
            }
        )
    return pd.DataFrame(rows)


def apply_normalized_conformal(
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    pred_col: str = "y_pred_delta_h_m",
    scale_col: str = "ensemble_std_m",
    out_prefix: str = "pi",
) -> pd.DataFrame:
    frame = predictions.copy()
    for row in calibration.to_dict(orient="records"):
        level = float(row["level"])
        qhat = float(row["qhat"])
        eps = float(row.get("eps", 1.0e-6))
        suffix = _level_suffix(level)
        scale = np.maximum(frame[scale_col].astype(float).to_numpy(), eps)
        radius = qhat * scale
        frame[f"{out_prefix}_{suffix}_radius_m"] = radius
        frame[f"{out_prefix}_{suffix}_lower_m"] = frame[pred_col].astype(float) - radius
        frame[f"{out_prefix}_{suffix}_upper_m"] = frame[pred_col].astype(float) + radius
    return frame
