from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg, spsolve
from torch_geometric.data import Data

from .dataset import _target_mean_std_for_horizon, _target_scaling_mode
from .train import _build_model
from .utils import (
    device_from_config,
    ensure_dir,
    load_active_grid,
    load_or_build_aem_cache,
    load_or_build_dynamic_cache,
    load_topography_cache,
    load_time_index,
    month_sin_cos,
    output_paths,
    pumping_log_dynamic_feature_name,
    pumping_raw_dynamic_feature_name,
    regression_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _display_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std_safe = np.where(np.asarray(std, dtype=np.float32) == 0.0, 1.0, np.asarray(std, dtype=np.float32))
    return ((np.asarray(values, dtype=np.float32) - np.asarray(mean, dtype=np.float32)) / std_safe).astype(np.float32)


def _smoothstep(z: float) -> float:
    z = float(np.clip(z, 0.0, 1.0))
    return float(3.0 * z * z - 2.0 * z * z * z)


def _soft_anchor_alpha(
    month_observation_count: int,
    abs_mismatch: np.ndarray,
    obs_ref_count: float,
    alpha_min: float,
    alpha_max: float,
    mismatch_scale_m: float,
) -> np.ndarray:
    base = float(np.clip(float(month_observation_count) / max(float(obs_ref_count), 1.0), alpha_min, alpha_max))
    mismatch_scale_m = max(float(mismatch_scale_m), 1e-6)
    alpha = base / (1.0 + np.asarray(abs_mismatch, dtype=np.float32) / mismatch_scale_m)
    return np.clip(alpha.astype(np.float32), 0.02, 1.0)


def _local_bridge_alpha(
    *,
    left_quality: float,
    right_quality: float,
    gap_months: int,
    alpha_min: float,
    alpha_max: float,
    gap_scale_months: float,
) -> float:
    quality_score = np.clip(0.5 * (float(left_quality) + float(right_quality)) / 5.0, 0.0, 1.0)
    alpha = float(alpha_min) + (float(alpha_max) - float(alpha_min)) * quality_score
    gap_penalty = 1.0 / (1.0 + max(float(gap_months - 1), 0.0) / max(float(gap_scale_months), 1.0))
    return float(np.clip(alpha * gap_penalty, 0.05, 0.95))


def _single_observation_anchor_alpha(
    *,
    quality: float,
    abs_mismatch: float,
    mismatch_scale_m: float,
    alpha_max: float,
) -> float:
    base = 0.05 + 0.15 * float(np.clip(float(quality) / 5.0, 0.0, 1.0))
    alpha = base / (1.0 + float(abs_mismatch) / max(float(mismatch_scale_m), 1e-6))
    return float(np.clip(alpha, 0.02, float(alpha_max)))


def build_local_anchor_priors(
    obs_values: np.ndarray,
    obs_mask: np.ndarray,
    obs_quality: np.ndarray,
    anchor_month_indices: list[int],
    *,
    multi_obs_min_count: int = 2,
    extrap_window_months: int = 12,
    single_obs_window_months: int = 3,
    interp_span_scale_months: float = 48.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    n_month, n_grid = obs_values.shape
    n_anchor = len(anchor_month_indices)
    prior_values = np.full((n_anchor, n_grid), np.nan, dtype=np.float32)
    prior_weights = np.zeros((n_anchor, n_grid), dtype=np.float32)
    records: list[dict[str, Any]] = []

    for grid_idx in range(n_grid):
        month_idx = np.where(obs_mask[:, grid_idx])[0]
        n_obs = int(len(month_idx))
        if n_obs == 0:
            continue
        vals = obs_values[month_idx, grid_idx].astype(np.float32)
        quals = obs_quality[month_idx, grid_idx].astype(np.float32)
        spread = float(np.nanstd(vals)) if n_obs > 1 else 0.0
        quality_score = float(np.clip(np.nanmean(quals) / 5.0, 0.0, 1.0))
        count_score = float(np.clip(n_obs / 6.0, 0.0, 1.0))
        spread_factor = 1.0 / (1.0 + spread / 5.0)
        multi_base_weight = float(np.clip(0.35 + 0.55 * count_score * spread_factor, 0.20, 0.90))
        single_base_weight = float(np.clip(0.05 + 0.10 * quality_score, 0.05, 0.20))

        for anchor_rank, anchor_month_idx in enumerate(anchor_month_indices):
            anchor_month_idx = int(anchor_month_idx)
            if obs_mask[anchor_month_idx, grid_idx]:
                continue

            mode = None
            prior = None
            weight = 0.0

            if n_obs >= int(multi_obs_min_count):
                if anchor_month_idx < int(month_idx[0]):
                    gap = int(month_idx[0] - anchor_month_idx)
                    if gap <= int(extrap_window_months):
                        prior = float(vals[0])
                        weight = multi_base_weight * max(0.0, 1.0 - gap / max(float(extrap_window_months) + 1.0, 1.0)) * 0.7
                        mode = "pre_nearest"
                elif anchor_month_idx > int(month_idx[-1]):
                    gap = int(anchor_month_idx - month_idx[-1])
                    if gap <= int(extrap_window_months):
                        prior = float(vals[-1])
                        weight = multi_base_weight * max(0.0, 1.0 - gap / max(float(extrap_window_months) + 1.0, 1.0)) * 0.7
                        mode = "post_nearest"
                else:
                    right_pos = int(np.searchsorted(month_idx, anchor_month_idx))
                    left_pos = max(right_pos - 1, 0)
                    if right_pos < len(month_idx):
                        left_month = int(month_idx[left_pos])
                        right_month = int(month_idx[right_pos])
                        left_val = float(vals[left_pos])
                        right_val = float(vals[right_pos])
                        span = max(right_month - left_month, 1)
                        frac = float(anchor_month_idx - left_month) / float(span)
                        prior = left_val + frac * (right_val - left_val)
                        span_factor = 1.0 / (1.0 + float(span) / max(float(interp_span_scale_months), 1.0))
                        weight = multi_base_weight * span_factor
                        mode = "interpolate"
            else:
                gap = int(abs(anchor_month_idx - int(month_idx[0])))
                if gap <= int(single_obs_window_months):
                    prior = float(vals[0])
                    weight = single_base_weight * max(0.0, 1.0 - gap / max(float(single_obs_window_months) + 1.0, 1.0))
                    mode = "single_near"

            if prior is None or weight <= 0.0:
                continue

            prior_values[anchor_rank, grid_idx] = np.float32(prior)
            prior_weights[anchor_rank, grid_idx] = np.float32(weight)
            records.append(
                {
                    "grid_index": int(grid_idx),
                    "anchor_rank": int(anchor_rank),
                    "anchor_month_idx": int(anchor_month_idx),
                    "n_observed_months": n_obs,
                    "mode": str(mode),
                    "prior_value_m": float(prior),
                    "prior_weight": float(weight),
                }
            )

    return prior_values, prior_weights, pd.DataFrame(records)


def _assimilate_observations_into_state(
    state: np.ndarray,
    *,
    month_idx: int,
    obs_values: np.ndarray,
    obs_mask: np.ndarray,
    obs_quality: np.ndarray,
    grid_observation_counts: np.ndarray,
    exact_obs_min_count: int,
    single_obs_alpha_max: float,
    single_obs_mismatch_scale_m: float,
) -> np.ndarray:
    out = np.asarray(state, dtype=np.float32).copy()
    month_mask = np.asarray(obs_mask[month_idx], dtype=bool)
    if not np.any(month_mask):
        return out

    idx = np.where(month_mask)[0]
    obs_month_values = np.asarray(obs_values[month_idx, idx], dtype=np.float32)
    counts = np.asarray(grid_observation_counts[idx], dtype=np.int64)
    quality = np.asarray(obs_quality[month_idx, idx], dtype=np.float32)

    exact = counts >= int(exact_obs_min_count)
    if np.any(exact):
        out[idx[exact]] = obs_month_values[exact]

    weak = ~exact
    if np.any(weak):
        current = out[idx[weak]].astype(np.float32)
        mismatch = np.abs(obs_month_values[weak] - current).astype(np.float32)
        alpha = np.asarray(
            [
                _single_observation_anchor_alpha(
                    quality=float(q),
                    abs_mismatch=float(m),
                    mismatch_scale_m=single_obs_mismatch_scale_m,
                    alpha_max=single_obs_alpha_max,
                )
                for q, m in zip(quality[weak], mismatch, strict=False)
            ],
            dtype=np.float32,
        )
        out[idx[weak]] = current + alpha * (obs_month_values[weak] - current)
    return out.astype(np.float32)


def apply_local_observation_anchors(
    reconstructed: np.ndarray,
    obs_values: np.ndarray,
    obs_mask: np.ndarray,
    obs_quality: np.ndarray,
    *,
    multi_obs_min_count: int = 2,
    bridge_alpha_min: float = 0.35,
    bridge_alpha_max: float = 0.80,
    bridge_gap_scale_months: float = 12.0,
    edge_window_months: int = 6,
    edge_alpha_max: float = 0.25,
    single_obs_alpha_max: float = 0.20,
    single_obs_mismatch_scale_m: float = 5.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    out = np.asarray(reconstructed, dtype=np.float32).copy()
    n_month, n_grid = out.shape
    records: list[dict[str, Any]] = []

    for grid_idx in range(n_grid):
        month_idx = np.where(obs_mask[:, grid_idx])[0]
        n_obs = int(len(month_idx))
        if n_obs == 0:
            continue

        if n_obs < int(multi_obs_min_count):
            idx = int(month_idx[0])
            obs_val = float(obs_values[idx, grid_idx])
            if idx == 0 and n_month > 1:
                baseline = float(out[1, grid_idx])
            elif idx == n_month - 1 and n_month > 1:
                baseline = float(out[n_month - 2, grid_idx])
            else:
                left = float(out[max(idx - 1, 0), grid_idx])
                right = float(out[min(idx + 1, n_month - 1), grid_idx])
                baseline = 0.5 * (left + right)
            alpha = _single_observation_anchor_alpha(
                quality=float(obs_quality[idx, grid_idx]),
                abs_mismatch=abs(obs_val - baseline),
                mismatch_scale_m=single_obs_mismatch_scale_m,
                alpha_max=single_obs_alpha_max,
            )
            out[idx, grid_idx] = np.float32(baseline + alpha * (obs_val - baseline))
            records.append(
                {
                    "grid_index": int(grid_idx),
                    "n_observed_months": n_obs,
                    "mode": "single_obs_weak_anchor",
                    "month_idx": idx,
                    "anchor_alpha": float(alpha),
                    "obs_value_m": obs_val,
                    "baseline_value_m": baseline,
                    "gap_months": 0,
                }
            )
            continue

        obs_months = month_idx.astype(int).tolist()
        obs_series = obs_values[obs_months, grid_idx].astype(np.float32)
        qual_series = obs_quality[obs_months, grid_idx].astype(np.float32)

        first_idx = obs_months[0]
        last_idx = obs_months[-1]
        first_val = float(obs_series[0])
        last_val = float(obs_series[-1])

        if first_idx > 0:
            window_start = max(0, first_idx - int(edge_window_months))
            for month in range(window_start, first_idx):
                distance = float(first_idx - month)
                alpha = float(edge_alpha_max) * (1.0 - distance / max(float(edge_window_months), 1.0))
                alpha = float(np.clip(alpha, 0.0, edge_alpha_max))
                out[month, grid_idx] = np.float32((1.0 - alpha) * out[month, grid_idx] + alpha * first_val)

        if last_idx < n_month - 1:
            window_stop = min(n_month - 1, last_idx + int(edge_window_months))
            for month in range(last_idx + 1, window_stop + 1):
                distance = float(month - last_idx)
                alpha = float(edge_alpha_max) * (1.0 - distance / max(float(edge_window_months), 1.0))
                alpha = float(np.clip(alpha, 0.0, edge_alpha_max))
                out[month, grid_idx] = np.float32((1.0 - alpha) * out[month, grid_idx] + alpha * last_val)

        for left_obs_idx, right_obs_idx, left_val, right_val, left_q, right_q in zip(
            obs_months[:-1],
            obs_months[1:],
            obs_series[:-1],
            obs_series[1:],
            qual_series[:-1],
            qual_series[1:],
            strict=False,
        ):
            gap = int(right_obs_idx - left_obs_idx)
            if gap <= 1:
                out[left_obs_idx, grid_idx] = np.float32(left_val)
                out[right_obs_idx, grid_idx] = np.float32(right_val)
                continue
            alpha = _local_bridge_alpha(
                left_quality=float(left_q),
                right_quality=float(right_q),
                gap_months=gap,
                alpha_min=bridge_alpha_min,
                alpha_max=bridge_alpha_max,
                gap_scale_months=bridge_gap_scale_months,
            )
            bridge = np.linspace(float(left_val), float(right_val), gap + 1, dtype=np.float32)
            current = out[left_obs_idx : right_obs_idx + 1, grid_idx].astype(np.float32)
            corrected = ((1.0 - alpha) * current + alpha * bridge).astype(np.float32)
            corrected[0] = np.float32(left_val)
            corrected[-1] = np.float32(right_val)
            out[left_obs_idx : right_obs_idx + 1, grid_idx] = corrected
            records.append(
                {
                    "grid_index": int(grid_idx),
                    "n_observed_months": n_obs,
                    "mode": "multi_obs_bridge",
                    "month_idx": int(left_obs_idx),
                    "right_month_idx": int(right_obs_idx),
                    "anchor_alpha": float(alpha),
                    "obs_value_m": float(left_val),
                    "right_obs_value_m": float(right_val),
                    "baseline_value_m": float(current[0]),
                    "gap_months": gap,
                }
            )

        out[np.asarray(obs_months, dtype=np.int64), grid_idx] = obs_series

    return out.astype(np.float32), pd.DataFrame(records)


def apply_monthly_residual_diffusion(
    reconstructed: np.ndarray,
    obs_values: np.ndarray,
    obs_mask: np.ndarray,
    obs_quality: np.ndarray,
    laplacian,
    *,
    smooth_lambda: float = 1.0,
    gain: float = 0.70,
    cap_m: float = 2.50,
    min_obs: int = 20,
    exact_at_obs: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    out = np.asarray(reconstructed, dtype=np.float32).copy()
    n_month, n_grid = out.shape
    rows: list[dict[str, Any]] = []
    gain = float(np.clip(gain, 0.0, 1.0))
    cap_m = max(float(cap_m), 0.0)

    for month_idx in range(n_month):
        month_mask = np.asarray(obs_mask[month_idx], dtype=bool)
        n_obs = int(month_mask.sum())
        if n_obs < int(min_obs):
            continue

        pre_residual = np.asarray(obs_values[month_idx, month_mask] - out[month_idx, month_mask], dtype=np.float32)
        if pre_residual.size == 0:
            continue

        residual_values = np.full(n_grid, np.nan, dtype=np.float32)
        residual_quality = np.zeros(n_grid, dtype=np.float32)
        residual_values[month_mask] = pre_residual
        residual_quality[month_mask] = np.maximum(np.asarray(obs_quality[month_idx, month_mask], dtype=np.float32), 1.0)

        residual_field = interpolate_background_field(
            laplacian=laplacian,
            observed_values=residual_values,
            observed_quality=residual_quality,
            smooth_lambda=float(smooth_lambda),
            exact_mask=month_mask,
            min_quality=0.0,
        )
        if cap_m > 0.0:
            residual_field = np.clip(residual_field, -cap_m, cap_m).astype(np.float32)

        correction = gain * residual_field
        out[month_idx] = (out[month_idx] + correction).astype(np.float32)
        if exact_at_obs:
            out[month_idx, month_mask] = obs_values[month_idx, month_mask]

        post_residual = np.asarray(obs_values[month_idx, month_mask] - out[month_idx, month_mask], dtype=np.float32)
        rows.append(
            {
                "month_idx": int(month_idx),
                "observation_count": n_obs,
                "pre_rmse_m": float(np.sqrt(np.mean(np.square(pre_residual, dtype=np.float64)))),
                "post_rmse_m": float(np.sqrt(np.mean(np.square(post_residual, dtype=np.float64)))),
                "pre_bias_m": float(np.mean(pre_residual, dtype=np.float64)),
                "post_bias_m": float(np.mean(post_residual, dtype=np.float64)),
                "mean_abs_correction_m": float(np.mean(np.abs(correction), dtype=np.float64)),
                "p95_abs_correction_m": float(np.quantile(np.abs(correction), 0.95)),
            }
        )

    return out.astype(np.float32), pd.DataFrame(rows)


@dataclass
class SpecialistPredictor:
    horizon: int
    checkpoint_path: Path
    config: dict[str, Any]
    experiment_name: str
    scalers: dict[str, Any]
    model: torch.nn.Module
    device: str
    amp_enabled: bool
    active_grid: pd.DataFrame
    grid_ids: np.ndarray
    month_labels: np.ndarray
    dynamic_data: np.ndarray
    dynamic_feature_names: list[str]
    aem_profile: torch.Tensor
    topography_features: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    root_mask: torch.Tensor
    target_scaling_mode: str
    max_horizon: int

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "SpecialistPredictor":
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = copy.deepcopy(checkpoint["config"])
        experiment_name = str(checkpoint.get("experiment_name", "aem_gnn"))
        scalers = checkpoint["scalers"]
        state_dict = checkpoint["model_state_dict"]

        # Reconstruction must reproduce the exact architecture stored in the
        # checkpoint. Some trained horizons used residual GCN projection and/or
        # raw topography concatenation; infer those modes from the weight keys
        # instead of changing the saved checkpoint or training notebooks.
        model_cfg = config.setdefault("model", {})
        features_cfg = config.setdefault("features", {})
        if any(key.startswith("gnn.residual_proj.") for key in state_dict):
            model_cfg["use_gnn_residual"] = True
        if bool(features_cfg.get("use_topography", False)):
            has_encoded_topography = any(key.startswith("topography_encoder.") for key in state_dict)
            model_cfg["topography_mode"] = "encoded" if has_encoded_topography else "raw"

        active_grid = load_active_grid(config)
        dynamic_cache = load_or_build_dynamic_cache(config)
        aem_cache = load_or_build_aem_cache(config)
        topo_cache = load_topography_cache(config)
        bundle = SimpleNamespace(aem_cache=aem_cache, topo_cache=topo_cache)
        model = _build_model(config, bundle, scalers)
        model.load_state_dict(state_dict)
        device = device_from_config()
        model = model.to(device)
        model.eval()

        graph_type = str(config["graph"]["graph_type"])
        graph_path = output_paths(config, experiment_name)["graph_dir"] / f"{graph_type}_graph.pt"
        graph_bundle = torch.load(graph_path, map_location="cpu", weights_only=False)

        grid_ids = np.asarray(active_grid["grid_id"].to_numpy(dtype=np.int64))
        if not np.array_equal(grid_ids, np.asarray(dynamic_cache["grid_ids"], dtype=np.int64)):
            raise ValueError("Active-grid order does not match dynamic cache grid order.")
        if not np.array_equal(grid_ids, np.asarray(aem_cache["grid_ids"], dtype=np.int64)):
            raise ValueError("Active-grid order does not match AEM cache grid order.")
        if not np.array_equal(grid_ids, np.asarray(topo_cache["grid_ids"], dtype=np.int64)):
            raise ValueError("Active-grid order does not match topography cache grid order.")
        if not np.array_equal(grid_ids, np.asarray(graph_bundle["grid_ids"], dtype=np.int64)):
            raise ValueError("Active-grid order does not match graph grid order.")

        aem_profile = torch.tensor(
            _standardize(
                np.asarray(aem_cache["profiles"], dtype=np.float32),
                np.asarray(scalers["aem_profile_mean"], dtype=np.float32),
                np.asarray(scalers["aem_profile_std"], dtype=np.float32),
            ),
            dtype=torch.float32,
            device=device,
        )
        topo_mean = np.asarray(
            scalers.get("topography_mean", np.zeros((topo_cache["features"].shape[1],), dtype=np.float32)),
            dtype=np.float32,
        )
        topo_std = np.asarray(
            scalers.get("topography_std", np.ones((topo_cache["features"].shape[1],), dtype=np.float32)),
            dtype=np.float32,
        )
        topography_features = torch.tensor(
            _standardize(
                np.asarray(topo_cache["features"], dtype=np.float32),
                topo_mean,
                topo_std,
            ),
            dtype=torch.float32,
            device=device,
        )
        edge_index = graph_bundle["edge_index"].to(device)
        edge_weight = graph_bundle["edge_weight"].to(device)
        root_mask = torch.ones(len(grid_ids), dtype=torch.bool, device=device)

        return cls(
            horizon=int(config["data"]["horizons"][0]),
            checkpoint_path=checkpoint_path,
            config=config,
            experiment_name=experiment_name,
            scalers=scalers,
            model=model,
            device=device,
            amp_enabled=bool(config["training"].get("use_amp", True)) and str(device).startswith("cuda"),
            active_grid=active_grid,
            grid_ids=grid_ids,
            month_labels=np.asarray(dynamic_cache["month_labels"]),
            dynamic_data=np.asarray(dynamic_cache["data"], dtype=np.float32),
            dynamic_feature_names=list(dynamic_cache["feature_names"]),
            aem_profile=aem_profile,
            topography_features=topography_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            root_mask=root_mask,
            target_scaling_mode=_target_scaling_mode(config),
            max_horizon=int(max(int(v) for v in config["data"]["horizons"])),
        )

    def validation_metrics(self) -> dict[str, float]:
        metrics_path = output_paths(self.config, self.experiment_name)["metric_dir"] / f"{self.experiment_name}_overall_metrics.csv"
        frame = pd.read_csv(metrics_path)
        row = frame.loc[frame["split"] == "validation"].iloc[0]
        return {
            "pearson_r": float(row["pearson_r"]) if "pearson_r" in row.index else float("nan"),
            "rmse": float(row["rmse"]),
            "mae": float(row["mae"]),
            "nse": float(row["nse"]) if "nse" in row.index else float(row["r2"]),
            "bias": float(row["bias"]),
        }

    def conformal_qhat(self, level: float = 0.95) -> float | None:
        path = output_paths(self.config, self.experiment_name)["output_root"] / "conformal" / f"{self.experiment_name}_split_conformal_calibration.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        match = frame.loc[np.isclose(frame["level"], float(level))]
        if match.empty:
            return None
        return float(match.iloc[0]["qhat_m"])

    def predict_delta_full(self, current_wtd_m_bls: np.ndarray, start_month_idx: int) -> np.ndarray:
        horizon = int(self.horizon)
        if start_month_idx < 0 or start_month_idx + horizon >= len(self.month_labels):
            raise IndexError(
                f"Cannot predict horizon {horizon} from start_month_idx={start_month_idx}; "
                f"valid range is 0..{len(self.month_labels) - horizon - 1}."
            )
        current_wtd = np.asarray(current_wtd_m_bls, dtype=np.float32).reshape(-1)
        if current_wtd.shape[0] != self.grid_ids.shape[0]:
            raise ValueError("Current WTD vector length does not match active-grid length.")

        sum_net_idx = self.dynamic_feature_names.index("monthly_sum_net_infiltration")
        pump_idx = self.dynamic_feature_names.index(
            pumping_raw_dynamic_feature_name(
                self.config,
                time_unit="month",
                for_log_input=bool(self.config["features"].get("use_log1p_pumping", False)),
            )
        )
        seq_cols = ["monthly_sum_net_infiltration"]
        seq_cols.append(
            pumping_log_dynamic_feature_name(self.config, time_unit="month")
            if bool(self.config["features"].get("use_log1p_pumping", False))
            else pumping_raw_dynamic_feature_name(self.config, time_unit="month", for_log_input=False)
        )
        seq_indices = [self.dynamic_feature_names.index(col) for col in seq_cols]

        future = self.dynamic_data[:, start_month_idx + 1 : start_month_idx + 1 + horizon, :]
        forcing_summary = np.column_stack(
            [
                np.sum(future[:, :, sum_net_idx], axis=1),
                np.log1p(np.clip(np.sum(future[:, :, pump_idx], axis=1), a_min=0.0, a_max=None))
                if bool(self.config["features"].get("use_log1p_pumping", False))
                else np.sum(future[:, :, pump_idx], axis=1),
            ]
        ).astype(np.float32)
        forcing_summary = _standardize(
            forcing_summary,
            np.asarray(self.scalers["forcing_mean"], dtype=np.float32),
            np.asarray(self.scalers["forcing_std"], dtype=np.float32),
        )

        seq_values = future[:, :, seq_indices].astype(np.float32)
        seq_values = _standardize(
            seq_values,
            np.asarray(self.scalers["forcing_sequence_mean"], dtype=np.float32),
            np.asarray(self.scalers["forcing_sequence_std"], dtype=np.float32),
        )
        forcing_sequence = np.zeros((len(self.grid_ids), self.max_horizon, len(seq_indices)), dtype=np.float32)
        forcing_sequence[:, : seq_values.shape[1], :] = seq_values
        forcing_sequence_length = np.full(len(self.grid_ids), seq_values.shape[1], dtype=np.int64)

        month_ts = pd.Timestamp(self.month_labels[start_month_idx])
        month_sin, month_cos = month_sin_cos(month_ts)
        horizon_features = np.asarray([float(horizon), float(month_sin), float(month_cos)], dtype=np.float32)
        horizon_features = _standardize(
            horizon_features,
            np.asarray(self.scalers["horizon_mean"], dtype=np.float32),
            np.asarray(self.scalers["horizon_std"], dtype=np.float32),
        )
        horizon_features = np.broadcast_to(horizon_features[None, :], (len(self.grid_ids), 3)).copy()

        root_wtd = _standardize(
            current_wtd[:, None],
            np.asarray(self.scalers["root_wtd_mean"], dtype=np.float32),
            np.asarray(self.scalers["root_wtd_std"], dtype=np.float32),
        )

        data = Data(
            aem_profile=self.aem_profile,
            topography_features=self.topography_features,
            horizon_features=torch.tensor(horizon_features, dtype=torch.float32, device=self.device),
            forcing_summary=torch.tensor(forcing_summary, dtype=torch.float32, device=self.device),
            forcing_sequence=torch.tensor(forcing_sequence, dtype=torch.float32, device=self.device),
            forcing_sequence_length=torch.tensor(forcing_sequence_length, dtype=torch.long, device=self.device),
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            root_mask=self.root_mask,
            root_wtd=torch.tensor(root_wtd, dtype=torch.float32, device=self.device),
            num_nodes=len(self.grid_ids),
        )

        autocast_device = "cuda" if str(self.device).startswith("cuda") else "cpu"
        with torch.no_grad(), torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=self.amp_enabled):
            pred_std = self.model(data).detach().cpu().numpy().astype(np.float32)

        mean, std = _target_mean_std_for_horizon(self.scalers, horizon, self.target_scaling_mode)
        pred = pred_std * np.asarray(std, dtype=np.float32).reshape(-1)[0] + np.asarray(mean, dtype=np.float32).reshape(-1)[0]
        return pred.astype(np.float32)


@dataclass
class BaselineAdjustedPredictor:
    base_predictor: SpecialistPredictor
    baseline_wtd_m: np.ndarray

    @property
    def horizon(self) -> int:
        return int(self.base_predictor.horizon)

    @property
    def grid_ids(self) -> np.ndarray:
        return self.base_predictor.grid_ids

    @property
    def active_grid(self) -> pd.DataFrame:
        return self.base_predictor.active_grid

    @property
    def month_labels(self) -> np.ndarray:
        return self.base_predictor.month_labels

    @property
    def edge_index(self) -> torch.Tensor:
        return self.base_predictor.edge_index

    @property
    def edge_weight(self) -> torch.Tensor:
        return self.base_predictor.edge_weight

    def validation_metrics(self) -> dict[str, float]:
        return self.base_predictor.validation_metrics()

    def conformal_qhat(self, level: float = 0.95) -> float | None:
        return self.base_predictor.conformal_qhat(level=level)

    def predict_delta_full(self, current_wtd_anomaly_m: np.ndarray, start_month_idx: int) -> np.ndarray:
        current_wtd_abs = np.asarray(current_wtd_anomaly_m, dtype=np.float32) + np.asarray(self.baseline_wtd_m, dtype=np.float32)
        return self.base_predictor.predict_delta_full(current_wtd_abs, start_month_idx)


def add_harmonic_pseudoobs(
    observed: dict[str, Any],
    *,
    min_real_obs: int = 12,
    pseudo_quality: float = 0.30,
    r2_threshold: float = 0.50,
    decay_scale_months: float = 6.0,
    min_pseudo_quality: float = 0.05,
) -> dict[str, Any]:
    """Enrich the observation matrices with harmonic-model pseudo-observations.

    For each grid with >= min_real_obs real observation months, fit an annual +
    semi-annual harmonic plus linear trend to the observed WTD time series, then
    fill unobserved months with model predictions (quality << real observations).

    This provides the Laplacian background-field interpolation with local
    seasonal estimates at sparsely observed months (e.g. drought trough months
    where only 3-6 % of nodes have real measurements), preventing the background
    field from being pulled toward shallower neighbouring values.

    Real observations are never modified.  Pseudo-obs quality is kept well below
    the >=1.0 floor of real observations, so assimilation always prefers real data.
    The per-grid observation count (grid_observation_counts) is intentionally NOT
    updated, so pseudo-obs never trigger the 'exact' assimilation branch.
    """
    from scipy.optimize import curve_fit  # lazy import

    obs_values  = np.array(observed["values"],  dtype=np.float32)
    obs_quality = np.array(observed["quality"], dtype=np.float32)
    obs_mask    = np.array(observed["mask"],    dtype=bool)
    n_month, n_grid = obs_values.shape
    t = np.arange(n_month, dtype=np.float64)

    def _harmonic(t, c0, c1, s1, c2, s2, trend):
        return (c0 + trend * t
                + c1 * np.cos(2 * np.pi * t / 12) + s1 * np.sin(2 * np.pi * t / 12)
                + c2 * np.cos(4 * np.pi * t / 12) + s2 * np.sin(4 * np.pi * t / 12))

    n_pseudo = 0
    for grid_idx in range(n_grid):
        real_t = np.where(obs_mask[:, grid_idx])[0]
        if len(real_t) < int(min_real_obs):
            continue
        real_v = obs_values[real_t, grid_idx].astype(np.float64)

        try:
            popt, _ = curve_fit(_harmonic, real_t.astype(np.float64), real_v,
                                maxfev=8000, method="trf")
        except Exception:
            continue

        pred_at_obs = _harmonic(real_t.astype(np.float64), *popt)
        ss_res = float(np.sum((real_v - pred_at_obs) ** 2))
        ss_tot = float(np.sum((real_v - real_v.mean()) ** 2))
        r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-6))
        if r2 < float(r2_threshold):
            continue

        unobs_t = np.where(~obs_mask[:, grid_idx])[0]
        if len(unobs_t) == 0:
            continue

        pseudo_vals = _harmonic(unobs_t.astype(np.float64), *popt).astype(np.float32)
        # Quality decays with distance to nearest real observation
        nearest_dist = np.array([
            float(np.min(np.abs(real_t - mt))) for mt in unobs_t
        ], dtype=np.float64)
        decay = np.exp(-nearest_dist / float(decay_scale_months)).astype(np.float32)
        q = np.clip(float(pseudo_quality) * float(r2) * decay, float(min_pseudo_quality), 0.99)

        obs_values[unobs_t, grid_idx]  = pseudo_vals
        obs_quality[unobs_t, grid_idx] = q.astype(np.float32)
        obs_mask[unobs_t, grid_idx]    = True
        n_pseudo += len(unobs_t)

    print(f"[harmonic pseudoobs] added {n_pseudo:,} pseudo-observations "
          f"across {(obs_mask.sum(axis=0) > 0).sum()} grids "
          f"(min_real_obs={min_real_obs}, r2≥{r2_threshold})")

    new_obs = dict(observed)
    new_obs["values"]  = obs_values
    new_obs["quality"] = obs_quality
    new_obs["mask"]    = obs_mask
    new_obs["monthly_observation_counts"] = obs_mask.sum(axis=1).astype(np.int64)
    # grid_observation_counts intentionally kept from original (real obs only),
    # so pseudo-obs never trigger the exact-assimilation branch.
    if "grid_observation_counts" not in new_obs:
        real_mask = np.array(observed["mask"], dtype=bool)
        new_obs["grid_observation_counts"] = real_mask.sum(axis=0).astype(np.int64)
    return new_obs


def build_observation_matrices(
    wtd_monthly_path: str | Path,
    active_grid: pd.DataFrame,
    month_labels: np.ndarray,
) -> dict[str, Any]:
    frame = pd.read_csv(wtd_monthly_path, parse_dates=["month_start"]).rename(
        columns={
            "node_id": "grid_id",
            "x_center": "x",
            "y_center": "y",
            "mean_wtd_m": "wtd_m_bls",
        }
    )
    label_to_idx = {label: idx for idx, label in enumerate(month_labels)}
    grid_to_idx = {int(grid_id): idx for idx, grid_id in enumerate(active_grid["grid_id"].to_numpy(dtype=np.int64))}
    n_month = len(month_labels)
    n_grid = len(active_grid)

    values = np.full((n_month, n_grid), np.nan, dtype=np.float32)
    quality = np.zeros((n_month, n_grid), dtype=np.float32)
    counts = np.zeros(n_month, dtype=np.int64)

    frame["month_label"] = frame["month_start"].dt.strftime("%Y-%m")
    frame = frame.loc[frame["month_label"].isin(label_to_idx)].copy()
    for row in frame.itertuples(index=False):
        month_idx = label_to_idx[str(row.month_label)]
        grid_idx = grid_to_idx.get(int(row.grid_id))
        if grid_idx is None:
            continue
        values[month_idx, grid_idx] = float(row.wtd_m_bls)
        q = float(row.n_days_observed) / max(float(row.n_unique_sites), 1.0)
        quality[month_idx, grid_idx] = max(q, 1.0)
    counts[:] = np.isfinite(values).sum(axis=1)
    return {
        "values": values,
        "quality": quality,
        "mask": np.isfinite(values),
        "monthly_observation_counts": counts,
    }


def load_longterm_baseline_field(
    longterm_csv_path: str | Path,
    active_grid: pd.DataFrame,
    *,
    value_col: str = "wtd_model_mean_m",
) -> dict[str, Any]:
    frame = pd.read_csv(longterm_csv_path)
    required = {"cell_id", value_col, "fill_source"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Long-term baseline file missing required columns: {sorted(missing)}")

    merged = active_grid.merge(
        frame[["cell_id", value_col, "fill_source", "has_direct_footprint_cell", "n_footprint_points"]],
        on="cell_id",
        how="left",
    )
    if merged[value_col].isna().any():
        missing_n = int(merged[value_col].isna().sum())
        raise ValueError(f"Long-term baseline does not cover {missing_n} active-grid cells.")

    fill_source = merged["fill_source"].fillna("unknown").astype(str)
    quality = np.select(
        [
            fill_source.eq("direct").to_numpy(),
            fill_source.eq("linear").to_numpy(),
            fill_source.eq("nearest").to_numpy(),
        ],
        [0.75, 0.35, 0.15],
        default=0.25,
    ).astype(np.float32)
    n_points = merged.get("n_footprint_points", pd.Series(np.zeros(len(merged), dtype=float))).fillna(0.0).to_numpy(dtype=np.float32)
    direct_bonus = np.clip(np.log1p(n_points) / np.log(6.0), 0.0, 1.0)
    quality = np.clip(quality + 0.10 * direct_bonus, 0.05, 1.0).astype(np.float32)

    return {
        "values": merged[value_col].to_numpy(dtype=np.float32),
        "quality": quality,
        "fill_source": fill_source.to_numpy(dtype=object),
        "has_direct_footprint_cell": merged.get("has_direct_footprint_cell", pd.Series(False, index=merged.index)).fillna(False).to_numpy(dtype=bool),
        "n_footprint_points": n_points,
        "value_col": value_col,
    }


def build_graph_laplacian(graph_grid_ids: np.ndarray, edge_index: torch.Tensor, edge_weight: torch.Tensor):
    n_nodes = int(len(graph_grid_ids))
    row = edge_index[0].detach().cpu().numpy().astype(np.int64)
    col = edge_index[1].detach().cpu().numpy().astype(np.int64)
    weight = edge_weight.detach().cpu().numpy().astype(np.float64)
    adjacency = coo_matrix((weight, (row, col)), shape=(n_nodes, n_nodes)).tocsr()
    adjacency = 0.5 * (adjacency + adjacency.T)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    laplacian = diags(degree) - adjacency
    return laplacian.tocsr()


def interpolate_background_field(
    laplacian,
    observed_values: np.ndarray,
    observed_quality: np.ndarray,
    smooth_lambda: float = 1.0,
    exact_mask: np.ndarray | None = None,
    min_quality: float = 1.0,
) -> np.ndarray:
    observed_values = np.asarray(observed_values, dtype=np.float64)
    observed_quality = np.asarray(observed_quality, dtype=np.float64)
    mask = np.isfinite(observed_values)
    if not np.any(mask):
        raise ValueError("Cannot build background field for month with zero observations.")

    n_nodes = observed_values.shape[0]
    obs_vector = np.where(mask, observed_values, 0.0)
    weights = np.where(mask, np.maximum(observed_quality, float(min_quality)), 0.0)
    system = diags(weights) + float(smooth_lambda) * laplacian
    rhs = weights * obs_vector

    diag = np.asarray(system.diagonal()).reshape(-1)
    preconditioner = diags(1.0 / np.clip(diag, 1e-6, None))
    x0 = np.full(n_nodes, float(np.nanmean(observed_values[mask])), dtype=np.float64)

    solution, info = cg(system, rhs, x0=x0, M=preconditioner, rtol=1e-6, atol=0.0, maxiter=4000)
    if info != 0:
        solution = spsolve(system, rhs)
    solution = np.asarray(solution, dtype=np.float64).reshape(-1)
    if exact_mask is None:
        exact_mask = mask
    else:
        exact_mask = np.asarray(exact_mask, dtype=bool) & mask
    solution[exact_mask] = observed_values[exact_mask]
    return solution.astype(np.float32)


def compute_horizon_costs(predictors: dict[int, SpecialistPredictor], level: float = 0.95) -> dict[int, float]:
    costs: dict[int, float] = {}
    for horizon, predictor in predictors.items():
        val = predictor.validation_metrics()
        qhat = predictor.conformal_qhat(level=level)
        if qhat is None:
            qhat = float(val["rmse"])
        denom = max(float(val["nse"]), 0.05)
        costs[int(horizon)] = float(qhat) / denom
    return costs


def best_paths_by_gap(max_gap: int, horizon_costs: dict[int, float]) -> dict[int, dict[str, Any]]:
    allowed = sorted(int(h) for h in horizon_costs)
    dp: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
    out: dict[int, dict[str, Any]] = {0: {"cost": 0.0, "path": [], "last_horizon": 0, "prev_gap": None}}
    for gap in range(1, max_gap + 1):
        best_cost = None
        best_path: list[int] | None = None
        best_h = None
        best_prev = None
        for h in allowed:
            prev = gap - h
            if prev < 0:
                continue
            prev_cost, prev_path = dp[prev]
            cand_cost = prev_cost + float(horizon_costs[h])
            cand_path = prev_path + [h]
            if (
                best_cost is None
                or cand_cost < best_cost - 1e-9
                or (abs(cand_cost - best_cost) < 1e-9 and len(cand_path) < len(best_path or []))
            ):
                best_cost = cand_cost
                best_path = cand_path
                best_h = h
                best_prev = prev
        if best_path is None:
            raise ValueError(f"Could not build a path for gap={gap}.")
        dp[gap] = (float(best_cost), list(best_path))
        out[gap] = {
            "cost": float(best_cost),
            "path": list(best_path),
            "last_horizon": int(best_h),
            "prev_gap": int(best_prev) if best_prev is not None else None,
        }
    return out


def default_anchor_month_indices(n_month: int) -> list[int]:
    anchors = list(range(0, n_month, 24))
    if anchors[-1] != n_month - 1:
        anchors.append(n_month - 1)
    return anchors


def _choose_split_month_idx(left_idx: int, right_idx: int) -> int | None:
    gap = int(right_idx - left_idx)
    if gap <= 6:
        return None
    if gap >= 18:
        split_idx = left_idx + 12
    elif gap >= 12:
        split_idx = left_idx + 6
    else:
        split_idx = left_idx + gap // 2
    if split_idx <= left_idx or split_idx >= right_idx:
        return None
    return int(split_idx)


def _segment_summary(
    segment_id: int,
    left_idx: int,
    right_idx: int,
    month_labels: np.ndarray,
    gap_paths: dict[int, dict[str, Any]],
    right_anchor: np.ndarray,
    propagated_right: np.ndarray,
    level: int,
    was_split: bool,
    effective_alpha: np.ndarray | None = None,
    effective_right_anchor: np.ndarray | None = None,
) -> dict[str, Any]:
    endpoint_diff = (right_anchor - propagated_right).astype(np.float32)
    mismatch_metrics = regression_metrics(right_anchor.astype(np.float64), propagated_right.astype(np.float64))
    gap = int(right_idx - left_idx)
    out = {
        "segment_id": int(segment_id),
        "recursion_level": int(level),
        "left_month_idx": int(left_idx),
        "left_month_label": str(month_labels[left_idx]),
        "right_month_idx": int(right_idx),
        "right_month_label": str(month_labels[right_idx]),
        "gap_months": gap,
        "right_endpoint_path": " + ".join(str(v) for v in gap_paths[gap]["path"]),
        "right_endpoint_path_cost": float(gap_paths[gap]["cost"]),
        "right_endpoint_rmse_before_correction_m": float(mismatch_metrics["rmse"]),
        "right_endpoint_mae_before_correction_m": float(mismatch_metrics["mae"]),
        "right_endpoint_bias_before_correction_m": float(mismatch_metrics["bias"]),
        "right_endpoint_p95_abs_diff_m": float(np.quantile(np.abs(endpoint_diff), 0.95)),
        "was_split": bool(was_split),
    }
    if effective_alpha is not None and effective_right_anchor is not None:
        eff_diff = (np.asarray(effective_right_anchor, dtype=np.float32) - propagated_right.astype(np.float32)).astype(np.float32)
        eff_metrics = regression_metrics(
            np.asarray(effective_right_anchor, dtype=np.float64),
            propagated_right.astype(np.float64),
        )
        out.update(
            {
                "soft_anchor_alpha_mean": float(np.mean(effective_alpha)),
                "soft_anchor_alpha_p05": float(np.quantile(effective_alpha, 0.05)),
                "soft_anchor_alpha_p50": float(np.quantile(effective_alpha, 0.50)),
                "soft_anchor_alpha_p95": float(np.quantile(effective_alpha, 0.95)),
                "effective_endpoint_rmse_before_correction_m": float(eff_metrics["rmse"]),
                "effective_endpoint_mae_before_correction_m": float(eff_metrics["mae"]),
                "effective_endpoint_bias_before_correction_m": float(eff_metrics["bias"]),
                "effective_endpoint_p95_abs_diff_m": float(np.quantile(np.abs(eff_diff), 0.95)),
            }
        )
    return out


def _reconstruct_segment_recursive(
    *,
    left_idx: int,
    right_idx: int,
    left_state: np.ndarray,
    get_anchor_state,
    predictors: dict[int, SpecialistPredictor],
    gap_paths: dict[int, dict[str, Any]],
    obs_values: np.ndarray,
    obs_quality: np.ndarray,
    obs_mask: np.ndarray,
    month_labels: np.ndarray,
    split_rmse_threshold_m: float,
    split_p95_threshold_m: float,
    level: int,
    next_segment_id: list[int],
    monthly_observation_counts: np.ndarray | None = None,
    soft_anchor: bool = False,
    soft_anchor_obs_ref_count: float = 100.0,
    soft_anchor_alpha_min: float = 0.10,
    soft_anchor_alpha_max: float = 0.80,
    soft_anchor_mismatch_scale_m: float = 3.0,
    direct_obs_exact_min_count: int = 2,
    direct_obs_single_alpha_max: float = 0.12,
    direct_obs_single_mismatch_scale_m: float = 6.0,
    grid_observation_counts: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    gap = int(right_idx - left_idx)
    propagated: list[np.ndarray] = [left_state.copy()]

    for local_gap in range(1, gap + 1):
        plan = gap_paths[local_gap]
        h = int(plan["last_horizon"])
        prev_gap = int(plan["prev_gap"])
        start_idx = left_idx + prev_gap
        start_state = propagated[prev_gap]
        delta_pred = predictors[h].predict_delta_full(start_state, start_idx)
        next_state = (start_state - delta_pred).astype(np.float32)
        month_idx = left_idx + local_gap
        if obs_mask[month_idx].any():
            next_state = _assimilate_observations_into_state(
                next_state,
                month_idx=month_idx,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_quality=obs_quality,
                grid_observation_counts=np.zeros(obs_mask.shape[1], dtype=np.int64) if grid_observation_counts is None else grid_observation_counts,
                exact_obs_min_count=direct_obs_exact_min_count,
                single_obs_alpha_max=direct_obs_single_alpha_max,
                single_obs_mismatch_scale_m=direct_obs_single_mismatch_scale_m,
            )
        propagated.append(next_state)

    right_anchor = get_anchor_state(right_idx).copy()
    if obs_mask[right_idx].any():
        right_anchor = _assimilate_observations_into_state(
            right_anchor,
            month_idx=right_idx,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_quality=obs_quality,
            grid_observation_counts=np.zeros(obs_mask.shape[1], dtype=np.int64) if grid_observation_counts is None else grid_observation_counts,
            exact_obs_min_count=direct_obs_exact_min_count,
            single_obs_alpha_max=direct_obs_single_alpha_max,
            single_obs_mismatch_scale_m=direct_obs_single_mismatch_scale_m,
        )

    effective_alpha = None
    effective_right_anchor = None
    if soft_anchor:
        month_obs_count = int(monthly_observation_counts[right_idx]) if monthly_observation_counts is not None else 0
        effective_alpha = _soft_anchor_alpha(
            month_observation_count=month_obs_count,
            abs_mismatch=np.abs(right_anchor - propagated[gap]),
            obs_ref_count=soft_anchor_obs_ref_count,
            alpha_min=soft_anchor_alpha_min,
            alpha_max=soft_anchor_alpha_max,
            mismatch_scale_m=soft_anchor_mismatch_scale_m,
        )
        if obs_mask[right_idx].any() and grid_observation_counts is not None:
            effective_alpha = effective_alpha.astype(np.float32)
            exact_mask = np.asarray(grid_observation_counts >= int(direct_obs_exact_min_count), dtype=bool) & np.asarray(obs_mask[right_idx], dtype=bool)
            effective_alpha[exact_mask] = 1.0
        effective_right_anchor = (propagated[gap] + effective_alpha * (right_anchor - propagated[gap])).astype(np.float32)
        if obs_mask[right_idx].any() and grid_observation_counts is not None:
            exact_mask = np.asarray(grid_observation_counts >= int(direct_obs_exact_min_count), dtype=bool) & np.asarray(obs_mask[right_idx], dtype=bool)
            effective_right_anchor[exact_mask] = obs_values[right_idx, exact_mask]
    else:
        effective_right_anchor = right_anchor.copy()

    summary = _segment_summary(
        segment_id=next_segment_id[0],
        left_idx=left_idx,
        right_idx=right_idx,
        month_labels=month_labels,
        gap_paths=gap_paths,
        right_anchor=right_anchor,
        propagated_right=propagated[gap],
        level=level,
        was_split=False,
        effective_alpha=effective_alpha,
        effective_right_anchor=effective_right_anchor,
    )

    split_idx = _choose_split_month_idx(left_idx, right_idx)
    should_split = (
        split_idx is not None
        and (
            summary["right_endpoint_rmse_before_correction_m"] > float(split_rmse_threshold_m)
            or summary["right_endpoint_p95_abs_diff_m"] > float(split_p95_threshold_m)
        )
    )
    if should_split:
        summary["was_split"] = True
        left_states, left_summaries = _reconstruct_segment_recursive(
            left_idx=left_idx,
            right_idx=int(split_idx),
            left_state=left_state,
            get_anchor_state=get_anchor_state,
            predictors=predictors,
            gap_paths=gap_paths,
            obs_values=obs_values,
            obs_quality=obs_quality,
            obs_mask=obs_mask,
            month_labels=month_labels,
            split_rmse_threshold_m=split_rmse_threshold_m,
            split_p95_threshold_m=split_p95_threshold_m,
            level=level + 1,
            next_segment_id=next_segment_id,
            monthly_observation_counts=monthly_observation_counts,
            soft_anchor=soft_anchor,
            soft_anchor_obs_ref_count=soft_anchor_obs_ref_count,
            soft_anchor_alpha_min=soft_anchor_alpha_min,
            soft_anchor_alpha_max=soft_anchor_alpha_max,
            soft_anchor_mismatch_scale_m=soft_anchor_mismatch_scale_m,
        )
        mid_state = left_states[-1].copy()
        right_states, right_summaries = _reconstruct_segment_recursive(
            left_idx=int(split_idx),
            right_idx=right_idx,
            left_state=mid_state,
            get_anchor_state=get_anchor_state,
            predictors=predictors,
            gap_paths=gap_paths,
            obs_values=obs_values,
            obs_quality=obs_quality,
            obs_mask=obs_mask,
            month_labels=month_labels,
            split_rmse_threshold_m=split_rmse_threshold_m,
            split_p95_threshold_m=split_p95_threshold_m,
            level=level + 1,
            next_segment_id=next_segment_id,
            monthly_observation_counts=monthly_observation_counts,
            soft_anchor=soft_anchor,
            soft_anchor_obs_ref_count=soft_anchor_obs_ref_count,
            soft_anchor_alpha_min=soft_anchor_alpha_min,
            soft_anchor_alpha_max=soft_anchor_alpha_max,
            soft_anchor_mismatch_scale_m=soft_anchor_mismatch_scale_m,
        )
        states = np.concatenate([left_states[:-1], right_states], axis=0)
        return states, [summary, *left_summaries, *right_summaries]

    endpoint_diff = (effective_right_anchor - propagated[gap]).astype(np.float32)
    corrected_states: list[np.ndarray] = []
    for local_gap in range(gap + 1):
        alpha = float((local_gap / gap) ** 3) if gap > 0 else 0.0
        corrected = (propagated[local_gap] + alpha * endpoint_diff).astype(np.float32)
        month_idx = left_idx + local_gap
        if obs_mask[month_idx].any():
            corrected[obs_mask[month_idx]] = obs_values[month_idx, obs_mask[month_idx]]
        corrected_states.append(corrected)

    next_segment_id[0] += 1
    return np.stack(corrected_states, axis=0), [summary]


def reconstruct_monthly_wtd(
    predictors: dict[int, SpecialistPredictor],
    observed: dict[str, Any],
    anchor_month_indices: list[int] | None = None,
    smooth_lambda: float = 1.0,
) -> dict[str, Any]:
    if not predictors:
        raise ValueError("At least one specialist predictor is required.")

    master_h = sorted(predictors)[0]
    master = predictors[master_h]
    month_labels = np.asarray(master.month_labels)
    n_month = len(month_labels)
    n_grid = len(master.grid_ids)
    anchor_month_indices = anchor_month_indices or default_anchor_month_indices(n_month)

    obs_values = np.asarray(observed["values"], dtype=np.float32)
    obs_quality = np.asarray(observed["quality"], dtype=np.float32)
    obs_mask = np.asarray(observed["mask"], dtype=bool)

    laplacian = build_graph_laplacian(master.grid_ids, master.edge_index, master.edge_weight)

    anchor_background = np.zeros((len(anchor_month_indices), n_grid), dtype=np.float32)
    anchor_observation_counts: list[int] = []
    for i, month_idx in enumerate(anchor_month_indices):
        background = interpolate_background_field(
            laplacian=laplacian,
            observed_values=obs_values[month_idx],
            observed_quality=obs_quality[month_idx],
            smooth_lambda=smooth_lambda,
        )
        background[obs_mask[month_idx]] = obs_values[month_idx, obs_mask[month_idx]]
        anchor_background[i] = background
        anchor_observation_counts.append(int(obs_mask[month_idx].sum()))

    horizon_costs = compute_horizon_costs(predictors)
    max_gap = max(anchor_month_indices[i + 1] - anchor_month_indices[i] for i in range(len(anchor_month_indices) - 1))
    gap_paths = best_paths_by_gap(max_gap, horizon_costs)

    reconstructed = np.full((n_month, n_grid), np.nan, dtype=np.float32)
    background_at_anchors = np.full((n_month, n_grid), np.nan, dtype=np.float32)
    is_anchor_month = np.zeros(n_month, dtype=bool)
    interval_id_per_month = np.full(n_month, -1, dtype=np.int32)
    interval_summaries: list[dict[str, Any]] = []

    for anchor_local_idx, month_idx in enumerate(anchor_month_indices):
        background_at_anchors[month_idx] = anchor_background[anchor_local_idx]
        is_anchor_month[month_idx] = True

    for interval_id, (left_idx, right_idx) in enumerate(zip(anchor_month_indices[:-1], anchor_month_indices[1:])):
        gap = int(right_idx - left_idx)
        propagated: list[np.ndarray] = [anchor_background[interval_id].copy()]
        propagated[0][obs_mask[left_idx]] = obs_values[left_idx, obs_mask[left_idx]]

        for local_gap in range(1, gap + 1):
            plan = gap_paths[local_gap]
            h = int(plan["last_horizon"])
            prev_gap = int(plan["prev_gap"])
            start_idx = left_idx + prev_gap
            start_state = propagated[prev_gap]
            delta_pred = predictors[h].predict_delta_full(start_state, start_idx)
            next_state = (start_state - delta_pred).astype(np.float32)
            month_idx = left_idx + local_gap
            if obs_mask[month_idx].any():
                next_state[obs_mask[month_idx]] = obs_values[month_idx, obs_mask[month_idx]]
            propagated.append(next_state)

        right_anchor = anchor_background[interval_id + 1].copy()
        right_anchor[obs_mask[right_idx]] = obs_values[right_idx, obs_mask[right_idx]]
        endpoint_diff = (right_anchor - propagated[gap]).astype(np.float32)
        mismatch_metrics = regression_metrics(right_anchor.astype(np.float64), propagated[gap].astype(np.float64))

        corrected_states: list[np.ndarray] = []
        for local_gap in range(gap + 1):
            alpha = _smoothstep(local_gap / gap) if gap > 0 else 0.0
            corrected = (propagated[local_gap] + alpha * endpoint_diff).astype(np.float32)
            month_idx = left_idx + local_gap
            if obs_mask[month_idx].any():
                corrected = _assimilate_observations_into_state(
                    corrected,
                    month_idx=month_idx,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_quality=obs_quality,
                    grid_observation_counts=np.zeros(obs_mask.shape[1], dtype=np.int64) if grid_observation_counts is None else grid_observation_counts,
                    exact_obs_min_count=direct_obs_exact_min_count,
                    single_obs_alpha_max=direct_obs_single_alpha_max,
                    single_obs_mismatch_scale_m=direct_obs_single_mismatch_scale_m,
                )
            corrected_states.append(corrected)
            if local_gap == 0 and interval_id > 0:
                continue
            reconstructed[month_idx] = corrected
            interval_id_per_month[month_idx] = interval_id

        interval_summaries.append(
            {
                "interval_id": interval_id,
                "left_month_idx": left_idx,
                "left_month_label": str(month_labels[left_idx]),
                "right_month_idx": right_idx,
                "right_month_label": str(month_labels[right_idx]),
                "gap_months": gap,
                "right_endpoint_path": " + ".join(str(v) for v in gap_paths[gap]["path"]),
                "right_endpoint_path_cost": float(gap_paths[gap]["cost"]),
                "right_endpoint_rmse_before_correction_m": float(mismatch_metrics["rmse"]),
                "right_endpoint_mae_before_correction_m": float(mismatch_metrics["mae"]),
                "right_endpoint_bias_before_correction_m": float(mismatch_metrics["bias"]),
                "right_endpoint_p95_abs_diff_m": float(np.quantile(np.abs(endpoint_diff), 0.95)),
            }
        )

    if np.isnan(reconstructed).any():
        missing_months = np.where(np.isnan(reconstructed).all(axis=1))[0].tolist()
        raise RuntimeError(f"Reconstruction left missing monthly states for month indices: {missing_months}")

    reconstructed[obs_mask] = obs_values[obs_mask]
    return {
        "grid_ids": master.grid_ids,
        "active_grid": master.active_grid.copy(),
        "month_labels": month_labels,
        "reconstructed_matrix": reconstructed,
        "anchor_month_indices": anchor_month_indices,
        "anchor_base_month_indices": sorted(int(v) for v in anchor_base_fields_by_month.keys()),
        "anchor_base_field_quality": float(anchor_base_field_quality),
        "anchor_background_matrix": anchor_background,
        "background_at_anchor_months": background_at_anchors,
        "is_anchor_month": is_anchor_month,
        "interval_id_per_month": interval_id_per_month,
        "interval_summary": pd.DataFrame(interval_summaries),
        "horizon_costs": horizon_costs,
        "gap_paths": gap_paths,
        "anchor_observation_counts": anchor_observation_counts,
    }


def reconstruct_monthly_wtd_adaptive(
    predictors: dict[int, SpecialistPredictor],
    observed: dict[str, Any],
    anchor_month_indices: list[int] | None = None,
    anchor_base_fields_by_month: dict[int, np.ndarray] | None = None,
    anchor_base_field_quality: float = 0.15,
    anchor_base_field_quality_by_month: dict[int, np.ndarray] | None = None,
    smooth_lambda: float = 1.0,
    split_rmse_threshold_m: float = 2.0,
    split_p95_threshold_m: float = 3.0,
    soft_anchor: bool = False,
    soft_anchor_obs_ref_count: float = 100.0,
    soft_anchor_alpha_min: float = 0.10,
    soft_anchor_alpha_max: float = 0.80,
    soft_anchor_mismatch_scale_m: float = 3.0,
    direct_obs_exact_min_count: int = 2,
    direct_obs_single_alpha_max: float = 0.12,
    direct_obs_single_mismatch_scale_m: float = 6.0,
    use_local_anchor_priors: bool = False,
    local_anchor_prior_multi_obs_min_count: int = 2,
    local_anchor_prior_extrap_window_months: int = 12,
    local_anchor_prior_single_obs_window_months: int = 3,
    local_anchor_prior_interp_span_scale_months: float = 48.0,
    smooth_local_anchor_priors_into_background: bool = False,
    local_anchor_prior_graph_quality_scale: float = 5.0,
    local_anchor_prior_graph_quality_floor: float = 0.10,
    use_local_observation_anchors: bool = False,
    local_anchor_multi_obs_min_count: int = 2,
    local_anchor_bridge_alpha_min: float = 0.35,
    local_anchor_bridge_alpha_max: float = 0.80,
    local_anchor_bridge_gap_scale_months: float = 12.0,
    local_anchor_edge_window_months: int = 6,
    local_anchor_edge_alpha_max: float = 0.25,
    local_anchor_single_obs_alpha_max: float = 0.20,
    local_anchor_single_obs_mismatch_scale_m: float = 5.0,
    use_monthly_residual_diffusion: bool = False,
    monthly_residual_smooth_lambda: float = 1.0,
    monthly_residual_gain: float = 0.70,
    monthly_residual_cap_m: float = 2.50,
    monthly_residual_min_obs: int = 20,
    monthly_residual_exact_at_obs: bool = True,
    allow_final_extrapolation: bool = False,
) -> dict[str, Any]:
    if not predictors:
        raise ValueError("At least one specialist predictor is required.")

    master_h = sorted(predictors)[0]
    master = predictors[master_h]
    month_labels = np.asarray(master.month_labels)
    n_month = len(month_labels)
    n_grid = len(master.grid_ids)
    anchor_month_indices = anchor_month_indices or default_anchor_month_indices(n_month)
    anchor_base_fields_by_month = anchor_base_fields_by_month or {}
    anchor_base_field_quality_by_month = anchor_base_field_quality_by_month or {}

    obs_values = np.asarray(observed["values"], dtype=np.float32)
    obs_quality = np.asarray(observed["quality"], dtype=np.float32)
    obs_mask = np.asarray(observed["mask"], dtype=bool)
    monthly_observation_counts = np.asarray(observed["monthly_observation_counts"], dtype=np.int64)
    local_anchor_prior_summary = pd.DataFrame()

    laplacian = build_graph_laplacian(master.grid_ids, master.edge_index, master.edge_weight)
    anchor_cache: dict[int, np.ndarray] = {}
    anchor_rank_by_month = {int(month_idx): int(rank) for rank, month_idx in enumerate(anchor_month_indices)}
    anchor_local_prior_values = np.full((len(anchor_month_indices), n_grid), np.nan, dtype=np.float32)
    anchor_local_prior_weights = np.zeros((len(anchor_month_indices), n_grid), dtype=np.float32)

    if use_local_anchor_priors:
        anchor_local_prior_values, anchor_local_prior_weights, local_anchor_prior_summary = build_local_anchor_priors(
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_quality=obs_quality,
            anchor_month_indices=anchor_month_indices,
            multi_obs_min_count=local_anchor_prior_multi_obs_min_count,
            extrap_window_months=local_anchor_prior_extrap_window_months,
            single_obs_window_months=local_anchor_prior_single_obs_window_months,
            interp_span_scale_months=local_anchor_prior_interp_span_scale_months,
        )

    def get_anchor_state(month_idx: int) -> np.ndarray:
        month_idx = int(month_idx)
        cached = anchor_cache.get(month_idx)
        if cached is not None:
            return cached
        anchor_rank = anchor_rank_by_month.get(month_idx)
        base_field = anchor_base_fields_by_month.get(month_idx)
        base_quality = anchor_base_field_quality_by_month.get(month_idx)

        if use_local_anchor_priors and smooth_local_anchor_priors_into_background and anchor_rank is not None:
            if base_field is not None:
                constrained_values = np.asarray(base_field, dtype=np.float32).copy()
                if base_quality is None:
                    constrained_quality = np.full(n_grid, float(anchor_base_field_quality), dtype=np.float32)
                else:
                    constrained_quality = np.asarray(base_quality, dtype=np.float32).copy()
            else:
                constrained_values = np.full(n_grid, np.nan, dtype=np.float32)
                constrained_quality = np.zeros(n_grid, dtype=np.float32)
            exact_mask = np.asarray(obs_mask[month_idx], dtype=bool).copy()

            if np.any(exact_mask):
                constrained_values[exact_mask] = obs_values[month_idx, exact_mask]
                constrained_quality[exact_mask] = np.maximum(obs_quality[month_idx, exact_mask], 1.0)

            weights = anchor_local_prior_weights[anchor_rank]
            valid = (weights > 0.0) & (~exact_mask)
            if np.any(valid):
                constrained_values[valid] = anchor_local_prior_values[anchor_rank, valid]
                constrained_quality[valid] = np.maximum(
                    weights[valid] * float(local_anchor_prior_graph_quality_scale),
                    float(local_anchor_prior_graph_quality_floor),
                )

            background = interpolate_background_field(
                laplacian=laplacian,
                observed_values=constrained_values,
                observed_quality=constrained_quality,
                smooth_lambda=smooth_lambda,
                exact_mask=exact_mask,
                min_quality=0.0,
            )
        else:
            if base_field is not None:
                constrained_values = np.asarray(base_field, dtype=np.float32).copy()
                if base_quality is None:
                    constrained_quality = np.full(n_grid, float(anchor_base_field_quality), dtype=np.float32)
                else:
                    constrained_quality = np.asarray(base_quality, dtype=np.float32).copy()
                exact_mask = np.asarray(obs_mask[month_idx], dtype=bool)
                if np.any(exact_mask):
                    constrained_values[exact_mask] = obs_values[month_idx, exact_mask]
                    constrained_quality[exact_mask] = np.maximum(obs_quality[month_idx, exact_mask], 1.0)
                background = interpolate_background_field(
                    laplacian=laplacian,
                    observed_values=constrained_values,
                    observed_quality=constrained_quality,
                    smooth_lambda=smooth_lambda,
                    exact_mask=exact_mask,
                    min_quality=0.0,
                )
            else:
                background = interpolate_background_field(
                    laplacian=laplacian,
                    observed_values=obs_values[month_idx],
                    observed_quality=obs_quality[month_idx],
                    smooth_lambda=smooth_lambda,
                )
            if obs_mask[month_idx].any():
                background[obs_mask[month_idx]] = obs_values[month_idx, obs_mask[month_idx]]
            if anchor_rank is not None and use_local_anchor_priors:
                weights = anchor_local_prior_weights[anchor_rank]
                valid = (weights > 0.0) & (~obs_mask[month_idx])
                if np.any(valid):
                    background[valid] = (
                        background[valid] + weights[valid] * (anchor_local_prior_values[anchor_rank, valid] - background[valid])
                    ).astype(np.float32)
        anchor_cache[month_idx] = background.astype(np.float32)
        return anchor_cache[month_idx]

    horizon_costs = compute_horizon_costs(predictors)
    max_anchor_gap = max(anchor_month_indices[i + 1] - anchor_month_indices[i] for i in range(len(anchor_month_indices) - 1))
    final_tail_gap = max(0, (n_month - 1) - int(anchor_month_indices[-1]))
    max_gap = max(max_anchor_gap, final_tail_gap if allow_final_extrapolation else 0)
    gap_paths = best_paths_by_gap(max_gap, horizon_costs)

    reconstructed = np.full((n_month, n_grid), np.nan, dtype=np.float32)
    background_at_anchors = np.full((n_month, n_grid), np.nan, dtype=np.float32)
    is_anchor_month = np.zeros(n_month, dtype=bool)
    interval_id_per_month = np.full(n_month, -1, dtype=np.int32)
    interval_summaries: list[dict[str, Any]] = []

    for month_idx in anchor_month_indices:
        background_at_anchors[month_idx] = get_anchor_state(month_idx)
        is_anchor_month[month_idx] = True

    next_segment_id = [0]
    for root_interval_id, (left_idx, right_idx) in enumerate(zip(anchor_month_indices[:-1], anchor_month_indices[1:])):
        left_state = get_anchor_state(left_idx).copy()
        states, summaries = _reconstruct_segment_recursive(
            left_idx=left_idx,
            right_idx=right_idx,
            left_state=left_state,
            get_anchor_state=get_anchor_state,
            predictors=predictors,
            gap_paths=gap_paths,
            obs_values=obs_values,
            obs_quality=obs_quality,
            obs_mask=obs_mask,
            month_labels=month_labels,
            split_rmse_threshold_m=split_rmse_threshold_m,
            split_p95_threshold_m=split_p95_threshold_m,
            level=0,
            next_segment_id=next_segment_id,
            monthly_observation_counts=monthly_observation_counts,
            soft_anchor=soft_anchor,
            soft_anchor_obs_ref_count=soft_anchor_obs_ref_count,
            soft_anchor_alpha_min=soft_anchor_alpha_min,
            soft_anchor_alpha_max=soft_anchor_alpha_max,
            soft_anchor_mismatch_scale_m=soft_anchor_mismatch_scale_m,
            direct_obs_exact_min_count=direct_obs_exact_min_count,
            direct_obs_single_alpha_max=direct_obs_single_alpha_max,
            direct_obs_single_mismatch_scale_m=direct_obs_single_mismatch_scale_m,
            grid_observation_counts=obs_mask.sum(axis=0).astype(np.int64),
        )
        for local_idx in range(states.shape[0]):
            month_idx = left_idx + local_idx
            if local_idx == 0 and root_interval_id > 0:
                continue
            reconstructed[month_idx] = states[local_idx]
            interval_id_per_month[month_idx] = root_interval_id
        for summary in summaries:
            summary["root_interval_id"] = int(root_interval_id)
        interval_summaries.extend(summaries)

    if allow_final_extrapolation and final_tail_gap > 0:
        left_idx = int(anchor_month_indices[-1])
        propagated: list[np.ndarray] = [get_anchor_state(left_idx).copy()]
        grid_observation_counts = obs_mask.sum(axis=0).astype(np.int64)
        for local_gap in range(1, final_tail_gap + 1):
            plan = gap_paths[local_gap]
            h = int(plan["last_horizon"])
            prev_gap = int(plan["prev_gap"])
            start_idx = left_idx + prev_gap
            start_state = propagated[prev_gap]
            delta_pred = predictors[h].predict_delta_full(start_state, start_idx)
            next_state = (start_state - delta_pred).astype(np.float32)
            month_idx = left_idx + local_gap
            if obs_mask[month_idx].any():
                next_state = _assimilate_observations_into_state(
                    next_state,
                    month_idx=month_idx,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_quality=obs_quality,
                    grid_observation_counts=grid_observation_counts,
                    exact_obs_min_count=direct_obs_exact_min_count,
                    single_obs_alpha_max=direct_obs_single_alpha_max,
                    single_obs_mismatch_scale_m=direct_obs_single_mismatch_scale_m,
                )
            propagated.append(next_state)

        tail_interval_id = max(0, len(anchor_month_indices) - 1)
        for local_idx, state in enumerate(propagated):
            month_idx = left_idx + local_idx
            if local_idx == 0 and not np.isnan(reconstructed[month_idx]).all():
                continue
            reconstructed[month_idx] = state.astype(np.float32)
            interval_id_per_month[month_idx] = tail_interval_id

        interval_summaries.append(
            {
                "segment_id": int(next_segment_id[0]),
                "recursion_level": 0,
                "left_month_idx": int(left_idx),
                "left_month_label": str(month_labels[left_idx]),
                "right_month_idx": int(n_month - 1),
                "right_month_label": str(month_labels[n_month - 1]),
                "gap_months": int(final_tail_gap),
                "right_endpoint_path": " + ".join(str(v) for v in gap_paths[final_tail_gap]["path"]),
                "right_endpoint_path_cost": float(gap_paths[final_tail_gap]["cost"]),
                "right_endpoint_rmse_before_correction_m": np.nan,
                "right_endpoint_mae_before_correction_m": np.nan,
                "right_endpoint_bias_before_correction_m": np.nan,
                "right_endpoint_p95_abs_diff_m": np.nan,
                "was_split": False,
                "was_final_extrapolation": True,
                "root_interval_id": int(tail_interval_id),
            }
        )
        next_segment_id[0] += 1

    if np.isnan(reconstructed).any():
        missing_months = np.where(np.isnan(reconstructed).all(axis=1))[0].tolist()
        raise RuntimeError(f"Adaptive reconstruction left missing monthly states for month indices: {missing_months}")

    local_anchor_summary = pd.DataFrame()
    if use_local_observation_anchors:
        reconstructed, local_anchor_summary = apply_local_observation_anchors(
            reconstructed=reconstructed,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_quality=obs_quality,
            multi_obs_min_count=local_anchor_multi_obs_min_count,
            bridge_alpha_min=local_anchor_bridge_alpha_min,
            bridge_alpha_max=local_anchor_bridge_alpha_max,
            bridge_gap_scale_months=local_anchor_bridge_gap_scale_months,
            edge_window_months=local_anchor_edge_window_months,
            edge_alpha_max=local_anchor_edge_alpha_max,
            single_obs_alpha_max=local_anchor_single_obs_alpha_max,
            single_obs_mismatch_scale_m=local_anchor_single_obs_mismatch_scale_m,
        )
    else:
        reconstructed[obs_mask] = obs_values[obs_mask]

    monthly_residual_diffusion_summary = pd.DataFrame()
    if use_monthly_residual_diffusion:
        reconstructed, monthly_residual_diffusion_summary = apply_monthly_residual_diffusion(
            reconstructed=reconstructed,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_quality=obs_quality,
            laplacian=laplacian,
            smooth_lambda=monthly_residual_smooth_lambda,
            gain=monthly_residual_gain,
            cap_m=monthly_residual_cap_m,
            min_obs=monthly_residual_min_obs,
            exact_at_obs=monthly_residual_exact_at_obs,
        )

    anchor_observation_counts = [int(obs_mask[idx].sum()) for idx in anchor_month_indices]
    return {
        "grid_ids": master.grid_ids,
        "active_grid": master.active_grid.copy(),
        "month_labels": month_labels,
        "reconstructed_matrix": reconstructed,
        "anchor_month_indices": anchor_month_indices,
        "anchor_background_matrix": np.stack([get_anchor_state(idx) for idx in anchor_month_indices], axis=0),
        "background_at_anchor_months": background_at_anchors,
        "is_anchor_month": is_anchor_month,
        "interval_id_per_month": interval_id_per_month,
        "interval_summary": pd.DataFrame(interval_summaries),
        "horizon_costs": horizon_costs,
        "gap_paths": gap_paths,
        "anchor_observation_counts": anchor_observation_counts,
        "split_rmse_threshold_m": float(split_rmse_threshold_m),
        "split_p95_threshold_m": float(split_p95_threshold_m),
        "soft_anchor": bool(soft_anchor),
        "soft_anchor_obs_ref_count": float(soft_anchor_obs_ref_count),
        "soft_anchor_alpha_min": float(soft_anchor_alpha_min),
        "soft_anchor_alpha_max": float(soft_anchor_alpha_max),
        "soft_anchor_mismatch_scale_m": float(soft_anchor_mismatch_scale_m),
        "direct_obs_exact_min_count": int(direct_obs_exact_min_count),
        "direct_obs_single_alpha_max": float(direct_obs_single_alpha_max),
        "direct_obs_single_mismatch_scale_m": float(direct_obs_single_mismatch_scale_m),
        "anchor_base_month_indices": [int(v) for v in sorted(anchor_base_fields_by_month.keys())],
        "anchor_base_field_quality": float(anchor_base_field_quality),
        "anchor_base_field_quality_by_month": {
            int(k): {
                "mean": float(np.mean(np.asarray(v, dtype=np.float32))),
                "median": float(np.median(np.asarray(v, dtype=np.float32))),
                "min": float(np.min(np.asarray(v, dtype=np.float32))),
                "max": float(np.max(np.asarray(v, dtype=np.float32))),
            }
            for k, v in anchor_base_field_quality_by_month.items()
        },
        "use_local_anchor_priors": bool(use_local_anchor_priors),
        "local_anchor_prior_multi_obs_min_count": int(local_anchor_prior_multi_obs_min_count),
        "local_anchor_prior_extrap_window_months": int(local_anchor_prior_extrap_window_months),
        "local_anchor_prior_single_obs_window_months": int(local_anchor_prior_single_obs_window_months),
        "local_anchor_prior_interp_span_scale_months": float(local_anchor_prior_interp_span_scale_months),
        "smooth_local_anchor_priors_into_background": bool(smooth_local_anchor_priors_into_background),
        "local_anchor_prior_graph_quality_scale": float(local_anchor_prior_graph_quality_scale),
        "local_anchor_prior_graph_quality_floor": float(local_anchor_prior_graph_quality_floor),
        "local_anchor_prior_summary": local_anchor_prior_summary,
        "use_local_observation_anchors": bool(use_local_observation_anchors),
        "local_anchor_multi_obs_min_count": int(local_anchor_multi_obs_min_count),
        "local_anchor_bridge_alpha_min": float(local_anchor_bridge_alpha_min),
        "local_anchor_bridge_alpha_max": float(local_anchor_bridge_alpha_max),
        "local_anchor_bridge_gap_scale_months": float(local_anchor_bridge_gap_scale_months),
        "local_anchor_edge_window_months": int(local_anchor_edge_window_months),
        "local_anchor_edge_alpha_max": float(local_anchor_edge_alpha_max),
        "local_anchor_single_obs_alpha_max": float(local_anchor_single_obs_alpha_max),
        "local_anchor_single_obs_mismatch_scale_m": float(local_anchor_single_obs_mismatch_scale_m),
        "local_anchor_summary": local_anchor_summary,
        "use_monthly_residual_diffusion": bool(use_monthly_residual_diffusion),
        "monthly_residual_smooth_lambda": float(monthly_residual_smooth_lambda),
        "monthly_residual_gain": float(monthly_residual_gain),
        "monthly_residual_cap_m": float(monthly_residual_cap_m),
        "monthly_residual_min_obs": int(monthly_residual_min_obs),
        "monthly_residual_exact_at_obs": bool(monthly_residual_exact_at_obs),
        "allow_final_extrapolation": bool(allow_final_extrapolation),
        "monthly_residual_diffusion_summary": monthly_residual_diffusion_summary,
    }


def save_reconstruction_outputs(
    reconstruction: dict[str, Any],
    observed: dict[str, Any],
    out_dir: str | Path,
) -> dict[str, Path]:
    out_dir = ensure_dir(out_dir)
    recon_dir = ensure_dir(out_dir / "reconstruction")
    meta_dir = ensure_dir(out_dir / "metadata")

    active_grid = reconstruction["active_grid"].copy()
    month_labels = np.asarray(reconstruction["month_labels"])
    reconstructed = np.asarray(reconstruction["reconstructed_matrix"], dtype=np.float32)
    reconstructed_absolute = np.asarray(reconstruction.get("reconstructed_absolute_matrix", reconstructed), dtype=np.float32)
    is_anchor_month = np.asarray(reconstruction["is_anchor_month"], dtype=bool)
    interval_id_per_month = np.asarray(reconstruction["interval_id_per_month"], dtype=np.int32)
    obs_mask = np.asarray(observed["mask"], dtype=bool)

    grid_lookup_path = meta_dir / "grid_lookup.csv"
    month_index_path = meta_dir / "month_index.csv"
    interval_summary_path = meta_dir / "interval_summary.csv"
    gap_paths_path = meta_dir / "gap_paths.json"
    matrix_path = recon_dir / "wtd_reconstructed_matrix.npy"

    active_grid.to_csv(grid_lookup_path, index=False, lineterminator="\n")
    month_frame = pd.DataFrame(
        {
            "month_idx": np.arange(len(month_labels), dtype=int),
            "month_label": month_labels,
            "is_anchor_month": is_anchor_month.astype(int),
            "interval_id": interval_id_per_month,
            "direct_observation_count": obs_mask.sum(axis=1).astype(int),
        }
    )
    month_frame.to_csv(month_index_path, index=False, lineterminator="\n")
    reconstruction["interval_summary"].to_csv(interval_summary_path, index=False, lineterminator="\n")
    gap_paths_serializable = {
        int(gap): {
            "cost": float(info["cost"]),
            "path": [int(v) for v in info["path"]],
            "last_horizon": int(info["last_horizon"]),
            "prev_gap": None if info["prev_gap"] is None else int(info["prev_gap"]),
        }
        for gap, info in reconstruction["gap_paths"].items()
    }
    gap_paths_path.write_text(json.dumps(gap_paths_serializable, indent=2), encoding="utf-8")

    def _replace_npy(path: Path, array: np.ndarray) -> None:
        tmp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
        try:
            tmp_path.unlink(missing_ok=True)
            np.save(tmp_path, array)
            tmp_path.replace(path)
        except OSError as exc:
            tmp_path.unlink(missing_ok=True)
            raise OSError(
                f"Could not replace {_display_path(path)}. Close any notebook, viewer, GIS, or Python process "
                "that may be using this file, then rerun the reconstruction."
            ) from exc

    _replace_npy(matrix_path, reconstructed_absolute)

    return {
        "grid_lookup_csv": grid_lookup_path,
        "month_index_csv": month_index_path,
        "interval_summary_csv": interval_summary_path,
        "gap_paths_json": gap_paths_path,
        "reconstructed_matrix_npy": matrix_path,
    }
