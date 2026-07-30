from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .dataset import (
    _target_mean_std_for_horizon,
    _target_scaling_mode,
    horizon_column,
    start_time_idx_column,
    target_time_idx_column,
)
from .plot_results import (
    plot_improvement_map,
    plot_metric_vs_horizon,
    plot_representative_cells,
    plot_residual_distribution,
    plot_residual_map,
    plot_test_scatter,
    plot_training_curves,
)
from .train import _build_dataloaders, _build_model, resolve_experiment
from .utils import (
    device_from_config,
    load_time_index,
    output_paths,
    regression_metrics,
    time_unit_from_config,
)


def _inverse_standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return values * std + mean


def _predict_split(model, loader, scalers: dict[str, Any], config: dict, split_name: str) -> pd.DataFrame:
    device = device_from_config()
    amp_enabled = bool(config["training"].get("use_amp", True)) and str(device).startswith("cuda")
    autocast_device = "cuda" if str(device).startswith("cuda") else "cpu"
    target_scaling_mode = _target_scaling_mode(config)
    model.eval()
    time_unit = time_unit_from_config(config)
    time_index = load_time_index(config).set_index("time_index")
    rows = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=amp_enabled):
                batch = batch.to(device)
                model_out = model(batch)
                target_std = batch.y.view(-1).detach().cpu().numpy()
                meta = {
                    "sample_id": batch.sample_id.detach().cpu().numpy().tolist(),
                    "grid_id": batch.grid_id.detach().cpu().numpy().tolist(),
                    "start_time_idx": batch.start_time_idx.detach().cpu().numpy().tolist(),
                    "target_time_idx": batch.target_time_idx.detach().cpu().numpy().tolist(),
                    "h_steps": batch.h_steps.detach().cpu().numpy().tolist(),
                    "x": batch.x_coord.detach().cpu().numpy().tolist(),
                    "y": batch.y_coord.detach().cpu().numpy().tolist(),
                    "n_days_observed_t": batch.n_days_observed_t.detach().cpu().numpy().tolist(),
                    "n_days_observed_target": batch.n_days_observed_target.detach().cpu().numpy().tolist(),
                }

            h_values = [int(v) for v in meta["h_steps"]]
            pred_std = model_out.detach().cpu().numpy()
            pred = np.empty(len(pred_std), dtype=np.float32)
            target = np.empty(len(target_std), dtype=np.float32)
            for i, horizon in enumerate(h_values):
                mean, std = _target_mean_std_for_horizon(scalers, horizon, target_scaling_mode)
                pred[i] = _inverse_standardize(
                    np.asarray([pred_std[i]], dtype=np.float32),
                    np.asarray(mean, dtype=np.float32),
                    np.asarray(std, dtype=np.float32),
                ).reshape(-1)[0]
                target[i] = _inverse_standardize(
                    np.asarray([target_std[i]], dtype=np.float32),
                    np.asarray(mean, dtype=np.float32),
                    np.asarray(std, dtype=np.float32),
                ).reshape(-1)[0]

            for i in range(len(pred)):
                start_idx = int(meta["start_time_idx"][i])
                target_idx = int(meta["target_time_idx"][i])
                start_row = time_index.loc[start_idx]
                target_row = time_index.loc[target_idx]
                start_ts = pd.Timestamp(start_row["time_start"])
                target_ts = pd.Timestamp(target_row["time_start"])
                rows.append(
                    {
                        "sample_id": int(meta["sample_id"][i]),
                        "grid_id": int(meta["grid_id"][i]),
                        "split": split_name,
                        "time_unit": time_unit,
                        "h_steps": int(meta["h_steps"][i]),
                        "start_time_idx": start_idx,
                        "target_time_idx": target_idx,
                        "start_time_label": str(start_row["time_label"]),
                        "target_time_label": str(target_row["time_label"]),
                        "start_calendar_month": int(start_ts.month),
                        "target_calendar_month": int(target_ts.month),
                        "x": float(meta["x"][i]),
                        "y": float(meta["y"][i]),
                        "n_days_observed_t": int(meta["n_days_observed_t"][i]),
                        "n_days_observed_target": int(meta["n_days_observed_target"][i]),
                        "y_true_delta_h_m": float(target[i]),
                        "y_pred_delta_h_m": float(pred[i]),
                    }
                )
    frame = pd.DataFrame(rows)
    if time_unit == "month":
        frame["h_months"] = frame["h_steps"]
        frame["start_month_idx"] = frame["start_time_idx"]
        frame["target_month_idx"] = frame["target_time_idx"]
        frame["start_month_label"] = frame["start_time_label"]
        frame["target_month_label"] = frame["target_time_label"]
    frame["residual_m"] = frame["y_pred_delta_h_m"] - frame["y_true_delta_h_m"]
    frame["abs_error_m"] = np.abs(frame["residual_m"])
    return frame


def _overall_metrics(predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for split, sub in predictions.groupby("split", sort=False):
        metrics = regression_metrics(sub["y_true_delta_h_m"].to_numpy(), sub["y_pred_delta_h_m"].to_numpy())
        metrics.update({"model_name": model_name, "split": split, "n_samples": int(len(sub))})
        rows.append(metrics)
    return pd.DataFrame(rows)


def _metrics_by_horizon(predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for (split, horizon), sub in predictions.groupby(["split", "h_steps"], sort=False):
        metrics = regression_metrics(sub["y_true_delta_h_m"].to_numpy(), sub["y_pred_delta_h_m"].to_numpy())
        metrics.update(
            {
                "model_name": model_name,
                "split": split,
                "time_unit": str(sub["time_unit"].iloc[0]) if "time_unit" in sub.columns else "month",
                "h_steps": int(horizon),
                "n_samples": int(len(sub)),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def _metrics_by_month(predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for (split, month), sub in predictions.groupby(["split", "start_calendar_month"], sort=False):
        metrics = regression_metrics(sub["y_true_delta_h_m"].to_numpy(), sub["y_pred_delta_h_m"].to_numpy())
        metrics.update(
            {
                "model_name": model_name,
                "split": split,
                "start_calendar_month": int(month),
                "n_samples": int(len(sub)),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)



def _add_combined_test_for_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if "test" in set(predictions["split"].astype(str)):
        return predictions
    test_parts = predictions[predictions["split"].astype(str).isin(["test_temporal", "test_spatial"])]
    if test_parts.empty:
        return predictions
    combined = test_parts.copy()
    combined["split"] = "test"
    return pd.concat([predictions, combined], ignore_index=True)


def evaluate_experiment(config: dict, experiment_name: str) -> dict[str, Any]:
    config = resolve_experiment(config, experiment_name)
    outputs = output_paths(config, experiment_name)
    checkpoint_path = outputs["experiment_dir"] / "best_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run training first.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scalers = checkpoint["scalers"]

    bundle, datasets, loaders = _build_dataloaders(config, experiment_name, scalers)
    model = _build_model(config, bundle, scalers)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device_from_config())
    predictions = []
    for split_name, loader in loaders.items():
        predictions.append(_predict_split(model, loader, scalers, config, split_name))
    predictions = pd.concat(predictions, ignore_index=True)
    predictions["model_name"] = experiment_name

    if time_unit_from_config(config) == "month" and "h_months" not in predictions.columns:
        predictions["h_months"] = predictions["h_steps"]

    metric_predictions = _add_combined_test_for_metrics(predictions)
    overall = _overall_metrics(metric_predictions, experiment_name)
    by_horizon = _metrics_by_horizon(metric_predictions, experiment_name)
    by_month = _metrics_by_month(metric_predictions, experiment_name)

    predictions.to_csv(
        outputs["prediction_dir"] / f"{experiment_name}_predictions_all_splits.csv",
        index=False,
        lineterminator="\n",
    )
    overall.to_csv(
        outputs["metric_dir"] / f"{experiment_name}_overall_metrics.csv",
        index=False,
        lineterminator="\n",
    )
    by_horizon.to_csv(
        outputs["metric_dir"] / f"{experiment_name}_metrics_by_horizon.csv",
        index=False,
        lineterminator="\n",
    )
    by_month.to_csv(
        outputs["metric_dir"] / f"{experiment_name}_metrics_by_month.csv",
        index=False,
        lineterminator="\n",
    )
    return {
        "predictions": predictions,
        "overall": overall,
        "by_horizon": by_horizon,
        "by_month": by_month,
        "history": pd.read_csv(outputs["experiment_dir"] / "train_history.csv"),
    }


def evaluate_all_experiments(config: dict, experiments: list[str] | None = None) -> dict[str, Any]:
    experiments = experiments or list(config["experiments"].keys())
    results = {experiment_name: evaluate_experiment(config, experiment_name) for experiment_name in experiments}
    outputs = output_paths(config, experiments[0])

    overall = pd.concat([results[name]["overall"] for name in experiments], ignore_index=True)
    by_horizon = pd.concat([results[name]["by_horizon"] for name in experiments], ignore_index=True)
    by_month = pd.concat([results[name]["by_month"] for name in experiments], ignore_index=True)
    predictions = pd.concat([results[name]["predictions"] for name in experiments], ignore_index=True)

    overall.to_csv(outputs["metric_dir"] / "model_comparison_overall.csv", index=False, lineterminator="\n")
    by_horizon.to_csv(outputs["metric_dir"] / "model_comparison_by_horizon.csv", index=False, lineterminator="\n")
    by_month.to_csv(outputs["metric_dir"] / "model_comparison_by_month.csv", index=False, lineterminator="\n")
    predictions[predictions["split"] == "test"].to_csv(
        outputs["prediction_dir"] / "predictions_test.csv",
        index=False,
        lineterminator="\n",
    )
    if predictions[predictions["split"] == "test"].empty:
        test_parts = predictions[predictions["split"].astype(str).isin(["test_temporal", "test_spatial"])].copy()
        if not test_parts.empty:
            test_parts["split"] = "test"
            test_parts.to_csv(
                outputs["prediction_dir"] / "predictions_test.csv",
                index=False,
                lineterminator="\n",
            )
    plot_test_scatter(predictions, outputs["figure_dir"] / "test_scatter.png")
    plot_metric_vs_horizon(by_horizon, "pearson_r", outputs["figure_dir"] / "pearson_r_vs_horizon.png")
    plot_metric_vs_horizon(by_horizon, "rmse", outputs["figure_dir"] / "rmse_vs_horizon.png")
    plot_metric_vs_horizon(by_horizon, "nse", outputs["figure_dir"] / "nse_vs_horizon.png")
    plot_residual_distribution(predictions, outputs["figure_dir"] / "residual_distribution_by_horizon.png")
    plot_residual_map(predictions, outputs["figure_dir"] / "residual_map.png")
    plot_improvement_map(predictions, outputs["figure_dir"] / "aem_improvement_map.png")
    plot_representative_cells(
        predictions,
        outputs["figure_dir"] / "representative_cells_predictions.png",
        count=6,
    )
    plot_training_curves(
        {name: results[name]["history"] for name in experiments},
        outputs["figure_dir"] / "train_val_loss_curves.png",
    )

    return {
        "overall": overall,
        "by_horizon": by_horizon,
        "by_month": by_month,
        "predictions": predictions,
    }
