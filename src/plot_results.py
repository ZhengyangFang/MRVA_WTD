from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["distance_gnn", "aem_gnn"]
MODEL_LABELS = {
    "distance_gnn": "Distance-GNN",
    "aem_gnn": "AEM-GNN",
}
METRIC_LABELS = {
    "pearson_r": "Pearson r",
    "rmse": "RMSE",
    "mae": "MAE",
    "nse": "NSE",
    "bias": "Bias",
}


def _horizon_col(frame: pd.DataFrame) -> str:
    return "h_months" if "h_months" in frame.columns else "h_steps"


def _horizon_axis_label(frame: pd.DataFrame) -> str:
    if "time_unit" in frame.columns:
        units = [str(v) for v in frame["time_unit"].dropna().unique()]
        if len(units) == 1:
            return f"Horizon H ({units[0]}s)"
    return "Horizon H"


def _ordered_models(frame: pd.DataFrame) -> list[str]:
    present = set(frame["model_name"].unique())
    return [name for name in MODEL_ORDER if name in present]


def plot_test_scatter(predictions: pd.DataFrame, out_path: str | Path) -> None:
    test = predictions[predictions["split"] == "test"].copy()
    models = _ordered_models(test)
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for ax, model_name in zip(axes[0], models):
        sub = test[test["model_name"] == model_name]
        ax.scatter(sub["y_true_delta_h_m"], sub["y_pred_delta_h_m"], s=8, alpha=0.25)
        lo = min(sub["y_true_delta_h_m"].min(), sub["y_pred_delta_h_m"].min())
        hi = max(sub["y_true_delta_h_m"].max(), sub["y_pred_delta_h_m"].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(MODEL_LABELS.get(model_name, model_name))
        ax.set_xlabel("Observed delta_h (m)")
        ax.set_ylabel("Predicted delta_h (m)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metric_vs_horizon(metrics_by_h: pd.DataFrame, metric: str, out_path: str | Path) -> None:
    test = metrics_by_h[metrics_by_h["split"] == "test"].copy()
    horizon_col = _horizon_col(test)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name in _ordered_models(test):
        sub = test[test["model_name"] == model_name].sort_values(horizon_col)
        ax.plot(sub[horizon_col], sub[metric], marker="o", label=MODEL_LABELS.get(model_name, model_name))
    ax.set_xlabel(_horizon_axis_label(test))
    label = METRIC_LABELS.get(metric, metric)
    ax.set_ylabel(label)
    ax.set_title(f"Test {label} by horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_residual_distribution(predictions: pd.DataFrame, out_path: str | Path) -> None:
    test = predictions[predictions["split"] == "test"].copy()
    models = _ordered_models(test)
    horizon_col = _horizon_col(test)
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 4 * len(models)), squeeze=False, sharex=True)
    horizon_values = sorted(test[horizon_col].unique())
    for ax, model_name in zip(axes[:, 0], models):
        sub = test[test["model_name"] == model_name]
        data = [sub.loc[sub[horizon_col] == h, "residual_m"].to_numpy() for h in horizon_values]
        ax.boxplot(data, positions=np.arange(len(horizon_values)), widths=0.7, showfliers=False)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_title(MODEL_LABELS.get(model_name, model_name))
        ax.set_ylabel("Residual (m)")
    axes[-1, 0].set_xticks(np.arange(len(horizon_values)))
    axes[-1, 0].set_xticklabels(horizon_values)
    axes[-1, 0].set_xlabel(_horizon_axis_label(test))
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_residual_map(predictions: pd.DataFrame, out_path: str | Path) -> None:
    test = predictions[predictions["split"] == "test"].copy()
    summary = (
        test.groupby(["model_name", "grid_id", "x", "y"], as_index=False)["residual_m"]
        .mean()
        .rename(columns={"residual_m": "mean_residual_m"})
    )
    models = _ordered_models(summary)
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    v = np.nanpercentile(np.abs(summary["mean_residual_m"]), 95)
    for ax, model_name in zip(axes[0], models):
        sub = summary[summary["model_name"] == model_name]
        sc = ax.scatter(sub["x"], sub["y"], c=sub["mean_residual_m"], s=10, cmap="coolwarm", vmin=-v, vmax=v)
        ax.set_title(MODEL_LABELS.get(model_name, model_name))
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, label="Mean test residual (m)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_improvement_map(predictions: pd.DataFrame, out_path: str | Path) -> None:
    test = predictions[predictions["split"] == "test"].copy()
    pivot = test.pivot_table(
        index=["sample_id", "grid_id", "x", "y"],
        columns="model_name",
        values="abs_error_m",
    ).reset_index()
    if "aem_gnn" not in pivot or "distance_gnn" not in pivot:
        return
    pivot["aem_minus_distance_abs_error"] = pivot["aem_gnn"] - pivot["distance_gnn"]
    summary = (
        pivot.groupby(["grid_id", "x", "y"], as_index=False)["aem_minus_distance_abs_error"]
        .mean()
        .rename(columns={"aem_minus_distance_abs_error": "mean_error_diff"})
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    v = np.nanpercentile(np.abs(summary["mean_error_diff"]), 95)
    sc = ax.scatter(summary["x"], summary["y"], c=summary["mean_error_diff"], s=10, cmap="PiYG", vmin=-v, vmax=v)
    ax.set_title("AEM-GNN improvement over Distance-GNN")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.colorbar(sc, ax=ax, label="Mean abs-error difference (AEM - Distance)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_representative_cells(predictions: pd.DataFrame, out_path: str | Path, count: int = 6) -> None:
    test = predictions[predictions["split"] == "test"].copy()
    density = test.groupby("grid_id").size().sort_values(ascending=False)
    grids = density.head(count).index.tolist()
    if not grids:
        return
    horizon_col = _horizon_col(test)
    fig, axes = plt.subplots(len(grids), 1, figsize=(10, 3.5 * len(grids)), squeeze=False)
    for ax, grid_id in zip(axes[:, 0], grids):
        sub = test[test["grid_id"] == grid_id].copy()
        true_by_h = sub.groupby(horizon_col, as_index=False)["y_true_delta_h_m"].mean()
        ax.plot(true_by_h[horizon_col], true_by_h["y_true_delta_h_m"], marker="o", color="k", label="Observed")
        for model_name in _ordered_models(sub):
            pred_by_h = (
                sub[sub["model_name"] == model_name]
                .groupby(horizon_col, as_index=False)["y_pred_delta_h_m"]
                .mean()
            )
            ax.plot(
                pred_by_h[horizon_col],
                pred_by_h["y_pred_delta_h_m"],
                marker="o",
                label=MODEL_LABELS.get(model_name, model_name),
            )
        ax.set_title(f"Representative grid {grid_id}")
        ax.set_xlabel(_horizon_axis_label(test))
        ax.set_ylabel("Mean delta_h (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_training_curves(history_by_model: dict[str, pd.DataFrame], out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    ax_train, ax_val = axes[0]
    for model_name, history in history_by_model.items():
        ax_train.plot(history["epoch"], history["train_loss"], label=MODEL_LABELS.get(model_name, model_name))
        ax_val.plot(history["epoch"], history["val_loss"], label=MODEL_LABELS.get(model_name, model_name))
    ax_train.set_title("Train loss")
    ax_val.set_title("Validation loss")
    for ax in [ax_train, ax_val]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
