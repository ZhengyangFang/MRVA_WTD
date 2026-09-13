from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import regression_metrics  # noqa: E402


ENSEMBLE_ROOTS = {
    "H1": ROOT / "outputs" / "GNN_H1",
    "H3": ROOT / "outputs" / "GNN_H3",
    "H6": ROOT / "outputs" / "GNN_H6",
}
PREDICTION_FILE = Path("predictions") / "aem_gnn_predictions_all_splits.csv"
OUT_DIR = ROOT / "outputs"
PER_SEED_CSV = OUT_DIR / "summary_spacetime_h1_h3_h6_metrics_per_seed.csv"
BY_SPLIT_CSV = OUT_DIR / "summary_spacetime_h1_h3_h6_metrics_by_split.csv"


def summarize_seed(model_name: str, seed_dir: Path) -> list[dict[str, float | int | str]]:
    pred_path = seed_dir / PREDICTION_FILE
    if not pred_path.exists():
        return []
    frame = pd.read_csv(pred_path)
    test_parts = frame[frame["split"].astype(str).isin(["test_temporal", "test_spatial"])].copy()
    if not test_parts.empty:
        test_parts["split"] = "test"
        frame = pd.concat([frame, test_parts], ignore_index=True)
    rows: list[dict[str, float | int | str]] = []
    for split, sub in frame.groupby("split", sort=False):
        metrics = regression_metrics(sub["y_true_delta_h_m"].to_numpy(), sub["y_pred_delta_h_m"].to_numpy())
        rows.append(
            {
                "model": model_name,
                "seed": seed_dir.name,
                "split": str(split),
                "R": metrics["pearson_r"],
                "RMSE": metrics["rmse"],
                "NSE": metrics["nse"],
                "n_samples": int(len(sub)),
            }
        )
    return rows


def main() -> None:
    per_seed_rows: list[dict[str, float | int | str]] = []
    for model_name, ensemble_root in ENSEMBLE_ROOTS.items():
        for seed_dir in sorted(p for p in ensemble_root.glob("seed*") if p.is_dir()):
            per_seed_rows.extend(summarize_seed(model_name, seed_dir))

    if not per_seed_rows:
        raise FileNotFoundError("No seed prediction files found under the configured ensemble roots.")

    per_seed = pd.DataFrame(per_seed_rows).sort_values(["model", "split", "seed"]).reset_index(drop=True)
    per_seed.to_csv(PER_SEED_CSV, index=False)

    by_split = (
        per_seed.groupby(["model", "split"], as_index=False)
        .agg(
            R_mean=("R", "mean"),
            R_std=("R", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            NSE_mean=("NSE", "mean"),
            NSE_std=("NSE", "std"),
            n_samples=("n_samples", "first"),
        )
        .sort_values(["model", "split"])
        .reset_index(drop=True)
    )
    by_split.to_csv(BY_SPLIT_CSV, index=False)

    print(f"Saved: {PER_SEED_CSV}")
    print(f"Saved: {BY_SPLIT_CSV}")
    print(by_split.to_string(index=False))


if __name__ == "__main__":
    main()
