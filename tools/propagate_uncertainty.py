from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conformal import fit_normalized_conformal  # noqa: E402
from src.reconstruct import SpecialistPredictor, build_observation_matrices  # noqa: E402


MAIN_RECON_ROOT = ROOT / "outputs" / "RECON_MAIN_2011_2023"
WTD_MONTHLY_PATH = ROOT / "data" / "2 well WTD" / "2011_2023" / "wtd_monthly.csv"
ENSEMBLE_ROOTS = {
    1: ROOT / "outputs" / "GNN_H1",
    3: ROOT / "outputs" / "GNN_H3",
    6: ROOT / "outputs" / "GNN_H6",
}
INTERVAL_LEVEL = 0.75
DISCOUNT_GAMMA = 0.55
OBSERVED_REFERENCE_HORIZON = 1
EXPECTED_SEEDS = ("seed11", "seed22", "seed33", "seed44", "seed55")


def display_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def project_path(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _level_suffix(level: float) -> str:
    return str(int(round(float(level) * 100.0)))


def _ensure_calibration_level(ensemble_root: Path, level: float = INTERVAL_LEVEL) -> pd.DataFrame:
    level_suffix = _level_suffix(level)
    summary_dir = ensemble_root / "summary"
    path = summary_dir / f"aem_gnn_normalized_conformal_calibration_{level_suffix}.csv"
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if (not frame.empty) and np.any(np.isclose(frame["level"], float(level))):
        return frame

    pred_candidates = [summary_dir / f"aem_gnn_ensemble_predictions_with_interval_{level_suffix}.csv"]
    pi75_path = summary_dir / "aem_gnn_ensemble_predictions_with_interval_75.csv"
    if pi75_path not in pred_candidates:
        pred_candidates.append(pi75_path)
    pred_path = next((candidate for candidate in pred_candidates if candidate.exists()), None)
    if pred_path is None:
        raise FileNotFoundError(
            f"Cannot derive level={level:.2f} qhat because ensemble prediction files are missing under "
            f"{display_path(summary_dir)}"
        )

    predictions = pd.read_csv(pred_path)
    first = frame.iloc[0].to_dict() if not frame.empty else {}
    pred_col = str(first.get("pred_col", "ensemble_mean_m"))
    true_col = str(first.get("true_col", "y_true_delta_h_m"))
    scale_col = str(first.get("scale_col", "ensemble_std_m"))
    fit_split = str(first.get("fit_split", "validation"))
    eps = float(first.get("eps", 1.0e-6))

    extra = fit_normalized_conformal(
        predictions,
        fit_split=fit_split,
        pred_col=pred_col,
        true_col=true_col,
        scale_col=scale_col,
        levels=(float(level),),
        eps=eps,
    )
    merged = (
        pd.concat([frame, extra], ignore_index=True)
        .sort_values("level", kind="stable")
        .drop_duplicates(subset=["level"], keep="last")
        .reset_index(drop=True)
    )
    merged.to_csv(path, index=False)
    return merged


def _load_qhat(ensemble_root: Path, level: float = INTERVAL_LEVEL) -> float:
    frame = _ensure_calibration_level(ensemble_root, level=level)
    row = frame.loc[np.isclose(frame["level"], float(level))].iloc[0]
    return float(row["qhat"])


def _load_ensemble_members(ensemble_root: Path, horizon: int) -> list[SpecialistPredictor]:
    missing: list[Path] = []
    ckpts: list[Path] = []
    for seed_name in EXPECTED_SEEDS:
        ckpt = ensemble_root / seed_name / "checkpoints" / "aem_gnn" / "best_checkpoint.pt"
        if ckpt.exists():
            ckpts.append(ckpt)
        else:
            missing.append(ckpt)
    if missing:
        missing_text = "\n".join(f" - {display_path(path)}" for path in missing)
        raise FileNotFoundError(
            f"Missing required uncertainty ensemble checkpoints under {display_path(ensemble_root)}:\n{missing_text}"
        )

    members = [SpecialistPredictor.from_checkpoint(path) for path in ckpts]
    ref = members[0]
    ref_grid_ids = np.asarray(ref.grid_ids)
    ref_month_labels = np.asarray(ref.month_labels).astype(str)
    for member_idx, member in enumerate(members[1:], start=2):
        if member.horizon != ref.horizon:
            raise ValueError(f"Ensemble member {member_idx} has horizon {member.horizon}, expected {ref.horizon}.")
        if not np.array_equal(np.asarray(member.grid_ids), ref_grid_ids):
            raise ValueError(f"Ensemble member {member_idx} has inconsistent grid_ids.")
        if not np.array_equal(np.asarray(member.month_labels).astype(str), ref_month_labels):
            raise ValueError(f"Ensemble member {member_idx} has inconsistent month_labels.")
    return members


def _unique_required_steps(interval_summary: pd.DataFrame, gap_paths: dict[str, dict[str, object]]) -> set[tuple[int, int]]:
    needed: set[tuple[int, int]] = set()
    leaf_segments = interval_summary.loc[~interval_summary["was_split"].astype(bool)].copy()
    for row in leaf_segments.itertuples(index=False):
        left = int(row.left_month_idx)
        right = int(row.right_month_idx)
        gap = right - left
        for local_gap in range(1, gap + 1):
            plan = [int(v) for v in gap_paths[str(local_gap)]["path"]]
            cursor = left
            for h in plan:
                needed.add((int(cursor), int(h)))
                cursor += int(h)
    return needed


def _compute_step_radius(
    members_by_horizon: dict[int, list[SpecialistPredictor]],
    qhat_by_horizon: dict[int, float],
    reconstructed_absolute: np.ndarray,
    needed_steps: set[tuple[int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    ordered = sorted(needed_steps, key=lambda x: (x[0], x[1]))
    for start_month_idx, horizon in ordered:
        members = members_by_horizon[int(horizon)]
        current = reconstructed_absolute[int(start_month_idx)]
        preds = np.stack(
            [member.predict_delta_full(current, int(start_month_idx)) for member in members],
            axis=0,
        )
        std = np.std(preds, axis=0, ddof=0).astype(np.float32)
        out[(int(start_month_idx), int(horizon))] = (float(qhat_by_horizon[int(horizon)]) * std).astype(np.float32)
    return out


def _discounted_rss_plan_radius(plan_radii: list[np.ndarray], *, discount_gamma: float) -> np.ndarray:
    if not plan_radii:
        raise ValueError("plan_radii cannot be empty")
    acc = np.zeros_like(plan_radii[0], dtype=np.float32)
    n_steps = len(plan_radii)
    for idx, r in enumerate(plan_radii):
        w = float(discount_gamma) ** float(n_steps - 1 - idx)
        acc += (w * w) * np.square(r, dtype=np.float32)
    return np.sqrt(acc, dtype=np.float32)


def _build_monthly_radius_matrix(
    *,
    n_month: int,
    n_grid: int,
    interval_summary: pd.DataFrame,
    gap_paths: dict[str, dict[str, object]],
    step_radius: dict[tuple[int, int], np.ndarray],
    discount_gamma: float,
) -> np.ndarray:
    radius = np.zeros((n_month, n_grid), dtype=np.float32)
    leaf_segments = interval_summary.loc[~interval_summary["was_split"].astype(bool)].copy()
    for row in leaf_segments.itertuples(index=False):
        left = int(row.left_month_idx)
        right = int(row.right_month_idx)
        gap = right - left
        for local_gap in range(1, gap + 1):
            plan = [int(v) for v in gap_paths[str(local_gap)]["path"]]
            plan_radii: list[np.ndarray] = []
            cursor = left
            for h in plan:
                plan_radii.append(step_radius[(int(cursor), int(h))])
                cursor += int(h)
            target = left + local_gap
            radius[target] = _discounted_rss_plan_radius(plan_radii, discount_gamma=discount_gamma)
    return radius


def _ensure_horizon_loaded(
    horizon: int,
    *,
    members_by_horizon: dict[int, list[SpecialistPredictor]],
    qhat_by_horizon: dict[int, float],
    interval_level: float,
) -> None:
    if horizon in members_by_horizon:
        return
    if int(horizon) not in ENSEMBLE_ROOTS:
        raise KeyError(f"Horizon {horizon} is not configured for mainline uncertainty propagation.")
    ensemble_root = ENSEMBLE_ROOTS[int(horizon)]
    members_by_horizon[horizon] = _load_ensemble_members(ensemble_root, horizon)
    qhat_by_horizon[horizon] = _load_qhat(ensemble_root, level=interval_level)


def _apply_model_once_radius_on_observed_months(
    *,
    radius_matrix: np.ndarray,
    observed: dict[str, object],
    step_radius: dict[tuple[int, int], np.ndarray],
    reconstructed_absolute: np.ndarray,
    members_by_horizon: dict[int, list[SpecialistPredictor]],
    qhat_by_horizon: dict[int, float],
    reference_horizon: int,
    interval_level: float,
) -> np.ndarray:
    out = np.asarray(radius_matrix, dtype=np.float32).copy()
    mask = np.asarray(observed["mask"], dtype=bool)
    n_month = out.shape[0]

    h = int(reference_horizon)
    if h <= 0:
        raise ValueError(f"reference_horizon must be >= 1, got {reference_horizon}")
    if h >= n_month:
        return out

    _ensure_horizon_loaded(
        h,
        members_by_horizon=members_by_horizon,
        qhat_by_horizon=qhat_by_horizon,
        interval_level=interval_level,
    )

    needed_steps = {(int(start), h) for start in range(0, n_month - h)}
    missing_steps = needed_steps.difference(step_radius.keys())
    if missing_steps:
        step_radius.update(
            _compute_step_radius(
                members_by_horizon=members_by_horizon,
                qhat_by_horizon=qhat_by_horizon,
                reconstructed_absolute=reconstructed_absolute,
                needed_steps=missing_steps,
            )
        )

    model_once = np.zeros_like(out, dtype=np.float32)
    for start in range(0, n_month - h):
        target = start + h
        model_once[target] = step_radius[(int(start), h)]

    out[mask] = model_once[mask]
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Propagate the fixed mainline PI75 model uncertainty over reconstruction paths."
    )
    parser.add_argument(
        "recon_root",
        nargs="?",
        default=str(MAIN_RECON_ROOT),
        help=f"Reconstruction root (default: {display_path(MAIN_RECON_ROOT)}).",
    )
    return parser


def _replace_npy(path: Path, array: np.ndarray) -> Path:
    tmp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        tmp_path.unlink(missing_ok=True)
        np.save(tmp_path, array)
        tmp_path.replace(path)
        return path
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise OSError(
            f"Could not replace {path}. Close any notebook, viewer, GIS, or Python process "
            "that may be using this file, then rerun uncertainty propagation."
        ) from exc


def _save_radius_outputs(
    recon_root: Path,
    radius_matrix: np.ndarray,
) -> dict[str, Path]:
    out_root = recon_root / "model_uncertainty"
    out_root.mkdir(parents=True, exist_ok=True)

    npy_path = out_root / "monthly_model_uncertainty_radius_matrix.npy"
    radius_matrix_npy_path = _replace_npy(npy_path, radius_matrix.astype(np.float32))

    return {
        "radius_matrix_npy": radius_matrix_npy_path,
    }


def main() -> None:
    args = _build_arg_parser().parse_args()
    interval_level = float(INTERVAL_LEVEL)
    if not 0.0 < interval_level < 1.0:
        raise ValueError(f"INTERVAL_LEVEL must be between 0 and 1, got {interval_level}")

    recon_root = project_path(args.recon_root)
    interval_summary = pd.read_csv(recon_root / "metadata" / "interval_summary.csv")
    gap_paths = json.loads((recon_root / "metadata" / "gap_paths.json").read_text(encoding="utf-8"))
    month_index = pd.read_csv(recon_root / "metadata" / "month_index.csv")
    month_col = "month_label" if "month_label" in month_index.columns else "month"
    month_labels = month_index[month_col].tolist()
    grid_lookup = pd.read_csv(recon_root / "metadata" / "grid_lookup.csv")
    reconstructed_absolute = np.load(recon_root / "reconstruction" / "wtd_reconstructed_matrix.npy").astype(np.float32)

    members_by_horizon: dict[int, list[SpecialistPredictor]] = {}
    qhat_by_horizon: dict[int, float] = {}
    used_horizons = sorted({int(h) for info in gap_paths.values() for h in info["path"]})
    for horizon in used_horizons:
        if horizon not in ENSEMBLE_ROOTS:
            raise KeyError(f"Horizon {horizon} is not configured for mainline uncertainty propagation.")
        ensemble_root = ENSEMBLE_ROOTS[horizon]
        members_by_horizon[horizon] = _load_ensemble_members(ensemble_root, horizon)
        qhat_by_horizon[horizon] = _load_qhat(ensemble_root, level=interval_level)

    observed = build_observation_matrices(
        wtd_monthly_path=WTD_MONTHLY_PATH,
        active_grid=members_by_horizon[used_horizons[0]][0].active_grid,
        month_labels=np.asarray(month_labels),
    )

    needed_steps = _unique_required_steps(interval_summary, gap_paths)
    step_radius = _compute_step_radius(
        members_by_horizon=members_by_horizon,
        qhat_by_horizon=qhat_by_horizon,
        reconstructed_absolute=reconstructed_absolute,
        needed_steps=needed_steps,
    )
    radius_matrix = _build_monthly_radius_matrix(
        n_month=len(month_labels),
        n_grid=reconstructed_absolute.shape[1],
        interval_summary=interval_summary,
        gap_paths=gap_paths,
        step_radius=step_radius,
        discount_gamma=float(DISCOUNT_GAMMA),
    )
    radius_matrix = _apply_model_once_radius_on_observed_months(
        radius_matrix=radius_matrix,
        observed=observed,
        step_radius=step_radius,
        reconstructed_absolute=reconstructed_absolute,
        members_by_horizon=members_by_horizon,
        qhat_by_horizon=qhat_by_horizon,
        reference_horizon=int(OBSERVED_REFERENCE_HORIZON),
        interval_level=interval_level,
    )

    outputs = _save_radius_outputs(
        recon_root=recon_root,
        radius_matrix=radius_matrix,
    )

    meta = {
        "used_horizons": used_horizons,
        "qhat_by_horizon": qhat_by_horizon,
        "n_members_by_horizon": {int(h): int(len(members)) for h, members in members_by_horizon.items()},
        "n_required_step_predictions": int(len(needed_steps)),
        "interval_level": interval_level,
        "interval_label": f"PI{_level_suffix(interval_level)}",
        "radius_matrix_npy_path": display_path(outputs["radius_matrix_npy"]),
        "accumulation_mode": "discounted_rss",
        "discount_gamma": float(DISCOUNT_GAMMA),
        "observed_mode": "model_once",
        "observed_reference_horizon": int(OBSERVED_REFERENCE_HORIZON),
        "note": (
            "Fixed mainline PI75 uncertainty radius propagated from horizon ensemble predictions. "
            "Multi-step gaps use discounted RSS; observed months use one-step model uncertainty."
        ),
    }
    meta_path = recon_root / "model_uncertainty" / "model_uncertainty_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Model uncertainty propagation finished.")
    for key, path in outputs.items():
        print(f"{key}: {display_path(path)}")
    print(f"metadata_json: {display_path(meta_path)}")


if __name__ == "__main__":
    main()
