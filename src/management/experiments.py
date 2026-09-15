from __future__ import annotations

import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reconstruct import SpecialistPredictor, best_paths_by_gap, compute_horizon_costs
from tools.recon_mass_correction import sample_specific_yield
from tools.run_recon import (
    ENSEMBLE_ROOTS,
    EXPECTED_SEEDS,
    EnsembleSpecialistPredictor,
    all_seed_checkpoints,
)


MANAGEMENT_DIR = ROOT / "outputs" / "MANAGEMENT_2012_2013"
LOCAL_RESPONSE_DIR = MANAGEMENT_DIR / "local_response"
RESULTS_DIR = MANAGEMENT_DIR / "equal_volume"
RECOVERY_DIR = MANAGEMENT_DIR / "recovery_outcomes"
CACHE_DIR = MANAGEMENT_DIR / "_cache"
RECON_DIR = ROOT / "outputs" / "RECON_MAIN_2011_2023"
for directory in (LOCAL_RESPONSE_DIR, RESULTS_DIR, RECOVERY_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

GRID_CSV = RECON_DIR / "metadata" / "grid_lookup.csv"
MONTH_INDEX_CSV = RECON_DIR / "metadata" / "month_index.csv"
WTD_MATRIX = RECON_DIR / "reconstruction" / "wtd_reconstructed_matrix.npy"
DROUGHT_METRICS_CSV = RECON_DIR / "metrics" / "drought_metrics.csv"
LABELS_CSV = RECON_DIR / "metrics" / "clustering" / "cluster_labels.csv"
DYNAMIC_CACHE = ROOT / "data" / "train_val_test_inputs" / "GNN_spacetime" / "H6" / "dynamic_monthly_cache.pt"
UNIT_CELLS_CSV = LOCAL_RESPONSE_DIR / "cell_metrics.csv.gz"
SEED_CELL_LEVERAGE_CSV = LOCAL_RESPONSE_DIR / "seed_cell_leverage.csv.gz"

SEEDS = tuple(EXPECTED_SEEDS)
RAW_LOG_PAIRS = (
    ("monthly_pumping", "monthly_pumping_log1p"),
    ("monthly_pumping_mm", "monthly_pumping_mm_log1p"),
)
CLASS_ORDER = ("Fast recovery", "Slow recovery", "Buffered")
CLASS_SHORT = {"Fast recovery": "F", "Slow recovery": "S", "Buffered": "B"}
CLASS_COLORS = {"Fast recovery": "#486A9A", "Slow recovery": "#B8647C", "Buffered": "#5B9C95"}
STRATEGY_COLORS = {
    "Uniform": "#4A4A4A",
    "High pumping": "#7489B5",
    "Fast recovery": "#486A9A",
    "Slow recovery": "#B8647C",
    "Leverage guided": "#16857C",
}
PI75_NORMAL_FACTOR = 1.150349
LOCAL_REDUCTION_CAP = 0.50
MAIN_MAX_BUDGET = 30
BUDGETS = tuple(range(0, 31, 5))
REPORT_BUDGET = 20
RANKING_VERSION = "2012_five_seed_mean_cell_response_allocation_v3"
ALLOCATION_SCALE = "1 km Cell"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 7.4,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


@dataclass(frozen=True)
class Window:
    name: str
    initial_month: str
    intervention_start: str
    intervention_end: str
    final_month: str
    endpoint_months: tuple[str, ...]


WINDOW_2012 = Window(
    name="May-Oct 2012 drought-period intervention",
    initial_month="2012-04",
    intervention_start="2012-05",
    intervention_end="2012-10",
    final_month="2013-04",
    endpoint_months=("2012-10", "2012-12", "2013-04"),
)
WINDOW_LEVERAGE = Window(
    name="Jan-Dec 2012 own-Unit response experiment for Cell ranking",
    initial_month="2011-12",
    intervention_start="2012-01",
    intervention_end="2012-12",
    final_month="2012-12",
    endpoint_months=("2012-12",),
)
WINDOW_LOCAL_RESPONSE = Window(
    name="Jan-Dec 2012 reduction followed by 12-month recovery",
    initial_month="2011-12",
    intervention_start="2012-01",
    intervention_end="2012-12",
    final_month="2013-12",
    endpoint_months=("2012-12", "2013-12"),
)
WINDOW_2013 = Window(
    name="May-Oct 2013 post-drought intervention",
    initial_month="2013-04",
    intervention_start="2013-05",
    intervention_end="2013-10",
    final_month="2014-04",
    endpoint_months=("2013-10", "2013-12", "2014-04"),
)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def mean_pi75(values) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    radius = float(PI75_NORMAL_FACTOR * values.std(ddof=0))
    return mean, mean - radius, mean + radius


def _first_hit_index(hit: np.ndarray) -> np.ndarray:
    any_hit = hit.any(axis=0)
    first = hit.argmax(axis=0).astype(float)
    first[~any_hit] = np.nan
    return first


class ExperimentContext:
    def __init__(self) -> None:
        required = (
            GRID_CSV,
            MONTH_INDEX_CSV,
            WTD_MATRIX,
            DROUGHT_METRICS_CSV,
            LABELS_CSV,
            DYNAMIC_CACHE,
        )
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)

        self.grid = pd.read_csv(GRID_CSV)
        self.month_index = pd.read_csv(MONTH_INDEX_CSV)
        self.month_index["month_label"] = self.month_index["month_label"].astype(str)
        self.drought = pd.read_csv(DROUGHT_METRICS_CSV)
        self.labels = pd.read_csv(LABELS_CSV)
        if UNIT_CELLS_CSV.exists():
            self.unit_cells = pd.read_csv(UNIT_CELLS_CSV, dtype={"unit_id": str})
        else:
            unit_keys = (
                self.grid.assign(unit_row=self.grid.row // 20, unit_col=self.grid.col // 20)
                [["unit_row", "unit_col"]]
                .drop_duplicates()
                .sort_values(["unit_row", "unit_col"])
                .reset_index(drop=True)
            )
            unit_keys["unit_id"] = [f"U{number:03d}" for number in range(1, len(unit_keys) + 1)]
            self.unit_cells = (
                self.grid.assign(unit_row=self.grid.row // 20, unit_col=self.grid.col // 20)
                .merge(unit_keys, on=["unit_row", "unit_col"], validate="many_to_one")
                [["grid_id", "row", "col", "x", "y", "unit_id"]]
            )
        self.seed_cell_leverage = None
        self.wtd = np.load(WTD_MATRIX, mmap_mode="r")

        cache = torch.load(DYNAMIC_CACHE, map_location="cpu", weights_only=False)
        self.dynamic = np.asarray(cache["data"], dtype=np.float32)
        self.feature_names = list(cache["feature_names"])
        self.grid_ids = np.asarray(cache["grid_ids"], dtype=np.int64)
        self.months = np.asarray(cache["month_labels"]).astype(str)

        if not np.array_equal(self.grid.grid_id.to_numpy(np.int64), self.grid_ids):
            raise ValueError("Grid lookup and dynamic cache use different Cell order.")
        if not np.array_equal(self.labels.grid_id.to_numpy(np.int64), self.grid_ids):
            raise ValueError("Response classes and dynamic cache use different Cell order.")
        if not np.array_equal(self.drought.grid_id.to_numpy(np.int64), self.grid_ids):
            raise ValueError("Drought metrics and dynamic cache use different Cell order.")
        unit_order = self.unit_cells.set_index("grid_id").reindex(self.grid_ids)
        if unit_order.unit_id.isna().any() or self.unit_cells.grid_id.duplicated().any():
            raise ValueError("Fig. S22 Unit assignment must cover each active Cell once.")
        self.unit_by_grid = unit_order.unit_id.to_numpy(str)
        self.unit_ids = np.unique(self.unit_by_grid)
        self.unit_indices = {unit: np.flatnonzero(self.unit_by_grid == unit) for unit in self.unit_ids}

        dx = float(np.median(np.diff(np.sort(self.grid.x.unique()))))
        dy = float(np.median(np.diff(np.sort(self.grid.y.unique()))))
        if not np.isclose(dx, 1000.0) or not np.isclose(dy, 1000.0):
            raise AssertionError("Expected the native 1 km grid.")
        self.cell_area_m2 = dx * dy
        self.sy = sample_specific_yield(self.grid).astype(np.float64)
        self.storage_weights_m2 = self.cell_area_m2 * self.sy

        self.response_class = self.labels.response_class.astype(str).to_numpy()
        self.class_fraction = self._unit_class_fraction()
        self.month_lookup = {month: int(np.flatnonzero(self.months == month)[0]) for month in self.months}
        self.raw_indices = {raw: self.feature_names.index(raw) for raw, _ in RAW_LOG_PAIRS}
        self.log_indices = {log: self.feature_names.index(log) for _, log in RAW_LOG_PAIRS}

        annual_idx = self.indices("2012-01", "2012-12")
        pump_idx = self.raw_indices["monthly_pumping"]
        self.reference_2012_pumping_m3 = float(self.dynamic[:, annual_idx, pump_idx].sum(dtype=np.float64))

        self.pre_wtd = self.drought.pre_wtd_m.to_numpy(float)
        self.peak_wtd = self.drought.drought_peak_wtd_m.to_numpy(float)
        self.peak_month_idx = self.drought.drought_peak_month_idx.to_numpy(int)
        self.valid_decline = self.drought.valid_decline.to_numpy(bool)
        self.d50 = self.peak_wtd - 0.5 * (self.peak_wtd - self.pre_wtd)
        self.baseline_t50_index = self._baseline_t50_index()
        published_t50 = self.drought.T50_months.to_numpy(float)
        reproduced = self.baseline_t50_index - self.peak_month_idx
        if not np.allclose(reproduced, published_t50, equal_nan=True, rtol=0, atol=0):
            raise AssertionError("Fixed D50 does not reproduce the paper's T50 metric.")

        self.members_by_horizon = None
        self.seed_maps = None
        self.paths = None
        self.horizon_costs = None

    def indices(self, start: str, end: str) -> np.ndarray:
        idx = np.flatnonzero((self.months >= start) & (self.months <= end))
        if not len(idx):
            raise ValueError(f"No months found for {start} to {end}.")
        return idx

    def window_indices(self, window: Window) -> tuple[int, np.ndarray, np.ndarray]:
        initial = self.month_lookup[window.initial_month]
        intervention = self.indices(window.intervention_start, window.intervention_end)
        output = self.indices(window.intervention_start, window.final_month)
        if intervention[0] != initial + 1 or output[0] != initial + 1:
            raise AssertionError("Window must start one month after the fixed initial state.")
        return initial, intervention, output

    def _unit_class_fraction(self) -> pd.DataFrame:
        frame = pd.DataFrame({"unit_id": self.unit_by_grid, "response_class": self.response_class})
        counts = pd.crosstab(frame.unit_id, frame.response_class).reindex(self.unit_ids, fill_value=0)
        totals = frame.groupby("unit_id").size().reindex(self.unit_ids).to_numpy(float)
        out = pd.DataFrame(index=self.unit_ids)
        for label in CLASS_ORDER:
            values = counts[label].to_numpy(float) if label in counts else np.zeros(len(self.unit_ids))
            out[label] = values / totals
        return out

    def _baseline_t50_index(self) -> np.ndarray:
        all_wtd = np.asarray(self.wtd, dtype=np.float32)
        month_grid = np.arange(len(self.months), dtype=int)[:, None]
        hit = (month_grid > self.peak_month_idx[None, :]) & (all_wtd <= self.d50[None, :])
        first = _first_hit_index(hit)
        first[~self.valid_decline] = np.nan
        return first

    def load_models(self) -> None:
        if self.members_by_horizon is not None:
            return
        members_by_horizon = {}
        shared_aem = shared_topography = shared_root_mask = None
        for horizon in (1, 3, 6):
            members = []
            graph_reference = None
            records = all_seed_checkpoints(horizon, ENSEMBLE_ROOTS[horizon])
            if len(records) != len(SEEDS):
                raise AssertionError(f"H{horizon}: incomplete five-member ensemble.")
            for checkpoint_path, _ in records:
                member = SpecialistPredictor.from_checkpoint(checkpoint_path)
                cfg = member.config.get("features", {})
                if member.horizon != horizon or not np.array_equal(member.grid_ids, self.grid_ids):
                    raise ValueError(f"Grid/horizon mismatch: {checkpoint_path}")
                if not cfg.get("use_log1p_pumping", False) or str(cfg.get("pumping_log_input", "")).lower() != "depth_mm":
                    raise ValueError(f"Pumping-input configuration mismatch: {checkpoint_path}")
                if member.dynamic_feature_names != self.feature_names or not np.array_equal(np.asarray(member.month_labels).astype(str), self.months):
                    raise ValueError(f"Dynamic-input mismatch: {checkpoint_path}")
                member.dynamic_data = self.dynamic
                if shared_aem is None:
                    shared_aem, shared_topography, shared_root_mask = member.aem_profile, member.topography_features, member.root_mask
                else:
                    member.aem_profile, member.topography_features, member.root_mask = shared_aem, shared_topography, shared_root_mask
                if graph_reference is None:
                    graph_reference = (member.edge_index, member.edge_weight)
                else:
                    member.edge_index, member.edge_weight = graph_reference
                members.append(member)
                gc.collect()
            members_by_horizon[horizon] = members
            print(f"H{horizon}: loaded {len(members)} model members")
        ensemble = {h: EnsembleSpecialistPredictor(members_by_horizon[h]) for h in (1, 3, 6)}
        self.horizon_costs = compute_horizon_costs(ensemble, level=0.75)
        self.paths = best_paths_by_gap(24, self.horizon_costs)
        self.members_by_horizon = members_by_horizon
        self.seed_maps = {seed: {h: members_by_horizon[h][k] for h in (1, 3, 6)} for k, seed in enumerate(SEEDS)}
        print("Forecast paths:", {gap: self.paths[gap]["path"] for gap in (6, 8, 12)})

    def forecast_seed(self, window: Window, seed: str) -> np.ndarray:
        self.load_models()
        initial_idx, _, output_idx = self.window_indices(window)
        initial_wtd = np.asarray(self.wtd[initial_idx], dtype=np.float32).copy()
        states = [initial_wtd]
        for gap in range(1, len(output_idx) + 1):
            plan = self.paths[gap]
            horizon = int(plan["last_horizon"])
            previous_gap = int(plan["prev_gap"])
            delta = self.seed_maps[seed][horizon].predict_delta_full(states[previous_gap], initial_idx + previous_gap)
            states.append((states[previous_gap] - delta).astype(np.float32))
        result = np.stack(states[1:])
        if result.shape != (len(output_idx), len(self.grid_ids)) or not np.isfinite(result).all():
            raise ValueError("Invalid counterfactual forecast.")
        return result

    def capture_window_pumping(self, window: Window) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        _, intervention_idx, _ = self.window_indices(window)
        baseline = {}
        for raw, log in RAW_LOG_PAIRS:
            raw_values = self.dynamic[:, intervention_idx, self.raw_indices[raw]].copy()
            log_values = self.dynamic[:, intervention_idx, self.log_indices[log]]
            if not np.allclose(log_values, np.log1p(np.clip(raw_values, 0, None)), rtol=0, atol=1e-6):
                raise AssertionError(f"{log} does not equal log1p of raw pumping.")
            baseline[raw] = raw_values
        cell_volume = baseline["monthly_pumping"].sum(axis=1, dtype=np.float64)
        depth_check = baseline["monthly_pumping_mm"].sum(dtype=np.float64) * self.cell_area_m2 / 1000.0
        if not np.isclose(cell_volume.sum(), depth_check, rtol=1e-8, atol=1.0):
            raise AssertionError("Pumping volume and depth inputs disagree.")
        return cell_volume, baseline

    def apply_reduction(self, window: Window, cell_fraction: np.ndarray, baseline: dict[str, np.ndarray]) -> None:
        _, intervention_idx, _ = self.window_indices(window)
        factor = 1.0 - np.asarray(cell_fraction, dtype=np.float32)[:, None]
        for raw, log in RAW_LOG_PAIRS:
            scaled = baseline[raw] * factor
            self.dynamic[:, intervention_idx, self.raw_indices[raw]] = scaled
            self.dynamic[:, intervention_idx, self.log_indices[log]] = np.log1p(np.clip(scaled, 0, None))

    def restore_window(self, window: Window, baseline: dict[str, np.ndarray]) -> None:
        _, intervention_idx, _ = self.window_indices(window)
        for raw, log in RAW_LOG_PAIRS:
            self.dynamic[:, intervention_idx, self.raw_indices[raw]] = baseline[raw]
            self.dynamic[:, intervention_idx, self.log_indices[log]] = np.log1p(np.clip(baseline[raw], 0, None))

    def unit_pumping(self, cell_volume: np.ndarray) -> pd.Series:
        return pd.Series(cell_volume, index=self.unit_by_grid).groupby(level=0).sum().reindex(self.unit_ids)

    def ensure_seed_cell_leverage(self, force: bool = False) -> pd.DataFrame:
        required = {
            "seed",
            "grid_id",
            "unit_id",
            "own_unit_response_dec2012_m",
            "pumping_reduction_depth_m",
            "cell_management_leverage",
        }
        if SEED_CELL_LEVERAGE_CSV.exists() and not force:
            cached = pd.read_csv(
                SEED_CELL_LEVERAGE_CSV,
                dtype={"seed": str, "unit_id": str},
            )
            complete = (
                required.issubset(cached.columns)
                and len(cached) == len(SEEDS) * len(self.grid_ids)
                and set(cached.seed) == set(SEEDS)
                and cached.groupby("seed").grid_id.nunique().eq(len(self.grid_ids)).all()
            )
            if complete:
                self.seed_cell_leverage = cached
                return cached

        print("Computing seed-specific Cell response scores from independent own-Unit interventions.")
        cell_volume, pumping_baseline = self.capture_window_pumping(WINDOW_LEVERAGE)
        self.load_models()
        self.restore_window(WINDOW_LEVERAGE, pumping_baseline)
        baseline = {seed: self.forecast_seed(WINDOW_LEVERAGE, seed)[-1] for seed in SEEDS}
        score_by_seed = {seed: np.full(len(self.grid_ids), np.nan, dtype=np.float32) for seed in SEEDS}
        response_by_seed = {seed: np.full(len(self.grid_ids), np.nan, dtype=np.float32) for seed in SEEDS}
        depth_by_cell = np.full(len(self.grid_ids), np.nan, dtype=np.float64)
        try:
            for unit_number, unit_id in enumerate(self.unit_ids, start=1):
                indices = self.unit_indices[unit_id]
                reduction_m3 = 0.20 * float(cell_volume[indices].sum(dtype=np.float64))
                reduction_depth = reduction_m3 / (len(indices) * self.cell_area_m2)
                depth_by_cell[indices] = reduction_depth
                if reduction_depth >= 0.001:
                    fractions = np.zeros(len(self.grid_ids), dtype=np.float32)
                    fractions[indices] = 0.20
                    self.apply_reduction(WINDOW_LEVERAGE, fractions, pumping_baseline)
                    for seed in SEEDS:
                        scenario = self.forecast_seed(WINDOW_LEVERAGE, seed)[-1]
                        own_response = baseline[seed][indices] - scenario[indices]
                        response_by_seed[seed][indices] = own_response
                        score_by_seed[seed][indices] = own_response / reduction_depth
                    self.restore_window(WINDOW_LEVERAGE, pumping_baseline)
                if unit_number == 1 or unit_number % 25 == 0 or unit_number == len(self.unit_ids):
                    print(f"Cell leverage completed {unit_number:>3d}/{len(self.unit_ids)} Units")
        finally:
            self.restore_window(WINDOW_LEVERAGE, pumping_baseline)

        frames = []
        for seed in SEEDS:
            frames.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "grid_id": self.grid_ids,
                        "unit_id": self.unit_by_grid,
                        "own_unit_response_dec2012_m": response_by_seed[seed],
                        "pumping_reduction_depth_m": depth_by_cell,
                        "cell_management_leverage": score_by_seed[seed],
                    }
                )
            )
        result = pd.concat(frames, ignore_index=True)
        result.to_csv(SEED_CELL_LEVERAGE_CSV, index=False)
        self.seed_cell_leverage = result
        print(f"Saved: {SEED_CELL_LEVERAGE_CSV.relative_to(ROOT)}")
        return result

    def five_seed_mean_cell_scores(self) -> np.ndarray:
        frame = self.ensure_seed_cell_leverage()
        matrix = frame.pivot(index="grid_id", columns="seed", values="cell_management_leverage").reindex(self.grid_ids)
        if matrix.shape != (len(self.grid_ids), len(SEEDS)) or set(matrix.columns) != set(SEEDS):
            raise ValueError("Incomplete 2012 seed-specific Cell leverage cache.")
        return matrix.mean(axis=1, skipna=True).to_numpy(dtype=np.float64)

    def cell_ranking(self, strategy: str, cell_volume: np.ndarray, held_out_seed: str | None = None) -> np.ndarray:
        """Rank 1 km Cells directly for every targeted allocation strategy."""
        cell_volume = np.asarray(cell_volume, dtype=np.float64)
        if strategy == "High pumping":
            score = cell_volume.copy()
        elif strategy == "Fast recovery":
            score = (self.response_class == "Fast recovery").astype(np.float64)
        elif strategy == "Slow recovery":
            score = (self.response_class == "Slow recovery").astype(np.float64)
        elif strategy == "Leverage guided":
            score = self.five_seed_mean_cell_scores()
        else:
            raise ValueError(strategy)

        ranking = pd.DataFrame(
            {
                "cell_index": np.arange(len(self.grid_ids), dtype=np.int64),
                "score": np.where(np.isfinite(score), score, -np.inf),
                "pumping": cell_volume,
                "grid_id": self.grid_ids,
            }
        )
        return ranking.sort_values(
            ["score", "pumping", "grid_id"], ascending=[False, False, True]
        ).cell_index.to_numpy(dtype=np.int64)

    def allocate(
        self,
        strategy: str,
        budget_nominal: int,
        cell_volume: np.ndarray,
        held_out_seed: str | None = None,
    ) -> tuple[np.ndarray, pd.DataFrame, dict]:
        cell_volume = np.asarray(cell_volume, dtype=np.float64)
        window_total = float(cell_volume.sum(dtype=np.float64))
        requested = self.reference_2012_pumping_m3 * budget_nominal / 100.0
        feasible_target = min(requested, LOCAL_REDUCTION_CAP * window_total)
        cell_fraction = np.zeros(len(self.grid_ids), dtype=np.float64)
        if feasible_target > 0:
            maximum_reduction = LOCAL_REDUCTION_CAP * window_total
            if np.isclose(feasible_target, maximum_reduction, rtol=1.0e-12, atol=1.0):
                cell_fraction[:] = LOCAL_REDUCTION_CAP
            elif strategy == "Uniform":
                cell_fraction[:] = feasible_target / window_total
            else:
                order = self.cell_ranking(strategy, cell_volume, held_out_seed)
                available = LOCAL_REDUCTION_CAP * cell_volume[order]
                cumulative = np.cumsum(available, dtype=np.float64)
                full_count = int(np.searchsorted(cumulative, feasible_target, side="right"))
                if full_count:
                    cell_fraction[order[:full_count]] = LOCAL_REDUCTION_CAP
                    allocated = float(cumulative[full_count - 1])
                else:
                    allocated = 0.0
                remaining = feasible_target - allocated
                if remaining > max(1.0e-8, feasible_target * 1.0e-14):
                    if full_count >= len(order) or cell_volume[order[full_count]] <= 0:
                        raise ValueError(f"Could not allocate {remaining:.1f} m3 for {strategy}.")
                    cell_fraction[order[full_count]] = remaining / cell_volume[order[full_count]]
                    remaining = 0.0
                if remaining > max(1.0, feasible_target * 1e-8):
                    raise ValueError(f"Could not allocate {remaining:.1f} m3 for {strategy}.")
        cell_fraction = cell_fraction.astype(np.float32)
        actual = float(np.sum(cell_volume * cell_fraction, dtype=np.float64))
        mismatch = abs(actual - feasible_target) / max(feasible_target, 1.0)
        if mismatch > 2e-7 or cell_fraction.min() < -1e-8 or cell_fraction.max() > LOCAL_REDUCTION_CAP + 1e-7:
            raise AssertionError("Equal-volume allocation or local reduction cap failed.")

        cell_reduction = cell_volume * cell_fraction
        allocation_cells = pd.DataFrame(
            {
                "unit_id": self.unit_by_grid,
                "cell_volume_m3": cell_volume,
                "cell_reduction_fraction": cell_fraction.astype(np.float64),
                "cell_reduction_m3": cell_reduction,
                "positive_reduction": cell_fraction > 0,
            }
        )
        unit_summary = allocation_cells.groupby("unit_id", sort=True).agg(
            unit_pumping_m3=("cell_volume_m3", "sum"),
            unit_reduction_m3=("cell_reduction_m3", "sum"),
            mean_cell_reduction_fraction=("cell_reduction_fraction", "mean"),
            maximum_cell_reduction_fraction=("cell_reduction_fraction", "max"),
            reduced_cell_count=("positive_reduction", "sum"),
            cell_count=("positive_reduction", "size"),
        ).reindex(self.unit_ids)
        unit_summary["unit_reduction_fraction"] = np.divide(
            unit_summary.unit_reduction_m3,
            unit_summary.unit_pumping_m3,
            out=np.zeros(len(unit_summary), dtype=np.float64),
            where=unit_summary.unit_pumping_m3.to_numpy(dtype=np.float64) > 0,
        )
        rows = pd.DataFrame(
            {
                "strategy": strategy,
                "budget_nominal": budget_nominal,
                "ranking_seed": "shared_5_seed_mean" if strategy == "Leverage guided" else "shared",
                "ranking_version": RANKING_VERSION if strategy == "Leverage guided" else "fixed",
                "allocation_scale": ALLOCATION_SCALE,
                "unit_id": self.unit_ids,
                "unit_pumping_m3": unit_summary.unit_pumping_m3.to_numpy(dtype=np.float64),
                "unit_reduction_fraction": unit_summary.unit_reduction_fraction.to_numpy(dtype=np.float64),
                "unit_reduction_m3": unit_summary.unit_reduction_m3.to_numpy(dtype=np.float64),
                "mean_cell_reduction_fraction": unit_summary.mean_cell_reduction_fraction.to_numpy(dtype=np.float64),
                "maximum_cell_reduction_fraction": unit_summary.maximum_cell_reduction_fraction.to_numpy(dtype=np.float64),
                "reduced_cell_count": unit_summary.reduced_cell_count.to_numpy(dtype=np.int64),
                "cell_count": unit_summary.cell_count.to_numpy(dtype=np.int64),
                "requested_dV_m3": requested,
                "feasible_target_dV_m3": feasible_target,
                "actual_dV_m3": actual,
                "feasibility_capped": requested > feasible_target + 1.0,
            }
        )
        info = {
            "allocation_scale": ALLOCATION_SCALE,
            "ranking_version": RANKING_VERSION if strategy == "Leverage guided" else "fixed",
            "requested_dV_m3": requested,
            "feasible_target_dV_m3": feasible_target,
            "actual_dV_m3": actual,
            "relative_mismatch": mismatch,
            "feasibility_capped": requested > feasible_target + 1.0,
        }
        return cell_fraction, rows, info

    def storage_benefit(self, response_m: np.ndarray) -> np.ndarray:
        return np.sum(response_m.astype(np.float64) * self.storage_weights_m2[None, :], axis=1, dtype=np.float64)

    def recovery_arrays(self, response_m: np.ndarray, window: Window) -> dict[str, np.ndarray]:
        if window is not WINDOW_2013:
            raise ValueError("Recovery-time arrays are defined for the 2013 intervention only.")
        _, _, output_idx = self.window_indices(window)
        reconstructed = np.asarray(self.wtd[output_idx], dtype=np.float32)
        managed = reconstructed - response_m
        managed_first_local = _first_hit_index(managed <= self.d50[None, :])
        managed_t50_index = np.where(np.isfinite(managed_first_local), output_idx[0] + managed_first_local, np.nan)
        reached_before = np.isfinite(self.baseline_t50_index) & (self.baseline_t50_index < output_idx[0])
        managed_t50_index[reached_before] = self.baseline_t50_index[reached_before]
        valid = self.valid_decline & np.isfinite(self.d50)
        base_months = self.baseline_t50_index - self.peak_month_idx
        managed_months = managed_t50_index - self.peak_month_idx
        delta = self.baseline_t50_index - managed_t50_index
        base_months[~valid] = np.nan
        managed_months[~valid] = np.nan
        delta[~valid | ~np.isfinite(self.baseline_t50_index) | ~np.isfinite(managed_t50_index)] = np.nan
        reached_during = valid & ~reached_before & np.isfinite(managed_t50_index)
        not_reached = valid & ~reached_before & ~np.isfinite(managed_t50_index)
        return {
            "valid": valid,
            "reached_before_intervention": reached_before,
            "reached_during_simulation": reached_during,
            "not_reached_by_end": not_reached,
            "baseline_t50_index": self.baseline_t50_index.copy(),
            "management_t50_index": managed_t50_index,
            "T50_base": base_months,
            "T50_management": managed_months,
            "delta_T50": delta,
        }

    def recovery_summary(self, arrays: dict[str, np.ndarray], response_m: np.ndarray) -> list[dict]:
        rows = []
        end_idx = self.month_lookup[WINDOW_2013.final_month]
        dec_idx = self.month_lookup["2013-12"]
        slow_mask = self.response_class == "Slow recovery"
        slow_benefit = np.sum(
            response_m[:, slow_mask].astype(np.float64) * self.storage_weights_m2[None, slow_mask],
            axis=1,
            dtype=np.float64,
        )
        for label in ("All", *CLASS_ORDER):
            class_mask = np.ones(len(self.grid_ids), dtype=bool) if label == "All" else self.response_class == label
            valid = arrays["valid"] & class_mask
            eligible = valid & ~arrays["reached_before_intervention"]
            comparable = eligible & np.isfinite(arrays["delta_T50"])
            delta = arrays["delta_T50"][comparable]
            base_reached_dec = eligible & np.isfinite(arrays["baseline_t50_index"]) & (arrays["baseline_t50_index"] <= dec_idx)
            mgmt_reached_dec = eligible & np.isfinite(arrays["management_t50_index"]) & (arrays["management_t50_index"] <= dec_idx)
            base_reached_end = eligible & np.isfinite(arrays["baseline_t50_index"]) & (arrays["baseline_t50_index"] <= end_idx)
            mgmt_reached_end = eligible & np.isfinite(arrays["management_t50_index"]) & (arrays["management_t50_index"] <= end_idx)
            rows.append(
                {
                    "response_class": label,
                    "valid_cells": int(valid.sum()),
                    "eligible_unrecovered_May2013_cells": int(eligible.sum()),
                    "comparable_delta_T50_cells": int(comparable.sum()),
                    "median_delta_T50_months": float(np.nanmedian(delta)) if len(delta) else np.nan,
                    "q25_delta_T50_months": float(np.nanpercentile(delta, 25)) if len(delta) else np.nan,
                    "q75_delta_T50_months": float(np.nanpercentile(delta, 75)) if len(delta) else np.nan,
                    "fraction_eligible_delta_T50_positive": float(np.sum(comparable & (arrays["delta_T50"] > 0)) / max(eligible.sum(), 1)),
                    "fraction_comparable_delta_T50_positive": float(np.mean(delta > 0)) if len(delta) else np.nan,
                    "fraction_baseline_reached_Dec2013": float(base_reached_dec.sum() / max(eligible.sum(), 1)),
                    "fraction_management_reached_Dec2013": float(mgmt_reached_dec.sum() / max(eligible.sum(), 1)),
                    "fraction_baseline_reached_Apr2014": float(base_reached_end.sum() / max(eligible.sum(), 1)),
                    "fraction_management_reached_Apr2014": float(mgmt_reached_end.sum() / max(eligible.sum(), 1)),
                    "not_reached_by_end_cells": int((arrays["not_reached_by_end"] & class_mask).sum()),
                    "storage_benefit_slow_end_m3": float(slow_benefit[5]) if label == "Slow recovery" else np.nan,
                    "storage_benefit_slow_6m_m3": float(slow_benefit[-1]) if label == "Slow recovery" else np.nan,
                }
            )
        return rows


_CONTEXT = None


def context() -> ExperimentContext:
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = ExperimentContext()
    return _CONTEXT


def ensure_local_response_metrics(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Cell response, persistence and leverage from independent Unit reductions."""
    cell_required = {
        "grid_id", "unit_id", "own_unit_response_dec2012_m",
        "own_unit_response_dec2013_m", "own_unit_persistence_fraction",
        "pumping_reduction_depth_2012_m", "management_leverage",
    }
    seed_required = {
        "seed", "grid_id", "unit_id", "own_unit_response_dec2012_m",
        "pumping_reduction_depth_m", "cell_management_leverage",
    }
    if UNIT_CELLS_CSV.exists() and SEED_CELL_LEVERAGE_CSV.exists() and not force:
        cells = pd.read_csv(UNIT_CELLS_CSV, dtype={"unit_id": str})
        seeds = pd.read_csv(SEED_CELL_LEVERAGE_CSV, dtype={"seed": str, "unit_id": str})
        complete = (
            cell_required.issubset(cells.columns)
            and seed_required.issubset(seeds.columns)
            and len(cells) == cells.grid_id.nunique()
            and len(seeds) == len(SEEDS) * len(cells)
        )
        if complete:
            print("Loaded complete local-response and five-seed leverage tables.")
            return cells, seeds

    ctx = context()
    cell_volume, pumping_baseline = ctx.capture_window_pumping(WINDOW_LOCAL_RESPONSE)
    _, _, output_idx = ctx.window_indices(WINDOW_LOCAL_RESPONSE)
    end_2012 = int(np.flatnonzero(ctx.months[output_idx] == "2012-12")[0])
    end_2013 = int(np.flatnonzero(ctx.months[output_idx] == "2013-12")[0])
    ctx.load_models()
    ctx.restore_window(WINDOW_LOCAL_RESPONSE, pumping_baseline)
    baseline = {seed: ctx.forecast_seed(WINDOW_LOCAL_RESPONSE, seed) for seed in SEEDS}
    response_12 = {seed: np.full(len(ctx.grid_ids), np.nan, np.float32) for seed in SEEDS}
    response_24 = {seed: np.full(len(ctx.grid_ids), np.nan, np.float32) for seed in SEEDS}
    reduction_depth = np.full(len(ctx.grid_ids), np.nan, np.float64)

    try:
        for number, unit_id in enumerate(ctx.unit_ids, start=1):
            indices = ctx.unit_indices[unit_id]
            depth = 0.20 * float(cell_volume[indices].sum(dtype=np.float64)) / (
                len(indices) * ctx.cell_area_m2
            )
            reduction_depth[indices] = depth
            if depth >= 0.001:
                fractions = np.zeros(len(ctx.grid_ids), dtype=np.float32)
                fractions[indices] = 0.20
                ctx.apply_reduction(WINDOW_LOCAL_RESPONSE, fractions, pumping_baseline)
                for seed in SEEDS:
                    scenario = ctx.forecast_seed(WINDOW_LOCAL_RESPONSE, seed)
                    response_12[seed][indices] = baseline[seed][end_2012, indices] - scenario[end_2012, indices]
                    response_24[seed][indices] = baseline[seed][end_2013, indices] - scenario[end_2013, indices]
                ctx.restore_window(WINDOW_LOCAL_RESPONSE, pumping_baseline)
            if number == 1 or number % 25 == 0 or number == len(ctx.unit_ids):
                print(f"Local response completed {number:>3d}/{len(ctx.unit_ids)} Units")
    finally:
        ctx.restore_window(WINDOW_LOCAL_RESPONSE, pumping_baseline)

    seed_frames = []
    for seed in SEEDS:
        persistence = np.divide(
            response_24[seed], response_12[seed],
            out=np.full(len(ctx.grid_ids), np.nan), where=response_12[seed] > 0.001,
        )
        leverage = np.divide(
            response_12[seed], reduction_depth,
            out=np.full(len(ctx.grid_ids), np.nan), where=reduction_depth >= 0.001,
        )
        seed_frames.append(pd.DataFrame({
            "seed": seed,
            "grid_id": ctx.grid_ids,
            "unit_id": ctx.unit_by_grid,
            "own_unit_response_dec2012_m": response_12[seed],
            "own_unit_response_dec2013_m": response_24[seed],
            "own_unit_persistence_fraction": persistence,
            "pumping_reduction_depth_m": reduction_depth,
            "cell_management_leverage": leverage,
        }))
    seeds = pd.concat(seed_frames, ignore_index=True)
    mean_12 = np.mean(np.stack([response_12[seed] for seed in SEEDS]), axis=0)
    mean_24 = np.mean(np.stack([response_24[seed] for seed in SEEDS]), axis=0)
    persistence = np.divide(mean_24, mean_12, out=np.full(len(ctx.grid_ids), np.nan), where=mean_12 > 0.001)
    leverage = np.divide(mean_12, reduction_depth, out=np.full(len(ctx.grid_ids), np.nan), where=reduction_depth >= 0.001)
    area_by_unit = pd.Series(ctx.unit_by_grid).value_counts().to_dict()
    cells = ctx.grid[["grid_id", "row", "col", "x", "y"]].copy()
    cells["unit_id"] = ctx.unit_by_grid
    cells["own_unit_response_dec2012_m"] = mean_12
    cells["own_unit_response_dec2013_m"] = mean_24
    cells["own_unit_persistence_fraction"] = persistence
    cells["effective_unit_area_m2"] = cells.unit_id.map(area_by_unit).astype(float) * ctx.cell_area_m2
    cells["pumping_reduction_depth_2012_m"] = reduction_depth
    cells["management_leverage"] = leverage
    cells.to_csv(UNIT_CELLS_CSV, index=False)
    seeds.to_csv(SEED_CELL_LEVERAGE_CSV, index=False)
    print(f"Saved: {UNIT_CELLS_CSV.relative_to(ROOT)}")
    print(f"Saved: {SEED_CELL_LEVERAGE_CSV.relative_to(ROOT)}")
    return cells, seeds


def _scenario_cache_complete(path: Path, strategies: tuple[str, ...], budgets: tuple[int, ...], required: set[str]) -> bool:
    if not path.exists():
        return False
    frame = pd.read_csv(path, dtype={"seed": str, "strategy": str})
    complete = (
        required.issubset(frame.columns)
        and len(frame) == len(strategies) * len(budgets) * len(SEEDS)
        and set(frame.strategy) == set(strategies)
        and set(frame.seed) == set(SEEDS)
        and set(frame.budget_nominal.astype(int)) == set(budgets)
    )
    if not complete:
        return False
    informed = frame[frame.strategy == "Leverage guided"]
    return informed.empty or set(informed.ranking_version.astype(str)) == {RANKING_VERSION}


def _run_window_grid(
    ctx: ExperimentContext,
    window: Window,
    strategies: tuple[str, ...],
    budgets: tuple[int, ...],
    metrics_path: Path,
    allocations_path: Path,
    include_recovery: bool,
    recovery_summary_path: Path | None = None,
    representative_cells_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    cell_volume, pumping_baseline = ctx.capture_window_pumping(window)
    initial_idx, intervention_idx, output_idx = ctx.window_indices(window)
    endpoint_local = {month: int(np.flatnonzero(ctx.months[output_idx] == month)[0]) for month in window.endpoint_months}
    ctx.load_models()
    ctx.restore_window(window, pumping_baseline)
    baseline = {seed: ctx.forecast_seed(window, seed) for seed in SEEDS}
    print(f"{window.name}: completed five paired 0% baselines")

    metric_rows = []
    allocation_frames = []
    recovery_rows = []
    representative_frames = []
    max_mismatch = 0.0

    def append_result(strategy, budget, seed, info, response):
        benefit = ctx.storage_benefit(response)
        row = {
            "timing": window.name,
            "strategy": strategy,
            "budget_nominal": int(budget),
            "seed": seed,
            **info,
        }
        for month, local_idx in endpoint_local.items():
            suffix = month.replace("-", "")
            row[f"benefit_{suffix}_m3"] = float(benefit[local_idx])
            row[f"eta_{suffix}"] = float(benefit[local_idx] / info["actual_dV_m3"]) if info["actual_dV_m3"] > 0 else np.nan
        metric_rows.append(row)
        if include_recovery:
            arrays = ctx.recovery_arrays(response, window)
            for summary in ctx.recovery_summary(arrays, response):
                recovery_rows.append({"strategy": strategy, "budget_nominal": int(budget), "seed": seed, "actual_dV_m3": info["actual_dV_m3"], **summary})
            if budget == REPORT_BUDGET and strategy == "Uniform" and representative_cells_path is not None:
                def labels_from_index(values):
                    result = np.full(len(values), "", dtype=object)
                    finite = np.isfinite(values)
                    result[finite] = ctx.months[values[finite].astype(int)]
                    return result
                representative_frames.append(
                    pd.DataFrame(
                        {
                            "cell_id": ctx.grid_ids,
                            "response_class": ctx.response_class,
                            "seed": seed,
                            "budget_nominal": int(budget),
                            "actual_dV_m3": info["actual_dV_m3"],
                            "T50_base": arrays["T50_base"],
                            "T50_management": arrays["T50_management"],
                            "delta_T50": arrays["delta_T50"],
                            "T50_base_month": labels_from_index(arrays["baseline_t50_index"]),
                            "T50_management_month": labels_from_index(arrays["management_t50_index"]),
                            "reached_before_intervention": arrays["reached_before_intervention"],
                            "reached_during_simulation": arrays["reached_during_simulation"],
                            "not_reached_by_end": arrays["not_reached_by_end"],
                            "valid_flag": arrays["valid"],
                        }
                    )
                )

    try:
        for budget in budgets:
            if budget == 0:
                zero = np.zeros((len(output_idx), len(ctx.grid_ids)), dtype=np.float32)
                for strategy in strategies:
                    cell_fraction, alloc, info = ctx.allocate(strategy, budget, cell_volume)
                    allocation_frames.append(alloc)
                    max_mismatch = max(max_mismatch, info["relative_mismatch"])
                    for seed in SEEDS:
                        append_result(strategy, budget, seed, info, zero)
                continue

            all_capped = all(ctx.allocate(strategy, budget, cell_volume)[2]["feasibility_capped"] for strategy in strategies)
            if all_capped:
                cell_fraction, _, info = ctx.allocate("Uniform", budget, cell_volume)
                ctx.apply_reduction(window, cell_fraction, pumping_baseline)
                response_by_seed = {seed: baseline[seed] - ctx.forecast_seed(window, seed) for seed in SEEDS}
                ctx.restore_window(window, pumping_baseline)
                for strategy in strategies:
                    _, alloc, strategy_info = ctx.allocate(strategy, budget, cell_volume)
                    allocation_frames.append(alloc)
                    max_mismatch = max(max_mismatch, strategy_info["relative_mismatch"])
                    for seed in SEEDS:
                        append_result(strategy, budget, seed, strategy_info, response_by_seed[seed])
                print(f"{window.name}: nominal {budget}% reached the common 50% Cell cap")
                continue

            for strategy in strategies:
                cell_fraction, alloc, info = ctx.allocate(strategy, budget, cell_volume)
                allocation_frames.append(alloc)
                max_mismatch = max(max_mismatch, info["relative_mismatch"])
                ctx.apply_reduction(window, cell_fraction, pumping_baseline)
                for seed in SEEDS:
                    response = baseline[seed] - ctx.forecast_seed(window, seed)
                    append_result(strategy, budget, seed, info, response)
                ctx.restore_window(window, pumping_baseline)
                print(f"{window.name}: completed {strategy}, nominal {budget}%")
    finally:
        ctx.restore_window(window, pumping_baseline)

    metrics = pd.DataFrame(metric_rows)
    allocations = pd.concat(allocation_frames, ignore_index=True)
    metrics.to_csv(metrics_path, index=False)
    allocations.to_csv(allocations_path, index=False)
    if include_recovery and recovery_summary_path is not None:
        pd.DataFrame(recovery_rows).to_csv(recovery_summary_path, index=False)
    if representative_cells_path is not None and representative_frames:
        pd.concat(representative_frames, ignore_index=True).to_csv(representative_cells_path, index=False)
    print(f"Maximum actual-dV allocation mismatch: {max_mismatch:.3e}")
    return metrics, pd.DataFrame(recovery_rows) if include_recovery else None


def ensure_2012_scenarios(force: bool = False) -> pd.DataFrame:
    metrics_path = RESULTS_DIR / "drought_strategy_seed_metrics.csv"
    allocations_path = RESULTS_DIR / "drought_cell_allocations.csv"
    strategies = ("Uniform", "High pumping", "Leverage guided")
    required = {"strategy", "budget_nominal", "seed", "ranking_version", "actual_dV_m3", "benefit_201210_m3", "benefit_201212_m3", "benefit_201304_m3"}
    if not force and _scenario_cache_complete(metrics_path, strategies, BUDGETS, required):
        print("Loaded complete May-Oct 2012 scenario cache.")
        return pd.read_csv(metrics_path, dtype={"seed": str, "strategy": str})
    metrics, _ = _run_window_grid(
        context(), WINDOW_2012, strategies, BUDGETS, metrics_path, allocations_path, include_recovery=False
    )
    return metrics


def ensure_2013_scenarios(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = RESULTS_DIR / "recovery_strategy_seed_metrics.csv"
    allocations_path = RESULTS_DIR / "recovery_cell_allocations.csv"
    recovery_path = RESULTS_DIR / "recovery_timing_by_class.csv"
    cells_path = RESULTS_DIR / "recovery_time_cells.csv.gz"
    strategies = ("Uniform", "High pumping", "Leverage guided")
    required = {"strategy", "budget_nominal", "seed", "ranking_version", "actual_dV_m3", "benefit_201310_m3", "benefit_201312_m3", "benefit_201404_m3"}
    expected_recovery = len(strategies) * len(BUDGETS) * len(SEEDS) * (1 + len(CLASS_ORDER))
    cache_ok = not force and _scenario_cache_complete(metrics_path, strategies, BUDGETS, required)
    if cache_ok and recovery_path.exists() and cells_path.exists():
        recovery = pd.read_csv(recovery_path, dtype={"seed": str, "strategy": str})
        if len(recovery) == expected_recovery:
            print("Loaded complete May-Oct 2013 scenario and recovery caches.")
            return pd.read_csv(metrics_path, dtype={"seed": str, "strategy": str}), recovery
    metrics, recovery = _run_window_grid(
        context(),
        WINDOW_2013,
        strategies,
        BUDGETS,
        metrics_path,
        allocations_path,
        include_recovery=True,
        recovery_summary_path=recovery_path,
        representative_cells_path=cells_path,
    )
    return metrics, recovery


def ensure_uniform_storage_trajectory(force: bool = False) -> pd.DataFrame:
    path = RESULTS_DIR / "uniform_storage_trajectory.csv"
    if path.exists() and not force:
        data = pd.read_csv(path, dtype={"seed": str})
        counts = data.groupby("intervention_period").month.nunique().to_dict()
        if sorted(counts.values()) == [12, 24] and len(data) == len(SEEDS) * 36:
            return data
    ctx = context()
    drought_extended = Window(
        name=WINDOW_2012.name,
        initial_month=WINDOW_2012.initial_month,
        intervention_start=WINDOW_2012.intervention_start,
        intervention_end=WINDOW_2012.intervention_end,
        final_month="2014-04",
        endpoint_months=("2012-10", "2012-12", "2013-04", "2013-10", "2014-04"),
    )
    frames = []
    for window in (drought_extended, WINDOW_2013):
        cell_volume, pumping_baseline = ctx.capture_window_pumping(window)
        _, _, output_idx = ctx.window_indices(window)
        ctx.load_models()
        ctx.restore_window(window, pumping_baseline)
        baseline = {seed: ctx.forecast_seed(window, seed) for seed in SEEDS}
        fractions, _, info = ctx.allocate("Uniform", REPORT_BUDGET, cell_volume)
        try:
            ctx.apply_reduction(window, fractions, pumping_baseline)
            for seed in SEEDS:
                response = baseline[seed] - ctx.forecast_seed(window, seed)
                frames.append(pd.DataFrame({
                    "intervention_period": window.name,
                    "seed": seed,
                    "month": ctx.months[output_idx],
                    "storage_benefit_m3": ctx.storage_benefit(response),
                    "actual_dV_m3": info["actual_dV_m3"],
                }))
        finally:
            ctx.restore_window(window, pumping_baseline)
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(path, index=False)
    return data


def prepare_equal_volume_results(force: bool = False) -> dict[str, Path]:
    """Run the two management windows and save the compact tables used by figures."""
    drought = ensure_2012_scenarios(force=force)
    recovery, _ = ensure_2013_scenarios(force=force)
    ensure_uniform_storage_trajectory(force=force)
    ctx = context()
    allocation_outputs = {}
    for label, window in (("drought", WINDOW_2012), ("recovery", WINDOW_2013)):
        cell_volume, _ = ctx.capture_window_pumping(window)
        fractions, _, info = ctx.allocate("Leverage guided", REPORT_BUDGET, cell_volume)
        allocation = ctx.grid[["grid_id", "row", "col", "x", "y"]].copy()
        allocation["unit_id"] = ctx.unit_by_grid
        allocation["response_class"] = ctx.response_class
        allocation["reduction_fraction"] = fractions
        allocation["reduction_m3"] = cell_volume * fractions
        if not np.isclose(allocation.reduction_m3.sum(), info["actual_dV_m3"], rtol=2e-7, atol=2.0):
            raise AssertionError("Saved Cell allocation does not match the regional budget.")
        allocation_path = RESULTS_DIR / f"{label}_leverage_cells.csv.gz"
        allocation.to_csv(allocation_path, index=False)
        allocation_outputs[f"{label}_leverage_cells"] = allocation_path
    dec_idx = ctx.month_lookup["2012-12"]
    deficit = float(
        np.sum(
            ctx.storage_weights_m2 * (np.asarray(ctx.wtd[dec_idx], float) - ctx.pre_wtd),
            dtype=np.float64,
        )
    )
    offset = drought.rename(columns={"benefit_201212_m3": "storage_benefit_m3"})[
        ["strategy", "budget_nominal", "actual_dV_m3", "seed", "storage_benefit_m3"]
    ].copy()
    offset["drought_deficit_m3"] = deficit
    offset["fraction_offset"] = offset.storage_benefit_m3 / deficit
    offset.to_csv(RESULTS_DIR / "drought_deficit_offset.csv", index=False)

    drought_timing = drought[drought.strategy == "Uniform"].rename(columns={
        "benefit_201210_m3": "benefit_end_m3",
        "benefit_201304_m3": "benefit_6m_m3",
        "eta_201210": "eta_end",
        "eta_201304": "eta_6m",
    }).assign(timing="2012 drought period")
    recovery_timing = recovery[recovery.strategy == "Uniform"].rename(columns={
        "benefit_201310_m3": "benefit_end_m3",
        "benefit_201404_m3": "benefit_6m_m3",
        "eta_201310": "eta_end",
        "eta_201404": "eta_6m",
    }).assign(timing="2013 recovery period")
    columns = [
        "timing", "budget_nominal", "actual_dV_m3", "seed",
        "benefit_end_m3", "benefit_6m_m3", "eta_end", "eta_6m",
    ]
    pd.concat([drought_timing[columns], recovery_timing[columns]], ignore_index=True).to_csv(
        RESULTS_DIR / "timing_seed_metrics.csv", index=False
    )
    return {
        "drought_metrics": RESULTS_DIR / "drought_strategy_seed_metrics.csv",
        "drought_allocations": RESULTS_DIR / "drought_cell_allocations.csv",
        "recovery_metrics": RESULTS_DIR / "recovery_strategy_seed_metrics.csv",
        "recovery_allocations": RESULTS_DIR / "recovery_cell_allocations.csv",
        "recovery_timing": RESULTS_DIR / "recovery_timing_by_class.csv",
        "recovery_cells": RESULTS_DIR / "recovery_time_cells.csv.gz",
        "drought_offset": RESULTS_DIR / "drought_deficit_offset.csv",
        "timing_comparison": RESULTS_DIR / "timing_seed_metrics.csv",
        "uniform_storage_trajectory": RESULTS_DIR / "uniform_storage_trajectory.csv",
        **allocation_outputs,
    }


def _summary_by_strategy(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    rows = []
    for (strategy, budget), group in frame.groupby(["strategy", "budget_nominal"], sort=False):
        mean, low, high = mean_pi75(group[value])
        rows.append(
            {
                "strategy": strategy,
                "budget_nominal": int(budget),
                "actual_dV_m3": float(group.actual_dV_m3.mean()),
                "mean": mean,
                "pi75_low": low,
                "pi75_high": high,
                "min": float(group[value].min()),
                "max": float(group[value].max()),
            }
        )
    return pd.DataFrame(rows)


def _paired(frame: pd.DataFrame, a: str, b: str, value: str, label: str) -> pd.DataFrame:
    left = frame.loc[frame.strategy == a, ["budget_nominal", "seed", "actual_dV_m3", value]].rename(columns={value: "a"})
    right = frame.loc[frame.strategy == b, ["budget_nominal", "seed", value]].rename(columns={value: "b"})
    out = left.merge(right, on=["budget_nominal", "seed"], validate="one_to_one")
    out["comparison"] = label
    out["difference"] = out.a - out.b
    return out


def _maximum_relative_volume_spread(frame: pd.DataFrame, maximum_budget: int = MAIN_MAX_BUDGET) -> float:
    data = frame[frame.budget_nominal <= maximum_budget]
    grouped = data.groupby("budget_nominal").actual_dV_m3.agg(["min", "max", "mean"])
    return float(((grouped["max"] - grouped["min"]) / grouped["mean"].replace(0, np.nan)).max())


def _nonmonotonic_group_count(frame: pd.DataFrame, value: str, tolerance: float = 1.0e3) -> int:
    count = 0
    for _, group in frame[frame.budget_nominal <= MAIN_MAX_BUDGET].groupby(["strategy", "seed"]):
        ordered = group.sort_values("actual_dV_m3")
        count += int(np.sum(np.diff(ordered[value].to_numpy(float)) < -tolerance))
    return count


def _style_axes(axes) -> None:
    for ax in np.ravel(axes):
        ax.tick_params(direction="out", length=2.4, width=0.65, pad=2.0)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)


def _save_publication_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(RESULTS_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(RESULTS_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _mean_interval_curve(ax, data: pd.DataFrame, x, scale: float, color: str, label: str) -> None:
    y = data["mean"].to_numpy(float) / scale
    low = data.pi75_low.to_numpy(float) / scale
    high = data.pi75_high.to_numpy(float) / scale
    ax.errorbar(
        x, y, yerr=np.vstack([y - low, high - y]), fmt="none",
        ecolor=color, elinewidth=0.75, capsize=1.4, capthick=0.65,
        alpha=0.48, zorder=1,
    )
    ax.plot(x, y, color=color, lw=1.45, marker="o", ms=2.35, label=label, zorder=2)


def _horizontal_interval(ax, y: float, values, color: str, marker: str = "o", open_marker: bool = False) -> tuple[float, float, float]:
    mean, low, high = mean_pi75(values)
    ax.errorbar(
        mean, y, xerr=[[mean - low], [high - mean]], fmt=marker, ms=4.0,
        mfc="white" if open_marker else color, mec=color, mew=0.8,
        ecolor=color, elinewidth=0.95, capsize=2.0, capthick=0.8, zorder=3,
    )
    return mean, low, high


def build_figs26(force: bool = False, output_path: Path | None = None) -> dict:
    metrics = ensure_2012_scenarios(force=force)
    ctx = context()
    dec_idx = ctx.month_lookup["2012-12"]
    dec_deficit = float(np.sum(ctx.storage_weights_m2 * (np.asarray(ctx.wtd[dec_idx], float) - ctx.pre_wtd), dtype=np.float64))
    if not np.isclose(dec_deficit, 14_849_654_274.015465, rtol=0, atol=2e5):
        raise AssertionError("December 2012 drought-deficit denominator changed.")

    offset = metrics.rename(columns={"benefit_201212_m3": "storage_benefit_m3"})[
        ["strategy", "budget_nominal", "actual_dV_m3", "seed", "storage_benefit_m3"]
    ].copy()
    offset["drought_deficit_m3"] = dec_deficit
    offset["fraction_offset"] = offset.storage_benefit_m3 / dec_deficit
    offset_path = RESULTS_DIR / "drought_deficit_offset.csv"
    offset.to_csv(offset_path, index=False)

    legacy_reference = None

    core = ("Uniform", "High pumping", "Leverage guided")
    paired = pd.concat(
        [
            _paired(offset, "Leverage guided", "Uniform", "fraction_offset", "Leverage - Uniform"),
            _paired(offset, "Leverage guided", "High pumping", "fraction_offset", "Leverage - High pumping"),
        ],
        ignore_index=True,
    )
    paired["positive_direction"] = paired.difference > 0
    summary = _summary_by_strategy(offset, "fraction_offset")

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.35), dpi=300, gridspec_kw={"width_ratios": [0.88, 1.12]})

    report = offset[(offset.budget_nominal == REPORT_BUDGET) & offset.strategy.isin(core)]
    for i, strategy in enumerate(core):
        values = 100 * report.loc[report.strategy == strategy, "fraction_offset"].to_numpy(float)
        _horizontal_interval(axes[0], i, values, STRATEGY_COLORS[strategy])
    axes[0].set_yticks(range(len(core)), ["Uniform", "High pumping", "Leverage"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Deficit offset (%)")
    axes[0].set_title("a  Response at 20%", loc="left", fontweight="bold")

    pair_summary = _summary_by_strategy(paired.rename(columns={"comparison": "strategy"}), "difference")
    shown_budgets = (10, 20, 30)
    contrast_specs = (("Leverage - Uniform", "#486A9A", -0.12), ("Leverage - High pumping", "#B77A61", 0.12))
    for comparison, color, offset_y in contrast_specs:
        for j, budget in enumerate(shown_budgets):
            row = pair_summary[(pair_summary.strategy == comparison) & (pair_summary.budget_nominal == budget)].iloc[0]
            mean, low, high = 100 * row["mean"], 100 * row.pi75_low, 100 * row.pi75_high
            label = ("Leverage - Uniform" if comparison.endswith("Uniform") else "Leverage - High pumping") if j == 0 else None
            axes[1].errorbar(mean, j + offset_y, xerr=[[mean - low], [high - mean]], fmt="o", ms=3.6, color=color, ecolor=color, capsize=1.8, elinewidth=0.9, label=label)
    axes[1].axvline(0, color="0.35", lw=0.7, ls="--")
    axes[1].set_yticks(range(len(shown_budgets)), [f"{b}%" for b in shown_budgets])
    axes[1].invert_yaxis()
    axes[1].set_title("b  Paired differences", loc="left", fontweight="bold")
    axes[1].set_xlabel("Additional deficit offset (%)")
    axes[1].set_ylabel("Conservation budget")
    axes[1].legend(loc="best", fontsize=5.4, handlelength=1.5, labelspacing=0.25)
    _style_axes(axes)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
    fig.tight_layout(w_pad=1.05)
    if output_path is None:
        _save_publication_figure(fig, "FigS26")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    row20 = offset[(offset.strategy == "Leverage guided") & (offset.budget_nominal == REPORT_BUDGET)]
    allocation_mismatch = float(metrics.relative_mismatch.max())
    qa = {
        "experiment": WINDOW_2012.name,
        "allocation_scale": ALLOCATION_SCALE,
        "leverage_score_scale": "1 km Cell",
        "leverage_source_experiment": "own-Unit intervention with Cell-resolved response",
        "ranking_design": "one shared Cell ranking from the mean of all five seed-specific scores",
        "active_cells": int(len(ctx.grid_ids)),
        "management_units": int(len(ctx.unit_ids)),
        "drought_deficit_m3": dec_deficit,
        "report_budget_nominal": REPORT_BUDGET,
        "report_actual_dV_m3": float(row20.actual_dV_m3.mean()),
        "report_five_seed_mean_offset": float(row20.fraction_offset.mean()),
        "report_seed_min_offset": float(row20.fraction_offset.min()),
        "report_seed_max_offset": float(row20.fraction_offset.max()),
        "actual_dV_range_m3": [float(metrics.actual_dV_m3.min()), float(metrics.actual_dV_m3.max())],
        "leverage_minus_uniform_positive_seeds": int(paired[(paired.comparison == "Leverage - Uniform") & (paired.budget_nominal == REPORT_BUDGET)].positive_direction.sum()),
        "leverage_minus_high_positive_seeds": int(paired[(paired.comparison == "Leverage - High pumping") & (paired.budget_nominal == REPORT_BUDGET)].positive_direction.sum()),
        "maximum_relative_actual_dV_mismatch": allocation_mismatch,
        "maximum_relative_strategy_dV_spread_0_to_30pct": _maximum_relative_volume_spread(metrics),
        "nonmonotonic_Dec2012_benefit_steps_0_to_30pct": _nonmonotonic_group_count(metrics, "benefit_201212_m3"),
        "legacy_full_year_sensitivity": legacy_reference,
        "note": "All targeted strategies rank and reduce 1 km Cells. Leverage-guided management uses one shared ranking from the five-seed mean Cell score.",
    }
    print(json.dumps(qa, indent=2))
    return qa


def build_figs27(force: bool = False, output_path: Path | None = None) -> dict:
    metrics, recovery = ensure_2013_scenarios(force=force)
    cells = pd.read_csv(RESULTS_DIR / "recovery_time_cells.csv.gz", dtype={"seed": str, "response_class": str})
    report = recovery[(recovery.strategy == "Uniform") & (recovery.budget_nominal == REPORT_BUDGET)].copy()
    eligible_cells = cells[cells.valid_flag & ~cells.reached_before_intervention]
    comparable = eligible_cells[np.isfinite(eligible_cells.delta_T50)]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55), dpi=300)
    shown_thresholds = np.array([1, 6, 12, 24, 36, 60])
    exceedance = []
    for _, group in comparable.groupby("seed"):
        gains = group.loc[group.delta_T50 > 0, "delta_T50"].to_numpy(float)
        exceedance.append([np.mean(gains >= threshold) for threshold in shown_thresholds])
    exceedance = np.asarray(exceedance, float)
    exceedance_mean = exceedance.mean(axis=0)
    exceedance_low = exceedance_mean - PI75_NORMAL_FACTOR * exceedance.std(axis=0, ddof=0)
    exceedance_high = exceedance_mean + PI75_NORMAL_FACTOR * exceedance.std(axis=0, ddof=0)
    bar_x = np.arange(len(shown_thresholds))
    bar_mean = 100 * exceedance_mean
    axes[0].bar(
        bar_x, bar_mean, width=0.68, color="#486A9A",
        edgecolor="#365579", linewidth=0.45,
    )
    axes[0].errorbar(
        bar_x, bar_mean,
        yerr=np.vstack([
            100 * (exceedance_mean - exceedance_low),
            100 * (exceedance_high - exceedance_mean),
        ]),
        fmt="none", color="#365579", ecolor="#365579",
        elinewidth=0.8, capsize=1.6, capthick=0.7,
    )
    axes[0].set_xticks(bar_x, shown_thresholds)
    axes[0].set_xlim(-0.6, len(bar_x) - 0.4)
    axes[0].set_ylim(0, 105)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].set_title("A  Recovery-time gains", loc="left", fontweight="bold")
    axes[0].set_xlabel("Time gained (months)")
    axes[0].set_ylabel("Improved Cells (%)")
    ctx = context()
    month_range = ctx.indices("2013-05", "2014-04")
    dates = pd.to_datetime(ctx.months[month_range])
    valid_first = cells.drop_duplicates("cell_id")
    eligibility = valid_first.valid_flag & ~valid_first.reached_before_intervention
    denominator = int(eligibility.sum())
    base_index = valid_first.loc[eligibility, "T50_base_month"].replace("", np.nan)
    base_dates = pd.to_datetime(base_index, errors="coerce")
    base_curve = np.array([(base_dates <= date).mean() for date in dates])
    managed_curves = []
    for seed, group in cells.groupby("seed"):
        mask = group.valid_flag & ~group.reached_before_intervention
        managed_dates = pd.to_datetime(group.loc[mask, "T50_management_month"].replace("", np.nan), errors="coerce")
        managed_curves.append([(managed_dates <= date).mean() for date in dates])
    managed_curves = np.asarray(managed_curves, float)
    mean_curve = managed_curves.mean(axis=0)
    low_curve = mean_curve - PI75_NORMAL_FACTOR * managed_curves.std(axis=0, ddof=0)
    high_curve = mean_curve + PI75_NORMAL_FACTOR * managed_curves.std(axis=0, ddof=0)
    axes[1].plot(dates, 100 * base_curve, color="0.25", lw=1.55, label="Baseline")
    axes[1].fill_between(dates, 100 * low_curve, 100 * high_curve, color="#486A9A", alpha=0.14, lw=0)
    axes[1].plot(dates, 100 * mean_curve, color="#486A9A", lw=1.45, label="Uniform")
    axes[1].set_ylim(0, 45)
    axes[1].set_yticks([0, 10, 20, 30, 40])
    axes[1].set_ylabel(r"Cells reaching $D_{50}$ (%)")
    axes[1].set_xlabel("Time")
    axes[1].set_title("B  All eligible Cells", loc="left", fontweight="bold")
    axes[1].legend(loc="upper left", fontsize=6.0, handlelength=1.6, labelspacing=0.25)

    slow = cells[(cells.response_class == "Slow recovery") & cells.valid_flag & ~cells.reached_before_intervention]
    slow_base_dates = pd.to_datetime(slow.drop_duplicates("cell_id").T50_base_month.replace("", np.nan), errors="coerce")
    slow_base_curve = np.array([(slow_base_dates <= date).mean() for date in dates])
    slow_managed_curves = []
    for seed, group in slow.groupby("seed"):
        managed_dates = pd.to_datetime(group.T50_management_month.replace("", np.nan), errors="coerce")
        slow_managed_curves.append([(managed_dates <= date).mean() for date in dates])
    slow_managed_curves = np.asarray(slow_managed_curves, float)
    slow_mean = slow_managed_curves.mean(axis=0)
    slow_low = slow_mean - PI75_NORMAL_FACTOR * slow_managed_curves.std(axis=0, ddof=0)
    slow_high = slow_mean + PI75_NORMAL_FACTOR * slow_managed_curves.std(axis=0, ddof=0)
    axes[2].plot(dates, 100 * slow_base_curve, color="0.25", lw=1.55)
    axes[2].fill_between(dates, 100 * slow_low, 100 * slow_high, color=CLASS_COLORS["Slow recovery"], alpha=0.13, lw=0)
    axes[2].plot(dates, 100 * slow_mean, color=CLASS_COLORS["Slow recovery"], lw=1.45)
    axes[2].set_ylim(0, 45)
    axes[2].set_yticks([0, 10, 20, 30, 40])
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel(r"Cells reaching $D_{50}$ (%)")
    axes[2].set_title("C  Slow-recovery Cells", loc="left", fontweight="bold")
    _style_axes(axes)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
    for ax in axes[1:]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.set_xlim(dates.min(), dates.max())
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.27, top=0.82, wspace=0.22)
    if output_path is None:
        _save_publication_figure(fig, "FigS27")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    all_rows = report[report.response_class == "All"]
    qa = {
        "experiment": WINDOW_2013.name,
        "allocation_scale": ALLOCATION_SCALE,
        "active_cells": int(len(ctx.grid_ids)),
        "management_units": int(len(ctx.unit_ids)),
        "report_budget_nominal": REPORT_BUDGET,
        "report_actual_dV_m3": float(metrics[(metrics.strategy == "Uniform") & (metrics.budget_nominal == REPORT_BUDGET)].actual_dV_m3.mean()),
        "valid_cells": int(cells.drop_duplicates("cell_id").valid_flag.sum()),
        "eligible_cells_unrecovered_by_May2013": denominator,
        "eligible_fraction": float(denominator / len(ctx.grid_ids)),
        "fast_recovery_cells_still_unrecovered_by_May2013": int(report[report.response_class == "Fast recovery"].eligible_unrecovered_May2013_cells.max()),
        "five_seed_median_delta_T50_months_mean": float(all_rows.median_delta_T50_months.mean()),
        "five_seed_median_delta_T50_months_range": [float(all_rows.median_delta_T50_months.min()), float(all_rows.median_delta_T50_months.max())],
        "five_seed_fraction_eligible_positive_mean": float(all_rows.fraction_eligible_delta_T50_positive.mean()),
        "not_reached_by_Apr2014_mean_cells": float(all_rows.not_reached_by_end_cells.mean()),
        "negative_delta_T50_cell_seed_pairs": int(np.sum(np.isfinite(eligible_cells.delta_T50) & (eligible_cells.delta_T50 < 0))),
        "actual_dV_range_m3": [float(metrics.actual_dV_m3.min()), float(metrics.actual_dV_m3.max())],
        "maximum_relative_actual_dV_mismatch": float(metrics.relative_mismatch.max()),
        "maximum_relative_strategy_dV_spread_0_to_30pct": _maximum_relative_volume_spread(metrics),
        "nonmonotonic_end_benefit_steps_0_to_30pct": _nonmonotonic_group_count(metrics, "benefit_201310_m3"),
        "censoring": "T50 values not reached by Apr 2014 remain missing and are flagged not_reached_by_end.",
    }
    print(json.dumps(qa, indent=2))
    return qa


def build_figs28(force: bool = False, output_path: Path | None = None) -> dict:
    timing_path = RESULTS_DIR / "timing_seed_metrics.csv"
    if timing_path.exists() and not force:
        timing = pd.read_csv(timing_path, dtype={"seed": str, "timing": str})
    else:
        m2012 = ensure_2012_scenarios(force=force)
        m2013, _ = ensure_2013_scenarios(force=force)
        drought = m2012[m2012.strategy == "Uniform"].rename(columns={
            "benefit_201210_m3": "benefit_end_m3", "benefit_201304_m3": "benefit_6m_m3",
            "eta_201210": "eta_end", "eta_201304": "eta_6m",
        }).assign(timing="2012 drought period")
        recovery = m2013[m2013.strategy == "Uniform"].rename(columns={
            "benefit_201310_m3": "benefit_end_m3", "benefit_201404_m3": "benefit_6m_m3",
            "eta_201310": "eta_end", "eta_201404": "eta_6m",
        }).assign(timing="2013 recovery period")
        columns = ["timing", "budget_nominal", "actual_dV_m3", "seed", "benefit_end_m3", "benefit_6m_m3", "eta_end", "eta_6m"]
        timing = pd.concat([drought[columns], recovery[columns]], ignore_index=True)
        timing.to_csv(timing_path, index=False)
    common = timing[timing.budget_nominal <= MAIN_MAX_BUDGET]
    volume_spread = common.groupby(["budget_nominal", "seed"]).actual_dV_m3.agg(lambda x: x.max() - x.min())
    max_relative = float((volume_spread / common.groupby(["budget_nominal", "seed"]).actual_dV_m3.mean().replace(0, np.nan)).max())
    if max_relative > 2e-7:
        raise AssertionError("2012 and 2013 timing experiments are not matched by actual dV.")

    paired_frames = []
    for metric in ("eta_end", "eta_6m"):
        left = timing[timing.timing == "2013 recovery period"][["budget_nominal", "seed", "actual_dV_m3", metric]].rename(columns={metric: "recovery"})
        right = timing[timing.timing == "2012 drought period"][["budget_nominal", "seed", metric]].rename(columns={metric: "drought"})
        paired = left.merge(right, on=["budget_nominal", "seed"], validate="one_to_one")
        paired["metric"] = metric
        paired["difference"] = paired.recovery - paired.drought
        paired_frames.append(paired)
    differences = pd.concat(paired_frames, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.55), dpi=300)
    timing_colors = {"2012 drought period": "#B77A61", "2013 recovery period": "#486A9A"}
    timing_labels = {"2012 drought period": "2012 drought", "2013 recovery period": "2013 recovery"}
    for panel, metric, title in (
        (axes[0], "eta_end", "A  Intervention end"),
        (axes[1], "eta_6m", "B  Six months later"),
    ):
        summary = _summary_by_strategy(timing.rename(columns={"timing": "strategy"}), metric)
        for name in timing_colors:
            data = summary[(summary.strategy == name) & (summary.budget_nominal <= MAIN_MAX_BUDGET)].sort_values("actual_dV_m3")
            x = data.actual_dV_m3.to_numpy(float) / 1e9
            _mean_interval_curve(panel, data, x, 1.0, timing_colors[name], timing_labels[name])
        panel.set_title(title, loc="left", fontweight="bold")
        panel.set_xlabel(r"Pumping reduction ($10^9$ m$^3$)")
        panel.set_xlim(0.6, 5.2)
        panel.set_xticks([1, 2, 3, 4, 5])
        panel.set_ylim(0.05, 0.40)
        panel.set_yticks([0.1, 0.2, 0.3, 0.4])
    axes[0].set_ylabel("Storage benefit per unit\npumping reduction")
    axes[1].set_ylabel("")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        fontsize=6.0,
        handlelength=1.7,
        columnspacing=1.0,
        labelspacing=0.2,
    )
    _style_axes(axes)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.22, top=0.84, wspace=0.18)
    if output_path is None:
        _save_publication_figure(fig, "FigS28")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    report = differences[differences.budget_nominal == REPORT_BUDGET]
    qa = {
        "comparison": "Uniform May-Oct 2013 versus Uniform May-Oct 2012 at matched actual dV",
        "allocation_scale": ALLOCATION_SCALE,
        "report_budget_nominal": REPORT_BUDGET,
        "report_actual_dV_m3": float(report.actual_dV_m3.mean()),
        "eta_end_difference_five_seed_mean": float(report[report.metric == "eta_end"].difference.mean()),
        "eta_end_positive_seeds": int(report[report.metric == "eta_end"].difference.gt(0).sum()),
        "eta_6m_difference_five_seed_mean": float(report[report.metric == "eta_6m"].difference.mean()),
        "eta_6m_positive_seeds": int(report[report.metric == "eta_6m"].difference.gt(0).sum()),
        "maximum_relative_actual_dV_mismatch": max_relative,
        "nonmonotonic_2012_end_benefit_steps_0_to_30pct": _nonmonotonic_group_count(
            timing[timing.timing == "2012 drought period"].assign(strategy="Uniform"), "benefit_end_m3"
        ),
        "nonmonotonic_2013_end_benefit_steps_0_to_30pct": _nonmonotonic_group_count(
            timing[timing.timing == "2013 recovery period"].assign(strategy="Uniform"), "benefit_end_m3"
        ),
    }
    print(json.dumps(qa, indent=2))
    return qa


def build_figs29(force: bool = False, output_path: Path | None = None) -> dict:
    metrics, recovery = ensure_2013_scenarios(force=force)
    ctx = context()

    paired_benefit = pd.concat(
        [
            _paired(metrics, "Leverage guided", "Uniform", "benefit_201310_m3", "Leverage - Uniform"),
            _paired(metrics, "Leverage guided", "High pumping", "benefit_201310_m3", "Leverage - High pumping"),
        ],
        ignore_index=True,
    )
    paired_benefit["positive_direction"] = paired_benefit.difference > 0

    recovery_paired_frames = []
    for response_class in ("All", "Slow recovery"):
        class_data = recovery[recovery.response_class == response_class]
        for value in ("median_delta_T50_months", "fraction_eligible_delta_T50_positive"):
            for control in ("Uniform", "High pumping"):
                comparison = _paired(
                    class_data,
                    "Leverage guided",
                    control,
                    value,
                    f"2012-informed - {control}",
                )
                comparison["response_class"] = response_class
                comparison["metric"] = value
                comparison["positive_direction"] = comparison.difference > 0
                recovery_paired_frames.append(comparison)
    paired_recovery = pd.concat(recovery_paired_frames, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.65), dpi=300)
    strategies = ("Uniform", "High pumping", "Leverage guided")
    display_labels = {
        "Uniform": "Uniform",
        "High pumping": "High pumping",
        "Leverage guided": "Leverage",
    }
    summary = _summary_by_strategy(metrics, "benefit_201310_m3")
    for strategy in strategies:
        data = summary[(summary.strategy == strategy) & (summary.budget_nominal <= MAIN_MAX_BUDGET)].sort_values("actual_dV_m3")
        _mean_interval_curve(
            axes[0], data, data.actual_dV_m3.to_numpy(float), 1.0,
            STRATEGY_COLORS[strategy], display_labels[strategy],
        )
    axes[0].set_title("A  Post-drought benefit", loc="left", fontweight="bold")
    axes[0].set_xlabel("Pumping reduction")
    axes[0].set_ylabel("Storage benefit")
    axes[0].ticklabel_format(axis="both", style="sci", scilimits=(9, 9), useMathText=True)
    axes[0].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    axes[0].legend(loc="upper left", fontsize=5.8, handlelength=1.8, labelspacing=0.25)

    benefit20 = metrics[(metrics.budget_nominal == REPORT_BUDGET) & metrics.strategy.isin(strategies)]
    for i, strategy in enumerate(strategies):
        values = benefit20.loc[benefit20.strategy == strategy, "benefit_201310_m3"].to_numpy(float)
        _horizontal_interval(axes[1], i, values, STRATEGY_COLORS[strategy])
    axes[1].set_yticks(range(len(strategies)), [display_labels[strategy] for strategy in strategies])
    axes[1].invert_yaxis()
    axes[1].set_title("B  Storage benefit at 20%", loc="left", fontweight="bold")
    axes[1].set_xlabel("Storage benefit")
    axes[1].set_ylabel("")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(9, 9), useMathText=True)
    axes[1].xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))

    recovery20 = recovery[(recovery.budget_nominal == REPORT_BUDGET) & recovery.response_class.isin(["All", "Slow recovery"])]
    offsets = {"All": -0.12, "Slow recovery": 0.12}
    markers = {"All": "o", "Slow recovery": "s"}
    for class_name in ("All", "Slow recovery"):
        subset = recovery20[recovery20.response_class == class_name]
        for i, strategy in enumerate(strategies):
            values = 100 * subset[subset.strategy == strategy].fraction_eligible_delta_T50_positive.to_numpy(float)
            mean, low, high = _horizontal_interval(
                axes[2], i + offsets[class_name], values,
                STRATEGY_COLORS[strategy], marker=markers[class_name],
                open_marker=class_name == "Slow recovery",
            )
    axes[2].plot([], [], "o", color="0.35", ms=3.8, label="All eligible")
    axes[2].plot([], [], "s", mfc="white", mec="0.35", mew=0.8, ms=3.8, label="Slow recovery")
    axes[2].set_yticks(range(3), ["Uniform", "High pumping", "Leverage"])
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 6.6)
    axes[2].set_xlabel("Cells recovering earlier (%)")
    axes[2].set_title("C  Earlier recovery at 20%", loc="left", fontweight="bold")
    axes[2].legend(loc="upper left", fontsize=5.3, handlelength=1.2, labelspacing=0.25)
    _style_axes(axes)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.locator_params(axis="x", nbins=5)
    fig.tight_layout(w_pad=1.15)
    if output_path is None:
        _save_publication_figure(fig, "FigS29")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    report = metrics[metrics.budget_nominal == REPORT_BUDGET]
    paired20 = paired_benefit[paired_benefit.budget_nominal == REPORT_BUDGET]
    recovery20_paired = paired_recovery[
        (paired_recovery.budget_nominal == REPORT_BUDGET)
        & (paired_recovery.response_class == "All")
        & (paired_recovery.metric == "fraction_eligible_delta_T50_positive")
    ]
    qa = {
        "experiment": "Frozen five-seed-mean 2012 Cell leverage ranking evaluated during May-Oct 2013",
        "allocation_scale": ALLOCATION_SCALE,
        "leverage_score_scale": "1 km Cell",
        "leverage_source_experiment": "own-Unit intervention with Cell-resolved response",
        "ranking_design": "one shared Cell ranking from the mean of all five seed-specific 2012 scores",
        "active_cells": int(len(ctx.grid_ids)),
        "management_units": int(len(ctx.unit_ids)),
        "ranking_version": RANKING_VERSION,
        "ranking_uses_2013_response": False,
        "report_budget_nominal": REPORT_BUDGET,
        "report_actual_dV_m3": float(report.actual_dV_m3.mean()),
        "actual_dV_range_m3": [float(metrics.actual_dV_m3.min()), float(metrics.actual_dV_m3.max())],
        "maximum_relative_actual_dV_mismatch": float(metrics.relative_mismatch.max()),
        "maximum_relative_strategy_dV_spread_0_to_30pct": _maximum_relative_volume_spread(metrics),
        "nonmonotonic_end_benefit_steps_0_to_30pct": _nonmonotonic_group_count(metrics, "benefit_201310_m3"),
        "informed_minus_uniform_positive_seeds": int(paired20[paired20.comparison == "2012-informed - Uniform"].positive_direction.sum()),
        "informed_minus_high_positive_seeds": int(paired20[paired20.comparison == "2012-informed - High pumping"].positive_direction.sum()),
        "earlier_recovery_fraction_informed_minus_uniform_positive_seeds": int(recovery20_paired[recovery20_paired.comparison == "2012-informed - Uniform"].positive_direction.sum()),
        "earlier_recovery_fraction_informed_minus_high_positive_seeds": int(recovery20_paired[recovery20_paired.comparison == "2012-informed - High pumping"].positive_direction.sum()),
        "informed_benefit_five_seed_mean_m3": float(report[report.strategy == "Leverage guided"].benefit_201310_m3.mean()),
        "informed_benefit_seed_range_m3": [float(report[report.strategy == "Leverage guided"].benefit_201310_m3.min()), float(report[report.strategy == "Leverage guided"].benefit_201310_m3.max())],
        "uniform_benefit_five_seed_mean_m3": float(report[report.strategy == "Uniform"].benefit_201310_m3.mean()),
        "high_pumping_benefit_five_seed_mean_m3": float(report[report.strategy == "High pumping"].benefit_201310_m3.mean()),
    }
    print(json.dumps(qa, indent=2))
    return qa


def build_all(force: bool = False) -> dict[str, dict]:
    figs26 = build_figs26(force=force)
    figs27 = build_figs27(force=force)
    return {
        "FigS26": figs26,
        "FigS27": figs27,
        "FigS28": build_figs28(force=False),
        "FigS29": build_figs29(force=False),
    }


if __name__ == "__main__":
    build_all(force="--force" in sys.argv)
