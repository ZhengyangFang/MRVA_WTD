"""Shared helpers for pumping-allocation experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from src.management.experiments import (
    LOCAL_REDUCTION_CAP,
    PI75_NORMAL_FACTOR,
    Window,
)


ROOT = Path(__file__).resolve().parents[2]
HYDROSTRAT_CSV = (
    ROOT
    / "data"
    / "14 MRVA hydrostratigraphy"
    / "MRVA_hydrostratigraphic_model_output.csv"
)
WINDOW_RECOVERY = Window(
    name="May-Oct 2013 recovery intervention",
    initial_month="2013-04",
    intervention_start="2013-05",
    intervention_end="2013-10",
    final_month="2014-10",
    endpoint_months=("2013-10", "2014-04", "2014-10"),
)


def mean_pi75(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(values.mean())
    radius = float(PI75_NORMAL_FACTOR * values.std(ddof=0))
    return mean, mean - radius, mean + radius


def _weighted_average(group: pd.DataFrame, column: str) -> float:
    weights = group["thickness_m"].to_numpy(float)
    values = group[column].to_numpy(float)
    good = np.isfinite(weights) & np.isfinite(values) & (weights > 0)
    return float(np.average(values[good], weights=weights[good])) if good.any() else np.nan


def load_hydrostratigraphy(grid: pd.DataFrame) -> pd.DataFrame:
    """Align the MRVA hydrostratigraphic columns to the 1-km model grid."""
    intervals = pd.read_csv(HYDROSTRAT_CSV)
    intervals["thickness_m"] = intervals["top_elev_m"] - intervals["bot_elev_m"]
    if (intervals["thickness_m"] <= 0).any():
        raise ValueError("Hydrostratigraphic intervals must have positive thickness.")

    facies = intervals.pivot_table(
        index=["cellnum", "x_albers_m", "y_albers_m"],
        columns="soiltype",
        values="thickness_m",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    for code in (0, 1, 2):
        if code not in facies:
            facies[code] = 0.0
    facies = facies.rename(
        columns={
            0: "clay_silt_thickness_m",
            1: "fine_sand_thickness_m",
            2: "gravel_sand_thickness_m",
        }
    )
    facies["sand_thickness_m"] = (
        facies["fine_sand_thickness_m"] + facies["gravel_sand_thickness_m"]
    )
    facies["total_thickness_m"] = (
        facies["sand_thickness_m"] + facies["clay_silt_thickness_m"]
    )
    facies["gravel_fraction"] = facies["gravel_sand_thickness_m"] / facies["total_thickness_m"]
    facies["clay_fraction"] = facies["clay_silt_thickness_m"] / facies["total_thickness_m"]

    weighted = intervals.groupby("cellnum", sort=False).apply(
        lambda group: pd.Series(
            {
                "sand_probability": _weighted_average(group, "sand_probability"),
                "estimation_variance": _weighted_average(group, "estimation_variance"),
            }
        ),
        include_groups=False,
    ).reset_index()
    max_clay = (
        intervals.loc[intervals["soiltype"] == 0]
        .groupby("cellnum", as_index=False)["thickness_m"]
        .max()
        .rename(columns={"thickness_m": "maximum_clay_bed_m"})
    )
    facies = facies.merge(weighted, on="cellnum", how="left", validate="one_to_one")
    facies = facies.merge(max_clay, on="cellnum", how="left", validate="one_to_one")
    facies["maximum_clay_bed_m"] = facies["maximum_clay_bed_m"].fillna(0.0)

    transformer = Transformer.from_crs("ESRI:102003", "EPSG:5070", always_xy=True)
    facies["x_5070"], facies["y_5070"] = transformer.transform(
        facies["x_albers_m"].to_numpy(float), facies["y_albers_m"].to_numpy(float)
    )
    distance_m, nearest = cKDTree(facies[["x_5070", "y_5070"]]).query(
        grid[["x", "y"]].to_numpy(float), k=1
    )
    feature_columns = [
        "cellnum",
        "clay_silt_thickness_m",
        "fine_sand_thickness_m",
        "gravel_sand_thickness_m",
        "sand_thickness_m",
        "total_thickness_m",
        "gravel_fraction",
        "clay_fraction",
        "maximum_clay_bed_m",
        "sand_probability",
        "estimation_variance",
    ]
    aligned = facies.iloc[nearest][feature_columns].reset_index(drop=True)
    aligned.insert(0, "hydrostrat_distance_m", distance_m)
    aligned.insert(0, "hydrostrat_covered", distance_m <= 10.0)
    aligned.loc[~aligned["hydrostrat_covered"], feature_columns] = np.nan
    return pd.concat(
        [grid[["grid_id", "row", "col", "x", "y"]].reset_index(drop=True), aligned],
        axis=1,
    )


def allocate_equal_volume(
    ctx,
    strategy: str,
    budget: int,
    cell_volume: np.ndarray,
    eligible: np.ndarray,
    scores: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Allocate the same requested pumping-reduction volume across Cell strategies."""
    cell_volume = np.asarray(cell_volume, dtype=np.float64)
    eligible_volume = float(cell_volume[eligible].sum(dtype=np.float64))
    requested = float(ctx.reference_2012_pumping_m3 * budget / 100.0)
    feasible = min(requested, LOCAL_REDUCTION_CAP * eligible_volume)
    fractions = np.zeros(len(cell_volume), dtype=np.float64)

    if strategy == "Uniform":
        fractions[eligible] = feasible / eligible_volume
    else:
        score = np.asarray(scores[strategy], dtype=float)
        candidates = np.flatnonzero(eligible & np.isfinite(score) & (cell_volume > 0))
        order = candidates[
            np.lexsort((ctx.grid_ids[candidates], -cell_volume[candidates], -score[candidates]))
        ]
        available = LOCAL_REDUCTION_CAP * cell_volume[order]
        cumulative = np.cumsum(available, dtype=np.float64)
        full_count = int(np.searchsorted(cumulative, feasible, side="right"))
        if full_count:
            fractions[order[:full_count]] = LOCAL_REDUCTION_CAP
            allocated = float(cumulative[full_count - 1])
        else:
            allocated = 0.0
        remaining = feasible - allocated
        if remaining > 1.0:
            if full_count >= len(order):
                raise ValueError(f"Insufficient eligible pumping for {strategy} at {budget}%.")
            fractions[order[full_count]] = remaining / cell_volume[order[full_count]]

    fractions = fractions.astype(np.float32)
    actual = float(np.sum(cell_volume * fractions, dtype=np.float64))
    mismatch = abs(actual - feasible) / max(feasible, 1.0)
    if mismatch > 2e-7 or fractions.min() < 0 or fractions.max() > LOCAL_REDUCTION_CAP + 1e-7:
        raise AssertionError("Equal-volume allocation failed.")
    return fractions, {
        "requested_dV_m3": requested,
        "feasible_target_dV_m3": feasible,
        "actual_dV_m3": actual,
        "relative_mismatch": mismatch,
        "feasibility_capped": requested > feasible + 1.0,
    }
