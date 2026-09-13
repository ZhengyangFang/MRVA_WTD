"""River-setting summaries for the Cell persistence experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import pearsonr, rankdata, spearmanr

from src.management.experiments import LOCAL_RESPONSE_DIR, UNIT_CELLS_CSV


ROOT = Path(__file__).resolve().parents[2]
TOPOGRAPHY_PATH = ROOT / "data" / "5 topography" / "topo_1km.csv"
STREAMBED_PATH = ROOT / "data" / "15 MRVA streambed connectiviy" / "MRVA_streambed_connectivity_1km_aligned.tif"
PERSISTENCE = "own_unit_persistence_fraction"
DISTANCE = "distance_to_order12_river_km"
VIC = "streambed_vic_s"


def _within_unit_rank(frame: pd.DataFrame, predictor: str) -> float:
    data = frame[["unit_id", predictor, PERSISTENCE]].dropna().copy()
    data["x"] = rankdata(data[predictor])
    data["y"] = rankdata(data[PERSISTENCE])
    data["x"] -= data.groupby("unit_id")["x"].transform("mean")
    data["y"] -= data.groupby("unit_id")["y"].transform("mean")
    return float(pearsonr(data.x, data.y).statistic)


def _trend(frame: pd.DataFrame, predictor: str, bins: int = 24) -> pd.DataFrame:
    data = frame[[predictor, PERSISTENCE]].dropna().copy()
    edges = np.unique(np.quantile(data[predictor], np.linspace(0, 1, bins + 1)))
    data["bin"] = pd.cut(data[predictor], edges, include_lowest=True, duplicates="drop")
    result = data.groupby("bin", observed=True).agg(
        x=(predictor, "median"),
        median=(PERSISTENCE, "median"),
        q25=(PERSISTENCE, lambda values: values.quantile(0.25)),
        q75=(PERSISTENCE, lambda values: values.quantile(0.75)),
        n=(PERSISTENCE, "size"),
    ).reset_index(drop=True)
    result.insert(0, "predictor", predictor)
    return result


def _joint_model(frame: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 17) -> pd.DataFrame:
    data = frame[["unit_id", PERSISTENCE, DISTANCE, VIC]].dropna().copy()
    within = []
    for column in (PERSISTENCE, DISTANCE, VIC):
        name = f"within_{column}"
        data[name] = rankdata(data[column])
        data[name] -= data.groupby("unit_id")[name].transform("mean")
        data[name] /= data[name].std(ddof=0)
        within.append(name)
    y = data[within[0]].to_numpy(float)
    x = data[within[1:]].to_numpy(float)
    estimate = np.linalg.lstsq(x, y, rcond=None)[0]
    unit_ids = data.unit_id.drop_duplicates().to_numpy()
    xtx, xty = [], []
    for unit_id in unit_ids:
        mask = data.unit_id.to_numpy() == unit_id
        unit_x, unit_y = x[mask], y[mask]
        xtx.append(unit_x.T @ unit_x)
        xty.append(unit_x.T @ unit_y)
    xtx, xty = np.stack(xtx), np.stack(xty)
    weights = np.random.default_rng(seed).multinomial(
        len(unit_ids), np.full(len(unit_ids), 1 / len(unit_ids)), size=n_bootstrap
    )
    boot = np.linalg.solve(
        np.einsum("bu,uij->bij", weights, xtx),
        np.einsum("bu,ui->bi", weights, xty)[..., None],
    )[..., 0]
    interval = np.quantile(boot, [0.025, 0.975], axis=0)
    return pd.DataFrame({
        "predictor": ["River distance", "Streambed VIC"],
        "coefficient": estimate,
        "ci_low": interval[0],
        "ci_high": interval[1],
    })


def prepare_persistence_results() -> dict[str, pd.DataFrame]:
    for path in (UNIT_CELLS_CSV, TOPOGRAPHY_PATH, STREAMBED_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    cells = pd.read_csv(UNIT_CELLS_CSV, dtype={"unit_id": str})
    topo = pd.read_csv(TOPOGRAPHY_PATH, usecols=["grid_id", DISTANCE])
    data = cells.merge(topo, on="grid_id", validate="one_to_one")
    with rasterio.open(STREAMBED_PATH) as source:
        vic = np.fromiter(
            (value[0] for value in source.sample(data[["x", "y"]].to_numpy())),
            dtype=float,
            count=len(data),
        )
        invalid = ~np.isfinite(vic) | (vic <= 0)
        if source.nodata is not None:
            invalid |= np.isclose(vic, source.nodata)
        vic[invalid] = np.nan
    data[VIC] = vic
    data = data[["grid_id", "unit_id", PERSISTENCE, DISTANCE, VIC]].copy()
    valid = data[np.isfinite(data[PERSISTENCE])]
    streambed = valid[np.isfinite(valid[VIC])]
    associations = []
    for frame, predictor, label in (
        (valid, DISTANCE, "River distance"),
        (streambed, VIC, "Streambed VIC"),
    ):
        unit = frame.groupby("unit_id")[[predictor, PERSISTENCE]].median().dropna()
        associations.append({
            "predictor": label,
            "cells": len(frame),
            "units": len(unit),
            "cell_spearman_rho": spearmanr(frame[predictor], frame[PERSISTENCE]).statistic,
            "unit_median_spearman_rho": spearmanr(unit[predictor], unit[PERSISTENCE]).statistic,
            "within_unit_rank_association": _within_unit_rank(frame, predictor),
        })
    outputs = {
        "river_persistence_cells.csv.gz": data,
        "river_persistence_trends.csv": pd.concat([
            _trend(valid, DISTANCE), _trend(streambed, VIC)
        ], ignore_index=True),
        "river_persistence_model.csv": _joint_model(streambed),
        "river_persistence_associations.csv": pd.DataFrame(associations),
    }
    for name, frame in outputs.items():
        frame.to_csv(LOCAL_RESPONSE_DIR / name, index=False)
    return outputs
