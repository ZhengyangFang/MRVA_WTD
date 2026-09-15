"""Evaluate recovery timing and spatial benefit distribution."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from src.management.experiments import (
    RESULTS_DIR, RECOVERY_DIR, CACHE_DIR, SEEDS, CLASS_ORDER, CLASS_COLORS, STRATEGY_COLORS,
    WINDOW_2013, Window, context, mean_pi75,
)
from src.reconstruct import best_paths_by_gap

VERSION = "recovery_definitions_v1"
BUDGETS = (20,)
STRATEGIES = ("Uniform", "High pumping", "Leverage guided")
LABELS = {"Uniform": "Uniform", "High pumping": "High-pumping", "Leverage guided": "Leverage-guided"}
WINDOW = Window("Recovery definition sensitivity", "2013-04", "2013-05", "2013-10",
                "2014-12", ("2013-10", "2014-04", "2014-10"))
ENDPOINTS = ("2013-10", "2014-04", "2014-10")
CACHE_DIR = CACHE_DIR / "recovery_responses"


def first_confirmation(wtd, threshold, peak_index, consecutive=1):
    """Return the last month in the first qualifying run; infinity means censored."""
    wtd = np.asarray(wtd)
    if consecutive < 1:
        raise ValueError("consecutive must be positive")
    threshold = np.asarray(threshold)
    hit = (wtd <= threshold[None, :]) & (np.arange(len(wtd))[:, None] > np.asarray(peak_index)[None, :])
    counts = np.zeros(wtd.shape[1], dtype=int)
    first = np.full(wtd.shape[1], np.inf)
    for month in range(len(wtd)):
        counts = np.where(hit[month], counts + 1, 0)
        new = (counts >= consecutive) & ~np.isfinite(first)
        first[new] = month
    return first


def recovery_comparison(base_first, managed_first, cohort, start, endpoint):
    """Compare observed events and restricted waiting times without imputing recovery."""
    cohort = np.asarray(cohort, bool)
    n = int(cohort.sum())
    if not n:
        return dict(n_cells=0, earlier_fraction=np.nan, delayed_fraction=np.nan,
                    newly_confirmed_fraction=np.nan, base_confirmed_fraction=np.nan,
                    managed_confirmed_fraction=np.nan, restricted_months_saved=np.nan,
                    base_censored_cells=0, managed_censored_cells=0)
    base = np.asarray(base_first)[cohort]
    managed = np.asarray(managed_first)[cohort]
    b_hit, m_hit = base <= endpoint, managed <= endpoint
    # Count newly confirmed recovery events.
    earlier = m_hit & (managed < base)
    delayed = b_hit & (base < managed)
    b_wait = np.clip(np.minimum(base, endpoint + 1) - start, 0, None)
    m_wait = np.clip(np.minimum(managed, endpoint + 1) - start, 0, None)
    return dict(n_cells=n, earlier_fraction=earlier.mean(), delayed_fraction=delayed.mean(),
                newly_confirmed_fraction=(m_hit & ~b_hit).mean(),
                base_confirmed_fraction=b_hit.mean(), managed_confirmed_fraction=m_hit.mean(),
                restricted_months_saved=float(np.mean(b_wait-m_wait)),
                base_censored_cells=int((~b_hit).sum()), managed_censored_cells=int((~m_hit).sum()))


def positive_concentration(contributions, percentiles=(1, 5, 10, 20, 50, 100)):
    """Concentration of gross positive benefit; signed totals are reported separately."""
    values = np.sort(np.maximum(np.asarray(contributions, float), 0))[::-1]
    total = values.sum()
    if not len(values) or total <= 0:
        return {p: np.nan for p in percentiles}
    cumulative = np.cumsum(values)
    return {p: float(cumulative[min(len(values)-1, max(0, int(np.ceil(p/100*len(values)))-1))]/total)
            for p in percentiles}


def signed_parts(values):
    values = np.asarray(values, float)
    return float(values.sum()), float(np.maximum(values, 0).sum()), float(np.minimum(values, 0).sum())


def scenario_path(strategy, budget, seed):
    key = {"Uniform": "uniform", "High pumping": "high_pumping", "Leverage guided": "leverage"}[strategy]
    return CACHE_DIR / f"{key}_{budget}_{seed}.npz"


def save_archive(path, **arrays):
    temporary = path.with_suffix(".pending.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def ensure_forecasts(ctx):
    CACHE_DIR.mkdir(exist_ok=True)
    cell_volume, original = ctx.capture_window_pumping(WINDOW)
    _, _, output_idx = ctx.window_indices(WINDOW)
    months = ctx.months[output_idx]
    allocations = {}
    missing = []
    for budget in BUDGETS:
        actual = []
        for strategy in STRATEGIES:
            fractions, _, info = ctx.allocate(strategy, budget, cell_volume)
            allocations[strategy, budget] = (fractions, info)
            actual.append(info["actual_dV_m3"])
            for seed in SEEDS:
                path = scenario_path(strategy, budget, seed)
                if not path.exists():
                    missing.append((strategy, budget, seed))
                    continue
                with np.load(path) as cache:
                    if (str(cache["version"]) != VERSION or not np.array_equal(cache["grid_ids"], ctx.grid_ids)
                            or not np.array_equal(cache["months"], months)
                            or not np.allclose(cache["fractions"], fractions, rtol=0, atol=1e-8)
                            or not np.array_equal(cache["sy"], ctx.sy)):
                        raise ValueError(f"Existing forecast does not match the current experiment: {path.name}")
        if np.ptp(actual) > max(1., np.mean(actual)*1e-6):
            raise AssertionError("Actual pumping reductions differ across strategies")
    if not missing:
        print("All 30 Cell-response forecasts available; no GNN rerun.", flush=True)
        return allocations, cell_volume
    ctx.load_models()
    ctx.paths = best_paths_by_gap(len(output_idx), ctx.horizon_costs)
    start = time.perf_counter()
    try:
        for seed in SEEDS:
            todo = [item for item in missing if item[2] == seed]
            if not todo:
                continue
            ctx.restore_window(WINDOW, original)
            baseline = ctx.forecast_seed(WINDOW, seed)
            for strategy, budget, _ in todo:
                fractions, info = allocations[strategy, budget]
                ctx.apply_reduction(WINDOW, fractions, original)
                response = baseline - ctx.forecast_seed(WINDOW, seed)
                save_archive(scenario_path(strategy, budget, seed), version=np.array(VERSION),
                             grid_ids=ctx.grid_ids, months=months, fractions=fractions, sy=ctx.sy,
                             response_m=response, actual_dV_m3=np.array(info["actual_dV_m3"]))
                ctx.restore_window(WINDOW, original)
                print(f"Saved {seed}, {LABELS[strategy]}, {budget}% ({time.perf_counter()-start:.0f}s)", flush=True)
    finally:
        ctx.restore_window(WINDOW, original)
    return allocations, cell_volume


def summarize_pairs(frame, group_columns, metrics):
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = group[group.strategy == "Uniform"].set_index("seed")
        for strategy in STRATEGIES[1:]:
            current = group[group.strategy == strategy].set_index("seed")
            common = base.index.intersection(current.index)
            for metric in metrics:
                delta = current.loc[common, metric] - base.loc[common, metric]
                values = delta.to_numpy(float)
                mean, low, high = mean_pi75(values)
                rows.append({**dict(zip(group_columns, keys)), "strategy": strategy,
                             "comparison": f"{LABELS[strategy]} minus Uniform", "metric": metric,
                             "mean_difference": mean, "pi75_low": low, "pi75_high": high,
                             "seed_min": float(delta.min()), "seed_max": float(delta.max()),
                             "positive_seeds": int((delta > 0).sum()), "negative_seeds": int((delta < 0).sum()),
                             "finite_seeds": int(np.isfinite(values).sum()),
                             **{str(seed): float(value) for seed, value in delta.items()}})
    return pd.DataFrame(rows)


def analyze():
    ctx = context()
    allocations, cell_volume = ensure_forecasts(ctx)
    initial, _, output_idx = ctx.window_indices(WINDOW)
    last = output_idx[-1]
    history = np.asarray(ctx.wtd[:last+1], dtype=np.float32)
    classes = np.asarray(ctx.labels.response_class.fillna("Unclassified"), dtype=str)
    valid = ctx.valid_decline & np.isfinite(ctx.pre_wtd) & (ctx.peak_wtd > ctx.pre_wtd)
    base_first = {}
    for threshold in (25, 50, 75):
        level = ctx.peak_wtd - threshold/100*(ctx.peak_wtd-ctx.pre_wtd)
        for duration in (1, 3):
            base_first[threshold, duration] = first_confirmation(history, level, ctx.peak_month_idx, duration)
    common = valid & (base_first[50, 1] > initial)
    original_common = valid & (~np.isfinite(ctx.baseline_t50_index) | (ctx.baseline_t50_index > initial))
    np.testing.assert_array_equal(common, original_common)
    status = np.where(~valid, "Invalid drought reference", np.where(common, "Not half recovered", "Previously half recovered"))
    initial_gap = np.maximum(history[initial]-ctx.d50, 0)
    gap_names = np.array(["At/below half-recovery level", "0-0.1 m", "0.1-0.5 m", "0.5-1 m", ">1 m"])
    gap_index = np.digitize(initial_gap, [0, .1, .5, 1], right=True)
    from src.management.allocation import load_hydrostratigraphy

    hydro = load_hydrostratigraphy(ctx.grid).set_index("grid_id").reindex(ctx.grid_ids)
    definitions, monthly, destinations, gaps, concentrations, checks = [], [], [], [], [], []
    legacy = pd.read_csv(RESULTS_DIR / "recovery_timing_by_class.csv")
    legacy_metrics = pd.read_csv(RESULTS_DIR / "recovery_strategy_seed_metrics.csv")
    for budget in BUDGETS:
        for strategy in STRATEGIES:
            fractions, info = allocations[strategy, budget]
            saved_volume = cell_volume*fractions
            for seed in SEEDS:
                with np.load(scenario_path(strategy, budget, seed)) as archive:
                    response = archive["response_m"].astype(np.float64)
                if not np.isfinite(response).all():
                    raise AssertionError("Non-finite response")
                managed = history.astype(np.float64).copy()
                managed[output_idx] -= response
                common_fields = dict(strategy=strategy, budget=budget, seed=seed, actual_dV_m3=info["actual_dV_m3"])
                for threshold in (25, 50, 75):
                    level = ctx.peak_wtd - threshold/100*(ctx.peak_wtd-ctx.pre_wtd)
                    for duration in (1, 3):
                        base = base_first[threshold, duration]
                        treated = first_confirmation(managed, level, ctx.peak_month_idx, duration)
                        eligible = valid & (base > initial)
                        for endpoint in ENDPOINTS:
                            end = ctx.month_lookup[endpoint]
                            for cohort_name, mask in (("Definition-specific", eligible), ("Fixed half-recovery cohort", common)):
                                for label in ("All", *CLASS_ORDER):
                                    selected = mask & ((classes == label) if label != "All" else True)
                                    definitions.append({**common_fields, "threshold_pct": threshold,
                                                        "consecutive_months": duration, "endpoint": endpoint,
                                                        "cohort": cohort_name, "response_class": label,
                                                        **recovery_comparison(base, treated, selected, initial+1, end)})
                # Calculate the legacy first-hit statistic.
                old = ctx.recovery_summary(ctx.recovery_arrays(response[:12].astype(np.float32), WINDOW_2013), response[:12])
                reference = legacy[(legacy.budget_nominal == budget) & (legacy.strategy == strategy) & (legacy.seed == seed)]
                for row in old:
                    previous = reference[reference.response_class == row["response_class"]].iloc[0]
                    difference = row["fraction_eligible_delta_T50_positive"]-previous.fraction_eligible_delta_T50_positive
                    if abs(difference) > 1e-10:
                        raise AssertionError("Old recovery result does not reproduce")
                    checks.append({**common_fields, "check": "FigS29 recovery reproduction", "group": row["response_class"], "difference": difference})
                previous = legacy_metrics[(legacy_metrics.budget_nominal == budget) & (legacy_metrics.strategy == strategy) & (legacy_metrics.seed == seed)].iloc[0]
                for k, month_index in enumerate(output_idx):
                    month = ctx.months[month_index]
                    contribution = response[k]*ctx.storage_weights_m2
                    base_deficit = np.maximum(history[month_index, common]-ctx.pre_wtd[common], 0)
                    managed_deficit = np.maximum(managed[month_index, common]-ctx.pre_wtd[common], 0)
                    avoidance = base_deficit-managed_deficit
                    monthly.append({**common_fields, "month": month, "net_storage_benefit_m3": contribution.sum(),
                                    "mean_water_level_benefit_m": response[k].mean(),
                                    "fixed_cohort_cells": int(common.sum()),
                                    "base_mean_deficit_m": base_deficit.mean(),
                                    "managed_mean_deficit_m": managed_deficit.mean(),
                                    "mean_deficit_avoided_m": avoidance.mean(),
                                    "positive_deficit_avoided_cells": int((avoidance > 0).sum())})
                    if month == "2013-10":
                        error = contribution.sum()-previous["benefit_201310_m3"]
                        if abs(error) > max(10., abs(previous["benefit_201310_m3"])*1e-5):
                            raise AssertionError("Old storage result does not reproduce")
                        checks.append({**common_fields, "check": "FigS29 storage reproduction m3", "group": "All", "difference": error})
                    if month not in ENDPOINTS:
                        continue
                    for dimension, names, categories in (("Recovery status", np.unique(status), status),
                                                         ("FSB", np.unique(classes), classes)):
                        for name in names:
                            mask = categories == name
                            net, positive, negative = signed_parts(contribution[mask])
                            pumping = saved_volume[mask].sum()
                            destinations.append({**common_fields, "endpoint": month, "dimension": dimension, "group": name,
                                                 "n_cells": int(mask.sum()), "net_storage_benefit_m3": net,
                                                 "positive_storage_benefit_m3": positive, "negative_storage_benefit_m3": negative,
                                                 "reduction_m3": pumping, "mean_sy": ctx.sy[mask].mean(),
                                                 "reduction_weighted_sy": np.average(ctx.sy[mask], weights=saved_volume[mask]) if pumping > 0 else np.nan,
                                                 "median_gravel_fraction": hydro.loc[mask, "gravel_fraction"].median(),
                                                 "median_clay_thickness_m": hydro.loc[mask, "clay_silt_thickness_m"].median()})
                    for name in gap_names:
                        mask = common & (gap_names[gap_index] == name)
                        b = np.maximum(history[month_index, mask]-ctx.pre_wtd[mask], 0)
                        m = np.maximum(managed[month_index, mask]-ctx.pre_wtd[mask], 0)
                        gaps.append({**common_fields, "endpoint": month, "initial_gap": name, "n_cells": int(mask.sum()),
                                     "mean_deficit_avoided_m": np.mean(b-m) if mask.any() else np.nan,
                                     "mean_water_level_benefit_m": np.mean(response[k, mask]) if mask.any() else np.nan,
                                     "net_storage_benefit_m3": contribution[mask].sum(),
                                     "mean_sy": np.mean(ctx.sy[mask]) if mask.any() else np.nan,
                                     "median_gravel_fraction": hydro.loc[mask, "gravel_fraction"].median(),
                                     "median_clay_thickness_m": hydro.loc[mask, "clay_silt_thickness_m"].median()})
                    for percentile, share in positive_concentration(contribution[common], range(1, 101)).items():
                        concentrations.append({**common_fields, "endpoint": month, "top_area_pct": percentile,
                                               "positive_benefit_share": share})
                print(f"Analyzed {seed}, {LABELS[strategy]}, {budget}%", flush=True)
    frames = {
        "unrecovered_monthly.csv": pd.DataFrame(monthly),
        "benefit_destinations.csv": pd.DataFrame(destinations),
        "recovery_gap.csv": pd.DataFrame(gaps),
        "benefit_concentration.csv": pd.DataFrame(concentrations),
    }
    for name, frame in frames.items():
        frame.to_csv(RECOVERY_DIR / name, index=False)
    return frames


def style(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.65)
    ax.tick_params(length=3, width=.65)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#EEEEEE", lw=.5)


def interval_points(ax, data, x, metric, scale=1., color="k", label=None, offset=0.):
    for i, value in enumerate(x):
        mean, low, high = mean_pi75(data[data["_x"] == value][metric])
        ax.errorbar(i+offset, mean*scale, yerr=[[(mean-low)*scale], [(high-mean)*scale]],
                    fmt="o", ms=3.4, capsize=2, lw=1, color=color, label=label if i == 0 else None)


def plot36():
    monthly = pd.read_csv(RESULTS_DIR / "FigS36_monthly_metrics.csv")
    paired = pd.read_csv(RESULTS_DIR / "FigS36_paired_recovery.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    fig.subplots_adjust(left=.12, right=.98, bottom=.12, top=.89, wspace=.65, hspace=.60)
    for ax in axes.flat:
        style(ax)
    ax = axes[0, 0]
    for offset, strategy in zip((-.17, 0, .17), STRATEGIES):
        data = monthly[(monthly.strategy == strategy) & (monthly.month == "2013-10")].copy()
        data["_x"] = data.budget
        interval_points(ax, data, BUDGETS, "net_storage_benefit_m3", 1e-9,
                        STRATEGY_COLORS[strategy], LABELS[strategy], offset)
    ax.set_xticks([0, 1], ["10%", "20%"])
    ax.set_xlabel("Pumping-reduction budget")
    ax.set_ylabel("Storage benefit ($10^9$ m$^3$)")
    ax.set_title("A  Immediate storage benefit", loc="left", fontweight="bold", pad=10)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=7,
               bbox_to_anchor=(.53, .99), frameon=False)
    configs = [(t, d) for d in (1, 3) for t in (25, 50, 75)]
    labels = [f"{t}% / {'first month' if d == 1 else '3 months'}" for t, d in configs]
    chosen = paired[(paired.strategy == STRATEGIES[-1]) & (paired.budget == 20)
                    & (paired.endpoint == "2014-04") & (paired.cohort == "Definition-specific")
                    & (paired.response_class == "All")]
    for ax, metric, scale, title, xlabel in (
        (axes[0, 1], "earlier_fraction", 100, "B  Earlier recovery", "Difference (percentage points)"),
        (axes[1, 1], "restricted_months_saved", 1, "D  Restricted waiting time saved", "Difference (months per Cell)")):
        for y, (threshold, duration) in enumerate(configs):
            row = chosen[(chosen.threshold_pct == threshold) & (chosen.consecutive_months == duration) & (chosen.metric == metric)].iloc[0]
            ax.errorbar(row.mean_difference*scale, y,
                        xerr=[[(row.mean_difference-row.pi75_low)*scale], [(row.pi75_high-row.mean_difference)*scale]],
                        fmt="o", color="#16857C", ms=3.5, lw=1.1, capsize=2)
        ax.axvline(0, color=".45", lw=.7, ls="--")
        ax.set_yticks(range(6), labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontweight="bold", pad=10)
        ax.text(.5, 1.005, "Leverage-guided minus Uniform", transform=ax.transAxes, ha="center", fontsize=6)
    ax = axes[1, 0]
    for strategy in STRATEGIES:
        group = monthly[(monthly.budget == 20) & (monthly.strategy == strategy) & (monthly.month <= "2014-10")]
        stats = np.array([mean_pi75(g.mean_deficit_avoided_m) for _, g in group.groupby("month")])
        x = np.arange(len(stats))
        xx = np.linspace(0, len(stats)-1, 200)
        smoothed = np.column_stack([PchipInterpolator(x, stats[:, k])(xx) for k in range(3)])
        ax.plot(xx, smoothed[:, 0]*1000, color=STRATEGY_COLORS[strategy], lw=1.35)
        ax.fill_between(xx, smoothed[:, 1]*1000, smoothed[:, 2]*1000, color=STRATEGY_COLORS[strategy], alpha=.13, lw=0)
    ax.axvline(5, color=".5", ls="--", lw=.7)
    ax.set_xticks([0, 5, 11, 17], ["May 2013", "Oct", "Apr 2014", "Oct"])
    ax.set_ylabel("Mean water-level deficit avoided (mm)")
    ax.set_title("C  Initially unrecovered Cells", loc="left", fontweight="bold", pad=10)
    ax.set_xlim(0, 17)
    ax.tick_params(axis="x", labelsize=6)
    fig.savefig(RESULTS_DIR / "FigS36.png", dpi=600, facecolor="white")
    fig.savefig(RESULTS_DIR / "FigS36.svg", facecolor="white")
    return fig


def plot37():
    destinations = pd.read_csv(RESULTS_DIR / "FigS37_benefit_destinations.csv")
    paired = pd.read_csv(RESULTS_DIR / "FigS37_paired_destinations.csv")
    concentration = pd.read_csv(RESULTS_DIR / "FigS37_benefit_concentration.csv")
    gaps = pd.read_csv(RESULTS_DIR / "FigS37_initial_deficit_groups.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    fig.subplots_adjust(left=.12, right=.98, bottom=.14, top=.89, wspace=.55, hspace=.65)
    for ax in axes.flat:
        style(ax)
    ax = axes[0, 0]
    group = paired[(paired.budget == 20) & (paired.endpoint == "2013-10") & (paired.dimension == "Recovery status")
                   & (paired.strategy == STRATEGIES[-1])]
    names = ["Previously half recovered", "Not half recovered"]
    for x, name in enumerate(names):
        row = group[group.group == name].iloc[0]
        ax.errorbar(x, row.mean_difference/1e6,
                    yerr=[[(row.mean_difference-row.pi75_low)/1e6], [(row.pi75_high-row.mean_difference)/1e6]],
                    color="#16857C", fmt="o", ms=4, lw=1.2, capsize=3)
    ax.axhline(0, color=".5", lw=.7, ls="--")
    ax.set_xlim(-.5, 1.5)
    ax.set_xticks([0, 1], ["Previously\nhalf recovered", "Not yet\nhalf recovered"])
    ax.set_ylabel("Extra storage benefit ($10^6$ m$^3$)")
    ax.set_title("A  Where extra benefit occurs", loc="left", fontweight="bold", pad=10)
    ax.text(.5, 1.005, "Leverage-guided minus Uniform", transform=ax.transAxes, ha="center", fontsize=6)
    ax = axes[0, 1]
    for strategy in STRATEGIES:
        group = concentration[(concentration.budget == 20) & (concentration.endpoint == "2013-10") & (concentration.strategy == strategy)]
        stats = np.array([mean_pi75(g.positive_benefit_share) for _, g in group.groupby("top_area_pct")])
        stats = np.vstack([np.zeros(3), stats])
        x = np.arange(101)
        ax.plot(x, stats[:, 0]*100, lw=1.4, color=STRATEGY_COLORS[strategy], label=LABELS[strategy])
        ax.fill_between(x, stats[:, 1]*100, stats[:, 2]*100, color=STRATEGY_COLORS[strategy], alpha=.13, lw=0)
    ax.plot([0, 100], [0, 100], color=".7", lw=.7, ls="--")
    ax.set(xlim=(0, 100), ylim=(0, 102), xlabel="Top fraction of unrecovered Cells (%)",
           ylabel="Cumulative positive benefit (%)")
    ax.set_title("B  Benefit concentration", loc="left", fontweight="bold", pad=10)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(.52, .99), frameon=False)
    ax = axes[1, 0]
    gap_names = ["0-0.1 m", "0.1-0.5 m", "0.5-1 m", ">1 m"]
    for offset, strategy in zip((-.18, 0, .18), STRATEGIES):
        group = gaps[(gaps.budget == 20) & (gaps.endpoint == "2013-10") & (gaps.strategy == strategy)].copy()
        group["_x"] = group.initial_gap
        interval_points(ax, group, gap_names, "mean_deficit_avoided_m", 1000, STRATEGY_COLORS[strategy], offset=offset)
    ax.set_xticks(range(4), ["0-0.1", "0.1-0.5", "0.5-1", ">1"], fontsize=6)
    ax.set_xlabel("Initial gap to half-recovery level (m)")
    ax.set_ylabel("Mean water-level deficit avoided (mm)")
    ax.set_title("C  Benefit versus recovery need", loc="left", fontweight="bold", pad=10)
    ax = axes[1, 1]
    group = destinations[(destinations.budget == 20) & (destinations.endpoint == "2013-10") & (destinations.dimension == "FSB")]
    bottom = np.zeros(3)
    for label in CLASS_ORDER:
        means = np.array([group[(group.strategy == strategy) & (group.group == label)].reduction_m3.mean() for strategy in STRATEGIES])
        total = np.array([group[group.strategy == strategy].groupby("seed").reduction_m3.sum().mean() for strategy in STRATEGIES])
        pct = means/total*100
        ax.bar(np.arange(3), pct, bottom=bottom, color=CLASS_COLORS[label], width=.6, label=label[0])
        for x, value in enumerate(pct):
            if value > 5:
                ax.text(x, bottom[x]+value/2, f"{label[0]}  {value:.0f}%", ha="center", va="center", fontsize=6, color="white")
        bottom += pct
    unclassified = np.array([group[(group.strategy == strategy) & (group.group == "Unclassified")].reduction_m3.mean()
                             for strategy in STRATEGIES]) / total * 100
    ax.bar(np.arange(3), unclassified, bottom=bottom, color="#B8B8B8", width=.6)
    ax.set_xticks(range(3), ["Uniform", "High-pumping", "Leverage-guided"], rotation=15, fontsize=6)
    ax.set_ylabel("Share of pumping reduction (%)")
    ax.set_ylim(0, 103)
    ax.set_title("D  Allocation across F, S and B", loc="left", fontweight="bold", pad=10)
    fig.savefig(RESULTS_DIR / "FigS37.png", dpi=600, facecolor="white")
    fig.savefig(RESULTS_DIR / "FigS37.svg", facecolor="white")
    return fig


def main():
    analyze()


if __name__ == "__main__":
    main()
