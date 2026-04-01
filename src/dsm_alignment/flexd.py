from __future__ import annotations

import pandas as pd

from dsm_alignment.common import (
    calibrate_weights_from_importance,
    classify_quadrants,
    find_column_by_keywords,
    merge_weight_overrides,
    weighted_composite,
)


DEFAULT_FLEXD_WEIGHTS = {
    "temporal_flexibility": {
        "full_time": 0.25,
        "not_in_labour_force": 0.20,
        "children": 0.20,
        "single_parent": 0.20,
        "commute_burden": 0.15,
    },
    "demand_elasticity": {
        "tenure_owner": 0.20,
        "household_size": 0.15,
        "occupancy_density": 0.15,
        "peak_intensity": 0.20,
        "peak_to_mean_ratio": 0.10,
        "heating_slope": 0.20,
    },
    "demand_relevance": {"winter_peak_share": 0.45, "heating_slope": 0.30, "peak_intensity": 0.25},
}


def _resolve_flexd_columns(df: pd.DataFrame, column_map: dict[str, str] | None = None) -> dict[str, str]:
    column_map = column_map or {}

    def pick(key: str, include: list[str], *, exclude: list[str] | None = None, prefer_percent: bool = False) -> str:
        if key in column_map:
            return column_map[key]
        found = find_column_by_keywords(df.columns, include=include, exclude=exclude, prefer_percent=prefer_percent)
        if found is None:
            raise KeyError(f"Could not resolve Flex D proxy column for '{key}'.")
        return found

    try:
        full_time = pick(
            "full_time",
            ["worked", "full year full time"],
            exclude=["part-time"],
            prefer_percent=True,
        )
    except KeyError:
        full_time = pick(
            "full_time",
            ["full year full time"],
            exclude=["part-time"],
            prefer_percent=True,
        )

    try:
        not_in_labour_force = pick("not_in_labour_force", ["not in labour force"], prefer_percent=True)
    except KeyError:
        not_in_labour_force = full_time
    try:
        single_parent = pick("single_parent", ["one-parent"], prefer_percent=True)
    except KeyError:
        single_parent = pick("single_parent", ["lone-parent"], prefer_percent=True)
    try:
        commute_burden = pick("commute_burden", ["commuting duration", "60 minutes"], prefer_percent=True)
    except KeyError:
        commute_burden = pick("commute_burden", ["main mode of commuting", "public transit"], prefer_percent=True)
    try:
        occupancy_density = pick("occupancy_density", ["persons per room"], prefer_percent=True)
    except KeyError:
        occupancy_density = pick("occupancy_density", ["more than one person per room"], prefer_percent=True)
    try:
        peak_to_mean_ratio = pick("peak_to_mean_ratio", ["am_pm_peak_ratio"])
    except KeyError:
        peak_to_mean_ratio = pick("peak_to_mean_ratio", ["winter", "peak", "intensity"])

    return {
        "winter_peak_share": pick("winter_peak_share", ["winter", "peak", "share"]),
        "heating_slope_per_hdd": pick("heating_slope_per_hdd", ["heating", "slope"]),
        "peak_intensity": pick("peak_intensity", ["winter", "peak", "intensity"]),
        "full_time": full_time,
        "not_in_labour_force": not_in_labour_force,
        "children": pick("children", ["0 to 14 years"], prefer_percent=True),
        "single_parent": single_parent,
        "commute_burden": commute_burden,
        "tenure_owner": pick("tenure_owner", ["owner"], exclude=["renter"], prefer_percent=True),
        "household_size": pick("household_size", ["average household size"]),
        "occupancy_density": occupancy_density,
        "peak_to_mean_ratio": peak_to_mean_ratio,
    }


def compute_temporal_flexibility(
    df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    cols = _resolve_flexd_columns(df, column_map=column_map)
    score_df = pd.DataFrame(index=df.index)
    score_df["full_time"] = pd.to_numeric(df[cols["full_time"]], errors="coerce")
    score_df["not_in_labour_force"] = pd.to_numeric(df[cols["not_in_labour_force"]], errors="coerce")
    score_df["children"] = pd.to_numeric(df[cols["children"]], errors="coerce")
    score_df["single_parent"] = pd.to_numeric(df[cols["single_parent"]], errors="coerce")
    score_df["commute_burden"] = pd.to_numeric(df[cols["commute_burden"]], errors="coerce")
    w = DEFAULT_FLEXD_WEIGHTS["temporal_flexibility"] if weights is None else weights
    return weighted_composite(score_df, w, invert={"not_in_labour_force", "children", "single_parent", "commute_burden"})


def compute_demand_elasticity(
    df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    cols = _resolve_flexd_columns(df, column_map=column_map)
    score_df = pd.DataFrame(index=df.index)
    score_df["tenure_owner"] = pd.to_numeric(df[cols["tenure_owner"]], errors="coerce")
    score_df["household_size"] = pd.to_numeric(df[cols["household_size"]], errors="coerce")
    score_df["occupancy_density"] = pd.to_numeric(df[cols["occupancy_density"]], errors="coerce")
    score_df["peak_intensity"] = pd.to_numeric(df[cols["peak_intensity"]], errors="coerce")
    score_df["peak_to_mean_ratio"] = pd.to_numeric(df[cols["peak_to_mean_ratio"]], errors="coerce")
    score_df["heating_slope"] = pd.to_numeric(df[cols["heating_slope_per_hdd"]], errors="coerce")
    w = DEFAULT_FLEXD_WEIGHTS["demand_elasticity"] if weights is None else weights
    return weighted_composite(score_df, w)


def evaluate_flexd_alignment(
    df: pd.DataFrame,
    *,
    column_map: dict[str, str] | None = None,
    weight_overrides: dict[str, dict[str, float]] | None = None,
    quantile: float = 0.5,
) -> pd.DataFrame:
    cols = _resolve_flexd_columns(df, column_map=column_map)
    weights = merge_weight_overrides(DEFAULT_FLEXD_WEIGHTS, weight_overrides)
    out = pd.DataFrame(index=df.index)

    out["temporal_flexibility"] = compute_temporal_flexibility(
        df,
        column_map=column_map,
        weights=weights["temporal_flexibility"],
    )
    out["demand_elasticity"] = compute_demand_elasticity(
        df,
        column_map=column_map,
        weights=weights["demand_elasticity"],
    )
    out["participation_capacity"] = (out["temporal_flexibility"] + out["demand_elasticity"]) / 2.0

    demand_df = pd.DataFrame(index=df.index)
    demand_df["winter_peak_share"] = pd.to_numeric(df[cols["winter_peak_share"]], errors="coerce")
    demand_df["heating_slope"] = pd.to_numeric(df[cols["heating_slope_per_hdd"]], errors="coerce")
    demand_df["peak_intensity"] = pd.to_numeric(df[cols["peak_intensity"]], errors="coerce")
    out["demand_relevance"] = weighted_composite(
        demand_df,
        weights["demand_relevance"],
    )

    out["alignment_class"] = classify_quadrants(
        out["demand_relevance"],
        out["participation_capacity"],
        x_high_label="high_demand",
        y_high_label="high_capacity",
        x_low_label="low_demand",
        y_low_label="low_capacity",
        quantile=quantile,
    )
    return out.sort_index()


def calibrate_flexd_weight_overrides(
    df: pd.DataFrame,
    feature_importance: pd.Series,
    *,
    column_map: dict[str, str] | None = None,
    alpha: float = 1.0,
    min_factor: float = 0.5,
) -> dict[str, dict[str, float]]:
    cols = _resolve_flexd_columns(df, column_map=column_map)
    return {
        "temporal_flexibility": calibrate_weights_from_importance(
            DEFAULT_FLEXD_WEIGHTS["temporal_flexibility"],
            {
                "full_time": cols["full_time"],
                "not_in_labour_force": cols["not_in_labour_force"],
                "children": cols["children"],
                "single_parent": cols["single_parent"],
                "commute_burden": cols["commute_burden"],
            },
            feature_importance,
            alpha=alpha,
            min_factor=min_factor,
        ),
        "demand_elasticity": calibrate_weights_from_importance(
            DEFAULT_FLEXD_WEIGHTS["demand_elasticity"],
            {
                "tenure_owner": cols["tenure_owner"],
                "household_size": cols["household_size"],
                "occupancy_density": cols["occupancy_density"],
                "peak_intensity": cols["peak_intensity"],
                "peak_to_mean_ratio": cols["peak_to_mean_ratio"],
                "heating_slope": cols["heating_slope_per_hdd"],
            },
            feature_importance,
            alpha=alpha,
            min_factor=min_factor,
        ),
        "demand_relevance": dict(DEFAULT_FLEXD_WEIGHTS["demand_relevance"]),
    }


__all__ = [
    "compute_temporal_flexibility",
    "compute_demand_elasticity",
    "evaluate_flexd_alignment",
    "calibrate_flexd_weight_overrides",
    "DEFAULT_FLEXD_WEIGHTS",
]
