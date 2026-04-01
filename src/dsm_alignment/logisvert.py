from __future__ import annotations

import pandas as pd

from dsm_alignment.common import (
    calibrate_weights_from_importance,
    classify_quadrants,
    find_column_by_keywords,
    merge_weight_overrides,
    weighted_composite,
)


DEFAULT_LOGISVERT_WEIGHTS = {
    "structural_demand_relevance": {
        "heating_slope": 0.35,
        "winter_intensity": 0.30,
        "seasonal_consumption": 0.20,
        "peak_intensity": 0.15,
    },
    "adoption_capacity": {"median_income": 0.45, "owner": 0.25, "single_detached": 0.15, "residential_stability": 0.15},
    "persistence_capacity": {"mobility_1y": 0.40, "renter": 0.30, "owner": 0.30},
}


def _resolve_logisvert_columns(df: pd.DataFrame, column_map: dict[str, str] | None = None) -> dict[str, str]:
    column_map = column_map or {}

    def pick(key: str, include: list[str], *, exclude: list[str] | None = None, prefer_percent: bool = False) -> str:
        if key in column_map:
            return column_map[key]
        found = find_column_by_keywords(df.columns, include=include, exclude=exclude, prefer_percent=prefer_percent)
        if found is None:
            raise KeyError(f"Could not resolve LogisVert proxy column for '{key}'.")
        return found

    return {
        "heating_slope_per_hdd": pick("heating_slope_per_hdd", ["heating", "slope"]),
        "winter_intensity": pick("winter_intensity", ["winter", "peak", "intensity"]),
        "peak_intensity": pick("peak_intensity", ["winter", "peak", "intensity"]),
        "seasonal_consumption": pick("seasonal_consumption", ["mean_load"]),
        "median_income": pick("median_income", ["median", "income"]),
        "owner": pick("owner", ["owner"], exclude=["renter"], prefer_percent=True),
        "renter": pick("renter", ["renter"], prefer_percent=True),
        "single_detached": pick("single_detached", ["single-detached house"], prefer_percent=True),
        "residential_stability": pick("residential_stability", ["mobility status 1 year ago", "non-movers"], prefer_percent=True),
        "mobility_1y": pick(
            "mobility_1y",
            ["mobility status 1 year ago", "non-movers"],
            prefer_percent=True,
        ),
    }


def evaluate_logisvert_alignment(
    df: pd.DataFrame,
    *,
    column_map: dict[str, str] | None = None,
    weight_overrides: dict[str, dict[str, float]] | None = None,
    quantile: float = 0.5,
) -> pd.DataFrame:
    cols = _resolve_logisvert_columns(df, column_map=column_map)
    weights = merge_weight_overrides(DEFAULT_LOGISVERT_WEIGHTS, weight_overrides)
    out = pd.DataFrame(index=df.index)

    structural_df = pd.DataFrame(index=df.index)
    structural_df["heating_slope"] = pd.to_numeric(df[cols["heating_slope_per_hdd"]], errors="coerce")
    structural_df["winter_intensity"] = pd.to_numeric(df[cols["winter_intensity"]], errors="coerce")
    structural_df["seasonal_consumption"] = pd.to_numeric(df[cols["seasonal_consumption"]], errors="coerce")
    structural_df["peak_intensity"] = pd.to_numeric(df[cols["peak_intensity"]], errors="coerce")
    out["structural_demand_relevance"] = weighted_composite(
        structural_df,
        weights["structural_demand_relevance"],
    )

    capacity_df = pd.DataFrame(index=df.index)
    capacity_df["median_income"] = pd.to_numeric(df[cols["median_income"]], errors="coerce")
    capacity_df["owner"] = pd.to_numeric(df[cols["owner"]], errors="coerce")
    capacity_df["single_detached"] = pd.to_numeric(df[cols["single_detached"]], errors="coerce")
    capacity_df["residential_stability"] = pd.to_numeric(df[cols["residential_stability"]], errors="coerce")
    out["adoption_capacity"] = weighted_composite(capacity_df, weights["adoption_capacity"])

    persistence_df = pd.DataFrame(index=df.index)
    persistence_df["mobility_1y"] = pd.to_numeric(df[cols["mobility_1y"]], errors="coerce")
    persistence_df["renter"] = pd.to_numeric(df[cols["renter"]], errors="coerce")
    persistence_df["owner"] = pd.to_numeric(df[cols["owner"]], errors="coerce")
    out["persistence_capacity"] = weighted_composite(persistence_df, weights["persistence_capacity"], invert={"renter"})

    out["overall_capacity"] = (out["adoption_capacity"] + out["persistence_capacity"]) / 2.0
    out["alignment_class"] = classify_quadrants(
        out["structural_demand_relevance"],
        out["overall_capacity"],
        x_high_label="high_structural_relevance",
        y_high_label="high_capacity",
        x_low_label="low_structural_relevance",
        y_low_label="low_capacity",
        quantile=quantile,
    )
    return out.sort_index()


def calibrate_logisvert_weight_overrides(
    df: pd.DataFrame,
    feature_importance: pd.Series,
    *,
    column_map: dict[str, str] | None = None,
    alpha: float = 1.0,
    min_factor: float = 0.5,
) -> dict[str, dict[str, float]]:
    cols = _resolve_logisvert_columns(df, column_map=column_map)
    return {
        "structural_demand_relevance": dict(DEFAULT_LOGISVERT_WEIGHTS["structural_demand_relevance"]),
        "adoption_capacity": calibrate_weights_from_importance(
            DEFAULT_LOGISVERT_WEIGHTS["adoption_capacity"],
            {
                "median_income": cols["median_income"],
                "owner": cols["owner"],
                "single_detached": cols["single_detached"],
                "residential_stability": cols["residential_stability"],
            },
            feature_importance,
            alpha=alpha,
            min_factor=min_factor,
        ),
        "persistence_capacity": calibrate_weights_from_importance(
            DEFAULT_LOGISVERT_WEIGHTS["persistence_capacity"],
            {"mobility_1y": cols["mobility_1y"], "renter": cols["renter"], "owner": cols["owner"]},
            feature_importance,
            alpha=alpha,
            min_factor=min_factor,
        ),
    }


__all__ = ["evaluate_logisvert_alignment", "calibrate_logisvert_weight_overrides", "DEFAULT_LOGISVERT_WEIGHTS"]
