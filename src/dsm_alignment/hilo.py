from __future__ import annotations

import pandas as pd

from dsm_alignment.common import (
    calibrate_weights_from_importance,
    classify_quadrants,
    find_column_by_keywords,
    merge_weight_overrides,
    weighted_composite,
)


DEFAULT_HILO_WEIGHTS = {
    "technical_eligibility": {"heating_slope": 0.5, "peak_intensity": 0.3, "single_detached": 0.2},
    "control_authority": {"owner": 0.35, "renter": 0.25, "multi_unit": 0.20, "residential_stability": 0.20},
    "curtailment_tolerance": {"household_size": 0.25, "occupancy_density": 0.20, "children": 0.25, "older": 0.30},
    "demand_relevance": {"winter_peak_share": 0.5, "heating_slope": 0.3, "peak_intensity": 0.2},
}


def _resolve_hilo_columns(df: pd.DataFrame, column_map: dict[str, str] | None = None) -> dict[str, str]:
    column_map = column_map or {}

    def pick(key: str, include: list[str], *, exclude: list[str] | None = None, prefer_percent: bool = False) -> str:
        if key in column_map:
            return column_map[key]
        found = find_column_by_keywords(df.columns, include=include, exclude=exclude, prefer_percent=prefer_percent)
        if found is None:
            raise KeyError(f"Could not resolve Hilo proxy column for '{key}'.")
        return found

    return {
        "winter_peak_share": pick("winter_peak_share", ["winter", "peak", "share"]),
        "heating_slope_per_hdd": pick("heating_slope_per_hdd", ["heating", "slope"]),
        "peak_intensity": pick("peak_intensity", ["winter", "peak", "intensity"]),
        "owner": pick("owner", ["owner"], exclude=["renter"], prefer_percent=True),
        "renter": pick("renter", ["renter"], prefer_percent=True),
        "single_detached": pick("single_detached", ["single-detached house"], prefer_percent=True),
        "multi_unit": pick("multi_unit", ["apartment"], prefer_percent=True),
        "residential_stability": pick("residential_stability", ["mobility status 1 year ago", "non-movers"], prefer_percent=True),
        "household_size": pick("household_size", ["average household size"]),
        "occupancy_density": pick("occupancy_density", ["persons per room"], prefer_percent=True),
        "children": pick("children", ["0 to 14 years"], prefer_percent=True),
        "older": pick("older", ["65 years and over"], prefer_percent=True),
    }


def evaluate_hilo_alignment(
    df: pd.DataFrame,
    *,
    column_map: dict[str, str] | None = None,
    weight_overrides: dict[str, dict[str, float]] | None = None,
    quantile: float = 0.5,
) -> pd.DataFrame:
    cols = _resolve_hilo_columns(df, column_map=column_map)
    weights = merge_weight_overrides(DEFAULT_HILO_WEIGHTS, weight_overrides)
    out = pd.DataFrame(index=df.index)

    technical_df = pd.DataFrame(index=df.index)
    technical_df["heating_slope"] = pd.to_numeric(df[cols["heating_slope_per_hdd"]], errors="coerce")
    technical_df["peak_intensity"] = pd.to_numeric(df[cols["peak_intensity"]], errors="coerce")
    technical_df["single_detached"] = pd.to_numeric(df[cols["single_detached"]], errors="coerce")
    out["technical_eligibility"] = weighted_composite(technical_df, weights["technical_eligibility"])

    control_df = pd.DataFrame(index=df.index)
    control_df["owner"] = pd.to_numeric(df[cols["owner"]], errors="coerce")
    control_df["renter"] = pd.to_numeric(df[cols["renter"]], errors="coerce")
    control_df["multi_unit"] = pd.to_numeric(df[cols["multi_unit"]], errors="coerce")
    control_df["residential_stability"] = pd.to_numeric(df[cols["residential_stability"]], errors="coerce")
    out["control_authority"] = weighted_composite(control_df, weights["control_authority"], invert={"renter", "multi_unit"})

    tolerance_df = pd.DataFrame(index=df.index)
    tolerance_df["household_size"] = pd.to_numeric(df[cols["household_size"]], errors="coerce")
    tolerance_df["occupancy_density"] = pd.to_numeric(df[cols["occupancy_density"]], errors="coerce")
    tolerance_df["children"] = pd.to_numeric(df[cols["children"]], errors="coerce")
    tolerance_df["older"] = pd.to_numeric(df[cols["older"]], errors="coerce")
    out["curtailment_tolerance"] = weighted_composite(
        tolerance_df,
        weights["curtailment_tolerance"],
        invert={"occupancy_density", "children", "older"},
    )

    out["hilo_suitability"] = (
        out["technical_eligibility"] + out["control_authority"] + out["curtailment_tolerance"]
    ) / 3.0

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
        out["hilo_suitability"],
        x_high_label="high_demand",
        y_high_label="high_hilo_suitability",
        x_low_label="low_demand",
        y_low_label="low_hilo_suitability",
        quantile=quantile,
    )
    return out.sort_index()


def calibrate_hilo_weight_overrides(
    df: pd.DataFrame,
    feature_importance: pd.Series,
    *,
    column_map: dict[str, str] | None = None,
    alpha: float = 1.0,
    min_factor: float = 0.5,
) -> dict[str, dict[str, float]]:
    cols = _resolve_hilo_columns(df, column_map=column_map)
    return {
        "technical_eligibility": dict(DEFAULT_HILO_WEIGHTS["technical_eligibility"]),
        "control_authority": calibrate_weights_from_importance(
            DEFAULT_HILO_WEIGHTS["control_authority"],
            {
                "owner": cols["owner"],
                "renter": cols["renter"],
                "multi_unit": cols["multi_unit"],
                "residential_stability": cols["residential_stability"],
            },
            feature_importance,
            alpha=alpha,
            min_factor=min_factor,
        ),
        "curtailment_tolerance": calibrate_weights_from_importance(
            DEFAULT_HILO_WEIGHTS["curtailment_tolerance"],
            {
                "household_size": cols["household_size"],
                "occupancy_density": cols["occupancy_density"],
                "children": cols["children"],
                "older": cols["older"],
            },
            feature_importance,
            alpha=alpha,
            min_factor=min_factor,
        ),
        "demand_relevance": dict(DEFAULT_HILO_WEIGHTS["demand_relevance"]),
    }


__all__ = ["evaluate_hilo_alignment", "calibrate_hilo_weight_overrides", "DEFAULT_HILO_WEIGHTS"]
