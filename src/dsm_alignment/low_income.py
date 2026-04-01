from __future__ import annotations

import pandas as pd

from dsm_alignment.common import (
    calibrate_weights_from_importance,
    classify_quadrants,
    find_column_by_keywords,
    merge_weight_overrides,
    weighted_composite,
)


DEFAULT_LOW_INCOME_WEIGHTS = {
    "energy_vulnerability": {
        "low_income_rate": 0.35,
        "single_parent": 0.20,
        "household_size": 0.15,
        "crowding": 0.15,
        "renter": 0.15,
    },
    "system_relevance": {"winter_peak_share": 0.4, "peak_intensity": 0.3, "heating_slope": 0.3},
}


def _resolve_low_income_columns(df: pd.DataFrame, column_map: dict[str, str] | None = None) -> dict[str, str]:
    column_map = column_map or {}

    def pick(key: str, include: list[str], *, exclude: list[str] | None = None, prefer_percent: bool = False) -> str:
        if key in column_map:
            return column_map[key]
        found = find_column_by_keywords(df.columns, include=include, exclude=exclude, prefer_percent=prefer_percent)
        if found is None:
            raise KeyError(f"Could not resolve low-income proxy column for '{key}'.")
        return found

    try:
        single_parent = pick("single_parent", ["one-parent"], prefer_percent=True)
    except KeyError:
        single_parent = pick("single_parent", ["lone-parent"], prefer_percent=True)

    return {
        "winter_peak_share": pick("winter_peak_share", ["winter", "peak", "share"]),
        "heating_slope_per_hdd": pick("heating_slope_per_hdd", ["heating", "slope"]),
        "peak_intensity": pick("peak_intensity", ["winter", "peak", "intensity"]),
        "low_income_rate": pick("low_income_rate", ["low income"], prefer_percent=True),
        "single_parent": single_parent,
        "household_size": pick("household_size", ["average household size"]),
        "crowding": pick("crowding", ["persons per room"]),
        "renter": pick("renter", ["renter"], prefer_percent=True),
    }


def evaluate_low_income_alignment(
    df: pd.DataFrame,
    *,
    column_map: dict[str, str] | None = None,
    weight_overrides: dict[str, dict[str, float]] | None = None,
    quantile: float = 0.5,
) -> pd.DataFrame:
    cols = _resolve_low_income_columns(df, column_map=column_map)
    weights = merge_weight_overrides(DEFAULT_LOW_INCOME_WEIGHTS, weight_overrides)
    out = pd.DataFrame(index=df.index)

    vuln_df = pd.DataFrame(index=df.index)
    vuln_df["low_income_rate"] = pd.to_numeric(df[cols["low_income_rate"]], errors="coerce")
    vuln_df["single_parent"] = pd.to_numeric(df[cols["single_parent"]], errors="coerce")
    vuln_df["household_size"] = pd.to_numeric(df[cols["household_size"]], errors="coerce")
    vuln_df["crowding"] = pd.to_numeric(df[cols["crowding"]], errors="coerce")
    vuln_df["renter"] = pd.to_numeric(df[cols["renter"]], errors="coerce")
    out["energy_vulnerability"] = weighted_composite(
        vuln_df,
        weights["energy_vulnerability"],
    )

    relevance_df = pd.DataFrame(index=df.index)
    relevance_df["winter_peak_share"] = pd.to_numeric(df[cols["winter_peak_share"]], errors="coerce")
    relevance_df["peak_intensity"] = pd.to_numeric(df[cols["peak_intensity"]], errors="coerce")
    relevance_df["heating_slope"] = pd.to_numeric(df[cols["heating_slope_per_hdd"]], errors="coerce")
    out["system_relevance"] = weighted_composite(
        relevance_df,
        weights["system_relevance"],
    )

    out["alignment_class"] = classify_quadrants(
        out["system_relevance"],
        out["energy_vulnerability"],
        x_high_label="high_system_relevance",
        y_high_label="high_vulnerability",
        x_low_label="low_system_relevance",
        y_low_label="low_vulnerability",
        quantile=quantile,
    )
    return out.sort_index()


def calibrate_low_income_weight_overrides(
    df: pd.DataFrame,
    feature_importance: pd.Series,
    *,
    column_map: dict[str, str] | None = None,
    alpha: float = 1.0,
    min_factor: float = 0.5,
) -> dict[str, dict[str, float]]:
    cols = _resolve_low_income_columns(df, column_map=column_map)
    return {
        "energy_vulnerability": calibrate_weights_from_importance(
            DEFAULT_LOW_INCOME_WEIGHTS["energy_vulnerability"],
            {
                "low_income_rate": cols["low_income_rate"],
                "single_parent": cols["single_parent"],
                "household_size": cols["household_size"],
                "crowding": cols["crowding"],
                "renter": cols["renter"],
            },
            feature_importance,
            alpha=alpha,
            min_factor=min_factor,
        ),
        "system_relevance": dict(DEFAULT_LOW_INCOME_WEIGHTS["system_relevance"]),
    }


__all__ = ["evaluate_low_income_alignment", "calibrate_low_income_weight_overrides", "DEFAULT_LOW_INCOME_WEIGHTS"]
