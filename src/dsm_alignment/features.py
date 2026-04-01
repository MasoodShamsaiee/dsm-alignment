from __future__ import annotations

import pandas as pd

from urban_energy_core.domain.city import City


def city_fsa_feature_table(
    city: City,
    *,
    per_capita: bool = True,
    weather_normalized: bool = False,
    prism_mode: str = "segmented",
    winter_only: bool = True,
    weekday_only: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    prism = city.compute_prism_table(
        per_capita=per_capita,
        weather_normalized=weather_normalized,
        mode=prism_mode,
        show_progress=show_progress,
    )
    short_term = city.compute_short_term_table(
        per_capita=per_capita,
        weather_normalized=weather_normalized,
        winter_only=winter_only,
        weekday_only=weekday_only,
        aggregate=True,
        show_progress=show_progress,
    )

    if prism.empty and short_term.empty:
        raise ValueError(f"No PRISM or short-term metrics produced for city '{city.name}'.")

    frames = []
    if not prism.empty:
        frames.append(prism)
    if not short_term.empty:
        frames.append(short_term)

    features = pd.concat(frames, axis=1)
    features = features.loc[:, ~features.columns.duplicated()].copy()

    census_rows = {}
    for code in city.list_fsa_codes():
        fsa = city.get_fsa(code)
        if fsa.census is None:
            continue
        if hasattr(fsa.census, "to_dict"):
            census_rows[code] = pd.Series(fsa.census).copy()

    if census_rows:
        census_df = pd.DataFrame(census_rows).T
        census_df.index.name = "fsa"
        features = features.join(census_df, how="left")

    if "peak_load" in features.columns:
        peak_total = float(pd.to_numeric(features["peak_load"], errors="coerce").sum())
        if peak_total > 0:
            features["winter_peak_share"] = pd.to_numeric(features["peak_load"], errors="coerce") / peak_total

    if {"p90_top10_mean", "mean_load"}.issubset(features.columns):
        mean_load = pd.to_numeric(features["mean_load"], errors="coerce")
        top = pd.to_numeric(features["p90_top10_mean"], errors="coerce")
        features["winter_peak_intensity"] = top / mean_load.replace(0, pd.NA)

    return features.sort_index()


def city_fsa_feature_table_from_cache(
    city: City,
    *,
    prism_attr: str = "prism_table",
    short_term_attr: str = "short_term_table",
) -> pd.DataFrame:
    """
    Build FSA feature table using precomputed tables already attached on the city object.
    This function does not recompute PRISM or short-term metrics.
    """
    prism = getattr(city, prism_attr, None)
    short_term = getattr(city, short_term_attr, None)

    if not isinstance(prism, pd.DataFrame) or prism.empty:
        raise ValueError(f"City '{city.name}' is missing non-empty cached '{prism_attr}'.")
    if not isinstance(short_term, pd.DataFrame) or short_term.empty:
        raise ValueError(f"City '{city.name}' is missing non-empty cached '{short_term_attr}'.")

    prism = prism.copy()
    short_term = short_term.copy()
    prism.index = prism.index.astype(str)
    short_term.index = short_term.index.astype(str)
    prism.index.name = "fsa"
    short_term.index.name = "fsa"

    features = pd.concat([prism, short_term], axis=1)
    features = features.loc[:, ~features.columns.duplicated()].copy()

    census_rows = {}
    for code in city.list_fsa_codes():
        fsa = city.get_fsa(code)
        if fsa.census is None:
            continue
        if hasattr(fsa.census, "to_dict"):
            census_rows[str(code)] = pd.Series(fsa.census).copy()

    if census_rows:
        census_df = pd.DataFrame(census_rows).T
        census_df.index.name = "fsa"
        features = features.join(census_df, how="left")

    if "peak_load" in features.columns and "winter_peak_share" not in features.columns:
        peak_total = float(pd.to_numeric(features["peak_load"], errors="coerce").sum())
        if peak_total > 0:
            features["winter_peak_share"] = pd.to_numeric(features["peak_load"], errors="coerce") / peak_total

    if {"p90_top10_mean", "mean_load"}.issubset(features.columns) and "winter_peak_intensity" not in features.columns:
        mean_load = pd.to_numeric(features["mean_load"], errors="coerce")
        top = pd.to_numeric(features["p90_top10_mean"], errors="coerce")
        features["winter_peak_intensity"] = top / mean_load.replace(0, pd.NA)

    return features.sort_index()
