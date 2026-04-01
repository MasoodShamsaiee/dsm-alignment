from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def normalized_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    s = _to_numeric(series)
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return s.rank(pct=True, ascending=ascending, method="average")


def weighted_composite(
    df: pd.DataFrame,
    components: dict[str, float],
    *,
    invert: Iterable[str] | None = None,
) -> pd.Series:
    invert = set(invert or [])
    missing = [c for c in components if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for composite score: {missing}")

    out = pd.Series(0.0, index=df.index, dtype=float)
    total_weight = float(sum(abs(w) for w in components.values()))
    if total_weight <= 0:
        raise ValueError("Composite weights must have a positive total.")

    for col, weight in components.items():
        asc = col not in invert
        out = out + normalized_rank(df[col], ascending=asc) * float(weight)
    return out / total_weight


def classify_quadrants(
    x: pd.Series,
    y: pd.Series,
    *,
    x_high_label: str,
    y_high_label: str,
    x_low_label: str,
    y_low_label: str,
    quantile: float = 0.5,
) -> pd.Series:
    x_thr = float(x.quantile(quantile))
    y_thr = float(y.quantile(quantile))
    x_high = x >= x_thr
    y_high = y >= y_thr

    out = pd.Series(index=x.index, dtype="string")
    out.loc[x_high & y_high] = f"{x_high_label}_{y_high_label}"
    out.loc[x_high & ~y_high] = f"{x_high_label}_{y_low_label}"
    out.loc[~x_high & y_high] = f"{x_low_label}_{y_high_label}"
    out.loc[~x_high & ~y_high] = f"{x_low_label}_{y_low_label}"
    return out


def find_column_by_keywords(
    columns: pd.Index,
    *,
    include: list[str],
    exclude: list[str] | None = None,
    prefer_percent: bool = False,
) -> str | None:
    exclude = exclude or []
    candidates: list[tuple[float, str]] = []
    for col in columns:
        col_l = str(col).lower()
        if any(k.lower() not in col_l for k in include):
            continue
        if any(k.lower() in col_l for k in exclude):
            continue

        score = float(sum(col_l.count(k.lower()) for k in include))
        if prefer_percent and "%" in col_l:
            score += 2.0
        if re.search(r"\btotal\b", col_l):
            score -= 1.0
        candidates.append((score, str(col)))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def merge_weight_overrides(
    defaults: dict[str, dict[str, float]],
    overrides: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {k: dict(v) for k, v in defaults.items()}
    if not overrides:
        return out
    for section, section_weights in overrides.items():
        if section not in out:
            out[section] = dict(section_weights)
            continue
        out[section].update(section_weights)
    return out


def calibrate_weights_from_importance(
    base_weights: dict[str, float],
    proxy_to_column: dict[str, str],
    feature_importance: pd.Series,
    *,
    alpha: float = 1.0,
    min_factor: float = 0.5,
) -> dict[str, float]:
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")
    if not (0.0 < min_factor <= 1.0):
        raise ValueError("min_factor must be in (0, 1].")

    fi = pd.to_numeric(feature_importance, errors="coerce")
    fi = fi[fi.notna() & (fi >= 0)]

    keys = list(base_weights.keys())
    proxy_scores = pd.Series(index=keys, dtype=float)
    for key in keys:
        col = proxy_to_column.get(key, key)
        proxy_scores.loc[key] = float(fi.get(col, np.nan))

    valid = proxy_scores[proxy_scores.notna() & (proxy_scores > 0)]
    if valid.empty:
        return dict(base_weights)

    rel = valid / float(valid.sum())
    scaled = rel * float(len(valid))  # mean=1 across valid proxies

    factors = pd.Series(1.0, index=keys, dtype=float)
    for key in valid.index:
        raw = float(scaled.loc[key])
        damped = (1.0 - alpha) + alpha * raw
        factors.loc[key] = max(min_factor, damped)

    out = {}
    for key, w in base_weights.items():
        out[key] = float(w) * float(factors.loc[key])

    z = float(sum(out.values()))
    if z > 0:
        out = {k: v / z for k, v in out.items()}
    return out
