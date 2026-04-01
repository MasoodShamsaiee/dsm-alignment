from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dsm_alignment.flexd import calibrate_flexd_weight_overrides
from dsm_alignment.hilo import calibrate_hilo_weight_overrides
from dsm_alignment.logisvert import calibrate_logisvert_weight_overrides
from dsm_alignment.low_income import calibrate_low_income_weight_overrides
from dsm_alignment.synthesis import alignment_overview_table, run_all_program_alignments


DEFAULT_DML_TARGETS = [
    "heating_slope_per_hdd",
    "cooling_slope_per_cdd",
    "heating_change_point_temp_c",
    "peak_load",
    "p90_top10_mean",
    "am_pm_peak_ratio",
    "ramp_up_rate",
]


@dataclass
class DMLImportanceResult:
    target_scores: pd.DataFrame
    feature_importance_by_target: pd.DataFrame
    global_feature_importance: pd.Series
    used_targets: list[str]
    used_feature_cols: list[str]


def _infer_census_feature_cols(df: pd.DataFrame, target_cols: list[str]) -> list[str]:
    excluded = {
        "winter_peak_share",
        "winter_peak_intensity",
        "mean_load",
        "mean_temp",
        "n_points",
        "r2",
        "cvrmse",
        "sse",
        "x0",
        "x1",
        "x2",
        "y0",
        "y1",
        "k0",
        "k1",
        "k2",
        "peak_load",
        "p90_top10_mean",
        "am_pm_peak_ratio",
        "ramp_up_rate",
    }
    excluded.update(target_cols)
    cols = []
    for c in df.columns:
        if c in excluded:
            continue
        if str(c).lower() in {"fsa", "city", "alignment_class"}:
            continue
        # Census columns in this project follow verbose label patterns with "/".
        if "/" not in str(c):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if int(s.notna().sum()) >= max(10, int(0.1 * len(df))):
            cols.append(str(c))
    return cols


def _get_xgb_regressor(**kwargs):
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "xgboost is required for DML workflow. Install xgboost in the current environment."
        ) from exc

    defaults = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "objective": "reg:squarederror",
        "n_jobs": -1,
    }
    defaults.update(kwargs)
    return XGBRegressor(**defaults)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _kfold_indices(n: int, n_splits: int, random_state: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    out = []
    for i in range(n_splits):
        te = np.asarray(folds[i], dtype=int)
        tr = np.concatenate([folds[j] for j in range(n_splits) if j != i]).astype(int)
        out.append((tr, te))
    return out


def estimate_census_importance_xgboost(
    df: pd.DataFrame,
    *,
    target_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
    min_samples: int = 80,
    n_splits: int = 5,
    target_weight_mode: str = "cv_r2_nonneg",
    xgb_params: dict | None = None,
) -> DMLImportanceResult:
    target_cols = target_cols or list(DEFAULT_DML_TARGETS)
    target_cols = [c for c in target_cols if c in df.columns]
    if not target_cols:
        raise ValueError("No target columns found in dataframe.")

    feature_cols = feature_cols or _infer_census_feature_cols(df, target_cols)
    if not feature_cols:
        raise ValueError("No usable numeric census feature columns found.")

    xgb_params = xgb_params or {}
    X_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    min_non_na = max(10, int(0.1 * len(X_raw)))
    usable_cols = [c for c in X_raw.columns if int(X_raw[c].notna().sum()) >= min_non_na]
    if not usable_cols:
        raise ValueError("No census feature column has enough non-missing samples.")
    X_raw = X_raw[usable_cols]
    X_filled = X_raw.fillna(X_raw.median(numeric_only=True)).fillna(0.0)

    target_score_rows = []
    importances: list[pd.Series] = []
    used_targets: list[str] = []

    for target in target_cols:
        y_all = pd.to_numeric(df[target], errors="coerce").astype(float)
        mask = y_all.notna()
        if int(mask.sum()) < min_samples:
            continue

        X = X_filled.loc[mask]
        y = y_all.loc[mask]

        model = _get_xgb_regressor(**xgb_params)
        k = max(2, min(int(n_splits), len(X) // 20))
        if k < 2:
            continue
        preds = pd.Series(index=X.index, dtype=float)
        for tr_idx, te_idx in _kfold_indices(len(X), n_splits=k, random_state=42):
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds.iloc[te_idx] = model.predict(X.iloc[te_idx])
        cv_r2 = _r2_score(y.to_numpy(), preds.to_numpy()) if preds.notna().all() else np.nan

        model.fit(X, y)
        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        imp = pd.Series(0.0, index=usable_cols, dtype=float)
        for i, col in enumerate(usable_cols):
            imp.loc[col] = float(gain.get(col, gain.get(f"f{i}", 0.0)))

        if float(imp.sum()) > 0:
            imp = imp / float(imp.sum())

        used_targets.append(target)
        importances.append(imp.rename(target))
        target_score_rows.append(
            {
                "target": target,
                "n_samples": int(len(X)),
                "cv_r2": cv_r2,
            }
        )

    if not importances:
        raise ValueError("No target model could be trained. Check missingness and min_samples.")

    imp_df = pd.concat(importances, axis=1).fillna(0.0)
    score_df = pd.DataFrame(target_score_rows).set_index("target").sort_index()

    if target_weight_mode == "cv_r2_nonneg":
        weights = score_df["cv_r2"].clip(lower=0.0).fillna(0.0)
        if float(weights.sum()) <= 0:
            weights = pd.Series(1.0, index=score_df.index)
    elif target_weight_mode == "uniform":
        weights = pd.Series(1.0, index=score_df.index)
    else:
        raise ValueError("target_weight_mode must be 'cv_r2_nonneg' or 'uniform'.")

    weights = weights / float(weights.sum())
    global_imp = (imp_df * weights.reindex(imp_df.columns)).sum(axis=1)
    if float(global_imp.sum()) > 0:
        global_imp = global_imp / float(global_imp.sum())

    return DMLImportanceResult(
        target_scores=score_df,
        feature_importance_by_target=imp_df.sort_index(),
        global_feature_importance=global_imp.sort_values(ascending=False),
        used_targets=used_targets,
        used_feature_cols=usable_cols,
    )


def build_weight_overrides_from_importance(
    df: pd.DataFrame,
    feature_importance: pd.Series,
    *,
    column_maps: dict[str, dict[str, str]] | None = None,
    alpha: float = 1.0,
    min_factor: float = 0.5,
) -> dict[str, dict[str, dict[str, float]]]:
    column_maps = column_maps or {}
    return {
        "flexd": calibrate_flexd_weight_overrides(
            df,
            feature_importance,
            column_map=column_maps.get("flexd"),
            alpha=alpha,
            min_factor=min_factor,
        ),
        "hilo": calibrate_hilo_weight_overrides(
            df,
            feature_importance,
            column_map=column_maps.get("hilo"),
            alpha=alpha,
            min_factor=min_factor,
        ),
        "logisvert": calibrate_logisvert_weight_overrides(
            df,
            feature_importance,
            column_map=column_maps.get("logisvert"),
            alpha=alpha,
            min_factor=min_factor,
        ),
        "low_income": calibrate_low_income_weight_overrides(
            df,
            feature_importance,
            column_map=column_maps.get("low_income"),
            alpha=alpha,
            min_factor=min_factor,
        ),
    }


def run_dml_weighted_alignment(
    df: pd.DataFrame,
    *,
    target_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
    column_maps: dict[str, dict[str, str]] | None = None,
    alpha: float = 1.0,
    min_factor: float = 0.5,
    min_samples: int = 80,
    n_splits: int = 5,
    quantile: float = 0.5,
    xgb_params: dict | None = None,
) -> dict:
    dml = estimate_census_importance_xgboost(
        df,
        target_cols=target_cols,
        feature_cols=feature_cols,
        min_samples=min_samples,
        n_splits=n_splits,
        xgb_params=xgb_params,
    )
    overrides = build_weight_overrides_from_importance(
        df,
        dml.global_feature_importance,
        column_maps=column_maps,
        alpha=alpha,
        min_factor=min_factor,
    )
    results = run_all_program_alignments(
        df,
        column_maps=column_maps,
        weight_overrides=overrides,
        quantile=quantile,
    )
    return {
        "dml": dml,
        "weight_overrides": overrides,
        "alignment_results": results,
        "alignment_overview": alignment_overview_table(results),
    }
