from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from dsm_alignment.dml import _infer_census_feature_cols, build_weight_overrides_from_importance
from dsm_alignment.synthesis import alignment_overview_table, run_all_program_alignments


DEFAULT_SHORT_TERM_DAILY_TARGETS = [
    "peak_load",
    "p90_top10_mean",
    "am_pm_peak_ratio",
    "ramp_up_rate",
]

DEFAULT_PRISM_TARGETS = [
    "heating_slope_per_hdd",
    "cooling_slope_per_cdd",
    "heating_change_point_temp_c",
    "baseload_intercept",
]


def _discover_dtw_targets(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        s = str(c)
        if not s.startswith("dtw_"):
            continue
        x = pd.to_numeric(df[s], errors="coerce")
        if int(x.notna().sum()) >= max(10, int(0.1 * len(df))):
            out.append(s)
    return out


@dataclass
class DistributionalImportanceResult:
    target_scores: pd.DataFrame
    feature_importance_by_target: pd.DataFrame
    global_feature_importance: pd.Series
    used_targets: list[str]
    used_feature_cols: list[str]
    source: str


def _get_xgb_classifier(**kwargs):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover
        raise ImportError("xgboost is required for distributional workflow.") from exc

    defaults = {
        "n_estimators": 350,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "mlogloss",
        "objective": "multi:softprob",
    }
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def _prepare_X(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    X_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    min_non_na = max(10, int(0.1 * len(X_raw)))
    usable_cols = [c for c in X_raw.columns if int(X_raw[c].notna().sum()) >= min_non_na]
    if not usable_cols:
        raise ValueError("No usable census feature columns after missingness filter.")
    X = X_raw[usable_cols].fillna(X_raw[usable_cols].median(numeric_only=True)).fillna(0.0)
    return X, usable_cols


def _quantile_bins(y: pd.Series, n_bins: int = 4) -> pd.Series:
    y_num = pd.to_numeric(y, errors="coerce")
    labels = pd.qcut(y_num, q=n_bins, labels=False, duplicates="drop")
    return labels.astype("Int64")


def _class_representatives(y: pd.Series, y_bins: pd.Series) -> dict[int, float]:
    tmp = pd.DataFrame({"y": pd.to_numeric(y, errors="coerce"), "bin": y_bins})
    tmp = tmp.dropna(subset=["y", "bin"])
    if tmp.empty:
        return {}
    reps = tmp.groupby("bin")["y"].median()
    out: dict[int, float] = {}
    for k, v in reps.items():
        try:
            out[int(k)] = float(v)
        except Exception:
            continue
    return out


def _fit_distribution_models(
    df: pd.DataFrame,
    *,
    target_cols: list[str],
    feature_cols: list[str] | None = None,
    n_bins: int = 4,
    min_samples: int = 200,
    min_class_count: int = 20,
    n_splits: int = 5,
    xgb_params: dict | None = None,
    source: str,
) -> DistributionalImportanceResult:
    target_cols = [c for c in target_cols if c in df.columns]
    if not target_cols:
        raise ValueError("No distributional targets found in dataframe.")

    feature_cols = feature_cols or _infer_census_feature_cols(df, target_cols)
    if not feature_cols:
        raise ValueError("No census feature columns provided/found for distributional modeling.")

    X_full, usable_cols = _prepare_X(df, feature_cols)
    xgb_params = xgb_params or {}

    score_rows: list[dict] = []
    importances: list[pd.Series] = []
    used_targets: list[str] = []

    for target in target_cols:
        y_bins = _quantile_bins(df[target], n_bins=n_bins)
        mask = y_bins.notna()
        if int(mask.sum()) < min_samples:
            continue

        X = X_full.loc[mask]
        y = y_bins.loc[mask].astype(int)
        vc = y.value_counts()
        if len(vc) < 2 or int(vc.min()) < min_class_count:
            continue

        k = min(int(n_splits), int(vc.min()), 8)
        if k < 2:
            continue

        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        y_pred = pd.Series(index=X.index, dtype=int)
        y_proba = pd.DataFrame(index=X.index, dtype=float)

        for tr_idx, te_idx in cv.split(X, y):
            model = _get_xgb_classifier(num_class=int(y.nunique()), **xgb_params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            proba = model.predict_proba(X.iloc[te_idx])
            pred = np.argmax(proba, axis=1)
            y_pred.iloc[te_idx] = pred
            fold_probs = pd.DataFrame(proba, index=X.index[te_idx], columns=[f"class_{i}" for i in range(proba.shape[1])])
            y_proba = pd.concat([y_proba, fold_probs], axis=0)

        y_true = y.loc[y_pred.index].astype(int)
        macro_f1 = float(f1_score(y_true, y_pred.astype(int), average="macro"))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred.astype(int)))

        ll = np.nan
        try:
            yp = y_proba.loc[y_true.index].sort_index()
            yt = y_true.sort_index()
            ll = float(log_loss(yt, yp.to_numpy(), labels=np.arange(yp.shape[1])))
        except Exception:
            pass

        final_model = _get_xgb_classifier(num_class=int(y.nunique()), **xgb_params)
        final_model.fit(X, y)
        gain = final_model.get_booster().get_score(importance_type="gain")
        imp = pd.Series(0.0, index=usable_cols, dtype=float)
        for i, col in enumerate(usable_cols):
            imp.loc[col] = float(gain.get(col, gain.get(f"f{i}", 0.0)))
        if float(imp.sum()) > 0:
            imp = imp / float(imp.sum())

        used_targets.append(target)
        importances.append(imp.rename(target))
        score_rows.append(
            {
                "target": target,
                "n_samples": int(len(X)),
                "n_classes": int(y.nunique()),
                "macro_f1": macro_f1,
                "balanced_accuracy": bal_acc,
                "cv_logloss": ll,
            }
        )

    if not importances:
        raise ValueError(f"No target distribution model could be trained for source='{source}'.")

    imp_df = pd.concat(importances, axis=1).fillna(0.0)
    score_df = pd.DataFrame(score_rows).set_index("target").sort_index()

    weights = score_df["macro_f1"].clip(lower=0.0).fillna(0.0)
    if float(weights.sum()) <= 0:
        weights = pd.Series(1.0, index=score_df.index)
    weights = weights / float(weights.sum())

    global_imp = (imp_df * weights.reindex(imp_df.columns)).sum(axis=1)
    if float(global_imp.sum()) > 0:
        global_imp = global_imp / float(global_imp.sum())

    return DistributionalImportanceResult(
        target_scores=score_df,
        feature_importance_by_target=imp_df.sort_index(),
        global_feature_importance=global_imp.sort_values(ascending=False),
        used_targets=used_targets,
        used_feature_cols=usable_cols,
        source=source,
    )


def build_daily_short_term_panel(
    city,
    *,
    per_capita: bool = True,
    weather_normalized: bool = False,
    winter_only: bool = True,
    weekday_only: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    daily = city.compute_short_term_table(
        per_capita=per_capita,
        weather_normalized=weather_normalized,
        winter_only=winter_only,
        weekday_only=weekday_only,
        aggregate=False,
        show_progress=show_progress,
    )
    if daily.empty:
        raise ValueError(f"No daily short-term panel generated for city '{city.name}'.")

    panel = daily.reset_index()
    census_rows = []
    for code in city.list_fsa_codes():
        row = city.get_fsa(code).census
        if row is None:
            continue
        s = pd.Series(row).copy()
        s["fsa"] = code
        census_rows.append(s)
    if not census_rows:
        return panel

    census_df = pd.DataFrame(census_rows)
    panel = panel.merge(census_df, on="fsa", how="left")
    return panel


def estimate_distributional_short_term_importance_xgboost(
    daily_panel: pd.DataFrame,
    *,
    target_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
    n_bins: int = 4,
    min_samples: int = 200,
    min_class_count: int = 20,
    n_splits: int = 5,
    xgb_params: dict | None = None,
) -> DistributionalImportanceResult:
    return _fit_distribution_models(
        daily_panel,
        target_cols=target_cols or list(DEFAULT_SHORT_TERM_DAILY_TARGETS),
        feature_cols=feature_cols,
        n_bins=n_bins,
        min_samples=min_samples,
        min_class_count=min_class_count,
        n_splits=n_splits,
        xgb_params=xgb_params,
        source="short_term_daily",
    )


def estimate_distributional_prism_importance_xgboost(
    fsa_features: pd.DataFrame,
    *,
    target_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
    n_bins: int = 4,
    min_samples: int = 60,
    min_class_count: int = 12,
    n_splits: int = 4,
    xgb_params: dict | None = None,
) -> DistributionalImportanceResult:
    return _fit_distribution_models(
        fsa_features,
        target_cols=target_cols or list(DEFAULT_PRISM_TARGETS),
        feature_cols=feature_cols,
        n_bins=n_bins,
        min_samples=min_samples,
        min_class_count=min_class_count,
        n_splits=n_splits,
        xgb_params=xgb_params,
        source="prism_fsa",
    )


def combine_distributional_importance(
    results: list[DistributionalImportanceResult],
    *,
    weights: list[float] | None = None,
) -> pd.Series:
    if not results:
        raise ValueError("results must not be empty.")
    if weights is None:
        weights = []
        for r in results:
            if r.target_scores.empty:
                weights.append(0.0)
            else:
                weights.append(float(r.target_scores["macro_f1"].clip(lower=0).mean()))
        if float(sum(weights)) <= 0:
            weights = [1.0] * len(results)
    w = np.asarray(weights, dtype=float)
    if float(w.sum()) <= 0:
        w = np.ones_like(w)
    w = w / float(w.sum())

    union_index = sorted(set().union(*[set(r.global_feature_importance.index) for r in results]))
    out = pd.Series(0.0, index=union_index, dtype=float)
    for wi, r in zip(w, results):
        g = r.global_feature_importance.reindex(union_index).fillna(0.0)
        out = out + wi * g
    if float(out.sum()) > 0:
        out = out / float(out.sum())
    return out.sort_values(ascending=False)


def predict_distribution_for_rows_xgboost(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    *,
    target_cols: list[str],
    feature_cols: list[str] | None = None,
    n_bins: int = 4,
    min_samples: int = 200,
    min_class_count: int = 20,
    xgb_params: dict | None = None,
) -> pd.DataFrame:
    """
    Train per-target distributional classifiers on train_df and predict class
    probabilities for each row in predict_df.

    Output columns per target:
    - {target}__q0, {target}__q1, ... (probabilities)
    - {target}__pred_q
    - {target}__pred_prob
    - {target}__pred_expected
    """
    targets = [c for c in target_cols if c in train_df.columns]
    if not targets:
        return pd.DataFrame(index=predict_df.index)

    use_feature_cols = feature_cols or _infer_census_feature_cols(train_df, targets)
    if not use_feature_cols:
        return pd.DataFrame(index=predict_df.index)

    xgb_params = xgb_params or {}

    X_train_raw = train_df[use_feature_cols].apply(pd.to_numeric, errors="coerce")
    min_non_na = max(10, int(0.1 * len(X_train_raw)))
    usable_cols = [c for c in X_train_raw.columns if int(X_train_raw[c].notna().sum()) >= min_non_na]
    if not usable_cols:
        return pd.DataFrame(index=predict_df.index)

    med = X_train_raw[usable_cols].median(numeric_only=True)
    X_train = X_train_raw[usable_cols].fillna(med).fillna(0.0)

    X_pred_raw = predict_df.reindex(columns=usable_cols).apply(pd.to_numeric, errors="coerce")
    X_pred = X_pred_raw.fillna(med).fillna(0.0)

    out = pd.DataFrame(index=predict_df.index)

    for target in targets:
        y_num = pd.to_numeric(train_df[target], errors="coerce")
        y_bins = _quantile_bins(y_num, n_bins=n_bins)
        mask = y_bins.notna()
        if int(mask.sum()) < min_samples:
            continue

        X = X_train.loc[mask]
        y = y_bins.loc[mask].astype(int)
        class_rep = _class_representatives(y_num.loc[mask], y)
        vc = y.value_counts()
        if len(vc) < 2 or int(vc.min()) < min_class_count:
            continue

        n_classes = int(y.nunique())
        model = _get_xgb_classifier(num_class=n_classes, **xgb_params)
        model.fit(X, y)

        proba = model.predict_proba(X_pred)
        for c in range(proba.shape[1]):
            out[f"{target}__q{c}"] = proba[:, c]
        out[f"{target}__pred_q"] = np.argmax(proba, axis=1).astype(int)
        out[f"{target}__pred_prob"] = np.max(proba, axis=1).astype(float)
        if class_rep:
            exp_vals = np.zeros(proba.shape[0], dtype=float)
            for c in range(proba.shape[1]):
                exp_vals += proba[:, c] * float(class_rep.get(c, np.nan))
            out[f"{target}__pred_expected"] = exp_vals

    return out


def run_distributional_weighted_alignment(
    city,
    fsa_features: pd.DataFrame,
    *,
    daily_panel: pd.DataFrame | None = None,
    recompute_short_term_if_missing: bool = False,
    feature_cols: list[str] | None = None,
    column_maps: dict[str, dict[str, str]] | None = None,
    alpha: float = 0.8,
    min_factor: float = 0.6,
    quantile: float = 0.5,
    include_prism: bool = True,
    include_dtw: bool = True,
    short_term_n_bins: int = 4,
    prism_n_bins: int = 4,
    dtw_n_bins: int = 4,
    xgb_params: dict | None = None,
    show_progress: bool = True,
) -> dict:
    if daily_panel is None and recompute_short_term_if_missing:
        daily_panel = build_daily_short_term_panel(
            city,
            winter_only=True,
            weekday_only=False,
            show_progress=show_progress,
        )
    elif daily_panel is not None:
        daily_panel = daily_panel.copy()
    else:
        raise ValueError(
            "daily_panel is required when recompute_short_term_if_missing=False. "
            "Provide a precomputed daily panel (e.g., city.short_term_daily_panel)."
        )

    if daily_panel.empty:
        raise ValueError("daily_panel is empty. Provide a non-empty daily panel or allow internal build.")

    if "fsa" in daily_panel.columns:
        daily_panel["fsa"] = daily_panel["fsa"].astype(str)

    # If daily panel lacks some selected census features, merge them from fsa_features.
    if feature_cols:
        missing_in_panel = [c for c in feature_cols if c not in daily_panel.columns]
        if missing_in_panel and "fsa" in daily_panel.columns:
            merge_cols = [c for c in missing_in_panel if c in fsa_features.columns]
            if merge_cols:
                right = fsa_features[merge_cols].copy()
                right.index = right.index.astype(str)
                right = right.reset_index().rename(columns={"index": "fsa"})
                daily_panel = daily_panel.merge(right, on="fsa", how="left")
    short_res = estimate_distributional_short_term_importance_xgboost(
        daily_panel,
        feature_cols=feature_cols,
        n_bins=short_term_n_bins,
        xgb_params=xgb_params,
    )
    short_pred = predict_distribution_for_rows_xgboost(
        daily_panel,
        fsa_features,
        target_cols=list(DEFAULT_SHORT_TERM_DAILY_TARGETS),
        feature_cols=feature_cols,
        n_bins=short_term_n_bins,
        min_samples=200,
        min_class_count=20,
        xgb_params=xgb_params,
    )
    results_for_combine = [short_res]

    prism_res = None
    prism_pred = None
    if include_prism:
        prism_res = estimate_distributional_prism_importance_xgboost(
            fsa_features,
            feature_cols=feature_cols,
            n_bins=prism_n_bins,
            xgb_params=xgb_params,
        )
        prism_pred = predict_distribution_for_rows_xgboost(
            fsa_features,
            fsa_features,
            target_cols=list(DEFAULT_PRISM_TARGETS),
            feature_cols=feature_cols,
            n_bins=prism_n_bins,
            min_samples=60,
            min_class_count=12,
            xgb_params=xgb_params,
        )
        results_for_combine.append(prism_res)

    dtw_res = None
    dtw_pred = None
    if include_dtw:
        dtw_targets = _discover_dtw_targets(fsa_features)
        if dtw_targets:
            dtw_res = _fit_distribution_models(
                fsa_features,
                target_cols=dtw_targets,
                feature_cols=feature_cols,
                n_bins=dtw_n_bins,
                min_samples=60,
                min_class_count=12,
                n_splits=4,
                xgb_params=xgb_params,
                source="dtw_fsa",
            )
            dtw_pred = predict_distribution_for_rows_xgboost(
                fsa_features,
                fsa_features,
                target_cols=dtw_targets,
                feature_cols=feature_cols,
                n_bins=dtw_n_bins,
                min_samples=60,
                min_class_count=12,
                xgb_params=xgb_params,
            )
            results_for_combine.append(dtw_res)

    global_imp = combine_distributional_importance(results_for_combine)
    overrides = build_weight_overrides_from_importance(
        fsa_features,
        global_imp,
        column_maps=column_maps,
        alpha=alpha,
        min_factor=min_factor,
    )
    alignment_results = run_all_program_alignments(
        fsa_features,
        column_maps=column_maps,
        weight_overrides=overrides,
        quantile=quantile,
    )
    out = {
        "short_term_distributional": short_res,
        "short_term_fsa_distribution_predictions": short_pred,
        "weight_overrides": overrides,
        "global_feature_importance": global_imp,
        "alignment_results": alignment_results,
        "alignment_overview": alignment_overview_table(alignment_results),
    }
    if prism_res is not None:
        out["prism_distributional"] = prism_res
    if prism_pred is not None:
        out["prism_fsa_distribution_predictions"] = prism_pred
    if dtw_res is not None:
        out["dtw_distributional"] = dtw_res
    if dtw_pred is not None:
        out["dtw_fsa_distribution_predictions"] = dtw_pred
    return out
