from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold


DEFAULT_PROBABILISTIC_TARGETS = [
    "heating_slope_per_hdd",
    "cooling_slope_per_cdd",
    "heating_change_point_temp_c",
    "baseload_intercept",
    "peak_load",
    "p90_top10_mean",
    "am_pm_peak_ratio",
    "ramp_up_rate",
]


@dataclass
class ProbabilisticTargetModel:
    target: str
    feature_cols: list[str]
    quantiles: list[float]
    models_by_quantile: dict[float, GradientBoostingRegressor]
    train_target_quantiles: dict[float, float]
    cv_pinball_loss_mean: float
    cv_pinball_loss_by_quantile: dict[float, float]
    n_samples: int


def _infer_census_feature_cols(df: pd.DataFrame, target_cols: list[str]) -> list[str]:
    cols = []
    excluded = set(target_cols)
    excluded.update({"fsa", "city", "date", "timestamp", "alignment_class"})
    for c in df.columns:
        sc = str(c)
        if sc in excluded:
            continue
        if "/" not in sc:
            continue
        x = pd.to_numeric(df[sc], errors="coerce")
        if int(x.notna().sum()) >= max(10, int(0.1 * len(df))):
            cols.append(sc)
    return cols


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    e = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def fit_probabilistic_energy_models(
    train_df: pd.DataFrame,
    *,
    target_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
    quantiles: list[float] | None = None,
    min_samples: int = 80,
    n_splits: int = 5,
    random_state: int = 42,
    model_params: dict | None = None,
) -> dict[str, ProbabilisticTargetModel]:
    targets = [c for c in (target_cols or DEFAULT_PROBABILISTIC_TARGETS) if c in train_df.columns]
    if not targets:
        raise ValueError("No probabilistic targets found in train_df.")

    use_quantiles = sorted(set(float(q) for q in (quantiles or [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])))
    if any((q <= 0.0 or q >= 1.0) for q in use_quantiles):
        raise ValueError("quantiles must be strictly in (0,1).")

    feats = feature_cols or _infer_census_feature_cols(train_df, targets)
    if not feats:
        raise ValueError("No usable census feature columns for probabilistic modeling.")

    X_raw = train_df[feats].apply(pd.to_numeric, errors="coerce")
    min_non_na = max(10, int(0.1 * len(X_raw)))
    usable_cols = [c for c in X_raw.columns if int(X_raw[c].notna().sum()) >= min_non_na]
    if not usable_cols:
        raise ValueError("No usable feature columns after missingness filter.")
    X_full = X_raw[usable_cols]
    med = X_full.median(numeric_only=True)
    X_full = X_full.fillna(med).fillna(0.0)

    params = {
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "random_state": random_state,
    }
    if model_params:
        params.update(model_params)

    out: dict[str, ProbabilisticTargetModel] = {}
    for target in targets:
        y = pd.to_numeric(train_df[target], errors="coerce")
        mask = y.notna()
        if int(mask.sum()) < min_samples:
            continue

        X = X_full.loc[mask]
        yv = y.loc[mask].astype(float)

        k = min(max(2, int(n_splits)), max(2, len(X) // 20))
        cv = KFold(n_splits=k, shuffle=True, random_state=random_state)

        cv_loss_by_q: dict[float, list[float]] = {q: [] for q in use_quantiles}
        for tr_idx, te_idx in cv.split(X):
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = yv.iloc[tr_idx], yv.iloc[te_idx]
            for q in use_quantiles:
                m = GradientBoostingRegressor(loss="quantile", alpha=q, **params)
                m.fit(Xtr, ytr)
                yp = m.predict(Xte)
                cv_loss_by_q[q].append(_pinball_loss(yte.to_numpy(), yp, q))

        fitted: dict[float, GradientBoostingRegressor] = {}
        for q in use_quantiles:
            m = GradientBoostingRegressor(loss="quantile", alpha=q, **params)
            m.fit(X, yv)
            fitted[q] = m

        cv_mean_by_q = {q: float(np.mean(v)) if v else np.nan for q, v in cv_loss_by_q.items()}
        cv_mean = float(np.nanmean(list(cv_mean_by_q.values())))
        train_q = {q: float(np.nanquantile(yv, q)) for q in use_quantiles}
        out[target] = ProbabilisticTargetModel(
            target=target,
            feature_cols=usable_cols,
            quantiles=use_quantiles,
            models_by_quantile=fitted,
            train_target_quantiles=train_q,
            cv_pinball_loss_mean=cv_mean,
            cv_pinball_loss_by_quantile=cv_mean_by_q,
            n_samples=int(len(X)),
        )

    if not out:
        raise ValueError("No probabilistic target model was trainable.")
    return out


def predict_probabilistic_energy(
    models: dict[str, ProbabilisticTargetModel],
    predict_df: pd.DataFrame,
    *,
    n_draws: int = 300,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Returns:
    - summary_wide: per-row summary columns, incl expected/p05/p50/p95/std.
    - draws_by_target: sampled predictive draws per target (rows=index of predict_df).
    """
    if not models:
        return pd.DataFrame(index=predict_df.index), {}

    idx = predict_df.index
    summary = pd.DataFrame(index=idx)
    draws_out: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(random_state)

    for target, tm in models.items():
        X = predict_df.reindex(columns=tm.feature_cols).apply(pd.to_numeric, errors="coerce")
        med = X.median(numeric_only=True)
        X = X.fillna(med).fillna(0.0)

        q_levels = np.array(sorted(tm.quantiles), dtype=float)
        pred_q = np.column_stack([tm.models_by_quantile[q].predict(X) for q in q_levels])
        # enforce monotonic quantile predictions per row
        pred_q = np.sort(pred_q, axis=1)

        # empirical draws from interpolated quantile function
        U = rng.uniform(0.0, 1.0, size=(len(X), int(n_draws)))
        S = np.zeros_like(U, dtype=float)
        for i in range(len(X)):
            S[i, :] = np.interp(U[i, :], q_levels, pred_q[i, :], left=pred_q[i, 0], right=pred_q[i, -1])

        draws = pd.DataFrame(S, index=idx, columns=[f"draw_{k+1}" for k in range(int(n_draws))])
        draws_out[target] = draws

        summary[f"{target}__pred_expected"] = np.mean(S, axis=1)
        summary[f"{target}__pred_std"] = np.std(S, axis=1)
        summary[f"{target}__pred_q05"] = np.quantile(S, 0.05, axis=1)
        summary[f"{target}__pred_q25"] = np.quantile(S, 0.25, axis=1)
        summary[f"{target}__pred_q50"] = np.quantile(S, 0.50, axis=1)
        summary[f"{target}__pred_q75"] = np.quantile(S, 0.75, axis=1)
        summary[f"{target}__pred_q95"] = np.quantile(S, 0.95, axis=1)

    return summary, draws_out


def fit_and_predict_probabilistic_energy(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    *,
    target_cols: list[str] | None = None,
    feature_cols: list[str] | None = None,
    quantiles: list[float] | None = None,
    min_samples: int = 80,
    n_splits: int = 5,
    n_draws: int = 300,
    random_state: int = 42,
    model_params: dict | None = None,
) -> dict:
    models = fit_probabilistic_energy_models(
        train_df,
        target_cols=target_cols,
        feature_cols=feature_cols,
        quantiles=quantiles,
        min_samples=min_samples,
        n_splits=n_splits,
        random_state=random_state,
        model_params=model_params,
    )
    summary, draws = predict_probabilistic_energy(
        models,
        predict_df,
        n_draws=n_draws,
        random_state=random_state,
    )
    model_scores = pd.DataFrame(
        [
            {
                "target": t,
                "n_samples": m.n_samples,
                "cv_pinball_loss_mean": m.cv_pinball_loss_mean,
            }
            for t, m in models.items()
        ]
    ).set_index("target")
    return {
        "models": models,
        "summary_predictions": summary,
        "draws_by_target": draws,
        "model_scores": model_scores,
    }
