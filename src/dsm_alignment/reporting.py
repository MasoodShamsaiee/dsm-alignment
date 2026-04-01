from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dsm_alignment.synthesis import alignment_overview_table


PROGRAM_SCORE_COLUMNS = {
    "flexd": ("demand_relevance", "participation_capacity"),
    "hilo": ("demand_relevance", "hilo_suitability"),
    "logisvert": ("structural_demand_relevance", "overall_capacity"),
    "low_income": ("system_relevance", "energy_vulnerability"),
}

PROGRAM_SUBINDEX_COLUMNS = {
    "flexd": ["temporal_flexibility", "demand_elasticity", "participation_capacity", "demand_relevance"],
    "hilo": ["technical_eligibility", "control_authority", "curtailment_tolerance", "hilo_suitability", "demand_relevance"],
    "logisvert": ["structural_demand_relevance", "adoption_capacity", "persistence_capacity", "overall_capacity"],
    "low_income": ["energy_vulnerability", "system_relevance"],
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _city_alignment_geodf(city, program_df: pd.DataFrame):
    import geopandas as gpd

    rows = []
    for code in city.list_fsa_codes():
        if code not in program_df.index:
            continue
        geom = city.get_fsa(code).geometry
        if geom is None:
            continue
        row = program_df.loc[code].to_dict()
        row["fsa"] = code
        row["geometry"] = geom
        rows.append(row)
    if not rows:
        raise ValueError(f"No geometry available for program map in city '{city.name}'.")
    return gpd.GeoDataFrame(rows, geometry="geometry")


def _alignment_map_figure(city, program_name: str, program_df: pd.DataFrame) -> go.Figure:
    gdf = _city_alignment_geodf(city, program_df)
    geojson = json.loads(gdf.to_json())
    centroid = gdf.geometry.union_all().centroid if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union.centroid
    center = {"lat": float(centroid.y), "lon": float(centroid.x)}
    bounds = gdf.total_bounds
    span = max(float(bounds[2] - bounds[0]), float(bounds[3] - bounds[1]))
    zoom = 10.0 if span <= 0 else float(np.clip(np.log2(360.0 / span) - 1.0, 3.0, 12.0))

    classes = gdf["alignment_class"].fillna("unknown").astype(str)
    unique_classes = sorted(classes.unique().tolist())
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Safe + px.colors.qualitative.Plotly
    class_to_color = {c: palette[i % len(palette)] for i, c in enumerate(unique_classes)}

    score_x, score_y = PROGRAM_SCORE_COLUMNS[program_name]
    fig = go.Figure()
    for cls in unique_classes:
        m = classes == cls
        g_part = gdf.loc[m].copy()
        if g_part.empty:
            continue
        hover_text = (
            "FSA: " + g_part["fsa"].astype(str)
            + "<br>Class: " + cls
            + "<br>" + score_x + ": " + pd.to_numeric(g_part[score_x], errors="coerce").round(3).astype(str)
            + "<br>" + score_y + ": " + pd.to_numeric(g_part[score_y], errors="coerce").round(3).astype(str)
        )
        color = class_to_color[cls]
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=geojson,
                locations=g_part["fsa"].astype(str),
                z=np.ones(len(g_part), dtype=float),
                featureidkey="properties.fsa",
                marker_opacity=0.20,
                marker_line_width=0.5,
                colorscale=[[0.0, color], [1.0, color]],
                showscale=False,
                name=cls,
                legendgroup=cls,
                showlegend=True,
                hovertext=hover_text,
                hoverinfo="text",
            )
        )
    fig.update_layout(
        title=f"{city.name} - {program_name} alignment map",
        mapbox=dict(style="carto-positron", center=center, zoom=zoom),
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        width=980,
        height=620,
        legend=dict(title="Alignment class", orientation="v"),
    )
    return fig


def _alignment_scatter_figure(program_name: str, program_df: pd.DataFrame) -> go.Figure:
    score_x, score_y = PROGRAM_SCORE_COLUMNS[program_name]
    df = program_df.copy()
    df["fsa"] = df.index.astype(str)
    fig = px.scatter(
        df,
        x=score_x,
        y=score_y,
        color="alignment_class",
        hover_name="fsa",
        title=f"{program_name} alignment scatter",
        width=820,
        height=580,
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85))
    fig.add_hline(y=float(df[score_y].quantile(0.5)), line_dash="dash", line_color="#666")
    fig.add_vline(x=float(df[score_x].quantile(0.5)), line_dash="dash", line_color="#666")
    return fig


def _alignment_class_bar(program_name: str, program_df: pd.DataFrame) -> go.Figure:
    vc = program_df["alignment_class"].value_counts().sort_values(ascending=False)
    fig = px.bar(
        x=vc.index.astype(str),
        y=vc.values,
        labels={"x": "alignment_class", "y": "count_fsa"},
        title=f"{program_name} alignment class distribution",
        width=820,
        height=420,
        template="plotly_white",
    )
    return fig


def _subindex_boxplot(program_name: str, program_df: pd.DataFrame) -> go.Figure:
    cols = [c for c in PROGRAM_SUBINDEX_COLUMNS.get(program_name, []) if c in program_df.columns]
    if not cols:
        return go.Figure()
    long_df = (
        program_df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .reset_index()
        .melt(id_vars=program_df.index.name or "index", var_name="sub_index", value_name="score")
    )
    fig = px.box(
        long_df,
        x="sub_index",
        y="score",
        points="outliers",
        title=f"{program_name} sub-index distributions",
        template="plotly_white",
        width=980,
        height=460,
    )
    fig.update_layout(xaxis_title="sub-index", yaxis_title="rank-based score")
    return fig


def _write_fig(fig: go.Figure, path: Path) -> None:
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def _write_table_html(df: pd.DataFrame, path: Path, title: str) -> None:
    html = (
        "<html><head><meta charset='utf-8'><title>"
        + title
        + "</title><style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;}th,td{border:1px solid #ddd;padding:6px 10px;}th{background:#f6f6f6;}</style></head><body>"
        + f"<h2>{title}</h2>"
        + df.to_html(index=True, border=0)
        + "</body></html>"
    )
    path.write_text(html, encoding="utf-8")


def _parameter_eval_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    x = df[available].apply(pd.to_numeric, errors="coerce")
    if x.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "n_non_null": x.notna().sum(),
            "mean": x.mean(),
            "std": x.std(),
            "min": x.min(),
            "p25": x.quantile(0.25),
            "median": x.quantile(0.5),
            "p75": x.quantile(0.75),
            "max": x.max(),
        }
    )
    return out.sort_index()


def _dist_boxplot_figure(df: pd.DataFrame, cols: list[str], title: str, height: int = 420) -> go.Figure:
    available = [c for c in cols if c in df.columns]
    if not available:
        return go.Figure()
    plot_df = df[available].apply(pd.to_numeric, errors="coerce")
    n = len(available)
    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=available,
        horizontal_spacing=0.03,
    )

    for i, col in enumerate(available):
        c = i + 1
        vals = plot_df[col].dropna()
        fig.add_trace(
            go.Box(
                y=vals,
                name=col,
                boxpoints="outliers",
                marker=dict(opacity=0.5, size=3),
                showlegend=False,
            ),
            row=1,
            col=c,
        )
        fig.update_yaxes(title_text=col, row=1, col=c)

    fig.update_layout(
        title=title,
        template="plotly_white",
        width=max(260 * n, 900),
        height=height,
    )
    return fig


def _corr_heatmap_figure(df: pd.DataFrame, cols: list[str], title: str, height: int = 720) -> go.Figure:
    available = [c for c in cols if c in df.columns]
    if len(available) < 2:
        return go.Figure()
    corr = df[available].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=title,
        width=980,
        height=height,
    )
    fig.update_layout(template="plotly_white")
    return fig


def _prism_baseload_heating_scatter(df: pd.DataFrame) -> go.Figure:
    needed = ["baseload_intercept", "heating_slope_per_hdd", "heating_change_point_temp_c"]
    available = [c for c in needed if c in df.columns]
    if len(available) < 3:
        return go.Figure()
    x = pd.to_numeric(df["baseload_intercept"], errors="coerce")
    y = pd.to_numeric(df["heating_slope_per_hdd"], errors="coerce")
    c = pd.to_numeric(df["heating_change_point_temp_c"], errors="coerce")
    work = pd.DataFrame(
        {
            "baseload_intercept": x,
            "heating_slope_per_hdd": y,
            "heating_change_point_temp_c": c,
            "fsa": df.index.astype(str),
        }
    ).dropna()
    if work.empty:
        return go.Figure()

    fig = px.scatter(
        work,
        x="baseload_intercept",
        y="heating_slope_per_hdd",
        color="heating_change_point_temp_c",
        hover_name="fsa",
        title="PRISM diagnostic: baseload vs heating slope (color = heating change-point temp)",
        color_continuous_scale="Turbo",
        template="plotly_white",
        width=980,
        height=600,
    )
    fig.update_traces(marker=dict(size=9, opacity=0.86))
    return fig


def _metric_notes_table() -> pd.DataFrame:
    rows = [
        {
            "metric": "cv_r2 (DML target models)",
            "good_values": "Higher is better. ~0.50+ strong, ~0.20-0.50 moderate.",
            "bad_values": "Near 0 weak predictive signal; negative means worse than mean baseline.",
        },
        {
            "metric": "macro_f1 / balanced_accuracy (distributional targets)",
            "good_values": "Higher is better. ~0.50+ often useful signal for multi-class quantile targets.",
            "bad_values": "Near random-chance levels indicates weak class separation from census features.",
        },
        {
            "metric": "heating_slope_per_hdd",
            "good_values": "Higher can indicate stronger heating-related demand relevance for winter DSM targeting.",
            "bad_values": "Very low/near-zero implies weak heating sensitivity; extreme outliers may indicate fit/noise issues.",
        },
        {
            "metric": "cooling_slope_per_cdd",
            "good_values": "Moderate positive values indicate plausible cooling sensitivity.",
            "bad_values": "Near-zero may be expected in winter-dominant contexts; extreme spikes can indicate unstable fit.",
        },
        {
            "metric": "heating_change_point_temp_c",
            "good_values": "Typically in plausible indoor-comfort balance range (often mid-teens C).",
            "bad_values": "Very low or very high values can indicate weak model identifiability for that FSA.",
        },
        {
            "metric": "baseload_intercept",
            "good_values": "Stable mid-range values relative to peer FSAs indicate consistent non-weather load.",
            "bad_values": "Very high or very low outliers may indicate structural anomalies or data quality issues.",
        },
        {
            "metric": "r2 (PRISM fit)",
            "good_values": "Higher is better; indicates temperature explains more variation.",
            "bad_values": "Low values indicate poor explanatory fit for temperature-driven segmentation.",
        },
        {
            "metric": "cvrmse (PRISM fit)",
            "good_values": "Lower is better; indicates lower normalized error.",
            "bad_values": "High values indicate noisy or poor-fitting PRISM relationships.",
        },
        {
            "metric": "peak_load / p90_top10_mean",
            "good_values": "Higher values indicate stronger peak relevance (useful for peak-oriented DSM).",
            "bad_values": "Very low values imply flatter demand profiles and lower peak leverage.",
        },
        {
            "metric": "am_pm_peak_ratio",
            "good_values": "Near 1 suggests balanced AM/PM peaks; >1 or <1 shows directional peak skew.",
            "bad_values": "Extreme ratios can indicate atypical load-shape behavior or unstable hourly patterns.",
        },
        {
            "metric": "ramp_up_rate",
            "good_values": "Moderate values indicate manageable intraday ramps.",
            "bad_values": "Very high values indicate steep ramps and stronger short-term operational stress.",
        },
        {
            "metric": "winter_peak_share / winter_peak_intensity",
            "good_values": "Higher values indicate stronger system relevance for winter peak interventions.",
            "bad_values": "Lower values indicate lower system-priority from peak-management perspective.",
        },
    ]
    return pd.DataFrame(rows)


def _pick_target_score_column(target_scores: pd.DataFrame) -> tuple[str, str]:
    if "cv_r2" in target_scores.columns:
        return "cv_r2", "DML target model CV R2"
    if "macro_f1" in target_scores.columns:
        return "macro_f1", "Distributional target model Macro-F1"
    if "balanced_accuracy" in target_scores.columns:
        return "balanced_accuracy", "Distributional target model Balanced Accuracy"
    numeric_cols = target_scores.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        return numeric_cols[0], f"Target model score ({numeric_cols[0]})"
    raise ValueError("No plottable numeric score column in target_scores.")


def _timeseries_sample_figure(
    city,
    features: pd.DataFrame | None = None,
    n_fsas: int = 6,
    sample_fsas: list[str] | None = None,
) -> go.Figure:
    picked = sample_fsas or _pick_sample_fsas(city, features=features, n_fsas=n_fsas)
    elec = city.electricity_frame()
    if elec.empty or not picked:
        return go.Figure()

    ts = elec[picked].copy()
    ts.index = pd.to_datetime(ts.index, errors="coerce")
    ts = ts.dropna(how="all").sort_index()

    melt = ts.reset_index().melt(id_vars=ts.index.name or "index", var_name="fsa", value_name="kwh")
    xcol = ts.index.name or "index"
    fig = px.line(
        melt,
        x=xcol,
        y="kwh",
        color="fsa",
        title=f"Sample FSA electricity time series (full range, n={len(picked)})",
        template="plotly_white",
        width=1020,
        height=500,
    )
    fig.update_layout(xaxis_title="date_time", yaxis_title="load (kWh)")
    return fig


def _pretty_program_name(name: str) -> str:
    mapping = {
        "flexd": "Tarif Flex D",
        "hilo": "Hilo",
        "logisvert": "LogisVert",
        "low_income": "Low-/Modest-Income Assistance",
    }
    return mapping.get(str(name), str(name).replace("_", " ").title())


def _pretty_source_name(name: str) -> str:
    mapping = {
        "short_term": "Short-Term",
        "prism": "PRISM",
        "short_term_prob": "Short-Term (Probabilistic)",
        "prism_prob": "PRISM (Probabilistic)",
    }
    return mapping.get(str(name), str(name).replace("_", " ").title())


def _pick_sample_fsas(city, features: pd.DataFrame | None = None, n_fsas: int = 6) -> list[str]:
    elec = city.electricity_frame()
    if elec.empty:
        return []

    candidates = [c for c in city.list_fsa_codes() if c in elec.columns]
    if features is not None and not features.empty:
        feat_idx = set(features.index.astype(str))
        candidates = [c for c in candidates if c in feat_idx]
    if not candidates:
        return []
    n = min(max(1, int(n_fsas)), len(candidates))
    rng = np.random.default_rng()
    return list(rng.choice(np.array(candidates, dtype=object), size=n, replace=False))


def _distribution_predictions_long(pred_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()

    targets = sorted(
        set(str(c).split("__")[0] for c in pred_df.columns if ("__pred_q" in str(c) or "__pred_expected" in str(c)))
    )
    rows = []
    for fsa, row in pred_df.iterrows():
        for t in targets:
            pred_q_col = f"{t}__pred_q"
            pred_p_col = f"{t}__pred_prob"
            pred_e_col = f"{t}__pred_expected"
            pred_l_col = f"{t}__pred_q05"
            pred_u_col = f"{t}__pred_q95"
            if pred_q_col not in pred_df.columns:
                q_val = np.nan
                p_val = np.nan
            else:
                q_val = row.get(pred_q_col, np.nan)
                p_val = row.get(pred_p_col, np.nan)
            e_val = row.get(pred_e_col, np.nan) if pred_e_col in pred_df.columns else np.nan
            l_val = row.get(pred_l_col, np.nan) if pred_l_col in pred_df.columns else np.nan
            u_val = row.get(pred_u_col, np.nan) if pred_u_col in pred_df.columns else np.nan
            if pd.isna(q_val) and pd.isna(e_val):
                continue
            rows.append(
                {
                    "source": source_name,
                    "fsa": str(fsa),
                    "target": t,
                    "pred_quantile_bin": None if pd.isna(q_val) else int(q_val),
                    "pred_class_probability": float(p_val) if pd.notna(p_val) else np.nan,
                    "pred_expected": float(e_val) if pd.notna(e_val) else np.nan,
                    "pred_q05": float(l_val) if pd.notna(l_val) else np.nan,
                    "pred_q95": float(u_val) if pd.notna(u_val) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _distribution_prediction_heatmap(long_df: pd.DataFrame, title: str) -> go.Figure:
    if long_df.empty:
        return go.Figure()
    value_col = "pred_quantile_bin"
    if value_col not in long_df.columns or not long_df[value_col].notna().any():
        return go.Figure()

    piv = long_df.pivot(index="fsa", columns="target", values=value_col)
    if piv.empty:
        return go.Figure()
    if piv.notna().sum().sum() == 0:
        return go.Figure()

    fig = px.imshow(
        piv,
        text_auto=".0f",
        color_continuous_scale="Viridis",
        title=f"{title} ({value_col})",
        width=1180,
        height=max(420, 90 + 60 * len(piv.index)),
    )
    fig.update_layout(template="plotly_white")
    return fig


def _actual_vs_pred_energy_long(
    features: pd.DataFrame,
    pred_df: pd.DataFrame,
    sample_fsas: list[str],
    source_name: str,
) -> pd.DataFrame:
    if features is None or features.empty or pred_df is None or pred_df.empty or not sample_fsas:
        return pd.DataFrame()
    fx = features.copy()
    fx.index = fx.index.astype(str)
    px = pred_df.copy()
    px.index = px.index.astype(str)

    targets = sorted({str(c).split("__")[0] for c in px.columns if "__pred_expected" in str(c)})
    rows = []
    for fsa in sample_fsas:
        if fsa not in fx.index or fsa not in px.index:
            continue
        for t in targets:
            pcol = f"{t}__pred_expected"
            lcol = f"{t}__pred_q05"
            ucol = f"{t}__pred_q95"
            if pcol not in px.columns or t not in fx.columns:
                continue
            actual = pd.to_numeric(pd.Series([fx.at[fsa, t]]), errors="coerce").iloc[0]
            pred = pd.to_numeric(pd.Series([px.at[fsa, pcol]]), errors="coerce").iloc[0]
            q05 = pd.to_numeric(pd.Series([px.at[fsa, lcol]]), errors="coerce").iloc[0] if lcol in px.columns else np.nan
            q95 = pd.to_numeric(pd.Series([px.at[fsa, ucol]]), errors="coerce").iloc[0] if ucol in px.columns else np.nan
            if pd.isna(actual) and pd.isna(pred):
                continue
            rows.append(
                {
                    "source": source_name,
                    "fsa": fsa,
                    "target": t,
                    "actual": float(actual) if pd.notna(actual) else np.nan,
                    "predicted_expected": float(pred) if pd.notna(pred) else np.nan,
                    "pred_q05": float(q05) if pd.notna(q05) else np.nan,
                    "pred_q95": float(q95) if pd.notna(q95) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _actual_vs_pred_energy_plot(long_df: pd.DataFrame, title: str) -> go.Figure:
    if long_df.empty:
        return go.Figure()
    work = long_df.copy()
    targets = sorted(work["target"].astype(str).unique().tolist())
    n = len(targets)
    cols = min(2, max(1, n))
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=targets, horizontal_spacing=0.08, vertical_spacing=0.12)

    rng = np.random.default_rng(42)
    n_draws = 180
    for i, t in enumerate(targets):
        rr = i // cols + 1
        cc = i % cols + 1
        d = work[work["target"].astype(str) == t].copy().sort_values("fsa")
        if d.empty:
            continue
        xs = []
        ys = []
        for _, r in d.iterrows():
            mu = pd.to_numeric(pd.Series([r.get("predicted_expected")]), errors="coerce").iloc[0]
            q05 = pd.to_numeric(pd.Series([r.get("pred_q05")]), errors="coerce").iloc[0]
            q95 = pd.to_numeric(pd.Series([r.get("pred_q95")]), errors="coerce").iloc[0]
            if pd.isna(mu):
                continue
            if pd.notna(q05) and pd.notna(q95) and q95 > q05:
                sd = float((q95 - q05) / (2.0 * 1.645))
            else:
                sd = max(abs(float(mu)) * 0.05, 1e-6)
            draws = rng.normal(loc=float(mu), scale=max(sd, 1e-6), size=n_draws)
            xs.extend([str(r["fsa"])] * len(draws))
            ys.extend(draws.tolist())

        if ys:
            fig.add_trace(
                go.Violin(
                    x=xs,
                    y=ys,
                    name="predicted_dist",
                    box_visible=True,
                    meanline_visible=False,
                    points=False,
                    fillcolor="rgba(99,110,250,0.35)",
                    line=dict(color="rgba(99,110,250,0.7)", width=1),
                    showlegend=(i == 0),
                    legendgroup="predicted_dist",
                ),
                row=rr,
                col=cc,
            )

        fig.add_trace(
            go.Scatter(
                x=d["fsa"].astype(str),
                y=d["actual"],
                mode="markers",
                marker=dict(size=8, color="black"),
                name="actual",
                showlegend=(i == 0),
                legendgroup="actual",
            ),
            row=rr,
            col=cc,
        )
        fig.add_trace(
            go.Scatter(
                x=d["fsa"].astype(str),
                y=d["predicted_expected"],
                mode="markers",
                marker=dict(size=7, color="#d62728", symbol="diamond"),
                name="pred_expected",
                showlegend=(i == 0),
                legendgroup="pred_expected",
            ),
            row=rr,
            col=cc,
        )

    fig.update_layout(title=title, template="plotly_white", width=1180, height=max(620, 320 * rows))
    return fig


def _actual_vs_pred_by_fsa_plot(long_df: pd.DataFrame, title: str) -> go.Figure:
    if long_df.empty:
        return go.Figure()
    work = long_df.copy()
    fsa_list = sorted(work["fsa"].astype(str).unique().tolist())
    n = len(fsa_list)
    cols = min(3, max(1, n))
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=fsa_list, horizontal_spacing=0.07, vertical_spacing=0.12)

    rng = np.random.default_rng(43)
    n_draws = 180
    for i, fsa in enumerate(fsa_list):
        rr = i // cols + 1
        cc = i % cols + 1
        d = work[work["fsa"].astype(str) == fsa].copy().sort_values("target")
        if d.empty:
            continue
        xs = []
        ys = []
        for _, r in d.iterrows():
            mu = pd.to_numeric(pd.Series([r.get("predicted_expected")]), errors="coerce").iloc[0]
            q05 = pd.to_numeric(pd.Series([r.get("pred_q05")]), errors="coerce").iloc[0]
            q95 = pd.to_numeric(pd.Series([r.get("pred_q95")]), errors="coerce").iloc[0]
            if pd.isna(mu):
                continue
            if pd.notna(q05) and pd.notna(q95) and q95 > q05:
                sd = float((q95 - q05) / (2.0 * 1.645))
            else:
                sd = max(abs(float(mu)) * 0.05, 1e-6)
            draws = rng.normal(loc=float(mu), scale=max(sd, 1e-6), size=n_draws)
            xs.extend([str(r["target"])] * len(draws))
            ys.extend(draws.tolist())
        if ys:
            fig.add_trace(
                go.Violin(
                    x=xs,
                    y=ys,
                    name="predicted_dist",
                    box_visible=True,
                    meanline_visible=False,
                    points=False,
                    fillcolor="rgba(99,110,250,0.35)",
                    line=dict(color="rgba(99,110,250,0.7)", width=1),
                    showlegend=(i == 0),
                    legendgroup="predicted_dist",
                ),
                row=rr,
                col=cc,
            )
        fig.add_trace(
            go.Scatter(
                x=d["target"].astype(str),
                y=d["actual"],
                mode="markers",
                marker=dict(size=8, color="black"),
                name="actual",
                showlegend=(i == 0),
                legendgroup="actual",
            ),
            row=rr,
            col=cc,
        )
        fig.add_trace(
            go.Scatter(
                x=d["target"].astype(str),
                y=d["predicted_expected"],
                mode="markers",
                marker=dict(size=7, color="#d62728", symbol="diamond"),
                name="pred_expected",
                showlegend=(i == 0),
                legendgroup="pred_expected",
            ),
            row=rr,
            col=cc,
        )
    fig.update_layout(title=title, template="plotly_white", width=1200, height=max(700, 320 * rows))
    return fig


def _predictive_draws_violin(
    draws_by_target: dict[str, pd.DataFrame],
    sample_fsas: list[str],
    title: str,
) -> go.Figure:
    if not draws_by_target or not sample_fsas:
        return go.Figure()
    rows = []
    for target, draw_df in draws_by_target.items():
        if draw_df is None or draw_df.empty:
            continue
        work = draw_df.copy()
        work.index = work.index.astype(str)
        keep = [f for f in sample_fsas if f in work.index]
        if not keep:
            continue
        sub = work.loc[keep]
        m = sub.reset_index().melt(id_vars=sub.index.name or "index", var_name="draw", value_name="value")
        idcol = sub.index.name or "index"
        m = m.rename(columns={idcol: "fsa"})
        m["target"] = str(target)
        rows.append(m[["fsa", "target", "value"]])
    if not rows:
        return go.Figure()
    long_df = pd.concat(rows, axis=0, ignore_index=True)
    fig = px.violin(
        long_df,
        x="fsa",
        y="value",
        color="fsa",
        facet_col="target",
        facet_col_wrap=2,
        box=True,
        points=False,
        title=title,
        template="plotly_white",
        width=1180,
        height=max(700, 280 + 180 * int(np.ceil(long_df["target"].nunique() / 2.0))),
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def generate_dsm_report(
    city,
    alignment_results: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    features: pd.DataFrame | None = None,
    dml_result=None,
    distribution_predictions: dict[str, pd.DataFrame] | None = None,
    distribution_draws: dict[str, dict[str, pd.DataFrame]] | None = None,
    metadata: dict[str, str] | None = None,
) -> dict[str, Path]:
    out_dir = _ensure_dir(Path(output_dir))
    figs_dir = _ensure_dir(out_dir / "figures")
    tables_dir = _ensure_dir(out_dir / "tables")

    produced: dict[str, Path] = {}
    overview = alignment_overview_table(alignment_results)
    overview_csv = tables_dir / "alignment_overview.csv"
    overview_html = tables_dir / "alignment_overview.html"
    overview.to_csv(overview_csv, index=False)
    _write_table_html(overview.set_index(["program", "alignment_class"]), overview_html, "Alignment Overview")
    produced["overview_csv"] = overview_csv
    produced["overview_html"] = overview_html

    summary_rows = []
    for program_name, program_df in alignment_results.items():
        program_csv = tables_dir / f"{program_name}_alignment_scores.csv"
        program_html = tables_dir / f"{program_name}_alignment_scores.html"
        program_df.to_csv(program_csv, index=True)
        _write_table_html(program_df, program_html, f"{program_name} alignment scores")
        produced[f"{program_name}_table_csv"] = program_csv
        produced[f"{program_name}_table_html"] = program_html

        sub_cols = [c for c in PROGRAM_SUBINDEX_COLUMNS.get(program_name, []) if c in program_df.columns]
        if sub_cols:
            sub_tbl = program_df[sub_cols].copy()
            sub_csv = tables_dir / f"{program_name}_subindexes.csv"
            sub_html = tables_dir / f"{program_name}_subindexes.html"
            sub_tbl.to_csv(sub_csv, index=True)
            _write_table_html(sub_tbl, sub_html, f"{program_name} sub-index scores")
            produced[f"{program_name}_subindexes_csv"] = sub_csv
            produced[f"{program_name}_subindexes_html"] = sub_html

            sub_fig = _subindex_boxplot(program_name, program_df)
            if len(sub_fig.data) > 0:
                sub_fig_path = figs_dir / f"{program_name}_subindexes_box.html"
                _write_fig(sub_fig, sub_fig_path)
                produced[f"{program_name}_subindexes_box"] = sub_fig_path

        map_fig = _alignment_map_figure(city, program_name, program_df)
        map_path = figs_dir / f"{program_name}_map.html"
        _write_fig(map_fig, map_path)
        produced[f"{program_name}_map"] = map_path

        scatter_fig = _alignment_scatter_figure(program_name, program_df)
        scatter_path = figs_dir / f"{program_name}_scatter.html"
        _write_fig(scatter_fig, scatter_path)
        produced[f"{program_name}_scatter"] = scatter_path

        bar_fig = _alignment_class_bar(program_name, program_df)
        bar_path = figs_dir / f"{program_name}_class_bar.html"
        _write_fig(bar_fig, bar_path)
        produced[f"{program_name}_class_bar"] = bar_path

        top_risk = program_df[program_df["alignment_class"].astype(str).str.contains("high_.*low", regex=True, na=False)]
        summary_rows.append(
            {
                "program": program_name,
                "n_fsa": int(len(program_df)),
                "n_high_relevance_low_capacity": int(len(top_risk)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("program")
    summary_csv = tables_dir / "program_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    produced["program_summary_csv"] = summary_csv

    sample_fsas = _pick_sample_fsas(city, features=features, n_fsas=6)
    ts_fig = _timeseries_sample_figure(city, features=features, n_fsas=6, sample_fsas=sample_fsas)
    if len(ts_fig.data) > 0:
        ts_path = figs_dir / "timeseries_sample_fsas.html"
        _write_fig(ts_fig, ts_path)
        produced["timeseries_sample_fsas"] = ts_path

    if distribution_predictions:
        for src, pred_df in distribution_predictions.items():
            if pred_df is None or pred_df.empty:
                continue
            work = pred_df.copy()
            work.index = work.index.astype(str)
            if sample_fsas:
                work = work.reindex(sample_fsas)
            work = work.dropna(axis=0, how="all")
            if work.empty:
                continue

            long_df = _distribution_predictions_long(work, source_name=str(src))
            if long_df.empty:
                continue

            base = f"distribution_predictions_{src}_sample_fsas"
            p_csv = tables_dir / f"{base}.csv"
            p_html = tables_dir / f"{base}.html"
            long_df.to_csv(p_csv, index=False)
            _write_table_html(long_df.set_index(["fsa", "target"]), p_html, f"Distribution predictions ({src}) - sampled FSAs")
            produced[f"{base}_csv"] = p_csv
            produced[f"{base}_html"] = p_html

            hfig = _distribution_prediction_heatmap(long_df, title=f"Predicted class probability ({src}) for sampled FSAs")
            if len(hfig.data) > 0:
                hpath = figs_dir / f"{base}_heatmap.html"
                _write_fig(hfig, hpath)
                produced[f"{base}_heatmap"] = hpath

            if features is not None and not features.empty:
                avp_long = _actual_vs_pred_energy_long(features, work, sample_fsas, source_name=str(src))
                if not avp_long.empty:
                    ap_csv = tables_dir / f"{base}_actual_vs_pred.csv"
                    ap_html = tables_dir / f"{base}_actual_vs_pred.html"
                    avp_long.to_csv(ap_csv, index=False)
                    _write_table_html(
                        avp_long.set_index(["fsa", "target"]),
                        ap_html,
                        f"Actual vs predicted energy parameters ({src}) - sampled FSAs",
                    )
                    produced[f"{base}_actual_vs_pred_csv"] = ap_csv
                    produced[f"{base}_actual_vs_pred_html"] = ap_html

                    ap_fig = _actual_vs_pred_energy_plot(
                        avp_long,
                        title=f"Actual vs predicted expected values ({src}) - sampled FSAs",
                    )
                    if len(ap_fig.data) > 0:
                        ap_fig_path = figs_dir / f"{base}_actual_vs_pred_plot.html"
                        _write_fig(ap_fig, ap_fig_path)
                        produced[f"{base}_actual_vs_pred_plot"] = ap_fig_path

                    ap_fsa_fig = _actual_vs_pred_by_fsa_plot(
                        avp_long,
                        title=f"Per-FSA actual vs predicted expected values ({src})",
                    )
                    if len(ap_fsa_fig.data) > 0:
                        ap_fsa_path = figs_dir / f"{base}_actual_vs_pred_by_fsa_plot.html"
                        _write_fig(ap_fsa_fig, ap_fsa_path)
                        produced[f"{base}_actual_vs_pred_by_fsa_plot"] = ap_fsa_path

    if distribution_draws:
        for src, draws_map in distribution_draws.items():
            if not draws_map:
                continue
            fig = _predictive_draws_violin(
                draws_map,
                sample_fsas=sample_fsas,
                title=f"Predictive value distributions by target ({src}) - sampled FSAs",
            )
            if len(fig.data) > 0:
                p = figs_dir / f"distribution_draws_{src}_sample_fsas_violin.html"
                _write_fig(fig, p)
                produced[f"distribution_draws_{src}_sample_fsas_violin"] = p

    if dml_result is not None:
        target_scores = getattr(dml_result, "target_scores", pd.DataFrame()).copy()
        fi_target = getattr(dml_result, "feature_importance_by_target", pd.DataFrame()).copy()
        fi_global = getattr(dml_result, "global_feature_importance", pd.Series(dtype=float)).copy()

        if not target_scores.empty:
            p = tables_dir / "dml_target_scores.csv"
            target_scores.to_csv(p, index=True)
            produced["dml_target_scores_csv"] = p
            p_html = tables_dir / "dml_target_scores.html"
            _write_table_html(target_scores, p_html, "DML Target Model Scores")
            produced["dml_target_scores_html"] = p_html

            score_col, score_title = _pick_target_score_column(target_scores)
            fig = px.bar(
                target_scores.reset_index(),
                x="target",
                y=score_col,
                title=score_title,
                template="plotly_white",
                width=880,
                height=460,
            )
            fp = figs_dir / "dml_target_cv_r2_bar.html"
            _write_fig(fig, fp)
            produced["dml_target_cv_r2_bar"] = fp

        if not fi_target.empty:
            p = tables_dir / "dml_feature_importance_by_target.csv"
            fi_target.to_csv(p, index=True)
            produced["dml_feature_importance_by_target_csv"] = p

        if len(fi_global) > 0:
            topn = fi_global.sort_values(ascending=False).head(40)
            p = tables_dir / "dml_global_feature_importance_top40.csv"
            topn.to_csv(p, header=["importance"])
            produced["dml_global_feature_importance_top40_csv"] = p
            p_html = tables_dir / "dml_global_feature_importance_top40.html"
            _write_table_html(topn.to_frame("importance"), p_html, "DML Global Census Feature Importance (Top 40)")
            produced["dml_global_feature_importance_top40_html"] = p_html

            fig = px.bar(
                topn.sort_values(ascending=True),
                x=topn.sort_values(ascending=True).values,
                y=topn.sort_values(ascending=True).index,
                orientation="h",
                title="Top 40 census feature importances (DML/XGBoost)",
                template="plotly_white",
                width=980,
                height=1000,
            )
            fig.update_layout(yaxis_title="census_feature", xaxis_title="importance")
            fp = figs_dir / "dml_global_feature_importance_top40_bar.html"
            _write_fig(fig, fp)
            produced["dml_global_feature_importance_top40_bar"] = fp

    if features is not None and not features.empty:
        prism_cols_base = [
            "heating_slope_per_hdd",
            "cooling_slope_per_cdd",
            "heating_change_point_temp_c",
            "cooling_change_point_temp_c",
            "baseload_intercept",
        ]
        if "cooling_change_point_temp_c" not in features.columns and "x2" in features.columns:
            prism_cols_base = [
                "heating_slope_per_hdd",
                "cooling_slope_per_cdd",
                "heating_change_point_temp_c",
                "x2",
                "baseload_intercept",
            ]

        prism_cols = prism_cols_base
        short_cols = [
            "peak_load",
            "p90_top10_mean",
            "am_pm_peak_ratio",
            "ramp_up_rate",
            "winter_peak_share",
            "winter_peak_intensity",
        ]
        short_cols = [c for c in short_cols if c in features.columns]

        for c in [x for x in features.columns if str(x).startswith("dtw_")]:
            if pd.api.types.is_numeric_dtype(pd.to_numeric(features[c], errors="coerce")):
                short_cols.append(c)

        prism_summary = _parameter_eval_summary(features, prism_cols)
        short_summary = _parameter_eval_summary(features, short_cols)

        if not prism_summary.empty:
            p = tables_dir / "prism_parameter_summary.csv"
            prism_summary.to_csv(p, index=True)
            produced["prism_parameter_summary_csv"] = p
            p_html = tables_dir / "prism_parameter_summary.html"
            _write_table_html(prism_summary, p_html, "PRISM Parameter Evaluation Summary")
            produced["prism_parameter_summary_html"] = p_html
            fig = _dist_boxplot_figure(features, prism_cols, "PRISM parameter distributions")
            fp = figs_dir / "prism_parameter_distributions.html"
            _write_fig(fig, fp)
            produced["prism_parameter_distributions"] = fp

        if not short_summary.empty:
            p = tables_dir / "short_term_parameter_summary.csv"
            short_summary.to_csv(p, index=True)
            produced["short_term_parameter_summary_csv"] = p
            p_html = tables_dir / "short_term_parameter_summary.html"
            _write_table_html(short_summary, p_html, "Short-Term Parameter Evaluation Summary")
            produced["short_term_parameter_summary_html"] = p_html
            fig = _dist_boxplot_figure(features, short_cols, "Short-term parameter distributions")
            fp = figs_dir / "short_term_parameter_distributions.html"
            _write_fig(fig, fp)
            produced["short_term_parameter_distributions"] = fp

        fig = _prism_baseload_heating_scatter(features)
        if len(fig.data) > 0:
            fp = figs_dir / "prism_baseload_heating_scatter.html"
            _write_fig(fig, fp)
            produced["prism_baseload_heating_scatter"] = fp

    notes_df = _metric_notes_table()
    notes_csv = tables_dir / "metric_notes_good_bad.csv"
    notes_html = tables_dir / "metric_notes_good_bad.html"
    notes_df.to_csv(notes_csv, index=False)
    _write_table_html(notes_df, notes_html, "Metric Interpretation Notes (Good vs Bad)")
    produced["metric_notes_csv"] = notes_csv
    produced["metric_notes_html"] = notes_html

    index_html = out_dir / "index.html"
    md = metadata or {}
    n_programs = len(alignment_results)
    n_fsas = int(summary_df["n_fsa"].max()) if not summary_df.empty else 0
    n_priority = int(summary_df["n_high_relevance_low_capacity"].sum()) if not summary_df.empty else 0

    blocks = [
        "<html><head><meta charset='utf-8'><title>DSM Alignment Report</title>",
        "<style>",
        ":root{--bg:#f6f8fb;--panel:#ffffff;--ink:#111827;--muted:#4b5563;--line:#d8dee8;--accent:#0f766e;}",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:var(--bg);color:var(--ink);line-height:1.5;}",
        ".wrap{max-width:1320px;margin:0 auto;padding:18px 20px 28px 20px;}",
        ".top{position:sticky;top:0;z-index:5;background:#ffffffd9;backdrop-filter:blur(4px);border-bottom:1px solid var(--line);}",
        ".top .inner{max-width:1320px;margin:0 auto;padding:10px 20px;display:flex;gap:12px;flex-wrap:wrap;align-items:center;}",
        ".top a{color:#0b3b8a;text-decoration:none;font-size:13px;padding:4px 8px;border:1px solid transparent;border-radius:8px;}",
        ".top a:hover{border-color:var(--line);background:#f8fafc;}",
        "h1{margin:8px 0 6px 0;font-size:30px;}",
        "h2{margin:28px 0 8px 0;font-size:22px;}",
        "h3{margin:18px 0 8px 0;font-size:17px;color:#0f172a;}",
        ".muted{color:var(--muted);font-size:12px;}",
        ".panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin:10px 0;}",
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:10px 0 14px 0;}",
        ".card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;}",
        ".card .k{font-size:12px;color:var(--muted);}",
        ".card .v{font-size:22px;font-weight:700;color:var(--accent);}",
        "ul{margin:4px 0 0 0;padding-left:18px;}",
        "li{margin:3px 0;}",
        "iframe{border:1px solid var(--line);border-radius:10px;width:100%;max-width:1240px;background:#fff;}",
        ".split{display:grid;grid-template-columns:1fr 1fr;gap:10px;}",
        "@media(max-width:980px){.split{grid-template-columns:1fr;}}",
        "</style></head><body>",
        "<div class='top'><div class='inner'>",
        "<a href='#summary'>Summary</a>",
        "<a href='#overview'>Overview</a>",
        "<a href='#programs'>Program Results</a>",
        "<a href='#distributional'>Distributional</a>",
        "<a href='#dml'>DML</a>",
        "<a href='#energy_eval'>Energy Parameter Eval</a>",
        "<a href='#metadata'>Run Metadata</a>",
        "</div></div>",
        "<div class='wrap'>",
        f"<h1>DSM Alignment Report - {city.name}</h1>",
        "<div class='muted'>Generated program-level maps/tables and model diagnostics for DSM targeting.</div>",
        "<section id='summary'>",
        "<h2>Executive Summary</h2>",
        "<div class='cards'>",
        f"<div class='card'><div class='k'>Programs</div><div class='v'>{n_programs}</div></div>",
        f"<div class='card'><div class='k'>FSAs in scope</div><div class='v'>{n_fsas}</div></div>",
        f"<div class='card'><div class='k'>High-relevance / low-capacity cases</div><div class='v'>{n_priority}</div></div>",
        f"<div class='card'><div class='k'>Outputs directory</div><div class='v' style='font-size:13px;color:#1f2937;'>{out_dir.name}</div></div>",
        "</div>",
        "<div class='panel'><b>Downloads</b><ul>",
        "<li><a href='tables/alignment_overview.html'>Alignment overview (HTML)</a> | <a href='tables/alignment_overview.csv'>CSV</a></li>",
        "<li><a href='tables/program_summary.csv'>Program summary (CSV)</a></li>",
        "<li><a href='tables/metric_notes_good_bad.html'>Metric interpretation notes</a></li>",
        "</ul></div>",
        "</section>",
        "<section id='overview'>",
        "<h2>Overview</h2>",
    ]
    if "timeseries_sample_fsas" in produced:
        blocks.append("<div class='panel'><h3>Time Series Sample</h3>")
        blocks.append("<iframe src='figures/timeseries_sample_fsas.html' height='560'></iframe>")
        if sample_fsas:
            blocks.append("<div class='muted'><b>Sampled FSAs:</b> " + ", ".join(sample_fsas) + "</div>")
        blocks.append("</div>")
    blocks.append("</section>")

    blocks.append("<section id='programs'><h2>Program Results</h2>")
    for program_name in alignment_results:
        blocks.append(f"<div class='panel'><h3>{_pretty_program_name(program_name)}</h3>")
        blocks.append("<ul>")
        if f"{program_name}_subindexes_html" in produced:
            blocks.append(f"<li><a href='tables/{program_name}_subindexes.html'>Sub-index scores (HTML)</a> | <a href='tables/{program_name}_subindexes.csv'>CSV</a></li>")
        blocks.append(f"<li><a href='tables/{program_name}_alignment_scores.html'>Alignment scores (HTML)</a> | <a href='tables/{program_name}_alignment_scores.csv'>CSV</a></li>")
        blocks.append("</ul>")
        if f"{program_name}_subindexes_box" in produced:
            blocks.append(f"<iframe src='figures/{program_name}_subindexes_box.html' height='500'></iframe>")
        blocks.append("<div class='split'>")
        blocks.append(f"<iframe src='figures/{program_name}_map.html' height='660'></iframe>")
        blocks.append(f"<iframe src='figures/{program_name}_scatter.html' height='660'></iframe>")
        blocks.append("</div>")
        blocks.append(f"<iframe src='figures/{program_name}_class_bar.html' height='440'></iframe>")
        blocks.append("</div>")
    blocks.append("</section>")

    dist_keys = [k for k in produced.keys() if k.startswith("distribution_predictions_")]
    draw_keys = [k for k in produced.keys() if k.startswith("distribution_draws_")]
    blocks.append("<section id='distributional'><h2>Distributional Diagnostics</h2>")
    if dist_keys:
        sources = sorted({k.split("_sample_fsas")[0].replace("distribution_predictions_", "") for k in dist_keys})
        for src in sources:
            src_title = _pretty_source_name(src)
            blocks.append(f"<div class='panel'><h3>{src_title}</h3><ul>")
            p_tbl = produced.get(f"distribution_predictions_{src}_sample_fsas_html")
            p_csv = produced.get(f"distribution_predictions_{src}_sample_fsas_csv")
            p_avp_tbl = produced.get(f"distribution_predictions_{src}_sample_fsas_actual_vs_pred_html")
            p_avp_csv = produced.get(f"distribution_predictions_{src}_sample_fsas_actual_vs_pred_csv")
            if p_tbl is not None:
                blocks.append(f"<li><a href='{p_tbl.relative_to(out_dir).as_posix()}'>Prediction summary (HTML)</a> | <a href='{p_csv.relative_to(out_dir).as_posix()}'>CSV</a></li>")
            if p_avp_tbl is not None:
                blocks.append(f"<li><a href='{p_avp_tbl.relative_to(out_dir).as_posix()}'>Actual vs predicted (HTML)</a> | <a href='{p_avp_csv.relative_to(out_dir).as_posix()}'>CSV</a></li>")
            blocks.append("</ul>")
            p_heat = produced.get(f"distribution_predictions_{src}_sample_fsas_heatmap")
            if p_heat is not None:
                blocks.append(f"<iframe src='{p_heat.relative_to(out_dir).as_posix()}' height='640'></iframe>")
            p_avp = produced.get(f"distribution_predictions_{src}_sample_fsas_actual_vs_pred_plot")
            if p_avp is not None:
                blocks.append(f"<iframe src='{p_avp.relative_to(out_dir).as_posix()}' height='760'></iframe>")
            p_avp_fsa = produced.get(f"distribution_predictions_{src}_sample_fsas_actual_vs_pred_by_fsa_plot")
            if p_avp_fsa is not None:
                blocks.append(f"<iframe src='{p_avp_fsa.relative_to(out_dir).as_posix()}' height='740'></iframe>")
            blocks.append("</div>")
    if draw_keys:
        for k in sorted(draw_keys):
            p = produced[k]
            src = str(k).replace("distribution_draws_", "").replace("_sample_fsas_violin", "")
            blocks.append(f"<div class='panel'><h3>{_pretty_source_name(src)}</h3>")
            blocks.append(f"<iframe src='{p.relative_to(out_dir).as_posix()}' height='880'></iframe></div>")
    blocks.append("</section>")

    if dml_result is not None:
        blocks.append("<section id='dml'><h2>Census Evaluation (DML/XGBoost)</h2><div class='panel'><ul>")
        if "dml_target_scores_html" in produced:
            blocks.append("<li><a href='tables/dml_target_scores.html'>DML target scores (HTML)</a> | <a href='tables/dml_target_scores.csv'>CSV</a></li>")
        if "dml_global_feature_importance_top40_html" in produced:
            blocks.append("<li><a href='tables/dml_global_feature_importance_top40.html'>Top census feature importance (HTML)</a> | <a href='tables/dml_global_feature_importance_top40.csv'>CSV</a></li>")
        if "dml_feature_importance_by_target_csv" in produced:
            blocks.append("<li><a href='tables/dml_feature_importance_by_target.csv'>Feature importance by target (CSV)</a></li>")
        blocks.append("</ul>")
        if "dml_target_cv_r2_bar" in produced:
            blocks.append("<iframe src='figures/dml_target_cv_r2_bar.html' height='500'></iframe>")
        if "dml_global_feature_importance_top40_bar" in produced:
            blocks.append("<iframe src='figures/dml_global_feature_importance_top40_bar.html' height='1040'></iframe>")
        blocks.append("</div></section>")

    if features is not None and not features.empty:
        blocks.append("<section id='energy_eval'><h2>Energy Parameter Evaluation</h2><div class='panel'><ul>")
        if "prism_parameter_summary_html" in produced:
            blocks.append("<li><a href='tables/prism_parameter_summary.html'>PRISM summary (HTML)</a> | <a href='tables/prism_parameter_summary.csv'>CSV</a></li>")
        if "short_term_parameter_summary_html" in produced:
            blocks.append("<li><a href='tables/short_term_parameter_summary.html'>Short-term summary (HTML)</a> | <a href='tables/short_term_parameter_summary.csv'>CSV</a></li>")
        blocks.append("</ul>")
        if "prism_parameter_distributions" in produced:
            blocks.append("<iframe src='figures/prism_parameter_distributions.html' height='560'></iframe>")
        if "short_term_parameter_distributions" in produced:
            blocks.append("<iframe src='figures/short_term_parameter_distributions.html' height='560'></iframe>")
        if "prism_baseload_heating_scatter" in produced:
            blocks.append("<iframe src='figures/prism_baseload_heating_scatter.html' height='640'></iframe>")
        blocks.append("</div></section>")

    blocks.append("<section id='metadata'><h2>Run Metadata</h2>")
    if md:
        blocks.append("<div class='panel'><ul>")
        for k, v in md.items():
            blocks.append(f"<li><b>{k}</b>: {v}</li>")
        blocks.append("</ul></div>")
    blocks.append("</section>")
    blocks.append("</div></body></html>")
    index_html.write_text("\n".join(blocks), encoding="utf-8")
    produced["index_html"] = index_html
    return produced
