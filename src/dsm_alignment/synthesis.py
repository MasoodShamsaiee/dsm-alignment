from __future__ import annotations

import pandas as pd

from dsm_alignment.flexd import evaluate_flexd_alignment
from dsm_alignment.hilo import evaluate_hilo_alignment
from dsm_alignment.logisvert import evaluate_logisvert_alignment
from dsm_alignment.low_income import evaluate_low_income_alignment


def run_all_program_alignments(
    features: pd.DataFrame,
    *,
    column_maps: dict[str, dict[str, str]] | None = None,
    weight_overrides: dict[str, dict[str, dict[str, float]]] | None = None,
    quantile: float = 0.5,
) -> dict[str, pd.DataFrame]:
    column_maps = column_maps or {}
    weight_overrides = weight_overrides or {}
    return {
        "flexd": evaluate_flexd_alignment(
            features,
            column_map=column_maps.get("flexd"),
            weight_overrides=weight_overrides.get("flexd"),
            quantile=quantile,
        ),
        "hilo": evaluate_hilo_alignment(
            features,
            column_map=column_maps.get("hilo"),
            weight_overrides=weight_overrides.get("hilo"),
            quantile=quantile,
        ),
        "logisvert": evaluate_logisvert_alignment(
            features,
            column_map=column_maps.get("logisvert"),
            weight_overrides=weight_overrides.get("logisvert"),
            quantile=quantile,
        ),
        "low_income": evaluate_low_income_alignment(
            features,
            column_map=column_maps.get("low_income"),
            weight_overrides=weight_overrides.get("low_income"),
            quantile=quantile,
        ),
    }


def alignment_overview_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for program, df in results.items():
        if "alignment_class" not in df.columns:
            continue
        vc = df["alignment_class"].value_counts(dropna=False)
        total = max(int(vc.sum()), 1)
        for cls, n in vc.items():
            rows.append(
                {
                    "program": program,
                    "alignment_class": cls,
                    "count_fsa": int(n),
                    "share_fsa": float(n) / float(total),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["program", "alignment_class", "count_fsa", "share_fsa"])
    return pd.DataFrame(rows).sort_values(["program", "count_fsa"], ascending=[True, False]).reset_index(drop=True)
