from __future__ import annotations

from typing import Any

from dsm_alignment import (
    city_fsa_feature_table,
    city_fsa_feature_table_from_cache,
    generate_dsm_report,
    run_all_program_alignments,
    run_distributional_weighted_alignment,
    run_dml_weighted_alignment,
)


def run_city_dsm_workflow(
    city,
    *,
    mode: str = "default_weights",
    use_cached_tables: bool = True,
    feature_kwargs: dict[str, Any] | None = None,
    dml_kwargs: dict[str, Any] | None = None,
    distributional_kwargs: dict[str, Any] | None = None,
    report_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feature_kwargs = feature_kwargs or {}
    report_kwargs = report_kwargs or {}

    if use_cached_tables:
        features = city_fsa_feature_table_from_cache(city, **feature_kwargs).copy()
    else:
        features = city_fsa_feature_table(city, **feature_kwargs).copy()
    features.index = features.index.astype(str)
    features.index.name = "fsa"

    if mode == "dml_weighted":
        out = run_dml_weighted_alignment(features, **(dml_kwargs or {}))
        alignment_results = out["alignment_results"]
        model_result = out.get("dml")
    elif mode == "distributional":
        out = run_distributional_weighted_alignment(features, **(distributional_kwargs or {}))
        alignment_results = out["alignment_results"]
        model_result = out.get("distributional")
    elif mode == "default_weights":
        alignment_results = run_all_program_alignments(features)
        out = {"alignment_results": alignment_results}
        model_result = None
    else:
        raise ValueError("mode must be one of: default_weights, dml_weighted, distributional")

    report_output_dir = report_kwargs.pop("output_dir", None)
    produced = None
    if report_output_dir is not None:
        produced = generate_dsm_report(
            city=city,
            alignment_results=alignment_results,
            output_dir=report_output_dir,
            features=features,
            dml_result=model_result,
            metadata={"city": getattr(city, "name", "city"), "mode": mode, **report_kwargs.pop("metadata", {})},
            **report_kwargs,
        )

    return {
        "features": features,
        "alignment_results": alignment_results,
        "model_result": model_result,
        "workflow_result": out,
        "report_outputs": produced,
    }
