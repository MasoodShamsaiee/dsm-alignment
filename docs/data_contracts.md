# Data Contracts

## Purpose

`dsm-alignment` consumes feature tables derived from urban energy analytics and emits program-specific alignment tables. This document defines the minimum expectations for those inputs and outputs.

## Feature table input

The package expects an FSA-indexed table where:

- index: FSA code or FSA-like identifier
- rows: one row per FSA
- columns: engineered feature columns plus optional census proxies

Typical required feature families:

- PRISM features such as `heating_slope_per_hdd`, `cooling_slope_per_cdd`, `heating_change_point_temp_c`, `baseload_intercept`
- short-term features such as `peak_load`, `p90_top10_mean`, `am_pm_peak_ratio`, `ramp_up_rate`
- census-style socio-demographic columns used as proxy inputs

## Alignment output

Program evaluators return one table per program with:

- index aligned to the input feature table
- sub-index columns specific to the program
- a final categorical column `alignment_class`

## Weighted workflows

DML and distributional workflows return dictionaries containing:

- `alignment_results`
- model metadata/result objects
- feature importance summaries

These keys should remain stable unless a versioned breaking change is announced.
