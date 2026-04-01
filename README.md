# dsm-alignment

`dsm-alignment` is the extracted DSM scoring and reporting package from the larger research codebase. It contains reusable feature-table assembly, program alignment logic, weighting/calibration workflows, probabilistic and distributional modeling helpers, and report generation utilities.

## What is included

- FSA-level DSM feature table assembly from `urban-energy-core` city objects
- program alignment scoring for Flex D, Hilo, LogisVert, and low-income targeting
- DML-style importance estimation and weight calibration
- distributional target modeling and weighted alignment
- probabilistic target prediction helpers
- HTML reporting utilities and workflow entry points

## Dependency

This package depends on `urban-energy-core` for the `City` object and precomputed energy analytics tables.

## Data contract

This package expects an FSA-level feature table with engineered energy features plus census-style socio-demographic proxies. See [docs/data_contracts.md](docs/data_contracts.md) for the minimum expected columns and output shape conventions.

## Package layout

```text
src/dsm_alignment/
  common.py
  features.py
  flexd.py
  hilo.py
  logisvert.py
  low_income.py
  dml.py
  distributional.py
  probabilistic.py
  reporting.py
  synthesis.py
  workflows.py
```

## Quick start

```powershell
conda run -n dsm_qc python -m pip install -e ..\\urban-energy-core
conda run -n dsm_qc python -m pip install -e .[dev]
conda run -n dsm_qc python -c "import dsm_alignment; print('ok')"
```

## Next cleanup items

- add higher-value behavioral tests for the alignment functions
- decide whether report templates/assets should remain here or move to a docs/examples layer
