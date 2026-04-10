# Monitor Scope Analysis Trial

## What changed

- Added a new cross-environment monitor-scope aggregation in:
  - `src/analysis/aggregate.py`
  - `src/analysis/tables.py`
  - `scripts/make_tables.py`
- Generated a compact reviewer-facing table summarizing proxy-only monitor label
  support and accepted hack-monitor status across environments.

## What was run

- `MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python scripts/make_tables.py --results-root results/second_pass --monitor-results-root results/second_pass/monitor_eval --latex-root latex`
- `MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python scripts/make_tables.py --results-root results/second_pass --monitor-results-root results/second_pass/monitor_eval --latex-root neurips_latex`

## What evidence was generated

- Cross-environment summary CSV:
  - `results/second_pass/aggregated_monitor_scope_results.csv`
- Main-paper table source:
  - `latex/tables/monitor_scope.tex`
- NeurIPS package table source:
  - `neurips_latex/tables/monitor_scope.tex`

Key evidence:
- `SensorTamper-v0`: `58/72` positive hack labels, accepted hack-monitor task in `restricted + rich`
- `ProxyFarming-v0`: `72/72` positive hack labels, degenerate hack-monitor task
- `GemTimeout-v0`: `71/72` positive hack labels, degenerate hack-monitor task
- Failure labels collapse to `72/72` positives in all three environments

## Whether it helped

Yes.

This converts the weakest part of the paper from a soft qualification into an
artifact-backed methodological result. The paper can now explain exactly why
valid monitor evidence is concentrated in `SensorTamper-v0` instead of merely
stating that it is.

## Whether it should be included in the paper

Yes.

The compact table belongs in the main NeurIPS paper because it directly
supports the narrow-monitorability claim and makes the negative monitor results
scientifically legible.

## Whether it changes the core story

It sharpens the core story without changing it.

The paper remains a framework / benchmark and evaluation-methodology paper; the
new analysis simply makes the monitor limitation more informative and more
reviewer-readable.
