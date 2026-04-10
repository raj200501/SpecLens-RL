# Figure and Table Consolidation Trial

## What changed

- Replaced the detailed SensorTamper monitor table in the main NeurIPS paper
  flow with the new cross-environment monitor-scope table.
- Kept the strongest main-paper artifact set centered on:
  - framework overview figure
  - cross-environment main results figure
  - SensorTamper monitorability figure
  - compact environment table
  - compact main results table
  - compact monitor-scope table
- Retained the more detailed SensorTamper monitor table as a support artifact
  in the repository rather than a main-paper dependency.

## What was run

- `MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python scripts/make_tables.py --results-root results/second_pass --monitor-results-root results/second_pass/monitor_eval --latex-root neurips_latex`
- Single-file source regeneration for:
  - `neurips_latex/final.tex`
- Verified package build:
  - `TECTONIC_CACHE_DIR=$PWD/.tectonic-cache tectonic --keep-logs --keep-intermediates final.tex`

## What evidence was generated

- Main-paper monitor-scope table:
  - `neurips_latex/tables/monitor_scope.tex`
- Updated single-file paper showing the final artifact order:
  - `neurips_latex/final.tex`
- Updated compiled PDF:
  - `neurips_latex/final.pdf`

## Whether it helped

Yes.

The final paper reads more like a scientific argument and less like an export
of every available artifact. The monitor section now uses the figure for the
accepted SensorTamper result and the table for the cross-environment scope
diagnosis, which is a cleaner division of labor.

## Whether it should be included in the paper

Yes.

This consolidation materially improves workshop readability.

## Whether it changes the core story

It sharpens the presentation of the core story without changing the underlying
result.
