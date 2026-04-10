# NeurIPS Spec Package Status

Date: `2026-04-10`

## Package mode

- Workshop mode used: `dblblindworkshop`
- Named official workshop known: `no`
- Package state: `generic-workshop-ready`
- Single-file `final.tex` status: `yes`
  - `neurips_latex/final.tex` contains no body-level `\input{}` dependencies

## Build verification

- `neurips_latex/final.tex` builds to `neurips_latex/final.pdf`: `yes`
- Verified build command:
  - `cd neurips_latex && TECTONIC_CACHE_DIR=$PWD/.tectonic-cache tectonic --keep-logs --keep-intermediates final.tex`
- Verified page count from `neurips_latex/final.log`:
  - `7` pages
- Current SHA-256:
  - `final.pdf`: `b6ad23cec400f416774d7339e684ddaa19e4a622607fa6b8b19aaf17f8199feb`
  - `final.tex`: `b26da198da82db31b2710f5fa0fb891eaf4cbf38472546702c2b369969862125`

## Check status

- Paper/package compile success: `yes`
- Warning-free clean compile: `no`
- Test suite status:
  - `./.venv/bin/pytest -q`
  - `21 passed in 22.11s`

## Remaining packaging issues

1. Tectonic still emits a repeated non-fatal internal consistency warning about
   checking whether `final.bbl` changed.
2. `lineno.sty` emits a non-fatal UTF-8 replacement warning in this Tectonic
   path.
3. Minor underfull/overfull box warnings remain in the generated PDF.

## Bottom line

- The package is truthfully `generic-workshop-ready`.
- The updated single-file source and PDF both exist and build successfully.
- No named official NeurIPS 2026 workshop was found, so no workshop-specific
  retargeting was applied.
