# NeurIPS Spec Package Status

Date: `2026-04-10`

## Package mode

- Package mode used: `eandd`
- Named official NeurIPS track known: `yes`
- Package state: `specific-track-ready`
- Single-file `final.tex` status: `yes`
  - `neurips_latex/final.tex` contains no body-level `\input{}` dependencies

## Build verification

- `neurips_latex/final.tex` builds to `neurips_latex/final.pdf`: `yes`
- Verified build command:
  - `cd neurips_latex && TECTONIC_CACHE_DIR=$PWD/.tectonic-cache tectonic --keep-logs --keep-intermediates final.tex`
- Verified page count from `neurips_latex/final.log`:
  - `7` pages
- Current SHA-256:
  - `final.pdf`: `ae51c70cc9bac07af58a6c2d5b941c8fccdb42eb83ceea1a02857c8de8e117e8`
  - `final.tex`: `6df2aa16db92e27aa5a86bbbcdb91acdf0da13dacf3ddcdfc81a4354abd11522`

## Check status

- Paper/package compile success: `yes`
- Warning-free clean compile: `no`
- Test suite status:
  - `./.venv/bin/pytest -q`
  - `22 passed in 25.05s`

## Remaining packaging issues

1. Tectonic still emits a repeated non-fatal internal consistency warning about
   checking whether `final.bbl` changed.
2. `lineno.sty` emits a non-fatal UTF-8 replacement warning in this Tectonic
   path.

## Bottom line

- The package is truthfully `specific-track-ready` for the anonymous NeurIPS
  2026 Evaluations \& Datasets track.
- The updated single-file source and PDF both exist and build successfully.
- The rebuilt package now includes the accepted SensorTamper tamper-off
  ablation-backed text.
- The package has also been retargeted away from a generic unnamed workshop and
  into the official anonymous `eandd` mode, which better matches the paper's
  benchmark-and-evaluation contribution.
- Avoidable table-layout box warnings were removed in the final rebuild; the
  remaining warnings are style-path or Tectonic-path issues rather than paper
  content issues.
