# NeurIPS Bundle Consistency

## Source-bundle checks

- `neurips_latex/final.tex` contains no body-level `\input{}` dependencies: `yes`
- Figures referenced by `neurips_latex/final.tex`:
  - `neurips_latex/figures/framework_overview.pdf`
  - `neurips_latex/figures/main_results_panel.pdf`
  - `neurips_latex/figures/sensortamper_monitorability.pdf`
- All referenced figures exist: `yes`
- Table sources preserved in the package:
  - `neurips_latex/tables/environment_summary_compact.tex`
  - `neurips_latex/tables/main_results_compact.tex`
  - `neurips_latex/tables/sensortamper_monitor.tex`
- Citation keys used in `neurips_latex/final.tex` all resolve against
  `neurips_latex/references.bib`: `yes`
- Official style file exists locally:
  - `neurips_latex/neurips_2026.sty`

## Verified rebuild path

- `neurips_latex/final.pdf` was regenerated from `neurips_latex/final.tex`
  after a clean removal of the prior auxiliary outputs.
- Verified rebuild command:
  - `cd neurips_latex && TECTONIC_CACHE_DIR=$PWD/.tectonic-cache tectonic -C --keep-logs --keep-intermediates final.tex > final_build.log 2>&1`
- That rebuild succeeded using only the repo-local `.tectonic-cache/` directory.

## Content correspondence

- `neurips_latex/final.pdf` corresponds to the current checked-in
  `neurips_latex/final.tex`: `yes`
- Current SHA-256:
  - `final.pdf`: `daa66481e171a134cca6c90547d8f37127b6c1f2c6310212887f1be8b8ca6a14`
  - `final.tex`: `eb8c0fec3f5d902046a558570767c1dbe3846889855a1b6e67a0f69927531cd7`

## Hidden dependency assessment

- No hidden paper asset dependency outside the repo remains.
- No hidden TeX resource cache outside the repo is required for the verified
  rebuild path because the workshop package includes a repo-local
  `.tectonic-cache/`.
- A TeX engine binary is still required. The verified engine on this machine
  was `tectonic`.

## Bottom line

- The NeurIPS workshop package is self-consistent: `yes`
- A clean user with this repository plus a `tectonic` binary can rebuild
  `neurips_latex/final.pdf` from the repository contents alone: `yes`
