# Final Asset Repair Audit

## `latex/final.tex` runtime dependencies

### `\includegraphics{}` targets

| Target in `final.tex` | Resolved repo path | Exists in workspace | Tracked in git before this pass | Status after this pass |
| --- | --- | --- | --- | --- |
| `figures/framework_overview.pdf` | `latex/figures/framework_overview.pdf` | yes | no | repaired |
| `figures/main_results_panel.pdf` | `latex/figures/main_results_panel.pdf` | yes | no | repaired |
| `figures/sensortamper_monitorability.pdf` | `latex/figures/sensortamper_monitorability.pdf` | yes | no | repaired |

### Table includes

- `latex/final.tex` contains no `\input{}` table includes.
- The three main-paper tables are embedded directly into `latex/final.tex`.
- Therefore the table fragment sources in `latex/tables/` are generation inputs for `final.tex`, not runtime dependencies of rebuilding `final.pdf` from `final.tex`.

### Bibliography

| Dependency | Resolved repo path | Exists | Notes |
| --- | --- | --- | --- |
| `\bibliography{references_seed}` | `latex/references_seed.bib` | yes | bib source used by the final paper |
| bibliography style | `latex/aaai2026.bst` | yes | used via the AAAI style package path |

### Style / class path

| Dependency | Resolved repo path | Exists | Notes |
| --- | --- | --- | --- |
| `\usepackage[submission]{aaai2026}` | `latex/aaai2026.sty` | yes | official AAAI-style package vendored in repo |

## Files that were missing from the source bundle contract

- No figure referenced by `latex/final.tex` was missing from the workspace.
- The real bundle defect was that the three final-paper figure PDFs were present locally but were not tracked in Git because `latex/figures/*` was ignored broadly.
- This pass repairs that by explicitly promoting those three figure files into the tracked source bundle.

## Hidden dependencies outside the repo

- A local TeX engine is still required to rebuild `latex/final.pdf`. On this machine the verified build path is `tectonic`.
- The clean rebuild verification in this pass used `tectonic -C`, which means it used only cached local TeX resource files and did not fetch network resources during the rebuild.
- Standard LaTeX package resources are therefore still an external toolchain dependency, but no hidden paper asset dependency remains outside the repo.

## Bottom line

- All paper-facing figure assets referenced by `latex/final.tex` exist locally.
- The only repair needed was to make those three figure PDFs part of the tracked repo contents.
