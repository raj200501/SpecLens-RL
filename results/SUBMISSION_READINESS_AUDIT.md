# Submission Readiness Audit

## Main paper `\input{}` references

| Reference | Exists | Resolved path |
| --- | --- | --- |
| `sections/intro` | `yes` | `latex/sections/intro.tex` |
| `sections/related_work` | `yes` | `latex/sections/related_work.tex` |
| `sections/framework` | `yes` | `latex/sections/framework.tex` |
| `sections/metrics` | `yes` | `latex/sections/metrics.tex` |
| `sections/method_overview` | `yes` | `latex/sections/method_overview.tex` |
| `sections/experiments` | `yes` | `latex/sections/experiments.tex` |
| `sections/results` | `yes` | `latex/sections/results.tex` |
| `sections/limitations` | `yes` | `latex/sections/limitations.tex` |
| `sections/broader_impact` | `yes` | `latex/sections/broader_impact.tex` |
| `sections/conclusion` | `yes` | `latex/sections/conclusion.tex` |

## Section-level table fragment references

| Section | Reference | Exists | Resolved path |
| --- | --- | --- | --- |
| `experiments` | `tables/environment_summary_compact` | `yes` | `latex/tables/environment_summary_compact.tex` |
| `results` | `tables/main_results_compact` | `yes` | `latex/tables/main_results_compact.tex` |
| `results` | `tables/sensortamper_monitor` | `yes` | `latex/tables/sensortamper_monitor.tex` |

## `\includegraphics{}` references used by the modular paper

| Graphic | Exists |
| --- | --- |
| `figures/framework_overview.pdf` | `yes` |
| `figures/main_results_panel.pdf` | `yes` |
| `figures/sensortamper_monitorability.pdf` | `yes` |

## Bibliography consistency

- Cited keys in modular + final paper sources: `['amodei2016concrete', 'armstrong2020pitfalls', 'cobbe2020leveraging', 'everitt2021reward', 'leike2017ai', 'schulman2017ppo', 'skalse2022rewardhacking']`
- Undefined citation keys: `[]`

## Final single-file paper contract

- Remaining `\input{}` dependencies in `latex/final.tex`: `[]`
- `\includegraphics{}` dependencies in `latex/final.tex`: `['figures/framework_overview.pdf', 'figures/main_results_panel.pdf', 'figures/sensortamper_monitorability.pdf']`
- `latex/final.tex` embeds all body text and tables directly; external dependencies are limited to figure PDFs and `references_seed.bib`.

## Compile status

- `aaai2026.sty` present locally: `yes`
- `aaai2026.bst` present locally: `yes`
- `tectonic` present locally: `yes`
- Conclusion: the modular paper is internally file-consistent and the repository now contains the AAAI-26 package files needed for an official local compile path.

## Scaffold-tone status

- The live paper now inputs `sections/experiments.tex` and `sections/results.tex` instead of placeholder-named sections.
- Compatibility files `experiments_placeholders.tex` and `results_placeholders.tex` still exist for repository/test compatibility, but they are no longer referenced by `latex/main.tex`.
- Auxiliary generated sentence files remain in `latex/sections/` but are not required for the visible paper text.

## Balance and page-discipline estimate

- Approximate abstract + body word count: `3401`
- Section word counts:
  - `intro`: `503` words
  - `related_work`: `377` words
  - `framework`: `624` words
  - `metrics`: `339` words
  - `method_overview`: `274` words
  - `experiments`: `163` words
  - `results`: `458` words
  - `limitations`: `216` words
  - `broader_impact`: `154` words
  - `conclusion`: `170` words

- The paper looks realistically main-paper shaped by content density: framework + metrics remain substantial, related work is concise, and results is proportionate.
- Exact AAAI page count is determined separately in `results/FINAL_PDF_BUILD_STATUS.md` from the local tectonic build.

## Missing or mismatched files

- No missing body `\input{}` or `\includegraphics{}` targets were found after the final assembly updates.
- No file-contract mismatches remain in the final paper source tree.
- The remaining question is page discipline under the official compile, not missing assets.