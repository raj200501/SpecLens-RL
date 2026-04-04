# Final Asset Mapping Repair

This pass did not regenerate experiments or swap in different scientific plots.

## Final-paper figure mapping

- `latex/final.tex` references `figures/framework_overview.pdf`
  - resolved to `latex/figures/framework_overview.pdf`
  - this is the intended final conceptual figure
  - action: kept the filename and made the file part of the tracked source bundle

- `latex/final.tex` references `figures/main_results_panel.pdf`
  - resolved to `latex/figures/main_results_panel.pdf`
  - this is the intended final main-results panel
  - action: kept the filename and made the file part of the tracked source bundle

- `latex/final.tex` references `figures/sensortamper_monitorability.pdf`
  - resolved to `latex/figures/sensortamper_monitorability.pdf`
  - this is the intended final monitorability figure
  - action: kept the filename and made the file part of the tracked source bundle

## Table mapping

- No remapping was required for tables in `latex/final.tex`.
- The final paper embeds its three main tables directly, so no runtime table include was broken.

## Repair summary

- No figure filenames were changed.
- No final-paper figure content was regenerated or substituted.
- The repair was purely a source-bundle fix: the exact final-paper figure assets now live inside the tracked `latex/figures/` bundle.
