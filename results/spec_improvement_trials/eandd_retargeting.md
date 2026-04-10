# Anonymous ED-Track Retargeting

## What changed

- Retargeted the NeurIPS package from generic anonymous workshop mode
  (`dblblindworkshop`) to the official anonymous NeurIPS 2026 Evaluations
  \& Datasets track mode (`eandd`).
- Updated the title to emphasize both benchmark design and evaluation
  protocol.
- Removed workshop-specific phrasing from the main paper narrative.
- Clarified the monitor-scope table so failure-task degeneracy is explicit.

## What was run

- Reviewed the public NeurIPS 2026 conference call and ED FAQ.
- Rebuilt `neurips_latex/final.tex` and `neurips_latex/final.pdf` with
  `tectonic`.

## What evidence was generated

- Official NeurIPS sources:
  - `https://neurips.cc/Conferences/2026/CallForPapers`
  - `https://neurips.cc/Conferences/2026/EvaluationsDatasetsFAQ`
- Updated package artifacts:
  - `neurips_latex/main.tex`
  - `neurips_latex/final.tex`
  - `neurips_latex/final.pdf`
- Updated status files:
  - `results/NEURIPS_SPEC_WORKSHOP_TARGET_RECHECK.md`
  - `results/NEURIPS_SPEC_PACKAGE_STATUS.md`

Key support from the official FAQ:
- ED papers appear in the same NeurIPS proceedings.
- ED uses the same LaTeX template.
- Authors should choose ED when the main contribution is an evaluation
  framework or empirical evaluation insight.

## Whether it helped

Yes.

This removes a real packaging weakness. The paper no longer presents itself as
an unnamed generic workshop submission with a placeholder workshop title even
though an official named NeurIPS track now exists whose review frame matches
the contribution more closely.

## Whether it should be included in the paper

Yes.

This should remain in the package because it improves target-fit honesty and
submission professionalism without changing the science.

## Whether it changes the core story

No.

The paper remains the same benchmark and evaluation-methodology contribution.
The change is in review-target alignment, not in the scientific claim.
