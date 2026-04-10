# Paper Narrative Rewrite Trial

## What changed

- Retitled the NeurIPS paper as a `benchmark` rather than a generic
  `framework`.
- Rewrote the abstract to foreground the strongest supported benchmark claim.
- Added an explicit contributions paragraph in the introduction.
- Tightened the framework, experiments, results, limitations, and conclusion
  sections so the paper reads as a diagnosis / evaluation paper rather than a
  method paper.
- Added a claim-to-artifact map:
  - `results/SPEC_VERIFY_BEFORE_WRITE.md`

## What was run

- Manual paper edits in:
  - `neurips_latex/main.tex`
  - `neurips_latex/sections/intro.tex`
  - `neurips_latex/sections/framework.tex`
  - `neurips_latex/sections/experiments.tex`
  - `neurips_latex/sections/results.tex`
  - `neurips_latex/sections/limitations.tex`
  - `neurips_latex/sections/conclusion.tex`
- Single-file source regeneration for:
  - `neurips_latex/final.tex`
- Verified package build:
  - `TECTONIC_CACHE_DIR=$PWD/.tectonic-cache tectonic --keep-logs --keep-intermediates final.tex`

## What evidence was generated

- Claim map:
  - `results/SPEC_VERIFY_BEFORE_WRITE.md`
- Updated single-file submission source:
  - `neurips_latex/final.tex`
- Updated compiled PDF:
  - `neurips_latex/final.pdf`

## Whether it helped

Yes.

The paper now states its real identity directly: compact benchmark first,
evaluation methodology second. That reduces the risk that reviewers misread the
submission as a small method paper with weak method breadth.

## Whether it should be included in the paper

Yes.

This rewrite is core to the final submission package.

## Whether it changes the core story

It sharpens the core story without changing the science.
