# Final Bundle Consistency

- Is the source bundle now self-consistent: `yes`.
- Does `latex/final.tex` reference only real repo files: `yes`.
- Do all three figure files referenced by `latex/final.tex` now exist inside `latex/figures/`: `yes`.
- Do all citation keys used in `latex/final.tex` resolve against `latex/references_seed.bib`: `yes`.
- Does the checked-in `latex/final.pdf` come from a clean rebuild of `latex/final.tex`: `yes`, from the rebuild verified in this pass.

## Rebuildability statement

- A clean rebuild from this repository was verified on this machine after deleting the stale auxiliary outputs.
- The verified command was:
  - `cd latex && tectonic -C --keep-logs --keep-intermediates final.tex > final_rebuild.log 2>&1`
- That command succeeded using only cached local TeX resources.

## Important qualification

- No hidden paper asset dependency remains outside the repo.
- A TeX toolchain is still required. In this environment that toolchain was `tectonic`, plus its already-cached standard LaTeX resources.
- So the paper bundle is self-consistent as a source bundle, but a completely bare machine with no TeX engine would still need a TeX installation before rebuilding `final.pdf`.

## What still needs human attention

- Review the remaining typography warnings in `latex/final_rebuild.log`.
- If exact byte-for-byte PDF determinism matters, note that the newly rebuilt `latex/final.pdf` differs from the previously checked-in PDF hash; this pass repaired that by replacing the checked-in PDF with the clean-rebuilt output.
