# Citation Audit

## Scope

- Audited all LaTeX `\cite*{...}` usages under `latex/`.
- Audited `latex/references_seed.bib` against the active cite graph.

## Outcome

- Undefined citation keys fixed: `0`
- Remaining undefined citation keys: `0`
- Unused bibliography entries after pruning: `0`

## Active cite keys

- `amodei2016concrete`
- `armstrong2020pitfalls`
- `cobbe2020leveraging`
- `everitt2021reward`
- `leike2017ai`
- `schulman2017ppo`
- `skalse2022rewardhacking`

## Changes made

- Removed undefined or stale citation references from the LaTeX sources.
- Pruned unused bibliography entries that were no longer cited in the tightened paper draft.
- Kept only verifiable seed references that match the current paper framing.
- Standardized the surviving entries toward cleaner venue and identifier metadata where known.

## Remaining caveat

- No local TeX toolchain is installed in this environment:
  - `latexmk`: not found
  - `pdflatex`: not found
- The cite graph and bibliography file are internally consistent, but a full venue-style bibliography compile still needs to be checked on a machine with a LaTeX toolchain.
