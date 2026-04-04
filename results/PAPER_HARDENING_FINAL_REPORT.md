# Paper Hardening Final Report

## What changed

- Reframed the paper around a tighter main-paper artifact set: one framework figure, one compact main-results figure, one SensorTamper monitor figure, and three main tables.
- Rewrote the experimental setup and results sections so they read as a scientific argument rather than an artifact log.
- Tightened the abstract, the closing paragraph of the introduction, and the empirical conclusion to match the actual evidence.
- Cleaned the bibliography and citation graph so the cited works and bib entries now match exactly.
- Added stronger publication-style plot and table generators while preserving the older repository artifacts.
- Hardened the paper-export code so partial local result bundles fall back cleanly instead of crashing the analysis smoke test.

## What improved scientifically

- The final narrative now centers the strongest honest claim the evidence supports:
  - `SpecLens-RL` measures specification gaps across all three lightweight environments.
  - `SensorTamper-v0` provides explicit corruption-channel measurement.
  - Leakage-controlled monitor evidence exists, but it is narrow and concentrated in `SensorTamper-v0`.
- The results section now treats the negative monitor findings in `ProxyFarming-v0` and `GemTimeout-v0` as part of the framework argument rather than as missing evidence to gloss over.
- The paper no longer implies broad monitorability or uniformly worsening OOD behavior.

## What improved cosmetically and presentation-wise

- Figures use cleaner labels, a more publication-friendly layout, and a tighter main-paper footprint.
- Tables were rewritten away from raw CSV aesthetics toward compact, reader-facing summaries.
- Visible method and channel names were normalized for presentation:
  - `Proxy PPO`
  - `Oracle True PPO`
  - `Restricted monitor`
  - `Rich monitor`
- Placeholder-style wording was reduced in the empirical sections where real outputs already exist.

## Cheap extra ablation

- No additional empirical ablation was run in this pass.
- Reason:
  - The highest-value improvements were citation cleanup, figure/table redesign, and narrative tightening.
  - Launching even a small new run would have added reviewer-visible complexity without improving the core claim as much as the presentation fixes did.

## Strongest honest final claim

- `SpecLens-RL` is now presented as a compact, reproducible framework paper that measures specification gaps and tampering risk across lightweight RL tasks, while showing that leakage-controlled monitorability evaluation is feasible but currently narrow, with valid evidence concentrated in `SensorTamper-v0`.

## What still prevents this from being fully AAAI-ready

- The paper has not been compiled locally in AAAI format because this machine does not currently have `latexmk` or `pdflatex`.
- Final page count, float placement, and bibliography rendering still need a venue-style compile check.
- The monitorability evidence remains scientifically narrow, which is honestly reflected in the text but still limits the scope of the empirical claim.
- The benchmark remains symbolic and small-scale; that framing is appropriate, but it constrains the ambition of the paper.

## Exact commands to reproduce the final paper artifacts

```bash
MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python scripts/make_tables.py \
  --results-root results/second_pass \
  --monitor-results-root results/second_pass/monitor_eval \
  --resource-results-roots results,results/second_pass

MPLCONFIGDIR=/tmp/.mpl ./.venv/bin/python scripts/make_plots.py \
  --results-root results/second_pass \
  --monitor-results-root results/second_pass/monitor_eval

./.venv/bin/python scripts/make_paper.py --results-root results/second_pass

./.venv/bin/pytest -q
```

## Final note

- This hardening pass improved the paper more by reducing clutter and sharpening claims than by adding new artifacts. That was the intended tradeoff.
