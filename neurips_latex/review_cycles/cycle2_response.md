# Cycle 2 Author Response

## Summary

Cycle 2 focused on robustness and artifact trustworthiness. The paper already
read better after Cycle 1; this pass made the evidential chain more explicit by
adding seed-level support artifacts, sharpening the negative-monitor narrative,
and documenting software assets and reproduction commands in a way that better
matches E\&D expectations.

## Reviewer 1

- Concern: aggregate means were not enough.
  Fix:
  - Generated [aggregated_seed_variability_results.csv](/Users/rajkashikar/SpecLens-RL/results/second_pass/aggregated_seed_variability_results.csv).
  - Generated [aggregated_seed_consistency_results.csv](/Users/rajkashikar/SpecLens-RL/results/second_pass/aggregated_seed_consistency_results.csv).
  - Added [seed_variability.tex](/Users/rajkashikar/SpecLens-RL/neurips_latex/tables/seed_variability.tex) and cited it in [appendix_repro.tex](/Users/rajkashikar/SpecLens-RL/neurips_latex/sections/appendix_repro.tex).
- Concern: appendix claims needed direct artifact support.
  Fix:
  - Updated [SPEC_VERIFY_BEFORE_WRITE.md](/Users/rajkashikar/SpecLens-RL/results/SPEC_VERIFY_BEFORE_WRITE.md) so the new seed-robustness and seed-variability claims map to concrete saved outputs.

## Reviewer 2

- Concern: OOD language still risked over-interpretation.
  Fix:
  - Tightened the wording in [experiments.tex](/Users/rajkashikar/SpecLens-RL/neurips_latex/sections/experiments.tex) and [framework.tex](/Users/rajkashikar/SpecLens-RL/neurips_latex/sections/framework.tex) so OOD is framed only as symbolic stress testing.
- Concern: tamper-off paragraph needed the true-score recovery.
  Fix:
  - Updated [results.tex](/Users/rajkashikar/SpecLens-RL/neurips_latex/sections/results.tex) to report the ID true-score improvement from `-3.346` to `0.288` under the tamper-off ablation.
- Concern: negative monitor result needed clearer interpretation.
  Fix:
  - Revised [results.tex](/Users/rajkashikar/SpecLens-RL/neurips_latex/sections/results.tex) so ProxyFarming and GemTimeout are explicitly described as not currently defining trustworthy grouped hack-monitor tasks under the accepted protocol.

## Reviewer 3

- Concern: software licenses and code access were still underdocumented.
  Fix:
  - Added explicit upstream software versions and licenses to [appendix_repro.tex](/Users/rajkashikar/SpecLens-RL/neurips_latex/sections/appendix_repro.tex).
  - Added exact regeneration commands to the same appendix.
  - Reflected the same commitments in [checklist.tex](/Users/rajkashikar/SpecLens-RL/neurips_latex/sections/checklist.tex).
- Concern: repo-level target/compliance notes needed to match the manuscript.
  Fix:
  - Updated [WORKSHOP_TARGET_STATUS.md](/Users/rajkashikar/SpecLens-RL/neurips_latex/WORKSHOP_TARGET_STATUS.md) to state the official template mode, conservative page-limit posture, and now-present checklist/appendix.
