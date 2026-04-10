# Porting Notes

## What was preserved

- The core task tuple:
  - $T = (M, R_p, J_t, C, M_{\mathrm{mon}}, \Delta, \Pi)$
- The framework-first contribution and diagnosis framing
- The three core figures:
  - framework overview
  - main results panel
  - SensorTamper monitorability figure
- The compact environment summary, main results table, and monitor table
- The strongest empirical claims:
  - clear specification gaps across all three tasks
  - explicit tampering measurement in SensorTamper-v0
  - leakage-cleared monitor evidence concentrated in SensorTamper-v0

## What changed from the AAAI-style paper

- The paper was reorganized into a cleaner NeurIPS evaluation-oriented flow:
  - Introduction
  - Related Work
  - Task Tuple and Evaluation Protocol
  - Experimental Setup
  - Results
  - Limitations
  - Conclusion
- The framework, metric, and method material was merged into a single cleaner
  section rather than spread across more main-track-style subsections.
- The results section now makes the methodological benchmark message more
  explicit: leakage and degeneracy are part of what the benchmark diagnoses.

## What was cut

- The standalone broader-impact section was cut from the NeurIPS package.
- The separate method-overview section was folded into the evaluation-protocol
  section.
- No main scientific figure, key formula, or important result table was removed.

## What was strengthened

- The motivation was sharpened around diagnosis and evaluation methodology.
- The monitorability cautionary story was made more explicit for an
  evaluation-focused review audience.
- The package now targets the official anonymous NeurIPS 2026 Evaluations
  \& Datasets track rather than a generic unnamed workshop mode.

## Relative length

- The NeurIPS version is comparable in technical density to the AAAI-style
  version, but uses a cleaner section structure for evaluation-first
  readability.

## Claim changes

- No empirical claim was strengthened beyond the saved outputs.
- The NeurIPS version keeps the same conservative final claim and does not
  present the work as a new method paper.
