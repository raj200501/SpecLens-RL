# NeurIPS Workshop Notes

- This package intentionally preserves the strongest parts of the current
  SpecLens-RL paper rather than weakening it into a generic workshop abstract.
- The official 2026 NeurIPS style package supports workshop modes directly via
  `sglblindworkshop` and `dblblindworkshop`.
- In this package, the safer anonymous fallback `dblblindworkshop` mode is used
  because no specific official NeurIPS 2026 workshop CFP was identified during
  packaging.
- `\workshoptitle{NeurIPS 2026 Workshop Submission (Target Workshop TBD)}` is a
  truthful placeholder required by the official style for workshop papers.
- The verified rebuild path uses the repo-local `.tectonic-cache/` directory so
  the package does not depend on a hidden home-directory TeX cache.

## Human updates still needed

- Replace the placeholder workshop title once a specific official workshop is chosen.
- Re-check the workshop mode if the eventual CFP explicitly requires single-blind review.
- Re-run the final compile once a workshop target is known, in case the workshop
  requests appendix, page-limit, or checklist adjustments.
