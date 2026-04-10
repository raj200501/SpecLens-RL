# NeurIPS Track Notes

- This package intentionally preserves the strongest parts of the current
  SpecLens-RL paper rather than weakening it into a generic abstract.
- As of `2026-04-10`, the public NeurIPS 2026 Evaluations \& Datasets FAQ
  confirms that the ED track uses the same LaTeX template as the main track and
  that papers whose primary contribution is evaluation framing should be
  submitted to ED rather than to another track.
- This package therefore uses the official anonymous `eandd` mode in
  `neurips_2026.sty` rather than a generic unnamed workshop mode.
- The verified rebuild path uses the repo-local `.tectonic-cache/` directory so
  the package does not depend on a hidden home-directory TeX cache.

## Human updates still needed

- Re-check the ED FAQ and handbook closer to submission in case formatting,
  checklist, or data-hosting requirements change.
- Switch to `nonanonymous` only if preparing a non-anonymous version allowed by
  the track rules.
