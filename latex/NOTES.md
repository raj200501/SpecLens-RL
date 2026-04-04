# Paper Notes

- This is a framework paper, not a method-SOTA paper.
- Do not insert empirical claims by hand unless they are supported by generated
  tables or figures.
- Keep the framing centered on the task schema, metric suite, monitor channels,
  leakage diagnostics, and reproducible protocol.
- If an environment degenerates or a monitor channel leaks labels, record that
  in `results/OVERRIDES.md` and reflect it in the paper.
- Before any submission cycle, manually re-check bibliography metadata and
  formatting against the target venue style.

## Human polish still needed before submission

- Compile the paper on a machine with `latexmk` or `pdflatex` and check the
  final AAAI page count, float placement, and bibliography formatting.
- Read captions and table notes line by line after compile; they are now much
  tighter, so small wording issues will be more visible.
- Manually verify author block, anonymization state, and venue-specific style
  details before any real submission.
- Reconfirm that the main-paper figure set stays limited to the conceptual
  overview, the compact main-results panel, and the SensorTamper monitor figure.
- Keep appendix or repository-only artifacts out of the core narrative unless a
  reviewer-facing revision specifically benefits from adding them.
