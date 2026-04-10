# NeurIPS Spec Workshop Target Recheck

Date checked: `2026-04-10`

## Official sources checked

- NeurIPS 2026 conference page:
  - https://neurips.cc/Conferences/2026
- NeurIPS 2026 downloads page:
  - https://neurips.cc/Downloads/2026
- NeurIPS 2026 dates page:
  - https://neurips.cc/Conferences/2026/Dates

## Current public situation

- The official NeurIPS 2026 conference site is live.
- The official dates page confirms that workshops are scheduled for `December 11-12, 2026`.
- The conference page still says that more information will be available soon.
- The official downloads page shows a `Workshops:` slot, but no named 2026 workshop list is publicly visible there yet.

## Was a named official 2026 workshop matching this paper found?

No named official target was found.

As of `2026-04-10`, the public official sources checked above confirm that NeurIPS 2026 workshops exist as part of the conference schedule, but they do not yet expose a named workshop list or a workshop-specific CFP that can be truthfully targeted in this repository.

## Addendum: official NeurIPS track fit now matters more than unnamed workshop fit

Since the original workshop recheck, the public NeurIPS 2026 Evaluations
\& Datasets FAQ has become the more relevant official source for this paper's
targeting decision. That FAQ states that:

- accepted ED papers appear in the same NeurIPS proceedings as main-track
  papers,
- the ED track uses the same LaTeX template, and
- papers whose main contribution is an evaluation framework or empirical
  evaluation insight should be framed for ED rather than for another track.

For this repository, that changes the practical recommendation: the strongest
named official NeurIPS target is now the anonymous ED track, not a generic
unnamed workshop package.

## Best-fit conclusion

- Best current fit family: `evaluation / benchmark design`
- Strong secondary fit families:
  - `AI safety`
  - `monitorability / oversight`
  - `reward misspecification diagnosis`

Why:
- The paper is strongest as a compact diagnostic framework and benchmark.
- The monitor story is real but narrow, so it fits better as a methodological evaluation result than as a broad oversight benchmark paper.
- The benchmark's strongest workshop value is honest measurement of proxy-vs-true divergence, explicit corruption-channel behavior, and leakage-aware monitor evaluation.

## Packaging recommendation

- Keep the package `generic-workshop-ready`: `no`
- Keep anonymized workshop mode: `no`
- Keep `dblblindworkshop` as the current default: `no`
- Retarget to the official anonymous ED track: `yes`

Reason:
- No official named workshop page was found.
- An official named NeurIPS track now exists whose stated review frame matches
  the paper better than an unnamed workshop.
- Using the official anonymous `eandd` mode removes the placeholder workshop
  title and improves target-fit honesty.

## Practical implication for this repo

- Do not invent a workshop target.
- Do not claim workshop-specific fit beyond the theme family.
- Prefer the official ED-track framing over generic workshop framing until a
  concrete workshop CFP appears that is clearly better aligned.
