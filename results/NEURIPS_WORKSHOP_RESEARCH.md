# NeurIPS Workshop Research

Date of research: `2026-04-04`

## Official sources checked

- NeurIPS 2026 conference page:
  - https://neurips.cc/Conferences/2026
- NeurIPS 2026 downloads page:
  - https://neurips.cc/Downloads/2026
- Official 2026 formatting zip downloaded from:
  - https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip

## What was found

- The official NeurIPS 2026 conference pages are live.
- The official NeurIPS 2026 style package exists and is named `neurips_2026.sty`.
- The official sample template explicitly documents two workshop modes:
  - `sglblindworkshop`
  - `dblblindworkshop`
- The official sample template also explicitly requires `\workshoptitle{...}`
  for workshop papers.

## Whether a specific workshop was found

- No specific official NeurIPS 2026 workshop CFP or workshop page was found
  during this packaging pass.
- Therefore no specific workshop name or workshop URL can be truthfully claimed.

## Workshop fit assessment

- A strong thematic fit likely exists for future workshops on:
  - evaluation and benchmark design,
  - AI safety,
  - monitorability or oversight,
  - reinforcement learning failure analysis.
- Because no official 2026 workshop CFP was identified, that fit is currently
  conceptual rather than workshop-specific.

## Official style path conclusion

- The correct style path used here is the official generic NeurIPS 2026
  workshop style via `neurips_2026.sty`, not a workshop-specific template.

## Review mode conclusion

- The official style supports both single-blind and double-blind workshop modes.
- No official specific workshop instruction was found to force one or the other.
- Fallback assumption used here:
  - generic anonymous workshop submission
  - implemented via `dblblindworkshop`
- Reason:
  - it is the safer submission-ready default when no specific workshop review
    policy is publicly available.

## What is known

- Official style package name: `neurips_2026.sty`
- Official workshop options available: `sglblindworkshop`, `dblblindworkshop`
- `\workshoptitle{...}` is required for workshop papers in the official template

## What is unknown

- No official specific NeurIPS 2026 workshop target was identified
- No workshop-specific page limit was identified
- No workshop-specific packaging rule beyond the generic template was identified

## Fallback assumption used

- This package is built as:
  - `workshop-ready, pending final workshop targeting`
- It uses the official generic NeurIPS 2026 workshop style path with:
  - `dblblindworkshop`
  - placeholder workshop title:
    - `NeurIPS 2026 Workshop Submission (Target Workshop TBD)`
