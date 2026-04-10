# NeurIPS Build

This folder is a parallel NeurIPS 2026 paper package derived
from the strongest current SpecLens-RL paper.

## Local build

From the repository root:

```bash
cd neurips_latex
TECTONIC_CACHE_DIR=$PWD/.tectonic-cache tectonic -C --keep-logs --keep-intermediates final.tex > final_build.log 2>&1
```

That command compiles the single-file submission source using:

- the repo-local `neurips_2026.sty` package,
- the figures and tables inside this folder,
- the repo-local `.tectonic-cache/` directory populated during packaging.

## Notes

- `main.tex` is the modular source.
- `final.tex` is the flattened single-file submission source.
- `final.pdf` is the compiled submission artifact produced from `final.tex`.
- `.tectonic-cache/` stores the TeX resources needed for the verified offline rebuild path.
- The package currently targets the anonymous NeurIPS 2026 Evaluations \&
  Datasets track via the official `eandd` option in `neurips_2026.sty`.
