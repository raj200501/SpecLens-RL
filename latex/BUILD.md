# Building the Paper

This folder is designed to be Overleaf-ready.

## Files

- `main.tex`: paper entrypoint
- `sections/`: section files
- `figures/`: generated figure PDFs
- `tables/`: generated LaTeX tables
- `references_seed.bib`: seed bibliography

## Local Build

Use a standard LaTeX workflow such as:

```bash
latexmk -pdf main.tex
```

or upload the entire `latex/` folder to Overleaf.

## Artifact Policy

- Non-experimental sections may be edited directly.
- Experimental placeholders should only be populated from generated outputs.
- Figures and tables in the final draft must correspond to real files on disk.
