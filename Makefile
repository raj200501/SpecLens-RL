PYTHON ?= python
WORKERS ?= 4

.PHONY: install test lint smoke experiments report paper all

install:  ## install the package with development extras
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## unit tests (about a minute on a laptop)
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

smoke:  ## the full pipeline at toy scale, into a scratch directory
	$(PYTHON) -m speclens.experiment --config configs/smoke.yaml --out /tmp/speclens-smoke/results --workers $(WORKERS)
	$(PYTHON) -m speclens.report --config configs/smoke.yaml --results /tmp/speclens-smoke/results --paper /tmp/speclens-smoke/paper

experiments:  ## every training run in the paper (a few CPU-hours; resumable)
	$(PYTHON) -m speclens.experiment --config configs/paper.yaml --out results --workers $(WORKERS)

report:  ## figures, tables and numbers from results/ (seconds)
	$(PYTHON) -m speclens.report --config configs/paper.yaml --results results --paper paper

paper: report  ## compile paper/main.pdf (needs latexmk or tectonic)
	cd paper && (latexmk -pdf -interaction=nonstopmode main.tex || tectonic main.tex)

all: experiments paper
