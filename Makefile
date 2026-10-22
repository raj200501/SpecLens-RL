PYTHON ?= python
WORKERS ?= 4

.PHONY: install test lint smoke experiments verify report paper all

install:  ## exact dependency versions behind the committed results, then the package
	$(PYTHON) -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install --no-deps -e .

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

verify:  ## check that committed results were produced by the current code and config
	$(PYTHON) -m speclens.experiment --config configs/paper.yaml --out results --verify

report:  ## figures, tables and numbers from results/ (about a minute)
	$(PYTHON) -m speclens.report --config configs/paper.yaml --results results --paper paper

paper: report  ## compile paper/main.pdf (needs latexmk or tectonic)
	cd paper && (latexmk -pdf -interaction=nonstopmode main.tex || tectonic main.tex)

all: experiments paper
