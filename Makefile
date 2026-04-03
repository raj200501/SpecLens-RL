PYTHON ?= ./.venv/bin/python

.PHONY: install preflight smoke stage1 stage2 plots tables paper clean_artifacts report

install:
	$(PYTHON) scripts/install_check.py --write-requirements-report

preflight:
	$(PYTHON) scripts/preflight.py

smoke:
	$(PYTHON) scripts/run_stage0.py

stage1:
	$(PYTHON) scripts/run_stage1.py

stage2:
	$(PYTHON) scripts/run_stage2.py

plots:
	$(PYTHON) scripts/make_plots.py

tables:
	$(PYTHON) scripts/make_tables.py

paper:
	$(PYTHON) scripts/make_paper.py

clean_artifacts:
	$(PYTHON) scripts/cleanup_artifacts.py

report:
	$(PYTHON) scripts/report_status.py
