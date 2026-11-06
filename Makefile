# Convenience targets. Usage: make pipeline SET=q-bio
# Recent years only: make harvest FROM=2021-01-01, then make filter YEARS=2021-2025
# Long unattended harvests: add RETRIES=40 to wait out internet outages of up to ~8.5 hours.
#
# Every target first makes sure .venv exists with this project installed in it (and installs
# again when pyproject.toml changes, e.g. after a git pull), so nothing needs activating.
SET ?= cs
FROM ?=
YEARS ?=
RETRIES ?=
PY := .venv/bin/python
ENV := .venv/.installed

.PHONY: install harvest update filter analyze pipeline repro test lint

$(ENV): pyproject.toml
	test -x $(PY) || python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	touch $(ENV)

install: $(ENV)

harvest: $(ENV)
	$(PY) -m informatics harvest --set $(SET) $(if $(FROM),--from $(FROM)) $(if $(RETRIES),--retries $(RETRIES))

update: $(ENV)  # fetch what changed since the last harvest, then re-run the pipeline
	$(PY) -m informatics update --set $(SET) $(if $(RETRIES),--retries $(RETRIES))

filter: $(ENV)
	$(PY) -m informatics filter --set $(SET) $(if $(YEARS),--years $(YEARS))

analyze: $(ENV)
	$(PY) -m informatics analyze --set $(SET)

pipeline: harvest filter analyze

repro: $(ENV)  # the DVC pipeline in dvc.yaml (pip install dvc; dvc init once)
	$(if $(wildcard .venv/bin/dvc),.venv/bin/dvc,dvc) repro

test: $(ENV)
	$(PY) -m pytest

lint: $(ENV)
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
