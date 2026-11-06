# Convenience targets. Usage: make pipeline SET=q-bio
# Recent years only: make harvest FROM=2021-01-01, then make filter YEARS=2021-2025
SET ?= cs
# The project's virtual environment (make install creates it); plain python3 if there is none.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
FROM ?=
YEARS ?=

.PHONY: install harvest update filter analyze pipeline repro test lint

install:
	test -x .venv/bin/python || python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e ".[dev]"

harvest:
	$(PY) -m informatics harvest --set $(SET) $(if $(FROM),--from $(FROM))

update:  # fetch what changed since the last harvest, then re-run the pipeline
	$(PY) -m informatics update --set $(SET)

filter:
	$(PY) -m informatics filter --set $(SET) $(if $(YEARS),--years $(YEARS))

analyze:
	$(PY) -m informatics analyze --set $(SET)

pipeline: harvest filter analyze

repro:  # the DVC pipeline in dvc.yaml (pip install dvc; dvc init once)
	dvc repro

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
