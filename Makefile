# Convenience targets. Usage: make pipeline SET=q-bio
SET ?= cs
PY ?= python

.PHONY: install harvest update filter analyze pipeline repro test lint

install:
	$(PY) -m pip install -e ".[dev]"

harvest:
	$(PY) -m informatics harvest --set $(SET)

update:  # fetch what changed since the last harvest, then re-run the pipeline
	$(PY) -m informatics update --set $(SET)

filter:
	$(PY) -m informatics filter --set $(SET)

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
