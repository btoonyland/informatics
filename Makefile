# Convenience targets. Usage: make pipeline SET=q-bio
SET ?= cs
PY ?= python

.PHONY: install harvest filter analyze pipeline test lint

install:
	$(PY) -m pip install -e ".[dev]"

harvest:
	$(PY) -m informatics harvest --set $(SET)

filter:
	$(PY) -m informatics filter --set $(SET)

analyze:
	$(PY) -m informatics analyze --set $(SET)

pipeline: harvest filter analyze

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
