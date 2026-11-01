from __future__ import annotations

from pathlib import Path

import pytest
from synthetic import make_corpus

from informatics.paths import Layout
from informatics.records import RAW_SCHEMA
from informatics.storage import write_parquet


@pytest.fixture(scope="session")
def corpus():
    """Synthetic papers with known kind (ai_medicine, ai_only, ...) and topic columns."""
    return make_corpus(40, seed=0)


@pytest.fixture
def layout(tmp_path: Path, corpus) -> Layout:
    """A data/results layout in a temporary folder, with the synthetic corpus as data/raw/cs."""
    layout = Layout(tmp_path / "data", tmp_path / "results")
    write_parquet(corpus.drop(columns=["kind", "topic"]), layout.raw("cs"), schema=RAW_SCHEMA)
    return layout
