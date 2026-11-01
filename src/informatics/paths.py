"""Where each pipeline step reads and writes its files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def slug(set_name: str) -> str:
    """File-name-safe set name: ``physics:astro-ph`` -> ``physics_astro-ph``."""
    return set_name.replace(":", "_").replace("/", "_")


@dataclass(frozen=True)
class Layout:
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")

    # data/raw: one canonical Parquet file per set
    def raw(self, set_name: str) -> Path:
        return self.data_dir / "raw" / f"{slug(set_name)}.parquet"

    def raw_info(self, set_name: str) -> Path:
        """How the raw file was made (source, query or filter, dates): the population."""
        return self.data_dir / "raw" / f"{slug(set_name)}_import.json"

    def harvest_dir(self, set_name: str) -> Path:
        return self.data_dir / "raw" / f"{slug(set_name)}_harvest"

    # data/interim: output of the keyword filter (the wide net) ...
    def candidates(self, set_name: str) -> Path:
        return self.data_dir / "interim" / f"{slug(set_name)}_candidates.parquet"

    def background(self, set_name: str) -> Path:
        return self.data_dir / "interim" / f"{slug(set_name)}_ai_only_sample.parquet"

    def filter_summary(self, set_name: str) -> Path:
        return self.data_dir / "interim" / f"{slug(set_name)}_filter_summary.json"

    # ... and of the second selection stage (informatics classify)
    def relevance(self, set_name: str) -> Path:
        return self.data_dir / "interim" / f"{slug(set_name)}_relevance.parquet"

    def relevance_info(self, set_name: str) -> Path:
        return self.data_dir / "interim" / f"{slug(set_name)}_relevance.json"

    def llm(self, set_name: str) -> Path:
        """Language-model answers (relevance and themes) for each candidate."""
        return self.data_dir / "interim" / f"{slug(set_name)}_llm.parquet"

    def llm_batches(self, set_name: str) -> Path:
        """Message batches submitted for a set, so an interrupted run can resume."""
        return self.data_dir / "interim" / f"{slug(set_name)}_llm_batches.json"

    def openalex(self, set_name: str) -> Path:
        return self.data_dir / "interim" / f"{slug(set_name)}_openalex.parquet"

    # caches shared by all sets
    def text_cache(self) -> Path:
        return self.data_dir / "interim" / "text_cache.parquet"

    def embeddings(self, model_slug: str) -> Path:
        return self.data_dir / "interim" / "embeddings" / f"{model_slug}.parquet"

    # results: one folder per analysed set (or combination of sets)
    def results(self, set_names: list[str]) -> Path:
        return self.results_dir / "+".join(slug(s) for s in set_names)
