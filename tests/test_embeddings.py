import numpy as np
import pytest
from synthetic import TOPIC_WORDS

from informatics.analysis import AnalysisConfig, run_analysis
from informatics.embeddings import embed, embedding_text, load_encoder, load_vectors, model_slug
from informatics.filtering import FilterConfig, run_filter
from informatics.storage import read_json, read_parquet


class CountingEncoder:
    """Fake sentence encoder: one direction per synthetic topic, plus a little noise."""

    def __init__(self):
        self.calls = 0
        self.rng = np.random.default_rng(0)

    def __call__(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            vector = np.array(
                [any(w in text.lower() for w in words) for words in TOPIC_WORDS.values()], float
            )
            vectors.append(vector + self.rng.normal(0, 0.05, size=len(vector)))
        return np.array(vectors)


def test_embed_caches_by_text(tmp_path):
    cache, encoder = tmp_path / "m.parquet", CountingEncoder()
    first = embed(["1", "2"], ["tumor segmentation", "mood tracking"], cache, encoder)
    again = embed(["1", "2"], ["tumor segmentation", "mood tracking"], cache, encoder)
    assert encoder.calls == 1 and np.allclose(first, again)
    embed(["1", "2"], ["tumor segmentation", "sleep patterns"], cache, encoder)  # 2 changed
    assert encoder.calls == 2 and len(read_parquet(cache)) == 2
    with pytest.raises(ValueError, match="informatics embed"):
        load_vectors(["1", "3"], ["tumor segmentation", "x"], cache)


def test_missing_dependency_has_install_hint():
    pytest.importorskip("numpy")
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match=r"\.\[embeddings\]"):
            load_encoder("any/model")


def test_analysis_clusters_on_cached_embeddings(layout):
    run_filter(layout.raw("cs"), "cs", layout.candidates("cs"), layout.background("cs"),
               layout.filter_summary("cs"), FilterConfig())  # fmt: skip
    selected = read_parquet(layout.candidates("cs"))
    selected = selected[selected["duplicate_of"].isna() & selected["keyword_rule"]]
    texts = [
        embedding_text(t, a) for t, a in zip(selected["title"], selected["abstract"], strict=True)
    ]
    embed(
        selected["id"].tolist(),
        texts,
        layout.embeddings(model_slug("fake/model")),
        CountingEncoder(),
    )
    config = AnalysisConfig(
        min_df=3, cluster_min_df=3, k_candidates=(2, 3, 4, 5, 6), embeddings="fake/model"
    )
    out = run_analysis(["cs"], layout, config=config)
    assert read_json(out / "run_info.json")["counts"]["k"] == 4
    assert "clustered on sentence embeddings (fake/model)" in (out / "report.md").read_text()
