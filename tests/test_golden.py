"""The pipeline's main tables for the synthetic corpus, compared with tests/golden/.

Run ``UPDATE_GOLDEN=1 pytest tests/test_golden.py`` after an intended change and review the
diff of the golden files: it shows exactly which numbers the change moved."""

import pandas as pd
from helpers import check_golden

from informatics.analysis import AnalysisConfig, run_analysis
from informatics.filtering import FilterConfig, run_filter

CONFIG = AnalysisConfig(min_df=3, cluster_min_df=3, k_candidates=(2, 3, 4, 5, 6),
                        trend_min_papers=5, trend_min_year_papers=1, html=False)  # fmt: skip


def _rounded(table: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    return table.round({c: digits for c in table.select_dtypes("float").columns})


def test_golden_outputs(layout):
    run_filter(layout.raw("cs"), "cs", layout.candidates("cs"), layout.background("cs"),
               layout.filter_summary("cs"), FilterConfig(background_size=100))  # fmt: skip
    out = run_analysis(["cs"], layout, config=CONFIG)
    read = pd.read_csv
    check_golden("top_phrases.csv", read(out / "top_phrases.csv").head(30))
    distinctive = read(out / "distinctive_phrases.csv").head(20)
    check_golden("distinctive_phrases.csv", _rounded(distinctive[["phrase", "papers", "z"]], 2))
    clusters = read(out / "clusters.csv")[["cluster", "papers", "top_terms"]]
    check_golden("clusters.csv", clusters)
    check_golden("themes.csv", _rounded(read(out / "themes.csv")))
    check_golden("concepts.csv", _rounded(read(out / "concepts.csv")))
    trends = read(out / "theme_trends.csv")
    columns = ["tag", "papers", "or_per_year", "adjusted_or_per_year", "adjusted_q_value"]
    check_golden("theme_trends.csv", _rounded(trends[columns], 3))
