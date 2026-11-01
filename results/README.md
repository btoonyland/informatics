# Results

`informatics analyze --set <set>` writes one folder per analysed set, or per combination of sets
(e.g. `results/cs+q-bio/` for `--set cs --set q-bio`, where cross-listed papers are counted once).

| File | Contents |
|---|---|
| `report.html` | Interactive report: charts with table views, and a searchable list of the papers. One self-contained file; start here. |
| `report.md` | The same findings as text and tables. |
| `top_phrases.csv` | Most common 2–3 word phrases: number and share of papers using each one |
| `distinctive_phrases.csv` | Phrases most over-represented compared with AI-only papers (log-odds z-score) |
| `rising_phrases.csv`, `falling_phrases.csv` | Phrases more / less common in the recent period than in the earlier one |
| `phrases_between_sets.csv` | Phrases that differ between the first two sets (only when analysing several) |
| `clusters.csv` | Topic clusters: size, distinctive terms, typical papers |
| `k_selection.csv` | Silhouette and stability for each number of clusters tried |
| `cluster_map.csv` | 2-D (t-SNE) position and cluster of up to 2,000 papers, as drawn in the topic map |
| `themes.csv` | Share of papers discussing each secondary theme (privacy, bias, regulation, …), with 95% intervals |
| `themes_by_year.csv`, `themes_by_cluster.csv`, `themes_by_set.csv` | The same shares per year, cluster and set (long format) |
| `theme_trends.csv` | Per theme: odds ratio per year for all papers and within topics, 95% intervals, p- and q-values, change in percentage points per year |
| `concepts.csv` | Share of papers per disease area, data type, clinical task and AI method, with 95% intervals |
| `concepts_by_year.csv`, `concepts_by_cluster.csv`, `concepts_by_set.csv` | The same per year, cluster and set |
| `concept_trends.csv` | Trend estimates as in `theme_trends.csv`, per concept |
| `papers_by_year.csv` | Selected papers per year; per set, all papers and candidates that year and the share selected |
| `publication_by_year.csv` | Evidence of publication per year (journal reference, comments; OpenAlex if enriched) |
| `countries.csv` | Countries of the authors' institutions (only after `informatics enrich`) |
| `papers.parquet`, `papers.csv` | One row per paper: cluster, keywords, themes, concepts, publication data |
| `run_info.json` | Settings, how each set's papers were selected, periods, package and normalisation versions, git commit, and the size and SHA-256 hash of every input file |

The `*_by_*` tables have one row per group and tag, with columns `papers_in_group`, `papers`,
`share`, `ci_low`, `ci_high`. In the CSV files, list columns (authors, themes, …) are
JSON-encoded, so they can be parsed back with `json.loads`. `papers.parquet` and `papers.csv`
can be large, so `.gitignore` excludes them; the other files are small enough to commit.

## legacy/

Output of the first version of the pipeline, kept for reference only. It has known errors; see
[legacy/README.md](legacy/README.md).
