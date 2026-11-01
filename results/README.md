# Results

`informatics analyze --set <set>` writes one folder per analysed set, or per combination of sets
(e.g. `results/cs+q-bio/` for `--set cs --set q-bio`, where cross-listed papers are counted once).

| File | Contents |
|---|---|
| `report.md` | Readable summary of everything below. Start here. |
| `top_phrases.csv` | Most common 2–3 word phrases: number and share of papers using each one |
| `distinctive_phrases.csv` | Phrases most over-represented compared with AI-only papers (log-odds z-score) |
| `rising_phrases.csv`, `falling_phrases.csv` | Phrases more / less common in the recent period than in the earlier one |
| `phrases_between_sets.csv` | Phrases that differ between the first two sets (only when analysing several) |
| `clusters.csv` | Topic clusters: size, distinctive terms, typical papers |
| `k_selection.csv` | Silhouette and stability for each number of clusters tried |
| `themes.csv` | Share of papers mentioning each secondary theme (privacy, bias, regulation, …), with 95% intervals |
| `themes_by_year.csv`, `themes_by_cluster.csv`, `themes_by_set.csv` | The same shares per year, cluster and set (long format) |
| `concepts.csv` | Share of papers per disease area, data type, clinical task and AI method, with 95% intervals |
| `concepts_by_year.csv`, `concepts_by_cluster.csv`, `concepts_by_set.csv` | The same per year, cluster and set |
| `papers_by_year.csv` | Selected papers per year, and their share of all papers in the set |
| `publication_by_year.csv` | Evidence of publication per year (journal reference, comments; OpenAlex if enriched) |
| `countries.csv` | Countries of the authors' institutions (only after `informatics enrich`) |
| `papers.parquet`, `papers.csv` | One row per paper: cluster, keywords, themes, concepts, publication data |
| `run_info.json` | Settings, periods, input files, package and normalisation versions, git commit |

The `*_by_*` tables have one row per group and tag, with columns `papers_in_group`, `papers`,
`share`, `ci_low`, `ci_high`. In the CSV files, list columns (authors, themes, …) are
JSON-encoded, so they can be parsed back with `json.loads`. `papers.parquet` and `papers.csv`
can be large, so `.gitignore` excludes them; the other files are small enough to commit.

## legacy/

Output of the first version of the pipeline, kept for reference only. It has known errors; see
[legacy/README.md](legacy/README.md).
