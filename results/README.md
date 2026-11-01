# Results

`informatics analyze --set <set>` writes one folder per analysed set, or per combination of sets
(e.g. `results/cs+q-bio/` for `--set cs --set q-bio`, where cross-listed papers are counted once).

| File | Contents |
|---|---|
| `report.md` | Readable summary of everything below. Start here. |
| `top_phrases.csv` | Most common 2–3 word phrases: number and share of papers using each one |
| `distinctive_phrases.csv` | Phrases most over-represented compared with AI-only papers (log-odds z-score) |
| `clusters.csv` | Topic clusters: size, distinctive terms, typical papers |
| `k_selection.csv` | Silhouette and stability for each number of clusters tried |
| `themes.csv` | Share of papers mentioning each secondary theme (privacy, bias, regulation, …) |
| `themes_by_year.csv`, `themes_by_cluster.csv` | The same shares per year and per cluster |
| `papers_by_year.csv` | Selected papers per year, and their share of all papers in the set |
| `papers.parquet`, `papers.csv` | One row per paper with its cluster, matched keywords and themes |
| `run_info.json` | Settings, input files, package versions and git commit of the run |

In the CSV files, list columns (authors, themes, …) are JSON-encoded, so they can be parsed back
with `json.loads`. `papers.parquet` and `papers.csv` can be large, so `.gitignore` excludes them;
the other files are small enough to commit.

## legacy/

Output of the first version of the pipeline, kept for reference only. It has known errors; see
[legacy/README.md](legacy/README.md).
