# AI in medicine on arXiv

Tools to collect arXiv metadata (e.g. the `cs` and `q-bio` archives) and select the papers about
AI in medicine with a keyword filter. They then analyse those papers: their most common and most
distinctive phrases, their topic clusters, and how often themes such as privacy, bias, regulation
and adoption are mentioned, year by year.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m pip install -e ".[dev]"
```

Get the raw data for a set, using **one** of these:

```bash
informatics harvest --set cs                                     # arXiv OAI-PMH; takes hours, resumable
```

```bash
informatics import-kaggle arxiv-metadata-oai-snapshot.json --set cs   # Cornell's Kaggle snapshot; minutes
```

```bash
informatics import-legacy-csv All_CS.csv --set cs                # reuse a CSV from the old scripts/arXivAPI.py
```

Then select and analyse:

```bash
informatics filter --set cs
```

```bash
informatics analyze --set cs
```

Open `results/cs/report.md`. Use `--set` more than once to analyse sets together; papers
cross-listed in both are then counted once:

```bash
informatics analyze --set cs --set q-bio
```

Every command has `--help`. `python -m informatics …` works without the console script, and
`make pipeline SET=q-bio` runs harvest, filter and analyze in one go.

## Pipeline

```
harvest | import-kaggle | import-legacy-csv  ->  data/raw/<set>.parquet
filter                                       ->  data/interim/<set>_ai_medicine.parquet
                                                 data/interim/<set>_ai_only_sample.parquet
                                                 data/interim/<set>_filter_summary.json
analyze                                      ->  results/<set>/report.md and tables
label-sample, label-evaluate                 ->  labels/  (precision and recall of the filter)
```

Parquet is the working format: it keeps arXiv IDs as text (`0708.2931`, not `708.2931`) and lists
as lists. CSV files are only written for reading in a spreadsheet, with list columns encoded as JSON.
`informatics to-csv <file.parquet>` converts any intermediate file.

| Command | What it does |
|---|---|
| `harvest --set S` | Downloads set S over OAI-PMH. Each page is saved as it arrives, so re-running continues an interrupted harvest. Requests time out, are retried a bounded number of times, and honour `Retry-After`. |
| `list-sets` | Lists the set names the OAI-PMH endpoint offers. |
| `count --set S` | All-time article count from the arXiv search API (query `cat:S*`, as before). |
| `import-kaggle FILE --set S` | Builds `data/raw/S.parquet` from the Kaggle snapshot (cross-listed papers included, like the OAI sets). |
| `import-legacy-csv FILE --set S` | Converts the old harvester's CSV, repairing IDs, garbled accents and list columns. |
| `filter --set S` | Selects AI-in-medicine papers and keeps a reproducible random sample of AI-only papers for comparison. |
| `analyze --set S [--set T]` | Phrases, clusters, themes and trends. Writes tables plus `report.md` and `run_info.json`. |
| `label-sample --set S` / `label-evaluate --set S` | Hand-label a sample to measure the filter's precision and recall. |

## How papers are selected

`src/informatics/keywords.py` holds all keyword lists, so the rules can be changed without touching
code. A paper is selected when its title and abstract contain at least one AI term and one medicine term
(`--min-ai-hits` and `--min-medicine-hits` raise the bar). Terms match whole words, including
plural, hyphenated and spelling variants: `patients?`, `neural net(work)s?`, `LLMs` / `large language
models`, `deep-learning`, `health-care`.

The secondary **themes** are the original terms (adoption, privacy, bias, regulation, …) plus
variants that keep the meaning ("personalised", "ethics", "regulatory"). Some themes have
**exclusions** for unrelated senses, such as "gene regulation", "MRI bias field" or "lung
compliance". `accuracy` appears in almost every machine-learning abstract, so treat it as a baseline.

Keyword rules make mistakes in both directions. To measure them:

1. `informatics label-sample --set cs` writes `labels/cs_to_label.csv`, a random sample of
   selected papers and of each kind of rejected paper, in random order. It also writes a key
   file; keep the key closed while labelling.
2. Mark each paper's `relevant` column 1 (about AI in medicine) or 0.
3. `informatics label-evaluate --set cs` reports precision with a 95% interval, and estimated
   recall (each rejected group's labelled share scaled up to its size).

Commit the labels: they are the evidence for the filter's quality.

## What the analysis computes

- **Phrases.** These are 2–3 word phrases from the original text. Stop words are checked only at a
  phrase's edges, and punctuation and brackets end a phrase, so "low back pain" stays intact (the old
  code produced "low pain"). Counts are numbers of papers, reported as shares. Fragments of longer phrases are
  dropped, and singular and plural are merged. **Distinctive phrases** compare the selected papers
  with the AI-only sample using a smoothed log-odds ratio (Monroe et al. 2008). This removes both the
  selection keywords and generic academic phrasing, which are common in both groups.
- **Topic clusters.** The features are TF-IDF words and two-word phrases from title and abstract,
  without the selection keywords (every paper has them). The pipeline reduces them with LSA, then
  runs k-means, choosing k by silhouette from several candidates and reporting stability. Each
  cluster is labelled by its most distinctive terms and listed with its size and typical papers.
- **Themes and trends.** For each theme it reports the share of papers mentioning it, overall,
  per year and per cluster. It also reports selected papers per year as a share of all papers in
  the set. Years come from the arXiv ID: the `created` field of the OAI `arXiv` format sometimes
  lags the first submission by months.

## Notes on the data sources

- arXiv rewrote its OAI-PMH service in 2025. The default endpoint is `https://oaipmh.arxiv.org/oai`;
  pass `--base-url` to use another one. Run `informatics list-sets` to check set names. The
  harvester waits 5 s between requests (`--delay`).
- `harvest --from/--until` filter on the date a record last *changed*, not its submission date.
  Without them the whole set is downloaded.
- The old `All_CS.csv` kept only surnames; `import-legacy-csv` cannot restore first names. A fresh
  harvest or the Kaggle snapshot has full names.
- The filter runs on a normal CPU with pandas, so the multi-GPU Dask setup is no longer needed.

## Development

```bash
make test
```

```bash
make lint
```

Tests use a synthetic corpus (`tests/synthetic.py`) and recorded OAI-PMH pages (`tests/fixtures/`),
so they need no network access. CI runs ruff and pytest on Python 3.10 and 3.13.

```
src/informatics/   the package (one module per step; keywords.py holds the keyword lists)
tests/             pytest suite
results/           analysis output; results/legacy/ holds the old, known-faulty output
misc/              stand-alone scripts unrelated to this pipeline (see misc/README.md)
data/, labels/     created when you run the pipeline (data/ is git-ignored)
```
