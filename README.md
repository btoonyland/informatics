# AI in medicine on arXiv

Tools to collect arXiv metadata (e.g. the `cs` and `q-bio` archives) and select the papers about
AI in medicine with a keyword filter. They then analyse those papers: their most common,
distinctive, rising and falling phrases, their topic clusters, what they are about (disease
area, data type, clinical task, AI method), how often themes such as privacy, bias, regulation
and adoption come up year by year, and whether and where the papers were published.

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
cross-listed in both are counted once, and the report compares the sets:

```bash
informatics analyze --set cs --set q-bio
```

Every command has `--help`. `python -m informatics …` works without the console script, and
`make pipeline SET=q-bio` runs harvest, filter and analyze in one go.

## Pipeline

```
harvest | import-kaggle | import-legacy-csv  ->  data/raw/<set>.parquet
import-pubmed (optional comparison corpus)   ->  data/raw/pubmed.parquet
filter                                       ->  data/interim/<set>_ai_medicine.parquet
                                                 data/interim/<set>_ai_only_sample.parquet
                                                 data/interim/<set>_filter_summary.json
enrich (optional, OpenAlex)                  ->  data/interim/<set>_openalex.parquet
embed (optional, sentence embeddings)        ->  data/interim/embeddings/<model>.parquet
analyze                                      ->  results/<set>/report.md and tables
label-sample / label-evaluate                ->  labels/  (precision and recall of the filter)
label-themes-sample / label-themes-evaluate  ->  labels/  (precision of each theme)
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
| `import-pubmed --set pubmed` | Builds a PubMed comparison set (default query `"artificial intelligence"[MeSH Terms]`, up to `--per-year` records per year). |
| `filter --set S` | Excludes withdrawn and non-English papers, selects AI-in-medicine papers, flags near-duplicates, and keeps a reproducible random sample of AI-only papers for comparison. |
| `enrich --set S` | Looks the selected papers up in OpenAlex: venue, institution types, countries, citations. |
| `embed --set S` | Caches sentence embeddings of the selected papers (needs `pip install -e ".[embeddings]"`). |
| `analyze --set S [--set T]` | Everything below. Writes tables, `report.md` and `run_info.json`. Options include `--recent 2023-2025 --earlier 2019-2022` and `--embeddings MODEL`. |
| `label-sample` / `label-evaluate` | Hand-label a sample to measure the filter's precision and recall. |
| `label-themes-sample` / `label-themes-evaluate` | Hand-label tagged papers to measure each theme's precision. |

## How papers are selected

`src/informatics/keywords.py` holds all keyword lists, so the rules can be changed without touching
code. A paper is selected when its title and abstract contain at least one AI term and one medicine term
(`--min-ai-hits` and `--min-medicine-hits` raise the bar). Terms match whole words, including
plural, hyphenated and spelling variants: `patients?`, `neural net(work)s?`, `LLMs` / `large language
models`, `deep-learning`, `health-care`.

Before that, the filter **excludes** withdrawn papers (arXiv's withdrawal notice in the comments or
abstract) and, among papers with AI terms, abstracts not written in English (fewer than half the
words are English; bilingual abstracts are kept). After selection it flags **near-duplicates**, the
same paper posted under two IDs (MinHash on word 5-grams, Jaccard ≥ 0.8). The extra copies stay in
the file with `duplicate_of` set and are left out of the analysis.

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
4. For the themes, `label-themes-sample` draws papers tagged with each theme and shows the words
   that matched; mark whether the paper really discusses the theme, then `label-themes-evaluate`
   reports each theme's precision.

Commit the labels: they are the evidence for the filter's quality.

## How text is normalised

Phrases are counted by a normalised key, merged *before* counting, so a paper that writes a term
two ways counts once. Tables show each key's most common written form. The rules are in
`src/informatics/normalize.py`, in this order:

1. **Characters.** Unicode NFKC (ligatures, full-width letters), one kind of hyphen, and TeX turned
   into text (`$\alpha$-synuclein`, `\emph{in vivo}`, `Gr\"obner`), which arXiv abstracts often contain.
2. **Acronyms defined in the abstract**, using the Schwartz & Hearst (2003) algorithm. After
   "artificial neural networks (ANNs)", later "ANNs" become the long form. Expansion is per abstract,
   because acronyms are ambiguous across papers (AD: Alzheimer's disease or anomaly detection).
3. **Curated synonyms** for terms with one meaning in this field: EHR / EMR / "electronic medical
   records" → electronic health records; undefined AI, ML, LLM, CNN, MRI, CT, ICU → long forms.
4. **British → American spelling** (tumour, visualisation, modelling, paediatric, centre). Suffix
   rules only apply when the American form is at least as common in the corpus, so "exercise"
   never becomes "exercize".
5. **Lemmas** from a dictionary (simplemma): studies → study, fine-tuned → fine-tune,
   pretrained → pre + train. Some words are protected where dictionaries are wrong for this field
   ("media" is not "medium", "data" is not "datum").
6. **Compounds**: multi-modal = multimodal = "multi modal" (when the corpus hyphenates it).

Stemming is deliberately not used. A stemmer merges words that mean different things here:
specific/specificity, medical/medication, organ/organization, experiment/experience. Topic-model
research also finds it does not improve topics (Schofield & Mimno 2016).
`tests/test_normalize.py` lists pairs that must merge and pairs that must not. Prepared texts are
cached in `data/interim/text_cache.parquet`; `--no-cache` ignores the cache.

## What the analysis computes

- **Phrases.** 2–3 word phrases; stop words are only checked at a phrase's edges, and punctuation
  ends a phrase. Counts are numbers of papers, reported as shares, and fragments of longer phrases
  are dropped. Four comparisons use a smoothed log-odds ratio (Monroe et al. 2008):
  - **distinctive**: selected papers against the AI-only sample;
  - **rising** and **falling**: the last three years against the four before (or
    `--recent`/`--earlier`);
  - **between sets**: papers only in the first set against papers only in the second.
- **Topic clusters.** The features are TF-IDF words and two-word phrases, without the selection
  keywords. The pipeline clusters with LSA + k-means, choosing k by silhouette and reporting
  stability; `--embeddings MODEL` clusters on sentence embeddings instead. Each cluster gets its
  distinctive terms, its most common concepts and its typical papers.
- **Themes and concepts.** Each theme's share of papers, with 95% Wilson intervals, overall and by
  year, cluster and set. Concepts (`src/informatics/concepts.py`) tag each paper's disease area,
  data type, clinical task and AI method in the same way.
- **Trends and publication.** Selected papers per year, as a share of all papers in the set.
  Evidence of publication comes from arXiv's journal reference and comments; with `enrich`, it
  also covers venue type, healthcare and company authors, countries and citations. Years come from
  the arXiv ID (PubMed records use their publication date).

## Notes on the data sources

- arXiv rewrote its OAI-PMH service in 2025. The default endpoint is `https://oaipmh.arxiv.org/oai`;
  pass `--base-url` to use another one. Run `informatics list-sets` to check set names. The
  harvester waits 5 s between requests (`--delay`).
- `harvest --from/--until` filter on the date a record last *changed*, not its submission date.
  Without them the whole set is downloaded.
- The old `All_CS.csv` kept only surnames; `import-legacy-csv` cannot restore first names. A fresh
  harvest or the Kaggle snapshot has full names.
- **OpenAlex** has required a free API key since February 2026 (openalex.org/settings/api). Pass
  `--api-key` or set `OPENALEX_API_KEY`. Papers are looked up by journal DOI, or else by arXiv's
  own DOI (10.48550/arXiv.<id>). OpenAlex shares in the report are among the papers it found.
- **PubMed** search returns at most 10,000 records per query, so `import-pubmed` queries one year
  at a time and keeps the first `--per-year` records: a sample of the clinical literature, not all
  of it. An NCBI API key (`--api-key` or `NCBI_API_KEY`) allows faster requests.
- **Embeddings** download the model from Hugging Face the first time. The default is
  `sentence-transformers/all-MiniLM-L6-v2`; `allenai/specter2_base` was trained on scientific
  papers.
- The filter runs on a normal CPU with pandas, so the multi-GPU Dask setup is no longer needed.

## Development

```bash
make test
```

```bash
make lint
```

Tests use a synthetic corpus (`tests/synthetic.py`) and recorded responses (`tests/fixtures/` and
inline), so they need no network access. CI runs ruff and pytest on Python 3.10 and 3.13.

```
src/informatics/   the package (one module per step; keywords.py and concepts.py hold the term lists)
tests/             pytest suite
results/           analysis output; results/legacy/ holds the old, known-faulty output
misc/              stand-alone scripts unrelated to this pipeline (see misc/README.md)
data/, labels/     created when you run the pipeline (data/ is git-ignored)
```
