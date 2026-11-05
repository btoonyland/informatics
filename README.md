# AI in medicine on arXiv

Tools to collect paper metadata (arXiv archives such as `cs` and `q-bio`, and optionally PubMed
or OpenAlex), find the papers about AI in medicine, and analyse them: their most common,
distinctive, rising and falling phrases, their topic clusters, what they are about (disease
area, data type, clinical task, AI method), how often themes such as privacy, bias, regulation
and adoption come up, how fast those themes grow within topics, and whether and where the
papers were published. Results come as CSV tables, a Markdown report and an interactive HTML
report.

Papers are selected in two stages: a deliberately wide keyword net finds candidates, then the
original keyword rule, a classifier trained on hand labels, or a language model decides which
candidates are relevant. Hand-labelled samples measure how well each stage works.

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

Then find the candidates and analyse them:

```bash
informatics filter --set cs
```

```bash
informatics analyze --set cs
```

Open `results/cs/report.html` (or `report.md`). Without further steps the papers are those that
meet the original keyword rule. To use a classifier instead, label a sample first (see
[How papers are selected](#how-papers-are-selected)):

```bash
informatics label-sample --set cs
```

```bash
informatics classify --set cs
```

Use `--set` more than once (or `--set cs,q-bio`) to analyse sets together; papers cross-listed
in both are counted once, and the reports compare the sets. Every command has `--help`, and
`python -m informatics …` works without the console script.

## Recent years only

To analyse, say, the five years 2021–2025, download only what changed since the start of the
window and keep only papers first submitted inside it:

```bash
informatics harvest --set cs --from 2021-01-01
```

```bash
informatics filter --set cs --years 2021-2025
```

`--from` selects records by the date they last *changed*, so it also brings older papers that
were revised or got a journal reference since then; `--years` drops those (and anything from
later years) before anything else is counted. Do not add `--until`: a paper submitted in 2025
and revised in 2026 would be lost. The reports state the window, and rising and falling phrases
compare periods inside it. To include the current year so far, use `--years 2021-2026`; its
share-based numbers are comparable, its paper count is not yet complete.

## Pipeline

```
harvest | update | import-kaggle | import-legacy-csv        ->  data/raw/<set>.parquet
import-pubmed | import-pubmed-files | import-openalex         ->  data/raw/<set>.parquet
                                                                 data/raw/<set>_import.json (how it was made)
filter          wide keyword net                             ->  data/interim/<set>_candidates.parquet
                                                                 data/interim/<set>_ai_only_sample.parquet
                                                                 data/interim/<set>_filter_summary.json
label-sample / label-evaluate                                ->  labels/  (hand labels; quality of each stage)
classify        classifier or language model (optional)      ->  data/interim/<set>_relevance.parquet
enrich          OpenAlex publication data (optional)         ->  data/interim/<set>_openalex.parquet
embed           sentence embeddings (optional)               ->  data/interim/embeddings/<model>.parquet
analyze                                                      ->  results/<set>/report.html, report.md, tables
label-themes-sample / label-themes-evaluate                  ->  labels/  (quality of the themes)
```

Parquet is the working format: it keeps arXiv IDs as text (`0708.2931`, not `708.2931`) and lists
as lists. CSV files are only written for reading in a spreadsheet, with list columns encoded as
JSON. `informatics to-csv <file.parquet>` converts any intermediate file.

| Command | What it does |
|---|---|
| `harvest --set S` | Downloads set S over OAI-PMH. Each page is saved as it arrives, so re-running continues an interrupted harvest. Requests time out, are retried a bounded number of times, and honour `Retry-After`. |
| `update --set S` | Harvests only the records changed since the newest one in `data/raw/S.parquet` and merges them in (changed papers replaced, new ones added). Run it monthly, then re-run the steps below (or `dvc repro`). |
| `list-sets`, `count --set S` | The set names the OAI-PMH endpoint offers; a set's article count from the arXiv search API. |
| `import-kaggle FILE --set S` | Builds `data/raw/S.parquet` from the Kaggle snapshot (cross-listed papers included, like the OAI sets). |
| `import-legacy-csv FILE --set S` | Converts the old harvester's CSV, repairing IDs, garbled accents and list columns. |
| `import-pubmed --set S` | A **sample** of PubMed through E-utilities: up to `--per-year` records per publication year matching `--query`. |
| `import-pubmed-files FILES --set S` | **All** PubMed records with an AI MeSH heading or AI keyword, from the annual baseline (and update) files. |
| `import-openalex --set S` | OpenAlex works matching `--filter` (default: AI topics in medicine), journals and preprints alike. |
| `filter --set S` | Excludes withdrawn and non-English papers, finds candidates with the wide keyword net, flags near-duplicates, and keeps a reproducible random sample of AI-only papers for comparison. `--years 2021-2025` keeps only papers from those years. |
| `label-sample` / `label-evaluate` | Draw papers to hand-label; measure the keyword rule, the wide net and the classifier against the labels. |
| `classify --set S` | Decide which candidates are relevant: `--method model` trains a classifier on the labels; `--method llm` asks a language model. |
| `label-themes-sample` / `label-themes-evaluate` | Measure the themes (keyword or language-model) against hand labels. |
| `enrich --set S` | Looks the selected papers up in OpenAlex: venue, institution types, countries, citations. |
| `embed --set S` | Caches sentence embeddings of the selected papers (needs `pip install -e ".[embeddings]"`). |
| `analyze --set S [--set T]` | Everything under [What the analysis computes](#what-the-analysis-computes). Options include `--selection`, `--themes llm`, `--recent 2023-2025 --earlier 2019-2022` and `--embeddings MODEL`. |

## How papers are selected

`src/informatics/keywords.py` holds every keyword list, so the rules can be changed without
touching code. Terms match whole words, including plural, hyphenated and spelling variants:
`patients?`, `neural net(work)s?`, `LLMs` / `large language models`, `deep-learning`,
`health-care`. Text is matched after TeX and odd characters are cleaned up.

**First stage: the wide net.** A paper is a *candidate* when its title or abstract has an AI
term and a medicine term from the original lists (`AI_TERMS`, `MEDICINE_TERMS`) or the wider
ones (`AI_CONTEXT_TERMS`: convolutional, U-Net, random forests, transformers, BERT, …;
`MEDICINE_CONTEXT_TERMS`: MRI, sepsis, clinicians, radiology, cancer, …). The wide lists favour
recall: a wrong match costs only some classification, a missed paper is lost for good. Words
with a common unrelated sense count only in phrases that fix the sense ("ischemic stroke", not
"pen strokes"). Every paper falls in one stratum:

| Stratum | Meaning |
|---|---|
| `strict` | Candidate that meets the original keyword rule (at least `--min-ai-hits` AI and `--min-medicine-hits` medicine keywords) |
| `wide_only` | Other candidate, found only by the wider lists |
| `ai_only`, `medicine_only`, `neither` | Not a candidate; the comparison sample comes from `ai_only` papers with an original AI keyword |
| `excluded` | Withdrawn, or not written in English (fewer than half the words are English; checked for candidates and comparison papers) |

Near-duplicates (the same paper posted under two IDs; MinHash on word 5-grams, Jaccard ≥ 0.8)
stay in the candidates file with `duplicate_of` set and are left out of everything downstream.

**Second stage: which candidates are relevant.** `analyze`, `enrich`, `embed` and the theme
labelling take `--selection`:

- `keywords`: the original rule (stratum `strict`);
- `relevance`: the decisions `informatics classify` wrote;
- `auto` (default): `relevance` when `classify` has run, otherwise `keywords`.

The reports say which was used.

**Measuring the stages.** Keyword rules and classifiers make mistakes both ways; only hand
labels show how many.

1. `informatics label-sample --set cs` writes `labels/cs_to_label.csv`, a random sample of
   strict candidates, other candidates and each kind of non-candidate, in random order, and a
   key file (`labels/cs_key.csv`) to keep closed while labelling. `--extend` adds papers to an
   existing sheet without touching its labels.
2. Mark each paper's `relevant` column 1 (about AI in medicine) or 0. A second labeller can fill
   `relevant_2` for some papers to measure agreement between labellers.
3. `informatics label-evaluate --set cs` reports precision and recall of the keyword rule, the
   recall of the wide net, and precision, recall and Cohen's kappa of each classifier, with 95%
   intervals. Each stratum's labelled papers are scaled up to the stratum's size; intervals for
   recall come from a bootstrap within strata.

Commit the labels: they are the evidence for the selection's quality.

**A trained classifier** (`informatics classify --set cs`, `relevance.py`): logistic regression
on TF-IDF of the normalised words and phrases of title and abstract, plus the keyword-rule
flag, with balanced class weights. Stratified cross-validation gives every labelled paper a
score from a model that did not see its label; the decision threshold maximises F1 with each
labelled paper weighted by its stratum's sampling rate, and those cross-validated predictions
are what `label-evaluate` measures. Hand labels override the classifier for labelled papers.
About 1,000 labels give a dependable classifier; it needs at least 5 relevant and 5 irrelevant.

**A language model** (`informatics classify --set cs --method llm`, `llm.py`) reads each
candidate against a written rubric (`llm.RUBRIC`; `--rubric FILE` replaces it) and answers in a
fixed JSON form: relevant or not, the themes the paper discusses (`llm.THEME_DEFINITIONS`), and a
one-sentence rationale. It needs `pip install -e ".[llm]"` and `ANTHROPIC_API_KEY`; the model is
`claude-opus-5-5` unless `--llm-model` says otherwise. Requests go through the Message Batches
API, which costs half as much as single requests and usually finishes within an hour; runs
resume, and only papers whose text, model or rubric changed are sent again. Suggested order:

```bash
informatics classify --set cs --method llm --dry-run          # what would be sent, and one request
```

```bash
informatics classify --set cs --method llm --labelled-only --wait   # the labelled papers only
```

```bash
informatics label-evaluate --set cs                          # check the model against your labels
```

```bash
informatics classify --set cs --method llm --wait            # then every candidate
```

Refusals and invalid requests are recorded and fall back to the keyword rule (counted in
`data/interim/<set>_relevance.json`); `--retry-failed` sends them again.

## Themes and concepts

The secondary **themes** are the original terms (adoption, privacy, bias, regulation, …) plus
variants that keep the meaning ("personalised", "ethics", "regulatory"). Some themes have
**exclusions** for unrelated senses, such as "gene regulation", "MRI bias field" or "lung
compliance". `accuracy` appears in almost every machine-learning abstract, so treat it as a
baseline. With `analyze --themes llm`, the language model's themes replace the keyword matches.

To measure the themes: `label-themes-sample --set cs` draws papers tagged with each theme (mark
whether each really discusses it; `label-themes-evaluate` reports precision), and
`label-themes-sample --set cs --random 200` draws papers at random to tag with every theme they
discuss, which measures precision **and recall** of the keyword themes and of the language
model's themes.

**Concepts** (`concepts.py`) tag each paper's disease area, data type, clinical task and AI
method with curated patterns.

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
  **distinctive** (selected papers against the AI-only sample), **rising** and **falling** (the
  last three years against the four before, or `--recent`/`--earlier`), and **between sets**.
- **Topic clusters.** TF-IDF words and two-word phrases without the selection keywords, reduced
  with LSA and clustered with k-means; k is chosen by silhouette (each candidate k scored with 3
  restarts, the chosen one refitted with 10) and its stability reported. `--embeddings MODEL`
  clusters on sentence embeddings instead. Each cluster gets its distinctive terms, its most
  common concepts and its typical papers; a t-SNE map places the papers in two dimensions.
  Sets too small to cluster are reported as one group.
- **Themes and concepts.** Each tag's share of papers, with 95% Wilson intervals, overall and by
  year, cluster and set.
- **Trends within topics** (`trends.py`). For each theme and concept, a logistic regression of
  "the paper has the tag" on its date (odds ratio per year), and the same adjusted for the topic
  cluster (and set). A theme can grow because papers within each topic mention it more, or
  because topics that always mentioned it grow faster; when the adjusted odds ratio is much
  closer to 1, it is the second. The tables also give the within-topic change in percentage
  points per year and false-discovery-rate q-values. Cluster effects have a weakly informative
  prior, so a topic without the tag cannot make the fit diverge.
- **Publication.** Evidence of publication from arXiv's journal reference and comments; with
  `enrich`, also venue type, healthcare and company authors, countries and citations.

`report.html` shows all of this as charts (papers per year, themes by year with intervals, a
forest plot of the trends, concepts, rising and falling phrases, the topic map) with a table
view under each chart, plus a searchable, filterable table of the papers. It is one
self-contained file that works offline, in light and dark themes.

## Population and dates

What a set's papers are a sample of decides which comparisons make sense, so every import
records how it was made (`data/raw/<set>_import.json`), and the reports state it per set:

- **arXiv** (`harvest`, `import-kaggle`): every paper listed in the archive, cross-lists included.
  Dates are the month of first submission, from the arXiv identifier.
- **PubMed sample** (`import-pubmed`): at most `--per-year` records per year (ESearch returns at
  most 10,000 per query). Counts per year are capped, so they cannot show growth.
- **PubMed in full** (`import-pubmed-files`): download the annual baseline (about 1,300 files)
  from https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ (and the update files if wanted), then:

  ```bash
  informatics import-pubmed-files "downloads/pubmed26n*.xml.gz" --set pubmed
  ```

  This keeps every record with an AI MeSH heading (`--mesh` to change them) or an AI keyword
  (`--keywords none|strict|wide`); MeSH indexing lags by months, which is why keywords count too.
  Update files are read first, so revised records replace their baseline versions and deleted
  citations are dropped.
- **OpenAlex** (`import-openalex`): every work matching the filter. Needs a free API key.

PubMed dates are the earliest date in the record (received, published online, or in print), the
closest equivalent to arXiv's first submission; OpenAlex dates are publication dates.

## Reproducibility

- `run_info.json` (next to each report) records the settings, the package versions, the git
  commit, and the size, row count and SHA-256 hash of every input file. The filter summary does
  the same for the raw file, and the classifier's record for the labels it was trained on.
- `dvc.yaml` and `params.yaml` describe the pipeline for [DVC](https://dvc.org), which re-runs
  exactly the steps whose inputs or settings changed and records hashes in `dvc.lock`:

  ```bash
  pip install dvc && dvc init
  ```

  ```bash
  dvc add data/raw/cs.parquet data/raw/q-bio.parquet
  ```

  ```bash
  dvc repro
  ```

- `informatics update --set cs` (monthly, say) fetches only what changed since the last harvest;
  `dvc add` the raw file again and `dvc repro` refreshes the results.

## Notes on the data sources

- arXiv rewrote its OAI-PMH service in 2025. The default endpoint is `https://oaipmh.arxiv.org/oai`;
  pass `--base-url` to use another one. Run `informatics list-sets` to check set names. The
  harvester waits 5 s between requests (`--delay`).
- `harvest --from/--until` filter on the date a record last *changed*, not its submission date.
  Without them the whole set is downloaded.
- The old `All_CS.csv` kept only surnames; `import-legacy-csv` cannot restore first names. A fresh
  harvest or the Kaggle snapshot has full names.
- **OpenAlex** has required a free API key since February 2026 (openalex.org/settings/api). Pass
  `--api-key` or set `OPENALEX_API_KEY`. `enrich` looks papers up by journal DOI, or else by
  arXiv's own DOI (10.48550/arXiv.<id>).
- **PubMed**: an NCBI API key (`--api-key` or `NCBI_API_KEY`) allows faster E-utilities requests.
- **Embeddings** download the model from Hugging Face the first time it is needed. The default
  is `sentence-transformers/all-MiniLM-L6-v2`; `allenai/specter2_base` was trained on scientific
  papers.

## Working in VS Code

The repository includes VS Code settings (`.vscode/`). Open the folder (`code .`, or File →
Open Folder), accept the recommended extensions (Python, Python Debugger, Ruff, Data Wrangler),
and create the environment in the integrated terminal (Terminal → New Terminal):

```bash
python3 -m venv .venv
```

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

Then pick `.venv` with **Python: Select Interpreter** (Ctrl+Shift+P) if VS Code has not done so;
new terminals then activate it automatically.

- **Terminal → Run Task** lists the pipeline steps (harvest, filter with a year window,
  analyze, open the HTML report, labelling, classify, update, tests). Each asks for the set and
  dates, with 2021–2025 filled in. A harvest takes hours and stops when VS Code closes; running
  the task again continues it.
- The **Testing** panel (flask icon) runs and debugs the tests.
- **Run and Debug** → "informatics (asks for the arguments)" runs any command under the
  debugger, e.g. `filter --set cs --years 2021-2025`.
- Files are formatted with Ruff on save, matching the CI checks. CSV and Parquet files open in
  Data Wrangler (right-click → Open in Data Wrangler).

## Development

```bash
make test
```

```bash
make lint
```

Tests use a synthetic corpus (`tests/synthetic.py`), 17 real arXiv records in their original
legacy form (`tests/fixtures/real_arxiv_sample.csv`) and recorded responses, so they need no
network access. Golden files in `tests/golden/` pin the main output tables; after an intended
change, run `UPDATE_GOLDEN=1 pytest tests/test_golden.py tests/test_real_data.py` and review
the diff. `INFORMATICS_LIVE=1 pytest -m live` checks the real arXiv, PubMed and OpenAlex
services with a few requests each.

CI runs ruff and pytest on Python 3.10 and 3.13 for every push, plus a job with the oldest
supported dependency versions (`ci/minimum-versions.txt`); the live tests run on demand
(Actions → CI → Run workflow) and upload the records they fetched.

```
src/informatics/   the package (one module per step; keywords.py and concepts.py hold the term lists)
tests/             pytest suite, fixtures and golden files
results/           analysis output; results/legacy/ holds the old, known-faulty output
misc/              stand-alone scripts unrelated to this pipeline (see misc/README.md)
data/, labels/     created when you run the pipeline (data/ is git-ignored)
```
