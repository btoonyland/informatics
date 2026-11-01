# Miscellaneous scripts

These scripts are **not part of the arXiv analysis pipeline**. They came from other projects and
are kept here until they move to repositories of their own. Each runs on its own; see
`python misc/<script>.py --help`.

| Script | What it does | Previously |
|---|---|---|
| `greentech_news.py` | Lists green-tech stories from an RSS feed (default: Quanta Magazine) | `scripts/sn_api.py` |
| `explore_dataset.py` | Counts a CSV column's values and saves the rows with one value | `scripts/explore_dataset.py` |
| `pdf_extract.py` | Extracts text and metadata from a folder of PDFs into JSON Lines | `scripts/pdfProc.py` |

What changed:

- **greentech_news.py**: keywords match whole words, so "wind" no longer matches "window".
  Ambiguous single words ("energy", "solar") were replaced by phrases. It also adds a request
  timeout, a `--feed`/`--keyword`/`--limit` interface, and lists which keywords matched.
- **explore_dataset.py**: the file, column and value are arguments instead of being hard-coded,
  and the stale tutorial comments are gone.
- **pdf_extract.py**: removed the first block, which was copied from another script and crashed
  on undefined names. It uses `pypdf` instead of the deprecated `PyPDF2`, overwrites its output
  instead of appending to it (re-runs used to duplicate everything), and drops the 0.5 s pause
  per file. It writes one JSON record per PDF instead of two concatenated text files.
