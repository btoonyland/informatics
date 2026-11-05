"""What each set's papers are a sample of, and what its dates mean.

Every import records how it was made in ``data/raw/<set>_import.json`` (the source, the query
or filter, the files and options); the filter copies it into its summary, and the reports state
each set's population from it. Comparing sets is only meaningful with this in view: an arXiv
set holds every paper of an archive, ``import-pubmed`` holds a capped sample per year, and
``import-pubmed-files`` holds every PubMed record that matches the AI definition.
"""

from __future__ import annotations

POPULATIONS = {
    "oai-pmh": "every paper listed in the arXiv set, cross-lists included (OAI-PMH harvest)",
    "kaggle": "every paper in the arXiv set, cross-lists included (Cornell's Kaggle snapshot)",
    "legacy-csv": "arXiv papers from the first version's harvest (surnames only)",
    "pubmed-sample": (
        "a sample of PubMed: at most a fixed number of records per publication year that match "
        "the query, so counts per year are capped and not the size of the literature"
    ),
    "pubmed-baseline": (
        "every record in the PubMed files read that has an AI MeSH heading or an AI keyword"
    ),
    "openalex": "every OpenAlex work that matches the filter",
}
DATES = {
    "oai-pmh": "month of first submission (from the arXiv identifier)",
    "kaggle": "month of first submission (from the arXiv identifier)",
    "legacy-csv": "month of first submission (from the arXiv identifier)",
    "pubmed-sample": "earliest date in the record: received, published online or in print",
    "pubmed-baseline": "earliest date in the record: received, published online or in print",
    "openalex": "publication date",
    "synthetic": "the test corpus's own dates",
}
DETAILS = {  # import options worth stating, and how to say them
    "query": "query {}",
    "per_year": "at most {} records per year",
    "years": "publication years {}",
    "filter": "filter {}",
    "mesh": "MeSH headings: {}",
    "keywords": "keywords: {}",
    "files": "{} files",
    "set_spec": "set {}",
    "date_from": "records changed from {}",
    "date_until": "records changed until {}",
}


def describe(set_name: str, summary: dict | None) -> list[str]:
    """Sentences stating a set's population and the meaning of its dates."""
    if not summary:
        return [f"{set_name}: no filter summary, so the population is unknown."]
    sources = summary.get("sources") or {}
    info = summary.get("import") or {}
    out = []
    for source, n in sorted(sources.items(), key=lambda item: -item[1]):
        population = POPULATIONS.get(source, f"papers from {source}")
        details = [
            template.format(", ".join(map(str, value)) if isinstance(value, list) else value)
            for key, template in DETAILS.items()
            if (value := info.get(key)) not in (None, "", [])
        ]
        if years := (summary.get("config") or {}).get("years"):
            details.append(f"only papers from {years[0]}-{years[1]}")
        text = f"{set_name}: {n:,} papers, {population}"
        if details:
            text += " (" + "; ".join(details) + ")"
        out.append(text + f". Dates: {DATES.get(source, 'as given by the source')}.")
    return out or [f"{set_name}: no papers."]
