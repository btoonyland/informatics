"""Count the articles matching an arXiv API query (replaces scripts/check_num_files.py)."""

from __future__ import annotations

from informatics.http_client import HttpClient, HttpError

API_URL = "https://export.arxiv.org/api/query"
OPENSEARCH = {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}


def default_query(set_name: str) -> str:
    """The query the original script used, e.g. ``cat:q-bio*`` for all q-bio categories."""
    return f"cat:{set_name}*"


def count_articles(query: str, client: HttpClient | None = None, api_url: str = API_URL) -> int:
    """All-time number of articles matching ``query`` (the API's ``totalResults``).

    Note: this counts by category query, while OAI-PMH sets are defined by arXiv; the two
    numbers need not agree exactly.
    """
    client = client or HttpClient()
    params = {"search_query": query, "start": 0, "max_results": 0}
    root = client.get_xml(api_url, params)  # retries when the response is not valid XML
    total = root.find("opensearch:totalResults", OPENSEARCH)
    if total is None or not (total.text or "").strip().isdigit():
        raise HttpError(f"no totalResults in the API response for {query!r}")
    return int(total.text.strip())
