"""Resumable OAI-PMH harvester for arXiv metadata.

Each response page is written to its own Parquet file as soon as it arrives, and the resumption
token is saved next to it, so an interrupted harvest continues where it stopped instead of losing
hours of downloads. When the list is complete the pages are merged into one file.
"""

from __future__ import annotations

import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from informatics.http_client import HttpClient
from informatics.records import RAW_SCHEMA, clean_text, normalize_records
from informatics.storage import read_json, write_json, write_parquet

log = logging.getLogger(__name__)

# arXiv rewrote its OAI-PMH service in 2025; the old address was http://export.arxiv.org/oai2.
DEFAULT_BASE_URL = "https://oaipmh.arxiv.org/oai"
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "arxiv": "http://arxiv.org/OAI/arXiv/"}
PAGE_COLUMNS = [c for c in RAW_SCHEMA.names if c not in ("source", "year", "month")]
PAGE_SCHEMA = pa.schema([RAW_SCHEMA.field(c) for c in PAGE_COLUMNS])


class HarvestError(RuntimeError):
    pass


class OaiError(HarvestError):
    def __init__(self, code: str, message: str):
        super().__init__(f"OAI-PMH error {code}: {message}")
        self.code = code


def _text(element: ET.Element, path: str) -> str | None:
    found = element.find(path, NS)
    return clean_text(found.text) if found is not None else None


def parse_record(record: ET.Element) -> dict | None:
    """One ``<record>`` in the ``arXiv`` metadata format -> dict (``None`` for deleted records)."""
    header = record.find("oai:header", NS)
    if header is None or header.get("status") == "deleted":
        return None
    meta = record.find("oai:metadata/arxiv:arXiv", NS)
    if meta is None:
        return None
    authors = []
    for author in meta.findall("arxiv:authors/arxiv:author", NS):
        parts = (_text(author, f"arxiv:{p}") for p in ("forenames", "keyname", "suffix"))
        name = " ".join(p for p in parts if p)
        if name:
            authors.append(name)
    return {
        "id": _text(meta, "arxiv:id"),
        "title": _text(meta, "arxiv:title"),
        "abstract": _text(meta, "arxiv:abstract"),
        "authors": authors,
        "categories": (_text(meta, "arxiv:categories") or "").split(),
        "created": _text(meta, "arxiv:created"),
        "updated": _text(meta, "arxiv:updated"),
        "doi": _text(meta, "arxiv:doi"),
        "journal_ref": _text(meta, "arxiv:journal-ref"),
        "comments": _text(meta, "arxiv:comments"),
        "license": _text(meta, "arxiv:license"),
        "datestamp": _text(header, "oai:datestamp"),
    }


def parse_page(root: ET.Element) -> tuple[list[dict], str | None, dict[str, str]]:
    """Return (records, resumption token or None, token attributes) for one ListRecords page."""
    error = root.find("oai:error", NS)
    if error is not None:
        code = error.get("code", "unknown")
        if code == "noRecordsMatch":
            return [], None, {}
        raise OaiError(code, (error.text or "").strip())
    records = [r for r in map(parse_record, root.iterfind(".//oai:record", NS)) if r is not None]
    token_el = root.find(".//oai:resumptionToken", NS)
    token = (token_el.text or "").strip() if token_el is not None else ""
    return records, token or None, dict(token_el.attrib) if token_el is not None else {}


def harvest(
    set_spec: str,
    out_dir: str | Path,
    *,
    base_url: str = DEFAULT_BASE_URL,
    metadata_prefix: str = "arXiv",
    date_from: str | None = None,
    date_until: str | None = None,
    delay: float = 5.0,
    restart: bool = False,
    max_pages: int | None = None,
    client: HttpClient | None = None,
) -> dict:
    """Harvest ``set_spec`` into ``out_dir/pages``; returns the saved state.

    Re-running with the same arguments resumes an interrupted harvest. ``date_from`` and
    ``date_until`` filter on the OAI datestamp (last metadata change), not the submission date.
    """
    out_dir = Path(out_dir)
    pages_dir, state_path = out_dir / "pages", out_dir / "state.json"
    params = {
        "base_url": base_url,
        "set": set_spec,
        "metadataPrefix": metadata_prefix,
        "from": date_from,
        "until": date_until,
    }
    if restart and out_dir.exists():
        shutil.rmtree(out_dir)
    state = read_json(state_path) if state_path.exists() else None
    if state is not None and state["params"] != params:
        raise HarvestError(
            f"{out_dir} holds a harvest with different settings ({state['params']}); "
            "pass --restart to discard it"
        )
    if state is None:
        state = {"params": params, "next_page": 1, "token": None, "records": 0, "complete": False}
    if state["complete"]:
        log.info("harvest of %s is already complete (%d records)", set_spec, state["records"])
        return state

    client = client or HttpClient()
    pages_this_run = 0
    while True:
        if state["token"]:
            query = {"verb": "ListRecords", "resumptionToken": state["token"]}
        else:
            query = {"verb": "ListRecords", "metadataPrefix": metadata_prefix, "set": set_spec}
            query.update({k: v for k, v in (("from", date_from), ("until", date_until)) if v})
        try:
            records, token, token_info = parse_page(client.get_xml(base_url, query))
        except OaiError as exc:
            if exc.code == "badResumptionToken":
                raise HarvestError(
                    "the server no longer accepts the saved resumption token (it has expired); "
                    "re-run with --restart to start the harvest again"
                ) from exc
            raise
        page = state["next_page"]
        page_df = pd.DataFrame.from_records(records, columns=PAGE_COLUMNS)
        write_parquet(page_df, pages_dir / f"page-{page:06d}.parquet", schema=PAGE_SCHEMA)
        state.update(
            next_page=page + 1,
            token=token,
            records=state["records"] + len(records),
            complete=token is None,
        )
        write_json(state, state_path)  # written after the page, so a crash only repeats a page
        size = token_info.get("completeListSize")
        log.info(
            "page %d: %d records (%d so far%s)",
            page,
            len(records),
            state["records"],
            f" of {size}" if size else "",
        )
        pages_this_run += 1
        if state["complete"]:
            return state
        if max_pages is not None and pages_this_run >= max_pages:
            log.info("stopping after %d pages; run again to continue", pages_this_run)
            return state
        client.sleep(delay)  # be polite to arXiv between requests


def consolidate(out_dir: str | Path, dest: str | Path) -> pd.DataFrame:
    """Merge harvested pages into one canonical Parquet file, one row per paper."""
    pages = sorted((Path(out_dir) / "pages").glob("page-*.parquet"))
    if not pages:
        raise HarvestError(f"no harvested pages in {out_dir}")
    table = pa.concat_tables([pq.read_table(p, schema=PAGE_SCHEMA) for p in pages])
    df = normalize_records(table.to_pandas(), source="oai-pmh")
    # A record changed while the harvest was running can be listed twice: keep the newest.
    df = df.sort_values("datestamp", kind="stable", na_position="first")
    df = df.drop_duplicates("id", keep="last").sort_values("id").reset_index(drop=True)
    write_parquet(df, dest, schema=RAW_SCHEMA)
    log.info("wrote %d records to %s", len(df), dest)
    return df


def list_sets(base_url: str = DEFAULT_BASE_URL, client: HttpClient | None = None) -> list[tuple]:
    """(setSpec, setName) pairs offered by the endpoint, to check which ``--set`` values exist."""
    client = client or HttpClient()
    sets, query = [], {"verb": "ListSets"}
    while True:
        root = client.get_xml(base_url, query)
        error = root.find("oai:error", NS)
        if error is not None:
            raise OaiError(error.get("code", "unknown"), (error.text or "").strip())
        for s in root.iterfind(".//oai:set", NS):
            sets.append((_text(s, "oai:setSpec"), _text(s, "oai:setName")))
        token_el = root.find(".//oai:resumptionToken", NS)
        if token_el is None or not (token_el.text or "").strip():
            return sets
        query = {"verb": "ListSets", "resumptionToken": token_el.text.strip()}
