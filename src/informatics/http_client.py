"""HTTP access with timeouts, bounded retries and arXiv's ``Retry-After`` flow control."""

from __future__ import annotations

import email.utils
import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

from informatics import __version__

log = logging.getLogger(__name__)

USER_AGENT = f"informatics/{__version__} (arXiv metadata harvester)"
RETRY_STATUS = {429, 500, 502, 503, 504}


class HttpError(RuntimeError):
    pass


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Seconds to wait from a ``Retry-After`` header (either seconds or an HTTP date)."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return max(0.0, (when - now).total_seconds())


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


@dataclass
class HttpClient:
    session: requests.Session = field(default_factory=new_session)
    timeout: float = 120.0
    max_retries: int = 8
    backoff: float = 10.0  # first wait in seconds; doubles on every retry
    max_wait: float = 900.0
    sleep: Callable[[float], None] = time.sleep

    def get(self, url: str, params: dict[str, Any] | None = None) -> bytes:
        """GET ``url`` and return the raw bytes (never ``response.text``: see :meth:`get_xml`)."""
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                reason, retry_after = f"network error ({exc})", None
            else:
                if response.status_code == 200:
                    return response.content
                if response.status_code not in RETRY_STATUS:
                    raise HttpError(
                        f"HTTP {response.status_code} from {response.url}: {response.text[:300]}"
                    )
                reason = f"HTTP {response.status_code}"
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
            if attempt == self.max_retries:
                raise HttpError(f"giving up on {url} after {self.max_retries} retries ({reason})")
            wait = retry_after if retry_after is not None else self.backoff * 2**attempt
            wait = min(wait, self.max_wait)
            log.warning(
                "%s; retrying in %.0f s (%d/%d)", reason, wait, attempt + 1, self.max_retries
            )
            self.sleep(wait)
        raise AssertionError("unreachable")

    def get_xml(
        self, url: str, params: dict[str, Any] | None = None, parse_retries: int = 3
    ) -> ET.Element:
        """GET and parse XML, retrying if the server returns something that is not XML.

        The XML is parsed from bytes so the parser follows the document's own encoding
        declaration. Parsing ``response.text`` instead lets ``requests`` guess the encoding,
        and for ``text/xml`` without a charset it guesses ISO-8859-1, garbling UTF-8 names.
        """
        for attempt in range(parse_retries + 1):
            content = self.get(url, params)
            try:
                return ET.fromstring(content)
            except ET.ParseError as exc:
                if attempt == parse_retries:
                    raise HttpError(
                        f"invalid XML from {url}: {exc}; starts {content[:200]!r}"
                    ) from exc
                log.warning("invalid XML (%s); retrying in %.0f s", exc, self.backoff)
                self.sleep(self.backoff)
        raise AssertionError("unreachable")
