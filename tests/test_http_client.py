from datetime import datetime, timezone

import pytest
import requests
from helpers import FakeResponse, client_for

from informatics.http_client import HttpError, parse_retry_after

UTF8_XML = '<?xml version="1.0" encoding="UTF-8"?><root><name>Zukić Müller</name></root>'.encode()


def test_xml_is_parsed_from_bytes_not_guessed_text():
    """The old harvester parsed response.text, which requests decodes as ISO-8859-1 for text/xml
    without a charset, garbling names. Parsing the bytes must give the real characters."""
    response = FakeResponse(200, UTF8_XML)
    assert "Ã" in response.text or "Ä" in response.text  # the bug the old code had
    root = client_for([response]).get_xml("https://example.test", {})
    assert root.findtext("name") == "Zukić Müller"


def test_retries_on_503_honouring_retry_after():
    sleeps = []
    client = client_for(
        [FakeResponse(503, headers={"Retry-After": "7"}), FakeResponse(200, b"ok")], sleeps
    )
    assert client.get("https://example.test") == b"ok"
    assert sleeps == [7.0]


def test_backoff_doubles_on_network_errors_and_gives_up():
    sleeps = []
    error = requests.ConnectionError("connection reset")
    client = client_for([error, error, error], sleeps, max_retries=2, backoff=10)
    with pytest.raises(HttpError, match="after 2 retries"):
        client.get("https://example.test")
    assert sleeps == [10, 20]


def test_client_errors_are_not_retried():
    sleeps = []
    client = client_for([FakeResponse(404, b"not found")], sleeps)
    with pytest.raises(HttpError, match="HTTP 404"):
        client.get("https://example.test")
    assert sleeps == []


def test_invalid_xml_is_retried():
    sleeps = []
    client = client_for([FakeResponse(200, b"<html>busy"), FakeResponse(200, UTF8_XML)], sleeps)
    assert client.get_xml("https://example.test").findtext("name") == "Zukić Müller"
    assert len(sleeps) == 1


def test_parse_retry_after():
    now = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after("30") == 30.0
    assert parse_retry_after("Sun, 01 Nov 2026 12:01:00 GMT", now=now) == 60.0
    assert parse_retry_after("Sun, 01 Nov 2026 11:00:00 GMT", now=now) == 0.0
    assert parse_retry_after("soon") is None
    assert parse_retry_after(None) is None
