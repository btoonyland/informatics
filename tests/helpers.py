"""Fake HTTP objects so the network code can be tested without arXiv."""

from __future__ import annotations

from pathlib import Path

from informatics.http_client import HttpClient

FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, status: int = 200, content: bytes = b"", headers: dict | None = None):
        self.status_code = status
        self.content = content
        self.headers = headers or {"Content-Type": "text/xml"}  # no charset, like the bug case
        self.url = "https://example.test/oai"
        # What requests' .text gives for text/xml without a charset: ISO-8859-1 decoding.
        self.text = content.decode("latin-1")


class FakeSession:
    """Replays a script of responses (or exceptions) and records the params of each request."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params or {}))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def ok(name_or_bytes: str | bytes) -> FakeResponse:
    content = name_or_bytes if isinstance(name_or_bytes, bytes) else fixture_bytes(name_or_bytes)
    return FakeResponse(200, content)


def client_for(script: list, sleeps: list | None = None, **kwargs) -> HttpClient:
    return HttpClient(
        session=FakeSession(script),
        sleep=(sleeps.append if sleeps is not None else lambda _seconds: None),
        **kwargs,
    )


def oai_error(code: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
        f'<error code="{code}">message for {code}</error></OAI-PMH>'
    ).encode()
