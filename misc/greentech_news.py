"""Print recent green-technology stories from an RSS feed.

Not part of the arXiv pipeline in this repository (see misc/README.md).

    python misc/greentech_news.py                        # Quanta Magazine, 5 stories
    python misc/greentech_news.py --limit 10 --feed URL --keyword hydrogen --keyword battery

Quanta Magazine is a general-science magazine, so only a few of its stories are about green
technology. Keywords are matched as whole words or phrases ("wind" no longer matches "window"),
and the ambiguous single words of the first version ("energy", "solar", "wind", "environment")
were replaced by phrases that avoid "dark energy", "solar system" or "learning environment".
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from html import unescape

import requests

DEFAULT_FEED = "https://api.quantamagazine.org/feed/"
DEFAULT_KEYWORDS = [
    "greentech", "green tech", "climate", "sustainability", "environmental",
    "renewable energy", "clean energy", "energy storage", "energy transition",
    "solar power", "solar energy", "solar panel", "solar cell", "photovoltaic",
    "wind power", "wind energy", "wind turbine", "wind farm",
]  # fmt: skip


def build_pattern(keywords: list[str]) -> re.Pattern[str]:
    """Whole-word match for any keyword, optionally followed by a plural "s"."""
    alternatives = sorted((re.escape(k.lower()) for k in keywords), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")s?\b", re.IGNORECASE)


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", unescape(text or ""))


def fetch_items(feed_url: str, timeout: float = 30) -> list[dict[str, str]]:
    response = requests.get(feed_url, timeout=timeout, headers={"User-Agent": "greentech-news"})
    response.raise_for_status()
    root = ET.fromstring(response.content)  # bytes, so the feed's own encoding is used
    fields = ("title", "description", "link", "pubDate")
    return [
        {f: (item.findtext(f) or "").strip() for f in fields}
        for item in root.iterfind("./channel/item")
    ]


def matching_items(items: list[dict], pattern: re.Pattern[str], limit: int) -> list[dict]:
    matches = []
    for item in items:
        text = f"{item['title']} {strip_html(item['description'])}"
        found = sorted({m.group(0).lower() for m in pattern.finditer(text)})
        if found:
            matches.append({**item, "matched": found})
            if len(matches) >= limit:
                break
    return matches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List green-tech stories from an RSS feed.")
    parser.add_argument("--feed", default=DEFAULT_FEED)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--keyword", action="append", help="replaces the default keywords; repeatable"
    )
    args = parser.parse_args(argv)
    try:
        items = fetch_items(args.feed)
    except (requests.RequestException, ET.ParseError) as exc:
        print(f"Could not read {args.feed}: {exc}", file=sys.stderr)
        return 1
    matches = matching_items(items, build_pattern(args.keyword or DEFAULT_KEYWORDS), args.limit)
    if not matches:
        print(f"No green-tech stories among the {len(items)} items in the feed.")
    for item in matches:
        date = item["pubDate"][:16] or "unknown date"
        print(f"[{date}] {item['title']}\n   -> {item['link']}")
        print(f"   matched: {', '.join(item['matched'])}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
