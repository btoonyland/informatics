"""Extract the text and metadata of every PDF in a folder into one JSON Lines file.

Not part of the arXiv pipeline (see misc/README.md). Needs the optional dependencies:
``pip install -e ".[pdf]"`` (pypdf, the maintained successor of PyPDF2, and pdfminer.six).

    python misc/pdf_extract.py /content/arxiv5 --out arxiv5.jsonl

Each line holds ``{"file", "pages", "metadata", "text"}``, or ``{"file", "error"}`` for a PDF
that could not be read, so text and metadata stay together and are easy to load again. The
output file is overwritten (``--append`` adds to it instead), so re-running never duplicates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def extract(path: Path) -> dict:
    from pdfminer.high_level import extract_text
    from pypdf import PdfReader

    reader = PdfReader(path)
    metadata = {str(k).lstrip("/"): str(v) for k, v in (reader.metadata or {}).items()}
    return {
        "file": path.name,
        "pages": len(reader.pages),
        "metadata": metadata,
        "text": extract_text(path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract text and metadata from PDFs.")
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("pdf_text.jsonl"))
    parser.add_argument("--append", action="store_true", help="add to --out instead of replacing")
    args = parser.parse_args(argv)
    try:
        import pdfminer  # noqa: F401
        import pypdf  # noqa: F401
    except ImportError:
        print('Missing dependencies: pip install -e ".[pdf]"', file=sys.stderr)
        return 1

    files = sorted(p for p in args.pdf_dir.iterdir() if p.suffix.lower() == ".pdf")
    print(f"Found {len(files)} PDFs in {args.pdf_dir}")
    failed = 0
    with open(args.out, "a" if args.append else "w", encoding="utf-8") as out:
        for path in files:
            try:
                record = extract(path)
            except Exception as exc:  # one damaged PDF should not stop the batch
                record = {"file": path.name, "error": f"{type(exc).__name__}: {exc}"}
                failed += 1
                print(f"Failed: {path.name}: {exc}", file=sys.stderr)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(files) - failed} documents to {args.out} ({failed} failed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
