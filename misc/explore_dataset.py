"""Show how often each value of a CSV column occurs, and optionally save the rows with one value.

Not part of the arXiv pipeline (see misc/README.md). The original, hard-coded use was:

    python misc/explore_dataset.py Datasets/all_files_ER_fulltexts_robotsenergy.csv \
        --column type --value CONSUMER_GOOD --out filtered_consumer_goods_robotsenergy.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def select_rows(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    return df[df[column] == value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explore one column of a CSV file.")
    parser.add_argument("csv")
    parser.add_argument("--column", default="type")
    parser.add_argument("--value", help="save the rows where the column equals this value")
    parser.add_argument("--out", help="output CSV (default: filtered_<value>.csv)")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    if args.column not in df.columns:
        print(f"No column {args.column!r}; columns: {', '.join(df.columns)}", file=sys.stderr)
        return 1
    print(df[args.column].value_counts().to_string())
    if args.value is not None:
        rows = select_rows(df, args.column, args.value)
        out = args.out or f"filtered_{args.value.lower()}.csv"
        rows.to_csv(out, index=False)
        print(f"Saved {len(rows)} rows where {args.column} == {args.value!r} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
