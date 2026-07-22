"""
Export all human corrections from DynamoDB to a CSV for retraining.

Output columns match the format label_correction_loop.py already consumes:
    text, label, manual_label, date, review_id, batch_id

Usage:
    python scripts/export_corrections.py
    python scripts/export_corrections.py --out path/to/corrections.csv
    python scripts/export_corrections.py --table MyCorrectionsTable
"""

import argparse
import csv
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

COLUMNS = ["text", "label", "manual_label", "date", "review_id", "batch_id"]


def scan_corrections(table_name: str, endpoint_url: str | None) -> list[dict]:
    kwargs = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    ddb = boto3.resource("dynamodb", **kwargs)
    table = ddb.Table(table_name)

    items: list[dict] = []
    scan_kwargs: dict = {}
    while True:
        resp = table.scan(**scan_kwargs)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DynamoDB corrections to CSV")
    parser.add_argument("--out", default="corrections_export.csv", help="Output CSV path")
    parser.add_argument("--table", default="Corrections", help="DynamoDB table name")
    parser.add_argument("--endpoint-url", default=None, help="Override DynamoDB endpoint (local dev)")
    args = parser.parse_args()

    print(f"Scanning table '{args.table}'…")
    items = scan_corrections(args.table, args.endpoint_url)

    if not items:
        print("No corrections found — nothing to export.")
        sys.exit(0)

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)

    print(f"Exported {len(items)} corrections → {out}")


if __name__ == "__main__":
    main()
