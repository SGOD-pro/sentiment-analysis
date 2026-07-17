"""
One-time script: rebuild Aggregates table from Reviews data.

Usage: cd backend && uv run python scripts/rebuild_aggregates.py

Scans the Reviews table, groups by batch_id, and writes aggregate rows
under the new PK=batch_id SK=agg_type schema. Safe to run multiple times
(overwrites existing aggregate rows with fresh counts).
"""

import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, ".")

from config import get_settings
from database import get_tables
from logger import get_logger

log = get_logger(__name__)


def _week_key(date_str: str) -> str:
    try:
        dt = datetime.fromisoformat(date_str)
        return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
    except (ValueError, TypeError):
        return "unknown"


def rebuild():
    tables = get_tables()
    ts = datetime.now(timezone.utc).isoformat()

    # Scan all reviews
    all_reviews = []
    scan_kwargs = {}
    while True:
        resp = tables.reviews.scan(**scan_kwargs)
        all_reviews.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    log.info(f"Scanned {len(all_reviews)} reviews")

    # Group by batch_id
    by_batch: dict[str, list] = defaultdict(list)
    for r in all_reviews:
        by_batch[r["batch_id"]].append(r)

    for batch_id, reviews in by_batch.items():
        agg: dict[str, dict[str, int]] = {}

        for r in reviews:
            category = r.get("category", "")
            review_date = r.get("review_date", "")
            sentiment = r.get("sentiment", "unknown")
            week = _week_key(review_date) if review_date else "unknown"

            # TREND
            trend_key = f"TREND#{category}#{week}"
            agg.setdefault(trend_key, {})
            agg[trend_key][sentiment] = agg[trend_key].get(sentiment, 0) + 1

            # CAT
            cat_key = f"CAT#{category}"
            agg.setdefault(cat_key, {})
            agg[cat_key][sentiment] = agg[cat_key].get(sentiment, 0) + 1

            # ISSUE
            if r.get("issue_tag"):
                issue_key = f"ISSUE#{r['issue_tag']}#{week}"
                agg.setdefault(issue_key, {})
                agg[issue_key]["count"] = agg[issue_key].get("count", 0) + 1

        # Write aggregates for this batch
        for agg_type, metrics in agg.items():
            item = {"batch_id": batch_id, "agg_type": agg_type, "updated_at": ts}
            item.update(metrics)
            tables.aggregates.put_item(Item=item)

        log.info(f"Rebuilt {len(agg)} aggregate rows for batch {batch_id}")

    log.info("Rebuild complete")


if __name__ == "__main__":
    rebuild()
