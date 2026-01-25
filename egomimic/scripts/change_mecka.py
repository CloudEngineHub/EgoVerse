#!/usr/bin/env python3
"""
Fix processed_path in the SQL episode table.

Goal:
- For rows where processed_path contains "mecka" (case-insensitive),
- ensure processed_path starts with: s3://rldb/

This script only updates the SQL table (no S3 file edits).
"""

import argparse
from typing import Optional

import pandas as pd

# Add parent directory to path to import egomimic modules (match your repo layout)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_table_to_df,
    episode_hash_to_table_row,
    update_episode,
)


TARGET_PREFIX = "s3://rldb/"


def normalize_to_rldb_s3_uri(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    if isinstance(p, float) and pd.isna(p):
        return p

    p = str(p).strip()

    # Already a proper s3 uri
    if p.startswith("s3://"):
        return p

    # Special case: your data sometimes stores "rldb:/..." (or "rldb:...")
    # Desired: "s3://rldb:/..."
    if p.startswith("rldb:"):
        return "s3://" + p

    # Otherwise treat as path/key and force into s3://rldb/<key>
    key = p.lstrip("/")
    return f"s3://rldb/{key}"


def main():
    parser = argparse.ArgumentParser(description="Fix processed_path to start with s3://rldb/ for mecka rows")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing to DB")
    parser.add_argument("--contains", type=str, default="mecka", help="Substring filter for processed_path (default: mecka)")
    parser.add_argument("--case-sensitive", action="store_true", help="Make substring match case-sensitive")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows updated (for testing)")
    args = parser.parse_args()

    engine = create_default_engine()

    df = episode_table_to_df(engine)

    if "processed_path" not in df.columns:
        raise RuntimeError("processed_path column not found in episode table dataframe")

    # Filter rows where processed_path contains substring (default 'mecka')
    # str.contains supports na=False to treat NaNs as False during filtering [web:49]
    mask = df["processed_path"].astype("string").str.contains(
        args.contains,
        case=args.case_sensitive,
        na=False,
        regex=False,
    )
    df = df[mask].copy()

    if args.limit is not None:
        df = df.head(args.limit)

    if df.empty:
        print("No rows matched; nothing to do.")
        return 0

    total = len(df)
    print(f"Matched {total} rows where processed_path contains '{args.contains}'")

    changed = 0
    skipped = 0
    errors = 0

    for i, row in enumerate(df.itertuples(index=False), 1):
        episode_hash = getattr(row, "episode_hash")
        old_path = getattr(row, "processed_path")

        new_path = normalize_to_rldb_s3_uri(old_path)

        if old_path == new_path:
            skipped += 1
            continue

        if args.dry_run:
            print(f"[{i}/{total}] DRY RUN {episode_hash}:")
            print(f"  old: {old_path}")
            print(f"  new: {new_path}")
            changed += 1
            continue

        try:
            table_row = episode_hash_to_table_row(engine, episode_hash)
            if table_row is None:
                print(f"[{i}/{total}] WARNING: episode_hash not found in SQL table: {episode_hash}")
                errors += 1
                continue

            table_row.processed_path = new_path
            update_episode(engine, table_row)

            print(f"[{i}/{total}] UPDATED {episode_hash}")
            changed += 1
        except Exception as e:
            print(f"[{i}/{total}] ERROR updating {episode_hash}: {e}")
            errors += 1

    print("\nDone.")
    print(f"  Changed: {changed}")
    print(f"  Skipped (already ok): {skipped}")
    print(f"  Errors: {errors}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
