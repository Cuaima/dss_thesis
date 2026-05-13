#!/usr/bin/env python3
# =============================================================================
# user_stats.py  –  count unique users by role (initial posters vs repliers)
#
# Usage:
#   python src/user_stats.py
#   python src/user_stats.py --input output/preprocessed/messages_community.csv
#   python src/user_stats.py --input output/messages_structured.csv --verbose
# =============================================================================

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# ── Defaults (mirror postprocess.py) ─────────────────────────────────────────
DEFAULT_INPUT = os.path.join("output", "preprocessed", "messages_community.csv")
DATE_COLUMN   = "PostDate"
POSTER_COLUMN = "PosterID"
TOPIC_COLUMN  = "ForumTopicID"


# ── Core logic ────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"ERROR: file not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} messages from {path}")
    return df


def build_roles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds is_initial_post using the same logic as postprocess.py:
    sort by topic + date, first message per thread = initial post.
    Skips sorting if the column is missing (e.g. already structured file).
    """
    if "is_initial_post" in df.columns:
        print("  'is_initial_post' column already present — skipping sort.")
        return df

    if DATE_COLUMN not in df.columns:
        sys.exit(
            f"ERROR: neither 'is_initial_post' nor '{DATE_COLUMN}' found.\n"
            f"       Run postprocess.py first, or pass a messages_structured.csv."
        )

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.sort_values([TOPIC_COLUMN, DATE_COLUMN]).reset_index(drop=True)
    df["reply_index"]     = df.groupby(TOPIC_COLUMN).cumcount()
    df["is_initial_post"] = df["reply_index"] == 0
    return df


def compute_stats(df: pd.DataFrame, verbose: bool = False) -> None:
    if POSTER_COLUMN not in df.columns:
        sys.exit(f"ERROR: column '{POSTER_COLUMN}' not found in the file.")

    initial = df[df["is_initial_post"]]
    replies = df[~df["is_initial_post"]]

    initial_posters = set(initial[POSTER_COLUMN].dropna().unique())
    reply_posters   = set(replies[POSTER_COLUMN].dropna().unique())

    only_initial   = initial_posters - reply_posters   # posted but never replied
    only_replied   = reply_posters - initial_posters   # replied but never posted
    both           = initial_posters & reply_posters   # did both
    all_users      = initial_posters | reply_posters

    n_threads = df[TOPIC_COLUMN].nunique()

    print("\n" + "=" * 52)
    print("  USER ROLE SUMMARY")
    print("=" * 52)
    print(f"  Total messages          : {len(df):>7,}")
    print(f"  Total threads           : {n_threads:>7,}")
    print(f"  Total unique users      : {len(all_users):>7,}")
    print("-" * 52)
    print(f"  Unique initial posters  : {len(initial_posters):>7,}")
    print(f"  Unique repliers         : {len(reply_posters):>7,}")
    print("-" * 52)
    print(f"  Only ever posted (no replies given) : {len(only_initial):>5,}")
    print(f"  Only ever replied (never posted OP) : {len(only_replied):>5,}")
    print(f"  Both posted and replied             : {len(both):>5,}")
    print("=" * 52)

    if verbose:
        # Per-user breakdown: how many threads started / replies given
        posts_per_user  = initial.groupby(POSTER_COLUMN).size().rename("threads_started")
        replies_per_user = replies.groupby(POSTER_COLUMN).size().rename("replies_given")

        summary = (
            pd.concat([posts_per_user, replies_per_user], axis=1)
            .fillna(0)
            .astype(int)
            .sort_values("threads_started", ascending=False)
        )
        summary["role"] = "both"
        summary.loc[summary["replies_given"] == 0, "role"] = "poster_only"
        summary.loc[summary["threads_started"] == 0, "role"] = "replier_only"

        print("\nTop 15 users by threads started:")
        print(summary.head(15).to_string())

        print("\nReply count distribution (among repliers):")
        print(replies_per_user.describe().to_string())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count unique users by role: initial posters vs repliers."
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Path to messages CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Also print per-user breakdown and reply distribution.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load(args.input)
    df = build_roles(df)
    compute_stats(df, verbose=args.verbose)


if __name__ == "__main__":
    main()