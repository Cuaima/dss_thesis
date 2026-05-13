"""
Reply distribution per user in messages_community.csv.

Computes is_self_reply from thread structure (no postprocess.py needed),
then reports how many replies each user contributed and what share would
be affected by a per-user cap.

Run from project root:
    python scripts/reply_distribution.py
"""

import os
import pandas as pd

INPUT_FILE = os.path.join("output", "preprocessed", "messages_community.csv")
DATE_COLUMN = "PostDate"
POSTER_COLUMN = "PosterID"


def main():
    df = pd.read_csv(INPUT_FILE)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    # Identify the first post per thread by date → is_initial_post
    df = df.sort_values(["ForumTopicID", DATE_COLUMN]).reset_index(drop=True)
    df["is_initial_post"] = df.groupby("ForumTopicID").cumcount() == 0

    # Flag self-replies: replies where PosterID == the thread's original poster
    initial_posters = (
        df[df["is_initial_post"]][["ForumTopicID", POSTER_COLUMN]]
        .rename(columns={POSTER_COLUMN: "initial_poster_id"})
    )
    df = df.merge(initial_posters, on="ForumTopicID", how="left")
    df["is_self_reply"] = (~df["is_initial_post"]) & (df[POSTER_COLUMN] == df["initial_poster_id"])

    eligible = df[~df["is_self_reply"]].copy()
    replies_per_user = eligible.groupby(POSTER_COLUMN).size().sort_values(ascending=False)

    print(replies_per_user.describe())
    print(f"\nTop 10 users and their reply counts:")
    print(replies_per_user.head(10))
    print(f"\nUsers with >5 replies: {(replies_per_user > 5).sum()}")
    print(f"Their share of total replies: {replies_per_user[replies_per_user > 5].sum() / len(eligible):.1%}")
    print(f"\nUsers with >10 replies: {(replies_per_user > 10).sum()}")
    print(f"Their share of total replies: {replies_per_user[replies_per_user > 10].sum() / len(eligible):.1%}")

    print("\nSampling pool size under different per-user caps:")
    for cap in [1, 2, 3, 5, 10]:
        capped = eligible.groupby(POSTER_COLUMN).head(cap)
        affected = (replies_per_user > cap).sum()
        print(f"  Cap {cap:3d}: pool={len(capped):6,} | "
              f"users affected={affected:3d} | "
              f"unique users={capped[POSTER_COLUMN].nunique()}")


if __name__ == "__main__":
    main()