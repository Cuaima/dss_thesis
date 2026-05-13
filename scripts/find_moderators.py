# =============================================================================
# find_moderators.py  –  one-time diagnostic to identify moderator posters
#
# Run from project root:
#   PYTHONPATH=./src python src/find_moderators.py
#
# Output: output/moderator_review.csv
#   Open this in Excel/Sheets, review each PosterID, and mark the ones
#   that are actual moderators. Then add those IDs to MODERATOR_POSTER_IDS
#   in config.py.
# =============================================================================

import os
import pandas as pd

DATA_DIR   = "data"
OUTPUT_DIR = "output"

SEARCH_TERMS = [
    r"\bmoderator\b",
    r"\bmoderatoren\b",
    r"\bbeheerder\b",      # Dutch: "administrator/manager"
]

def main():
    messages = pd.read_csv(os.path.join(DATA_DIR, "messages.csv"), on_bad_lines="warn")
    groups   = pd.read_csv(os.path.join(DATA_DIR, "groups.csv"))
    topics   = pd.read_csv(os.path.join(DATA_DIR, "topics.csv"))

    # Join group name onto messages — both topics and groups may have duplicate IDs,
    # so use merge + drop_duplicates rather than map() to avoid index issues.
    topic_groups = (
        topics[["ForumTopicID", "ForumGroupID"]]
        .merge(
            groups[["ForumGroupID", "Name"]].drop_duplicates("ForumGroupID"),
            on="ForumGroupID", how="left",
        )
        .drop_duplicates("ForumTopicID")[["ForumTopicID", "Name"]]
        .rename(columns={"Name": "GroupName"})
    )
    messages = messages.merge(topic_groups, on="ForumTopicID", how="left")
    messages["GroupName"] = messages["GroupName"].fillna("")

    # Find all messages containing any moderator-related term
    pattern = "|".join(SEARCH_TERMS)
    mask    = messages["MessageText"].fillna("").str.contains(pattern, case=False, regex=True)
    hits    = messages[mask].copy()

    print(f"Found {len(hits)} messages mentioning moderator-related terms.")
    print(f"Unique posters in those messages: {hits['PosterID'].nunique()}")

    # For each poster who appears in any hit, collect all their posts so you
    # can see their full behaviour — not just the one mention
    poster_ids = hits["PosterID"].dropna().unique()
    all_posts  = messages[messages["PosterID"].isin(poster_ids)].copy()

    # Summarise: one row per poster with stats + sample text
    summary_rows = []
    for pid, group in all_posts.groupby("PosterID"):
        mod_posts  = hits[hits["PosterID"] == pid]
        sample_mod = mod_posts["MessageText"].iloc[0][:300] if len(mod_posts) else ""
        summary_rows.append({
            "PosterID":           pid,
            "total_posts":        len(group),
            "mod_term_mentions":  len(mod_posts),
            "groups_posted_in":   ", ".join(group["GroupName"].unique()[:5]),
            "sample_mod_message": sample_mod,
            "is_moderator":       "",   # <- fill this in manually: TRUE / FALSE
        })

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("mod_term_mentions", ascending=False)
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "moderator_review.csv")
    summary.to_csv(out_path, index=False)

    print(f"\nReview file written to: {out_path}")
    print("Open it, fill in the 'is_moderator' column (TRUE/FALSE), then")
    print("add the TRUE PosterIDs to MODERATOR_POSTER_IDS in config.py.")

    # Also show word-game candidate groups for reference
    print("\n── Word game / off-topic group names (for your review) ──")
    group_reply_stats = (
        messages.groupby("GroupName")["MessageText"]
        .apply(lambda texts: texts.fillna("").apply(lambda t: len(t.split())).median())
        .reset_index(name="median_word_count")
        .sort_values("median_word_count")
    )
    print(group_reply_stats[group_reply_stats["median_word_count"] <= 4].to_string(index=False))


if __name__ == "__main__":
    main()
