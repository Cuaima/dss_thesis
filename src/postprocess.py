# =============================================================================
# postprocess.py  –  thread structure → filter → normalize → sample
#
# Run this AFTER the original preprocess.py pipeline has already produced
# cleaned output. This script picks up from messages_community.csv and adds:
#
#   1. load_cleaned_data()            – load messages_community.csv
#   2. filter_intro_groups()          – drop welcome / off-topic threads
#   3. build_thread_structure()       – flag initial posts vs replies
#   4. label_thread_success()         – threads with 0 replies = negative class
#   5. normalize_text()               – lowercase, whitespace, repeated chars
#   6. filter_short_initial_posts()   – drop threads whose OP < 5 words
#   7. sanity_check_lengths()         – warn on suspiciously short messages
#   8. extract_annotation_sample()    – 500 replies for manual annotation
#   9. save_outputs()                 – write final dataset + annotation files
#
# Input:  output/messages_community.csv  (from original preprocess.py)
# Output: output/messages_structured.csv
#         output/annotation_sample.csv
#         output/annotation_sample_with_context.csv
# =============================================================================

from __future__ import annotations

import os
import re
import pandas as pd

from config import INTRO_GROUP_KEYWORDS

# ── Config ────────────────────────────────────────────────────────────────────
# Update these to match your config.py if they differ
INPUT_FILE    = os.path.join("output", "preprocessed", "messages_community.csv")
OUTPUT_DIR    = "output"
TEXT_COLUMN   = "MessageText"
DATE_COLUMN   = "PostDate"        # first date column from config DATE_COLUMNS
ANNOTATION_N      = 500           # number of replies to sample for annotation
USER_REPLY_CAP    = 4             # max times a single user can appear in the sample
RANDOM_STATE      = 42            # for reproducibility
MIN_WORDS_INITIAL = 5             # threads whose initial post is below this are dropped


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def write_csv(df: pd.DataFrame, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ── Step 1: Load cleaned data ─────────────────────────────────────────────────

def load_cleaned_data() -> pd.DataFrame:
    print(f"\n[1] Loading cleaned data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df)} messages, {df['ForumTopicID'].nunique()} threads.")

    # Parse date column if not already datetime
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    return df


# ── Step 2: Filter intro / welcome / off-topic groups ────────────────────────

def filter_intro_groups(
    messages: pd.DataFrame,
    keywords: set = INTRO_GROUP_KEYWORDS,
) -> pd.DataFrame:
    """
    Drops threads belonging to intro/welcome/off-topic groups.
    Relies on the 'GroupName' column written by preprocess.py's save_outputs().
    Skips gracefully if the column is absent or empty.
    """
    print("\n[2] Filtering intro / welcome / off-topic groups...")

    if "GroupName" not in messages.columns or messages["GroupName"].fillna("").eq("").all():
        print("  SKIP: 'GroupName' column not found or empty — re-run preprocess.py to populate it.")
        return messages

    mask = messages["GroupName"].fillna("").str.lower().apply(
        lambda g: any(kw in g for kw in keywords)
    )
    intro_thread_ids = messages.loc[mask, "ForumTopicID"].unique()

    before = messages["ForumTopicID"].nunique()
    messages = messages[~messages["ForumTopicID"].isin(intro_thread_ids)].copy()
    print(
        f"  Removed {before - messages['ForumTopicID'].nunique()} intro/welcome/off-topic threads — "
        f"{messages['ForumTopicID'].nunique()} threads remain."
    )
    return messages


# ── Step 3: Build thread structure ───────────────────────────────────────────

def build_thread_structure(messages: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts by ForumTopicID + PostDate, then adds:
      - is_initial_post (bool): True for the first message in each thread
      - reply_index (int):      0 for initial post, 1..N for replies
    """
    print("\n[3] Building thread structure...")

    if DATE_COLUMN not in messages.columns:
        raise ValueError(
            f"Date column '{DATE_COLUMN}' not found. "
            f"Available columns: {list(messages.columns)}"
        )

    messages = messages.sort_values(
        ["ForumTopicID", DATE_COLUMN]
    ).reset_index(drop=True)

    messages["reply_index"]     = messages.groupby("ForumTopicID").cumcount()
    messages["is_initial_post"] = messages["reply_index"] == 0

    n_initial  = messages["is_initial_post"].sum()
    n_replies  = (~messages["is_initial_post"]).sum()
    n_threads  = messages["ForumTopicID"].nunique()
    print(f"  {n_threads} threads: {n_initial} initial posts, {n_replies} replies.")

    return messages


# ── Step 4: Label thread success ─────────────────────────────────────────────

def label_thread_success(messages: pd.DataFrame) -> pd.DataFrame:
    """
    Labels threads based on whether they received any replies.

    Threads WITH replies    → thread_has_replies = True  (positive class candidate)
    Threads WITHOUT replies → thread_has_replies = False (negative class)

    Note: 'thread_has_replies' is a structural label only.
    The final 'successful' label (whether at least one reply is supportive)
    is assigned after manual annotation.

    Threads with no replies are KEPT as they form a natural negative class
    for the classification task.
    """
    print("\n[4] Labeling thread success...")

    reply_counts = (
        messages[~messages["is_initial_post"]]
        .groupby("ForumTopicID")
        .size()
        .reset_index(name="reply_count")
    )

    messages = messages.merge(reply_counts, on="ForumTopicID", how="left")
    messages["reply_count"]       = messages["reply_count"].fillna(0).astype(int)
    messages["thread_has_replies"] = messages["reply_count"] > 0

    no_reply  = (~messages["thread_has_replies"] & messages["is_initial_post"]).sum()
    has_reply = (messages["thread_has_replies"]  & messages["is_initial_post"]).sum()
    print(f"  Threads with replies:    {has_reply} (positive class candidates)")
    print(f"  Threads without replies: {no_reply} (negative class)")

    return messages


# ── Step 5: Normalize text ────────────────────────────────────────────────────

def _normalize_dutch_text(text: str) -> str:
    """
    Light normalization for Dutch NLP / LIWC feature extraction.

    Deliberately preserves:
      - Punctuation (question marks, exclamation points carry linguistic signal)
      - Stopwords (relevant for psycholinguistic LIWC features)
      - Sentence boundaries

    Applies:
      - Lowercase
      - Reduce pathological character repetition (e.g. 'haaaai' -> 'haai')
      - Normalize horizontal whitespace
      - Cap consecutive newlines at two
    """
    text = str(text).lower()
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)  # 4+ repeated chars -> 2
    text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)        # max two consecutive newlines
    return text.strip()


def normalize_text(messages: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'text_normalized' column for downstream feature extraction.
    The original TEXT_COLUMN is preserved for annotation and review.
    """
    print("\n[5] Normalizing text...")

    if TEXT_COLUMN not in messages.columns:
        print(f"  SKIP: column '{TEXT_COLUMN}' not found.")
        return messages

    messages = messages.copy()
    messages["text_normalized"] = (
        messages[TEXT_COLUMN].fillna("").apply(_normalize_dutch_text)
    )
    print("  Done -> column 'text_normalized' added.")
    return messages


# ── Step 6: Filter short initial posts ───────────────────────────────────────

def filter_short_initial_posts(
    messages: pd.DataFrame,
    min_words: int = MIN_WORDS_INITIAL,
) -> pd.DataFrame:
    """
    Drops threads where the initial post has fewer than `min_words` words.
    Removes the full thread (initial post + all replies) to avoid orphaned rows.
    Word count is computed on 'text_normalized'.
    """
    print(f"\n[6] Filtering threads with initial posts < {min_words} words...")

    wc = messages["text_normalized"].fillna("").apply(lambda x: len(x.split()))
    short_thread_ids = messages.loc[
        messages["is_initial_post"] & (wc < min_words), "ForumTopicID"
    ]

    before_threads = messages["ForumTopicID"].nunique()
    before_msgs    = len(messages)
    messages = messages[~messages["ForumTopicID"].isin(short_thread_ids)].copy()

    print(
        f"  Removed {before_threads - messages['ForumTopicID'].nunique()} threads "
        f"({before_msgs - len(messages)} messages total) — "
        f"{messages['ForumTopicID'].nunique()} threads remain."
    )
    return messages


# ── Step 7: Sanity check ──────────────────────────────────────────────────────

def sanity_check_lengths(messages: pd.DataFrame) -> None:
    """
    Quick sanity check: reports distribution of word counts in initial posts
    and replies so you can spot any suspiciously short or empty messages
    that slipped through the original pipeline.
    """
    print("\n[7] Sanity checking message lengths...")

    wc = messages["text_normalized"].fillna("").apply(lambda x: len(x.split()))
    messages = messages.copy()
    messages["_wc"] = wc

    for label, subset in [
        ("Initial posts", messages[messages["is_initial_post"]]),
        ("Replies",       messages[~messages["is_initial_post"]]),
    ]:
        q = subset["_wc"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        print(
            f"  {label}: "
            f"median={q[0.5]:.0f} words, "
            f"5th pct={q[0.05]:.0f}, "
            f"95th pct={q[0.95]:.0f}"
        )

    very_short = (wc < 3).sum()
    if very_short > 0:
        print(
            f"  WARNING: {very_short} messages have fewer than 3 words. "
            f"Consider reviewing these."
        )


# ── Step 8: Extract annotation sample ────────────────────────────────────────

def extract_annotation_sample(
    messages: pd.DataFrame,
    n: int = ANNOTATION_N,
    random_state: int = RANDOM_STATE,
    user_cap: int = USER_REPLY_CAP,
) -> pd.DataFrame:
    """
    Draws a reproducible random sample of `n` replies for manual annotation.
    Only replies (is_initial_post == False) are included.

    Sampling procedure (applied in order):
      1. Exclude self-replies (OP replying to their own thread).
      2. Apply a per-user cap of `user_cap` replies across the whole corpus to
         prevent the sample from being dominated by highly active users.
         Corpus analysis showed that 161 users (25.6% of repliers) accounted
         for 91.3% of all eligible replies; a cap of 4 was chosen to retain
         all 549 unique users while keeping the pool above the 500-sample target.
      3. Sample at most one reply per thread to preserve independence.
      4. Draw a random sample of `n` from the resulting pool.

    Exports two files:
      - annotation_sample.csv:              reply text + blank label columns
      - annotation_sample_with_context.csv: same + matched initial post text

    random_state=42 ensures reproducibility — required for the methodology chapter.
    """
    print(f"\n[8] Extracting annotation sample (n={n}, user_cap={user_cap})...")

    # Identify the original poster for each thread
    initial_posters = (
        messages[messages["is_initial_post"]][["ForumTopicID", "PosterID"]]
        .rename(columns={"PosterID": "initial_poster_id"})
    )

    # Step 1: Exclude self-replies (OP updating their own thread).
    replies = (
        messages[~messages["is_initial_post"]]
        .merge(initial_posters, on="ForumTopicID", how="left")
        .query("PosterID != initial_poster_id")
        .drop(columns=["initial_poster_id"])
        .copy()
    )
    excluded = (~messages["is_initial_post"]).sum() - len(replies)
    print(f"  Excluded {excluded} self-replies.")
    print(f"  Eligible pool before cap: {len(replies):,} replies from {replies['PosterID'].nunique()} users.")

    # Step 2: Per-user cap across the whole corpus.
    # Shuffle first so the cap selects randomly rather than always the earliest replies.
    replies_capped = (
        replies
        .sample(frac=1, random_state=random_state)
        .groupby("PosterID")
        .head(user_cap)
        .reset_index(drop=True)
    )
    n_affected = (replies.groupby("PosterID").size() > user_cap).sum()
    print(
        f"  After user cap ({user_cap}): {len(replies_capped):,} replies "
        f"({n_affected} users affected, {replies_capped['PosterID'].nunique()} unique users retained)."
    )

    # Step 3: One reply per thread to ensure independence.
    replies_deduped = replies_capped.groupby("ForumTopicID").sample(
        n=1, random_state=random_state
    )
    print(
        f"  After per-thread dedup: {len(replies_deduped):,} replies "
        f"across {replies_deduped['ForumTopicID'].nunique()} threads — "
        f"{'sufficient' if len(replies_deduped) >= n else 'INSUFFICIENT'} for n={n}."
    )

    if len(replies_deduped) < n:
        print(f"  WARNING: pool size {len(replies_deduped)} < n={n}; sampling all available.")
        n = len(replies_deduped)

    # Step 4: Draw final sample.
    sample = replies_deduped.sample(n=n, random_state=random_state)

    # Columns for annotators
    export_cols = [
        col for col in [
            "ForumMessageID", "ForumTopicID",
            "text_normalized",
            DATE_COLUMN, "reply_index",
        ]
        if col in sample.columns
    ]

    # Add empty label columns for annotators to fill in
    sample = sample[export_cols].copy()
    sample["label"]        = ""   # SS (Social Support) or NSS (Not Social Support)
    sample["support_type"] = ""   # informational / emotional / other / N/A

    write_csv(sample, "annotation_sample.csv")
    print(f"  annotation_sample.csv: {n} replies ready for labeling.")

    # Add initial post as context
    initial_posts = (
        messages[messages["is_initial_post"]][["ForumTopicID", TEXT_COLUMN]]
        .rename(columns={TEXT_COLUMN: "initial_post_text"})
    )
    sample_with_context = sample.merge(initial_posts, on="ForumTopicID", how="left")
    write_csv(sample_with_context, "annotation_sample_with_context.csv")
    print("  annotation_sample_with_context.csv: same sample + initial post text.")

    return sample


# ── Step 9: Save structured dataset ──────────────────────────────────────────

def save_outputs(messages: pd.DataFrame):
    print("\n[9] Saving structured dataset...")

    # Drop any temporary helper columns before export
    drop_cols = ["_wc", "GroupName"]
    messages_clean = messages.drop(columns=drop_cols, errors="ignore")
    write_csv(messages_clean, "messages_structured.csv")
    print(
        f"  messages_structured.csv: {len(messages_clean)} messages, "
        f"{messages_clean['ForumTopicID'].nunique()} threads."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    ensure_output_dir()

    messages = load_cleaned_data()
    messages = filter_intro_groups(messages)
    messages = build_thread_structure(messages)
    messages = label_thread_success(messages)
    messages = normalize_text(messages)
    messages = filter_short_initial_posts(messages)
    sanity_check_lengths(messages)
    extract_annotation_sample(messages, n=ANNOTATION_N, random_state=RANDOM_STATE, user_cap=USER_REPLY_CAP)
    save_outputs(messages)

    print("\nPostprocessing complete.")
    print("Next steps:")
    print("  1. Open output/annotation_sample_with_context.csv")
    print("  2. Fill in 'label' (SS/NSS) and 'support_type' columns")
    print("  3. Share annotation_sample.csv with your second rater")
    print("  4. Calculate Cohen's Kappa after both raters are done")


if __name__ == "__main__":
    run()