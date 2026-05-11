# =============================================================================
# postprocess.py  –  thread structure → filter → normalize → sample
#
# Run this AFTER the original preprocess.py pipeline has already produced
# cleaned output. This script picks up from messages_community.csv and adds:
#
#   1. build_thread_structure()       – flag initial posts vs replies
#   2. label_thread_success()         – threads with 0 replies = negative class
#   3. normalize_text()               – lowercase, whitespace, repeated chars
#   4. sanity_check_quote_stripping() – warn if stripping removed too much text
#   5. extract_annotation_sample()    – 500 replies for manual annotation
#   6. save_outputs()                 – write final dataset + annotation files
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

# ── Config ────────────────────────────────────────────────────────────────────
# Update these to match your config.py if they differ
INPUT_FILE   = os.path.join("data", "messages_community.csv")
OUTPUT_DIR   = "output"
TEXT_COLUMN  = "MessageText"
DATE_COLUMN  = "PostDate"        # first date column from config DATE_COLUMNS
ANNOTATION_N = 500               # number of replies to sample for annotation
RANDOM_STATE = 42                # for reproducibility


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


# ── Step 2: Build thread structure ───────────────────────────────────────────

def build_thread_structure(messages: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts by ForumTopicID + PostDate, then adds:
      - is_initial_post (bool): True for the first message in each thread
      - reply_index (int):      0 for initial post, 1..N for replies
    """
    print("\n[2] Building thread structure...")

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


# ── Step 3: Label thread success ─────────────────────────────────────────────

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
    print("\n[3] Labeling thread success...")

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


# ── Step 4: Normalize text ────────────────────────────────────────────────────

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
    print("\n[4] Normalizing text...")

    if TEXT_COLUMN not in messages.columns:
        print(f"  SKIP: column '{TEXT_COLUMN}' not found.")
        return messages

    messages = messages.copy()
    messages["text_normalized"] = (
        messages[TEXT_COLUMN].fillna("").apply(_normalize_dutch_text)
    )
    print("  Done -> column 'text_normalized' added.")
    return messages


# ── Step 5: Sanity check ──────────────────────────────────────────────────────

def sanity_check_lengths(messages: pd.DataFrame) -> None:
    """
    Quick sanity check: reports distribution of word counts in initial posts
    and replies so you can spot any suspiciously short or empty messages
    that slipped through the original pipeline.
    """
    print("\n[5] Sanity checking message lengths...")

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


# ── Step 6: Extract annotation sample ────────────────────────────────────────

def extract_annotation_sample(
    messages: pd.DataFrame,
    n: int = ANNOTATION_N,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Draws a reproducible random sample of `n` replies for manual annotation.
    Only replies (is_initial_post == False) are included.

    Exports two files:
      - annotation_sample.csv:
          The sample for annotators. Contains reply text only.
          Add a 'label' column (SS / NSS) and 'support_type' column
          (informational / emotional / other) when annotating.

      - annotation_sample_with_context.csv:
          Same sample but with the matched initial post text added as context,
          so annotators can see what the reply is responding to.

    random_state=42 ensures reproducibility across runs — required for the
    methodology chapter.
    """
    print(f"\n[6] Extracting annotation sample (n={n})...")

        # Identify the original poster for each thread
    initial_posters = (
        messages[messages["is_initial_post"]][["ForumTopicID", "PosterID"]]
        .rename(columns={"PosterID": "initial_poster_id"})
    )

    # Keep only replies from someone OTHER than the original poster
    # Self-replies (OP updating their own thread) are not peer support
    replies = (
        messages[~messages["is_initial_post"]]
        .merge(initial_posters, on="ForumTopicID", how="left")
        .query("PosterID != initial_poster_id")
        .drop(columns=["initial_poster_id"])
        .copy()
    )

    excluded = (~messages["is_initial_post"]).sum() - len(replies)
    print(f"  Excluded {excluded} self-replies from annotation pool.")

    if len(replies) < n:
        print(
            f"  WARNING: only {len(replies)} replies available; "
            f"sampling all instead of {n}."
        )
        n = len(replies)

    # Sample at most one reply per thread to ensure independence
    # and maximize thread coverage in the annotation sample
    replies_deduped = replies.groupby("ForumTopicID").sample(
        n=1, random_state=random_state
    )
    if len(replies_deduped) < n:
        print(
            f"  WARNING: only {len(replies_deduped)} unique threads available; "
            f"sampling all instead of {n}."
        )
        n = len(replies_deduped)
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


# ── Step 7: Save structured dataset ──────────────────────────────────────────

def save_outputs(messages: pd.DataFrame):
    print("\n[7] Saving structured dataset...")

    # Drop any temporary helper columns before export
    drop_cols = ["_wc"]
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
    messages = build_thread_structure(messages)
    messages = label_thread_success(messages)
    messages = normalize_text(messages)
    sanity_check_lengths(messages)
    extract_annotation_sample(messages, n=ANNOTATION_N, random_state=RANDOM_STATE)
    save_outputs(messages)

    print("\nPostprocessing complete.")
    print("Next steps:")
    print("  1. Open output/annotation_sample_with_context.csv")
    print("  2. Fill in 'label' (SS/NSS) and 'support_type' columns")
    print("  3. Share annotation_sample.csv with your second rater")
    print("  4. Calculate Cohen's Kappa after both raters are done")


if __name__ == "__main__":
    run()