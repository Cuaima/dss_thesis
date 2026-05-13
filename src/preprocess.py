# =============================================================================
# preprocess.py  –  load → filter → clean → anonymize
#
# Execution order:
#   1. load_raw_data()         – read CSVs from disk
#   2. build_topic_account_map() – join topics → groups → accounts
#   3. remove_superusers()     – drop posters from test/demo forums
#   4. clean_dataframe()       – strip HTML, convert dates
#   5. filter_text_quality()   – min length, language
#   6. anonymize_ids()         – replace PosterID with user_N
#   7. anonymize_text_columns()– NER-based text anonymization
#   8. save_outputs()          – attach GroupName, write per-account CSVs
#
# Note: intro/welcome group filtering is handled in postprocess.py (step 2),
# so that it can be adjusted without re-running the expensive anonymization step.
# =============================================================================

from __future__ import annotations

import os
import re
import warnings
import pandas as pd
from collections import defaultdict
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

from config import (
    DATA_DIR, OUTPUT_DIR, PREPROCESS_DIR,
    CSV_FILES, ID_COLUMN, DATE_COLUMNS, TEXT_COLUMN,
    SUPERUSER_ACCOUNT_IDS, COMMUNITY_ACCOUNT_IDS,
    MODERATOR_POSTER_IDS,
    INTRO_GROUP_KEYWORDS,
    MIN_WORD_COUNT, LANGUAGE_FILTER, TARGET_LANGUAGE,
    ANONYMIZE_TEXT, REPLACE_ORIGINAL_TEXT, EXPORT_ENTITY_REVIEW,
)

_ANON_AVAILABLE = False        # ← always defined at module level first
_LANGDETECT_AVAILABLE = False

# Optional: language detection  (pip install langdetect)
if LANGUAGE_FILTER:
    try:
        from langdetect import detect, LangDetectException
        _LANGDETECT_AVAILABLE = True
    except ImportError:
        print("WARNING: langdetect not installed – language filter disabled.")

# Optional: text anonymization  (pip install text-anonymizer)
if ANONYMIZE_TEXT:
    try:
        from custom_text_anonymizer import anonymize as ta_anonymize
        _ANON_AVAILABLE = True
    except Exception as e:
        print(f"WARNING: custom_text_anonymizer unavailable ({type(e).__name__}: {e})")
        print("  → Run with: conda activate thesis_env && PYTHONPATH=./src python src/preprocess.py")


print("preprocess module loaded.")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def ensure_output_dir():
    os.makedirs(PREPROCESS_DIR, exist_ok=True)


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{name}.csv")
    print(f"  Loading {path}")
    return pd.read_csv(path, on_bad_lines="warn", **kwargs)


def write_csv(df: pd.DataFrame, filename: str):
    df.to_csv(os.path.join(PREPROCESS_DIR, filename), index=False)


# ── Step 1: Load raw data ─────────────────────────────────────────────────────

def load_raw_data() -> dict[str, pd.DataFrame]:
    print("\n[1] Loading raw data…")
    dfs = {}
    # Load integrated messages if it exists, fall back to raw
    integrated_path = os.path.join(OUTPUT_DIR, "integrated_messages.csv")
    if os.path.exists(integrated_path):
        print(f"  Loading integrated messages from {integrated_path}")
        dfs["messages"] = pd.read_csv(integrated_path, on_bad_lines="skip")
    else:
        dfs["messages"] = read_csv("messages")

    # Topics, groups and accounts still come from raw
    for name in ["topics", "groups", "accounts"]:
        dfs[name] = read_csv(name)

    return dfs


# ── Step 2: Build topic → account mapping ────────────────────────────────────

def build_topic_account_map(dfs: dict[str, pd.DataFrame]) -> dict:
    """
    Returns {ForumTopicID: AccountID} with exactly one AccountID per topic.
    Also attaches GroupName so we can filter on it later.
    """
    topics = dfs["topics"][["ForumTopicID", "ForumGroupID"]].copy()
    groups = dfs["groups"][["ForumGroupID", "AccountID", "Name"]].copy()
    groups = groups.rename(columns={"Name": "GroupName"})

    merged = topics.merge(groups, on="ForumGroupID", how="left")
    merged = (
        merged
        .dropna(subset=["AccountID"])
        .drop_duplicates(subset=["ForumTopicID"])
    )

    topic_to_account = dict(zip(merged["ForumTopicID"], merged["AccountID"].astype(int)))
    topic_to_group   = dict(zip(merged["ForumTopicID"], merged["GroupName"].fillna("")))

    return topic_to_account, topic_to_group


# ── Step 3: Identify and remove superuser posters ────────────────────────────

def get_superuser_ids(
    messages: pd.DataFrame,
    topic_to_account: dict,
    superuser_accounts: set = SUPERUSER_ACCOUNT_IDS,
) -> set:
    """
    Returns the set of PosterIDs who have ever posted in a superuser account
    forum (test / demo).  These posters are excluded from all datasets.
    """
    df = messages.copy()
    df["AccountID"] = df["ForumTopicID"].map(topic_to_account)

    superuser_posters = set(
        df.loc[df["AccountID"].isin(superuser_accounts), ID_COLUMN].dropna()
    )
    print(f"  Identified {len(superuser_posters)} superuser posters to exclude.")
    return superuser_posters


def remove_superusers(
    messages: pd.DataFrame,
    superuser_ids: set,
) -> pd.DataFrame:
    before = len(messages)
    messages = messages[~messages[ID_COLUMN].isin(superuser_ids)].copy()
    print(f"  Removed superusers: {before} → {len(messages)} messages.")
    return messages


# ── Step 3b: Remove confirmed moderator posters ──────────────────────────────

def remove_moderators(
    messages: pd.DataFrame,
    moderator_ids: set = MODERATOR_POSTER_IDS,
) -> pd.DataFrame:
    if not moderator_ids:
        print("  No moderator IDs configured — skipping.")
        return messages

    before = len(messages)

    # Drop entire threads where the first post (by date) was by a moderator
    date_col = DATE_COLUMNS[0]
    if date_col in messages.columns:
        first_posts = (
            messages.sort_values(["ForumTopicID", date_col])
            .drop_duplicates(subset=["ForumTopicID"], keep="first")
        )
        mod_threads = set(
            first_posts.loc[first_posts[ID_COLUMN].isin(moderator_ids), "ForumTopicID"]
        )
        messages = messages[~messages["ForumTopicID"].isin(mod_threads)].copy()
        print(f"  Removed {len(mod_threads)} moderator-initiated threads.")
    else:
        print(f"  WARNING: date column '{date_col}' not found; skipping thread-level filter.")

    # Also drop individual moderator messages remaining in other threads
    before_replies = len(messages)
    messages = messages[~messages[ID_COLUMN].isin(moderator_ids)].copy()
    removed_replies = before_replies - len(messages)
    if removed_replies:
        print(f"  Removed {removed_replies} individual moderator messages from other threads.")

    print(f"  Total removed: {before} → {len(messages)} messages.")
    return messages


# ── Step 4: Remove introduction / welcome groups ──────────────────────────────

def get_intro_topic_ids(
    topic_to_group: dict,
    keywords: set = INTRO_GROUP_KEYWORDS,
) -> set:
    """
    Returns ForumTopicIDs whose group name contains any intro keyword.
    """
    intro_ids = {
        tid
        for tid, gname in topic_to_group.items()
        if any(kw in gname.lower() for kw in keywords)
    }
    print(f"  Identified {len(intro_ids)} intro/welcome topics to exclude.")
    return intro_ids


def remove_intro_topics(
    messages: pd.DataFrame,
    intro_topic_ids: set,
) -> pd.DataFrame:
    before = len(messages)
    messages = messages[~messages["ForumTopicID"].isin(intro_topic_ids)].copy()
    print(f"  Removed intro topics: {before} → {len(messages)} messages.")
    return messages


# ── Step 5: Clean dataframe (HTML stripping, date parsing) ───────────────────

def _parse_html(text: str) -> str:
    return BeautifulSoup(str(text), "html.parser").get_text(separator=" ").strip()


def _strip_forum_quotes(text: str) -> str:
    """
    Remove quoted reply blocks common in forum posts.
    Handles patterns like:
      - Lines starting with '>'
      - [quote]…[/quote] BBCode blocks
      - <blockquote>…</blockquote> (already handled by HTML strip above)
    """
    # BBCode quotes
    text = re.sub(r"\[quote[^\]]*\].*?\[/quote\]", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Markdown-style quote lines
    text = re.sub(r"^>.*$", "", text, flags=re.MULTILINE)
    # Collapse extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

    # Drop fully-empty rows
    df = df.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all").reset_index(drop=True)

    # Strip HTML from all columns
    for col in df.columns:
        df[col] = df[col].astype(str).apply(_parse_html)

    # Strip quoted reply blocks from the main text column
    if TEXT_COLUMN in df.columns:
        df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(_strip_forum_quotes)

    # Parse date columns
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def standardize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies post-cleaning text standardization to TEXT_COLUMN.
    Keeps original in MessageText, adds MessageText_normalized for model use.
    """
    if TEXT_COLUMN not in df.columns:
        return df

    df = df.copy()
    text = df[TEXT_COLUMN].fillna("").astype(str)

    # ── Remove URLs ───────────────────────────────────────────────────────────
    text = text.apply(lambda t: re.sub(r"https?://\S+|www\.\S+", "", t))

    # ── Normalize repeated punctuation ────────────────────────────────────────
    text = text.apply(lambda t: re.sub(r"([!?.]){2,}", r"\1", t))

    # ── Collapse newlines and tabs into single space ──────────────────────────
    text = text.apply(lambda t: re.sub(r"[\r\n\t]+", " ", t))

    # ── Collapse multiple spaces ──────────────────────────────────────────────
    text = text.apply(lambda t: re.sub(r" {2,}", " ", t))

    # ── Strip leading/trailing whitespace ────────────────────────────────────
    text = text.str.strip()

    # ── Write normalized version (lowercased) to separate column ─────────────
    df[f"{TEXT_COLUMN}_normalized"] = text.str.lower()

    # ── Update original column with everything except lowercasing ────────────
    df[TEXT_COLUMN] = text

    return df


# ── Step 6: Text quality filters ──────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(str(text).split())


def _detect_language(text: str) -> str:
    if not _LANGDETECT_AVAILABLE:
        return TARGET_LANGUAGE
    try:
        return detect(str(text))
    except Exception:
        return "unknown"


def filter_text_quality(
    df: pd.DataFrame,
    min_words: int = MIN_WORD_COUNT,
    language_filter: bool = LANGUAGE_FILTER,
    target_lang: str = TARGET_LANGUAGE,
) -> pd.DataFrame:
    if TEXT_COLUMN not in df.columns:
        return df

    before = len(df)

    # Minimum word count
    df = df[df[TEXT_COLUMN].fillna("").apply(_word_count) >= min_words].copy()
    print(f"  Min-word filter ({min_words}): {before} → {len(df)} messages.")

    # Language filter
    if language_filter and _LANGDETECT_AVAILABLE:
        before = len(df)
        df["_lang"] = df[TEXT_COLUMN].apply(_detect_language)
        df = df[df["_lang"] == target_lang].drop(columns=["_lang"])
        print(f"  Language filter ({target_lang}): {before} → {len(df)} messages.")

    return df


# ── Step 7: ID anonymization ──────────────────────────────────────────────────

def anonymize_ids(dfs: dict[str, pd.DataFrame]) -> dict[str, str]:
    """
    Replaces all PosterID values with user_N across every DataFrame.
    Writes the mapping to output/anonymization_mapping.csv.
    Returns the mapping dict.
    """
    all_ids: set = set()
    for df in dfs.values():
        if ID_COLUMN in df.columns:
            all_ids.update(df[ID_COLUMN].dropna())

    mapping = {uid: f"user_{i + 1}" for i, uid in enumerate(sorted(all_ids))}

    for df in dfs.values():
        if ID_COLUMN in df.columns:
            df[ID_COLUMN] = df[ID_COLUMN].map(mapping)

    write_csv(
        pd.DataFrame(mapping.items(), columns=["OriginalID", "AnonymizedID"]),
        "anonymization_mapping.csv",
    )
    print(f"  Anonymized {len(mapping)} unique poster IDs.")
    return mapping





# ── Step 8: Text anonymization ───────────────────────────────────────────────

def _strip_at_mentions(text: str) -> str:
    # Converts @username to username
    # So NER can recognize it as a name entity instead of a social media handle
    return re.sub(r"@(\w+)", lambda m: m.group(1).replace("_", " "), text)


def anonymize_text_column(
    df: pd.DataFrame,
    column: str,
    export_review: bool = EXPORT_ENTITY_REVIEW,
    replace_original: bool = REPLACE_ORIGINAL_TEXT,
) -> pd.DataFrame:
    if column not in df.columns:
        print(f"  SKIP anonymization: column '{column}' not found.")
        return df
    if not _ANON_AVAILABLE:
        print("  SKIP anonymization: custom_text_anonymizer unavailable.")
        return df

    anon_texts, anon_entities = [], []
    for text in df[column].fillna("").astype(str):
        cleaned = _strip_at_mentions(text)
        anon, entities = ta_anonymize(cleaned)
        anon_texts.append(anon)
        anon_entities.append(entities)

    df = df.copy()
    df[f"{column}_anon"]     = anon_texts
    # df[f"{column}_entities"] = anon_entities  # (omitted from export for now to avoid large JSON blobs in review file)

    # Entities written to a separate file for review
    if "ForumMessageID" in df.columns:
        ref_col = "ForumMessageID"
    else:
        ref_col = df.index.name or "index"
        df = df.reset_index()

    entities_df = pd.DataFrame({
        ref_col: df[ref_col],
        "column": column,
        "entities": anon_entities,
    })
    write_csv(entities_df, f"entities_{column}.csv")


    if export_review:
        write_csv(
            pd.DataFrame({
                "original_text":   df[column],
                "anonymized_text": df[f"{column}_anon"],
                # "entities":        df[f"{column}_entities"],  # (omitted from export for now to avoid large JSON blobs in review file)
            }),
            f"review_anonymization_{column}.csv",
        )

    if replace_original:
        df[column] = df[f"{column}_anon"]
        df = df.drop(columns=[f"{column}_anon"])

    # Drop the index column if we added it just for the reference
    if ref_col == "index" and "index" in df.columns:
        df = df.drop(columns=["index"])

    return df


def anonymize_text_columns(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    if columns is None:
        columns = [TEXT_COLUMN, "Name"]
    for col in columns:
        if col in df.columns:
            df = anonymize_text_column(df, col)
    return df


# ── Step 9: Save outputs ─────────────────────────────────────────

# AFTER
def save_outputs(messages: pd.DataFrame, topic_to_account: dict):
    messages = messages.copy()
    integrated_path = os.path.join(OUTPUT_DIR, "integrated_messages.csv")

    if os.path.exists(integrated_path):
        # Integrated data is already filtered to community accounts
        community = messages
    else:
        # Original flow — filter by account type
        topic_to_account_str = {str(int(float(k))): v for k, v in topic_to_account.items()}
        messages["AccountID"] = (
            messages["ForumTopicID"]
            .apply(lambda x: str(int(float(x))) if pd.notna(x) else None)
            .map(topic_to_account_str)
        )
        matched = messages["AccountID"].notna().sum()
        print(f"  Matched {matched}/{len(messages)} messages to an account.")
        community = messages[
            messages["AccountID"].isin(COMMUNITY_ACCOUNT_IDS)
        ].drop(columns=["AccountID"])

    write_csv(community, "messages_community.csv")
    print(f"  Wrote {len(community)} messages → messages_community.csv")


# ── Main pipeline ─────────────────────────────────────────────────────────────

# AFTER
def run_pipeline():
    ensure_output_dir()

    integrated_path = os.path.join(OUTPUT_DIR, "integrated_messages.csv")
    using_integrated = os.path.exists(integrated_path)

    # 1. Load
    dfs = load_raw_data()

    # 2. Build maps
    print("\n[2] Building topic → account map…")
    raw_topics = pd.read_csv(os.path.join(DATA_DIR, "topics.csv"))
    raw_groups = pd.read_csv(os.path.join(DATA_DIR, "groups.csv"))
    topic_to_account, topic_to_group = build_topic_account_map({
        "topics": raw_topics,
        "groups": raw_groups,
    })

    if using_integrated:
        print("\n[3] Skipping superuser removal — already applied in integrate_datasets.py")
        print("\n[4] Skipping intro group removal — already applied in integrate_datasets.py")
    else:
        # 3. Superuser removal
        print("\n[3] Identifying superusers…")
        superuser_ids = get_superuser_ids(dfs["messages"], topic_to_account)
        dfs["messages"] = remove_superusers(dfs["messages"], superuser_ids)

        # 3b. Remove confirmed moderators
        print("\n[3b] Removing moderators…")
        dfs["messages"] = remove_moderators(dfs["messages"])

        # 4. Remove intro/welcome threads
        print("\n[4] Removing intro/welcome groups…")
        intro_topic_ids = get_intro_topic_ids(topic_to_group)
        dfs["messages"] = remove_intro_topics(dfs["messages"], intro_topic_ids)
        dfs["topics"]   = dfs["topics"][~dfs["topics"]["ForumTopicID"].isin(intro_topic_ids)].copy()

    # 5. Clean all DataFrames
    print("\n[5] Cleaning dataframes (HTML, dates, quote stripping)…")
    for name in dfs:
        dfs[name] = clean_dataframe(dfs[name])

    # 5b. Standardize text
    print("\n[5b] Standardizing text…")
    dfs["messages"] = standardize_text(dfs["messages"])

    # 6. Text quality filters (messages only)
    print("\n[6] Filtering text quality…")
    dfs["messages"] = filter_text_quality(dfs["messages"])

    # 7. ID anonymization
    print("\n[7] Anonymizing poster IDs…")
    anonymize_ids(dfs)

    # 8. Text anonymization
    if ANONYMIZE_TEXT:
        print("\n[8] Anonymizing text…")
        dfs["messages"] = anonymize_text_columns(dfs["messages"], columns=[TEXT_COLUMN])
        dfs["topics"]   = anonymize_text_columns(dfs["topics"], columns=["Name"])
        # ← remove the standardize_text() call that was here

    # 9. Write cleaned files
    print("\n[9] Saving outputs…")
    for name, df in dfs.items():
        write_csv(df, f"{name}_cleaned.csv")

    save_outputs(dfs["messages"], topic_to_account)

    print("\n✓ Pipeline complete.")
    return dfs


if __name__ == "__main__":
    run_pipeline()
