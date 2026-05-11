# =============================================================================
# liwc_extractor.py  –  Extract LIWC features from Dutch text using .dic files
#
# Uses the Dutch LIWC dictionary files directly — no commercial LIWC software
# required. Parses the .dic format, matches words/wildcards against text, and
# produces category proportion scores identical to LIWC's output format.
#
# Supports both LIWC2007 and LIWC2015 .dic formats.
#
# Usage:
#   python src/liwc_extractor.py
#
# Input:  output/messages_structured.csv
#         data/LIWC2015Dutch.dic  (or Dutch_LIWC2007_Dictionary_final.dic)
# Output: output/liwc_output.csv  (ForumTopicID + one column per LIWC category)
# =============================================================================

from __future__ import annotations

import os
import re
import csv
import pandas as pd
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
# Use 2015 dictionary — more categories, more recent
DIC_FILE        = os.path.join("data", "LIWC2015Dutch.dic")
STRUCTURED_FILE = os.path.join("output", "messages_structured.csv")
OUTPUT_FILE     = os.path.join("output", "liwc_output.csv")
TEXT_COLUMN     = "text_normalized"

# If True, only extract initial posts (X variable for classification)
# If False, extract all messages (useful for annotation feature analysis)
INITIAL_POSTS_ONLY = True


# ── Parse .dic file ───────────────────────────────────────────────────────────

def parse_dic_file(dic_path: str) -> tuple[dict[int, str], dict[str, list[int]]]:
    """
    Parses a LIWC .dic file into:
      - category_map:  {category_id: category_name}
      - word_map:      {word_or_pattern: [category_id, ...]}

    The .dic format has two sections separated by '%':
      Section 1: category definitions (id TAB name)
      Section 2: word entries (word TAB cat_id TAB cat_id ...)

    Words ending in '*' are prefix wildcards — match any word starting
    with that prefix.
    """
    print(f"  Parsing dictionary: {dic_path}")

    category_map: dict[int, str] = {}
    word_map: dict[str, list[int]] = {}

    with open(dic_path, encoding="utf-8-sig") as f:
        content = f.read()

    # Split on '%' — first section is categories, second is words
    parts = content.strip().split("%")

    # Handle both 2-part and 3-part formats
    # Format: % [categories] % [words] (the leading % is part[0] = empty)
    if len(parts) < 3:
        raise ValueError(
            f"Unexpected .dic format in {dic_path}. "
            f"Expected 3 sections separated by '%', got {len(parts)}."
        )

    # Section 1: categories (part[1])
    for line in parts[1].strip().splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = line.split("\t")
        if len(tokens) >= 2:
            try:
                cat_id   = int(tokens[0].strip())
                cat_name = tokens[1].strip()
                category_map[cat_id] = cat_name
            except ValueError:
                continue

    # Section 2: words (part[2])
    for line in parts[2].strip().splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = line.split("\t")
        if len(tokens) < 2:
            continue
        word = tokens[0].strip().lower()
        try:
            cat_ids = [int(t.strip()) for t in tokens[1:] if t.strip().isdigit()]
        except ValueError:
            continue
        if cat_ids:
            word_map[word] = cat_ids

    print(f"  Categories: {len(category_map)}")
    print(f"  Dictionary entries: {len(word_map)}")
    return category_map, word_map


def build_lookup(
    word_map: dict[str, list[int]],
) -> tuple[dict[str, list[int]], list[tuple[str, list[int]]]]:
    """
    Splits word_map into:
      - exact_lookup:   {word: [cat_ids]} for exact matches
      - prefix_lookup:  [(prefix, [cat_ids])] for wildcard matches (word*)
    """
    exact_lookup: dict[str, list[int]] = {}
    prefix_lookup: list[tuple[str, list[int]]] = []

    for word, cat_ids in word_map.items():
        if word.endswith("*"):
            prefix_lookup.append((word[:-1], cat_ids))
        else:
            exact_lookup[word] = cat_ids

    # Sort prefix lookup by length descending so longer prefixes match first
    prefix_lookup.sort(key=lambda x: len(x[0]), reverse=True)

    return exact_lookup, prefix_lookup


# ── Score a single text ───────────────────────────────────────────────────────

def score_text(
    text: str,
    exact_lookup: dict[str, list[int]],
    prefix_lookup: list[tuple[str, list[int]]],
    category_map: dict[int, str],
) -> dict[str, float]:
    """
    Scores a single text against the LIWC dictionary.

    For each word in the text:
      1. Check exact match in exact_lookup
      2. If no exact match, check prefix matches
      3. Accumulate category hits

    Returns a dict of {category_name: proportion} where proportion is
    the percentage of words in the text matching that category.
    This matches LIWC's standard output format.
    """
    tokens = text.lower().split()
    n_words = len(tokens)

    if n_words == 0:
        return {name: 0.0 for name in category_map.values()}

    # Count hits per category
    cat_counts: dict[int, int] = defaultdict(int)

    for token in tokens:
        # Strip punctuation from token for matching
        clean = re.sub(r"[^\w]", "", token)
        if not clean:
            continue

        matched_cats: list[int] = []

        # 1. Exact match
        if clean in exact_lookup:
            matched_cats = exact_lookup[clean]
        else:
            # 2. Prefix match — take the longest matching prefix
            for prefix, cats in prefix_lookup:
                if clean.startswith(prefix):
                    matched_cats = cats
                    break

        for cat_id in matched_cats:
            cat_counts[cat_id] += 1

    # Convert counts to proportions (% of words)
    scores: dict[str, float] = {}
    for cat_id, cat_name in category_map.items():
        count = cat_counts.get(cat_id, 0)
        scores[cat_name] = round(100 * count / n_words, 4)

    # Also add raw word count and word count as WC (standard LIWC output)
    scores["WC"] = n_words

    return scores


# ── Process corpus ────────────────────────────────────────────────────────────

def extract_liwc_features(
    structured_file: str = STRUCTURED_FILE,
    dic_file: str = DIC_FILE,
    output_file: str = OUTPUT_FILE,
    initial_posts_only: bool = INITIAL_POSTS_ONLY,
    text_column: str = TEXT_COLUMN,
) -> pd.DataFrame:
    """
    Extracts LIWC features for all initial posts in the corpus.

    Each row in the output corresponds to one initial post, identified
    by ForumTopicID. The pipeline.py build_features() merges this output
    on ForumTopicID.

    Output columns:
      ForumTopicID, WC, [category_name per LIWC category]
    """
    print("\n[1] Loading corpus...")
    df = pd.read_csv(structured_file)

    if initial_posts_only:
        df = df[df["is_initial_post"]].copy()
        print(f"  Initial posts: {len(df)}")
    else:
        print(f"  All messages: {len(df)}")

    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found. "
            f"Run postprocess.py first to generate text_normalized."
        )

    print("\n[2] Parsing LIWC dictionary...")
    category_map, word_map = parse_dic_file(dic_file)
    exact_lookup, prefix_lookup = build_lookup(word_map)

    print("\n[3] Scoring texts...")
    rows = []
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 200 == 0:
            print(f"  {i}/{n} processed...", end="\r")

        text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        scores = score_text(text, exact_lookup, prefix_lookup, category_map)
        scores["ForumTopicID"] = row["ForumTopicID"]
        rows.append(scores)

    print(f"  {n}/{n} processed.   ")

    # Build output dataframe
    result = pd.DataFrame(rows)

    # Put ForumTopicID first
    cols = ["ForumTopicID", "WC"] + [
        c for c in result.columns if c not in ("ForumTopicID", "WC")
    ]
    result = result[cols]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result.to_csv(output_file, index=False)
    print(f"\n  LIWC output saved: {output_file}")
    print(f"  Shape: {result.shape} ({len(result)} posts, {result.shape[1]-1} features)")

    # Quick sanity check: show mean scores for key affect categories
    affect_cols = [c for c in result.columns
                   if c in ("posemo", "negemo", "anx", "anger", "sad", "affect",
                             "social", "cogproc", "WC")]
    if affect_cols:
        print("\n  Mean scores for key categories:")
        print(result[affect_cols].mean().round(3).to_string())

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Check dictionary file exists
    if not os.path.exists(DIC_FILE):
        # Try alternate path — user may have placed it differently
        alt_path = os.path.join("data", "Dutch_LIWC2007_Dictionary_final.dic")
        if os.path.exists(alt_path):
            print(f"  2015 dictionary not found, using 2007: {alt_path}")
            dic_to_use = alt_path
        else:
            raise FileNotFoundError(
                f"No LIWC dictionary found at {DIC_FILE} or {alt_path}. "
                f"Place either LIWC2015Dutch.dic or "
                f"Dutch_LIWC2007_Dictionary_final.dic in the data/ folder."
            )
    else:
        dic_to_use = DIC_FILE

    result = extract_liwc_features(dic_file=dic_to_use)
    print("\nDone. Run pipeline.py to include LIWC features in model training.")
