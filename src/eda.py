# =============================================================================
# eda.py  –  Exploratory Data Analysis on annotated Depression Connect data
#
# Produces figures and summary statistics that inform:
#   - Class imbalance handling strategy (SMOTE vs class weights)
#   - Feature engineering decisions
#   - Train/val/test split stratification
#   - Thesis methodology and results sections
#
# Input:  output/messages_structured.csv      (from postprocess.py)
#         output/annotation_sample_with_context.csv  (your annotated file)
# Output: output/eda/  (figures + summary CSVs)
# =============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
STRUCTURED_FILE  = os.path.join("output", "messages_structured.csv")
ANNOTATED_FILE   = os.path.join("data", "annotation_sample_with_context_claudia - annotation_sample_with_context.csv")
OUTPUT_DIR       = os.path.join("output", "eda")
TEXT_COLUMN      = "text_normalized"
DATE_COLUMN      = "PostDate"
POSTER_COLUMN    = "PosterID"

# Plotting style
sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (8, 5)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def savefig(name: str):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n[1] Loading data...")
    structured = pd.read_csv(STRUCTURED_FILE)
    annotated  = pd.read_csv(ANNOTATED_FILE)

    if DATE_COLUMN in structured.columns:
        structured[DATE_COLUMN] = pd.to_datetime(structured[DATE_COLUMN], errors="coerce")
    if DATE_COLUMN in annotated.columns:
        annotated[DATE_COLUMN]  = pd.to_datetime(annotated[DATE_COLUMN],  errors="coerce")

    print(f"  Structured: {len(structured)} messages, "
          f"{structured['ForumTopicID'].nunique()} threads.")
    print(f"  Annotated:  {len(annotated)} replies, "
          f"labels: {annotated['label'].value_counts().to_dict()}")
    return structured, annotated


# ── Section 1: Corpus overview ────────────────────────────────────────────────

def eda_corpus_overview(structured: pd.DataFrame):
    print("\n[2] Corpus overview...")

    initial = structured[structured["is_initial_post"]]
    replies  = structured[~structured["is_initial_post"]]

    print(f"  Total messages:   {len(structured)}")
    print(f"  Initial posts:    {len(initial)}")
    print(f"  Replies:          {len(replies)}")
    print(f"  Unique users:     {structured[POSTER_COLUMN].nunique()}")
    print(f"  Unique threads:   {structured['ForumTopicID'].nunique()}")

    # Threads with/without replies
    has_replies = structured.groupby("ForumTopicID")["thread_has_replies"].first()
    print(f"  Threads with replies:    {has_replies.sum()}")
    print(f"  Threads without replies: {(~has_replies).sum()}")

    # Save summary
    summary = pd.DataFrame({
        "Metric": [
            "Total messages", "Initial posts", "Replies",
            "Unique users", "Unique threads",
            "Threads with replies", "Threads without replies"
        ],
        "Value": [
            len(structured), len(initial), len(replies),
            structured[POSTER_COLUMN].nunique(),
            structured["ForumTopicID"].nunique(),
            int(has_replies.sum()), int((~has_replies).sum())
        ]
    })
    summary.to_csv(os.path.join(OUTPUT_DIR, "corpus_summary.csv"), index=False)

    # Plot: messages over time
    if DATE_COLUMN in structured.columns:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        structured.set_index(DATE_COLUMN).resample("M")["ForumTopicID"].count().plot(
            ax=ax, color="steelblue"
        )
        ax.set_title("Messages per Month")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Messages")
        savefig("messages_over_time.png")


# ── Section 2: Class distribution ────────────────────────────────────────────

def eda_class_distribution(annotated: pd.DataFrame):
    print("\n[3] Class distribution...")

    # Label distribution
    label_counts = annotated["label"].value_counts()
    print(f"  Label distribution:\n{label_counts.to_string()}")

    total = len(annotated)
    for label, count in label_counts.items():
        print(f"    {label}: {count} ({100*count/total:.1f}%)")

    # Plot: label distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    label_counts.plot(kind="bar", ax=ax, color=["steelblue", "salmon"], edgecolor="white")
    ax.set_title("Annotation Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for bar in ax.patches:
        ax.annotate(
            f"{int(bar.get_height())} ({100*bar.get_height()/total:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=10
        )
    savefig("label_distribution.png")

    # Support type distribution (SS only)
    ss_only = annotated[annotated["label"] == "SS"]
    if "support_type" in annotated.columns and len(ss_only) > 0:
        type_counts = ss_only["support_type"].value_counts()
        print(f"\n  Support type distribution (SS only):\n{type_counts.to_string()}")

        fig, ax = plt.subplots(figsize=(6, 4))
        type_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title("Support Type Distribution (SS replies only)")
        ax.set_xlabel("Support Type")
        ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        savefig("support_type_distribution.png")

    # Class imbalance ratio
    if len(label_counts) == 2:
        majority = label_counts.max()
        minority = label_counts.min()
        ratio = majority / minority
        print(f"\n  Class imbalance ratio: {ratio:.2f}:1")
        if ratio > 3:
            print("  WARNING: significant class imbalance detected.")
            print("  Recommendation: use SMOTE or class weighting in modeling.")


# ── Section 3: Thread-level label rollup ──────────────────────────────────────

def eda_thread_labels(annotated: pd.DataFrame, structured: pd.DataFrame):
    """
    Rolls up reply-level labels to thread level.
    A thread is 'successful' if it has at least one SS reply.
    This produces the target variable for the classification task.
    """
    print("\n[4] Thread-level label rollup...")

    # Only SS/NSS labeled replies
    labeled = annotated[annotated["label"].isin(["SS", "NSS"])].copy()

    # Thread is successful if at least one reply is SS
    thread_labels = (
        labeled.groupby("ForumTopicID")["label"]
        .apply(lambda x: "successful" if "SS" in x.values else "unsuccessful")
        .reset_index()
        .rename(columns={"label": "thread_label"})
    )

    # Threads with no replies from annotation (not in sample) → unsuccessful
    all_threads = structured[structured["is_initial_post"]][["ForumTopicID"]].copy()
    thread_labels_full = all_threads.merge(thread_labels, on="ForumTopicID", how="left")
    thread_labels_full["thread_label"] = thread_labels_full["thread_label"].fillna(
        "unsuccessful"
    )

    counts = thread_labels_full["thread_label"].value_counts()
    print(f"  Thread label distribution:\n{counts.to_string()}")
    total = len(thread_labels_full)
    for label, count in counts.items():
        print(f"    {label}: {count} ({100*count/total:.1f}%)")

    # Save thread labels — this becomes your y variable
    thread_labels_full.to_csv(
        os.path.join(OUTPUT_DIR, "thread_labels.csv"), index=False
    )
    print("  Thread labels saved -> output/eda/thread_labels.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["steelblue", "salmon"], edgecolor="white")
    ax.set_title("Thread-Level Label Distribution")
    ax.set_xlabel("Thread Label")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for bar in ax.patches:
        ax.annotate(
            f"{int(bar.get_height())} ({100*bar.get_height()/total:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=10
        )
    savefig("thread_label_distribution.png")

    return thread_labels_full


# ── Section 4: Message length distribution ────────────────────────────────────

def eda_message_lengths(structured: pd.DataFrame, annotated: pd.DataFrame):
    print("\n[5] Message length distribution...")

    if TEXT_COLUMN not in structured.columns:
        print("  SKIP: text_normalized column not found.")
        return

    structured = structured.copy()
    structured["word_count"] = structured[TEXT_COLUMN].fillna("").apply(
        lambda x: len(str(x).split())
    )

    # By message type
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, subset) in zip(axes, [
        ("Initial Posts", structured[structured["is_initial_post"]]),
        ("Replies",       structured[~structured["is_initial_post"]]),
    ]):
        subset["word_count"].clip(upper=500).plot(
            kind="hist", bins=40, ax=ax, color="steelblue", edgecolor="white"
        )
        ax.set_title(f"Word Count Distribution — {label}")
        ax.set_xlabel("Word Count (capped at 500)")
        ax.set_ylabel("Frequency")
        median = subset["word_count"].median()
        ax.axvline(median, color="red", linestyle="--", label=f"Median: {median:.0f}")
        ax.legend()
    plt.tight_layout()
    savefig("message_length_distribution.png")

    # By label in annotated sample
    if TEXT_COLUMN in annotated.columns and "label" in annotated.columns:
        annotated = annotated.copy()
        annotated["word_count"] = annotated[TEXT_COLUMN].fillna("").apply(
            lambda x: len(str(x).split())
        )
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for label, group in annotated.groupby("label"):
            group["word_count"].clip(upper=300).plot(
                kind="hist", bins=30, ax=ax, alpha=0.6,
                label=label, edgecolor="white"
            )
        ax.set_title("Reply Word Count by Label")
        ax.set_xlabel("Word Count (capped at 300)")
        ax.set_ylabel("Frequency")
        ax.legend()
        savefig("reply_length_by_label.png")

        # Print summary stats
        print("  Word count by label:")
        print(annotated.groupby("label")["word_count"].describe().round(1).to_string())


# ── Section 5: User activity distribution ────────────────────────────────────

def eda_user_activity(structured: pd.DataFrame):
    print("\n[6] User activity distribution...")

    if POSTER_COLUMN not in structured.columns:
        print("  SKIP: PosterID column not found.")
        return

    user_counts = structured.groupby(POSTER_COLUMN).size().sort_values(ascending=False)

    print(f"  Total unique users: {len(user_counts)}")
    print(f"  Messages per user — median: {user_counts.median():.0f}, "
          f"max: {user_counts.max()}")

    # Categorize users by activity level — used for error analysis in SQ4
    def activity_level(n):
        if n == 1:
            return "one-time"
        elif n <= 5:
            return "low"
        elif n <= 20:
            return "medium"
        else:
            return "high"

    structured = structured.copy()
    user_activity_map = user_counts.apply(activity_level).to_dict()
    structured["user_activity"] = structured[POSTER_COLUMN].map(user_activity_map)

    activity_counts = structured["user_activity"].value_counts()
    print(f"  User activity levels:\n{activity_counts.to_string()}")

    # Save user activity map for use in modeling/error analysis
    pd.DataFrame({
        POSTER_COLUMN: list(user_activity_map.keys()),
        "activity_level": list(user_activity_map.values()),
        "message_count": [user_counts[u] for u in user_activity_map.keys()]
    }).to_csv(os.path.join(OUTPUT_DIR, "user_activity.csv"), index=False)

    # Plot: user activity distribution
    fig, ax = plt.subplots(figsize=FIGSIZE)
    user_counts.clip(upper=100).plot(
        kind="hist", bins=40, ax=ax, color="steelblue", edgecolor="white"
    )
    ax.set_title("Messages per User (capped at 100)")
    ax.set_xlabel("Number of Messages")
    ax.set_ylabel("Number of Users")
    savefig("user_activity_distribution.png")

    return structured


# ── Section 6: Reply count distribution ──────────────────────────────────────

def eda_reply_counts(structured: pd.DataFrame):
    print("\n[7] Reply count distribution per thread...")

    reply_counts = (
        structured[~structured["is_initial_post"]]
        .groupby("ForumTopicID")
        .size()
        .reset_index(name="reply_count")
    )

    print(f"  Threads with replies: {len(reply_counts)}")
    print(f"  Replies per thread — "
          f"median: {reply_counts['reply_count'].median():.0f}, "
          f"mean: {reply_counts['reply_count'].mean():.1f}, "
          f"max: {reply_counts['reply_count'].max()}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    reply_counts["reply_count"].clip(upper=30).plot(
        kind="hist", bins=30, ax=ax, color="steelblue", edgecolor="white"
    )
    ax.set_title("Replies per Thread (capped at 30)")
    ax.set_xlabel("Number of Replies")
    ax.set_ylabel("Number of Threads")
    median = reply_counts["reply_count"].median()
    ax.axvline(median, color="red", linestyle="--", label=f"Median: {median:.0f}")
    ax.legend()
    savefig("reply_count_distribution.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    ensure_output_dir()
    structured, annotated = load_data()
    eda_corpus_overview(structured)
    eda_class_distribution(annotated)
    thread_labels = eda_thread_labels(annotated, structured)
    eda_message_lengths(structured, annotated)
    eda_user_activity(structured)
    eda_reply_counts(structured)

    print("\nEDA complete.")
    print("Outputs saved to output/eda/")
    print("\nKey outputs for thesis:")
    print("  output/eda/corpus_summary.csv      -> Dataset description table")
    print("  output/eda/thread_labels.csv        -> Your y variable for modeling")
    print("  output/eda/user_activity.csv        -> For error analysis in SQ4")
    print("  output/eda/*.png                    -> Figures for results section")


if __name__ == "__main__":
    run()
