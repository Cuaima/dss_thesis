# Depression Connect — Peer Support Prediction Pipeline

**Thesis:** Predicting Supportive Peer Responses in an Online Depression Forum  
**Programme:** MSc Data Science & Society, Tilburg University  
**Dataset:** Depression Connect (Apr 2019 – Oct 2022) — 16 000+ messages, 1 280 threads, 646 users

---

## Overview

This pipeline classifies forum threads as *socially supported* (SS) or *not socially supported* (NSS) based on the content of the initial post. It uses lexical, structural, and LIWC psycholinguistic features with four classifiers (Logistic Regression, SVM, Random Forest, XGBoost).

The pipeline has a mandatory human annotation step between `postprocess.py` and `eda.py`. These two stages are intentionally separate.

---

## Repository Structure

```
thesis_project/
├── src/
│   ├── config.py                  # All pipeline settings (single source of truth)
│   ├── preprocess.py              # Load → filter → clean → anonymize
│   ├── postprocess.py             # Thread structure → normalize → annotation sample
│   ├── find_moderators.py         # One-time diagnostic: identify moderator accounts
│   ├── eda.py                     # EDA on annotated data → figures + thread_labels.csv
│   ├── liwc_extractor.py          # LIWC feature extraction from Dutch .dic files
│   ├── pipeline.py                # Full ML pipeline: features → split → train → evaluate
│   └── custom_text_anonymizer/    # spaCy-based Dutch NER anonymizer
├── data/                          # Raw data (PRIVATE — not in repo)
├── output/                        # All generated outputs (PRIVATE — not in repo)
│   ├── preprocessed/              # Output from preprocess.py
│   ├── eda/                       # Figures and CSVs from eda.py
│   └── results/                   # Model results from pipeline.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

> **Data privacy:** `data/` and `output/` are excluded from version control. The dataset is pseudonymized and held under a formal data-sharing agreement.

---

## Setup

### Requirements

- Anaconda with the `thesis_env` conda environment (Python 3.13, spaCy 3.8, `nl_core_news_lg`)
- The base Anaconda Python (3.9) does **not** have spaCy and will silently skip anonymization

### Why the environment matters

`custom_text_anonymizer` loads the Dutch spaCy model `nl_core_news_lg` at import time. If you run without `thesis_env` active, the shell defaults to Anaconda base (Python 3.9, no spaCy), the import fails, and anonymization is silently skipped.

### Activate the environment

```bash
conda activate thesis_env
```

Do this once per terminal session before running any pipeline script.

### Install (first time only)

```bash
conda activate thesis_env
pip install -r requirements.txt
```

### Verify the anonymizer works

```bash
conda activate thesis_env
PYTHONPATH=./src python -c "from custom_text_anonymizer import anonymize; print('anonymizer OK')"
```

If you see `anonymizer OK`, text anonymization is active. Any other output means the wrong environment is active.

All scripts must be run from the **project root** with `PYTHONPATH=./src`:

```bash
conda activate thesis_env
PYTHONPATH=./src python src/<script>.py
```

---

## Pipeline

### Step 0 — One-time moderator identification

```bash
PYTHONPATH=./src python src/find_moderators.py
```

Searches messages for self-identified moderators ("moderator", "moderatoren", "beheerder") and writes `output/moderator_review.csv`. Open the file, fill in the `is_moderator` column (TRUE/FALSE), then copy the TRUE `PosterID` values into `MODERATOR_POSTER_IDS` in `src/config.py`.

Also prints group-level word-count statistics to help identify word-game and off-topic groups. Add any confirmed group name substrings to `INTRO_GROUP_KEYWORDS` in `config.py`.

---

### Step 1 — Preprocessing

```bash
PYTHONPATH=./src python src/preprocess.py
```

**Input:** `data/` (accounts.csv, messages.csv, topics.csv, groups.csv)  
**Output:** `output/preprocessed/`

| File | Contents |
|---|---|
| `messages_community.csv` | Cleaned, anonymized community messages |
| `accounts_community.csv` | Filtered accounts |
| `topics_community.csv` | Filtered topics |
| `groups_community.csv` | Filtered groups |

Steps applied:
1. Load raw CSVs
2. Filter to community accounts (IDs 2 and 3); drop test/demo accounts
3. Remove moderator accounts (`MODERATOR_POSTER_IDS` in config.py)
4. Drop intro/admin/off-topic groups (`INTRO_GROUP_KEYWORDS` in config.py)
5. Strip HTML and quote blocks from message text
6. Drop messages below `MIN_WORD_COUNT` words
7. Anonymize named entities (person names, locations, organisations) using Dutch spaCy NER

---

### Step 2 — Postprocessing

```bash
PYTHONPATH=./src python src/postprocess.py
```

**Input:** `output/preprocessed/messages_community.csv`  
**Output:** `output/`

| File | Contents |
|---|---|
| `messages_structured.csv` | Full dataset with thread structure and normalized text |
| `annotation_sample.csv` | 500-reply sample for manual annotation (blank label columns) |
| `annotation_sample_with_context.csv` | Same sample + matched initial post text for context |

Steps applied:
1. Build thread structure (sort by thread + date; flag initial posts vs replies)
2. Label threads by whether they received any replies (structural label only)
3. Normalize text (lowercase, collapse repeated characters, normalize whitespace)
4. Sanity-check message lengths
5. Extract annotation sample — one reply per thread, excluding self-replies

---

### Step 3 — Manual annotation (human step)

Open `output/annotation_sample_with_context.csv` and fill in:

| Column | Values |
|---|---|
| `label` | `SS` (social support) or `NSS` (not social support) |
| `support_type` | `informational` / `emotional` / `other` / `N/A` |

Share `output/annotation_sample.csv` (without context) with your second rater and calculate inter-rater agreement (Cohen's Kappa) before merging labels.

Save the completed annotation file to:
```
data/annotation_sample_with_context_claudia - annotation_sample_with_context.csv
```

---

### Step 4 — EDA

```bash
PYTHONPATH=./src python src/eda.py
```

**Input:**
- `output/messages_structured.csv`
- `data/annotation_sample_with_context_claudia - annotation_sample_with_context.csv`

**Output:** `output/eda/` — figures and summary CSVs including `thread_labels.csv` and `user_activity.csv`

Produces class distribution plots, temporal activity charts, message length distributions, and the thread-level label file consumed by `pipeline.py`.

---

### Step 5 — LIWC feature extraction

```bash
PYTHONPATH=./src python src/liwc_extractor.py
```

**Input:**
- `output/messages_structured.csv`
- `data/LIWC2015Dutch.dic` (or `Dutch_LIWC2007_Dictionary_final.dic`)

**Output:** `output/liwc_output.csv` — one row per thread with one column per LIWC category

Parses the Dutch LIWC `.dic` dictionary directly (no commercial LIWC software required) and computes category proportion scores for each message.

---

### Step 6 — ML pipeline

```bash
PYTHONPATH=./src python src/pipeline.py
```

**Input:**
- `output/messages_structured.csv`
- `output/eda/thread_labels.csv`
- `output/eda/user_activity.csv`
- `output/liwc_output.csv` (optional)

**Output:** `output/results/`

Steps:
1. Build features — lexical (TF-IDF), structural (length, reply index), LIWC
2. User-stratified 70/15/15 train/validation/test split (`GroupShuffleSplit` on `PosterID`)
3. Train LR, SVM, Random Forest, XGBoost with SMOTE and class weighting
4. Evaluate on held-out test set (primary metric: macro F1)
5. Error analysis — confusion matrices, SHAP feature importances, subgroup analysis
6. Subclassification: informational vs emotional support (SQ3)

---

## Configuration reference

All settings are in `src/config.py`:

| Key | Purpose |
|---|---|
| `DATA_DIR` | Raw data directory (`data/`) |
| `OUTPUT_DIR` | Top-level output directory (`output/`) |
| `PREPROCESS_DIR` | Preprocessed outputs (`output/preprocessed/`) |
| `SUPERUSER_ACCOUNT_IDS` | Account IDs to exclude (test/demo) |
| `COMMUNITY_ACCOUNT_IDS` | Account IDs for the two real communities |
| `MODERATOR_POSTER_IDS` | UUIDs of confirmed moderators to exclude |
| `INTRO_GROUP_KEYWORDS` | Group name substrings marking admin/off-topic channels |
| `MIN_WORD_COUNT` | Minimum words for a message to be kept |
| `ANONYMIZE_TEXT` | Whether to run NER anonymization |
| `RANDOM_STATE` | Seed for reproducibility (default: 42) |
