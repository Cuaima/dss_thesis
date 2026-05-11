# =============================================================================
# config.py  –  single source of truth for all pipeline settings
# =============================================================================

# ── Directories ──────────────────────────────────────────────────────────────
DATA_DIR   = "data"
OUTPUT_DIR = "output"

# ── Source files ─────────────────────────────────────────────────────────────
CSV_FILES = ["accounts", "groups", "messages", "topics"]

# ── Column names ─────────────────────────────────────────────────────────────
ID_COLUMN      = "PosterID"
DATE_COLUMNS   = ["PostDate", "StartDate"]
TEXT_COLUMN    = "MessageText"
TEXT_COLUMNS_TO_CLEAN = {TEXT_COLUMN, "Name",} 

# ── Account types ─────────────────────────────────────────────────────────────
# 1=test, 2=community1, 3=community2, 4=demo
SUPERUSER_ACCOUNT_IDS  = {1, 4}   # test + demo → exclude their posters
COMMUNITY_ACCOUNT_IDS  = {2, 3}   # the two real communities

# ── Group-name filters ────────────────────────────────────────────────────────
# Groups whose names contain any of these substrings (case-insensitive) are
# treated as introduction/admin channels and dropped entirely.
INTRO_GROUP_KEYWORDS = {"welkom", "teamberichten", "stel je jezelf", "voorstellen"}

# ── Text quality filters ──────────────────────────────────────────────────────
MIN_WORD_COUNT      = 1     # posts shorter than this are dropped
LANGUAGE_FILTER     = False   # drop non-English posts
TARGET_LANGUAGE     = "nl"

# ── Anonymization ─────────────────────────────────────────────────────────────
ANONYMIZE_TEXT        = True
REPLACE_ORIGINAL_TEXT = True
EXPORT_ENTITY_REVIEW  = True

# ── Classification dataset ────────────────────────────────────────────────────
# For build_classification_dataset: only use messages from community accounts
CLASSIFICATION_ACCOUNT_IDS = COMMUNITY_ACCOUNT_IDS