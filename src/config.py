# =============================================================================
# config.py  –  single source of truth for all pipeline settings
# =============================================================================

# ── Directories ──────────────────────────────────────────────────────────────
DATA_DIR        = "data"
OUTPUT_DIR      = "output"
PREPROCESS_DIR  = "output/preprocessed"

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

# ── Moderator exclusions ──────────────────────────────────────────────────────
# Raw PosterID UUIDs of confirmed moderators, identified via find_moderators.py.
# Add IDs here after reviewing output/moderator_review.csv.
MODERATOR_POSTER_IDS: set[str] = set([
    "22047A60-621D-4CB5-AC22-D68C649B3990", 
    "4DAC7C7F-D353-4342-96BA-DE923C27E3B6", 
    "516E4BAA-BE68-4D8F-AF69-7462B28EE6A1", 
    "BE493FA7-B016-44E9-ACC8-E1A582ECCA4E",
    "C9E46028-A9F8-4F52-BB96-E76976BF9C60",
    "25B2B59F-CEB6-49DE-82F9-7E50F0955287",
    "7EBB120B-B504-4BB5-B070-D7D57D4E1D60",
    "2C882C5C-B9F0-4A5A-9C54-1973CA3D345E"
     ])

# ── Group-name filters ────────────────────────────────────────────────────────
# Groups whose names contain any of these substrings (case-insensitive) are
# treated as introduction/admin channels and dropped entirely.
# Include word game and off-topic groups identified via find_moderators.py output.
INTRO_GROUP_KEYWORDS = {
    "welkom", 
    "teamberichten",
    "stel je jezelf", 
    "voorstellen", 
    "gedichten", 
    "ontspanning", 
    "ontspanningsruimte",
    "gedichten en schrijfsels",
    "overig",
    "off-topic", 
    "off topic", 
    "wordgame", 
    "woordspel", 
    "woordspelletjes"
    }

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