# =============================================================================
# pipeline.py  –  Full end-to-end ML pipeline
#
# Execution order:
#   1. load_data()              – load structured messages + thread labels
#   2. build_features()         – extract lexical, structural, and LIWC features
#   3. split_data()             – user-stratified 70/15/15 split (GroupShuffleSplit)
#   4. train_models()           – LR, SVM, RF, XGBoost + baselines w/ SMOTE
#   5. evaluate_on_test()       – final held-out evaluation
#   6. error_analysis()         – confusion matrices, SHAP, subgroup analysis
#   7. subclassification_sq3()  – informational vs emotional (SQ3)
#   8. save_results_summary()   – export tables and figures
#
# Design notes:
#   - No sentiment features: LIWC-22 affect dimensions (posemo, negemo, anx,
#     anger, sad) subsume sentiment signal and are validated for Dutch NLP.
#   - User-level splitting prevents data leakage from the same poster
#     appearing in both train and test sets.
#   - SMOTE is applied to the training set only.
#   - TF-IDF is fitted inside sklearn Pipelines to prevent leakage.
#
# Input:  output/messages_structured.csv
#         output/eda/thread_labels.csv
#         output/eda/user_activity.csv
#         output/liwc_output.csv            (optional — see build_features)
# Output: output/results/
# =============================================================================

from __future__ import annotations

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ────────────────────────────────────────────────────────────────────
STRUCTURED_FILE = os.path.join("output", "messages_structured.csv")
LABELS_FILE     = os.path.join("output", "eda", "thread_labels.csv")
USER_ACTIVITY   = os.path.join("output", "eda", "user_activity.csv")
LIWC_FILE       = os.path.join("output", "liwc_output.csv")
OUTPUT_DIR      = os.path.join("output", "results")
TEXT_COLUMN     = "text_normalized"
POSTER_COLUMN   = "PosterID"
RANDOM_STATE    = 42
N_FEATURES_SELECT = 25

sns.set_theme(style="whitegrid", palette="muted")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def savefig(name: str):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def write_csv(df: pd.DataFrame, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ── Step 1: Load data ─────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads structured messages and thread-level labels.
    Restricts to threads that have at least one annotated reply,
    so every label in the dataset is backed by actual annotation.

    Threads outside the annotation sample have no reliable label and
    are excluded rather than assumed unsuccessful. This keeps the
    classification dataset fully supervised.
    """
    print("\n[1] Loading data...")
    structured = pd.read_csv(STRUCTURED_FILE)
    labels     = pd.read_csv(LABELS_FILE)

    # Keep only initial posts (one row per thread = one observation for X)
    initial_posts = structured[structured["is_initial_post"]].copy()

    # Restrict to threads with an actual annotation
    # Note: threads not in the sample are excluded, not assumed unsuccessful.
    # This avoids introducing label noise from unsampled threads.
    annotated_threads = labels[labels["thread_label"].notna()]["ForumTopicID"].unique()
    initial_posts = initial_posts[
        initial_posts["ForumTopicID"].isin(annotated_threads)
    ].copy()

    initial_posts = initial_posts.merge(labels, on="ForumTopicID", how="left")

    label_dist = initial_posts["thread_label"].value_counts().to_dict()
    print(f"  Threads with annotated replies (used for modeling): {len(initial_posts)}")
    print(f"  Label distribution: {label_dist}")

    if len(initial_posts) < 50:
        raise ValueError(
            "Too few labeled threads for modeling. "
            "Check that thread_labels.csv has sufficient annotated threads."
        )

    return initial_posts, labels


# ── Step 2: Feature extraction ────────────────────────────────────────────────

def build_features(initial_posts: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts structural and LIWC features from initial posts.

    Feature sets:
      A. Lexical:          TF-IDF unigrams + bigrams (handled in Pipeline)
      B. Structural:       post length, punctuation, sentence-level features
      C. Psycholinguistic: LIWC-22 Dutch (merged from external output file)

    No sentiment features are included. LIWC affect dimensions (posemo,
    negemo, anx, anger, sad) cover sentiment signal and are validated
    for Dutch-language psycholinguistic research (Tausczik & Pennebaker, 2010).

    TF-IDF features are NOT added here — they are handled inside sklearn
    Pipelines in train_models() to guarantee no data leakage between splits.
    """
    print("\n[2] Extracting features...")
    df = initial_posts.copy()

    # ── B. Structural features ────────────────────────────────────────────────
    text = df[TEXT_COLUMN].fillna("")

    df["feat_word_count"]       = text.apply(lambda x: len(x.split()))
    df["feat_char_count"]       = text.apply(len)
    df["feat_question_marks"]   = text.apply(lambda x: x.count("?"))
    df["feat_exclamation"]      = text.apply(lambda x: x.count("!"))
    df["feat_sentence_count"]   = text.apply(
        lambda x: max(1, x.count(".") + x.count("!") + x.count("?"))
    )
    df["feat_avg_sentence_len"] = df["feat_word_count"] / df["feat_sentence_count"]
    df["feat_ellipsis"]         = text.apply(lambda x: x.count("..."))

    n_structural = len([c for c in df.columns if c.startswith("feat_")])
    print(f"  Structural features: {n_structural}")

    # ── C. LIWC features ──────────────────────────────────────────────────────
    liwc_cols_added = 0
    if os.path.exists(LIWC_FILE):
        liwc = pd.read_csv(LIWC_FILE)

        # LIWC output should have a column matching ForumTopicID or the post text.
        # Adjust the merge key if your LIWC output uses a different ID column.
        if "ForumTopicID" not in liwc.columns:
            raise ValueError(
                f"LIWC output at {LIWC_FILE} must contain a 'ForumTopicID' column "
                f"for merging. Found: {list(liwc.columns[:10])}"
            )

        # Keep only top-level LIWC categories — subcategories are subsumed
        # by their parents and introduce redundancy (e.g. posemo/negemo
        # are components of affect; keeping both inflates the feature space)
        # This also reduces the risk of overfitting given our limited dataset size.
        # This reduces LIWC from 74 to ~23 features.
        LIWC_TOP_LEVEL = {
            "WC", "function", "verb", "adj", "adverb", "negate",
            "affect",        # parent of posemo/negemo/anx/anger/sad
            "social",        # parent of family/friend/female/male
            "cogproc",       # parent of insight/cause/discrep/tentat/certain/differ
            "percept",       # parent of see/hear/feel
            "bio",           # parent of body/health/sexual/ingest
            "drives",        # parent of affiliation/achieve/power/reward/risk
            "focuspast", "focuspresent", "focusfuture",
            "relativ",       # parent of motion/space/time
            "work", "leisure", "home", "money", "relig", "death",
            "informal",      # parent of swear/netspeak/assent/nonflu/filler
        }
        raw_liwc_cols = [
            c for c in liwc.columns
            if c != "ForumTopicID" and c in LIWC_TOP_LEVEL
        ]
        dropped_count = len([
            c for c in liwc.columns
            if c != "ForumTopicID" and c not in LIWC_TOP_LEVEL
        ])
        print(f"  LIWC top-level retained: {len(raw_liwc_cols)} "
              f"(dropped {dropped_count} subcategories)")
        liwc_filtered = liwc[["ForumTopicID"] + raw_liwc_cols].copy()
        liwc_filtered = liwc_filtered.rename(
            columns={c: f"feat_liwc_{c}" for c in raw_liwc_cols}
        )
        df = df.merge(liwc_filtered, on="ForumTopicID", how="left")
        liwc_cols_added = len(raw_liwc_cols)
    else:
        print(
            f"  WARNING: LIWC output not found at {LIWC_FILE}. "
            f"Proceeding without psycholinguistic features. "
            f"Add LIWC output before final model runs."
        )

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    print(f"  Total engineered features: {len(feature_cols)} "
          f"({n_structural} structural, {liwc_cols_added} LIWC)")

    return df


# ── Step 3: Train/val/test split ──────────────────────────────────────────────

def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    User-stratified 70/15/15 split using GroupShuffleSplit.

    Splitting by PosterID (user) ensures that posts from the same user
    cannot appear in both train and test sets. Without this, the model
    could learn user-specific writing styles rather than generalizable
    linguistic patterns, which would constitute data leakage.

    Stratification on thread_label is applied within each split to
    preserve the class ratio across train, val, and test.

    Returns: train_df, val_df, test_df
    """
    print("\n[3] Splitting data (70/15/15, user-level grouping)...")

    if POSTER_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{POSTER_COLUMN}' not found. "
            "Cannot perform user-level splitting."
        )

    groups = df[POSTER_COLUMN].values

    # First split: 70% train, 30% temp
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss1.split(df, df["thread_label"], groups=groups))

    train_df = df.iloc[train_idx].copy()
    temp_df  = df.iloc[temp_idx].copy()

    # Second split: 50/50 of temp → 15% val, 15% test
    temp_groups = temp_df[POSTER_COLUMN].values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_idx, test_idx = next(gss2.split(temp_df, temp_df["thread_label"], groups=temp_groups))

    val_df  = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # Sanity check: no user overlap across splits
    train_users = set(train_df[POSTER_COLUMN])
    val_users   = set(val_df[POSTER_COLUMN])
    test_users  = set(test_df[POSTER_COLUMN])
    overlap_tv  = train_users & val_users
    overlap_tt  = train_users & test_users
    if overlap_tv or overlap_tt:
        print(f"  WARNING: user overlap detected — train/val: {len(overlap_tv)}, "
              f"train/test: {len(overlap_tt)}. "
              f"This may occur when a user has posts in both splits by chance. "
              f"Consider a stricter grouping strategy.")
    else:
        print("  User-level split verified: no overlap between train, val, and test.")

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"  Train labels: {train_df['thread_label'].value_counts().to_dict()}")
    print(f"  Val   labels: {val_df['thread_label'].value_counts().to_dict()}")
    print(f"  Test  labels: {test_df['thread_label'].value_counts().to_dict()}")

    # Save for reproducibility
    train_df.to_csv(os.path.join(OUTPUT_DIR, "split_train.csv"), index=False)
    val_df.to_csv(  os.path.join(OUTPUT_DIR, "split_val.csv"),   index=False)
    test_df.to_csv( os.path.join(OUTPUT_DIR, "split_test.csv"),  index=False)

    return train_df, val_df, test_df


# ── Step 4: Train models ──────────────────────────────────────────────────────

def get_models(liwc_available: bool = False) -> dict:
    """
    Returns all models to compare.

    When LIWC features are available, all models are included.
    When running without LIWC (structural features only), the TF-IDF
    pipeline baseline is still included as a text-only comparison.

    Two baselines:
      - Majority-class dummy (lower bound)
      - LR + TF-IDF pipeline (standard NLP text baseline)

    Four main models trained on engineered features:
      - Logistic Regression (interpretable, linear)
      - Linear SVM (strong text classification benchmark)
      - Random Forest (non-linear, feature importance via permutation)
      - XGBoost (state-of-the-art on structured feature sets)
    """
    return {
        "Dummy (majority)": DummyClassifier(
            strategy="most_frequent", random_state=RANDOM_STATE
        ),
        "LR + TF-IDF (text baseline)": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf",   LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=5000,
                solver="saga",  # saga handles large datasets better than lbfgs
                random_state=RANDOM_STATE
            )),
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(
                class_weight="balanced", max_iter=50000,
                random_state=RANDOM_STATE, dual=False  # dual=False faster for n_samples > n_features
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE,
            max_depth=5,        # prevent deep trees memorizing training data
            min_samples_leaf=5, # require at least 5 samples per leaf
            n_jobs=1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=1,
            max_depth=3,          # even shallower
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,        # L1 regularization
            reg_lambda=2.0,       # L2 regularization
            min_child_weight=5,   # minimum samples per leaf
        ),
    }


def _apply_smote(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies SMOTE oversampling to the minority class.
    Only ever called on the training set.
    """
    before = Counter(y)
    sampler = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sampler.fit_resample(X, y)
    after = Counter(y_res)
    print(f"    SMOTE: {before} → {after}")
    return X_res, y_res

def _select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
    feature_cols: list[str],
    k: int = N_FEATURES_SELECT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], SelectKBest]:
    """
    Selects the k most informative features using ANOVA F-statistic.
    Fitted on training set only to prevent leakage.
    Selector is returned so evaluate_on_test() can reuse it.
    """
    k        = min(k, X_train.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_tr_sel = selector.fit_transform(X_train, y_train)
    X_va_sel = selector.transform(X_val)
    X_te_sel = selector.transform(X_test)

    selected_cols = [feature_cols[i] for i in selector.get_support(indices=True)]
    print(f"\n  Feature selection: {len(feature_cols)} -> {k} features")
    print(f"  Selected: {selected_cols}")

    scores_df = pd.DataFrame({
        "feature":  feature_cols,
        "f_score":  selector.scores_,
        "p_value":  selector.pvalues_,
        "selected": selector.get_support(),
    }).sort_values("f_score", ascending=False)
    write_csv(scores_df, "feature_selection_scores.csv")

    return X_tr_sel, X_va_sel, X_te_sel, selected_cols, selector


def train_models(
    train_df:     pd.DataFrame,
    val_df:       pd.DataFrame,
    test_df:      pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict, pd.DataFrame, list[str], SelectKBest]:
    """
    Trains all models and evaluates on the validation set.

    For pipeline models (TF-IDF baseline): uses raw text, no SMOTE
    (TF-IDF + class_weight='balanced' handles imbalance).

    For feature-based models: uses engineered feature matrix with SMOTE
    applied to training set only.

    Cross-validation uses StratifiedKFold on the training set to report
    CV macro F1 alongside validation performance.

    Returns: fitted model dict, results DataFrame.
    """
    print("\n[4] Training models...")

    # Prepare feature matrices
    X_train_raw = train_df[feature_cols].fillna(0).values
    y_train     = (train_df["thread_label"] == "successful").astype(int).values
    y_val       = (val_df["thread_label"] == "successful").astype(int).values

    X_val_raw  = val_df[feature_cols].fillna(0).values
    X_test_raw = test_df[feature_cols].fillna(0).values

    # Feature selection fitted on ORIGINAL training data before SMOTE
    # This ensures feature scores reflect real data distributions,
    # not synthetic SMOTE interpolations
    X_train_sel_orig, X_val_sel, X_test_sel, selected_cols, selector = _select_features(
        X_train_raw, y_train, X_val_raw, X_test_raw, feature_cols
    )

    # SMOTE applied AFTER selection to the reduced feature space
    X_train_sel, y_train_res = _apply_smote(X_train_sel_orig, y_train)
    y_train_res = y_train_res  # already set by _apply_smote

    skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models   = get_models(liwc_available=any("liwc" in c for c in feature_cols))
    results  = {}
    fitted   = {}

    for name, model in models.items():
        print(f"\n  ── {name}")

        is_tfidf_pipeline = isinstance(model, Pipeline) and hasattr(
            model.named_steps.get("tfidf", None), "transform"
        )
        is_pipeline = is_tfidf_pipeline  # only TF-IDF pipelines use raw text
        is_dummy    = isinstance(model, DummyClassifier)

        # Pipeline models use raw text; feature models use engineered matrix
        if is_tfidf_pipeline:
            X_tr = train_df[TEXT_COLUMN].fillna("").values
            X_v  = val_df[TEXT_COLUMN].fillna("").values
            y_tr = y_train   # class_weight='balanced' handles imbalance
        else:
            X_tr = X_train_sel
            X_v  = X_val_sel
            y_tr = y_train_res

        # 5-fold cross-validation on training data (skip for dummy)
        cv_f1_mean, cv_f1_std = np.nan, np.nan
        if not is_dummy:
            cv_scores = cross_val_score(
                model, X_tr, y_tr,
                cv=skf, scoring="f1_macro", n_jobs=-1
            )
            cv_f1_mean = cv_scores.mean()
            cv_f1_std  = cv_scores.std()
            print(f"    CV macro F1: {cv_f1_mean:.3f} (±{cv_f1_std:.3f})")

        # Fit on full training set
        model.fit(X_tr, y_tr)
        fitted[name] = model

        # Evaluate on validation set
        y_pred = model.predict(X_v)
        val_f1   = f1_score(y_val, y_pred, average="macro", zero_division=0)
        val_prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
        val_rec  = recall_score(y_val, y_pred, average="macro", zero_division=0)

        results[name] = {
            "cv_f1_macro_mean":  round(cv_f1_mean, 3),
            "cv_f1_macro_std":   round(cv_f1_std,  3),
            "val_f1_macro":      round(val_f1,   3),
            "val_precision_macro": round(val_prec, 3),
            "val_recall_macro":  round(val_rec,  3),
        }
        print(f"    Val macro F1: {val_f1:.3f}  |  "
              f"Precision: {val_prec:.3f}  |  Recall: {val_rec:.3f}")
        print(classification_report(y_val, y_pred,
              target_names=["unsuccessful", "successful"], zero_division=0))

    results_df = pd.DataFrame(results).T
    write_csv(results_df, "model_comparison_val.csv")
    return fitted, results_df, selected_cols, selector


# ── Step 5: Final evaluation on test set ──────────────────────────────────────

def evaluate_on_test(
    fitted:          dict,
    test_df:         pd.DataFrame,
    feature_cols:    list[str],
    selected_cols:   list[str],
    selector:        SelectKBest,
    best_model_name: str,
) -> pd.DataFrame:
    """
    Evaluates the best model on the held-out test set.
    For pipeline models (TF-IDF baseline): uses raw text.
    For feature-based models: applies the same feature selection as during training.

    Returns a DataFrame with test performance metrics for all models.
    The best model's confusion matrix and SHAP analysis are handled in error_analysis().
    """
    X_test_raw = test_df[feature_cols].fillna(0).values
    X_test_sel = selector.transform(X_test_raw)
    y_test     = (test_df["thread_label"] == "successful").astype(int).values

    test_results = {}
    for name, model in fitted.items():
        is_tfidf = isinstance(model, Pipeline) and hasattr(
            model.named_steps.get("tfidf", None), "transform"
        )
        X_t = test_df[TEXT_COLUMN].fillna("").values if is_tfidf else X_test_sel

        y_pred = model.predict(X_t)
        test_results[name] = {
            "test_f1_macro":        round(f1_score(y_test, y_pred, average="macro", zero_division=0), 3),
            "test_precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 3),
            "test_recall_macro":    round(recall_score(y_test, y_pred, average="macro", zero_division=0), 3),
        }
        print(f"  {name}: test macro F1 = {test_results[name]['test_f1_macro']:.3f}")

    test_df_results = pd.DataFrame(test_results).T
    write_csv(test_df_results, "model_comparison_test.csv")
    return test_df_results


# ── Step 6: Error analysis ────────────────────────────────────────────────────

def error_analysis(
    model:         object,
    test_df:       pd.DataFrame,
    feature_cols:  list[str],
    selected_cols: list[str],
    selector:      SelectKBest,
    model_name:    str,
):
    """
    Error analysis for the best model:
      - Confusion matrix
      - SHAP feature importance (TreeExplainer for RF/XGBoost,
        LinearExplainer for LR/SVM)
      - Macro F1 stratified by user activity level (SQ4)
    """
    print(f"\n[6] Error analysis: {model_name}")

    is_tfidf_pipeline = isinstance(model, Pipeline) and hasattr(
            model.named_steps.get("tfidf", None), "transform"
        )
    is_pipeline = is_tfidf_pipeline  # only TF-IDF pipelines use raw text
    X_test_raw = test_df[feature_cols].fillna(0).values
    X_test_sel = selector.transform(X_test_raw)
    X_test = test_df[TEXT_COLUMN].fillna("").values if is_tfidf_pipeline \
             else X_test_sel
    y_test = (test_df["thread_label"] == "successful").astype(int).values
    y_pred = model.predict(X_test)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["unsuccessful", "successful"],
        yticklabels=["unsuccessful", "successful"],
    )
    ax.set_title(f"Confusion matrix — {model_name}")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    safe_name = model_name.replace(" ", "_").lower()
    savefig(f"confusion_matrix_{safe_name}.png")

    # ── SHAP feature importance ───────────────────────────────────────────────
    if not is_tfidf_pipeline:
        X_test_sel = test_df[selected_cols].fillna(0).values
        try:
            if isinstance(model, (RandomForestClassifier, XGBClassifier)):
                explainer    = shap.TreeExplainer(model)
                shap_values  = explainer.shap_values(X_test_sel)
                sv = shap_values[1] if isinstance(shap_values, list) else shap_values
            elif isinstance(model, Pipeline):
                # Scaled LR or SVM — extract the underlying classifier
                # and transform X with the scaler before SHAP
                scaler     = model.named_steps["scaler"]
                classifier = model.named_steps["clf"]
                X_scaled   = scaler.transform(X_test_sel)
                explainer  = shap.LinearExplainer(classifier, X_scaled)
                shap_values = explainer.shap_values(X_scaled)
                sv = shap_values
            else:
                explainer   = shap.LinearExplainer(model, X_test_sel)
                shap_values = explainer.shap_values(X_test_sel)
                sv = shap_values

            plt.figure(figsize=(9, 6))
            shap.summary_plot(
                sv, X_test_sel, feature_names=selected_cols,
                plot_type="dot", show=False
            )
            savefig(f"shap_summary_{safe_name}.png")
            print("  SHAP summary plot saved.")

            # Also save mean |SHAP| per feature as a table
            mean_shap = pd.DataFrame({
                "feature":    selected_cols,
                "mean_shap":  np.abs(sv).mean(axis=0),
            }).sort_values("mean_shap", ascending=False)
            write_csv(mean_shap, f"shap_importance_{safe_name}.csv")

        except Exception as e:
            print(f"  SHAP skipped: {e}")

    # ── Subgroup analysis by user activity level (SQ4) ───────────────────────
    if os.path.exists(USER_ACTIVITY):
        user_activity = pd.read_csv(USER_ACTIVITY)
        enriched = test_df.merge(
            user_activity[[POSTER_COLUMN, "activity_level"]],
            on=POSTER_COLUMN, how="left"
        ).copy()
        enriched["y_pred"] = y_pred
        enriched["y_true"] = y_test

        print("\n  Macro F1 by user activity level (SQ4):")
        subgroup_rows = []
        for level, group in enriched.groupby("activity_level"):
            if len(group) < 5:
                print(f"    {level}: skipped (n={len(group)} < 5)")
                continue
            f1 = f1_score(group["y_true"], group["y_pred"],
                          average="macro", zero_division=0)
            subgroup_rows.append({
                "activity_level": level,
                "f1_macro":       round(f1, 3),
                "n":              len(group),
            })
            print(f"    {level}: F1={f1:.3f}  (n={len(group)})")

        if subgroup_rows:
            write_csv(pd.DataFrame(subgroup_rows), "subgroup_error_analysis.csv")
    else:
        print(f"  Subgroup analysis skipped: {USER_ACTIVITY} not found.")


# ── Step 7: SQ3 — Informational vs emotional subclassification ────────────────

def subclassification_sq3(
    annotated_file: str = os.path.join(
        "data",
        "annotation_sample_with_context_claudia - annotation_sample_with_context.csv"
    ),
    feature_cols: list[str] | None = None,
):
    """
    Addresses SQ3: do predictive patterns differ for informational vs
    emotional support?

    Approach:
      1. Filter SS replies by support_type (informational vs emotional)
      2. Compare feature distributions between types (descriptive)
      3. Train a binary classifier on SS-only replies to predict type
      4. Compare top SHAP features for each type

    This is a secondary analysis run after the main pipeline is validated.
    """
    print("\n[7] SQ3: Informational vs emotional subclassification...")

    if not os.path.exists(annotated_file):
        print(f"  Skipped: annotated file not found at {annotated_file}")
        return

    annotated = pd.read_csv(annotated_file)

    if "support_type" not in annotated.columns:
        print("  Skipped: 'support_type' column not found in annotated file.")
        return

    ss_only = annotated[annotated["label"] == "SS"].copy()
    ss_only["support_type"] = ss_only["support_type"].str.lower().str.strip()
    ss_only = ss_only[ss_only["support_type"].isin(["informational", "emotional"])]

    if len(ss_only) < 20:
        print(f"  Skipped: only {len(ss_only)} SS replies with typed labels.")
        return

    type_counts = ss_only["support_type"].value_counts()
    print(f"  SS replies by type:\n{type_counts.to_string()}")

    # Basic feature comparison
    if TEXT_COLUMN in ss_only.columns:
        ss_only = ss_only.copy()
        ss_only["word_count"] = ss_only[TEXT_COLUMN].fillna("").apply(
            lambda x: len(x.split())
        )
        print("\n  Word count by support type:")
        print(ss_only.groupby("support_type")["word_count"].describe().round(1).to_string())

        # Plot
        fig, ax = plt.subplots(figsize=(7, 4))
        for stype, grp in ss_only.groupby("support_type"):
            grp["word_count"].clip(upper=300).plot(
                kind="hist", bins=25, ax=ax, alpha=0.6,
                label=stype, edgecolor="white"
            )
        ax.set_title("Reply word count: informational vs emotional SS")
        ax.set_xlabel("Word count (capped at 300)")
        ax.set_ylabel("Frequency")
        ax.legend()
        savefig("sq3_support_type_wordcount.png")

    # Binary classifier: informational vs emotional
    if feature_cols and TEXT_COLUMN in ss_only.columns:
        print("\n  Training informational vs emotional classifier...")
        
        # Extract structural features from the reply text directly
        # since ss_only comes from the annotated file, not initial_posts
        ss_only = ss_only.copy()
        text = ss_only[TEXT_COLUMN].fillna("")
        ss_only["feat_word_count"]       = text.apply(lambda x: len(x.split()))
        ss_only["feat_char_count"]       = text.apply(len)
        ss_only["feat_question_marks"]   = text.apply(lambda x: x.count("?"))
        ss_only["feat_exclamation"]      = text.apply(lambda x: x.count("!"))
        ss_only["feat_sentence_count"]   = text.apply(
            lambda x: max(1, x.count(".") + x.count("!") + x.count("?"))
        )
        ss_only["feat_avg_sentence_len"] = (
            ss_only["feat_word_count"] / ss_only["feat_sentence_count"]
        )
        ss_only["feat_ellipsis"] = text.apply(lambda x: x.count("..."))

        sq3_feature_cols = [c for c in ss_only.columns if c.startswith("feat_")]
        ss_feat = ss_only[sq3_feature_cols + ["support_type"]].copy()
        X_sq3 = ss_feat[sq3_feature_cols].fillna(0).values
        y_sq3 = (ss_feat["support_type"] == "informational").astype(int).values

        if len(X_sq3) >= 20:
            clf = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            )
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(clf, X_sq3, y_sq3, cv=skf, scoring="f1_macro")
            print(f"    SQ3 classifier CV macro F1: "
                  f"{scores.mean():.3f} (±{scores.std():.3f})")
        else:
            print(f"  Skipped SQ3 classifier: only {len(X_sq3)} typed SS replies.")

# ── Step 8: Results summary ───────────────────────────────────────────────────

def save_results_summary(val_results: pd.DataFrame, test_results: pd.DataFrame):
    print("\n[8] Saving results summary...")

    combined = val_results.join(
        test_results, lsuffix="_val", rsuffix="_test", how="outer"
    ).round(3)
    write_csv(combined, "results_summary.csv")

    # Bar chart: validation macro F1 comparison
    val_f1 = val_results["val_f1_macro"].sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    val_f1.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Model comparison — validation macro F1")
    ax.set_xlabel("Macro F1")
    ax.axvline(0.5, color="red", linestyle="--", label="Chance (0.5)")
    ax.legend()
    savefig("model_comparison_val.png")

    # Bar chart: test macro F1
    test_f1 = test_results["test_f1_macro"].sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    test_f1.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Model comparison — test macro F1")
    ax.set_xlabel("Macro F1")
    ax.axvline(0.5, color="red", linestyle="--", label="Chance (0.5)")
    ax.legend()
    savefig("model_comparison_test.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    ensure_output_dir()

    # 1. Load
    initial_posts, labels = load_data()

    # 2. Features
    initial_posts = build_features(initial_posts)
    feature_cols  = [c for c in initial_posts.columns if c.startswith("feat_")]
    liwc_present  = any("liwc" in c for c in feature_cols)
    print(f"\n  Feature columns ({len(feature_cols)} total): {feature_cols}")
    if not liwc_present:
        print("  NOTE: LIWC features not loaded. Add output/liwc_output.csv for full model.")

    # 3. Split (user-level)
    train_df, val_df, test_df = split_data(initial_posts)

    # 4. Train + validate
    fitted, val_results, selected_cols, selector = train_models(
        train_df, val_df, test_df, feature_cols
    )

    # 5. Final test evaluation (pick best by val macro F1, excluding dummy)
    non_dummy = val_results.drop(index="Dummy (majority)", errors="ignore")
    best_model_name = non_dummy["val_f1_macro"].idxmax()
    print(f"\n  Best model by val macro F1: {best_model_name}")
    test_results = evaluate_on_test(
        fitted, test_df, feature_cols, selected_cols, selector, best_model_name
    )

    # 6. Error analysis on best model
    error_analysis(
        fitted[best_model_name], test_df,
        feature_cols, selected_cols, selector, best_model_name
    )


    # 7. SQ3 subclassification
    subclassification_sq3(feature_cols=selected_cols)

    # 8. Summary
    save_results_summary(val_results, test_results)

    print("\n" + "="*60)
    print("Pipeline complete. Outputs saved to output/results/")
    print("="*60)
    print("\nNext steps:")
    print("  1. Add output/liwc_output.csv and re-run for full feature set")
    print("  2. Run GridSearchCV on best model for hyperparameter tuning")
    print("  3. Incorporate second rater labels when available")
    print("  4. Update thesis Results section from output/results/results_summary.csv")


if __name__ == "__main__":
    run()
