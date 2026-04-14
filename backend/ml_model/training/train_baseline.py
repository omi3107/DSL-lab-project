"""
Train Baseline — end-to-end pipeline for the Linear SVM classifier.

Orchestrates the full workflow:
    1. Load labelled CSV dataset
    2. Exploratory Data Analysis (EDA) with visualisations
    3. Text preprocessing / cleaning
    4. TF-IDF vectorisation
    5. Train / test split
    6. Model training (Linear SVM)
    7. Evaluation (accuracy, precision, recall, F1, confusion matrix)
    8. Save artifacts (vectorizer + classifier)

Designed to run seamlessly on **Google Colab** — just upload the repo and
execute this script.

Usage (Colab cell / terminal):
    %run backend/ml_model/training/train_baseline.py
    # or
    !python backend/ml_model/training/train_baseline.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Path bootstrap — works whether run from repo root, training/ dir, or Colab
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ML_MODEL_ROOT = _SCRIPT_DIR.parent            # backend/ml_model
_PROJECT_ROOT = _ML_MODEL_ROOT.parent.parent    # MeetingMind-AI-Redesigned

# Add ml_model root to path so sibling packages are importable
if str(_ML_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_MODEL_ROOT))

# ---------------------------------------------------------------------------
# Dependencies (install instructions for Colab)
# ---------------------------------------------------------------------------
# fmt: off
# !pip install -q scikit-learn pandas matplotlib seaborn nltk wordcloud joblib
# fmt: on

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Try importing wordcloud — optional (skip if not installed)
try:
    from wordcloud import WordCloud
    _HAS_WORDCLOUD = True
except ImportError:
    _HAS_WORDCLOUD = False
    print("[!] wordcloud not installed — word cloud plots will be skipped.")
    print("    Install with: pip install wordcloud\n")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Local imports
from preprocessing.text_cleaner import clean_text, clean_series
from models.tfidf_vectorizer import build_vectorizer, fit_transform, transform
from models.baseline_classifier import build_classifier, train, predict
from training.evaluate_baseline import (
    evaluate_model,
    plot_confusion_matrix,
    plot_per_class_metrics,
)
from training.save_model import save_trained_model

# ---------------------------------------------------------------------------
# Matplotlib backend — allows inline rendering on Colab, Agg on headless
# ---------------------------------------------------------------------------
try:
    _ip = get_ipython()  # type: ignore[name-defined]
    if "google.colab" in str(_ip):
        pass  # Colab already configures inline
except NameError:
    matplotlib.use("Agg")  # headless fallback

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH = _ML_MODEL_ROOT / "dataset" / "labelled_data.csv"
SAVE_DIR = _ML_MODEL_ROOT / "models" / "saved"
RESULTS_DIR = _ML_MODEL_ROOT / "training" / "results"

TEST_SIZE = 0.2
RANDOM_STATE = 42

LABELS = ["Decision", "Task", "Deadline", "Issue", "General"]


# ===========================================================================
#  1.  DATA LOADING
# ===========================================================================
def load_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Read the labelled CSV and perform basic sanity checks."""
    print("=" * 60)
    print("  STEP 1 — Loading Dataset")
    print("=" * 60)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            f"Expected columns: id, sentence, label, source"
        )

    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows  ·  {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Labels : {sorted(df['label'].unique())}\n")

    # Drop rows with missing text
    before = len(df)
    df = df.dropna(subset=["sentence", "label"])
    df = df[df["sentence"].str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing/empty text.\n")

    return df


# ===========================================================================
#  2.  EDA & VISUALISATION
# ===========================================================================
def run_eda(df: pd.DataFrame) -> None:
    """Exploratory Data Analysis with visualisations."""
    print("=" * 60)
    print("  STEP 2 — Exploratory Data Analysis")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 2a. Class distribution -------------------------------------------
    print("\n  Label distribution:")
    label_counts = df["label"].value_counts()
    for lbl, cnt in label_counts.items():
        pct = cnt / len(df) * 100
        print(f"    {lbl:>10s}  {cnt:>8,}  ({pct:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    colors = sns.color_palette("Set2", n_colors=len(label_counts))
    label_counts.plot.bar(ax=axes[0], color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("Label Distribution (Count)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)

    # Pie chart
    axes[1].pie(
        label_counts.values,
        labels=label_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    axes[1].set_title("Label Distribution (%)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "label_distribution.png", dpi=150)
    plt.show()

    # ---- 2b. Sentence length distribution ---------------------------------
    df["_word_count"] = df["sentence"].str.split().str.len()
    df["_char_count"] = df["sentence"].str.len()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["_word_count"], bins=50, color="#4C72B0", edgecolor="black", alpha=0.8)
    axes[0].set_title("Word Count Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Words per sentence")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(df["_word_count"].median(), color="red", linestyle="--", label=f"Median={df['_word_count'].median():.0f}")
    axes[0].legend()

    axes[1].hist(df["_char_count"], bins=50, color="#55A868", edgecolor="black", alpha=0.8)
    axes[1].set_title("Character Count Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Characters per sentence")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(df["_char_count"].median(), color="red", linestyle="--", label=f"Median={df['_char_count'].median():.0f}")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "length_distribution.png", dpi=150)
    plt.show()

    # ---- 2c. Word count per label (box plot) ------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    order = sorted(df["label"].unique())
    sns.boxplot(data=df, x="label", y="_word_count", order=order, palette="Set2", ax=ax)
    ax.set_title("Word Count by Label", fontsize=13, fontweight="bold")
    ax.set_xlabel("Label")
    ax.set_ylabel("Words per sentence")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "wordcount_by_label.png", dpi=150)
    plt.show()

    # ---- 2d. Word clouds per label (optional) -----------------------------
    if _HAS_WORDCLOUD:
        print("\n  Generating word clouds…")
        n_labels = len(LABELS)
        fig, axes = plt.subplots(1, n_labels, figsize=(5 * n_labels, 5))
        for ax, label in zip(axes, LABELS):
            subset = df[df["label"] == label]["sentence"]
            text = " ".join(subset.astype(str).tolist())
            if text.strip():
                wc = WordCloud(
                    width=400, height=300,
                    background_color="white",
                    colormap="viridis",
                    max_words=80,
                ).generate(text)
                ax.imshow(wc, interpolation="bilinear")
            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.axis("off")
        plt.suptitle("Word Clouds by Label", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "word_clouds.png", dpi=150, bbox_inches="tight")
        plt.show()

    # ---- 2e. Source file distribution -------------------------------------
    if "source" in df.columns:
        print(f"\n  Sources: {df['source'].nunique()} unique files")
        top_sources = df["source"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 4))
        top_sources.plot.barh(ax=ax, color="#C44E52", edgecolor="black", alpha=0.85)
        ax.set_title("Top 10 Source Files", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of sentences")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "source_distribution.png", dpi=150)
        plt.show()

    # Clean up temp columns
    df.drop(columns=["_word_count", "_char_count"], inplace=True, errors="ignore")
    print()


# ===========================================================================
#  3.  PREPROCESSING
# ===========================================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning and encode labels."""
    print("=" * 60)
    print("  STEP 3 — Text Preprocessing")
    print("=" * 60)

    t0 = time.time()
    df = df.copy()
    df["clean_text"] = clean_series(df["sentence"])
    elapsed = time.time() - t0
    print(f"  Cleaned {len(df):,} sentences in {elapsed:.1f}s")

    # Drop rows that became empty after cleaning
    before = len(df)
    df = df[df["clean_text"].str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with empty cleaned text")

    print(f"  Final dataset size: {len(df):,}\n")

    # Show sample
    print("  Sample (raw → cleaned):")
    sample = df.sample(min(5, len(df)), random_state=RANDOM_STATE)
    for _, row in sample.iterrows():
        print(f"    [{row['label']:>10s}]  {row['sentence'][:60]}")
        print(f"    {'':>12s}→ {row['clean_text'][:60]}\n")

    return df


# ===========================================================================
#  4.  FULL PIPELINE
# ===========================================================================
def run_pipeline(dataset_path: str | Path | None = None) -> dict:
    """Execute the complete training pipeline.

    Returns:
        Dictionary of evaluation metrics.
    """
    start = time.time()
    print("\n" + "=" * 60)
    print("  MeetingMind-AI — Baseline SVM Training Pipeline")
    print("=" * 60 + "\n")

    # ── 1. Load ────────────────────────────────────────────────────────────
    df = load_dataset(dataset_path or DATASET_PATH)

    # ── 2. EDA ─────────────────────────────────────────────────────────────
    run_eda(df)

    # ── 3. Preprocess ──────────────────────────────────────────────────────
    df = preprocess(df)

    # ── 4. Encode labels ───────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(LABELS)  # fixed order
    y = le.transform(df["label"])
    label_names = list(le.classes_)
    print(f"  Label encoding: {dict(zip(label_names, le.transform(label_names)))}\n")

    # ── 5. Train / test split ──────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 4 — Train / Test Split")
    print("=" * 60)
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        df["clean_text"],
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  Train : {len(X_text_train):,}  ({1 - TEST_SIZE:.0%})")
    print(f"  Test  : {len(X_text_test):,}  ({TEST_SIZE:.0%})\n")

    # ── 6. TF-IDF vectorisation ────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 5 — TF-IDF Vectorisation")
    print("=" * 60)
    vectorizer = build_vectorizer()
    X_train = fit_transform(vectorizer, X_text_train)
    X_test  = transform(vectorizer, X_text_test)
    print(f"  Vocabulary size     : {len(vectorizer.vocabulary_):,}")
    print(f"  Train matrix shape  : {X_train.shape}")
    print(f"  Test  matrix shape  : {X_test.shape}\n")

    # ── 7. Train SVM ──────────────────────────────────────────────────────-
    print("=" * 60)
    print("  STEP 6 — Training Linear SVM")
    print("=" * 60)
    t0 = time.time()
    clf = build_classifier(class_weight="balanced", calibrated=True)
    clf = train(clf, X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s\n")

    # ── 8. Evaluate ──────────────────────────────────────────────────────--
    print("=" * 60)
    print("  STEP 7 — Evaluation")
    print("=" * 60)
    y_pred_encoded = predict(clf, X_test)
    y_pred_labels = le.inverse_transform(y_pred_encoded)
    y_test_labels = le.inverse_transform(y_test)

    metrics = evaluate_model(
        y_test_labels,
        y_pred_labels,
        labels=label_names,
    )

    # Confusion matrix plot
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        labels=label_names,
        title="Baseline SVM — Confusion Matrix",
        save_path=str(RESULTS_DIR / "confusion_matrix.png"),
    )

    # Per-class metrics chart
    plot_per_class_metrics(
        y_test_labels,
        y_pred_labels,
        labels=label_names,
        save_path=str(RESULTS_DIR / "per_class_metrics.png"),
    )

    # ── 9. Save artifacts ─────────────────────────────────────────────────-
    print("=" * 60)
    print("  STEP 8 — Saving Artifacts")
    print("=" * 60)
    save_trained_model(
        vectorizer=vectorizer,
        classifier=clf,
        label_encoder=le,
        save_dir=SAVE_DIR,
        verify=True,
    )

    # ── 10. Quick inference demo ──────────────────────────────────────────-
    print("=" * 60)
    print("  BONUS — Quick Inference Demo")
    print("=" * 60)

    demo_sentences = [
        "We decided to go with React for the frontend.",
        "Rahul will prepare the presentation slides by tomorrow.",
        "Submit the report by end of week.",
        "The API integration is still pending and blocked.",
        "Let's move on to the next topic.",
    ]

    from inference.predict_text import SentenceClassifier  # noqa: E402
    predictor = SentenceClassifier(model_dir=SAVE_DIR)

    for sent in demo_sentences:
        result = predictor.predict(sent)
        conf = f"{result['confidence']:.2%}" if result["confidence"] else "N/A"
        print(f"  [{result['label']:>10s}]  {conf}  |  {sent}")

    # ── Summary ────────────────────────────────────────────────────────────
    total_time = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline completed in {total_time:.1f}s")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Macro : {metrics['f1_macro']:.4f}")
    print(f"  Artifacts: {SAVE_DIR}")
    print(f"  Results  : {RESULTS_DIR}")
    print(f"{'=' * 60}\n")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
