"""
insight_extractor.py — ML-powered meeting transcript analyser.

Uses the trained SVM classifier to categorise transcript sentences into:
    Decision · Task · Deadline · Issue · General

Returns structured data for the AI framing layer.
"""

from __future__ import annotations

import json
import sys
import logging
from pathlib import Path

import nltk

# Ensure nltk sentence tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize

# ── Path setup for ML model imports ──────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ML_MODEL_ROOT = _PROJECT_ROOT / "backend" / "ml_model"

if str(_ML_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_MODEL_ROOT))

from inference.predict_text import SentenceClassifier

logger = logging.getLogger(__name__)

# ── Singleton classifier instance ────────────────────────
_classifier: SentenceClassifier | None = None


def _get_classifier() -> SentenceClassifier:
    """Lazy-load the SentenceClassifier (model loads once)."""
    global _classifier
    if _classifier is None:
        _classifier = SentenceClassifier()
    return _classifier


def extract_insights(text: str) -> dict:
    """Classify transcript sentences using the trained ML model.

    Splits the transcript into sentences, runs batch prediction,
    and groups results by label.

    Returns:
        Structured dict with sentences grouped by predicted label,
        plus metadata for the AI framing layer.
    """
    insights = {
        "meeting_summary": {"title": "Meeting Analysis", "participants": []},
        "decisions": [],
        "tasks": [],
        "deadlines": [],
        "issues": [],
        "general": [],
        "responsibility_map": {},
        "intelligence_score": {"score": 0, "flags": []},
        "errors": [],
        "raw_transcript": text,
    }

    if not text or not text.strip():
        insights["errors"].append("Empty transcript provided.")
        return insights

    # ── Split transcript into sentences ───────────────────
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        insights["errors"].append("No sentences found in transcript.")
        return insights

    # ── Classify using the ML model ──────────────────────
    try:
        clf = _get_classifier()
        results = clf.predict_batch(sentences)
    except Exception as e:
        logger.error("ML model prediction failed: %s", e, exc_info=True)
        insights["errors"].append(f"ML classification error: {e}")
        return insights

    # ── Group sentences by predicted label ────────────────
    for sentence, result in zip(sentences, results):
        label = result["label"]

        if label == "Decision":
            insights["decisions"].append(sentence)
        elif label == "Task":
            insights["tasks"].append({
                "assignee": "Unassigned",
                "task": sentence,
                "deadline": None,
            })
        elif label == "Deadline":
            insights["deadlines"].append({"description": sentence})
        elif label == "Issue":
            insights["issues"].append(sentence)
        else:  # General
            insights["general"].append(sentence)

    # ── Calculate intelligence score ─────────────────────
    n_dec = len(insights["decisions"])
    n_task = len(insights["tasks"])
    n_dl = len(insights["deadlines"])
    n_iss = len(insights["issues"])

    score = min(100, n_dec * 12 + n_task * 10 + n_dl * 8 + n_iss * 5)

    flags = []
    if n_dec == 0:
        flags.append(("⚠️", "No decisions detected"))
    if n_task == 0:
        flags.append(("⚠️", "No tasks assigned"))
    if n_dl > 0:
        flags.append(("✅", f"{n_dl} deadline(s) identified"))
    if n_iss > 0:
        flags.append(("🔴", f"{n_iss} open issue(s)"))
    if n_dec > 3:
        flags.append(("✅", "Highly productive meeting"))

    insights["intelligence_score"] = {"score": score, "flags": flags}

    return insights


def insights_to_json(data: dict) -> str:
    """Serialise insights to JSON string (excludes raw transcript)."""
    export = {k: v for k, v in data.items() if k != "raw_transcript"}
    return json.dumps(export, indent=2, default=str)