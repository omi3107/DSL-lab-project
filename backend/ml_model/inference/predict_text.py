"""
Predict Text — inference entry-point for the baseline SVM pipeline.

Loads saved model artifacts (TF-IDF vectorizer + SVM classifier) and
classifies raw meeting-transcript sentences into one of five labels:
    Decision · Task · Deadline · Issue · General

Usage (programmatic):
    from inference.predict_text import SentenceClassifier

    clf = SentenceClassifier()                   # loads default saved model
    result = clf.predict("We decided to launch next week.")
    print(result)
    # {'label': 'Decision', 'confidence': 0.87}

Usage (CLI):
    python -m inference.predict_text
    python -m inference.predict_text --text "Submit slides by Friday"
    python -m inference.predict_text --batch sentences.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Ensure ml_model root is importable
# ---------------------------------------------------------------------------
_ML_MODEL_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_MODEL_ROOT))

from models.model_utils import load_artifacts          # noqa: E402
from preprocessing.text_cleaner import clean_text      # noqa: E402


class SentenceClassifier:
    """Stateful wrapper that loads model once and exposes predict methods."""

    def __init__(self, model_dir: str | Path | None = None):
        """Load saved artifacts.

        Args:
            model_dir: Directory containing ``*.joblib`` files.  Defaults to
                       ``backend/ml_model/models/saved/``.
        """
        self.vectorizer, self.classifier, self.label_encoder = load_artifacts(
            model_dir
        )

    # ---- single sentence ---------------------------------------------------

    def predict(self, text: str) -> dict[str, Any]:
        """Classify a single raw sentence.

        Args:
            text: Raw meeting transcript sentence.

        Returns:
            Dict with ``label`` (str) and ``confidence`` (float or None).
        """
        cleaned = clean_text(text)
        if not cleaned:
            return {"label": "General", "confidence": None}

        X = self.vectorizer.transform([cleaned])
        pred = self.classifier.predict(X)[0]

        # Decode via LabelEncoder if available
        if self.label_encoder is not None:
            label = self.label_encoder.inverse_transform([pred])[0]
        else:
            label = pred

        confidence = None
        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X)[0]
            confidence = float(np.max(probs))

        return {"label": label, "confidence": confidence}

    # ---- batch --------------------------------------------------------------

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Classify a list of sentences.

        Args:
            texts: List of raw sentence strings.

        Returns:
            List of result dicts (same format as ``predict``).
        """
        cleaned = [clean_text(t) for t in texts]
        X = self.vectorizer.transform(cleaned)
        preds = self.classifier.predict(X)

        # Decode via LabelEncoder if available
        if self.label_encoder is not None:
            labels = self.label_encoder.inverse_transform(preds)
        else:
            labels = preds

        confidences: list[float | None]
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(X)
            confidences = [float(np.max(row)) for row in proba]
        else:
            confidences = [None] * len(labels)

        return [
            {"label": lbl, "confidence": conf}
            for lbl, conf in zip(labels, confidences)
        ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _interactive(clf: SentenceClassifier) -> None:
    """Start an interactive prediction loop."""
    print("\n=== Interactive Sentence Classifier ===")
    print("Type a sentence and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        result = clf.predict(text)
        conf = f"{result['confidence']:.2%}" if result["confidence"] else "N/A"
        print(f"    Label: {result['label']}  |  Confidence: {conf}\n")

    print("Bye!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict labels for meeting transcript sentences."
    )
    parser.add_argument(
        "--text", type=str, default=None, help="Single sentence to classify."
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Path to a text file with one sentence per line.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to the saved model directory.",
    )
    args = parser.parse_args()

    clf = SentenceClassifier(model_dir=args.model_dir)

    if args.text:
        result = clf.predict(args.text)
        conf = f"{result['confidence']:.2%}" if result["confidence"] else "N/A"
        print(f"Label: {result['label']}  |  Confidence: {conf}")

    elif args.batch:
        path = Path(args.batch)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        texts = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        results = clf.predict_batch(texts)
        for text, res in zip(texts, results):
            conf = f"{res['confidence']:.2%}" if res["confidence"] else "N/A"
            print(f"[{res['label']:>10}]  {conf}  |  {text}")

    else:
        _interactive(clf)


if __name__ == "__main__":
    main()
