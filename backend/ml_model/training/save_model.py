"""
Save Model — convenience wrapper used by the training script.

Delegates to ``models.model_utils.save_artifacts`` and adds a small
verification step (load-back sanity check).

Usage:
    from training.save_model import save_trained_model
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Ensure the ml_model root is on sys.path for standalone execution
_ML_MODEL_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_MODEL_ROOT))

from models.model_utils import save_artifacts, load_artifacts  # noqa: E402


def save_trained_model(
    vectorizer,
    classifier,
    label_encoder=None,
    save_dir: str | Path | None = None,
    verify: bool = True,
) -> Path:
    """Save model artifacts and optionally verify round-trip loading.

    Args:
        vectorizer:    Fitted TF-IDF vectorizer.
        classifier:    Fitted SVM classifier.
        label_encoder: Optional fitted LabelEncoder.
        save_dir:      Override directory (defaults to ``models/saved/``).
        verify:        If True, reload artifacts to confirm integrity.

    Returns:
        Path to the saved artifacts directory.
    """
    print(f"\n{'=' * 50}")
    print(f"  Saving model artifacts — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'=' * 50}")

    out = save_artifacts(vectorizer, classifier, label_encoder, save_dir)

    if verify:
        print("  Verifying round-trip load…")
        vec, clf, le = load_artifacts(out)
        assert vec is not None, "Vectorizer reload failed"
        assert clf is not None, "Classifier reload failed"
        print("  [✓] Verification passed\n")

    return out
