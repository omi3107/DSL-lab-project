"""Training package — training, evaluation, and persistence scripts."""

from training.evaluate_baseline import (
    evaluate_model,
    plot_confusion_matrix,
    plot_per_class_metrics,
)
from training.save_model import save_trained_model

__all__ = [
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_per_class_metrics",
    "save_trained_model",
]
