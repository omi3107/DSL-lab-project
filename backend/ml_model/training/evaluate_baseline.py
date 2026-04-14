"""
Evaluate Baseline — metrics and visualisation for the SVM classifier.

Computes accuracy, precision, recall, F1 (macro & per-class), prints a
classification report, and renders a confusion matrix heatmap.

Usage:
    from training.evaluate_baseline import evaluate_model, plot_confusion_matrix
"""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    y_true,
    y_pred,
    labels: list[str] | None = None,
    print_report: bool = True,
) -> dict[str, Any]:
    """Compute all evaluation metrics.

    Args:
        y_true:       Ground-truth labels.
        y_pred:       Predicted labels.
        labels:       Ordered list of label names (for the report).
        print_report: Whether to print the classification report to stdout.

    Returns:
        Dictionary with keys ``accuracy``, ``precision_macro``,
        ``recall_macro``, ``f1_macro``, ``classification_report``, and
        ``confusion_matrix``.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true, y_pred, target_names=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if print_report:
        print("=" * 60)
        print("              EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Accuracy          : {acc:.4f}")
        print(f"  Precision (macro) : {prec:.4f}")
        print(f"  Recall    (macro) : {rec:.4f}")
        print(f"  F1-score  (macro) : {f1:.4f}")
        print("-" * 60)
        print(report)
        print("=" * 60)

    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    save_path: str | None = None,
) -> None:
    """Render a confusion matrix as a seaborn heatmap.

    Args:
        cm:        Confusion matrix array.
        labels:    Class label names.
        title:     Plot title.
        figsize:   Figure size.
        cmap:      Matplotlib colour map.
        save_path: If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[✓] Confusion matrix saved to {save_path}")

    plt.show()


def plot_per_class_metrics(
    y_true,
    y_pred,
    labels: list[str],
    save_path: str | None = None,
) -> None:
    """Bar chart of per-class precision, recall, and F1.

    Args:
        y_true:    Ground-truth labels.
        y_pred:    Predicted labels.
        labels:    Ordered class names.
        save_path: Optional file path to save the figure.
    """
    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1   = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, prec, width, label="Precision", color="#4C72B0")
    ax.bar(x, rec, width, label="Recall", color="#55A868")
    ax.bar(x + width, f1, width, label="F1-Score", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Metrics", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[✓] Per-class metrics chart saved to {save_path}")

    plt.show()
