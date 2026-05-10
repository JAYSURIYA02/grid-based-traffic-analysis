import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CLASSES = ["Empty", "Car", "Van", "Bus"]


def compute_metrics(cm):
    precision, recall, f1 = {}, {}, {}

    for i, cls in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision[cls] = tp / (tp + fp + 1e-6)
        recall[cls] = tp / (tp + fn + 1e-6)
        f1[cls] = 2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls] + 1e-6)

    return precision, recall, f1


def flatten_metrics(precision, recall, f1):
    flat = {}
    for cls in CLASSES:
        flat[f"precision_{cls}"] = precision.get(cls, 0.0)
        flat[f"recall_{cls}"] = recall.get(cls, 0.0)
        flat[f"f1_{cls}"] = f1.get(cls, 0.0)
    return flat


def plot_heatmap(cm, title, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_combined_heatmap(cm, title, path):
    """Normalized confusion matrix heatmap (row-wise)."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    cm = np.array(cm, dtype=float)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-6)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_per_class_heatmaps(cm, save_prefix):
    """Per-class normalized heatmaps (one row per class)."""
    cm = np.array(cm, dtype=float)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-6)

    for i, cls in enumerate(CLASSES):
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_norm[i:i + 1, :],
                    annot=True,
                    fmt=".2f",
                    cmap="Reds",
                    xticklabels=CLASSES,
                    yticklabels=[cls])
        plt.title(f"{cls} Classification")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_{cls}.png")
        plt.close()


def plot_metric_lines(df, metric_prefix, title, path):
    """Line chart comparing Car/Van/Bus metrics across categories."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    categories = df["category"].tolist()
    plt.figure(figsize=(7, 4))

    for cls in ["Car", "Van", "Bus"]:
        col = f"{metric_prefix}_{cls}"
        if col in df.columns:
            plt.plot(categories, df[col], marker="o", label=cls)

    ylabel = "F1 Score" if metric_prefix == "f1" else metric_prefix.capitalize()
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel(ylabel)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_f1_bar(df, title, path):
    """Grouped bar chart for F1 by class across categories."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    categories = df["category"].tolist()
    x = np.arange(len(categories))
    width = 0.22

    plt.figure(figsize=(7, 4))
    plt.bar(x, df["f1_Car"], width=width, label="Car")
    plt.bar(x + width, df["f1_Van"], width=width, label="Van")
    plt.bar(x + 2 * width, df["f1_Bus"], width=width, label="Bus")

    plt.xticks(x + width, categories)
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_confusion_matrix_csv(path, cm):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    import pandas as pd
    df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    df.to_csv(path)
