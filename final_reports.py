import ast
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import CLASSES

VEHICLE_CLASSES = ["Car", "Van", "Bus"]
BASE_OUT_DIR = "final_report_3"
OUT_DIR = BASE_OUT_DIR
HEATMAP_DIR = os.path.join(OUT_DIR, "heatmaps")
METRICS_DIR = os.path.join(OUT_DIR, "metrics")


def parse_conf_matrix(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, np.ndarray)):
        return np.array(value, dtype=int)
    if isinstance(value, str):
        try:
            return np.array(json.loads(value), dtype=int)
        except json.JSONDecodeError:
            try:
                return np.array(ast.literal_eval(value), dtype=int)
            except (ValueError, SyntaxError):
                return None
    return None


def get_next_report_dir(base_dir):
    if not os.path.exists(base_dir):
        return base_dir

    match = re.match(r"^(.*?)(\d+)$", base_dir)
    if match:
        prefix = match.group(1)
        start_num = int(match.group(2))
    else:
        prefix = base_dir.rstrip("_") + "_"
        start_num = 1

    next_num = start_num + 1
    while os.path.exists(f"{prefix}{next_num}"):
        next_num += 1

    return f"{prefix}{next_num}"


def plot_raw_heatmap(cm, classes, title, path, cmap="Blues", figsize=(6, 5)):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    cm = np.array(cm, dtype=int)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_class_metrics(precision, recall, f1, title, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    classes = VEHICLE_CLASSES
    x = np.arange(len(classes))
    width = 0.26

    precision_vals = [precision.get(c, 0.0) for c in classes]
    recall_vals = [recall.get(c, 0.0) for c in classes]
    f1_vals = [f1.get(c, 0.0) for c in classes]

    plt.figure(figsize=(7, 4))
    plt.bar(x - width, precision_vals, width=width, label="Precision")
    plt.bar(x, recall_vals, width=width, label="Recall")
    plt.bar(x + width, f1_vals, width=width, label="F1")

    plt.xticks(x, classes)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compute_vehicle_metrics(cm):
    metrics = {}
    for i, cls in enumerate(VEHICLE_CLASSES):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        metrics[cls] = (precision, recall, f1)

    return metrics


def plot_vehicle_heatmap_with_metrics(cm, title, path, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=VEHICLE_CLASSES,
        yticklabels=VEHICLE_CLASSES,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    fig.subplots_adjust(bottom=0.28)
    lines = []
    for cls in VEHICLE_CLASSES:
        precision, recall, f1 = metrics[cls]
        lines.append(f"{cls}: P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")
    fig.text(0.5, 0.02, "\n".join(lines), ha="center", va="bottom", fontsize=9)

    plt.savefig(path)
    plt.close(fig)


def main():
    global OUT_DIR, HEATMAP_DIR, METRICS_DIR
    OUT_DIR = get_next_report_dir(BASE_OUT_DIR)
    HEATMAP_DIR = os.path.join(OUT_DIR, "heatmaps")
    METRICS_DIR = os.path.join(OUT_DIR, "metrics")

    if not os.path.exists("category_results.csv"):
        raise FileNotFoundError("category_results.csv not found.")

    df = pd.read_csv("category_results.csv")
    if df.empty:
        raise ValueError("category_results.csv is empty.")

    os.makedirs(HEATMAP_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    for _, row in df.iterrows():
        category = str(row.get("category", "Unknown"))
        grid = str(row.get("grid", "unknown"))
        tag = f"{category}_{grid}".replace(" ", "_")

        grid_cm = parse_conf_matrix(row.get("conf_matrix"))
        if grid_cm is not None:
            plot_raw_heatmap(
                grid_cm,
                CLASSES,
                f"Multi-Class Confusion Matrix ({category} - {grid})",
                os.path.join(HEATMAP_DIR, f"{tag}_grid.png"),
                cmap="Blues",
                figsize=(6, 5),
            )

        vehicle_cm = parse_conf_matrix(row.get("vehicle_conf_matrix"))
        if vehicle_cm is not None:
            metrics = compute_vehicle_metrics(vehicle_cm)
            plot_vehicle_heatmap_with_metrics(
                vehicle_cm,
                f"Vehicle-Level Confusion Matrix ({category} - {grid})",
                os.path.join(HEATMAP_DIR, f"{tag}_vehicle.png"),
                metrics,
            )
            plot_class_metrics(
                {k: v[0] for k, v in metrics.items()},
                {k: v[1] for k, v in metrics.items()},
                {k: v[2] for k, v in metrics.items()},
                f"Vehicle Metrics ({category} - {grid})",
                os.path.join(METRICS_DIR, f"{tag}_metrics.png"),
            )


if __name__ == "__main__":
    main()
