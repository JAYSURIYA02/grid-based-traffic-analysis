import ast
import json
import os

import numpy as np
import pandas as pd

from utils import (
    plot_combined_heatmap,
    plot_f1_bar,
    plot_metric_lines,
    plot_per_class_heatmaps,
)

HEATMAP_DIR = "heatmaps"
PLOTS_DIR = "plots"


def parse_conf_matrix(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, np.ndarray)):
        return np.array(value, dtype=float)

    if isinstance(value, str):
        try:
            return np.array(json.loads(value), dtype=float)
        except json.JSONDecodeError:
            try:
                return np.array(ast.literal_eval(value), dtype=float)
            except (ValueError, SyntaxError):
                return None

    return None


def generate_category_heatmaps(category_df):
    os.makedirs(HEATMAP_DIR, exist_ok=True)

    for _, row in category_df.iterrows():
        cm = parse_conf_matrix(row.get("conf_matrix"))
        if cm is None:
            continue

        category = row.get("category", "Unknown")
        grid = row.get("grid", "unknown")
        prefix = f"{category}_{grid}"

        plot_combined_heatmap(
            cm,
            f"{category} - {grid}",
            os.path.join(HEATMAP_DIR, f"{prefix}_combined.png"),
        )
        plot_per_class_heatmaps(cm, os.path.join(HEATMAP_DIR, prefix))


def generate_metric_plots(category_df):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    default_order = ["Cloudy", "Sunny", "Rainy", "Night"]

    for grid in category_df["grid"].unique():
        subset = category_df[
            (category_df["grid"] == grid) & (category_df["category"] != "ALL")
        ].copy()
        if subset.empty:
            continue

        order = [c for c in default_order if c in subset["category"].unique()]
        if order:
            subset["category"] = pd.Categorical(
                subset["category"], categories=order, ordered=True
            )
            subset = subset.sort_values("category")

        plot_metric_lines(
            subset,
            "f1",
            f"F1 Score Comparison ({grid})",
            os.path.join(PLOTS_DIR, f"f1_{grid}.png"),
        )
        plot_metric_lines(
            subset,
            "precision",
            f"Precision Comparison ({grid})",
            os.path.join(PLOTS_DIR, f"precision_{grid}.png"),
        )
        plot_metric_lines(
            subset,
            "recall",
            f"Recall Comparison ({grid})",
            os.path.join(PLOTS_DIR, f"recall_{grid}.png"),
        )
        plot_f1_bar(
            subset,
            f"F1 Comparison (Bar) - {grid}",
            os.path.join(PLOTS_DIR, f"bar_f1_{grid}.png"),
        )


def generate_video_heatmaps(per_video_path):
    if not os.path.exists(per_video_path):
        return

    df = pd.read_csv(per_video_path)
    if df.empty:
        return

    os.makedirs(HEATMAP_DIR, exist_ok=True)

    for _, row in df.iterrows():
        cm = parse_conf_matrix(row.get("conf_matrix"))
        if cm is None:
            continue

        video = row.get("video", "video")
        grid = row.get("grid", "grid")
        safe_name = os.path.splitext(os.path.basename(video))[0]
        prefix = f"{safe_name}_{grid}"

        plot_combined_heatmap(
            cm,
            f"{safe_name} - {grid}",
            os.path.join(HEATMAP_DIR, f"{prefix}_combined.png"),
        )
        plot_per_class_heatmaps(cm, os.path.join(HEATMAP_DIR, prefix))


def main():
    category_path = "category_results.csv"
    per_video_path = "per_video_results.csv"

    if not os.path.exists(category_path):
        raise FileNotFoundError("category_results.csv not found.")

    category_df = pd.read_csv(category_path)
    if category_df.empty:
        raise ValueError("category_results.csv is empty.")

    generate_category_heatmaps(category_df)
    generate_metric_plots(category_df)
    generate_video_heatmaps(per_video_path)


if __name__ == "__main__":
    main()
