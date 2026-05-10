import json
import os

import numpy as np
import pandas as pd

from evaluation import process_video
from utils import (
    compute_metrics,
    flatten_metrics,
    plot_combined_heatmap,
    plot_per_class_heatmaps,
    plot_metric_lines,
    plot_f1_bar,
    save_confusion_matrix_csv,
)

VIDEO_ROOT = "DETRAC_VIDEOS_eval"
ANNOT_ROOT = "DETRAC-Annotations"
HEATMAP_DIR = "heatmaps"
MATRIX_DIR = "combined_matrices"
PLOTS_DIR = "plots"

GRID_SIZES = [(6, 6), (10, 16), (2, 3)]


def safe_video_list(folder):
    if not os.path.isdir(folder):
        return []
    return [f for f in sorted(os.listdir(folder)) if f.lower().endswith(".mp4")]


def main():
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    os.makedirs(MATRIX_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_results = []
    category_results = []

    categories = [d for d in sorted(os.listdir(VIDEO_ROOT))
                  if os.path.isdir(os.path.join(VIDEO_ROOT, d))]

    for rows, cols in GRID_SIZES:
        grid_total_cm = np.zeros((4, 4), dtype=int)
        grid_video_count = 0
        grid_total_vehicle_cm = np.zeros((3, 3), dtype=int)

        for category in categories:
            cat_path = os.path.join(VIDEO_ROOT, category)
            videos = safe_video_list(cat_path)
            if not videos:
                continue

            category_cm = np.zeros((4, 4), dtype=int)
            category_vehicle_cm = np.zeros((3, 3), dtype=int)
            processed = 0

            for video_file in videos:
                video_path = os.path.join(cat_path, video_file)
                xml_path = os.path.join(ANNOT_ROOT, video_file.replace(".mp4", ".xml"))

                if not os.path.exists(xml_path):
                    print(f"Missing XML for {video_file}; skipping.")
                    continue

                print(f"Processing {video_file} ({rows}x{cols})")
                result = process_video(video_path, xml_path, rows, cols)

                cm = result["conf_matrix"]
                vehicle_cm = result.get("vehicle_conf_matrix")
                category_cm += cm
                grid_total_cm += cm
                if vehicle_cm is not None:
                    category_vehicle_cm += vehicle_cm
                    grid_total_vehicle_cm += vehicle_cm
                processed += 1

                row = {
                    "video": video_file,
                    "category": category,
                    "grid": f"{rows}x{cols}",
                    "frames": result.get("frame_count", 0),
                    "conf_matrix": json.dumps(cm.tolist()),
                    "vehicle_conf_matrix": json.dumps(vehicle_cm.tolist()) if vehicle_cm is not None else None,
                }
                row.update(flatten_metrics(result["precision"], result["recall"], result["f1"]))
                all_results.append(row)

                plot_combined_heatmap(
                    cm,
                    f"{video_file} {rows}x{cols}",
                    os.path.join(HEATMAP_DIR, f"{video_file}_{rows}x{cols}_combined.png")
                )

            if processed == 0:
                continue

            grid_video_count += processed

            cat_precision, cat_recall, cat_f1 = compute_metrics(category_cm)
            cat_row = {
                "category": category,
                "grid": f"{rows}x{cols}",
                "videos": processed,
                "conf_matrix": json.dumps(category_cm.tolist()),
                "vehicle_conf_matrix": json.dumps(category_vehicle_cm.tolist()),
            }
            cat_row.update(flatten_metrics(cat_precision, cat_recall, cat_f1))
            category_results.append(cat_row)

            save_confusion_matrix_csv(
                os.path.join(MATRIX_DIR, f"conf_matrix_{category}_{rows}x{cols}.csv"),
                category_cm
            )
            pd.DataFrame(
                category_vehicle_cm,
                index=["Car", "Van", "Bus"],
                columns=["Car", "Van", "Bus"],
            ).to_csv(os.path.join(MATRIX_DIR, f"vehicle_conf_matrix_{category}_{rows}x{cols}.csv"))
            plot_combined_heatmap(
                category_cm,
                f"{category} {rows}x{cols}",
                os.path.join(HEATMAP_DIR, f"{category}_{rows}x{cols}_combined.png")
            )
            plot_per_class_heatmaps(
                category_cm,
                os.path.join(HEATMAP_DIR, f"{category}_{rows}x{cols}")
            )

        if grid_total_cm.sum() > 0:
            grid_precision, grid_recall, grid_f1 = compute_metrics(grid_total_cm)
            grid_row = {
                "category": "ALL",
                "grid": f"{rows}x{cols}",
                "videos": grid_video_count,
                "conf_matrix": json.dumps(grid_total_cm.tolist()),
                "vehicle_conf_matrix": json.dumps(grid_total_vehicle_cm.tolist()),
            }
            grid_row.update(flatten_metrics(grid_precision, grid_recall, grid_f1))
            category_results.append(grid_row)

            save_confusion_matrix_csv(
                os.path.join(MATRIX_DIR, f"conf_matrix_ALL_{rows}x{cols}.csv"),
                grid_total_cm
            )
            pd.DataFrame(
                grid_total_vehicle_cm,
                index=["Car", "Van", "Bus"],
                columns=["Car", "Van", "Bus"],
            ).to_csv(os.path.join(MATRIX_DIR, f"vehicle_conf_matrix_ALL_{rows}x{cols}.csv"))
            plot_combined_heatmap(
                grid_total_cm,
                f"ALL {rows}x{cols}",
                os.path.join(HEATMAP_DIR, f"ALL_{rows}x{cols}_combined.png")
            )
            plot_per_class_heatmaps(
                grid_total_cm,
                os.path.join(HEATMAP_DIR, f"ALL_{rows}x{cols}")
            )

    pd.DataFrame(all_results).to_csv("per_video_results.csv", index=False)
    category_df = pd.DataFrame(category_results)
    category_df.to_csv("category_results.csv", index=False)

    if not category_df.empty:
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
                os.path.join(PLOTS_DIR, f"f1_{grid}.png")
            )
            plot_metric_lines(
                subset,
                "precision",
                f"Precision Comparison ({grid})",
                os.path.join(PLOTS_DIR, f"precision_{grid}.png")
            )
            plot_metric_lines(
                subset,
                "recall",
                f"Recall Comparison ({grid})",
                os.path.join(PLOTS_DIR, f"recall_{grid}.png")
            )
            plot_f1_bar(
                subset,
                f"F1 Comparison (Bar) - {grid}",
                os.path.join(PLOTS_DIR, f"bar_f1_{grid}.png")
            )


if __name__ == "__main__":
    main()
