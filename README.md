# Lattice-Layer Vehicle Detection

> **BTP Project** — Grid-based vehicle detection and classification from traffic surveillance video using the [UA-DETRAC](https://detrac-db.rit.albany.edu/) dataset.

## Overview

Divides each video frame into a configurable ROI grid and classifies every cell as **Empty / Car / Van / Bus** using classical computer vision (Gaussian blur → threshold → dilation → contour detection). Two execution strategies are compared: sequential (single-threaded) and parallel (persistent `ThreadPoolExecutor` pools with Histogram Equalization).

## Project Structure

```
.
├── app.py                         # Flask web UI — upload, calibrate, run
├── seq.py                         # Sequential detection engine  ← primary runtime
├── rparallel_he3.py               # Parallel detection engine    ← primary runtime
├── utils.py                       # Shared metrics & plotting helpers
├── calibrate_pixels_per_meter.py  # Interactive speed-calibration tool
├── gui.py                         # Legacy standalone GUI (pre-Flask)
│
├── bench_sysinfo.py               # Hardware/software provenance written into every metrics file
│
├── engines/
│   └── archive/                   # Dated dev snapshots (seq5–seq8, etc.)
│
├── evaluation/                    # Accuracy evaluation pipeline (38-video batch study)
│   ├── evaluation.py              # GridDetector, process_video, confusion matrices
│   ├── annotation_parser.py       # UA-DETRAC XML parser
│   ├── visualize_results.py       # Regenerate heatmaps/plots from saved CSVs
│   └── final_reports.py           # Consolidated report chart generator
│
├── benchmarking/                  # Performance benchmark pipeline (paper)
│   ├── orchestrate_benchmark.py   # 12 videos × 4 workloads → all paper tables' inputs
│   ├── check_equivalence.py       # Exact seq-vs-parallel Lane-1 equivalence check
│   └── batch_runner.py            # Accuracy batch runner (evaluation/evaluation.py)
│
├── benchmark_results/             # v1 benchmark data behind the current paper tables
├── verification/                  # Equivalence/smoke runs in other environments (NOT paper timings)
├── archive/                       # Superseded artifacts and scripts (see archive/README.md)
├── paper/                         # LaTeX source of the paper
│
├── outputs/                       # Generated outputs (gitignored)
│   └── video_writer.ipynb
│
├── static/                        # Flask static files
│   └── uploads/                   # Uploaded videos & XMLs (gitignored)
├── templates/
│   └── index.html                 # Flask frontend
│
├── user_input_data.example.json   # Config template — copy & rename to user_input_data.json
└── user_input_data.json           # Active config (gitignored — machine-specific paths)
```

## Quick Start

### 1. Install dependencies
```bash
pip install flask opencv-python numpy pandas openpyxl matplotlib seaborn psutil
```

### 2. Configure
```bash
cp user_input_data.example.json user_input_data.json
# Edit user_input_data.json with your video path, grid size, and execution mode
```

### 3. Run via Web UI (recommended)
```bash
python app.py
# Open http://127.0.0.1:5000 in your browser
```

### 4. Run engines directly
```bash
python seq.py               # Sequential mode (ROI 1)
python rparallel_he3.py     # Parallel mode (ROI 1 + ROI 2, persistent thread pools)
```
Optional environment switches (defaults reproduce the interactive behaviour):

| Variable | Values | Effect |
|---|---|---|
| `BTP_DISPLAY` | `1` / `0` | live preview window and `waitKey` polling |
| `BTP_PROFILE` | `1` / `0` | run under `cProfile` and write `profile_*.prof/.txt` |
| `BTP_SAVE_FRAMES` | `1` / `0` | `seq.py` only: per-frame JPEG dump to `output_seq/` |
| `BTP_LANES` | `2` / `1` | `rparallel_he3.py` only: process ROI 1 + 2, or ROI 1 only |
| `BTP_EXECUTOR` | `threaded` / `serial` | `rparallel_he3.py` only: thread pools, or the same code with no threads |

Each run writes `seq_frame_metrics.json` / `rpar_frame_metrics.json`
(total time, per-frame processing time, per-frame I/O time, memory, CPU
time, hardware/software record) and, with an XML, `eval_metrics.json`.

### 5. Speed calibration
```bash
python calibrate_pixels_per_meter.py <video_path>
```

## Benchmarking (reproduces the paper)

```bash
python benchmarking/orchestrate_benchmark.py --dry-run   # show the 12 selected videos
python benchmarking/orchestrate_benchmark.py             # full run (≈4 workloads × 12 videos)
python benchmarking/orchestrate_benchmark.py --resume    # continue an interrupted run
```
Runs, per video, the four workloads A `seq.py`, B `rparallel_he3.py
BTP_LANES=1`, C `rparallel_he3.py BTP_EXECUTOR=serial`, D `rparallel_he3.py`
(clean pass; A and D also profiled), with display and frame dumps off.
Results go to a **new** folder `benchmark_results_v2/` (never overwritten):
`FINAL_SUMMARY.txt`, `consolidated_results.csv`, `equivalence.csv`,
`speedup_by_video.csv`, `amdahl_by_video.csv`, per-category/tier summaries,
`system_info.json`, and one folder per run with its log and metrics.

Timing definitions used everywhere: `total_time_seconds` = end-to-end
(module start → end of main loop); `frame_times_seconds` = per-frame
processing **and** overlay drawing, excluding output write, preview and
next-frame decode (those are in `frame_io_times_seconds`). Both scripts use
identical boundaries.

## Accuracy Evaluation

```bash
# Batch accuracy run (requires DETRAC_VIDEOS_eval/ and DETRAC-Annotations/):
python benchmarking/batch_runner.py

# Regenerate plots from saved CSVs:
python evaluation/visualize_results.py
```

## Dataset

**UA-DETRAC** — not included in the repo (too large). Download from https://detrac-db.rit.albany.edu/ and place as:
```
DETRAC_VIDEOS_eval/
  Cloudy/   Night/   Rainy/   Sunny/
DETRAC-Annotations/
  MVI_xxxxx.xml  ...
```

## Configuration Reference (`user_input_data.json`)

| Key | Type | Description |
|-----|------|-------------|
| `video` | string | Absolute path to the video file |
| `color_channel` | string | Difference-image channel(s): `"H"`, `"S"`, `"V"`, `"H+S"`, `"H+V"`, `"S+V"`, `"H+S+V"` or `"gray"` (histogram equalisation always uses V) |
| `grids.rows` | int | Number of grid rows in the ROI |
| `grids.cols` | int | Number of grid columns in the ROI |
| `execution_mode` | string | `"Sequential"` or `"Parallel"` |
| `xml_path` | string | Path to UA-DETRAC XML annotation (optional) |
| `pixels_per_meter` | float | Calibration constant for speed estimation (`0` = disabled) |
