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
├── engines/                       # Benchmarking copies & historical archives
│   ├── seq_bench.py               # seq.py + benchmark instrumentation
│   ├── rparallel_he3_bench.py     # rparallel_he3.py + benchmark instrumentation
│   ├── rparallel_he3_instrumented.py
│   ├── seq_bench.diff             # What changed from seq.py → seq_bench.py
│   ├── rparallel_he3_bench.diff
│   └── archive/                   # Dated dev snapshots (seq5–seq8, etc.)
│
├── evaluation/                    # Accuracy evaluation pipeline
│   ├── evaluation.py              # GridDetector, process_video, confusion matrices
│   ├── annotation_parser.py       # UA-DETRAC XML parser
│   ├── batch_runner.py            # Batch accuracy run over all videos & grid sizes
│   ├── analyze_results.py         # Seq vs parallel timing analysis → LaTeX table
│   ├── visualize_results.py       # Regenerate heatmaps/plots from saved CSVs
│   └── final_reports.py          # Consolidated report chart generator
│
├── benchmarking/                  # Performance benchmark pipeline
│   ├── orchestrate_benchmark.py   # Master orchestrator (multi-video, multi-weather)
│   ├── benchmark_runner.py        # Head-to-head seq vs parallel timing (3 trials)
│   └── batch_runner.py            # Accuracy batch runner
│
├── profiling/                     # Profiling artifacts (gitignored)
│   ├── profile_seq.prof / profile_rpar.prof / …
│   ├── *_frame_metrics.json
│   └── *_cumtime.txt / *_tottime.txt
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
pip install flask opencv-python numpy pandas openpyxl matplotlib seaborn psutil anyio
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
python seq.py               # Sequential mode
python rparallel_he3.py     # Parallel mode (with histogram equalization)
```

### 5. Speed calibration
```bash
python calibrate_pixels_per_meter.py <video_path>
```

## Benchmarking

```bash
# Quick head-to-head (3 trials, single video):
python benchmarking/benchmark_runner.py

# Full multi-video benchmark across all UA-DETRAC weather categories:
python benchmarking/orchestrate_benchmark.py
```

## Accuracy Evaluation

```bash
# Batch accuracy run (requires DETRAC_VIDEOS_eval/ and DETRAC-Annotations/):
python evaluation/batch_runner.py

# Analyse timing results:
python evaluation/analyze_results.py

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
| `color_channel` | string | Channel used for detection: `"V"` (HSV Value), `"R"`, `"G"`, `"B"` |
| `grids.rows` | int | Number of grid rows in the ROI |
| `grids.cols` | int | Number of grid columns in the ROI |
| `execution_mode` | string | `"Sequential"` or `"Parallel"` |
| `xml_path` | string | Path to UA-DETRAC XML annotation (optional) |
| `pixels_per_meter` | float | Calibration constant for speed estimation (`0` = disabled) |
