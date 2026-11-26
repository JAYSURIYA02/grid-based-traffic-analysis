# Lattice-Layer Vehicle Detection

> **BTP Project** — Lightweight grid-based vehicle classification, counting and speed estimation from traffic surveillance video, evaluated on the [UA-DETRAC](https://detrac-db.rit.albany.edu/) dataset.

## Overview

Each video frame's region of interest (ROI) is divided into a configurable grid. Every cell is classified as **Empty / Car / Van / Bus** using classical computer vision only, with no neural network: frame differencing → Gaussian blur → threshold → dilation → contour detection. Detected vehicles are tracked across frames, counted per lane, and optionally assigned a speed from a pixels-per-metre calibration.

Two execution strategies are implemented and benchmarked against each other:

- **Sequential** (`seq.py`): one lane (ROI 1), single-threaded.
- **Parallel** (`rparallel_he3.py`): two lanes (ROI 1 + ROI 2) processed by persistent `ThreadPoolExecutor` pools, with histogram equalisation.

The repository also contains the benchmark data, a grid-resolution sensitivity study, and the LaTeX source of the paper with its full revision and audit trail.

## Requirements

| | |
|---|---|
| **Python** | 3.10 (developed and tested on 3.10.11 / 3.10.12) |
| **OS** | Windows, Linux or macOS |
| **Packages** | listed in [`requirements.txt`](requirements.txt) |

| Package | Used for |
|---|---|
| `opencv-python` | Video I/O and all image processing |
| `numpy`, `pandas` | Grid matrices, metrics, CSV tables |
| `openpyxl` | Per-frame result matrix export (`result_matrix1.xlsx`) |
| `flask` | Web UI (`app.py`) |
| `matplotlib`, `seaborn` | Plots, heatmaps, paper figures |
| `psutil` | Memory / CPU provenance in benchmark metrics |
| `Pillow` | Legacy Tkinter GUI (`gui.py`) |

`tkinter` (for `gui.py`) ships with standard Python installers. A LaTeX distribution, or Overleaf, is only needed to build the paper.

## Installation

```bash
git clone https://github.com/JAYSURIYA02/grid-based-traffic-analysis.git
cd grid-based-traffic-analysis

# (recommended) create a virtual environment
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
```

With Conda: `conda create -n btp python=3.10`, `conda activate btp`, then `pip install -r requirements.txt`.

## Running the Application

### Option 1: Web UI (recommended)

```bash
python app.py
```

Open **http://127.0.0.1:5000** in a browser, then:

1. Upload a traffic video and, optionally, its UA-DETRAC XML annotation (this enables accuracy metrics).
2. Choose the grid size (rows × columns), the colour channel and the execution mode (**Sequential** or **Parallel**).
3. *(Optional)* Calibrate speed: the UI shows a frame with the ROI grid so you can mark a known distance and compute pixels per metre.
4. Click **Process**. The app writes `user_input_data.json`, runs `seq.py` or `rparallel_he3.py`, and plays the annotated result video in the browser.

Uploaded files are stored in `static/uploads/`. The result video is saved as `static/seq_output.mp4` (sequential) or `static/test_output1.mp4` (parallel). All of these are gitignored.

### Option 2: Run the engines directly

Create a config from the template and edit it (see the [Configuration reference](#configuration-reference-user_input_datajson) below):

```bash
# Linux/macOS
cp user_input_data.example.json user_input_data.json
# Windows (PowerShell)
Copy-Item user_input_data.example.json user_input_data.json
```

Then run one of the engines:

```bash
python seq.py               # Sequential: ROI 1
python rparallel_he3.py     # Parallel: ROI 1 + ROI 2, persistent thread pools
```

**Outputs** (written to the working directory):

| File | Written by | Content |
|---|---|---|
| `seq_output.mp4` / `test_output1.mp4` | seq / parallel | Annotated video (grid, classes, counts, speed) |
| `result_matrix1.xlsx` | both | Per-frame grid classification matrix |
| `seq_frame_metrics.json` / `rpar_frame_metrics.json` | seq / parallel | Total time, per-frame processing and I/O time, memory, CPU time, hardware/software record |
| `eval_metrics.json` | both, only if `xml_path` is set | Precision / recall / F1 against UA-DETRAC ground truth |

**Optional environment switches** (the defaults reproduce the interactive behaviour):

| Variable | Values | Effect |
|---|---|---|
| `BTP_DISPLAY` | `1` / `0` | Live preview window and `waitKey` polling |
| `BTP_PROFILE` | `1` / `0` | Run under `cProfile` and write `profile_*.prof/.txt` |
| `BTP_SAVE_FRAMES` | `1` / `0` | `seq.py` only: per-frame JPEG dump to `output_seq/` |
| `BTP_LANES` | `2` / `1` | `rparallel_he3.py` only: process ROI 1 + 2, or ROI 1 only |
| `BTP_EXECUTOR` | `threaded` / `serial` | `rparallel_he3.py` only: thread pools, or the same code without threads |

Example of a headless run (no preview window):

```bash
# Linux/macOS
BTP_DISPLAY=0 python rparallel_he3.py
# Windows (PowerShell)
$env:BTP_DISPLAY = "0"; python rparallel_he3.py
```

### Option 3: Legacy desktop GUI

```bash
python gui.py
```

A Tkinter interface from before the Flask UI existed. It is kept for reference; the web UI is the maintained front end.

### Speed calibration (stand-alone)

```bash
python calibrate_pixels_per_meter.py <video_path> [frame_number] [json_path]
```

A window shows the chosen frame (default 0):

1. Click two points that span a known real-world distance, then type that distance in metres.
2. Press `n` to add more samples, `r` to re-click the current pair, and `q` to finish.

The averaged value is saved as `pixels_per_meter` in `user_input_data.json` (or in `json_path`, if given). A value of `0` disables speed estimation.

## Configuration reference (`user_input_data.json`)

| Key | Type | Description |
|-----|------|-------------|
| `video` | string | Absolute path to the video file |
| `color_channel` | string | Difference-image channel(s): `"H"`, `"S"`, `"V"`, `"H+S"`, `"H+V"`, `"S+V"`, `"H+S+V"` or `"gray"` (histogram equalisation always uses V) |
| `grids.rows` | int | Number of grid rows in the ROI |
| `grids.cols` | int | Number of grid columns in the ROI |
| `execution_mode` | string | `"Sequential"` or `"Parallel"` |
| `xml_path` | string | Path to the UA-DETRAC XML annotation (optional, `""` to skip evaluation) |
| `pixels_per_meter` | float | Calibration constant for speed estimation (`0` = disabled) |

`user_input_data.json` contains machine-specific absolute paths, so it is gitignored. Only `user_input_data.example.json` is committed.

## Dataset

**UA-DETRAC** is not included in the repository because it is too large. Download it from https://detrac-db.rit.albany.edu/ and place it in the repository root as:

```
DETRAC_VIDEOS_eval/
  Cloudy/   Night/   Rainy/   Sunny/      # MVI_xxxxx.mp4 grouped by weather
DETRAC-Annotations/
  MVI_xxxxx.xml  ...
```

The web UI and the engines work on any traffic video. The dataset is only needed for accuracy evaluation, benchmarking and the grid-resolution study.

## Reproducing the paper results

All commands run from the repository root.

### Performance benchmark

```bash
python benchmarking/orchestrate_benchmark.py --dry-run   # show the 12 selected videos
python benchmarking/orchestrate_benchmark.py             # full run (4 workloads × 12 videos)
python benchmarking/orchestrate_benchmark.py --resume    # continue an interrupted run
```

For each video it runs four workloads, with display and frame dumps off:

- **A**: `seq.py`
- **B**: `rparallel_he3.py` with `BTP_LANES=1`
- **C**: `rparallel_he3.py` with `BTP_EXECUTOR=serial`
- **D**: `rparallel_he3.py`

A and D also get a profiled pass. Results go to a new folder that is never overwritten: `FINAL_SUMMARY.txt`, `consolidated_results.csv`, `equivalence.csv`, `speedup_by_video.csv`, `amdahl_by_video.csv`, per-category summaries, `system_info.json`, and one folder per run with its log and metrics.

To verify that two runs give bit-identical results (the orchestrator runs this automatically):

```bash
python benchmarking/check_equivalence.py <runA_dir> <laneA> <runB_dir> <laneB>
```

Use `Lane1` / `Lane2` for `rparallel_he3.py` runs and `-` for `seq.py` runs.

**Timing definitions:**
- `total_time_seconds` is end-to-end, from module start to the end of the main loop.
- `frame_times_seconds` is per-frame processing **and** overlay drawing. It excludes output write, preview and next-frame decode, which are recorded in `frame_io_times_seconds`.

Both engines use identical timing boundaries.

### Accuracy evaluation

```bash
python benchmarking/batch_runner.py           # batch accuracy run (needs DETRAC_VIDEOS_eval/ + DETRAC-Annotations/)
python evaluation/visualize_results.py        # regenerate heatmaps/plots from saved CSVs
python evaluation/final_reports.py            # consolidated report charts
```

### Grid-resolution sensitivity analysis

The study compares grids from 4×6 to 10×12 using site-grouped cross-validation, ground-truth counting and an unseen-site test set. The full step-by-step commands are in [`grid_resolution_analysis/reproduction_commands.md`](grid_resolution_analysis/reproduction_commands.md). The final aggregates and figures come from:

```bash
python grid_resolution_analysis/final_analysis.py
python grid_resolution_analysis/verify_corrected.py
```

### Paper

- `paper/`: original manuscript (`paper_final.tex`) and its figure scripts (`make_figures.py`, `make_grid_figure.py`).
- `paper_revision/`: revised manuscript, audit trail and every regenerated table and figure. See [`paper_revision/README.md`](paper_revision/README.md). To rebuild it:

```bash
python paper_revision/scripts/revision_analysis.py all
python paper_revision/figure_sources/make_figures_revised.py
python paper_revision/figure_sources/make_grid_figures.py
python paper_revision/scripts/build_revised_tex.py
python paper_revision/scripts/verify_revision.py
```

Then compile `revised_paper.tex` with IEEEtran, or upload `revised_paper_overleaf.zip` to Overleaf. The bundled `revised_paper.pdf` is a layout preview built with a stand-in class, not an IEEE build.

## Project Structure

```
.
├── app.py                         # Flask web UI: upload, calibrate, run, view result
├── seq.py                         # Sequential detection engine   ← primary runtime
├── rparallel_he3.py               # Parallel detection engine     ← primary runtime
├── utils.py                       # Shared metrics & plotting helpers
├── calibrate_pixels_per_meter.py  # Stand-alone speed-calibration tool
├── bench_sysinfo.py               # Hardware/software provenance for every metrics file
├── gui.py                         # Legacy Tkinter GUI (pre-Flask)
├── requirements.txt               # Python dependencies
├── user_input_data.example.json   # Config template → copy to user_input_data.json
│
├── templates/index.html           # Flask frontend
├── static/                        # Flask static files (js/, style.css)
│   └── uploads/                   # Uploaded videos & XMLs (gitignored)
│
├── engines/archive/               # Dated development snapshots of the engines
├── evaluation/                    # Accuracy evaluation pipeline
│   ├── evaluation.py              #   GridDetector, process_video, confusion matrices
│   ├── annotation_parser.py       #   UA-DETRAC XML parser
│   ├── visualize_results.py       #   Heatmaps/plots from saved CSVs
│   └── final_reports.py           #   Consolidated report charts
├── benchmarking/                  # Performance benchmark pipeline
│   ├── orchestrate_benchmark.py   #   12 videos × 4 workloads → paper tables
│   ├── check_equivalence.py       #   Seq-vs-parallel Lane-1 equivalence check
│   └── batch_runner.py            #   Accuracy batch runner
│
├── benchmark_results/             # v1 benchmark data
├── benchmark_results_v2/          # v2 four-workload benchmark data (used by the paper)
├── verification/                  # Equivalence/smoke runs on other machines (not paper timings)
├── grid_resolution_analysis/      # Grid-resolution sensitivity study (scripts, runs, results)
├── paper/                         # Original paper LaTeX source, figures, audit
├── paper_revision/                # Revised paper, audit trail, tables, figure scripts
└── archive/                       # Superseded scripts and data (see archive/README.md)
```

**Gitignored, generated locally:** the dataset folders, `outputs/`, `heatmaps/`, `plots/`, `combined_matrices/`, `final_report*/`, all videos (`*.mp4`, `*.mov`, …), spreadsheets (`*.xlsx`), caches and `user_input_data.json`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: openpyxl` / `flask` | Run `pip install -r requirements.txt` in the environment you run Python from |
| Web UI says "output video was not generated" | Check the Flask console for the engine's error. The most common cause is a video path or codec OpenCV cannot read |
| Engine opens a window / blocks on a server | Set `BTP_DISPLAY=0` for headless runs |
| `FileNotFoundError: user_input_data.json` | Copy `user_input_data.example.json` to `user_input_data.json`, or run once through the web UI |
| `app.py` runs the wrong Python | It calls `python` from `PATH`, so activate the environment where the requirements are installed before starting it |
