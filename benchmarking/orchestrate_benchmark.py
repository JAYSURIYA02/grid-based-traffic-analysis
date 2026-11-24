"""
orchestrate_benchmark.py
-------------------------
Drives the multi-video benchmark of the SHIPPED implementations
(seq.py and rparallel_he3.py at the repository root) across the UA-DETRAC
Cloudy/Night/Rainy/Sunny folder structure, and aggregates the results.

Workloads run per video (all with BTP_DISPLAY=0, BTP_SAVE_FRAMES=0):
  A  sequential      seq.py                              1 lane, no pools
  B  parallel_1lane  rparallel_he3.py  BTP_LANES=1       1 lane, threaded
  C  serial_2lane    rparallel_he3.py  BTP_EXECUTOR=serial 2 lanes, no threads
  D  parallel        rparallel_he3.py                    2 lanes, threaded
Each gets a clean (profiler-off) pass used for timing; A and D also get a
profiled pass (cProfile) used for the Amdahl analysis.

Video selection: by default the 12 videos of the existing
benchmark_results/selection_report.json are re-used, so a re-run measures
the same sequences as the paper. Pass --reselect to re-derive the
selection from density terciles instead.

Outputs go to a NEW directory (benchmark_results_v2/, or a timestamped
sibling if that already holds a finished run); existing results are never
overwritten.

This script does not modify any detection/tracking/evaluation algorithm;
it only writes each run's user_input_data.json, launches the scripts as
subprocesses and reads their output files.

USAGE
    python benchmarking/orchestrate_benchmark.py --dry-run      # selection only
    python benchmarking/orchestrate_benchmark.py                # full run
    python benchmarking/orchestrate_benchmark.py --max-videos 2 # quick subset
    python benchmarking/orchestrate_benchmark.py --no-profile   # skip profiled passes
    python benchmarking/orchestrate_benchmark.py --resume       # continue an interrupted run
                                                                # (same BTP_OUTPUT_ROOT), skipping
                                                                # runs whose outputs already exist
Paths can be overridden with BTP_DATASET_ROOT, BTP_ANNOTATIONS_ROOT,
BTP_OUTPUT_ROOT, BTP_SELECTION_REPORT environment variables.
"""
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════
# CONFIG -- edit these paths/values for your machine before running.
# ══════════════════════════════════════════════════════════════════════════

REPO_ROOT = Path(__file__).resolve().parent.parent

# UA-DETRAC video root, containing Cloudy/ Night/ Rainy/ Sunny/ subfolders.
DATASET_ROOT = Path(os.environ.get("BTP_DATASET_ROOT", REPO_ROOT / "DETRAC_VIDEOS_eval"))

# UA-DETRAC XML ground-truth annotations root (searched recursively).
ANNOTATIONS_ROOT = Path(os.environ.get("BTP_ANNOTATIONS_ROOT", REPO_ROOT / "DETRAC-Annotations"))

# Optional batch_runner.py output used to pick videos by density; if absent
# or unparsable, density is computed from each video's XML instead.
PER_VIDEO_RESULTS_CSV = REPO_ROOT / "evaluation" / "per_video_results.csv"

# The shipped implementations (not copies).
SEQ_SCRIPT = REPO_ROOT / "seq.py"
PAR_SCRIPT = REPO_ROOT / "rparallel_he3.py"

# Fixed video selection to reuse (the paper's 12 videos). --reselect ignores it.
SELECTION_REPORT = Path(os.environ.get(
    "BTP_SELECTION_REPORT", REPO_ROOT / "benchmark_results" / "selection_report.json"))

# Where benchmark outputs are written. Never an existing finished run.
_DEFAULT_OUTPUT_ROOT = Path(os.environ.get("BTP_OUTPUT_ROOT", REPO_ROOT / "benchmark_results_v2"))

# ── Selection knobs ─────────────────────────────────────────────────────
CATEGORIES = ["Cloudy", "Night", "Rainy", "Sunny"]
TIERS = ["light", "medium", "heavy"]   # terciles of density, computed per category
# Cap on total videos selected (across all categories/tiers combined).
# None = no cap (up to len(CATEGORIES) * len(TIERS) = 12 videos). Lower this
# if you want a faster/cheaper run -- selection is done round-robin across
# (category, tier) pairs so a lower cap still stays balanced across
# categories rather than exhausting one category first.
MAX_TOTAL_VIDEOS = None

# ── Pipeline config injected into each run's user_input_data.json ───────
GRID_ROWS = 6
GRID_COLS = 6
COLOR_CHANNEL = "V"
# UA-DETRAC videos are not calibrated to any real-world pixels-per-meter
# constant, so speed estimation is left disabled (0) for this benchmark --
# see seq_bench.diff for why this is now safe on seq_bench.py as well.
PIXELS_PER_METER = 0

# ── Execution knobs ──────────────────────────────────────────────────────
# Running BOTH a profiled and a clean pass roughly doubles total runtime.
# Set to False to skip the profiled pass entirely (faster, but you lose the
# per-function cProfile breakdown used in the paper's Discussion section).
RUN_PROFILED_PASS = True
RUN_TIMEOUT_SECONDS = 60 * 60  # 1 hour per script invocation; adjust as needed

# (name, script, extra env, also run a profiled pass?)
WORKLOADS = [
    ("sequential",     SEQ_SCRIPT, {},                         True),   # A
    ("parallel_1lane", PAR_SCRIPT, {"BTP_LANES": "1"},         False),  # B
    ("serial_2lane",   PAR_SCRIPT, {"BTP_EXECUTOR": "serial"}, False),  # C
    ("parallel",       PAR_SCRIPT, {},                         True),   # D
]

# ══════════════════════════════════════════════════════════════════════════
# END CONFIG
# ══════════════════════════════════════════════════════════════════════════


DENSITY_CSV_VIDEO_CANDIDATES = [
    "video_name", "video_id", "videoname", "video", "sequence", "seq_id", "seq",
]
DENSITY_CSV_DENSITY_CANDIDATES = [
    "avg_vehicle", "vehicle_count", "num_vehicle", "n_vehicle", "gt_vehicle",
    "gt_count", "density", "total_vehicle", "mean_vehicle",
]
DENSITY_CSV_BLACKLIST = ["f1", "precision", "recall", "empty"]
DENSITY_CSV_GRID_ROW_CANDIDATES = ["grid_rows", "rows"]
DENSITY_CSV_GRID_COL_CANDIDATES = ["grid_cols", "grid_col", "cols", "columns"]
DENSITY_CSV_GRID_COMBO_CANDIDATES = ["grid_size", "grid"]


# ──────────────────────────────────────────────────────────────────────────
# 1. Video discovery
# ──────────────────────────────────────────────────────────────────────────
def discover_videos(dataset_root, categories):
    """Return {category: [Path, ...]} for every .mp4 found."""
    found = {}
    for cat in categories:
        cat_dir = dataset_root / cat
        if not cat_dir.is_dir():
            print(f"[WARN] Category folder not found: {cat_dir}")
            found[cat] = []
            continue
        vids = sorted(cat_dir.glob("*.mp4"))
        found[cat] = vids
        print(f"[INFO] {cat}: {len(vids)} video(s) found.")
    return found


# ──────────────────────────────────────────────────────────────────────────
# 2. XML annotation resolution (requirement #4)
# ──────────────────────────────────────────────────────────────────────────
def find_xml_for_video(video_path, annotations_root):
    """Recursively search annotations_root for an XML whose stem matches the
    video's stem, regardless of internal folder layout."""
    stem = video_path.stem
    if not annotations_root.is_dir():
        return None
    matches = list(annotations_root.rglob(f"{stem}.xml"))
    if not matches:
        # Case-insensitive fallback, in case of MVI_xxx vs mvi_xxx mismatches.
        matches = [
            p for p in annotations_root.rglob("*.xml")
            if p.stem.lower() == stem.lower()
        ]
    return matches[0] if matches else None


def build_xml_index(videos_by_category, annotations_root):
    """Returns (xml_map, missing) where xml_map[video_path] = xml_path for
    every video that has one, and missing is the list of videos that don't."""
    xml_map = {}
    missing = []
    for cat, vids in videos_by_category.items():
        for v in vids:
            xml_path = find_xml_for_video(v, annotations_root)
            if xml_path is None:
                missing.append(v)
            else:
                xml_map[v] = xml_path
    return xml_map, missing


# ──────────────────────────────────────────────────────────────────────────
# 3. Density metric (requirement #2) -- CSV first, XML fallback
# ──────────────────────────────────────────────────────────────────────────
def _find_col(columns, candidates, blacklist=None):
    blacklist = blacklist or []
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        for lc, orig in cols_lower.items():
            if cand in lc and not any(b in lc for b in blacklist):
                return orig
    return None


def load_density_from_csv(csv_path, grid_rows, grid_cols):
    """Try to build {video_stem_lower: density_value} from per_video_results.csv.
    Returns (density_map, info) where info describes what was detected/used,
    or (None, info) if the CSV could not be used at all."""
    info = {"csv_path": str(csv_path), "used": False, "reason": None}

    if not csv_path.exists():
        info["reason"] = f"File not found: {csv_path}"
        return None, info

    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = reader.fieldnames or []
    except Exception as e:
        info["reason"] = f"Could not read CSV: {e}"
        return None, info

    if not rows:
        info["reason"] = "CSV has no rows."
        return None, info

    video_col = _find_col(columns, DENSITY_CSV_VIDEO_CANDIDATES)
    density_col = _find_col(columns, DENSITY_CSV_DENSITY_CANDIDATES, DENSITY_CSV_BLACKLIST)
    grid_row_col = _find_col(columns, DENSITY_CSV_GRID_ROW_CANDIDATES)
    grid_col_col = _find_col(columns, DENSITY_CSV_GRID_COL_CANDIDATES)
    already_claimed = {video_col, density_col, grid_row_col, grid_col_col}
    grid_combo_col = None
    if not (grid_row_col and grid_col_col):
        grid_combo_col = _find_col(
            [c for c in columns if c not in already_claimed], DENSITY_CSV_GRID_COMBO_CANDIDATES
        )

    info["columns_found"] = columns
    info["video_col"] = video_col
    info["density_col"] = density_col
    info["grid_row_col"] = grid_row_col
    info["grid_col_col"] = grid_col_col
    info["grid_combo_col"] = grid_combo_col

    if video_col is None or density_col is None:
        info["reason"] = (
            f"Could not confidently identify a video-name column and a "
            f"density/vehicle-count column among: {columns}. "
            f"Falling back to XML-derived density for all videos."
        )
        return None, info

    # Filter to the target grid config if a grid column was identified, so
    # a CSV with multiple grid configs per video (e.g. your 6x6/10x16/2x3
    # batch_runner.py output) doesn't average across incompatible rows.
    def row_matches_grid(row):
        try:
            if grid_row_col and grid_col_col:
                return (
                    int(float(row[grid_row_col])) == grid_rows
                    and int(float(row[grid_col_col])) == grid_cols
                )
            if grid_combo_col:
                val = str(row[grid_combo_col])
                nums = re.findall(r"\d+", val)
                if len(nums) >= 2:
                    return int(nums[0]) == grid_rows and int(nums[1]) == grid_cols
        except (ValueError, KeyError, TypeError):
            return True  # can't parse -- don't filter this row out
        return True  # no grid column at all -- keep every row

    filtered_rows = [r for r in rows if row_matches_grid(r)]
    if grid_row_col or grid_col_col or grid_combo_col:
        info["grid_filter_applied"] = True
        info["rows_before_grid_filter"] = len(rows)
        info["rows_after_grid_filter"] = len(filtered_rows)
        if not filtered_rows:
            info["reason"] = (
                f"Grid column(s) found ({grid_row_col or grid_combo_col}) but no "
                f"rows matched target grid {grid_rows}x{grid_cols}. "
                f"Falling back to XML-derived density for all videos."
            )
            return None, info
    else:
        info["grid_filter_applied"] = False

    density_map = {}
    skipped_unparseable = 0
    for row in filtered_rows:
        vid_raw = str(row.get(video_col, "")).strip()
        if not vid_raw:
            continue
        vid_stem = Path(vid_raw).stem.lower()
        try:
            val = float(row[density_col])
        except (ValueError, TypeError):
            skipped_unparseable += 1
            continue
        density_map.setdefault(vid_stem, []).append(val)

    density_map = {k: statistics.mean(v) for k, v in density_map.items()}
    info["skipped_unparseable_density_values"] = skipped_unparseable
    info["videos_mapped"] = len(density_map)

    if not density_map:
        info["reason"] = "No usable (video, density) pairs after parsing. Falling back to XML."
        return None, info

    info["used"] = True
    info["reason"] = f"Using CSV column '{density_col}' keyed by '{video_col}'."
    return density_map, info


def compute_xml_density(xml_path):
    """Fallback density metric computed directly from ground truth: mean
    number of annotated vehicles per frame."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        return None, f"XML parse error: {e}"
    root = tree.getroot()
    frame_counts = []
    unique_ids = set()
    for frame in root.findall("frame"):
        target_list = frame.find("target_list")
        n = 0
        if target_list is not None:
            for target in target_list.findall("target"):
                n += 1
                tid = target.get("id")
                if tid is not None:
                    unique_ids.add(tid)
        frame_counts.append(n)
    if not frame_counts:
        return None, "No <frame> elements found in XML."
    return {
        "avg_vehicles_per_frame": statistics.mean(frame_counts),
        "unique_vehicle_count": len(unique_ids),
        "frame_count": len(frame_counts),
    }, None


def load_density_data(videos_by_category, xml_map, csv_path, grid_rows, grid_cols):
    """Returns {video_path: density_value_float} using CSV where possible,
    XML-derived density otherwise, plus a report dict for logging."""
    csv_density, csv_info = load_density_from_csv(csv_path, grid_rows, grid_cols)

    density = {}
    source = {}
    xml_fallback_errors = []

    for cat, vids in videos_by_category.items():
        for v in vids:
            stem_lower = v.stem.lower()
            if csv_density is not None and stem_lower in csv_density:
                density[v] = csv_density[stem_lower]
                source[v] = "csv"
                continue
            # CSV unusable, OR this specific video wasn't found in the CSV
            # (e.g. it's outside your original 38-video batch) -- fall back
            # to XML for this video specifically.
            xml_path = xml_map.get(v)
            if xml_path is None:
                continue  # no annotation at all; handled separately as "missing"
            result, err = compute_xml_density(xml_path)
            if result is None:
                xml_fallback_errors.append((v, err))
                continue
            density[v] = result["avg_vehicles_per_frame"]
            source[v] = "xml_fallback"

    report = {
        "csv_info": csv_info,
        "videos_using_csv_density": sum(1 for s in source.values() if s == "csv"),
        "videos_using_xml_fallback_density": sum(1 for s in source.values() if s == "xml_fallback"),
        "xml_fallback_errors": [(str(v), e) for v, e in xml_fallback_errors],
    }
    return density, source, report


# ──────────────────────────────────────────────────────────────────────────
# 4. Selection: tier by density within each category (requirements #2, #3)
# ──────────────────────────────────────────────────────────────────────────
def select_videos(videos_by_category, xml_map, density, source, tiers, max_total):
    """Returns (selection, selection_log) where selection is a list of dicts:
    {video, category, tier, density, density_source, xml_path}."""
    per_category_tiers = {}  # cat -> {tier_name: (video, density)}
    log_lines = []

    for cat, vids in videos_by_category.items():
        # Only videos that have BOTH an XML annotation AND a density value
        # are eligible -- requirement #4 says missing annotations must be
        # reported, not silently worked around.
        eligible = [(v, density[v]) for v in vids if v in xml_map and v in density]
        excluded = [v for v in vids if v not in xml_map]
        for v in excluded:
            log_lines.append(f"[EXCLUDE] {cat}/{v.name}: no matching XML annotation found.")
        no_density = [v for v in vids if v in xml_map and v not in density]
        for v in no_density:
            log_lines.append(f"[EXCLUDE] {cat}/{v.name}: has XML but density could not be computed.")

        if not eligible:
            log_lines.append(f"[WARN] {cat}: no eligible videos (with XML + density) at all.")
            per_category_tiers[cat] = {}
            continue

        eligible.sort(key=lambda t: t[1])  # ascending density
        n = len(eligible)
        n_tiers = len(tiers)
        tier_picks = {}
        if n < n_tiers:
            log_lines.append(
                f"[NOTE] {cat}: only {n} eligible video(s), fewer than the "
                f"{n_tiers} requested tiers ({', '.join(tiers)}). "
                f"Selecting {n} video(s) spanning the available density range."
            )
            chosen_idxs = sorted(set(min(i * (n - 1) // max(1, n_tiers - 1), n - 1) for i in range(n_tiers))) \
                if n_tiers > 1 else [n // 2]
            chosen_idxs = chosen_idxs[:n]
            for tier, idx in zip(tiers, chosen_idxs):
                tier_picks[tier] = eligible[idx]
        else:
            # Terciles (or n-iles for len(tiers) != 3): split sorted list into
            # len(tiers) contiguous buckets, pick the middle video of each.
            bucket_size = n / n_tiers
            for i, tier in enumerate(tiers):
                lo = int(round(i * bucket_size))
                hi = int(round((i + 1) * bucket_size))
                hi = max(hi, lo + 1)
                bucket = eligible[lo:hi]
                pick = bucket[len(bucket) // 2]  # middle of the bucket
                tier_picks[tier] = pick

        per_category_tiers[cat] = tier_picks
        for tier, (v, d) in tier_picks.items():
            log_lines.append(
                f"[SELECT] {cat}/{tier}: {v.name}  (density={d:.3f}, "
                f"source={source.get(v, '?')}, n_eligible_in_category={n})"
            )

    # Flatten round-robin across (category, tier) so a MAX_TOTAL_VIDEOS cap
    # stays balanced across categories instead of draining one category first.
    flat = []
    for tier in tiers:
        for cat in videos_by_category:
            pick = per_category_tiers.get(cat, {}).get(tier)
            if pick:
                v, d = pick
                flat.append({
                    "video": v,
                    "category": cat,
                    "tier": tier,
                    "density": d,
                    "density_source": source.get(v, "?"),
                    "xml_path": xml_map[v],
                })

    if max_total is not None and len(flat) > max_total:
        log_lines.append(
            f"[NOTE] {len(flat)} candidates selected but MAX_TOTAL_VIDEOS={max_total}; "
            f"truncating (round-robin order, so this drops the highest tiers/last "
            f"categories reached first rather than any one category entirely)."
        )
        flat = flat[:max_total]

    return flat, log_lines


# ──────────────────────────────────────────────────────────────────────────
# 5. Run isolation + execution (requirements #5, #6, #7, #8, #12)
# ──────────────────────────────────────────────────────────────────────────
def write_config(workdir, video_path, xml_path):
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "video": str(video_path.resolve()),
        "color_channel": COLOR_CHANNEL,
        "grids": {"rows": GRID_ROWS, "cols": GRID_COLS},
        "xml_path": str(xml_path.resolve()) if xml_path else "",
        "pixels_per_meter": PIXELS_PER_METER,
    }
    with open(workdir / "user_input_data.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def run_one(script_path, workdir, video_path, xml_path, profile_on, extra_env=None):
    write_config(workdir, video_path, xml_path)
    env = os.environ.copy()
    env["BTP_PROFILE"] = "1" if profile_on else "0"
    env["BTP_DISPLAY"] = "0"
    env["BTP_SAVE_FRAMES"] = "0"
    env.setdefault("MPLBACKEND", "Agg")  # evaluation plots must not need a display
    for k, v in (extra_env or {}).items():
        env[k] = v
    log_path = workdir / "run.log"
    record = {
        "workdir": str(workdir),
        "script": script_path.name,
        "profile_on": profile_on,
        "extra_env": dict(extra_env or {}),
        "log": str(log_path),
    }
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(workdir),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=RUN_TIMEOUT_SECONDS,
            )
        record["wall_seconds"] = time.time() - t0
        record["returncode"] = result.returncode
        record["status"] = "ok" if result.returncode == 0 else "nonzero_exit"
    except subprocess.TimeoutExpired:
        record["wall_seconds"] = time.time() - t0
        record["status"] = "timeout"
        record["returncode"] = None
    except Exception as e:
        record["wall_seconds"] = time.time() - t0
        record["status"] = "exception"
        record["returncode"] = None
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()

    # Extra sanity check beyond returncode: did the expected metrics file
    # actually get written? A script can exit 0 but still not have produced
    # usable output in unusual edge cases.
    expected = "seq_frame_metrics.json" if script_path.name.startswith("seq") else "rpar_frame_metrics.json"
    record["expected_output_present"] = (workdir / expected).exists()
    if record["status"] == "ok" and not record["expected_output_present"]:
        record["status"] = "ok_but_missing_output"

    return record


def process_video(entry, output_root, run_profiled_pass, resume=False):
    """Run all 4 (impl x pass) combinations for one selected video. Never
    raises -- any failure is captured in the returned record so the caller
    can continue with the next video (requirement #12)."""
    video = entry["video"]
    cat = entry["category"]
    tier = entry["tier"]
    xml_path = entry["xml_path"]

    video_dir = output_root / cat / video.stem
    result = {
        "video": video.name, "category": cat, "tier": tier,
        "density": entry["density"], "density_source": entry["density_source"],
        "xml_path": str(xml_path), "runs": {},
    }

    plan = []
    for impl_name, script, extra_env, profile_too in WORKLOADS:
        plan.append((impl_name, script, "clean", False, extra_env))
        if run_profiled_pass and profile_too:
            plan.append((impl_name, script, "profiled", True, extra_env))

    for impl_name, script, pass_name, profile_on, extra_env in plan:
        workdir = video_dir / impl_name / pass_name
        key = f"{impl_name}/{pass_name}"
        expected = workdir / ("seq_frame_metrics.json" if script.name.startswith("seq") else "rpar_frame_metrics.json")
        if resume and expected.exists() and (workdir / "eval_metrics.json").exists():
            result["runs"][key] = {"workdir": str(workdir), "script": script.name, "profile_on": profile_on,
                                   "extra_env": dict(extra_env or {}), "log": str(workdir / "run.log"),
                                   "status": "ok", "resumed": True, "expected_output_present": True}
            print(f"    [{key}] already complete (resumed) -> {workdir}")
            continue
        try:
            record = run_one(script, workdir, video, xml_path, profile_on, extra_env)
        except Exception as e:  # belt-and-suspenders; run_one already catches internally
            record = {
                "status": "exception", "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(), "workdir": str(workdir),
            }
        result["runs"][key] = record
        status = record.get("status")
        symbol = "OK" if status == "ok" else f"FAILED ({status})"
        print(f"    [{key}] {symbol}  ({record.get('wall_seconds', 0):.1f}s)  -> {workdir}")

    return result


# ──────────────────────────────────────────────────────────────────────────
# 6. Aggregation (requirements #9, #10, #11)
# ──────────────────────────────────────────────────────────────────────────
def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _run_dir(entry, key):
    rec = entry["runs"].get(key)
    return Path(rec["workdir"]) if rec and rec.get("workdir") else None


def _metrics_file(impl):
    return "seq_frame_metrics.json" if impl == "sequential" else "rpar_frame_metrics.json"


def _load_run(entry, impl, pass_name="clean"):
    d = _run_dir(entry, f"{impl}/{pass_name}")
    if d is None:
        return None, None
    return _read_json(d / _metrics_file(impl)), _read_json(d / "eval_metrics.json")


def timing_fields(m):
    keys = ["frames", "lanes", "total_time_seconds", "mean_frame_time_ms", "mean_frame_io_ms",
            "fps", "roi_frame_throughput", "peak_memory_mb", "cpu_utilization_of_one_core"]
    if m is None:
        return {k: None for k in keys}
    ft = m.get("frame_times_seconds", [])
    io = m.get("frame_io_times_seconds", [])
    lanes = m.get("lanes", 1)
    frames = m.get("frames", 0)
    total = m.get("total_time_seconds", 0) or 0
    mem = m.get("memory_usages_mb", [])
    return {
        "frames": frames,
        "lanes": lanes,
        "total_time_seconds": total,
        "mean_frame_time_ms": (statistics.mean(ft) * 1000) if ft else None,
        "mean_frame_io_ms": (statistics.mean(io) * 1000) if io else None,
        "fps": (frames / total) if total else None,
        "roi_frame_throughput": ((frames * lanes) / total) if total else None,
        "peak_memory_mb": max(mem) if mem else None,
        "cpu_utilization_of_one_core": m.get("cpu_utilization_of_one_core"),
    }


def eval_fields(payload):
    out = {}
    m = (payload or {}).get("metrics", {})
    for c in ["Car", "Van", "Bus"]:
        cm = m.get(c, {})
        for k in ["precision", "recall", "f1"]:
            out[f"{c.lower()}_{k}"] = cm.get(k)
    for k in ["unique_gt_vehicles", "unique_predicted_tracks", "matched_gt_vehicles",
              "matched_predicted_tracks"]:
        out[k] = (payload or {}).get(k)
    return out


def build_consolidated_rows(run_results, output_root):
    """One row per (video, workload, lane), clean pass only. Timing fields
    describe the whole run (all lanes); accuracy fields are per lane."""
    rows = []
    for entry in run_results:
        stem = Path(entry["video"]).stem
        for impl, _script, _env, _prof in WORKLOADS:
            m, ev = _load_run(entry, impl)
            if impl == "sequential":
                lane_payloads = [("Lane1(ROI1)", ev)]
            else:
                lane_payloads = [(ln, (ev or {}).get(ln)) for ln in ("Lane1", "Lane2")
                                 if ev is None or ln in ev]
            for lane_name, payload in lane_payloads:
                row = {"video": stem, "category": entry["category"], "tier": entry["tier"],
                       "density": entry.get("density"), "implementation": impl, "lane": lane_name}
                row.update(timing_fields(m))
                row.update(eval_fields(payload))
                row["run_status"] = (entry["runs"].get(f"{impl}/clean") or {}).get("status")
                rows.append(row)
    return rows


def compute_equivalence_rows(run_results):
    """Exact Lane-1 equivalence of every workload against seq.py, plus Lane-2
    equivalence between the serial and threaded 2-lane runs."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_equivalence import compare_payloads
    rows = []
    for entry in run_results:
        stem = Path(entry["video"]).stem
        _, a = _load_run(entry, "sequential")
        _, b = _load_run(entry, "parallel_1lane")
        _, c = _load_run(entry, "serial_2lane")
        _, d = _load_run(entry, "parallel")
        _, a_p = _load_run(entry, "sequential", "profiled")
        _, d_p = _load_run(entry, "parallel", "profiled")
        checks = [
            ("A_seq_vs_B_par1lane_Lane1", a, (b or {}).get("Lane1")),
            ("A_seq_vs_C_serial2lane_Lane1", a, (c or {}).get("Lane1")),
            ("A_seq_vs_D_par2lane_Lane1", a, (d or {}).get("Lane1")),
            ("C_serial_vs_D_threaded_Lane2", (c or {}).get("Lane2"), (d or {}).get("Lane2")),
        ]
        if a_p is not None and d_p is not None:
            checks.append(("A_seq_vs_D_par2lane_Lane1_profiledpass", a_p, (d_p or {}).get("Lane1")))
        for name, x, y in checks:
            equal, diffs = compare_payloads(x, y)
            rows.append({"video": stem, "category": entry["category"], "tier": entry["tier"],
                         "check": name, "identical": equal, "differing_fields": ";".join(diffs)})
    return rows


def _ratio(a, b):
    return (a / b) if (a is not None and b) else None


def compute_speedup_rows(run_results, sysinfo):
    """Per-video speedups. Every figure names its workload pair:
      naive_speedup                 T_A / T_D        (1 lane vs 2 lanes, raw)
      normalized_speedup            2*T_A / T_D      (ROI-frame throughput, end-to-end)
      normalized_speedup_per_frame  2*f_A / f_D      (same, timed per-frame loop only)
      same_workload_speedup_2lane   T_C / T_D        (threads vs no threads, 2 lanes)
      same_workload_speedup_2lane_per_frame  f_C / f_D
      same_workload_speedup_1lane   T_A / T_B        (seq.py vs threaded LaneTracker, 1 lane)
      same_workload_speedup_1lane_per_frame  f_A / f_B
    T = total_time_seconds, f = mean frame_times_seconds. Efficiency
    E(N) = S/N is reported for the threads-only speedup (T_C/T_D) at
    N = 2 (_LANE_POOL width), physical cores and logical processors."""
    n_phys = (sysinfo or {}).get("physical_cores")
    n_log = (sysinfo or {}).get("logical_processors")
    rows = []
    for entry in run_results:
        r = {"video": Path(entry["video"]).stem, "category": entry["category"],
             "tier": entry["tier"], "density": entry.get("density")}
        t = {}
        f = {}
        for key, impl in (("A", "sequential"), ("B", "parallel_1lane"),
                          ("C", "serial_2lane"), ("D", "parallel")):
            m, _ = _load_run(entry, impl)
            t[key] = (m or {}).get("total_time_seconds")
            ft = (m or {}).get("frame_times_seconds") or []
            f[key] = statistics.mean(ft) if ft else None
            r[f"T_{key}_s"] = t[key]
            r[f"frame_ms_{key}"] = f[key] * 1000 if f[key] else None
        r["naive_speedup"] = _ratio(t["A"], t["D"])
        r["normalized_speedup"] = _ratio(2 * t["A"], t["D"]) if t["A"] else None
        r["normalized_speedup_per_frame"] = _ratio(2 * f["A"], f["D"]) if f["A"] else None
        r["percent_improvement_normalized"] = ((r["normalized_speedup"] - 1) * 100) if r["normalized_speedup"] else None
        r["same_workload_speedup_2lane"] = _ratio(t["C"], t["D"])
        r["same_workload_speedup_2lane_per_frame"] = _ratio(f["C"], f["D"])
        r["same_workload_speedup_1lane"] = _ratio(t["A"], t["B"])
        r["same_workload_speedup_1lane_per_frame"] = _ratio(f["A"], f["B"])
        s2 = r["same_workload_speedup_2lane"]
        r["efficiency_N2_lanepool"] = _ratio(s2, 2)
        r["efficiency_N_physical"] = _ratio(s2, n_phys)
        r["efficiency_N_logical"] = _ratio(s2, n_log)
        rows.append(r)
    return rows


NATIVE_BROAD = ["dilate", "GaussianBlur", "cvtColor", "absdiff", "split",
                "'read' of 'cv2.VideoCapture", "'write' of 'cv2.VideoWriter"]


def _amdahl(p, n):
    return 1.0 / ((1.0 - p) + p / n) if (p is not None and n) else None


def compute_amdahl_rows(run_results, sysinfo):
    """Per-video Amdahl estimates from seq.py's profiled pass.
    narrow p = cumtime(process_grid_channel + apply_histogram_equalization)/total
      -- the work rparallel_he3.py actually dispatches to _CELL_POOL;
    broad p  = tottime of the seven dominant native calls / total."""
    import pstats
    n_phys = (sysinfo or {}).get("physical_cores")
    n_log = (sysinfo or {}).get("logical_processors")
    rows = []
    for entry in run_results:
        d = _run_dir(entry, "sequential/profiled")
        prof = d / "profile_seq.prof" if d else None
        if not prof or not prof.exists():
            continue
        st = pstats.Stats(str(prof))
        total = st.total_tt
        narrow_t = broad_t = 0.0
        for (fname, line, func), (cc, nc, tt, ct, callers) in st.stats.items():
            if func in ("process_grid_channel", "apply_histogram_equalization") and fname.endswith("seq.py"):
                narrow_t += ct
            if fname == "~" and any(k in func for k in NATIVE_BROAD):
                broad_t += tt
        row = {"video": Path(entry["video"]).stem, "category": entry["category"], "tier": entry["tier"],
               "profiled_total_s": total, "narrow_parallel_s": narrow_t, "broad_native_s": broad_t,
               "p_narrow": narrow_t / total if total else None, "p_broad": broad_t / total if total else None}
        for label, p in (("narrow", row["p_narrow"]), ("broad", row["p_broad"])):
            row[f"S_{label}_N2"] = _amdahl(p, 2)
            row[f"S_{label}_Nphys"] = _amdahl(p, n_phys)
            row[f"S_{label}_Nlogical"] = _amdahl(p, n_log)
            row[f"S_{label}_inf"] = (1.0 / (1.0 - p)) if p is not None and p < 1 else None
        rows.append(row)
    return rows


def _pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else None


def write_csv(rows, path, fieldnames=None):
    if not rows:
        print(f"[WARN] No rows to write for {path}")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[OK] Wrote {path} ({len(rows)} rows)")


def compute_speedup_rows(run_results):
    """Per-video normalized speedup: (parallel ROI-frame throughput) /
    (sequential ROI-frame throughput), using the clean pass only."""
    rows = []
    for entry in run_results:
        video, cat, tier = entry["video"], entry["category"], entry["tier"]
        seq_dir = Path(entry["runs"]["sequential/clean"]["workdir"])
        par_dir = Path(entry["runs"]["parallel/clean"]["workdir"])
        seq_m = _read_json(seq_dir / "seq_frame_metrics.json")
        par_m = _read_json(par_dir / "rpar_frame_metrics.json")
        if not seq_m or not par_m:
            rows.append({
                "video": Path(video).stem, "category": cat, "tier": tier,
                "naive_speedup": None, "normalized_speedup": None,
                "percent_improvement_normalized": None,
                "note": "missing seq or parallel clean-pass metrics (see run_status)",
            })
            continue
        seq_total = seq_m.get("total_time_seconds", 0) or 0
        par_total = par_m.get("total_time_seconds", 0) or 0
        seq_thr = (seq_m["frames"] * seq_m.get("lanes", 1)) / seq_total if seq_total else None
        par_thr = (par_m["frames"] * par_m.get("lanes", 2)) / par_total if par_total else None
        naive = (seq_total / par_total) if par_total else None
        norm = (par_thr / seq_thr) if (seq_thr and par_thr) else None
        rows.append({
            "video": Path(video).stem, "category": cat, "tier": tier,
            "seq_total_time_s": seq_total, "par_total_time_s": par_total,
            "naive_speedup": naive,
            "normalized_speedup": norm,
            "percent_improvement_normalized": ((norm - 1) * 100) if norm is not None else None,
        })
    return rows


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return (None, None, 0)
    if len(values) == 1:
        return (values[0], 0.0, 1)
    return (statistics.mean(values), statistics.stdev(values), len(values))


def summarize(rows, group_key, value_keys):
    """rows: list of dicts. Returns list of summary dicts, one per distinct
    group_key value, with mean/std/n for each of value_keys."""
    groups = {}
    for r in rows:
        groups.setdefault(r[group_key], []).append(r)
    out = []
    for g, grows in groups.items():
        summary = {group_key: g, "n": len(grows)}
        for vk in value_keys:
            m, s, n = mean_std([r.get(vk) for r in grows])
            summary[f"{vk}_mean"] = m
            summary[f"{vk}_std"] = s
            summary[f"{vk}_n"] = n
        out.append(summary)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    dry_run = "--dry-run" in sys.argv
    reselect = "--reselect" in sys.argv
    resume = "--resume" in sys.argv
    run_profiled = RUN_PROFILED_PASS and "--no-profile" not in sys.argv
    max_videos = None
    if "--max-videos" in sys.argv:
        max_videos = int(sys.argv[sys.argv.index("--max-videos") + 1])

    OUTPUT_ROOT = _DEFAULT_OUTPUT_ROOT
    if (OUTPUT_ROOT / "FINAL_SUMMARY.txt").exists() and not resume:
        OUTPUT_ROOT = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        print(f"[INFO] {_DEFAULT_OUTPUT_ROOT} already holds a finished run; writing to {OUTPUT_ROOT} instead.")

    print("=" * 78)
    print("STEP 1: Discovering videos")
    print("=" * 78)
    videos_by_category = discover_videos(DATASET_ROOT, CATEGORIES)
    total_videos = sum(len(v) for v in videos_by_category.values())
    if total_videos == 0:
        print("[FATAL] No videos found under DATASET_ROOT. Check the path in the "
              "CONFIG block at the top of this script.")
        return

    print("\n" + "=" * 78)
    print("STEP 2: Resolving DETRAC XML annotations (requirement #4)")
    print("=" * 78)
    xml_map, missing_xml = build_xml_index(videos_by_category, ANNOTATIONS_ROOT)
    print(f"[INFO] {len(xml_map)} / {total_videos} videos have a matching XML annotation.")
    if missing_xml:
        print(f"[WARN] {len(missing_xml)} video(s) are MISSING an XML annotation "
              f"and will be EXCLUDED from selection:")
        for v in missing_xml:
            print(f"         - {v}")

    print("\n" + "=" * 78)
    print("STEP 3: Computing traffic density per video (requirement #2)")
    print("=" * 78)
    density, source, density_report = load_density_data(
        videos_by_category, xml_map, PER_VIDEO_RESULTS_CSV, GRID_ROWS, GRID_COLS
    )
    print(f"[INFO] CSV info: {density_report['csv_info']}")
    print(f"[INFO] Videos using CSV density: {density_report['videos_using_csv_density']}")
    print(f"[INFO] Videos using XML-fallback density: {density_report['videos_using_xml_fallback_density']}")
    if density_report["xml_fallback_errors"]:
        print(f"[WARN] {len(density_report['xml_fallback_errors'])} video(s) failed XML-density fallback too:")
        for v, e in density_report["xml_fallback_errors"]:
            print(f"         - {v}: {e}")

    print("\n" + "=" * 78)
    print("STEP 4: Selecting videos (requirement #3: light/medium/heavy, balanced by category)")
    print("=" * 78)
    selection, selection_log = select_videos(
        videos_by_category, xml_map, density, source, TIERS, MAX_TOTAL_VIDEOS
    )
    for line in selection_log:
        print(line)
    if not reselect and SELECTION_REPORT.exists():
        fixed = json.load(open(SELECTION_REPORT))["selection"]
        by_stem = {v.stem: (cat, v) for cat, vids in videos_by_category.items() for v in vids}
        new_sel = []
        for e in fixed:
            stem = Path(e["video"].replace("\\", "/")).stem
            if stem not in by_stem or by_stem[stem][1] not in xml_map:
                print(f"[WARN] fixed-selection video {stem} not found locally (or no XML); skipped.")
                continue
            cat, vpath = by_stem[stem]
            new_sel.append({"video": vpath, "category": e["category"], "tier": e["tier"],
                            "density": e["density"], "density_source": e.get("density_source", "?"),
                            "xml_path": xml_map[vpath]})
        print(f"[INFO] Using FIXED selection from {SELECTION_REPORT} ({len(new_sel)} video(s)); "
              "pass --reselect to re-derive it.")
        selection = new_sel
    if max_videos is not None:
        selection = selection[:max_videos]
        print(f"[INFO] --max-videos {max_videos}: running a subset.")
    print(f"\n[INFO] Final selection: {len(selection)} video(s).")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from bench_sysinfo import collect_system_info
        sysinfo = collect_system_info()
    except Exception as e:
        sysinfo = {"error": f"{type(e).__name__}: {e}"}
    with open(OUTPUT_ROOT / "system_info.json", "w") as f:
        json.dump(sysinfo, f, indent=2)
    selection_report_path = OUTPUT_ROOT / "selection_report.json"
    with open(selection_report_path, "w") as f:
        json.dump({
            "selection": [
                {"video": str(e["video"]), "category": e["category"], "tier": e["tier"],
                 "density": e["density"], "density_source": e["density_source"],
                 "xml_path": str(e["xml_path"])}
                for e in selection
            ],
            "missing_xml_annotations": [str(v) for v in missing_xml],
            "density_report": density_report,
            "selection_log": selection_log,
        }, f, indent=2)
    print(f"[OK] Selection report written to {selection_report_path}")

    if dry_run:
        print("\n[DRY RUN] Stopping before execution (--dry-run passed). "
              "Re-run without --dry-run to actually process these videos.")
        return

    if not selection:
        print("[FATAL] No videos selected -- nothing to run. Check the warnings above "
              "(likely: missing XML annotations, or DATASET_ROOT/ANNOTATIONS_ROOT paths).")
        return

    print("\n" + "=" * 78)
    print(f"STEP 5: Running both implementations on {len(selection)} video(s)")
    print(f"        (profiled pass: {'ON' if run_profiled else 'OFF'})")
    print("=" * 78)
    run_results = []
    failures = []
    for i, entry in enumerate(selection, 1):
        print(f"\n[{i}/{len(selection)}] {entry['category']}/{entry['tier']}: {entry['video'].name}")
        try:
            result = process_video(entry, OUTPUT_ROOT, run_profiled, resume)
        except Exception as e:
            # Should not happen (process_video already catches internally),
            # but requirement #12 demands the whole batch never dies.
            print(f"    [FATAL FOR THIS VIDEO ONLY] {type(e).__name__}: {e}")
            failures.append({
                "video": entry["video"].name, "category": entry["category"],
                "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
            })
            continue
        run_results.append(result)
        for key, rec in result["runs"].items():
            if rec.get("status") not in ("ok",):
                failures.append({
                    "video": entry["video"].name, "category": entry["category"],
                    "run": key, "status": rec.get("status"),
                    "error": rec.get("error"), "log": rec.get("log"),
                })

    run_results_path = OUTPUT_ROOT / "run_results_raw.json"
    with open(run_results_path, "w") as f:
        json.dump(
            [{**r, "runs": {k: {kk: vv for kk, vv in v.items()} for k, v in r["runs"].items()}} for r in run_results],
            f, indent=2, default=str,
        )

    if failures:
        write_csv(failures, OUTPUT_ROOT / "failures.csv")
        print(f"\n[WARN] {len(failures)} run(s) failed. See failures.csv and run.log files "
              f"in the corresponding workdirs.")
    else:
        print("\n[OK] All runs completed with status 'ok'.")

    print("\n" + "=" * 78)
    print("STEP 6: Aggregating results (requirements #9, #10, #11)")
    print("=" * 78)
    rows = build_consolidated_rows(run_results, OUTPUT_ROOT)
    write_csv(rows, OUTPUT_ROOT / "consolidated_results.csv")

    equiv_rows = compute_equivalence_rows(run_results)
    write_csv(equiv_rows, OUTPUT_ROOT / "equivalence.csv")

    speedup_rows = compute_speedup_rows(run_results, sysinfo)
    write_csv(speedup_rows, OUTPUT_ROOT / "speedup_by_video.csv")

    amdahl_rows = compute_amdahl_rows(run_results, sysinfo) if run_profiled else []
    if amdahl_rows:
        write_csv(amdahl_rows, OUTPUT_ROOT / "amdahl_by_video.csv")

    value_keys = ["mean_frame_time_ms", "mean_frame_io_ms", "fps", "roi_frame_throughput",
                  "peak_memory_mb", "cpu_utilization_of_one_core",
                  "car_f1", "van_f1", "bus_f1"]
    for group in ("category", "tier"):
        out = []
        for impl, lane in (("sequential", "Lane1(ROI1)"), ("parallel_1lane", "Lane1"),
                           ("serial_2lane", "Lane1"), ("parallel", "Lane1")):
            sel = [r for r in rows if r["implementation"] == impl and r["lane"] == lane]
            for srow in summarize(sel, group, value_keys):
                srow["implementation"] = f"{impl}({lane})"
                out.append(srow)
        write_csv(out, OUTPUT_ROOT / f"summary_by_{group}.csv")

    speed_keys = ["naive_speedup", "normalized_speedup", "normalized_speedup_per_frame",
                  "same_workload_speedup_2lane", "same_workload_speedup_2lane_per_frame",
                  "same_workload_speedup_1lane", "same_workload_speedup_1lane_per_frame",
                  "efficiency_N2_lanepool", "efficiency_N_physical", "efficiency_N_logical"]
    write_csv(summarize(speedup_rows, "category", speed_keys), OUTPUT_ROOT / "speedup_summary_by_category.csv")
    write_csv(summarize(speedup_rows, "tier", speed_keys), OUTPUT_ROOT / "speedup_summary_by_tier.csv")

    overall = {k: mean_std([r.get(k) for r in speedup_rows]) for k in speed_keys}
    impl_overall = {}
    for impl, lane in (("sequential", "Lane1(ROI1)"), ("parallel_1lane", "Lane1"),
                       ("serial_2lane", "Lane1"), ("parallel", "Lane1")):
        sel = [r for r in rows if r["implementation"] == impl and r["lane"] == lane]
        impl_overall[impl] = {k: mean_std([r.get(k) for r in sel])
                              for k in ("mean_frame_time_ms", "mean_frame_io_ms", "fps",
                                        "total_time_seconds", "peak_memory_mb",
                                        "cpu_utilization_of_one_core")}
    r_density = _pearson([r.get("density") for r in speedup_rows],
                         [r.get("normalized_speedup") for r in speedup_rows])
    amdahl_overall = {k: mean_std([r.get(k) for r in amdahl_rows])
                      for k in ("p_narrow", "p_broad", "S_narrow_N2", "S_broad_N2",
                                "S_narrow_Nphys", "S_broad_Nphys")} if amdahl_rows else {}
    n_equiv = sum(1 for r in equiv_rows if r["identical"])

    def fmt(ms):
        m, s, n = ms
        return f"{m:.4f} ± {s:.4f} (n={n})" if m is not None else "n/a"

    lines = ["=" * 78, "BENCHMARK COMPLETE", "=" * 78,
             f"\nOutput root: {OUTPUT_ROOT}",
             f"\nSystem: {sysinfo.get('cpu_model')} | physical cores {sysinfo.get('physical_cores')} | "
             f"logical {sysinfo.get('logical_processors')} | {sysinfo.get('platform')} | "
             f"Python {sysinfo.get('python_version')} | OpenCV {sysinfo.get('opencv_version')}",
             f"\nVideos ({len(selection)}):"]
    for e in selection:
        lines.append(f"  - {e['category']}/{e['tier']}: {e['video'].name}  (density={e['density']:.3f})")
    lines.append(f"\nFailed runs: {len(failures)} (see failures.csv)")
    lines.append(f"\nEquivalence checks identical: {n_equiv}/{len(equiv_rows)} (see equivalence.csv)")
    lines.append("\nPer-workload means over videos (clean pass, Lane-1 rows):")
    for impl, vals in impl_overall.items():
        lines.append(f"  {impl:15s} " + "  ".join(f"{k}={fmt(v)}" for k, v in vals.items()))
    lines.append("\nSpeedups (mean ± sample std over videos):")
    for k in speed_keys:
        lines.append(f"  {k:40s} {fmt(overall[k])}")
    lines.append(f"\nPearson r(density, normalized_speedup) = "
                 f"{r_density:.3f}" if r_density is not None else "\nPearson r: n/a")
    if amdahl_overall:
        lines.append("\nAmdahl (seq.py profiled pass, per-video then averaged):")
        for k, v in amdahl_overall.items():
            lines.append(f"  {k:20s} {fmt(v)}")
    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    with open(OUTPUT_ROOT / "FINAL_SUMMARY.txt", "w") as f:
        f.write(summary_text + "\n")

if __name__ == "__main__":
    main()
