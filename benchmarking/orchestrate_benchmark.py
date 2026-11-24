"""
orchestrate_benchmark.py
-------------------------
Drives a multi-video benchmark of seq_bench.py vs rparallel_he3_bench.py
across the UA-DETRAC Cloudy/Night/Rainy/Sunny folder structure, selecting
videos by traffic density (light/medium/heavy) within each weather
category, running both implementations in isolated per-run directories,
and aggregating results into consolidated CSVs.

WHAT THIS SCRIPT DOES NOT DO:
It does not modify seq.py or rparallel_he3.py's detection, tracking,
classification, counting, density, or speed algorithms. It only invokes
the *_bench.py copies (which add opt-in instrumentation toggles and one
crash-fix, all documented separately -- see seq_bench.diff /
rparallel_he3_bench.diff) as external subprocesses, writes their
per-run config file, and reads their output files.

===============================================================================
STEP 0 -- EDIT THE CONFIG BLOCK BELOW FOR YOUR MACHINE, THEN RUN:
    python orchestrate_benchmark.py --dry-run     # preview selection only
    python orchestrate_benchmark.py               # run the full benchmark
===============================================================================
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
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════
# CONFIG -- edit these paths/values for your machine before running.
# ══════════════════════════════════════════════════════════════════════════

# UA-DETRAC video root, containing Cloudy/ Night/ Rainy/ Sunny/ subfolders
# (each full of .mp4 files), matching the structure in your screenshot.
DATASET_ROOT = Path(
    r"C:\Users\jaysu\OneDrive\Desktop\PROJECTS\BTP\Lattice-Layer-Vehicle-Detection-\DETRAC_VIDEOS_eval"
)

# UA-DETRAC XML ground-truth annotations root. Searched recursively, so it
# does not matter whether this is flat or split into subfolders internally.
ANNOTATIONS_ROOT = Path(
    r"C:\Users\jaysu\OneDrive\Desktop\PROJECTS\BTP\Lattice-Layer-Vehicle-Detection-\DETRAC-Annotations"
)

# Your existing batch_runner.py output, used to pick videos by density
# rather than randomly. If this file is missing, unreadable, or its columns
# can't be confidently identified, the script automatically falls back to
# computing density directly from each video's XML annotation instead --
# see load_density_data() below -- and prints a clear warning saying so.
PER_VIDEO_RESULTS_CSV = Path(
    r"C:\Users\jaysu\OneDrive\Desktop\PROJECTS\BTP\Lattice-Layer-Vehicle-Detection-\per_video_results.csv"
)

# Where the two benchmark scripts live (the *_bench.py copies, not the
# originals -- see the diffs for exactly what differs and why).
SEQ_SCRIPT = Path(__file__).resolve().parent.parent / "engines" / "seq_bench.py"
PAR_SCRIPT = Path(__file__).resolve().parent.parent / "engines" / "rparallel_he3_bench.py"

# Where all benchmark outputs will be written.
OUTPUT_ROOT = Path(r".\benchmark_results").resolve()

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


def run_one(script_path, workdir, video_path, xml_path, profile_on):
    write_config(workdir, video_path, xml_path)
    env = os.environ.copy()
    env["BTP_PROFILE"] = "1" if profile_on else "0"
    env["BTP_DISPLAY"] = "0"
    env["BTP_SAVE_FRAMES"] = "0"
    log_path = workdir / "run.log"
    record = {
        "workdir": str(workdir),
        "script": script_path.name,
        "profile_on": profile_on,
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
    expected = "seq_frame_metrics.json" if "seq" in script_path.name else "rpar_frame_metrics.json"
    record["expected_output_present"] = (workdir / expected).exists()
    if record["status"] == "ok" and not record["expected_output_present"]:
        record["status"] = "ok_but_missing_output"

    return record


def process_video(entry, output_root, run_profiled_pass):
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

    plan = [("sequential", SEQ_SCRIPT, "clean", False)]
    if run_profiled_pass:
        plan.append(("sequential", SEQ_SCRIPT, "profiled", True))
    plan.append(("parallel", PAR_SCRIPT, "clean", False))
    if run_profiled_pass:
        plan.append(("parallel", PAR_SCRIPT, "profiled", True))

    for impl_name, script, pass_name, profile_on in plan:
        workdir = video_dir / impl_name / pass_name
        key = f"{impl_name}/{pass_name}"
        try:
            record = run_one(script, workdir, video, xml_path, profile_on)
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


def build_consolidated_rows(run_results, output_root):
    """One row per (video, implementation, lane) using the CLEAN pass only
    for timing (requirement #8), and that same clean pass's eval_metrics.json
    for accuracy (identical DETRAC logic regardless of profiling)."""
    rows = []
    for entry in run_results:
        video, cat, tier = entry["video"], entry["category"], entry["tier"]
        stem = Path(video).stem

        seq_clean_dir = Path(entry["runs"]["sequential/clean"]["workdir"])
        par_clean_dir = Path(entry["runs"]["parallel/clean"]["workdir"])

        seq_metrics = _read_json(seq_clean_dir / "seq_frame_metrics.json")
        par_metrics = _read_json(par_clean_dir / "rpar_frame_metrics.json")
        seq_eval = _read_json(seq_clean_dir / "eval_metrics.json")
        par_eval = _read_json(par_clean_dir / "eval_metrics.json")  # {"Lane1": {...}, "Lane2": {...}}

        def timing_fields(m):
            if m is None:
                return {"frames": None, "total_time_seconds": None,
                        "mean_frame_time_ms": None, "fps": None,
                        "roi_frame_throughput": None, "peak_memory_mb": None}
            ft = m.get("frame_times_seconds", [])
            lanes = m.get("lanes", 1)
            frames = m.get("frames", 0)
            total = m.get("total_time_seconds", 0) or 0
            mem = m.get("memory_usages_mb", [])
            return {
                "frames": frames,
                "total_time_seconds": total,
                "mean_frame_time_ms": (statistics.mean(ft) * 1000) if ft else None,
                "fps": (frames / total) if total else None,
                "roi_frame_throughput": ((frames * lanes) / total) if total else None,
                "peak_memory_mb": max(mem) if mem else None,
            }

        def eval_fields(metrics_dict):
            if not metrics_dict:
                return {f"{c.lower()}_{k}": None for c in ["car", "van", "bus"] for k in ["precision", "recall", "f1"]}
            m = metrics_dict.get("metrics", {})
            out = {}
            for c in ["Car", "Van", "Bus"]:
                cm = m.get(c, {})
                out[f"{c.lower()}_precision"] = cm.get("precision")
                out[f"{c.lower()}_recall"] = cm.get("recall")
                out[f"{c.lower()}_f1"] = cm.get("f1")
            return out

        # Sequential row (single ROI == same coordinates as parallel Lane1)
        row = {"video": stem, "category": cat, "tier": tier, "implementation": "sequential", "lane": "Lane1(ROI1)"}
        row.update(timing_fields(seq_metrics))
        row.update(eval_fields(seq_eval))
        row["run_status"] = entry["runs"]["sequential/clean"].get("status")
        rows.append(row)

        # Parallel rows, one per lane
        for lane_name in ("Lane1", "Lane2"):
            row = {"video": stem, "category": cat, "tier": tier, "implementation": "parallel", "lane": lane_name}
            row.update(timing_fields(par_metrics))  # timing is per-run (both lanes), not per-lane
            row.update(eval_fields((par_eval or {}).get(lane_name)))
            row["run_status"] = entry["runs"]["parallel/clean"].get("status")
            rows.append(row)

    return rows


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
    print(f"\n[INFO] Final selection: {len(selection)} video(s).")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
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
    print(f"        (profiled pass: {'ON' if RUN_PROFILED_PASS else 'OFF'})")
    print("=" * 78)
    run_results = []
    failures = []
    for i, entry in enumerate(selection, 1):
        print(f"\n[{i}/{len(selection)}] {entry['category']}/{entry['tier']}: {entry['video'].name}")
        try:
            result = process_video(entry, OUTPUT_ROOT, RUN_PROFILED_PASS)
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

    speedup_rows = compute_speedup_rows(run_results)
    write_csv(speedup_rows, OUTPUT_ROOT / "speedup_by_video.csv")

    # Category summary: sequential vs parallel(Lane1) timing + accuracy
    cat_rows_seq = [r for r in rows if r["implementation"] == "sequential"]
    cat_rows_par = [r for r in rows if r["implementation"] == "parallel" and r["lane"] == "Lane1"]
    value_keys = ["mean_frame_time_ms", "fps", "roi_frame_throughput",
                  "car_f1", "van_f1", "bus_f1"]
    summary_cat_seq = summarize(cat_rows_seq, "category", value_keys)
    summary_cat_par = summarize(cat_rows_par, "category", value_keys)
    for r in summary_cat_seq:
        r["implementation"] = "sequential"
    for r in summary_cat_par:
        r["implementation"] = "parallel(Lane1)"
    write_csv(summary_cat_seq + summary_cat_par, OUTPUT_ROOT / "summary_by_category.csv")

    summary_tier_seq = summarize(cat_rows_seq, "tier", value_keys)
    summary_tier_par = summarize(cat_rows_par, "tier", value_keys)
    for r in summary_tier_seq:
        r["implementation"] = "sequential"
    for r in summary_tier_par:
        r["implementation"] = "parallel(Lane1)"
    write_csv(summary_tier_seq + summary_tier_par, OUTPUT_ROOT / "summary_by_tier.csv")

    speedup_summary_cat = summarize(speedup_rows, "category", ["naive_speedup", "normalized_speedup", "percent_improvement_normalized"])
    write_csv(speedup_summary_cat, OUTPUT_ROOT / "speedup_summary_by_category.csv")
    speedup_summary_tier = summarize(speedup_rows, "tier", ["naive_speedup", "normalized_speedup", "percent_improvement_normalized"])
    write_csv(speedup_summary_tier, OUTPUT_ROOT / "speedup_summary_by_tier.csv")

    overall_speedup_mean, overall_speedup_std, overall_speedup_n = mean_std(
        [r["normalized_speedup"] for r in speedup_rows]
    )

    # ── Final human-readable summary (requirement: "at the end, tell me...") ──
    lines = []
    lines.append("=" * 78)
    lines.append("BENCHMARK COMPLETE")
    lines.append("=" * 78)
    lines.append(f"\nOutput root: {OUTPUT_ROOT}")
    lines.append(f"\nVideos selected ({len(selection)}):")
    for e in selection:
        lines.append(f"  - {e['category']}/{e['tier']}: {e['video'].name}  "
                      f"(density={e['density']:.3f}, source={e['density_source']})")
    lines.append(f"\nMissing XML annotations ({len(missing_xml)}), excluded from selection:")
    for v in missing_xml:
        lines.append(f"  - {v}")
    lines.append(f"\nFailed runs: {len(failures)} (see failures.csv)")
    lines.append(f"\nOverall normalized speedup (parallel vs sequential, "
                  f"workload-corrected, clean pass only):")
    lines.append(f"  mean = {overall_speedup_mean:.3f}x  "
                  f"std = {overall_speedup_std:.3f}  (n={overall_speedup_n})"
                  if overall_speedup_mean is not None else "  (no valid runs)")
    lines.append("\nPer-category speedup: see speedup_summary_by_category.csv")
    lines.append("Per-tier (light/medium/heavy) speedup: see speedup_summary_by_tier.csv")
    lines.append("Per-category accuracy (F1) mean±std: see summary_by_category.csv")
    lines.append("Per-tier accuracy (F1) mean±std: see summary_by_tier.csv")
    lines.append("Full per-video, per-implementation, per-lane rows: consolidated_results.csv")
    lines.append("Full per-video speedup: speedup_by_video.csv")
    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    with open(OUTPUT_ROOT / "FINAL_SUMMARY.txt", "w") as f:
        f.write(summary_text + "\n")


if __name__ == "__main__":
    main()
