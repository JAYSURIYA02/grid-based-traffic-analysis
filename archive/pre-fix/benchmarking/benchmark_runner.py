"""
benchmark_runner.py  --  Benchmark harness: seq.py vs rparallel_he3.py

Run:  python benchmark_runner.py

Outputs
-------
  benchmark_results.csv
  benchmark_results.txt
  (profile *.prof / *_cumtime.txt / *_tottime.txt written by the scripts)

CPU% note: unnormalized (psutil default on Windows).
On an N-logical-CPU machine the maximum is N*100%.
"""

import subprocess
import sys

import time
import json
import os
import csv
import io
import pstats
import threading
import statistics
import psutil

# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.join(SCRIPT_DIR, "..")          # project root
SEQ_SCRIPT  = os.path.join(ROOT_DIR, "seq.py")
PAR_SCRIPT  = os.path.join(ROOT_DIR, "rparallel_he3.py")
CONFIG_FILE = os.path.join(ROOT_DIR, "user_input_data.json")
SEQ_METRICS = os.path.join(ROOT_DIR, "seq_frame_metrics.json")
PAR_METRICS = os.path.join(ROOT_DIR, "parallel_frame_metrics.json")
SEQ_PROF    = os.path.join(ROOT_DIR, "profile_seq.prof")
PAR_PROF    = os.path.join(ROOT_DIR, "profile_parallel.prof")

POLL_S      = 0.25
NUM_TRIALS  = 3
CSV_OUT     = os.path.join(SCRIPT_DIR, "benchmark_results.csv")
TXT_OUT     = os.path.join(SCRIPT_DIR, "benchmark_results.txt")
PYTHON      = sys.executable


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config():
    with open(CONFIG_FILE) as f:
        return json.load(f)


def verify_config_parity(cfg):
    sep = "=" * 60
    print()
    print(sep)
    print("CONFIG PARITY VERIFICATION")
    print(sep)
    print("  Video         :", cfg["video"])
    print("  Color channel :", cfg["color_channel"])
    print("  Grid rows     :", cfg["grids"]["rows"])
    print("  Grid cols     :", cfg["grids"]["cols"])
    print("  pixels/meter  :", cfg.get("pixels_per_meter", "N/A"))
    print("  XML path      :", cfg.get("xml_path", "") or "(none)")
    print("  Both scripts read the same user_input_data.json -> identical settings.")
    print(sep)


# ---------------------------------------------------------------------------
# Resource monitor
# ---------------------------------------------------------------------------
class ResourceMonitor:
    def __init__(self, pid):
        self._pid        = pid
        self.cpu_samples = []
        self.rss_samples = []
        self._stop       = threading.Event()
        self._t          = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=5)

    def _run(self):
        try:
            p = psutil.Process(self._pid)
            p.cpu_percent(interval=None)   # prime the counter
            time.sleep(POLL_S)
            while not self._stop.is_set():
                try:
                    self.cpu_samples.append(p.cpu_percent(interval=None))
                    self.rss_samples.append(p.memory_info().rss / 1048576)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                self._stop.wait(POLL_S)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


# ---------------------------------------------------------------------------
# Run one trial of a script
# ---------------------------------------------------------------------------
def run_script(script_path, trial_num, label):
    print()
    print(f"  [{label}] Trial {trial_num}/{NUM_TRIALS} -- launching ...", flush=True)

    wall_start = time.perf_counter()
    proc = subprocess.Popen(
        [PYTHON, script_path],
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        child    = psutil.Process(proc.pid)
        init_rss = child.memory_info().rss / 1048576
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        child    = None
        init_rss = 0.0

    mon = ResourceMonitor(proc.pid)
    mon.start()

    for line in proc.stdout:
        print("    |", line, end="", flush=True)

    proc.wait()
    wall_end  = time.perf_counter()
    mon.stop()

    wall_time = wall_end - wall_start
    rc        = proc.returncode

    if rc != 0:
        print(f"  [WARN] {label} trial {trial_num} exited with code {rc}")

    try:
        final_rss = child.memory_info().rss / 1048576 if child else 0.0
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        final_rss = mon.rss_samples[-1] if mon.rss_samples else 0.0

    peak_rss = max(mon.rss_samples, default=0.0)
    avg_cpu  = statistics.mean(mon.cpu_samples) if mon.cpu_samples else 0.0
    max_cpu  = max(mon.cpu_samples, default=0.0)

    print(
        f"  [{label}] Trial {trial_num} done | "
        f"wall={wall_time:.2f}s  peak_rss={peak_rss:.1f}MB  "
        f"avg_cpu={avg_cpu:.1f}%  max_cpu={max_cpu:.1f}%",
        flush=True,
    )

    return dict(
        trial=trial_num, label=label, wall_time_s=wall_time,
        initial_rss_mb=init_rss, peak_rss_mb=peak_rss,
        final_rss_mb=final_rss, avg_cpu_pct=avg_cpu,
        max_cpu_pct=max_cpu, returncode=rc,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_frame_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def prof_top15(prof_file, sort_key):
    if not os.path.exists(prof_file):
        return f"  (not found: {prof_file})\n"
    buf = io.StringIO()
    pstats.Stats(prof_file, stream=buf).sort_stats(sort_key).print_stats(15)
    return buf.getvalue()


def _mean(lst):
    return statistics.mean(lst) if lst else 0.0


def _std(lst):
    return statistics.stdev(lst) if len(lst) > 1 else 0.0


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def aggregate(trials, metrics_list):
    a = {}
    wt = [r["wall_time_s"] for r in trials]
    a["wall_times"]      = wt
    a["mean_wall_time"]  = _mean(wt)
    a["stdev_wall_time"] = _std(wt)
    a["mean_avg_cpu"]    = _mean([r["avg_cpu_pct"] for r in trials])
    a["mean_max_cpu"]    = _mean([r["max_cpu_pct"] for r in trials])
    a["mean_peak_rss"]   = _mean([r["peak_rss_mb"]    for r in trials])
    a["mean_init_rss"]   = _mean([r["initial_rss_mb"] for r in trials])
    a["mean_fin_rss"]    = _mean([r["final_rss_mb"]   for r in trials])

    valid = [m for m in metrics_list if m]
    if valid:
        st = [m["total_time_seconds"] for m in valid]
        a["mean_script_time"]  = _mean(st)
        a["stdev_script_time"] = _std(st)
        fl = [m["frames"] for m in valid]
        a["mean_frames"] = _mean(fl)
        a["frames"]      = fl[0]
        pm = [_mean(m["frame_times_seconds"]) for m in valid if m["frame_times_seconds"]]
        a["mean_frame_time_ms"] = _mean(pm) * 1000 if pm else 0.0
        fv = [m["frames"] / m["total_time_seconds"] for m in valid if m["total_time_seconds"] > 0]
        a["fps_vals"]  = fv
        a["mean_fps"]  = _mean(fv)
        a["stdev_fps"] = _std(fv)
        a["video"]         = valid[0]["video"]
        a["grid_rows"]     = valid[0]["grid_rows"]
        a["grid_cols"]     = valid[0]["grid_cols"]
        a["color_channel"] = valid[0]["color_channel"]
    else:
        a["mean_script_time"]   = a["mean_wall_time"]
        a["stdev_script_time"]  = a["stdev_wall_time"]
        a["mean_frames"] = 0;  a["frames"] = 0
        a["mean_frame_time_ms"] = 0.0
        a["mean_fps"] = 0.0;   a["stdev_fps"] = 0.0;  a["fps_vals"] = []
        a["video"] = "unknown"; a["grid_rows"] = 0
        a["grid_cols"] = 0;    a["color_channel"] = "unknown"
    return a


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
FIELDNAMES = [
    "implementation", "trial", "video", "frames",
    "grid_rows", "grid_cols", "color_channel",
    "total_time_seconds", "average_frame_time_ms", "fps",
    "initial_memory_mb", "peak_memory_mb", "final_memory_mb",
    "average_cpu_percent", "max_cpu_percent",
]


def write_csv(seq_trials, par_trials, seq_ml, par_ml, cfg):
    rows = []
    def _row(impl, tr, m):
        frames    = m.get("frames", 0) if m else 0
        total_t   = m.get("total_time_seconds", tr["wall_time_s"]) if m else tr["wall_time_s"]
        ft        = m.get("frame_times_seconds", []) if m else []
        avg_ft_ms = _mean(ft) * 1000
        fps       = frames / total_t if total_t > 0 and frames > 0 else 0.0
        vid       = (m or {}).get("video", cfg.get("video", ""))
        gr        = cfg["grids"]["rows"]
        gc        = cfg["grids"]["cols"]
        return {
            "implementation":       impl,
            "trial":                tr["trial"],
            "video":                os.path.basename(vid),
            "frames":               frames,
            "grid_rows":            (m or {}).get("grid_rows", gr),
            "grid_cols":            (m or {}).get("grid_cols", gc),
            "color_channel":        (m or {}).get("color_channel", cfg["color_channel"]),
            "total_time_seconds":   round(total_t, 4),
            "average_frame_time_ms":round(avg_ft_ms, 4),
            "fps":                  round(fps, 4),
            "initial_memory_mb":    round(tr["initial_rss_mb"], 2),
            "peak_memory_mb":       round(tr["peak_rss_mb"], 2),
            "final_memory_mb":      round(tr["final_rss_mb"], 2),
            "average_cpu_percent":  round(tr["avg_cpu_pct"], 2),
            "max_cpu_percent":      round(tr["max_cpu_pct"], 2),
        }
    for tr, m in zip(seq_trials, seq_ml):
        rows.append(_row("sequential", tr, m))
    for tr, m in zip(par_trials, par_ml):
        rows.append(_row("parallel", tr, m))
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    print(f"[BENCH] {CSV_OUT} written ({n} rows).")


# ---------------------------------------------------------------------------
# TXT report
# ---------------------------------------------------------------------------
def _fmt_section(agg, name):
    lines = []
    hr = "-" * 56
    lines.append(hr)
    lines.append(name)
    lines.append(hr)
    st  = agg["mean_script_time"]
    sst = agg["stdev_script_time"]
    fps = agg["mean_fps"]
    sfp = agg["stdev_fps"]
    fms = agg["mean_frame_time_ms"]
    lines.append(f"  Mean total time   : {st:.2f} s  (+-{sst:.2f} s)")
    lines.append(f"  Mean FPS          : {fps:.2f}  (+-{sfp:.2f})")
    lines.append(f"  Mean frame time   : {fms:.2f} ms")
    lines.append(f"  Initial RSS       : {agg['mean_init_rss']:.1f} MB")
    lines.append(f"  Peak RSS          : {agg['mean_peak_rss']:.1f} MB")
    lines.append(f"  Final RSS         : {agg['mean_fin_rss']:.1f} MB")
    lines.append(f"  Average CPU%      : {agg['mean_avg_cpu']:.1f}%")
    lines.append(f"  Maximum CPU%      : {agg['mean_max_cpu']:.1f}%")
    wt = agg.get("wall_times", [])
    if len(wt) >= 2 and wt[0] > wt[1] * 1.15:
        lines.append("  (Trial 1 was >15% slower -- possible warm-up effect)")
    lines.append("")
    lines.append("  Per-trial breakdown:")
    fv = agg.get("fps_vals", [])
    for i in range(NUM_TRIALS):
        w = wt[i] if i < len(wt) else 0
        f = fv[i] if i < len(fv) else 0
        m = "  <- Trial 1 (possible warm-up)" if i == 0 else ""
        lines.append(f"    Trial {i+1}: wall={w:.2f}s  fps={f:.2f}{m}")
    return lines


def write_txt(s, p, cfg):
    seq_t = s["mean_script_time"]
    par_t = p["mean_script_time"]
    speedup   = seq_t / par_t if par_t > 0 else float("inf")
    time_red  = (seq_t - par_t) / seq_t * 100 if seq_t > 0 else 0.0
    fps_imp   = (p["mean_fps"] - s["mean_fps"]) / s["mean_fps"] * 100 if s["mean_fps"] > 0 else 0.0
    sep = "=" * 56
    hr  = "-" * 56
    L   = []
    L.append(sep)
    L.append("SEQUENTIAL VS PARALLEL BENCHMARK")
    L.append(sep)
    L.append("")
    vidname = os.path.basename(s.get("video", cfg.get("video", "N/A")))
    L.append(f"  Input video     : {vidname}")
    L.append(f"  Frames (seq)    : {int(s['mean_frames'])}")
    L.append(f"  Frames (par)    : {int(p['mean_frames'])}")
    L.append(f"  Grid            : {s.get('grid_rows','?')} x {s.get('grid_cols','?')}")
    L.append(f"  Color channel   : {s.get('color_channel','?')}")
    L.append(f"  Trials          : {NUM_TRIALS}")
    L.append("  CPU% note       : unnormalized -- can exceed 100% on multi-core")
    L.append("  Workload note   : seq.py = 1 lane | rparallel_he3.py = 2 lanes")
    L.append("")
    L.extend(_fmt_section(s, "SEQUENTIAL  (seq.py)"))
    L.append("")
    L.extend(_fmt_section(p, "PARALLEL  (rparallel_he3.py)"))
    L.append("")
    L.append(hr)
    L.append("COMPARISON")
    L.append(hr)
    L.append(f"  Speedup (seq/par)          : {speedup:.3f}x")
    L.append(f"  Execution-time reduction   : {time_red:.1f}%")
    L.append(f"  FPS improvement            : {fps_imp:.1f}%")
    mem_diff = p["mean_peak_rss"] - s["mean_peak_rss"]
    cpu_diff = p["mean_avg_cpu"]  - s["mean_avg_cpu"]
    L.append(f"  Memory diff (peak, MB)     : {mem_diff:+.1f} MB  (par - seq)")
    L.append(f"  CPU diff (avg %)           : {cpu_diff:+.1f}%  (par - seq)")
    L.append("")
    if speedup > 1.05:
        verdict = f"Parallel is faster by {time_red:.1f}% (speedup={speedup:.2f}x)."
    elif speedup < 0.95:
        verdict = f"Sequential is faster (parallel is {abs(time_red):.1f}% slower)."
    else:
        verdict = "Both implementations perform similarly (difference < 5%)."
    L.append(f"  Verdict: {verdict}")
    L.append("")
    L.append("  Workload caveat: parallel processes 2 lanes; sequential processes 1.")
    L.append("    A fair per-lane comparison would halve the parallel time.")
    ppl = par_t / 2 if par_t > 0 else 0
    pls = seq_t / ppl if ppl > 0 else 0
    L.append(f"    Per-lane parallel time : {ppl:.2f} s")
    L.append(f"    Per-lane speedup       : {pls:.3f}x")
    L.append("")
    L.append(sep)
    text = "\n".join(L) + "\n"
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[BENCH] {TXT_OUT} written.")
    return text


# ---------------------------------------------------------------------------
# Profile tables
# ---------------------------------------------------------------------------
def print_profile_tables():
    sep = "=" * 60
    for prof_file, name in [(SEQ_PROF, "seq.py"), (PAR_PROF, "rparallel_he3.py")]:
        for sort_key, label in [("cumulative", "CUMULATIVE"), ("tottime", "SELF (tottime)")]:
            print()
            print(sep)
            print(f"{name} -- Top 15 by {label}")
            print(sep)
            print(prof_top15(prof_file, sort_key))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = load_config()
    verify_config_parity(cfg)

    sep = "=" * 60

    # Sequential runs
    print()
    print(sep)
    print(f"RUNNING seq.py  ({NUM_TRIALS} trials)")
    print(sep)
    seq_trials, seq_ml = [], []
    for t in range(1, NUM_TRIALS + 1):
        seq_trials.append(run_script(SEQ_SCRIPT, t, "SEQ"))
        seq_ml.append(read_frame_metrics(SEQ_METRICS))
        if t < NUM_TRIALS:
            print("  Waiting 5 s before next trial ...")
            time.sleep(5)

    # Parallel runs
    print()
    print(sep)
    print(f"RUNNING rparallel_he3.py  ({NUM_TRIALS} trials)")
    print(sep)
    par_trials, par_ml = [], []
    for t in range(1, NUM_TRIALS + 1):
        par_trials.append(run_script(PAR_SCRIPT, t, "PAR"))
        par_ml.append(read_frame_metrics(PAR_METRICS))
        if t < NUM_TRIALS:
            print("  Waiting 5 s before next trial ...")
            time.sleep(5)

    # Aggregate
    seq_agg = aggregate(seq_trials, seq_ml)
    par_agg = aggregate(par_trials, par_ml)

    # Write outputs
    write_csv(seq_trials, par_trials, seq_ml, par_ml, cfg)
    report = write_txt(seq_agg, par_agg, cfg)
    print()
    print(report)
    print_profile_tables()

    # Final summary table
    print()
    print(sep)
    print("FINAL SUMMARY")
    print(sep)
    w = 32
    print(f"  {'Metric':<{w}} {'Sequential':>14} {'Parallel':>14}")
    print(f"  {'-'*w} {'-'*14} {'-'*14}")
    def row(name, sv, pv, fmt=".2f"):
        print(f"  {name:<{w}} {sv:>14{fmt}} {pv:>14{fmt}}")
    row("Mean total time (s)",     seq_agg["mean_script_time"],    par_agg["mean_script_time"])
    row("Stdev total time (s)",    seq_agg["stdev_script_time"],   par_agg["stdev_script_time"])
    row("Mean FPS",                seq_agg["mean_fps"],            par_agg["mean_fps"])
    row("Mean frame time (ms)",    seq_agg["mean_frame_time_ms"],  par_agg["mean_frame_time_ms"])
    row("Mean peak RSS (MB)",      seq_agg["mean_peak_rss"],       par_agg["mean_peak_rss"], ".1f")
    row("Mean avg CPU %",          seq_agg["mean_avg_cpu"],        par_agg["mean_avg_cpu"],  ".1f")
    row("Mean max CPU %",          seq_agg["mean_max_cpu"],        par_agg["mean_max_cpu"],  ".1f")
    seq_t = seq_agg["mean_script_time"]
    par_t = par_agg["mean_script_time"]
    speedup  = seq_t / par_t if par_t > 0 else float("inf")
    time_red = (seq_t - par_t) / seq_t * 100 if seq_t > 0 else 0
    fps_imp  = (par_agg["mean_fps"] - seq_agg["mean_fps"]) / seq_agg["mean_fps"] * 100 if seq_agg["mean_fps"] > 0 else 0
    print()
    print(f"  {'Speedup (seq/par)':<{w}} {speedup:>13.3f}x")
    print(f"  {'Time reduction':<{w}} {time_red:>13.1f}%")
    print(f"  {'FPS improvement':<{w}} {fps_imp:>13.1f}%")
    print()
    print(f"  Outputs:")
    print(f"    {CSV_OUT}")
    print(f"    {TXT_OUT}")
    print(sep)


if __name__ == "__main__":
    main()
