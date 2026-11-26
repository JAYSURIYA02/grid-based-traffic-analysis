"""
timing_trials.py -- dedicated, repeated timing runs (serial, one process at a
time, no track logging, no evaluation) on the Linux VM. Grid order is rotated
per trial to spread thermal/background drift. Results are VM numbers and must
not be compared with the Windows benchmark-machine numbers in the paper.

    python grid_resolution_analysis/timing_trials.py run --budget 150   (resumable)
    python grid_resolution_analysis/timing_trials.py summary
"""
import json, os, subprocess, sys, time, csv, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
GRIDS = ["4x6", "6x6", "6x8", "8x8", "8x10", "10x12"]
VIDEOS = [("Sunny", "MVI_20011"), ("Night", "MVI_40774"), ("Rainy", "MVI_63525")]
TRIALS = 3
OUT = os.path.join(HERE, "timing_trials")


def plan():
    for t in range(TRIALS):
        order = GRIDS[t * 2 % 6:] + GRIDS[:t * 2 % 6]
        for g in order:
            for cat, v in VIDEOS:
                yield t + 1, g, cat, v


def run(budget):
    t0 = time.time()
    for t, g, cat, v in plan():
        wd = os.path.join(OUT, f"trial{t}", g, v)
        if os.path.exists(os.path.join(wd, "rpar_frame_metrics.json")):
            continue
        if time.time() - t0 > budget:
            print("budget; resume later"); return
        os.makedirs(wd, exist_ok=True); r, c = map(int, g.split("x"))
        json.dump({"video": os.path.join(REPO, "DETRAC_VIDEOS_eval", cat, v + ".mp4"), "color_channel": "V",
                   "grids": {"rows": r, "cols": c}, "xml_path": "", "pixels_per_meter": 0},
                  open(os.path.join(wd, "user_input_data.json"), "w"))
        with open(os.path.join(wd, "run.log"), "w") as lf:
            p = subprocess.run([sys.executable, "-u", os.path.join(HERE, "experiment.py"), "worker"], cwd=wd,
                               stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ, GRID_NOLOG="1"))
        print(t, g, v, "rc", p.returncode, flush=True)
    print("all trials done")


def summary():
    rows = []
    for t, g, cat, v in plan():
        f = os.path.join(OUT, f"trial{t}", g, v, "rpar_frame_metrics.json")
        if not os.path.exists(f):
            continue
        m = json.load(open(f))
        rows.append({"trial": t, "grid": g, "video": v, "frames": m["frames"], "time_s": round(m["total_time_seconds"], 3),
                     "fps": round(m["frames"] / m["total_time_seconds"], 2), "peak_mem_mb": round(m["peak_memory_mb"], 1),
                     "mean_frame_ms": round(1000 * st.mean(m["frame_times_seconds"]), 3)})
    last = f
    with open(os.path.join(HERE, "timing_trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    by = {}
    for g in GRIDS:
        rs = [r for r in rows if r["grid"] == g]
        if not rs:
            continue
        tot = {}
        for r in rs:
            a = tot.setdefault(r["trial"], [0, 0]); a[0] += r["frames"]; a[1] += r["time_s"]
        fps = [fr / tm for fr, tm in tot.values()]
        by[g] = {"n_runs": len(rs), "trials_complete": sum(1 for t in tot if sum(1 for r in rs if r["trial"] == t) == len(VIDEOS)),
                 "fps_median": round(st.median(fps), 2), "fps_min": round(min(fps), 2), "fps_max": round(max(fps), 2),
                 "peak_mem_mb_median": round(st.median(r["peak_mem_mb"] for r in rs), 1),
                 "mean_frame_ms_median": round(st.median(r["mean_frame_ms"] for r in rs), 3)}
    sysinfo = json.load(open(last))["system"] if rows else {}
    json.dump({"machine": "Linux VM (device bridge), 2 vCPU; serial executor; not comparable with Windows numbers",
               "system": sysinfo, "videos": [v for _, v in VIDEOS], "trials": TRIALS, "by_grid": by},
              open(os.path.join(HERE, "timing_trials_summary.json"), "w"), indent=1)
    print(json.dumps(by, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 1e9)
    else:
        summary()
