"""
experiment.py -- grid-resolution sensitivity analysis for the geometry-based
vehicle classifier. Does NOT modify seq.py / rparallel_he3.py.

How it works
------------
`run`    For every (grid, video) it creates runs/<grid>/<video>/ with a
         user_input_data.json and launches `experiment.py worker` there.
`worker` Imports the shipped rparallel_he3.py as a module (module-level setup
         reads the config from the cwd; main() is not auto-run), forces the
         unthreaded two-lane workload (BTP_EXECUTOR=serial, so Lane 1 is
         identical to seq.py), disables video writing, and wraps
         LaneTracker.suppress_duplicate_tracks to log, for every track that
         voted in a frame, its geometric features, vote weight inputs, bbox,
         frame_age and final label. It then runs main() and the original
         evaluate_lane() and saves both.
`analyze` Replays label voting + the original vehicle-level evaluation offline
         from the logs (verified to reproduce evaluate_lane exactly with the
         original thresholds), computes all metrics, diagnostics, plots and
         the threshold-tuning experiment.

Commands (from the repository root)
    python grid_resolution_analysis/experiment.py run --set benchmark --resume
    python grid_resolution_analysis/experiment.py run --set validation --grids 6x6,8x8 --resume
    python grid_resolution_analysis/experiment.py tune --grids 6x6
    python grid_resolution_analysis/experiment.py tune --grids 8x8
    python grid_resolution_analysis/experiment.py tune --grids 10x12
    python grid_resolution_analysis/experiment.py analyze
"""
import argparse, gzip, importlib.util, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RUNS = os.path.join(HERE, "runs")
GRIDS = ["4x6", "6x6", "6x8", "8x8", "8x10", "10x12"]          # rows x cols
BASELINE = "6x6"
DATASET = os.path.join(REPO, "DETRAC_VIDEOS_eval")
ANNOT = os.path.join(REPO, "DETRAC-Annotations")
SELECTION = os.path.join(REPO, "benchmark_results", "selection_report.json")


def video_sets():
    sel = json.load(open(SELECTION))["selection"]
    bench = [(e["category"], os.path.basename(e["video"].replace("\\", "/"))[:-4]) for e in sel]
    bench_ids = {v for _, v in bench}
    allv = []
    for cat in ("Cloudy", "Night", "Rainy", "Sunny"):
        for f in sorted(os.listdir(os.path.join(DATASET, cat))):
            if f.endswith(".mp4"):
                allv.append((cat, f[:-4]))
    val = [(c, v) for c, v in allv if v not in bench_ids and os.path.exists(os.path.join(ANNOT, v + ".xml"))]
    return {"benchmark": bench, "validation": val}


# ───────────────────────────── worker ─────────────────────────────
def worker():
    os.environ.update({"BTP_EXECUTOR": "serial", "BTP_LANES": "2", "BTP_DISPLAY": "0",
                       "BTP_PROFILE": "0", "BTP_SAVE_FRAMES": "0", "MPLBACKEND": "Agg"})
    sys.path.insert(0, REPO)
    spec = importlib.util.spec_from_file_location("rpar_mod", os.path.join(REPO, "rparallel_he3.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)           # module-level setup only (no __main__ block)
    m.OUTPUT_WRITER_OK = False           # do not write the annotated video
    H = int(m.frame_height)
    log = []
    orig_supp = m.LaneTracker.suppress_duplicate_tracks

    def logged_suppress(self, frame_count):
        voters = [(k, v) for k, v in self.vehicles.items() if v.get("last_seen") == frame_count]
        suppressed = orig_supp(self, frame_count)
        for k, v in voters:
            bx, by, bw, bh = v["bbox"]
            log.append([self.name, frame_count, k, v["frame_age"], v["col_span"], v["row_span"],
                        v["cell_count"], bx, by, bw, bh, v["centroid"][1], v["final_label"],
                        1 if k in suppressed else 0])
        return suppressed
    m.LaneTracker.suppress_duplicate_tracks = logged_suppress

    t0 = time.time()
    m.main()
    wall = time.time() - t0
    gt = m.load_detrac(m.xml_path)
    ev = {}
    for lane in m.ACTIVE_LANES:
        _, cmv, met, ids = m.evaluate_lane(lane, gt, show_grid_heatmap=False, debug_grid_metrics=False)
        ev[lane.name] = {"confusion_matrix_vehicle": cmv.tolist(), **ids,
                         "metrics": {c: list(map(float, met[c])) for c in ("Car", "Van", "Bus")}}
    meta = {"rows": m.num_rows, "cols": m.num_cols, "frame_height": H, "frames": m.frame_count,
            "lanes": {l.name: {"roi": [l.roi_x, l.roi_y, l.roi_width, l.roi_height],
                               "grid_w": l.grid_width, "grid_h": l.grid_height} for l in m.ACTIVE_LANES},
            "main_wall_seconds": wall}
    json.dump(ev, open("eval_original.json", "w"), indent=1)
    json.dump(meta, open("meta.json", "w"), indent=1)
    cols = ["lane", "frame", "track", "frame_age", "col_span", "row_span", "cell_count",
            "bx", "by", "bw", "bh", "cy", "final_label", "suppressed"]
    with gzip.open("track_log.json.gz", "wt") as f:
        json.dump({"columns": cols, "rows": log}, f)
    for fn in ("test_output1.mp4",):
        try:
            os.remove(fn)
        except OSError:
            pass
    print("[GRID] worker done", m.num_rows, m.num_cols, m.frame_count, "frames", round(wall, 1), "s")


# ───────────────────────────── runner ─────────────────────────────
def run(args):
    sets = video_sets()
    vids = sets[args.set]
    grids = args.grids.split(",") if args.grids else GRIDS
    failures = []
    for g in grids:
        r, c = map(int, g.split("x"))
        for cat, v in vids:
            wd = os.path.join(RUNS, args.set, g, v)
            if args.resume and os.path.exists(os.path.join(wd, "track_log.json.gz")):
                continue
            os.makedirs(wd, exist_ok=True)
            json.dump({"video": os.path.join(DATASET, cat, v + ".mp4"), "color_channel": "V",
                       "grids": {"rows": r, "cols": c}, "xml_path": os.path.join(ANNOT, v + ".xml"),
                       "pixels_per_meter": 0}, open(os.path.join(wd, "user_input_data.json"), "w"), indent=1)
            t0 = time.time()
            with open(os.path.join(wd, "run.log"), "w") as lf:
                p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "worker"],
                                   cwd=wd, stdout=lf, stderr=subprocess.STDOUT)
            ok = p.returncode == 0 and os.path.exists(os.path.join(wd, "track_log.json.gz"))
            print(f"[{args.set}] {g} {v}: {'OK' if ok else 'FAILED rc=%s' % p.returncode} ({time.time() - t0:.1f}s)", flush=True)
            if not ok:
                failures.append((g, v, p.returncode))
    print("failures:", failures)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "worker", "analyze", "tune"])
    ap.add_argument("--set", default="benchmark", choices=["benchmark", "validation"])
    ap.add_argument("--grids", default="")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.cmd == "worker":
        worker()
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "tune":
        from analysis import tune
        for g in (a.grids.split(",") if a.grids else ["6x6", "8x8", "10x12"]):
            tune(g)
    else:
        from analysis import analyze
        analyze()
