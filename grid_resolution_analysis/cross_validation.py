"""
cross_validation.py -- site-grouped cross-validation for grid selection and
threshold tuning, using ONLY the 26 validation videos.

Protocol (fixed before any fold result was computed):
  * Folds: leave-one-site-group-out over the site groups of the validation
    videos (site_groups_manual.json). 10 groups -> 10 folds.
  * Inside each fold, using the training videos only:
      1. for every grid, choose classification thresholds that maximise the
         pooled combined-lane macro-F1 on the training videos, over the same
         810 candidates used earlier (families A: spans, B: spans + aspect);
         ties -> van_row closest to 0.45 (the original value), then list order;
      2. score every grid on the training videos with the objective below and
         select the best grid (cost guard and tie rule below);
      3. evaluate the selected (grid, thresholds) AND every grid with its own
         training-tuned thresholds on the held-out site group.
  * Out-of-fold (OOF) results are pooled over the 10 held-out groups.
  * The final configuration is obtained by running steps 1-2 once on all 26
    validation videos and is frozen in selected_configuration.json.

Objective J (higher is better), all terms pooled over the videos scored:
    J = 0.30 macroF1 + 0.10 min(F1_Car, F1_Van, F1_Bus)
      + 0.25 countF1  + 0.15 detection recall
      + 0.10 (1 - fragmentation rate) + 0.10 (1 - unmatched-detection rate)
  macroF1/F1_k : vehicle-level, combined lanes (replay of evaluate_lane)
  countF1      : event-level counting F1, combined lanes, GT = downward
                 trajectory crossings (counting_evaluation.py)
  detection recall : matched GT vehicles / GT vehicles (combined lanes)
  fragmentation    : GT observations covered by >= 2 detections / GT observations
  unmatched rate   : predicted tracks matching no GT vehicle / predicted tracks
Cost guard: a grid is eligible only if its median throughput in the dedicated
timing trials (timing_trials.py, same VM, serial) is >= 25 FPS (real time for
UA-DETRAC) and its peak memory <= 1.5x the 6x6 baseline. If timing trials are
missing, the guard is skipped and this is reported. Tie rule: grids within
0.005 of the best J -> the one with fewer cells.

Usage:  python grid_resolution_analysis/cross_validation.py prepare --grids 6x6   (resumable, per grid)
        python grid_resolution_analysis/cross_validation.py cv
"""
import csv, json, os, sys, time, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from replay import Run, FastRun, ORIGINAL, CLASSES
from analysis import prf, diagnostics, counts
import counting_evaluation as ce

GRIDS = ["4x6", "6x6", "6x8", "8x8", "8x10", "10x12"]
LANES = ("Lane1", "Lane2")
W = {"macro_f1": 0.30, "min_class_f1": 0.10, "count_f1": 0.25, "det_recall": 0.15, "frag": 0.10, "unmatched": 0.10}
TIE = 0.005
CACHE = os.path.join(HERE, "cv_cache"); os.makedirs(CACHE, exist_ok=True)

CANDS = ([dict(bus_col=a, bus_both=b, van_row=c, van_max_aspect=None)
          for a in (0.6, 0.7, 0.75, 0.8, 0.9, 1.01) for b in (0.45, 0.55, 0.65, 0.75, 1.01)
          for c in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01)] +
         [dict(bus_col=a, bus_both=b, van_row=c, van_max_aspect=d)
          for a in (0.7, 0.75, 0.9, 1.01) for b in (0.55, 0.75, 1.01)
          for c in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8) for d in (1.0, 1.5, 2.0, 2.5, 3.0)])
TIEKEY = np.array([abs(t["van_row"] - 0.45) for t in CANDS])


def site_of():
    g = json.load(open(os.path.join(HERE, "site_groups_manual.json")))["groups"]
    return {v: k for k, vs in g.items() for v in vs}


def video_stats(set_name, g, v, base=None):
    """Threshold-independent per-video stats + per-candidate confusion matrices (cached)."""
    base = base or os.path.join(HERE, "runs")
    f = os.path.join(CACHE, f"{os.path.basename(base)}_{set_name}_{g}_{v}.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True); return z["cms"], json.loads(str(z["stats"]))
    wd = os.path.join(base, set_name, g, v)
    run = Run(wd); fr = FastRun(run)
    cms = np.stack([fr.cm(th, LANES) for th in CANDS]).astype(np.int32)
    c = counts(run, LANES); d = diagnostics(run)
    cnt = ce.evaluate_run(run)
    comb = {r["gt_def"]: r for r in cnt["rows"] if r["scope"] == "Combined"}
    m = json.load(open(os.path.join(wd, "rpar_frame_metrics.json")))
    stats = {"unique_gt": c["unique_gt"], "matched_gt": c["matched_gt"], "pred_tracks": c["pred_tracks"],
             "unmatched_tracks": c["unmatched_tracks"], "gt_obs": d["gt_obs"], "frags": d["frags"],
             "boxes": d["boxes"], "merges": d["merges"], "count_verified": cnt["verified"],
             "count_down": {k: comb["down"][k] for k in ("gt", "pred", "tp", "duplicate", "false", "missed")},
             "count_all": {k: comb["all"][k] for k in ("gt", "pred", "tp", "duplicate", "false", "missed")},
             "frames": m["frames"], "time_s": m["total_time_seconds"], "peak_mem_mb": m.get("peak_memory_mb")}
    np.savez_compressed(f, cms=cms, stats=json.dumps(stats))
    return cms, stats


def pooled(stats_list, cm):
    s = lambda k: sum(x[k] for x in stats_list)
    cd = lambda k: sum(x["count_down"][k] for x in stats_list)
    ca = lambda k: sum(x["count_all"][k] for x in stats_list)
    p = prf(cm)
    out = {"macro_f1": p["macro_f1"], "f1_Car": p["Car"][2], "f1_Van": p["Van"][2], "f1_Bus": p["Bus"][2],
           "min_class_f1": min(p[c][2] for c in CLASSES), "accuracy": p["accuracy"],
           "det_recall": s("matched_gt") / max(1, s("unique_gt")),
           "frag": s("frags") / max(1, s("gt_obs")), "merge": s("merges") / max(1, s("boxes")),
           "unmatched": s("unmatched_tracks") / max(1, s("pred_tracks")),
           "count_f1": 2 * cd("tp") / max(1, cd("pred") + cd("gt")),
           "count_f1_all": 2 * ca("tp") / max(1, ca("pred") + ca("gt")),
           "count_gt_down": cd("gt"), "count_gt_all": ca("gt"), "count_pred": cd("pred"),
           "count_abs_err_sum": sum(abs(x["count_down"]["pred"] - x["count_down"]["gt"]) for x in stats_list),
           "count_abs_err_all_sum": sum(abs(x["count_all"]["pred"] - x["count_all"]["gt"]) for x in stats_list),
           "n_videos": len(stats_list), "frames": s("frames")}
    out["J"] = (W["macro_f1"] * out["macro_f1"] + W["min_class_f1"] * out["min_class_f1"] + W["count_f1"] * out["count_f1"]
                + W["det_recall"] * out["det_recall"] + W["frag"] * (1 - out["frag"]) + W["unmatched"] * (1 - out["unmatched"]))
    return out


def timing_guard():
    f = os.path.join(HERE, "timing_trials_summary.json")
    if not os.path.exists(f):
        return None
    t = json.load(open(f))["by_grid"]
    base_mem = t["6x6"]["peak_mem_mb_median"]
    return {g: (t[g]["fps_median"] >= 25 and t[g]["peak_mem_mb_median"] <= 1.5 * base_mem) for g in t}


def tune(cms, idx):
    """Best candidate on the pooled training confusion matrices."""
    tot = cms[:, idx].sum(1)                               # (C,3,3)
    f1 = np.array([prf(m)["macro_f1"] for m in tot])
    order = np.lexsort((np.arange(len(CANDS)), TIEKEY, -np.round(f1, 12)))
    return int(order[0]), float(f1[order[0]])


def select(train_idx, data, guard):
    rows = {}
    for g in GRIDS:
        cms, stats = data[g]
        k, _ = tune(cms, train_idx)
        rows[g] = {"cand": k, **pooled([stats[i] for i in train_idx], cms[k, train_idx].sum(0))}
    elig = [g for g in GRIDS if guard is None or guard.get(g, False)]
    best = max(rows[g]["J"] for g in elig)
    near = [g for g in elig if rows[g]["J"] >= best - TIE]
    cells = lambda g: int(g.split("x")[0]) * int(g.split("x")[1])
    return min(near, key=cells), rows


def cv():
    site = site_of()
    vids = sorted(os.listdir(os.path.join(HERE, "runs", "validation", "6x6")))
    data = {}
    for g in GRIDS:
        cl, sl = [], []
        for v in vids:
            c, s = video_stats("validation", g, v); cl.append(c); sl.append(s)
        data[g] = (np.stack(cl, 1), sl)                    # cms: (C, V, 3, 3)
        bad = [v for v, s in zip(vids, sl) if not s["count_verified"]]
        if bad:
            print("WARNING count replay not verified:", g, bad)
    guard = timing_guard()
    groups = sorted({site[v] for v in vids})
    out_rows = []; oof_sel = []; oof = {g: [] for g in GRIDS}; chosen = []
    for gi, grp in enumerate(groups):
        test_idx = [i for i, v in enumerate(vids) if site[v] == grp]
        train_idx = [i for i, v in enumerate(vids) if site[v] != grp]
        sel, rows = select(train_idx, data, guard); chosen.append(sel)
        for g in GRIDS:
            cms, stats = data[g]; k = rows[g]["cand"]
            te = pooled([stats[i] for i in test_idx], cms[k, test_idx].sum(0))
            oof[g].append((test_idx, k))
            if g == sel:
                oof_sel.append((g, test_idx, k))
            out_rows.append({"fold": gi + 1, "held_out_site": grp, "held_out_videos": " ".join(vids[i] for i in test_idx),
                             "n_train": len(train_idx), "n_test": len(test_idx), "grid": g, "selected_in_fold": int(g == sel),
                             "thresholds": json.dumps(CANDS[k]),
                             **{f"train_{m}": round(rows[g][m], 4) for m in ("J", "macro_f1", "min_class_f1", "count_f1", "det_recall", "frag", "unmatched")},
                             **{f"test_{m}": round(te[m], 4) for m in ("J", "macro_f1", "f1_Car", "f1_Van", "f1_Bus", "count_f1", "count_f1_all", "det_recall", "frag", "unmatched")},
                             "test_count_gt_down": te["count_gt_down"], "test_count_pred": te["count_pred"]})
    # OOF aggregates: pool held-out confusion matrices / counts across folds
    agg = []
    def oof_pool(items):
        st, cm = [], np.zeros((3, 3), int)
        for g, idx, k in items:
            cms, stats = data[g]; st += [stats[i] for i in idx]; cm += cms[k, idx].sum(0)
        return pooled(st, cm), cm
    for g in GRIDS:
        p, cm = oof_pool([(g, idx, k) for idx, k in oof[g]])
        orig_k = CANDS.index(ORIGINAL) if ORIGINAL in CANDS else None
        po, _ = oof_pool([(g, list(range(len(vids))), orig_k)]) if orig_k is not None else ({}, None)
        agg.append({"row": "OOF_fixed_grid", "grid": g, **{k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()},
                    "cm": json.dumps(cm.tolist()), "macro_f1_original_thresholds": round(po.get("macro_f1", float("nan")), 4)})
    p, cm = oof_pool(oof_sel)
    agg.append({"row": "OOF_selection_procedure", "grid": "per-fold", **{k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()},
                "cm": json.dumps(cm.tolist()), "macro_f1_original_thresholds": ""})
    # final selection on all 26 validation videos
    sel, rows = select(list(range(len(vids))), data, guard)
    with open(os.path.join(HERE, "cross_validation_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    with open(os.path.join(HERE, "cross_validation_oof_summary.csv"), "w", newline="") as f:
        keys = list(agg[0].keys())
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(agg)
    cfg = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "selected_grid": sel, "rows": int(sel.split("x")[0]), "cols": int(sel.split("x")[1]),
           "thresholds": CANDS[rows[sel]["cand"]], "original_thresholds": ORIGINAL,
           "selection_data": f"{len(vids)} validation videos, {len(groups)} site groups (site_groups_manual.json)",
           "objective_weights": W, "tie_rule": f"within {TIE} of best J -> fewer cells",
           "cost_guard": "applied" if guard is not None else "NOT applied (timing_trials_summary.json missing)",
           "cost_guard_eligibility": guard,
           "fold_choices": dict(zip(groups, chosen)),
           "validation_scores_all_grids": {g: {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[g].items() if k != "cand"} | {"thresholds": CANDS[rows[g]["cand"]]} for g in GRIDS},
           "fixed_pipeline": {"executor": "serial (BTP_EXECUTOR=serial)", "lanes": 2, "color_channel": "V",
                              "roi_lane1": [47, 202, 441, 141], "roi_lane2": [500, 202, 441, 141],
                              "counting_line": "roi_y + roi_h//2 (y=272)", "clustering": "4-connected (production)"},
           "note": "Frozen before any use of the site-independent test set (test_set.json). Do not edit."}
    s = json.dumps(cfg, indent=1, sort_keys=True); cfg["sha256_of_content"] = hashlib.sha256(s.encode()).hexdigest()
    json.dump(cfg, open(os.path.join(HERE, "selected_configuration.json"), "w"), indent=1)
    print("fold choices:", dict(zip(groups, chosen))); print("FINAL:", sel, CANDS[rows[sel]["cand"]])
    for a in agg:
        print(a["row"], a["grid"], "J", a["J"], "mF1", a["macro_f1"], "cF1", a["count_f1"], "rec", a["det_recall"], "frag", a["frag"])


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        set_name = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "validation"
        grids = sys.argv[sys.argv.index("--grids") + 1].split(",") if "--grids" in sys.argv else GRIDS
        budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 1e9
        t0 = time.time()
        for g in grids:
            d = os.path.join(HERE, "runs", set_name, g)
            for v in sorted(os.listdir(d)):
                if time.time() - t0 > budget:
                    print("budget"); sys.exit(0)
                if os.path.exists(os.path.join(d, v, "track_log.json.gz")):
                    video_stats(set_name, g, v)
        print("prepared")
    else:
        cv()
