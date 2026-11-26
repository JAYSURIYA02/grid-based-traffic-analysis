"""
analysis.py -- metrics, diagnostics, plots and threshold tuning for the
grid-resolution experiment. Called by `experiment.py analyze`.
All classification metrics come from the replayed original evaluation
(replay.py, verified against evaluate_lane). Pooled = confusion matrices
summed over videos; macro-F1 = unweighted mean of Car/Van/Bus F1.
"""
import csv, itertools, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from replay import Run, FastRun, ORIGINAL, CLASSES, in_roi

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
GRIDS = ["4x6", "6x6", "6x8", "8x8", "8x10", "10x12"]
BASELINE = "6x6"
EPS = 1e-6
SCOPES = {"Lane1": ("Lane1",), "Lane2": ("Lane2",), "Combined": ("Lane1", "Lane2")}
plt.rcParams.update({"font.size": 8, "font.family": "serif"})


def prf(cm):
    out = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        p = tp / (tp + fp + EPS); r = tp / (tp + fn + EPS); f = 2 * p * r / (p + r + EPS)
        out[c] = (p, r, f)
    out["macro_f1"] = float(np.mean([out[c][2] for c in CLASSES]))
    tot = cm.sum(); out["accuracy"] = float(np.trace(cm) / tot) if tot else 0.0
    out["car_to_van"] = float(cm[0, 1] / cm[0].sum()) if cm[0].sum() else 0.0
    return out


def load(set_name, grids):
    runs = {}
    for g in grids:
        d = os.path.join(RUNS, set_name, g)
        if not os.path.isdir(d):
            continue
        for v in sorted(os.listdir(d)):
            wd = os.path.join(d, v)
            if os.path.exists(os.path.join(wd, "track_log.json.gz")):
                runs.setdefault(g, {})[v] = Run(wd)
    return runs


def counts(run, lanes):
    """Detection counts with the combined-lane GT de-duplication of Run.evaluate."""
    return run.evaluate(ORIGINAL, lanes=lanes, combine=True)["+".join(lanes)]


def diagnostics(run):
    """Merge / fragmentation / box-fit / feature statistics against GT at t+1."""
    r = run; merges = boxes = frags = gts = 0; ious = []; wr = []; hr = []
    feats = {c: {"nrow": [], "ncol": []} for c in CLASSES}
    for ln, L in r.meta["lanes"].items():
        roi = L["roi"]; m = (r.lane == ln) & (r.supp == 0)
        frames = np.unique(r.frame[m])
        for fr in frames:
            idx = np.flatnonzero(m & (r.frame == fr))
            g = [o for o in r.gt.get(int(fr) + 1, []) if in_roi(o, roi)]
            gc = [(o["bbox"][0] + o["bbox"][2] / 2, o["bbox"][1] + o["bbox"][3] / 2) for o in g]
            gc_in = [(x, y) for x, y in gc if roi[0] <= x <= roi[0] + roi[2] and roi[1] <= y <= roi[1] + roi[3]]
            pcs = []
            for i in idx:
                bx, by, bw, bh = r.bbox[i]; boxes += 1
                if sum(bx <= x <= bx + bw and by <= y <= by + bh for x, y in gc_in) >= 2:
                    merges += 1
                pcs.append((bx + bw / 2, by + bh / 2))
                best, bo = 0, None
                for o in g:
                    gx, gy, gw, gh = o["bbox"]
                    ix = max(0, min(bx + bw, gx + gw) - max(bx, gx)); iy = max(0, min(by + bh, gy + gh) - max(by, gy))
                    u = bw * bh + gw * gh - ix * iy; iou = ix * iy / u if u > 0 else 0
                    if iou > best:
                        best, bo = iou, o
                if best > 0.1:
                    ious.append(best); wr.append(bw / bo["bbox"][2]); hr.append(bh / bo["bbox"][3])
                    if bo["label"] in CLASSES and r.age[i] >= 5:
                        feats[bo["label"]]["nrow"].append(r.nrow[i]); feats[bo["label"]]["ncol"].append(r.ncol[i])
            for o, (x, y) in zip(g, gc):
                if not (roi[0] <= x <= roi[0] + roi[2] and roi[1] <= y <= roi[1] + roi[3]):
                    continue
                gts += 1; gx, gy, gw, gh = o["bbox"]
                if sum(gx <= px <= gx + gw and gy <= py <= gy + gh for px, py in pcs) >= 2:
                    frags += 1
    return {"boxes": boxes, "merges": merges, "gt_obs": gts, "frags": frags,
            "ious": ious, "w_ratio": wr, "h_ratio": hr, "feats": feats}


def auc(pos, neg):
    """P(score_pos > score_neg) with ties counted 1/2 (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    a = np.array(pos); b = np.array(neg)
    allv = np.concatenate([a, b]); ranks = allv.argsort().argsort().astype(float) + 1
    # tie correction
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks); ranks = (sums / cnt)[inv]
    return float((ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def perf(run):
    m = json.load(open(os.path.join(run.wd, "rpar_frame_metrics.json")))
    return m["total_time_seconds"], m["frames"], m.get("peak_memory_mb"), float(np.mean(m["frame_times_seconds"]) * 1000)


def bootstrap_diff(per_video_cms_a, per_video_cms_b, n=2000, seed=0):
    """Paired bootstrap over videos of pooled macro-F1(b) - macro-F1(a)."""
    rng = np.random.default_rng(seed); k = len(per_video_cms_a); d = []
    A = np.array(per_video_cms_a); B = np.array(per_video_cms_b)
    for _ in range(n):
        s = rng.integers(0, k, k)
        d.append(prf(B[s].sum(0))["macro_f1"] - prf(A[s].sum(0))["macro_f1"])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def analyze():
    out_cm = os.path.join(HERE, "confusion_matrices"); os.makedirs(out_cm, exist_ok=True)
    bench = load("benchmark", GRIDS)
    status = {g: len(v) for g, v in bench.items()}
    # 1) verification of the offline replay
    verify = {g: {v: r.verify() for v, r in vs.items()} for g, vs in bench.items()}
    fast_ok = {}
    for g, vs in bench.items():
        for v, r in vs.items():
            fr = FastRun(r)
            fast_ok[(g, v)] = all(np.array_equal(fr.cm(ORIGINAL, l), r.evaluate(ORIGINAL, lanes=l, combine=True)["+".join(l)]["cm"])
                                  for l in SCOPES.values())
    ver_summary = {"runs": sum(len(v) for v in verify.values()),
                   "labels_reproduced": sum(a for vs in verify.values() for a, _ in vs.values()),
                   "evaluation_reproduced": sum(b for vs in verify.values() for _, b in vs.values()),
                   "fast_replay_equal": sum(fast_ok.values())}
    print("verification:", ver_summary)

    rows, summary, per_video = [], {}, {}
    videos = sorted(set.intersection(*[set(vs) for vs in bench.values()]))
    for g in GRIDS:
        if g not in bench:
            continue
        summary[g] = {}
        for sc, lanes in SCOPES.items():
            cms = []; agg = dict(unique_gt=0, matched_gt=0, missed_gt=0, pred_tracks=0, matched_tracks=0, unmatched_tracks=0)
            for v in videos:
                c = counts(bench[g][v], lanes); cms.append(c["cm"])
                for k in agg:
                    agg[k] += c[k]
            cm = np.sum(cms, 0); met = prf(cm)
            pv = [prf(x)["macro_f1"] for x in cms]
            per_video.setdefault(g, {})[sc] = cms
            T = [perf(bench[g][v]) for v in videos]
            tot = sum(t[0] for t in T); frames = sum(t[1] for t in T)
            row = {"grid": g, "rows": int(g.split("x")[0]), "cols": int(g.split("x")[1]), "scope": sc,
                   **{f"{c.lower()}_{k}": round(met[c][j], 4) for c in CLASSES for j, k in enumerate(("precision", "recall", "f1"))},
                   "macro_f1": round(met["macro_f1"], 4),
                   "all_car_baseline_macro_f1": round(prf(np.array([[cm[0].sum(), 0, 0], [cm[1].sum(), 0, 0], [cm[2].sum(), 0, 0]]))["macro_f1"], 4),
                   "macro_f1_per_video_mean": round(float(np.mean(pv)), 4),
                   "macro_f1_per_video_std": round(float(np.std(pv, ddof=1)), 4),
                   "accuracy_matched": round(met["accuracy"], 4), "car_to_van_rate": round(met["car_to_van"], 4),
                   "correct": int(np.trace(cm)), "misclassified": int(cm.sum() - np.trace(cm)),
                   "gt_vehicles": agg["unique_gt"], "detected_matched_gt": agg["matched_gt"], "missed_gt": agg["missed_gt"],
                   "pred_tracks": agg["pred_tracks"], "matched_tracks": agg["matched_tracks"], "unmatched_detections": agg["unmatched_tracks"],
                   "detection_recall": round(agg["matched_gt"] / agg["unique_gt"], 4) if agg["unique_gt"] else 0,
                   "total_time_s": round(tot, 2), "fps": round(frames / tot, 2),
                   "mean_loop_ms": round(float(np.mean([t[3] for t in T])), 3),
                   "peak_memory_mb": round(float(np.max([t[2] for t in T])), 1),
                   "cm": cm.tolist()}
            rows.append(row); summary[g][sc] = row
            np.savetxt(os.path.join(out_cm, f"cm_{g}_{sc}.csv"), cm, fmt="%d", delimiter=",", header="rows=GT Car,Van,Bus; cols=Pred Car,Van,Bus")
        fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.4))
        for ax, sc in zip(axs, SCOPES):
            cm = np.array(summary[g][sc]["cm"]); ax.imshow(cm, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
            ax.set_title(f"{g} {sc} (macro-F1 {summary[g][sc]['macro_f1']:.3f})"); ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
        fig.tight_layout(); fig.savefig(os.path.join(out_cm, f"cm_{g}.png"), dpi=200); plt.close(fig)

    # bootstrap CIs vs baseline (combined)
    for g in summary:
        lo, hi = bootstrap_diff(per_video[BASELINE]["Combined"], per_video[g]["Combined"])
        summary[g]["Combined"]["delta_macro_f1_vs_baseline_CI95"] = [round(lo, 4), round(hi, 4)]
        allcar = [np.array([[c[0].sum(), 0, 0], [c[1].sum(), 0, 0], [c[2].sum(), 0, 0]]) for c in per_video[g]["Combined"]]
        lo, hi = bootstrap_diff(allcar, per_video[g]["Combined"])
        summary[g]["Combined"]["delta_macro_f1_vs_all_car_CI95"] = [round(lo, 4), round(hi, 4)]
        if "8x8" in per_video:
            lo, hi = bootstrap_diff(per_video["8x8"]["Combined"], per_video[g]["Combined"])
            summary[g]["Combined"]["delta_macro_f1_vs_8x8_CI95"] = [round(lo, 4), round(hi, 4)]
    # per-video wins vs baseline
    for g in summary:
        a = [prf(x)["macro_f1"] for x in per_video[BASELINE]["Combined"]]; b = [prf(x)["macro_f1"] for x in per_video[g]["Combined"]]
        summary[g]["Combined"]["videos_better_same_worse_vs_baseline"] = [int(sum(y > x + 1e-9 for x, y in zip(a, b))), int(sum(abs(y - x) <= 1e-9 for x, y in zip(a, b))), int(sum(y < x - 1e-9 for x, y in zip(a, b)))]

    # diagnostics
    diag = {}
    for g in summary:
        D = [diagnostics(bench[g][v]) for v in videos]
        feats = {c: {k: sum((d["feats"][c][k] for d in D), []) for k in ("nrow", "ncol")} for c in CLASSES}
        ious = sum((d["ious"] for d in D), []); wr = sum((d["w_ratio"] for d in D), []); hr = sum((d["h_ratio"] for d in D), [])
        diag[g] = {"merge_rate": round(sum(d["merges"] for d in D) / max(1, sum(d["boxes"] for d in D)), 4),
                   "fragmentation_rate": round(sum(d["frags"] for d in D) / max(1, sum(d["gt_obs"] for d in D)), 4),
                   "pred_boxes": sum(d["boxes"] for d in D), "gt_observations": sum(d["gt_obs"] for d in D),
                   "mean_iou_matched": round(float(np.mean(ious)), 4), "median_w_ratio": round(float(np.median(wr)), 3),
                   "median_h_ratio": round(float(np.median(hr)), 3),
                   "feature_means": {c: {k: [round(float(np.mean(feats[c][k])), 3) if feats[c][k] else None,
                                             round(float(np.std(feats[c][k])), 3) if feats[c][k] else None, len(feats[c][k])]
                                         for k in ("nrow", "ncol")} for c in CLASSES},
                   "auc_van_vs_car_nrow": round(auc(feats["Van"]["nrow"], feats["Car"]["nrow"]), 3),
                   "auc_bus_vs_rest_ncol": round(auc(feats["Bus"]["ncol"], feats["Car"]["ncol"] + feats["Van"]["ncol"]), 3),
                   "car_frac_nrow_gt_0.45": round(float(np.mean(np.array(feats["Car"]["nrow"]) > 0.45)), 3) if feats["Car"]["nrow"] else None}
        print("diag", g, {k: v for k, v in diag[g].items() if k != "feature_means"})

    # results.csv
    keys = [k for k in rows[0] if k != "cm"]
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["merge_rate", "fragmentation_rate", "mean_iou_matched"])
        w.writeheader()
        for r in rows:
            d = {k: r[k] for k in keys}
            if r["scope"] == "Combined":
                d.update({k: diag[r["grid"]][k] for k in ("merge_rate", "fragmentation_rate", "mean_iou_matched")})
            w.writerow(d)

    # plots
    gs = [g for g in GRIDS if g in summary]; x = np.arange(len(gs))
    fig, ax = plt.subplots(figsize=(6, 2.6))
    for k, c in enumerate(CLASSES):
        ax.bar(x + (k - 1) * 0.26, [summary[g]["Combined"][f"{c.lower()}_f1"] for g in gs], 0.26, label=c)
    ax.set_xticks(x); ax.set_xticklabels([g.replace("x", "×") for g in gs]); ax.set_xlabel("grid (rows × cols)")
    ax.set_ylabel("F1 (pooled, both lanes)"); ax.legend(frameon=False, ncol=3); ax.grid(axis="y", alpha=0.4)
    ax.set_title("Per-class F1 by grid resolution (12 benchmark videos)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "classification_comparison.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 2.6))
    for sc, mk in (("Combined", "o-"), ("Lane1", "s--"), ("Lane2", "^--")):
        ax.plot(x, [summary[g][sc]["macro_f1"] for g in gs], mk, label=f"{sc} (pooled)")
    base = summary[BASELINE]["Combined"]["macro_f1"]
    lo = [base + summary[g]["Combined"]["delta_macro_f1_vs_baseline_CI95"][0] for g in gs]
    hi = [base + summary[g]["Combined"]["delta_macro_f1_vs_baseline_CI95"][1] for g in gs]
    ax.fill_between(x, lo, hi, alpha=0.15, label="95% CI of change vs 6×6 (combined)")
    ax.axhline(summary[BASELINE]["Combined"]["all_car_baseline_macro_f1"], color="k", ls=":", lw=1, label="trivial 'always Car' (combined)")
    ax.set_xticks(x); ax.set_xticklabels([g.replace("x", "×") for g in gs]); ax.set_xlabel("grid (rows × cols)")
    ax.set_ylabel("macro-F1"); ax.legend(frameon=False, fontsize=6.5, loc="lower right"); ax.grid(alpha=0.4)
    ax.set_title("Macro-F1 by grid resolution")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "macro_f1_comparison.png"), dpi=200); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.4))
    axs[0].bar(x, [summary[g]["Combined"]["total_time_s"] for g in gs], color="#4C72B0"); axs[0].set_ylabel("total time, 12 videos (s)")
    axs[1].bar(x, [summary[g]["Combined"]["fps"] for g in gs], color="#55A868"); axs[1].set_ylabel("FPS (end-to-end)")
    axs[2].bar(x, [summary[g]["Combined"]["peak_memory_mb"] for g in gs], color="#DD8452"); axs[2].set_ylabel("peak memory (MB)")
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels([g.replace("x", "×") for g in gs], rotation=45); ax.grid(axis="y", alpha=0.4)
    fig.suptitle("Unthreaded two-lane workload, 2-vCPU Linux VM (relative comparison only)", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "performance_comparison.png"), dpi=200); plt.close(fig)

    tuning = {}
    for g in GRIDS:
        p = os.path.join(HERE, f"tuning_{g}.json")
        if os.path.exists(p):
            tuning[g] = json.load(open(p))
    for g in list(tuning):
        # paired bootstrap over the 12 test videos: tuned vs original, tuned vs trivial all-Car
        th = tuning[g]["A_spans"]["thresholds"]; lanes = ("Lane1", "Lane2")
        frs = [FastRun(bench[g][v]) for v in videos]
        cm_t = [f.cm(th, lanes) for f in frs]; cm_o = [f.cm(ORIGINAL, lanes) for f in frs]
        cm_c = [np.array([[c[0].sum(), 0, 0], [c[1].sum(), 0, 0], [c[2].sum(), 0, 0]]) for c in cm_o]
        tuning[g]["test_CI95_tuned_minus_original"] = [round(x, 4) for x in bootstrap_diff(cm_o, cm_t)]
        tuning[g]["test_CI95_tuned_minus_all_car"] = [round(x, 4) for x in bootstrap_diff(cm_c, cm_t)]
        tuning[g]["test_all_car_macro_f1"] = round(prf(np.sum(cm_c, 0))["macro_f1"], 4)
    if tuning:
        tg = [g for g in GRIDS if g in tuning]; xt = np.arange(len(tg))
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.bar(xt - 0.2, [tuning[g]["original"]["test_macro_f1"] for g in tg], 0.4, label="original thresholds")
        ax.bar(xt + 0.2, [tuning[g]["A_spans"]["test_macro_f1"] for g in tg], 0.4, label="thresholds tuned on 26 validation videos")
        ax.plot(xt, [summary[g]["Combined"]["all_car_baseline_macro_f1"] for g in tg], "k_", ms=30, mew=2, label="trivial 'always Car'")
        ax.set_xticks(xt); ax.set_xticklabels([g.replace("x", "×") for g in tg]); ax.set_ylabel("macro-F1 on 12 test videos")
        ax.set_ylim(0, 0.5); ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=1); ax.grid(axis="y", alpha=0.4); ax.set_title("Threshold tuning: held-out test performance")
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "threshold_tuning_comparison.png"), dpi=200); plt.close(fig)
    json.dump({"verification": ver_summary, "runs_per_grid": status, "videos": videos,
               "summary": {g: {sc: {k: v for k, v in d.items()} for sc, d in s.items()} for g, s in summary.items()},
               "diagnostics": diag, "tuning": tuning}, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
    print("analysis written")


def tune(g):
    """Threshold tuning for grid g: search on the validation videos only, then
    evaluate the selected thresholds once on the 12 held-out benchmark videos."""
    bench = load("benchmark", [g])[g]; val = load("validation", [g])[g]
    fv = [FastRun(r) for r in val.values()]
    ft = [FastRun(bench[v]) for v in sorted(bench)]
    lanes = ("Lane1", "Lane2")

    def score(runs, th, ln=lanes):
        cm = sum((f.cm(th, ln) for f in runs), np.zeros((3, 3), int)); return prf(cm), cm
    grid_A = [dict(bus_col=a, bus_both=b, van_row=c, van_max_aspect=None)
              for a in (0.6, 0.7, 0.75, 0.8, 0.9, 1.01) for b in (0.45, 0.55, 0.65, 0.75, 1.01)
              for c in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01)]
    grid_B = [dict(bus_col=a, bus_both=b, van_row=c, van_max_aspect=d)
              for a in (0.7, 0.75, 0.9, 1.01) for b in (0.55, 0.75, 1.01)
              for c in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8) for d in (1.0, 1.5, 2.0, 2.5, 3.0)]
    res = {"n_validation_videos": len(fv), "n_test_videos": len(ft), "objective": "pooled macro-F1, both lanes, validation videos"}

    def entry(th, vsel=None):
        vm, vcm = score(fv, th); tm, tcm = score(ft, th)
        lane_test = {ln: round(score(ft, th, (ln,))[0]["macro_f1"], 4) for ln in lanes}
        return {"thresholds": th, "validation_macro_f1": round(vm["macro_f1"], 4),
                "validation_per_class_f1": {c: round(vm[c][2], 4) for c in CLASSES},
                "test_macro_f1": round(tm["macro_f1"], 4), "test_per_class_f1": {c: round(tm[c][2], 4) for c in CLASSES},
                "test_accuracy": round(tm["accuracy"], 4), "test_car_to_van": round(tm["car_to_van"], 4),
                "test_macro_f1_by_lane": lane_test, "test_cm": tcm.tolist(), "validation_cm": vcm.tolist()}
    res["original"] = entry(ORIGINAL)
    for fam, cands in (("A_spans", grid_A), ("B_spans_plus_aspect", grid_B)):
        scored = [(score(fv, th)[0]["macro_f1"], th) for th in cands]
        scored.sort(key=lambda t: (-t[0], abs(t[1]["van_row"] - 0.45)))
        res[fam] = entry(scored[0][1])
        res[fam]["n_candidates"] = len(cands)
        res[fam]["validation_top5"] = [[round(s_, 4), th] for s_, th in scored[:5]]
    json.dump(res, open(os.path.join(HERE, f"tuning_{g}.json"), "w"), indent=1)
    print(g, {k: (v["validation_macro_f1"], v["test_macro_f1"]) for k, v in res.items() if isinstance(v, dict) and "test_macro_f1" in v})
