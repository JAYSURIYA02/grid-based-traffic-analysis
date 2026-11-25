"""
final_analysis.py -- aggregates for updated_analysis_report.md and the two
figures. Reads only files produced by the other scripts:
cross_validation_oof_summary.csv, selected_configuration.json,
frozen_thresholds_by_grid.json, counting_results.csv, timing_trials_summary.json,
fragmentation_validation.csv and the cached per-video statistics (cv_cache/).

    python grid_resolution_analysis/final_analysis.py
"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from cross_validation import video_stats, pooled, CANDS, ORIGINAL
from analysis import prf
GRIDS6 = ["4x6", "6x6", "6x8", "8x8", "8x10", "10x12"]; GRIDS4 = ["6x6", "6x8", "8x8", "10x12"]
COL = ["#257de0", "#f47230", "#00b894", "#f2a600"]; INK = "#222222"; INK2 = "#666666"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
FROZEN = json.load(open(os.path.join(HERE, "frozen_thresholds_by_grid.json")))
SEL = json.load(open(os.path.join(HERE, "selected_configuration.json")))


def session(v):                     # resampling unit for the test set: recording session (first 4 digits)
    return v[4:8]


def all_car(cm):
    cm = np.array(cm); c = np.zeros((3, 3), int); c[:, 0] = cm.sum(1); return prf(c)["macro_f1"]


def classification(set_name, grids):
    out = {}
    d0 = os.path.join(HERE, "runs", set_name)
    for g in grids:
        vids = sorted(v for v in os.listdir(os.path.join(d0, g)) if os.path.exists(os.path.join(d0, g, v, "track_log.json.gz")))
        cms, st = zip(*[video_stats(set_name, g, v) for v in vids]); cms = np.stack(cms, 1)
        for tag, th in (("original", ORIGINAL), ("frozen", FROZEN[g])):
            k = CANDS.index(th); p = pooled(list(st), cms[k].sum(0))
            p["all_car_macro_f1"] = all_car(cms[k].sum(0)); p["cm"] = cms[k].sum(0).tolist()
            out[(g, tag)] = p
        out[(g, "_per_video")] = (vids, cms, list(st))
    return out


def boot(set_name, data, ga, gb, tag, metric, n=2000, seed=1):
    va, ca, sa = data[(ga, "_per_video")]; vb, cb, sb = data[(gb, "_per_video")]; assert va == vb
    ka, kb = CANDS.index(ORIGINAL if tag == "original" else FROZEN[ga]), CANDS.index(ORIGINAL if tag == "original" else FROZEN[gb])
    units = sorted({session(v) for v in va}); by = {u: [i for i, v in enumerate(va) if session(v) == u] for u in units}
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(n):
        idx = [i for u in rng.choice(units, len(units)) for i in by[u]]
        pa = pooled([sa[i] for i in idx], ca[ka, idx].sum(0)); pb = pooled([sb[i] for i in idx], cb[kb, idx].sum(0))
        diffs.append(pb[metric] - pa[metric])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def counting_summary():
    rows = list(csv.DictReader(open(os.path.join(HERE, "counting_results.csv"))))
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["variant"], r["set"], r["grid"], r["thresholds"], r["gt_def"], r["scope"])
        for f in ("gt", "pred", "tp", "duplicate", "false", "missed", "false_from_opposite_direction", "abs_error",
                  "gt_Car", "gt_Van", "gt_Bus", "pred_Car", "pred_Van", "pred_Bus", "tp_Car_classok", "tp_Van_classok", "tp_Bus_classok"):
            agg[k][f].append(float(r[f]))
        if r["ape"] not in ("", "None"):
            agg[k]["ape"].append(float(r["ape"]))
    out = {}
    for k, a in agg.items():
        G, P, TP = sum(a["gt"]), sum(a["pred"]), sum(a["tp"])
        out[k] = {"videos": len(a["gt"]), "gt": int(G), "pred": int(P), "tp": int(TP), "duplicate": int(sum(a["duplicate"])),
                  "false": int(sum(a["false"])), "false_opp_dir": int(sum(a["false_from_opposite_direction"])),
                  "missed": int(sum(a["missed"])), "MAE": float(np.mean(a["abs_error"])),
                  "MAPE": float(np.mean(a["ape"])) if a["ape"] else None, "n_mape": len(a["ape"]),
                  "pooled_count_accuracy": max(0.0, 1 - abs(P - G) / G) if G else None,
                  "event_precision": TP / P if P else None, "event_recall": TP / G if G else None,
                  "event_f1": 2 * TP / (P + G) if P + G else None,
                  **{f"{x}_{c}": int(sum(a[f"{x}_{c}"])) for x in ("gt", "pred") for c in ("Car", "Van", "Bus")},
                  **{f"tp_classok_{c}": int(sum(a[f"tp_{c}_classok"])) for c in ("Car", "Van", "Bus")}}
    return out


def main():
    C = counting_summary()
    test = classification("test", GRIDS4); bench = classification("benchmark", GRIDS4)
    timing = json.load(open(os.path.join(HERE, "timing_trials_summary.json")))["by_grid"]
    oof = {r["grid"]: r for r in csv.DictReader(open(os.path.join(HERE, "cross_validation_oof_summary.csv")))}
    frag = list(csv.DictReader(open(os.path.join(HERE, "fragmentation_validation.csv"))))
    res = {"counting": {"|".join(k): v for k, v in C.items()},
           "test_classification": {f"{g}|{t}": {k: v for k, v in p.items()} for (g, t), p in test.items() if t != "_per_video"},
           "benchmark_classification": {f"{g}|{t}": p for (g, t), p in bench.items() if t != "_per_video"},
           "bootstrap_test_vs_6x6_session_level": {}}
    for g in ["6x8", "8x8", "10x12"]:
        for tag in ("frozen", "original"):
            for m in ("macro_f1", "count_f1", "J", "det_recall"):
                res["bootstrap_test_vs_6x6_session_level"][f"{g}|{tag}|{m}"] = boot("test", test, "6x6", g, tag, m)
    res["bootstrap_test_8x8_vs_10x12"] = {m: boot("test", test, "8x8", "10x12", "frozen", m) for m in ("macro_f1", "count_f1", "J")}
    res["test_sessions"] = len({session(v) for v in test[("6x6", "_per_video")][0]})
    json.dump(res, open(os.path.join(HERE, "final_numbers.json"), "w"), indent=1, default=float)

    # ---------- figure 1: trade-off ----------
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.6)); x = np.arange(len(GRIDS4))
    def ln(a, ys, i, lab, ls="-"):
        a.plot(x, ys, ls, color=COL[i], lw=2, marker="o", ms=4, label=lab)
    a = ax[0]
    ln(a, [float(oof[g]["macro_f1"]) for g in GRIDS4], 0, "Validation OOF (tuned)")
    ln(a, [test[(g, "frozen")]["macro_f1"] for g in GRIDS4], 1, "Test (frozen thr.)")
    ln(a, [test[(g, "original")]["macro_f1"] for g in GRIDS4], 2, "Test (original thr.)")
    a.axhline(test[("6x6", "frozen")]["all_car_macro_f1"], color=INK2, lw=1, ls="--"); a.text(0, test[("6x6", "frozen")]["all_car_macro_f1"] - 0.03, "always-Car baseline (test)", ha="left", fontsize=6.5, color=INK2)
    a.set_title("(a) Classification macro-F1", fontsize=8); a.set_ylim(0, 0.45)
    a = ax[1]
    ln(a, [C[("original", "validation", g, "original", "down", "Combined")]["event_f1"] for g in GRIDS4], 0, "Validation")
    ln(a, [C[("original", "test", g, "original", "down", "Combined")]["event_f1"] for g in GRIDS4], 1, "Test")
    ln(a, [C[("original", "test", g, "original", "all", "Combined")]["event_f1"] for g in GRIDS4], 2, "Test, both directions", "--")
    a.set_title("(b) Counting event-F1", fontsize=8); a.set_ylim(0, 0.8)
    a = ax[2]
    ln(a, [test[(g, "frozen")]["det_recall"] for g in GRIDS4], 0, "Detection recall")
    ln(a, [test[(g, "frozen")]["unmatched"] for g in GRIDS4], 1, "Unmatched-track rate")
    ln(a, [10 * test[(g, "frozen")]["frag"] for g in GRIDS4], 2, "Fragmentation ×10")
    a.set_title("(c) Detection quality (test)", fontsize=8); a.set_ylim(0, 1.0)
    a = ax[3]
    ln(a, [timing[g]["fps_median"] for g in GRIDS4], 0, "Median FPS (VM)")
    a.axhline(25, color=INK2, lw=1, ls="--"); a.text(0, 27, "25 FPS", fontsize=6.5, color=INK2)
    a.set_title("(d) Throughput, 3 trials", fontsize=8); a.set_ylim(0, 110)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(GRIDS4); a.grid(axis="y", color="#e6e6e6", lw=0.6); a.set_axisbelow(True)
        a.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "grid_tradeoff_comparison.png"), dpi=300); plt.close(fig)

    # ---------- figure 2: counting ----------
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})
    a = ax[0]; w = 0.36
    for j, (setn, off) in enumerate((("validation", -w / 2 - 0.01), ("test", w / 2 + 0.01))):
        bottoms = np.zeros(len(GRIDS4))
        segs = [("tp", "Correct counts", COL[0]), ("false", "False counts", COL[1]), ("duplicate", "Duplicate counts", COL[3])]
        for key, lab, c in segs:
            vals = np.array([C[("original", setn, g, "original", "all", "Combined")][key] for g in GRIDS4], float)
            gts = np.array([C[("original", setn, g, "original", "all", "Combined")]["gt"] for g in GRIDS4], float)
            a.bar(x + off, vals / gts, w, bottom=bottoms, color=c, edgecolor="white", lw=1, label=lab if j == 0 else None)
            bottoms += vals / gts
        for i in range(len(GRIDS4)):
            a.text(x[i] + off, bottoms[i] + 0.02, "V" if setn == "validation" else "T", ha="center", fontsize=6, color=INK2)
    a.axhline(1.0, color=INK, lw=1); a.text(3.55, 1.02, "all GT crossings = 1", fontsize=6.5, color=INK, ha="right")
    gd = C[("original", "test", "6x6", "original", "down", "Combined")]["gt"] / C[("original", "test", "6x6", "original", "all", "Combined")]["gt"]
    a.axhline(gd, color=INK2, lw=1, ls="--"); a.text(3.5, gd + 0.02, "downward share (test)", fontsize=6.5, color=INK2, ha="right")
    a.set_xticks(x); a.set_xticklabels(GRIDS4); a.set_ylabel("Counts / GT crossings")
    a.set_title("(a) Count composition, combined lanes (V = validation, T = test)", fontsize=8); a.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    a = ax[1]
    for i, (lane, ls) in enumerate((("Lane1", "-"), ("Lane2", "-"), ("Combined", "--"))):
        a.plot(x, [C[("original", "test", g, "original", "down", lane)]["MAE"] for g in GRIDS4], ls, color=COL[i], lw=2, marker="o", ms=4, label=lane)
    a.set_xticks(x); a.set_xticklabels(GRIDS4); a.set_ylabel("MAE (vehicles / video)")
    a.set_title("(b) Test count MAE (GT: downward)", fontsize=8); a.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3); a.set_ylim(0, None)
    for a in ax:
        a.grid(axis="y", color="#e6e6e6", lw=0.6); a.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "counting_comparison.png"), dpi=300); plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
