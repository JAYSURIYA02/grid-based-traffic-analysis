"""
revision_analysis.py -- statistics, counting decomposition, classification
tables and the dataset/site mapping for the paper revision.
Reads only existing experiment outputs (grid_resolution_analysis/runs, cv_cache,
counting_results.csv, selected_configuration.json, ...); never modifies them.
Outputs go to paper_revision/updated_tables/.

    python paper_revision/scripts/revision_analysis.py [stats|counting|classification|mapping|objective|all]
"""
import csv, json, os, sys, math, glob, time
from collections import defaultdict, Counter
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRA = os.path.join(REPO, "grid_resolution_analysis"); sys.path.insert(0, GRA)
OUT = os.path.join(REPO, "paper_revision", "updated_tables"); os.makedirs(OUT, exist_ok=True)
from cross_validation import video_stats, pooled, CANDS, ORIGINAL, select, timing_guard, GRIDS, W, TIE
from analysis import prf
import counting_evaluation as ce
from replay import Run, match_vehicle, CLASSES
G4 = ["6x6", "6x8", "8x8", "10x12"]
CACHE = os.path.join(REPO, "paper_revision", "scripts", "cache"); os.makedirs(CACHE, exist_ok=True)
T0 = time.time(); BUDGET = float(os.environ.get("BUDGET", "1e9"))
FROZEN = json.load(open(os.path.join(GRA, "frozen_thresholds_by_grid.json")))
TEST = json.load(open(os.path.join(GRA, "test_set.json")))["videos"]

# ---- test-set site groups (visual inspection of site_grouping_all_sheet.png; conservative merges) ----
TEST_SITES = {
 "T01_park_road": ["MVI_39051"], "T02_bus_terminal_plaza": ["MVI_39271"], "T03_wide_grey_road": ["MVI_39361"],
 "T04_fenced_road_buildings": ["MVI_39371"],
 "T05_diagonal_fenced_road": ["MVI_39781", "MVI_39801", "MVI_39811", "MVI_39821", "MVI_40152", "MVI_40161", "MVI_40162", "MVI_40171", "MVI_40172", "MVI_40181"],
 "T06_diagonal_hedge_road": ["MVI_39931"],
 "T07_turning_junction": ["MVI_40701", "MVI_40711", "MVI_40712", "MVI_40714"],
 "T08_bus_corridor_signboards": ["MVI_40732", "MVI_40742", "MVI_40743", "MVI_40751", "MVI_40752"],
 "T09_night_red_barrier": ["MVI_40761", "MVI_40762", "MVI_40763"],
 "T10_crosswalk_intersection": ["MVI_40851", "MVI_40852", "MVI_40853", "MVI_40854", "MVI_40855", "MVI_40891", "MVI_40892",
                                "MVI_40901", "MVI_40902", "MVI_40903", "MVI_40904", "MVI_40905"],
 "T11_rainy_bus_road": ["MVI_40863", "MVI_40864", "MVI_40871"],
 "T12_overpass_pillar": ["MVI_40991", "MVI_40992"]}
TEST_UNCERTAIN = {
 "T05_diagonal_fenced_road": "merged on visual similarity (diagonal carriageway with white fence); may be 2-3 nearby cameras",
 "T06_diagonal_hedge_road": "hedge strip resembles eval site L (40201/40204) but road geometry differs; kept in test",
 "T07_turning_junction": "40701 is hazy; merged on lane markings",
 "T08_bus_corridor_signboards": "40742/40743 framed tighter than 40732/40751/40752; merged",
 "T10_crosswalk_intersection": "4085x/4089x/4090x merged (same crosswalk layout; automatic clustering also merged 4085x+4089x)",
 "T11_rainy_bus_road": "40871 merged with 40863/40864 on scene similarity",
 "T03_wide_grey_road": "39361 and 39371 recorded in adjacent sessions; kept separate (different framing)",
 "T04_fenced_road_buildings": "see T03"}
SITE_OF_TEST = {v: k for k, vs in TEST_SITES.items() for v in vs}
assert set(SITE_OF_TEST) == set(TEST), set(TEST) ^ set(SITE_OF_TEST)


def per_video(set_name, g, tag="frozen"):
    d = os.path.join(GRA, "runs", set_name, g)
    vids = sorted(v for v in os.listdir(d) if os.path.exists(os.path.join(d, v, "track_log.json.gz")))
    cms, st = zip(*[video_stats(set_name, g, v) for v in vids])
    k = CANDS.index(ORIGINAL if tag == "original" else FROZEN[g])
    return vids, np.stack(cms, 1)[k], list(st)        # cm per video (V,3,3)


def metric(stats, cms, name):
    if name in ("count_mae", "count_rmse"):
        e = np.array([s["count_down"]["pred"] - s["count_down"]["gt"] for s in stats], float)
        return float(np.mean(np.abs(e))) if name == "count_mae" else float(np.sqrt(np.mean(e ** 2)))
    return pooled(stats, cms.sum(0))[name]


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def stats():
    METRICS = ["macro_f1", "J", "count_f1", "det_recall", "count_mae", "count_rmse"]
    data = {g: per_video("test", g) for g in G4}
    vids = data["6x6"][0]
    units = {"site": [SITE_OF_TEST[v] for v in vids], "session": [v[4:8] for v in vids]}
    rows = []
    rng = np.random.default_rng(20261126); B = int(os.environ.get("NBOOT", "10000"))
    for unit, lab in units.items():
        U = sorted(set(lab)); idx = {u: [i for i, x in enumerate(lab) if x == u] for u in U}
        boots = [np.concatenate([idx[u] for u in rng.choice(U, len(U))]).astype(int) for _ in range(B)]
        flips = rng.integers(0, 2, size=(B, len(U)))
        for m in METRICS:
            fam = []
            for g in ["6x8", "8x8", "10x12"]:
                _, ca, sa = data["6x6"]; _, cb, sb = data[g]
                obs = metric(sb, cb, m) - metric(sa, ca, m)
                d = np.array([metric([sb[i] for i in b], cb[b], m) - metric([sa[i] for i in b], ca[b], m) for b in boots])
                p_boot = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
                # paired permutation: swap the two grids' results within each resampling unit
                perm = []
                for f in flips:
                    swap = set(np.concatenate([idx[u] for u, s in zip(U, f) if s]).astype(int)) if f.any() else set()
                    A_s = [sb[i] if i in swap else sa[i] for i in range(len(vids))]; A_c = np.stack([cb[i] if i in swap else ca[i] for i in range(len(vids))])
                    B_s = [sa[i] if i in swap else sb[i] for i in range(len(vids))]; B_c = np.stack([ca[i] if i in swap else cb[i] for i in range(len(vids))])
                    perm.append(metric(B_s, B_c, m) - metric(A_s, A_c, m))
                perm = np.array(perm); p_perm = float((np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (B + 1))
                lo, hi = np.percentile(d, [2.5, 97.5]); blo, bhi = np.percentile(d, [100 * 0.05 / 6, 100 - 100 * 0.05 / 6])
                fam.append({"unit": unit, "n_units": len(U), "metric": m, "comparison": f"{g} - 6x6", "observed_diff": round(obs, 4),
                            "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4), "ci_bonf3_lo": round(blo, 4), "ci_bonf3_hi": round(bhi, 4),
                            "p_bootstrap": round(p_boot, 4), "p_permutation": round(p_perm, 4)})
            for r, a in zip(fam, holm(np.array([r["p_permutation"] for r in fam]))):
                r["p_perm_holm_within_metric"] = round(float(a), 4)
            rows += fam
        print("done", unit)
    # Holm across all 18 tests of a unit
    for unit in units:
        sub = [r for r in rows if r["unit"] == unit]
        for r, a in zip(sub, holm(np.array([r["p_permutation"] for r in sub]))):
            r["p_perm_holm_all18"] = round(float(a), 4)
    with open(os.path.join(OUT, "significance_tests_test_set.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r["unit"], r["metric"], r["comparison"], r["observed_diff"], (r["ci95_lo"], r["ci95_hi"]), r["p_permutation"], r["p_perm_holm_within_metric"], r["p_perm_holm_all18"])


def count_one(wd):
    run = Run(wd); out = {}
    regs = ce.ignored_regions(run.cfg["xml_path"]); evs = ce.replay_counts(run)
    preds = []
    for ln, f, i in evs:
        x = run.bbox[i, 0] + int(run.bbox[i, 2]) // 2; L = run.meta["lanes"][ln]["roi"]
        if ce.in_ignored(x, L[1] + L[3] // 2, regs):
            continue
        preds.append({"lane": ln, "frame": f, "x": float(x), "ident": match_vehicle(tuple(run.bbox[i]), run.gt.get(f + 1, [])), "cls": "Car"})
    gts, outside = ce.gt_events(run)
    for scope in ("Lane1", "Lane2", "Combined"):
        P = [p for p in preds if scope == "Combined" or p["lane"] == scope]
        G = [x for x in gts if scope == "Combined" or x["lane"] == scope]
        pm, gm, dup, fal, opp = ce.match(P, G, same_lane=(scope != "Combined"))
        Gd = [i for i, x in enumerate(G) if x["dir"] == "down"]
        det_miss = assoc_miss = 0
        for b in Gd:
            if b in gm:
                continue
            e = G[b]; ok = False
            for i in np.flatnonzero((run.lane == e["lane"]) & (np.abs(run.frame - e["frame"]) <= ce.DT) & (run.supp == 0)):
                if match_vehicle(tuple(run.bbox[i]), [o for o in run.gt.get(int(run.frame[i]) + 1, []) if o["id"] == e["id"]]) is not None:
                    ok = True; break
            assoc_miss += ok; det_miss += (not ok)
        unmatched = [p for a, p in enumerate(P) if a not in pm]
        cnt = {"gt_down": len(Gd), "gt_up": len(G) - len(Gd), "pred": len(P), "tp": len(pm), "false": fal,
               "false_opposite_direction": opp, "duplicate": dup, "missed_down": len(Gd) - len(gm),
               "missed_down_not_detected": det_miss, "missed_down_detected_not_counted": assoc_miss,
               "unmatched_pred_no_vehicle_overlap": sum(1 for p in unmatched if p["ident"] is None),
               "gt_outside_zones": outside if scope == "Combined" else 0}
        out[scope] = (cnt, [len(P), len(Gd), len(G)])
    return out


def counting():
    """Per grid / lane decomposition on the test, validation and benchmark sets (downward GT matching, original thresholds)."""
    rows = []
    for set_name, grids in (("test", G4), ("validation", G4), ("benchmark", G4)):
        for g in grids:
            d = os.path.join(GRA, "runs", set_name, g)
            acc = defaultdict(lambda: defaultdict(float)); errs = defaultdict(list)
            for v in sorted(os.listdir(d)):
                cf = os.path.join(CACHE, f"cnt_{set_name}_{g}_{v}.json")
                if os.path.exists(cf):
                    res = json.load(open(cf))
                else:
                    if time.time() - T0 > BUDGET:
                        print("budget; rerun"); return
                    res = count_one(os.path.join(d, v)); json.dump(res, open(cf, "w"))
                for scope, (cnt, err) in res.items():
                    for k_, v_ in cnt.items():
                        acc[scope][k_] += v_
                    errs[scope].append(tuple(err))
            for scope in ("Lane1", "Lane2", "Combined"):
                a = acc[scope]; e = np.array(errs[scope], float)
                dd = e[:, 0] - e[:, 1]; da = e[:, 0] - e[:, 2]
                r = {"set": set_name, "grid": g, "scope": scope, "videos": len(e), **{k: int(v) for k, v in a.items()}}
                r["precision"] = round(a["tp"] / a["pred"], 4) if a["pred"] else None
                r["recall_down"] = round(a["tp"] / a["gt_down"], 4) if a["gt_down"] else None
                r["recall_all"] = round(a["tp"] / (a["gt_down"] + a["gt_up"]), 4) if a["gt_down"] + a["gt_up"] else None
                r["f1_down"] = round(2 * a["tp"] / (a["pred"] + a["gt_down"]), 4) if a["pred"] + a["gt_down"] else None
                r["f1_all"] = round(2 * a["tp"] / (a["pred"] + a["gt_down"] + a["gt_up"]), 4)
                r["mae_down"] = round(float(np.mean(np.abs(dd))), 3); r["rmse_down"] = round(float(np.sqrt(np.mean(dd ** 2))), 3)
                m = e[:, 1] > 0; r["mape_down"] = round(float(np.mean(np.abs(dd[m]) / e[m, 1])), 3); r["mape_down_n"] = int(m.sum())
                r["mae_all"] = round(float(np.mean(np.abs(da))), 3); r["rmse_all"] = round(float(np.sqrt(np.mean(da ** 2))), 3)
                m2 = e[:, 2] > 0; r["mape_all"] = round(float(np.mean(np.abs(da[m2]) / e[m2, 2])), 3)
                rows.append(r); print(r)
    with open(os.path.join(OUT, "counting_by_lane_and_error_type.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def classification():
    rows = []; boot = {}
    rng = np.random.default_rng(7)
    for g in G4:
        for tag in ("frozen", "original"):
            vids, cms, st = per_video("test", g, tag); cm = cms.sum(0)
            p = prf(cm); sup = cm.sum(1); tot = sup.sum()
            allcar = np.zeros((3, 3), int); allcar[:, 0] = sup; pa = prf(allcar)
            wf1 = float(sum(p[c][2] * sup[i] for i, c in enumerate(CLASSES)) / tot)
            wf1a = float(sum(pa[c][2] * sup[i] for i, c in enumerate(CLASSES)) / tot)
            r = {"grid": g, "thresholds": tag, "cm_rows_true_Car_Van_Bus": json.dumps(cm.tolist()),
                 **{f"support_{c}": int(sup[i]) for i, c in enumerate(CLASSES)},
                 **{f"{m}_{c}": round(p[c][j], 4) for c in CLASSES for j, m in enumerate(("precision", "recall", "f1"))},
                 "macro_f1": round(p["macro_f1"], 4), "weighted_f1": round(wf1, 4), "accuracy": round(p["accuracy"], 4),
                 "car_to_van_rate": round(cm[0, 1] / sup[0], 4), "car_to_bus_rate": round(cm[0, 2] / sup[0], 4),
                 "van_to_car_rate": round(cm[1, 0] / sup[1], 4), "van_to_bus_rate": round(cm[1, 2] / sup[1], 4),
                 "bus_to_car_rate": round(cm[2, 0] / sup[2], 4), "bus_to_van_rate": round(cm[2, 1] / sup[2], 4),
                 "always_car_macro_f1": round(pa["macro_f1"], 4), "always_car_weighted_f1": round(wf1a, 4), "always_car_accuracy": round(pa["accuracy"], 4)}
            # site-level bootstrap of (classifier - always-Car) macro-F1 and weighted-F1
            lab = [SITE_OF_TEST[v] for v in vids]; U = sorted(set(lab)); idx = {u: [i for i, x in enumerate(lab) if x == u] for u in U}
            dm, dw = [], []
            for _ in range(5000):
                b = np.concatenate([idx[u] for u in rng.choice(U, len(U))]).astype(int); c = cms[b].sum(0); s = c.sum(1)
                ac = np.zeros((3, 3), int); ac[:, 0] = s; pc, pac = prf(c), prf(ac)
                dm.append(pc["macro_f1"] - pac["macro_f1"])
                dw.append(sum(pc[k][2] * s[i] for i, k in enumerate(CLASSES)) / s.sum() - sum(pac[k][2] * s[i] for i, k in enumerate(CLASSES)) / s.sum())
            r["macro_f1_minus_always_car_ci95"] = json.dumps([round(x, 4) for x in np.percentile(dm, [2.5, 97.5])])
            r["weighted_f1_minus_always_car_ci95"] = json.dumps([round(x, 4) for x in np.percentile(dw, [2.5, 97.5])])
            rows.append(r); print(g, tag, r["macro_f1"], r["weighted_f1"], r["always_car_weighted_f1"], r["macro_f1_minus_always_car_ci95"], r["weighted_f1_minus_always_car_ci95"])
    with open(os.path.join(OUT, "classification_test_set.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def mapping():
    import xml.etree.ElementTree as ET, cv2
    man = json.load(open(os.path.join(GRA, "site_groups_manual.json")))
    eval_site = {v: k for k, vs in man["groups"].items() for v in vs}
    sel = json.load(open(os.path.join(REPO, "benchmark_results", "selection_report.json")))["selection"]
    bench = {os.path.basename(e["video"].replace("\\", "/"))[:-4] for e in sel}
    ts = json.load(open(os.path.join(GRA, "test_set.json")))
    excl = ts["excluded_resembling_eval_sites"]
    auto = json.load(open(os.path.join(GRA, "site_groups.json")))["site_of_video"]
    unc_eval = {"A_bus_stop_boulevard": "20051/20052 and 20032-35 are wider/shifted views; merged with 20011/2006x",
                "B_hedged_divider_road": "40141 not certainly the same camera as 39761/39771; merged",
                "C_wet_post_divider_road": "63544 framed differently; merged (same rainy session as 63521/63525)"}
    rows = []
    for f in sorted(glob.glob(os.path.join(REPO, "DETRAC_VIDEOS", "*.mp4"))):
        v = os.path.basename(f)[:-4]; xml = os.path.join(REPO, "DETRAC-Annotations", v + ".xml")
        cap = cv2.VideoCapture(f); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        root = ET.parse(xml).getroot() if os.path.exists(xml) else None
        sa = root.find("sequence_attribute") if root is not None else None
        nann = len(root.findall("frame")) if root is not None else 0
        if v in bench: part, site = "benchmark_12 (retrospective; site shared with validation for 8/12)", eval_site[v]
        elif v in eval_site: part, site = "validation_26 (CV selection and tuning)", eval_site[v]
        elif v in SITE_OF_TEST: part, site = "test_44 (unseen-site final evaluation)", SITE_OF_TEST[v]
        else: part, site = "excluded (unused; resembles an eval site)", "resembles_eval_site"
        unc = unc_eval.get(site, "") or TEST_UNCERTAIN.get(site, "")
        if v in excl: unc = excl[v]
        rows.append({"video_id": v, "site_id": site, "partition": part, "frames": n, "annotation_xml": int(root is not None),
                     "annotated_frames": nann, "weather_attr": sa.get("sence_weather") if sa is not None else "",
                     "camera_state_attr": sa.get("camera_state") if sa is not None else "", "recording_session_prefix": v[4:8],
                     "automatic_cluster_id_th0.4": auto.get(v, ""), "grouping_uncertainty": unc})
    with open(os.path.join(REPO, "paper_revision", "dataset_split_and_site_mapping.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    c = Counter(r["partition"].split(" ")[0] for r in rows); print(c)
    # checks
    val_sites = {eval_site[v] for v in eval_site}
    print("test sites overlapping eval sites:", set(SITE_OF_TEST.values()) & val_sites)


def objective():
    """Recompute the frozen selection from cached validation data and report the exact objective."""
    vids = sorted(os.listdir(os.path.join(GRA, "runs", "validation", "6x6")))
    data = {}
    for g in GRIDS:
        cl, sl = zip(*[video_stats("validation", g, v) for v in vids]); data[g] = (np.stack(cl, 1), list(sl))
    guard = timing_guard()
    sel, rows = select(list(range(len(vids))), data, guard)
    frozen = json.load(open(os.path.join(GRA, "selected_configuration.json")))
    out = {"weights": W, "tie_rule": TIE, "guard": guard, "recomputed_selection": sel, "frozen_selection": frozen["selected_grid"],
           "match": sel == frozen["selected_grid"] and CANDS[rows[sel]["cand"]] == frozen["thresholds"],
           "per_grid": {g: {k: round(v, 4) for k, v in rows[g].items() if k in ("J", "macro_f1", "min_class_f1", "count_f1", "det_recall", "frag", "unmatched")} | {"thresholds": CANDS[rows[g]["cand"]]} for g in GRIDS},
           "frozen_at": frozen["frozen_at"],
           "selected_configuration_mtime": os.path.getmtime(os.path.join(GRA, "selected_configuration.json")),
           "test_set_json_mtime": os.path.getmtime(os.path.join(GRA, "test_set.json")),
           "earliest_test_run_mtime": min(os.path.getmtime(p) for p in glob.glob(os.path.join(GRA, "runs", "test", "*", "*", "meta.json"))),
           "timing_summary_mtime": os.path.getmtime(os.path.join(GRA, "timing_trials_summary.json"))}
    json.dump(out, open(os.path.join(OUT, "objective_verification.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_grid"}, indent=1)); print(json.dumps(out["per_grid"], indent=0))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in ("objective", "mapping", "classification", "counting", "stats"):
        if what in (name, "all"):
            globals()[name]()
