"""
counting_evaluation.py -- ground-truth evaluation of the line-crossing counter.

The production counter (rparallel_he3.LaneTracker.compute_tracking, STEP 5) is
NOT modified. Its count events are replayed offline from the per-frame track
logs written by experiment.py, using the exact production rule:

    a track that is not yet `counted`, has age >= 2 and whose centroid moved
    from prev_y < line_y to y >= line_y in this update is counted, unless a
    crossing was already counted in the same lane within 30 frames and 60 px
    (is_duplicate_crossing -> track marked counted, no count). Tracks removed
    by suppress_duplicate_tracks are marked counted. line_y = roi_y + roi_h//2.

The replay is checked against the lane_vehicle_counts that the production run
itself wrote (rpar_frame_metrics.json); a run that does not reproduce exactly
is reported and excluded.

Ground-truth crossing events come from annotation trajectories, not from the
number of annotated vehicles: for each UA-DETRAC target id, the centre of its
box (left+w/2, top+h/2) in consecutive annotated frames; an event is recorded
when the centre crosses the same line_y. Direction: "down" (prev < line <= cur,
the only direction the production counter can count) or "up" (prev >= line > cur).
Only "down" GT events can be matched (the counter emits only downward
crossings); under the "all" GT definition every upward event is therefore a
missed count, which isolates the design limitation from detection errors.
The event is assigned to the lane whose x-range contains the centre at the
crossing frame; centres in the 12 px gap between the two ROIs or outside both
are reported as `outside_zone` and are excluded. Predicted events whose
crossing point lies inside a UA-DETRAC ignored region are excluded (those
vehicles are unannotated); the same exclusion is applied to every grid.
GT frame numbers are 1-based; pipeline frame t is compared with GT frame t+1
(as in evaluate_lane).

Matching (identical for every grid): a predicted event and a GT event are
compatible if |dt| <= 25 frames (1 s at 25 fps) and |dx| <= 80 px; in the
per-lane scope they must also be in the same lane. Pairs are assigned one-to-
one greedily by cost |dt|/25 + |dx|/80 - 1[identity agrees], where identity is
the GT id overlapping the counted track's box (IoU > 0.1) at that frame.
An unmatched prediction is a DUPLICATE if the vehicle it overlaps (identity)
or a compatible GT event was already matched to another prediction, else a
FALSE count. Unmatched GT events are MISSED counts. In the combined scope the
GT events of both lanes form one list and predictions of both lanes compete
for them, so a vehicle counted in both lanes yields one TP and one duplicate
(never two TPs).

Metrics per video / lane / scope: GT count, predicted count, signed and
absolute error, MAPE (GT>0 only), count accuracy = max(0, 1-|P-G|/G) and the
event-level F1 = 2TP/(P+G), which unlike count accuracy does not reward
false and missed counts that cancel out. Per-class counts use the final label
of the counted track (end of track) and the GT vehicle_type of the event
("others" counted as Car, as in the classification evaluation).

Usage:
    python grid_resolution_analysis/counting_evaluation.py --set benchmark --grids 6x6,8x8,10x12
"""
import argparse, csv, json, os, sys
from collections import defaultdict, Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from replay import Run, ORIGINAL, CLASSES, match_vehicle, load_gt

DT, DX = 25, 80
COOLDOWN_F, COOLDOWN_PX = 30, 60


def ignored_regions(xml):
    import xml.etree.ElementTree as ET
    ir = ET.parse(xml).getroot().find("ignored_region")
    return [] if ir is None else [tuple(float(b.get(k)) for k in ("left", "top", "width", "height")) for b in ir.findall("box")]


def in_ignored(x, y, regs):
    return any(l <= x <= l + w and t <= y <= t + h for l, t, w, h in regs)


def replay_counts(run):
    """Exact replay of STEP 5. Returns list of events (lane, frame, row_index)."""
    ev = []
    for ln, L in run.meta["lanes"].items():
        x, y, w, h = L["roi"]; line = y + h // 2
        idx = np.flatnonzero(run.lane == ln)          # temporal order, track-id order within a frame
        counted, last_cy, crossings = set(), {}, []
        cur = None
        for i in idx:
            f = int(run.frame[i]); k = int(run.track[i])
            if f != cur:
                # rows of one frame: suppression already applied when logged; process in id order
                cur = f
                crossings = [(pf, px) for pf, px in crossings if f - pf < COOLDOWN_F]
            prev = last_cy.get(k, run.cy[i])
            last_cy[k] = run.cy[i]
            if run.supp[i]:
                counted.add(k); continue
            if k in counted or run.age[i] < 2:
                continue
            if prev < line <= run.cy[i]:
                cx = run.bbox[i, 0] + int(run.bbox[i, 2]) // 2
                if any(f - pf < COOLDOWN_F and abs(cx - px) < COOLDOWN_PX for pf, px in crossings):
                    counted.add(k); continue
                counted.add(k); crossings.append((f, cx)); ev.append((ln, f, int(i)))
    return ev


def gt_events(run):
    gt = run.gt; lanes = run.meta["lanes"]
    line = {ln: L["roi"][1] + L["roi"][3] // 2 for ln, L in lanes.items()}
    lines = set(line.values()); assert len(lines) == 1, "both lanes share the counting line"
    ly = lines.pop()
    traj = defaultdict(list)
    for fid in sorted(gt):
        for o in gt[fid]:
            l, t, w, h = o["bbox"]
            traj[o["id"]].append((fid, l + w / 2, t + h / 2, o["label"]))
    out, outside = [], 0
    for gid, tr in traj.items():
        for (f0, x0, y0, _), (f1, x1, y1, lab) in zip(tr, tr[1:]):
            d = "down" if y0 < ly <= y1 else ("up" if y0 >= ly > y1 else None)
            if d is None:
                continue
            lane = next((ln for ln, L in lanes.items() if L["roi"][0] <= x1 < L["roi"][0] + L["roi"][2]), None)
            if lane is None:
                outside += 1; continue
            out.append({"lane": lane, "frame": f1 - 1, "x": x1, "id": gid, "dir": d,
                        "cls": lab if lab in CLASSES else "Car", "gap": f1 - f0})
    return out, outside


def final_label_per_track(run, th):
    from replay import FastRun
    fl = FastRun(run).final_labels(th)
    last = {}
    for i in range(run.n):
        last[run.key[i]] = CLASSES[fl[i]]
    return last


def match(preds, gts, same_lane):
    pairs = []
    for a, p in enumerate(preds):
        for b, g in enumerate(gts):
            if same_lane and p["lane"] != g["lane"]:
                continue
            if g["dir"] != "down":          # the counter only emits downward crossings
                continue
            dt = abs(p["frame"] - g["frame"]); dx = abs(p["x"] - g["x"])
            if dt <= DT and dx <= DX:
                pairs.append((dt / DT + dx / DX - (1.0 if p["ident"] == g["id"] else 0.0), a, b))
    pairs.sort()
    pm, gm = {}, {}
    for c, a, b in pairs:
        if a not in pm and b not in gm:
            pm[a] = b; gm[b] = a
    matched_ids = {gts[b]["id"] for b in gm}
    dup = fal = opp = 0
    up_ids = {g["id"] for g in gts if g["dir"] == "up"}
    for a, p in enumerate(preds):
        if a in pm:
            continue
        near = any((not same_lane or p["lane"] == gts[b]["lane"]) and abs(p["frame"] - gts[b]["frame"]) <= DT
                   and abs(p["x"] - gts[b]["x"]) <= DX for b in gm)
        if (p["ident"] is not None and p["ident"] in matched_ids) or near:
            dup += 1
        else:
            fal += 1
            near_up = any(g["dir"] == "up" and (not same_lane or p["lane"] == g["lane"])
                          and abs(p["frame"] - g["frame"]) <= DT and abs(p["x"] - g["x"]) <= DX for g in gts)
            if (p["ident"] is not None and p["ident"] in up_ids) or near_up:
                opp += 1
    return pm, gm, dup, fal, opp


def evaluate_run(run, th=ORIGINAL):
    regs = ignored_regions(run.cfg["xml_path"])
    evs = replay_counts(run)
    met = json.load(open(os.path.join(run.wd, "rpar_frame_metrics.json")))
    produced = met["lane_vehicle_counts"]
    rep = Counter(ln for ln, _, _ in evs)
    verified = all(rep.get(ln, 0) == produced[ln] for ln in produced)
    labels = final_label_per_track(run, th)
    preds, excluded = [], 0
    for ln, f, i in evs:
        x = run.bbox[i, 0] + int(run.bbox[i, 2]) // 2
        L = run.meta["lanes"][ln]["roi"]; ly = L[1] + L[3] // 2
        if in_ignored(x, ly, regs):
            excluded += 1; continue
        gts_f = [o for o in run.gt.get(f + 1, [])]
        ident = match_vehicle(tuple(run.bbox[i]), gts_f)
        preds.append({"lane": ln, "frame": f, "x": float(x), "ident": ident, "cls": labels[run.key[i]]})
    gts_all, outside = gt_events(run)
    rows = []
    for gtdef in ("all", "down"):
        gts = [g for g in gts_all if gtdef == "all" or g["dir"] == "down"]
        for scope in ("Lane1", "Lane2", "Combined"):
            P = [p for p in preds if scope == "Combined" or p["lane"] == scope]
            G = [g for g in gts if scope == "Combined" or g["lane"] == scope]
            pm, gm, dup, fal, opp = match(P, G, same_lane=(scope != "Combined"))
            r = {"gt_def": gtdef, "scope": scope, "gt": len(G), "pred": len(P), "tp": len(pm),
                 "duplicate": dup, "false": fal, "false_from_opposite_direction": opp, "missed": len(G) - len(gm),
                 "gt_up_events": sum(g["dir"] == "up" for g in G)}
            for c in CLASSES:
                r[f"gt_{c}"] = sum(g["cls"] == c for g in G); r[f"pred_{c}"] = sum(p["cls"] == c for p in P)
                r[f"tp_{c}_classok"] = sum(1 for a, b in pm.items() if P[a]["cls"] == c and G[b]["cls"] == c)
            rows.append(r)
    return {"verified": verified, "replayed": dict(rep), "produced": produced,
            "excluded_ignored": excluded, "gt_outside_zone": outside, "rows": rows}


def derived(r):
    G, P, TP = r["gt"], r["pred"], r["tp"]
    e = P - G
    return {"error": e, "abs_error": abs(e), "ape": abs(e) / G if G else None,
            "count_accuracy": max(0.0, 1 - abs(e) / G) if G else None,
            "event_precision": TP / P if P else None, "event_recall": TP / G if G else None,
            "event_f1": 2 * TP / (P + G) if (P + G) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="benchmark")
    ap.add_argument("--grids", default="6x6,8x8,10x12")
    ap.add_argument("--variant", default="")
    ap.add_argument("--thresholds", default="", help="json file {grid: thresholds}; default original")
    ap.add_argument("--out", default=os.path.join(HERE, "counting_results.csv"))
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    ths = json.load(open(a.thresholds)) if a.thresholds else {}
    base = os.path.join(HERE, "runs_variants", a.variant) if a.variant else os.path.join(HERE, "runs")
    cache = os.path.join(HERE, "counting_cache"); os.makedirs(cache, exist_ok=True)
    out = []; failures = []
    for g in a.grids.split(","):
        th = ths.get(g, ORIGINAL); tag = "tuned" if g in ths else "original"
        d = os.path.join(base, a.set, g)
        for v in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            wd = os.path.join(d, v)
            if not os.path.exists(os.path.join(wd, "track_log.json.gz")):
                failures.append((g, v, "no run")); continue
            cf = os.path.join(cache, f"{a.variant or 'orig'}_{a.set}_{g}_{v}_{tag}.json")
            if os.path.exists(cf):
                res = json.load(open(cf))
            else:
                res = evaluate_run(Run(wd), th); json.dump(res, open(cf, "w"))
            if not res["verified"]:
                failures.append((g, v, f"replay {res['replayed']} != produced {res['produced']}"))
                continue
            for r in res["rows"]:
                out.append({"variant": a.variant or "original", "set": a.set, "grid": g, "thresholds": tag,
                            "video": v, **r, **derived(r), "excluded_ignored": res["excluded_ignored"],
                            "gt_outside_zone": res["gt_outside_zone"]})
    mode = "a" if a.append and os.path.exists(a.out) else "w"
    with open(a.out, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        if mode == "w":
            w.writeheader()
        w.writerows(out)
    print(f"{len(out)} rows -> {a.out}; failures: {failures}")
    with open(os.path.join(HERE, "failures.log"), "a") as fl:
        for x in failures:
            fl.write(f"counting {a.variant or 'orig'} {a.set} {x}\n")


if __name__ == "__main__":
    main()
