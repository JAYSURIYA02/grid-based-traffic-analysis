"""
fragmentation_analysis.py -- where does fragmentation come from as the grid gets finer?

1. Detection stage (analytic + numeric): per-cell dilation kernel and the
   fixed 100 px^2 minimum contour area, for every grid size.
2. Clustering stage: for every GT observation covered by >= 2 detections
   (the fragmentation definition of analysis.py), the geometric relation of
   the fragment pairs in cell units: vertical gap, horizontal gap, and whether
   their boxes touch only at a corner (candidate diagonal-only adjacency).
   Causal test: experimental variants run separately (experiment.py --variant):
     conn8  : 8-connected grouping (diagonal neighbours join)
     bridge : 8-connected + bridging one empty cell vertically
   If a variant removes most fragmentation, the cause is clustering of a
   sparse hot-cell pattern (detection gaps inside a vehicle), not tracking.
3. Tracking stage: per GT vehicle, number of distinct predicted tracks matched
   over its lifetime (IoU > 0.1 at t+1, any age) and identity switches.

    python grid_resolution_analysis/fragmentation_analysis.py --set validation
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from replay import Run, in_roi, match_vehicle
GRIDS = ["4x6", "6x6", "6x8", "8x8", "8x10", "10x12"]
ROI_W, ROI_H = 441, 141


def detection_stage():
    out = {}
    for g in GRIDS:
        r, c = map(int, g.split("x")); cw, ch = ROI_W // c, ROI_H // r
        img = np.zeros((ch, cw), np.uint8); img[ch // 2, cw // 2] = 255         # one supra-threshold pixel
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, cw), max(1, ch // 2)))
        d = cv2.dilate(img, k, iterations=5)
        cs, _ = cv2.findContours(d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img2 = np.zeros((ch, cw), np.uint8); img2[0, 0] = 255                   # corner pixel (worst case)
        d2 = cv2.dilate(img2, k, iterations=5); cs2, _ = cv2.findContours(d2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out[g] = {"cell_w_px": cw, "cell_h_px": ch, "cell_area_px2": cw * ch, "kernel": [max(1, cw), max(1, ch // 2)],
                  "single_pixel_contour_area_px2": float(max(cv2.contourArea(x) for x in cs)),
                  "corner_pixel_contour_area_px2": float(max(cv2.contourArea(x) for x in cs2)),
                  "area_filter_100_binding": bool(max(cv2.contourArea(x) for x in cs2) <= 100)}
    return out


def clustering_and_tracking(run):
    r = run; pairs = []; frag_obs = obs = 0
    per_gt_tracks = defaultdict(list)                                         # gid -> [(frame, key)]
    for ln, L in r.meta["lanes"].items():
        roi = L["roi"]; gw, gh = L["grid_w"], L["grid_h"]
        m = (r.lane == ln) & (r.supp == 0)
        for fr in np.unique(r.frame[m]):
            idx = np.flatnonzero(m & (r.frame == fr))
            g = [o for o in r.gt.get(int(fr) + 1, []) if in_roi(o, roi)]
            boxes = [tuple(r.bbox[i]) for i in idx]
            for i, b in zip(idx, boxes):
                gid = match_vehicle(b, g)
                if gid is not None:
                    per_gt_tracks[(ln, gid)].append((int(fr), r.key[i]))
            for o in g:
                x, y, w, h = o["bbox"]; cx, cy = x + w / 2, y + h / 2
                if not (roi[0] <= cx <= roi[0] + roi[2] and roi[1] <= cy <= roi[1] + roi[3]):
                    continue
                obs += 1
                inside = [b for b in boxes if x <= b[0] + b[2] / 2 <= x + w and y <= b[1] + b[3] / 2 <= y + h]
                if len(inside) < 2:
                    continue
                frag_obs += 1
                for a in range(len(inside)):
                    for b_ in range(a + 1, len(inside)):
                        A, B = inside[a], inside[b_]
                        vg = (max(A[1], B[1]) - min(A[1] + A[3], B[1] + B[3])) / gh   # >0 gap, <=0 overlap/touch
                        hg = (max(A[0], B[0]) - min(A[0] + A[2], B[0] + B[2])) / gw
                        pairs.append((round(vg, 2), round(hg, 2)))
    sw = []; ntr = []
    for k, lst in per_gt_tracks.items():
        lst.sort(); keys = [t for _, t in lst]
        ntr.append(len(set(keys))); sw.append(sum(1 for a, b in zip(keys, keys[1:]) if a != b))
    return obs, frag_obs, pairs, ntr, sw


def classify_pair(vg, hg):
    if vg > 0.01 and hg <= 0.01:
        return "vertical_gap" if vg <= 1.01 else "vertical_gap_gt1"
    if hg > 0.01 and vg <= 0.01:
        return "horizontal_gap" if hg <= 1.01 else "horizontal_gap_gt1"
    if abs(vg) <= 0.01 and abs(hg) <= 0.01:
        return "corner_touch"
    if vg > 0.01 and hg > 0.01:
        return "diagonal_gap"
    return "overlapping_boxes"          # interleaved non-rectangular clusters


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="validation")
    ap.add_argument("--variants", default=",conn8,bridge"); a = ap.parse_args()
    rows = []
    for var in a.variants.split(","):
        base = os.path.join(HERE, "runs_variants", var) if var else os.path.join(HERE, "runs")
        for g in GRIDS:
            d = os.path.join(base, a.set, g)
            if not os.path.isdir(d):
                continue
            vids = [v for v in sorted(os.listdir(d)) if os.path.exists(os.path.join(d, v, "track_log.json.gz"))]
            O = F = 0; P = []; NT = []; SW = []
            for v in vids:
                o, f, p, nt, sw = clustering_and_tracking(Run(os.path.join(d, v))); O += o; F += f; P += p; NT += nt; SW += sw
            cls = defaultdict(int)
            for vg, hg in P:
                cls[classify_pair(vg, hg)] += 1
            rows.append({"variant": var or "original", "set": a.set, "grid": g, "n_videos": len(vids),
                         "gt_observations": O, "fragmented_observations": F, "fragmentation_rate": round(F / max(1, O), 4),
                         "fragment_pairs": len(P), **{f"pairs_{k}": cls[k] for k in ("vertical_gap", "vertical_gap_gt1", "horizontal_gap", "horizontal_gap_gt1", "corner_touch", "diagonal_gap", "overlapping_boxes")},
                         "gt_vehicles_tracked": len(NT), "mean_tracks_per_gt_vehicle": round(float(np.mean(NT)), 3) if NT else None,
                         "share_gt_vehicles_with_id_switch": round(float(np.mean([s > 0 for s in SW])), 4) if SW else None,
                         "id_switches_total": int(sum(SW))})
            print(rows[-1])
    with open(os.path.join(HERE, f"fragmentation_{a.set}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(detection_stage(), open(os.path.join(HERE, "fragmentation_detection_stage.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
