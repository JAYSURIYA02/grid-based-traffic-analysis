"""
replay.py -- offline re-implementation of LaneTracker label voting and of the
vehicle-level part of rparallel_he3.evaluate_lane, driven by the per-frame
track logs written by `experiment.py worker`. With the original thresholds it
must reproduce eval_original.json exactly (checked by verify_run()).
Matching (IoU > 0.1 to ground truth at frame t+1) is threshold-independent and
computed once per run.
"""
import gzip, json, os, sys
from collections import Counter, defaultdict
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
CLASSES = ["Car", "Van", "Bus"]
IDX = {c: i for i, c in enumerate(CLASSES)}
ALPHA = 0.15
MIN_FRAMES = 5
ORIGINAL = {"bus_col": 0.75, "bus_both": 0.55, "van_row": 0.45, "van_max_aspect": None}

_gt_cache = {}


def load_gt(xml):
    """Same parsing as rparallel_he3.load_detrac (re-implemented to avoid importing the script)."""
    if xml in _gt_cache:
        return _gt_cache[xml]
    import xml.etree.ElementTree as ET
    m = {"car": "Car", "van": "Van", "bus": "Bus", "others": "Unknown"}
    gt = {}
    for fr in ET.parse(xml).getroot().findall("frame"):
        fid = int(fr.get("num")); gt[fid] = []
        tl = fr.find("target_list")
        if tl is None:
            continue
        for t in tl.findall("target"):
            b, a = t.find("box"), t.find("attribute")
            if b is None or a is None:
                continue
            gt[fid].append({"id": int(t.get("id", -1)),
                            "bbox": (float(b.get("left")), float(b.get("top")), float(b.get("width")), float(b.get("height"))),
                            "label": m.get(a.get("vehicle_type", "others").lower(), "Unknown")})
    _gt_cache[xml] = gt
    return gt


def in_roi(o, roi):
    x, y, w, h = roi; bx, by, bw, bh = o["bbox"]
    return not (bx + bw < x or bx > x + w or by + bh < y or by > y + h)


def match_vehicle(pb, gts):
    best, mx = None, 0
    px, py, pw, ph = pb
    for o in gts:
        gx, gy, gw, gh = o["bbox"]
        ix = max(0, min(px + pw, gx + gw) - max(px, gx)); iy = max(0, min(py + ph, gy + gh) - max(py, gy))
        inter = ix * iy; union = pw * ph + gw * gh - inter
        iou = inter / union if union > 0 else 0
        if iou > mx:
            mx, best = iou, o.get("id")
    return best if mx > 0.1 else None


class Run:
    """All threshold-independent data of one (grid, video) run."""
    def __init__(self, wd):
        self.wd = wd
        self.meta = json.load(open(os.path.join(wd, "meta.json")))
        self.cfg = json.load(open(os.path.join(wd, "user_input_data.json")))
        self.ev = json.load(open(os.path.join(wd, "eval_original.json")))
        d = json.load(gzip.open(os.path.join(wd, "track_log.json.gz"), "rt"))
        c = {n: i for i, n in enumerate(d["columns"])}
        rows = d["rows"]
        self.n = len(rows)
        self.lane = np.array([r[c["lane"]] for r in rows])
        self.frame = np.array([r[c["frame"]] for r in rows], int)
        self.track = np.array([r[c["track"]] for r in rows], int)
        self.age = np.array([r[c["frame_age"]] for r in rows], int)
        self.ncol = np.array([r[c["col_span"]] for r in rows], float) / self.meta["cols"]
        self.nrow = np.array([r[c["row_span"]] for r in rows], float) / self.meta["rows"]
        self.cells = np.array([r[c["cell_count"]] for r in rows], int)
        self.bbox = np.array([[r[c["bx"]], r[c["by"]], r[c["bw"]], r[c["bh"]]] for r in rows], float).reshape(-1, 4)
        self.cy = np.array([r[c["cy"]] for r in rows], float)
        self.final_logged = np.array([r[c["final_label"]] for r in rows])
        self.supp = np.array([r[c["suppressed"]] for r in rows], int)
        self.weight = (self.cy / max(1, self.meta["frame_height"])) * (1 + ALPHA) ** self.age
        self.aspect = self.bbox[:, 2] / np.maximum(self.bbox[:, 3], 1)
        gt = load_gt(self.cfg["xml_path"])
        self.gt = gt
        # threshold-independent matching
        self.gt_id = np.full(self.n, -10**9, dtype=np.int64)
        for i in range(self.n):
            if self.supp[i] or self.age[i] < MIN_FRAMES:
                continue
            roi = self.meta["lanes"][self.lane[i]]["roi"]
            gts = [o for o in gt.get(int(self.frame[i]) + 1, []) if in_roi(o, roi)]
            g = match_vehicle(tuple(self.bbox[i]), gts)
            if g is not None:
                self.gt_id[i] = g
        # ground-truth sets per lane (as evaluate_lane)
        self.gt_ids, self.gt_labels = {}, {}
        for ln, L in self.meta["lanes"].items():
            ids, labs = set(), {}
            for fid, objs in gt.items():
                for o in objs:
                    if not in_roi(o, L["roi"]):
                        continue
                    if o["id"] >= 0:
                        ids.add(o["id"])
                    if o["label"] in CLASSES:
                        labs[o["id"]] = o["label"]
            self.gt_ids[ln], self.gt_labels[ln] = ids, labs
        # track order for vote replay: rows are logged in frame order per lane
        self.key = np.char.add(self.lane.astype(str), np.char.add("#", self.track.astype(str)))

    def raw_labels(self, th):
        lab = np.zeros(self.n, int)  # Car
        van = self.nrow > th["van_row"]
        if th.get("van_max_aspect") is not None:
            van &= self.aspect <= th["van_max_aspect"]
        lab[van] = 1
        bus = (self.ncol > th["bus_col"]) | ((self.ncol > th["bus_both"]) & (self.nrow > th["bus_both"]))
        lab[bus] = 2
        return lab

    def final_labels(self, th):
        raw = self.raw_labels(th)
        out = np.zeros(self.n, int)
        votes = defaultdict(lambda: np.zeros(3))
        for i in range(self.n):              # logged in temporal order
            v = votes[self.key[i]]
            v[raw[i]] += self.weight[i]
            out[i] = int(np.argmax(v))       # first max wins, like max(dict) over Car,Van,Bus
        return out

    def evaluate(self, th, lanes=("Lane1", "Lane2"), combine=False):
        """Returns per-lane (or combined) dict with confusion matrix and ID counts."""
        fl = self.final_labels(th)
        res = {}
        groups = [tuple(lanes)] if combine else [(l,) for l in lanes]
        for grp in groups:
            pred = defaultdict(list); matched_tracks = set(); all_tracks = set()
            for i in range(self.n):
                if self.lane[i] not in grp or self.supp[i]:
                    continue
                all_tracks.add(self.key[i])
                if self.gt_id[i] == -10**9:
                    continue
                pred[int(self.gt_id[i])].append(CLASSES[fl[i]])
                matched_tracks.add(self.key[i])
            gt_ids = set().union(*[self.gt_ids[l] for l in grp])
            gt_lab = {}
            for l in grp:
                gt_lab.update(self.gt_labels[l])
            cm = np.zeros((3, 3), int)
            for g, lst in pred.items():
                p = Counter(lst).most_common(1)[0][0]
                t = gt_lab.get(g, "Car")
                cm[IDX[t if t in IDX else "Car"]][IDX[p]] += 1
            res["+".join(grp)] = {"cm": cm, "unique_gt": len(gt_ids), "matched_gt": len(pred),
                                  "missed_gt": len(gt_ids - set(pred)), "pred_tracks": len(all_tracks),
                                  "matched_tracks": len(matched_tracks),
                                  "unmatched_tracks": len(all_tracks - matched_tracks)}
        return res

    def verify(self):
        """Replay with original thresholds vs the logged final labels and eval_original."""
        fl = self.final_labels(ORIGINAL)
        ok_labels = bool(np.all(np.array(CLASSES)[fl] == self.final_logged))
        r = self.evaluate(ORIGINAL)
        ok_eval = True
        for ln in ("Lane1", "Lane2"):
            e = self.ev[ln]
            ok_eval &= r[ln]["cm"].tolist() == e["confusion_matrix_vehicle"]
            ok_eval &= r[ln]["unique_gt"] == e["unique_gt_vehicles"] and r[ln]["matched_gt"] == e["matched_gt_vehicles"]
            ok_eval &= r[ln]["matched_tracks"] == e["matched_predicted_tracks"]
            ok_eval &= r[ln]["unmatched_tracks"] == e["unmatched_predicted_tracks"]
        return ok_labels, ok_eval


# ───────────── vectorized replay (used for threshold search) ─────────────
class FastRun:
    """Vectorized equivalent of Run.final_labels/evaluate for speed."""
    def __init__(self, run):
        self.r = run
        codes, inv = np.unique(run.key, return_inverse=True)
        self.order = np.argsort(inv, kind="stable")
        g = inv[self.order]
        self.start = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
        self.grp_of_sorted = g
        self.inv_order = np.empty_like(self.order); self.inv_order[self.order] = np.arange(run.n)
        self.elig = (run.supp == 0) & (run.gt_id != -10**9)
        self.gidx = np.searchsorted(self.start, np.arange(run.n), side="right") - 1

    def final_labels(self, th):
        """Exact per-track cumulative votes: tracks are rows of a padded
        array, so each row's cumsum adds the same values in the same order as
        the sequential replay (no cross-track floating-point contamination;
        vote weights grow as 1.15**age and reach ~1e60)."""
        r = self.r
        if not hasattr(self, "_pos"):
            self._pos = np.arange(r.n) - self.start[self.gidx]
            self._G = len(self.start); self._L = int(self._pos.max()) + 1
        raw = r.raw_labels(th)[self.order]
        W = np.zeros((self._G, self._L, 3))
        W[self.gidx, self._pos, raw] = r.weight[self.order]
        cs = np.cumsum(W, axis=1)
        lab_sorted = np.argmax(cs[self.gidx, self._pos], axis=1)
        return lab_sorted[self.inv_order]

    def cm(self, th, lanes):
        r = self.r
        fl = self.final_labels(th)
        m = self.elig & np.isin(r.lane, lanes)
        idx = np.flatnonzero(m)
        if idx.size == 0:
            return np.zeros((3, 3), int)
        gids, ginv = np.unique(r.gt_id[idx], return_inverse=True)
        cnt = np.zeros((len(gids), 3)); first = np.full((len(gids), 3), np.inf)
        np.add.at(cnt, (ginv, fl[idx]), 1)
        np.minimum.at(first, (ginv, fl[idx]), np.arange(idx.size))
        score = cnt * 1e7 - np.where(np.isfinite(first), first, 1e6)
        pred = np.argmax(score, axis=1)
        gt_lab = {}
        for l in lanes:
            gt_lab.update(r.gt_labels[l])
        cm = np.zeros((3, 3), int)
        for g, p in zip(gids, pred):
            t = gt_lab.get(int(g), "Car")
            cm[IDX[t if t in IDX else "Car"]][p] += 1
        return cm
