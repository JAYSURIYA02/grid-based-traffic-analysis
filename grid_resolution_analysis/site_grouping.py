"""Infer camera-site groups for the 100 UA-DETRAC videos.

UA-DETRAC XMLs carry no location field, so sites are inferred from the
static scene: a per-video median background (15 evenly spaced frames) is
reduced to a gradient-magnitude map (robust to day/night/weather changes),
compared pairwise with normalised cross-correlation, and clustered by
single-linkage at a fixed threshold. A contact sheet is written so the
grouping can be checked by eye. Resumable: backgrounds are cached.
"""
import os, sys, json, glob
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "site_grouping")
VIDS = os.path.join(REPO, "DETRAC_VIDEOS")
os.makedirs(OUT, exist_ok=True)

def background(path, n=15):
    cap = cv2.VideoCapture(path); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fr = []
    for i in np.linspace(0, max(N - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
        if ok: fr.append(cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.median(np.stack(fr), 0).astype(np.uint8)

def feat(bg):
    g = cv2.GaussianBlur(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    m = cv2.resize(m, (80, 45), interpolation=cv2.INTER_AREA).ravel()
    m = (m - m.mean()) / (m.std() + 1e-6)
    return m

def step_bg(budget=150):
    import time; t0 = time.time()
    for p in sorted(glob.glob(os.path.join(VIDS, "*.mp4"))):
        v = os.path.basename(p)[:-4]; f = os.path.join(OUT, v + ".png")
        if os.path.exists(f): continue
        if time.time() - t0 > budget: print("budget hit"); return False
        cv2.imwrite(f, background(p))
    return True

def cluster(th):
    files = sorted(glob.glob(os.path.join(OUT, "MVI_*.png")))
    names = [os.path.basename(f)[:-4] for f in files]
    F = np.stack([feat(cv2.imread(f)) for f in files])
    S = F @ F.T / F.shape[1]
    par = list(range(len(names)))
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if S[i, j] >= th: par[find(i)] = find(j)
    groups = {}
    for i, n in enumerate(names): groups.setdefault(find(i), []).append(n)
    gl = sorted(groups.values(), key=lambda g: g[0])
    site = {n: f"S{k+1:02d}" for k, g in enumerate(gl) for n in g}
    # nearest neighbour similarity, for the report
    nn = {n: (names[int(np.argsort(-S[i])[1])], float(np.sort(S[i])[-2])) for i, n in enumerate(names)}
    return names, S, gl, site, nn

def sheet(gl):
    rows = []
    for k, g in enumerate(gl):
        ims = [cv2.imread(os.path.join(OUT, n + ".png")) for n in g]
        for im, n in zip(ims, g):
            cv2.putText(im, n[4:], (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
        lab = np.zeros((90, 40, 3), np.uint8); cv2.putText(lab, f"S{k+1:02d}", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        rows.append(np.hstack([lab] + ims))
    W = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((90, W - r.shape[1], 3), np.uint8)]) for r in rows]
    cv2.imwrite(os.path.join(HERE, "site_contact_sheet.png"), np.vstack(rows))

if __name__ == "__main__":
    if sys.argv[1] == "bg": print("done" if step_bg() else "partial")
    else:
        th = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        names, S, gl, site, nn = cluster(th)
        for g in gl: print(len(g), " ".join(x[4:] for x in g))
        if sys.argv[1] == "save":
            sheet(gl)
            json.dump({"method": __doc__.strip(), "threshold": th, "site_of_video": site,
                       "groups": {site[g[0]]: g for g in gl},
                       "nearest_neighbour": nn}, open(os.path.join(HERE, "site_groups.json"), "w"), indent=1)
