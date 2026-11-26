"""
make_figures.py -- regenerate every figure of paper/paper_8p.tex from the
repository's own data (UA-DETRAC frames and benchmark_results_v2/).

    python paper/make_figures.py          (run from the repository root)

Charts are written as vector PDF, frames as JPEG, into paper/figures/.
The processing-stage figure re-implements the per-ROI preprocessing and
per-cell detection with exactly the constants of seq.py (Table II).
"""
import csv
import json
import os
import statistics as st

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "paper_revision", "figures")  # revised copy; original paper/figures untouched
V2 = os.path.join(REPO, "benchmark_results_v2")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.pad_inches": 0.02,
})
COL = 3.45  # IEEE column width (in)
C = {"A": "#4C72B0", "B": "#55A868", "C": "#DD8452", "D": "#C44E52",
     "norm": "#C44E52", "amort": "#DD8452", "same": "#4C72B0"}

ROI1 = (47, 202, 441, 141)
ROI2 = (500, 202, 441, 141)
ROWS = COLS = 6
VIDEO = os.path.join(REPO, "DETRAC_VIDEOS_eval", "Cloudy", "MVI_39761.mp4")
OVERLAY = os.path.join(V2, "Cloudy", "MVI_39761", "parallel", "clean", "test_output1.mp4")


def read_frames(path, idx, n=2):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    frames = [cap.read()[1] for _ in range(n)]
    cap.release()
    return frames


def he_v(frame, roi):
    x, y, w, h = roi
    hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    frame[y:y + h, x:x + w] = cv2.cvtColor(cv2.merge([hh, ss, cv2.equalizeHist(vv)]), cv2.COLOR_HSV2BGR)
    return frame


def cell_detect(chan, gw, gh):
    blur = cv2.GaussianBlur(chan, (5, 5), 1)
    _, th = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, gw), max(1, gh // 2)))
    dil = cv2.dilate(th, k, iterations=5)
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return dil, [c for c in cnts if cv2.contourArea(c) > 100]


def stages(idx):
    f1, f2 = read_frames(VIDEO, idx)
    raw = f1.copy()
    a, b = he_v(f1.copy(), ROI1), he_v(f2.copy(), ROI1)
    diff = cv2.absdiff(cv2.GaussianBlur(a, (7, 7), 0), cv2.GaussianBlur(b, (7, 7), 0))
    v = cv2.split(cv2.cvtColor(diff, cv2.COLOR_BGR2HSV))[2]
    x, y, w, h = ROI1
    gw, gh = w // COLS, h // ROWS
    vroi = v[y:y + h, x:x + w]
    mask = np.zeros_like(vroi)
    hot = np.zeros((ROWS, COLS), int)
    for r in range(ROWS):
        for c in range(COLS):
            cell = vroi[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
            dil, cnts = cell_detect(cell, gw, gh)
            mask[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw] = dil
            hot[r, c] = 1 if cnts else 0
    return raw, a, vroi, mask, hot, (gw, gh)


# ── choose a frame with several vehicles in ROI 1 ─────────────────────────
best = max(range(200, 1400, 50), key=lambda i: stages(i)[4].sum())
raw, he, vroi, mask, hot, (gw, gh) = stages(best)
x, y, w, h = ROI1

# Fig: sample frame with ROIs
fr = raw.copy()
for roi, colr, name in ((ROI1, (60, 200, 60), "ROI 1 (Lane 1)"), (ROI2, (0, 140, 255), "ROI 2 (Lane 2)")):
    rx, ry, rw, rh = roi
    cv2.rectangle(fr, (rx, ry), (rx + rw, ry + rh), colr, 3)
    cv2.line(fr, (rx, ry + rh // 2), (rx + rw, ry + rh // 2), (255, 255, 255), 2)
    cv2.putText(fr, name, (rx + 6, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colr, 2)
cv2.imwrite(os.path.join(OUT, "fig_roi_frame.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 88])

# Fig: processing stages of ROI 1
crop = raw[y:y + h, x:x + w]
hecrop = he[y:y + h, x:x + w]
vis_diff = cv2.normalize(vroi, None, 0, 255, cv2.NORM_MINMAX)
grid = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
grid = (0.55 * grid).astype(np.uint8)
over = hecrop.copy()
for r in range(ROWS):
    for c in range(COLS):
        p0, p1 = (c * gw, r * gh), ((c + 1) * gw - 1, (r + 1) * gh - 1)
        colr = (40, 220, 40) if hot[r, c] else (40, 40, 230)
        cv2.rectangle(over, p0, p1, colr, 1 if not hot[r, c] else 2)
panels = [("(a) ROI 1", crop), ("(b) V-channel hist. eq.", hecrop),
          ("(c) $|\\Delta|$ V (stretched)", vis_diff), ("(d) per-cell mask", grid),
          ("(e) hot cells", over)]
fig, axs = plt.subplots(1, 5, figsize=(7.16, 0.95))
for ax, (t, im) in zip(axs, panels):
    if im.ndim == 2:
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_title(t, pad=2); ax.axis("off")
plt.subplots_adjust(wspace=0.04)
fig.savefig(os.path.join(OUT, "fig_stages.pdf"), dpi=300); plt.close(fig)

# Fig: actual two-lane output overlay from the benchmark run
ov = read_frames(OVERLAY, best - 1, 1)[0]
if ov is not None:
    cv2.imwrite(os.path.join(OUT, "fig_overlay.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 88])

# ── benchmark data ───────────────────────────────────────────────────────
rows = list(csv.DictReader(open(os.path.join(V2, "consolidated_results.csv"))))
sp = list(csv.DictReader(open(os.path.join(V2, "speedup_by_video.csv"))))
amd = list(csv.DictReader(open(os.path.join(V2, "amdahl_by_video.csv"))))
f = lambda v: float(v) if v not in ("", None) else np.nan
ms = lambda v: (st.mean(v), st.stdev(v))

# Fig: F1 by category and tier
seq = [r for r in rows if r["implementation"] == "sequential"]
fig, axs = plt.subplots(1, 2, figsize=(COL, 1.55), sharey=True, gridspec_kw={"width_ratios": [4, 3]})
for ax, key, groups in ((axs[0], "category", ["Cloudy", "Night", "Rainy", "Sunny"]),
                        (axs[1], "tier", ["light", "medium", "heavy"])):
    xs = np.arange(len(groups))
    for k, (cls, colr) in enumerate((("car", "#4C72B0"), ("van", "#55A868"), ("bus", "#DD8452"))):
        m = [ms([f(r[f"{cls}_f1"]) for r in seq if r[key] == g]) for g in groups]
        ax.bar(xs + (k - 1) * 0.26, [a for a, _ in m], 0.26, yerr=[b for _, b in m],
               color=colr, label=cls.capitalize(), error_kw={"lw": 0.5, "capsize": 1.5})
    ax.set_xticks(xs); ax.set_xticklabels([g.capitalize() for g in groups])
    ax.set_ylim(0, 1); ax.grid(axis="y", lw=0.3, alpha=0.6)
axs[0].set_ylabel("F1"); axs[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, borderaxespad=0.1)
fig.savefig(os.path.join(OUT, "fig_f1.pdf")); plt.close(fig)

# Fig: four workloads (loop + I/O stacked, CPU cores on twin axis)
wl = [("A", "sequential", "Lane1(ROI1)"), ("B", "parallel_1lane", "Lane1"),
      ("C", "serial_2lane", "Lane1"), ("D", "parallel", "Lane1")]
lab = ["A\nseq, 1 lane", "B\nthreads, 1 lane", "C\nno threads,\n2 lanes", "D\nthreads,\n2 lanes"]
fig, ax = plt.subplots(figsize=(COL, 1.75))
xs = np.arange(4)
loop = [ms([f(r["mean_frame_time_ms"]) for r in rows if r["implementation"] == i and r["lane"] == l]) for _, i, l in wl]
io = [ms([f(r["mean_frame_io_ms"]) for r in rows if r["implementation"] == i and r["lane"] == l]) for _, i, l in wl]
cpu = [ms([f(r["cpu_utilization_of_one_core"]) for r in rows if r["implementation"] == i and r["lane"] == l]) for _, i, l in wl]
ax.bar(xs, [a for a, _ in loop], 0.55, yerr=[b for _, b in loop], color=[C[k] for k, _, _ in wl],
       error_kw={"lw": 0.5, "capsize": 2}, label="processing loop")
ax.bar(xs, [a for a, _ in io], 0.55, bottom=[a for a, _ in loop], color="#BBBBBB", label="output I/O + decode")
for xi, (a, _), (b, _) in zip(xs, loop, io):
    ax.text(xi, a + b + 2.2, f"{a:.1f}", ha="center", va="bottom", fontsize=7)
ax.set_ylabel("time per frame (ms)"); ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=7)
ax.set_ylim(0, 27); ax.grid(axis="y", lw=0.3, alpha=0.6); ax.legend(loc="upper left", frameon=False)
ax2 = ax.twinx()
ax2.errorbar(xs + 0.36, [a for a, _ in cpu], yerr=[b for _, b in cpu], fmt="D", ms=3, color="k", lw=0.6, capsize=1.5)
ax2.set_ylim(0, 8); ax2.set_ylabel("CPU (cores)", fontsize=7)
fig.savefig(os.path.join(OUT, "fig_workloads.pdf")); plt.close(fig)

# Fig: per-video ratios vs density
d = np.array([f(r["density"]) for r in sp])
order = np.argsort(d)
norm = np.array([f(r["normalized_speedup"]) for r in sp])
amort = np.array([2 * f(r["T_A_s"]) / f(r["T_C_s"]) for r in sp])
same = np.array([f(r["same_workload_speedup_2lane"]) for r in sp])
fig, ax = plt.subplots(figsize=(COL, 1.85))
ax.axhline(1.0, color="k", lw=0.6, ls="--")
ax.plot(d[order], amort[order], "s-", ms=3, color=C["amort"], label=r"$S_{\mathrm{amort}}=2T_A/T_C$ (2 lanes, no threads)")
ax.plot(d[order], norm[order], "o-", ms=3, color=C["norm"], label=r"$S_{\mathrm{norm}}=2T_A/T_D$ (2 lanes, threads)")
ax.plot(d[order], same[order], "^-", ms=3, color=C["same"], label=r"$S_{\mathrm{same}}=T_C/T_D$ (threads only)")
ax.set_xlabel("traffic density (mean annotated vehicles per frame)"); ax.set_ylabel("ratio (dimensionless)")
ax.set_ylim(0.6, 2.5); ax.grid(lw=0.3, alpha=0.6); ax.legend(loc="upper right", frameon=False, fontsize=7, ncol=1)
fig.savefig(os.path.join(OUT, "fig_ratios.pdf")); plt.close(fig)

# Fig: Amdahl ceilings vs measured threads-only ratio
m39 = [r for r in amd if r["video"] == "MVI_39311"][0]
pn, pb = f(m39["p_narrow"]), f(m39["p_broad"])
N = np.linspace(1, 8, 200)
A_ = lambda p, n: 1 / ((1 - p) + p / n)
fig, ax = plt.subplots(figsize=(COL, 1.7))
ax.plot(N, A_(pb, N), color="#8172B2", label=f"broad, $p$={pb:.3f}")
ax.plot(N, A_(pn, N), color="#55A868", label=f"narrow (cell-pool work), $p$={pn:.3f}")
sm, ss = ms(list(same))
ax.errorbar([2], [sm], yerr=[ss], fmt="o", color=C["same"], ms=4, capsize=2, label=f"measured $T_C/T_D$ = {sm:.2f}")
ax.axhline(1.0, color="k", lw=0.5, ls="--")
for n in (2, 4, 8):
    ax.axvline(n, color="#999999", lw=0.3)
ax.set_xlabel("workers $N$"); ax.set_ylabel("speedup $S(N)$"); ax.set_xlim(1, 8); ax.set_ylim(0.6, 3.4)
ax.legend(loc="upper left", frameon=False, borderpad=0.1, ncol=1); ax.grid(lw=0.3, alpha=0.5)
fig.savefig(os.path.join(OUT, "fig_amdahl.pdf")); plt.close(fig)

# Fig: speed-estimation calibration and a qualitative calibrated run.
# Sources (paper/assets/): a screenshot of the web UI's two-point
# calibration step and a frame of a screen recording of seq.py running with
# that calibration. Speeds in the frame are unvalidated (see paper).
ASSETS = os.path.join(REPO, "paper", "assets")
cal_p = os.path.join(ASSETS, "calibration_ui_screenshot.png")
run_p = os.path.join(ASSETS, "speed_run_frame847.png")
if os.path.exists(cal_p) and os.path.exists(run_p):
    cal = cv2.imread(cal_p)[98:552, 396:1202]      # video frame area of the UI
    run = cv2.imread(run_p)[34:440, 0:640]          # ROI region below the window title bar
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 1.95), gridspec_kw={"width_ratios": [cal.shape[1] / cal.shape[0], run.shape[1] / run.shape[0]]})
    for ax, im, t in ((axs[0], cal, "(a) two-point calibration (web UI): 6 m reference, 8.721 px/m"),
                      (axs[1], run, "(b) calibrated run: per-track speed in the overlay")):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(t, pad=2); ax.axis("off")
    plt.subplots_adjust(wspace=0.03)
    fig.savefig(os.path.join(OUT, "fig_speed.jpg"), dpi=300, pil_kwargs={"quality": 90}); plt.close(fig)

# ── underlying plotting data (one CSV per chart) ────────────────────────────
DATA = os.path.join(OUT, "data"); os.makedirs(DATA, exist_ok=True)
with open(os.path.join(DATA, "fig_f1.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["group_type", "group", "class", "f1_mean", "f1_std", "n_videos"])
    for key, groups in (("category", ["Cloudy", "Night", "Rainy", "Sunny"]), ("tier", ["light", "medium", "heavy"])):
        for g in groups:
            for cls in ("car", "van", "bus"):
                v = [f(r[f"{cls}_f1"]) for r in seq if r[key] == g]
                w.writerow([key, g, cls, round(st.mean(v), 4), round(st.stdev(v), 4), len(v)])
with open(os.path.join(DATA, "fig_workloads.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["workload", "loop_ms_mean", "loop_ms_std", "io_ms_mean", "io_ms_std", "cpu_cores_mean", "cpu_cores_std"])
    for (k, _, _), a, b, c in zip(wl, loop, io, cpu):
        w.writerow([k, round(a[0], 4), round(a[1], 4), round(b[0], 4), round(b[1], 4), round(c[0], 4), round(c[1], 4)])
with open(os.path.join(DATA, "fig_ratios.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["video", "density", "S_amort_2TA_over_TC", "S_norm_2TA_over_TD", "S_same_TC_over_TD"])
    for i in order:
        w.writerow([sp[i]["video"], round(d[i], 4), round(amort[i], 4), round(norm[i], 4), round(same[i], 4)])
with open(os.path.join(DATA, "fig_amdahl.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["N", "S_broad", "S_narrow", "p_broad", "p_narrow", "measured_TC_over_TD_mean", "measured_std"])
    for n in range(1, 9):
        w.writerow([n, round(A_(pb, n), 4), round(A_(pn, n), 4), round(pb, 4), round(pn, 4), round(sm, 4), round(ss, 4)])

json.dump({"stage_frame_index": best, "hot_cells": int(hot.sum()), "video": "MVI_39761"},
          open(os.path.join(OUT, "figures_meta.json"), "w"), indent=2)
print("figures written; stage frame", best, "hot cells", int(hot.sum()))
