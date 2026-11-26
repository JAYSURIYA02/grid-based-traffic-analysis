"""Grid-resolution figure for the revised paper (fig_grid.pdf), from
paper_revision/updated_tables/{classification_test_set,counting_by_lane_and_error_type}.csv.
Plotted values are also written to figures/data/fig_grid.csv."""
import csv, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = os.path.join(REPO, "paper_revision", "updated_tables"); OUT = os.path.join(REPO, "paper_revision", "figures")
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02, "axes.spines.top": False, "axes.spines.right": False})
G = ["6x6", "6x8", "8x8", "10x12"]; L = [g.replace("x", "×") for g in G]; x = range(4)
cl = {(r["grid"], r["thresholds"]): r for r in csv.DictReader(open(os.path.join(T, "classification_test_set.csv")))}
cn = {(r["set"], r["grid"], r["scope"]): r for r in csv.DictReader(open(os.path.join(T, "counting_by_lane_and_error_type.csv")))}
f = lambda r, k: float(r[k])
C1, C2, C3, C4, GR = "#1f5fa8", "#d1652a", "#2a9d6f", "#b8952a", "#666666"
fig, ax = plt.subplots(2, 2, figsize=(3.45, 2.75), gridspec_kw={"hspace": 0.62, "wspace": 0.45})
a = ax[0, 0]
a.plot(x, [f(cl[(g, "frozen")], "macro_f1") for g in G], "o-", ms=3.5, color=C1, label="tuned")
a.plot(x, [f(cl[(g, "original")], "macro_f1") for g in G], "s--", ms=3.5, color=C2, label="original thr.")
a.plot(x, [f(cl[(g, "frozen")], "always_car_macro_f1") for g in G], ":", color=GR, label="always-Car")
a.set_ylim(0.25, 0.47); a.set_title("(a) Macro-F1", pad=2)
a = ax[0, 1]
a.plot(x, [f(cl[(g, "frozen")], "weighted_f1") for g in G], "o-", ms=3.5, color=C1, label="tuned")
a.plot(x, [f(cl[(g, "original")], "weighted_f1") for g in G], "s--", ms=3.5, color=C2, label="original thr.")
a.plot(x, [f(cl[(g, "frozen")], "always_car_weighted_f1") for g in G], ":", color=GR, label="always-Car")
a.set_ylim(0.45, 0.9); a.set_title("(b) Weighted-F1", pad=2)
h, l = a.get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04), handlelength=2.0)
a = ax[1, 0]; bot = [0] * 4
segs = [("tp", "correct", C1), ("false_opp", "false: opposite dir.", C4), ("false_other", "false: other", C2), ("duplicate", "duplicate", "#9a9a9a")]
rows = []
for g in G:
    r = cn[("test", g, "Combined")]
    rows.append({"grid": g, "tp": int(r["tp"]), "false_opp": int(r["false_opposite_direction"]), "false_other": int(r["false"]) - int(r["false_opposite_direction"]),
                 "duplicate": int(r["duplicate"]), "gt_down": int(r["gt_down"]), "gt_all": int(r["gt_down"]) + int(r["gt_up"]),
                 "mae": f(r, "mae_down"), "rmse": f(r, "rmse_down"), "f1_l1": f(cn[("test", g, "Lane1")], "f1_down"), "f1_l2": f(cn[("test", g, "Lane2")], "f1_down"),
                 "f1_comb": f(r, "f1_down"),
                 **{k: cl[(g, t)][m] for t in ("frozen", "original") for m in ("macro_f1", "weighted_f1") for k in [f"{m}_{t}"]},
                 "always_car_macro_f1": cl[(g, "frozen")]["always_car_macro_f1"], "always_car_weighted_f1": cl[(g, "frozen")]["always_car_weighted_f1"]})
for key, name, colr in segs:
    v = [r[key] for r in rows]; a.bar(x, v, 0.62, bottom=bot, color=colr, edgecolor="white", lw=0.4, label=name); bot = [p + q for p, q in zip(bot, v)]
a.axhline(rows[0]["gt_all"], color="k", lw=0.7, label="GT, both dir.")
a.axhline(rows[0]["gt_down"], color="k", lw=0.7, ls="--", label="GT, downward")
a.set_ylim(0, 2750); a.set_ylabel("events (44 videos)"); a.set_title("(c) Predicted crossings", pad=2)
a.legend(frameon=False, loc="upper left", ncol=2, fontsize=5.4, handlelength=1.0, borderpad=0.1, labelspacing=0.15, columnspacing=0.5, bbox_to_anchor=(-0.02, 1.03))
a = ax[1, 1]
a.plot(x, [r["f1_l1"] for r in rows], "o-", ms=3.5, color=C1, label="Lane 1")
a.plot(x, [r["f1_l2"] for r in rows], "^-", ms=3.5, color=C2, label="Lane 2")
a.plot(x, [r["f1_comb"] for r in rows], "s--", ms=3.5, color=C3, label="combined")
a.set_ylim(0, 0.5); a.set_title("(d) Counting event-F1", pad=2); a.legend(frameon=False, loc="lower center", ncol=3, fontsize=6, handlelength=1.4, columnspacing=0.6, borderpad=0.1)
for a in ax.ravel():
    a.set_xticks(list(x)); a.set_xticklabels(L); a.grid(axis="y", lw=0.3, alpha=0.6); a.set_axisbelow(True)
fig.savefig(os.path.join(OUT, "fig_grid.pdf")); fig.savefig(os.path.join(OUT, "fig_grid_preview.png"), dpi=250)
with open(os.path.join(OUT, "data", "fig_grid.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("ok")
