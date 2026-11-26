"""Fig. fig_grid.pdf for the grid-resolution subsection. Reads only
grid_resolution_analysis/final_numbers.json (and writes the plotted values
to figures/data/fig_grid.csv)."""
import csv, json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "paper", "figures")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 7.5,
    "axes.labelsize": 7.5, "legend.fontsize": 6.0, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False})
COL = 3.45
d = json.load(open(os.path.join(REPO, "grid_resolution_analysis", "final_numbers.json")))
G = ["6x6", "6x8", "8x8", "10x12"]; lab = [g.replace("x", "×") for g in G]
tc = d["test_classification"]; cn = d["counting"]
rows = []
for g in G:
    c = cn[f"original|test|{g}|original|all|Combined"]
    rows.append({"grid": g, "macro_f1_tuned": tc[f"{g}|frozen"]["macro_f1"], "macro_f1_original": tc[f"{g}|original"]["macro_f1"],
                 "all_car": tc[f"{g}|frozen"]["all_car_macro_f1"], "tp": c["tp"], "false": c["false"], "duplicate": c["duplicate"],
                 "gt_all": c["gt"], "gt_down": cn[f"original|test|{g}|original|down|Combined"]["gt"]})
with open(os.path.join(OUT, "data", "fig_grid.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 1.55), gridspec_kw={"wspace": 0.42})
x = range(len(G))
a.plot(x, [r["macro_f1_tuned"] for r in rows], "o-", ms=3, color="#4C72B0", label="tuned per grid")
a.plot(x, [r["macro_f1_original"] for r in rows], "s--", ms=3, color="#DD8452", label="original thresholds")
a.axhline(rows[0]["all_car"], color="#777777", lw=0.6, ls=":"); a.text(3.1, rows[0]["all_car"] + 0.006, "always-Car", fontsize=6, color="#555555", ha="right")
a.set_ylim(0.2, 0.47); a.set_ylabel("Macro-F1"); a.set_title("(a) Classification", pad=3)
a.legend(frameon=False, loc="lower right", handlelength=1.6)
bot = [0] * len(G)
for key, colr, name in (("tp", "#4C72B0", "correct"), ("false", "#DD8452", "false"), ("duplicate", "#CCB974", "duplicate")):
    v = [r[key] for r in rows]; b.bar(x, v, 0.6, bottom=bot, color=colr, edgecolor="white", lw=0.4, label=name)
    bot = [p + q for p, q in zip(bot, v)]
b.axhline(rows[0]["gt_all"], color="k", lw=0.6, label="GT, both dir.")
b.axhline(rows[0]["gt_down"], color="k", lw=0.6, ls="--", label="GT, downward")
b.set_ylim(0, 2700); b.set_ylabel("Counts (44 videos)"); b.set_title("(b) Counting", pad=3)
b.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.6, fontsize=5.5)
for ax in (a, b):
    ax.set_xticks(list(x)); ax.set_xticklabels(lab)
fig.savefig(os.path.join(OUT, "fig_grid.pdf")); fig.savefig(os.path.join(OUT, "fig_grid_preview.png"), dpi=250)
print(rows)
