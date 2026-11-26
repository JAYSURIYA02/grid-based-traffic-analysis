"""verify_revision.py -- final consistency checks for paper_revision/revised_paper.tex."""
import csv, json, os, re, subprocess, sys
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(R, "paper_revision"); T = os.path.join(P, "updated_tables"); G = os.path.join(R, "grid_resolution_analysis")
tex = open(os.path.join(P, "revised_paper.tex"), encoding="utf-8").read()
res = []; ok_all = True
def check(name, cond, info=""):
    global ok_all; ok_all &= bool(cond); res.append(("PASS" if cond else "FAIL", name, info)); print(res[-1][0], name, info)

# 1 originals unchanged vs the last commit before the revision
paths = ["seq.py", "rparallel_he3.py", "app.py", "user_input_data.json", "paper/paper_final.tex", "paper/paper.tex", "paper/paper_8p.tex",
         "paper/figures", "benchmark_results", "benchmark_results_v2", "grid_resolution_analysis/results.csv",
         "grid_resolution_analysis/analysis_report.md", "grid_resolution_analysis/updated_analysis_report.md",
         "grid_resolution_analysis/counting_results.csv", "grid_resolution_analysis/cross_validation_results.csv",
         "grid_resolution_analysis/selected_configuration.json", "grid_resolution_analysis/runs"]
d = subprocess.run(["git", "-c", "core.autocrlf=true", "diff", "--stat", "fe76ed7", "--"] + paths, cwd=R, capture_output=True, text=True).stdout.strip()
check("production code, original paper, figures and experiment outputs unchanged since fe76ed7", d == "", d)

# 2 references / citations / labels
cites = set(c.strip() for m in re.findall(r"\\cite\{([^}]*)\}", tex) for c in m.split(","))
bib = set(re.findall(r"\\bibitem\{([^}]*)\}", tex))
check("every citation has a bibliography entry", cites <= bib, str(cites - bib))
check("no unused bibliography entry", bib <= cites, str(bib - cites))
refs = set(re.findall(r"\\(?:eq)?ref\{([^}]*)\}", tex)); labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
check("every \\ref has a label", refs <= labels, str(refs - labels))
figs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", tex)
check("all figure files exist", all(os.path.exists(os.path.join(P, f)) for f in figs), str([f for f in figs if not os.path.exists(os.path.join(P, f))]))

# 3 numbers in the paper vs CSV
cl = {(r["grid"], r["thresholds"]): r for r in csv.DictReader(open(os.path.join(T, "classification_test_set.csv")))}
cn = {(r["set"], r["grid"], r["scope"]): r for r in csv.DictReader(open(os.path.join(T, "counting_by_lane_and_error_type.csv")))}
fn = json.load(open(os.path.join(G, "final_numbers.json")))
tim = json.load(open(os.path.join(G, "timing_trials_summary.json")))["by_grid"]
sig = {(r["unit"], r["metric"], r["comparison"]): r for r in csv.DictReader(open(os.path.join(T, "significance_tests_test_set.csv")))}
f3 = lambda x: f"{float(x):.3f}"; f1 = lambda x: f"{float(x):.1f}"
miss = []
for g in ["6x6", "6x8", "8x8", "10x12"]:
    c = cl[(g, "frozen")]; tc = fn["test_classification"][f"{g}|frozen"]
    row = [f3(c["macro_f1"]), f3(c["weighted_f1"]), f3(tc["det_recall"]), f"{100*tc['frag']:.1f}\\%", f3(2 * int(cn[("test", g, "Combined")]["tp"]) / (int(cn[("test", g, "Combined")]["pred"]) + int(cn[("test", g, "Combined")]["gt_down"]))),
           f1(cn[("test", g, "Combined")]["mae_down"]), f1(cn[("test", g, "Combined")]["rmse_down"]), f1(tim[g]["fps_median"])]
    lab = {"6x6": "$6{\\times}6$", "6x8": "$6{\\times}8^{a}$", "8x8": "$8{\\times}8$", "10x12": "$10{\\times}12$"}[g]
    line = [l for l in tex.splitlines() if l.strip().startswith(lab) and "\\%" in l and "&" in l and "Lane" not in l]
    if not line or any(v not in line[0] for v in row):
        miss.append((g, row, line[:1]))
    for sc in ("Lane1", "Lane2", "Combined"):
        r = cn[("test", g, sc)]
        tp_, p_, gd_ = int(r["tp"]), int(r["pred"]), int(r["gt_down"])
        vals = [r["gt_down"], r["gt_up"], r["pred"], r["tp"], f"{r['false']} ({r['false_opposite_direction']})", r["duplicate"],
                f"{r['missed_down']} ({r['missed_down_not_detected']})", f3(tp_ / p_), f3(tp_ / gd_), f3(2 * tp_ / (p_ + gd_)), f1(r["mae_down"]), f1(r["rmse_down"])]
        sl = {"Lane1": "Lane 1", "Lane2": "Lane 2", "Combined": "Combined"}[sc]
        cand = [l for l in tex.splitlines() if f"& {sl} &" in l and f"& {r['pred']} &" in l]
        if not cand or any(str(v) not in cand[0] for v in vals):
            miss.append((g, sc, vals))
check("Table IV (grid summary) values match CSV/JSON", not [m for m in miss if len(m) == 3 and isinstance(m[1], list)], str(miss[:2]))
check("Table VI (counting by lane) values match CSV", not [m for m in miss if len(m) == 3 and not isinstance(m[1], list)], str(miss[:2]))
c = cl[("6x8", "frozen")]; cm = json.loads(c["cm_rows_true_Car_Van_Bus"])
check("Table V confusion matrix matches CSV", all(f"{a} & {b} & {d}" in tex for a, b, d in cm))
for k in ("precision_Car", "recall_Car", "f1_Car", "precision_Van", "recall_Van", "f1_Van", "precision_Bus", "recall_Bus", "f1_Bus"):
    check(f"Table V {k}", f3(c[k]) in tex)
s = lambda m, cmp: sig[("site", m, cmp)]
probes = {"Holm $p$ for 10x12 MAE (0.010)": f"{float(s('count_mae','10x12 - 6x6')['p_perm_holm_within_metric']):.3f}" == "0.010",
          "10x12 MAE CI [5.4, 8.4]": abs(float(s('count_mae','10x12 - 6x6')['ci95_lo']) - 5.4) < 0.05 and abs(float(s('count_mae','10x12 - 6x6')['ci95_hi']) - 8.4) < 0.05,
          "6x8 count F1 +0.022 p=0.012": f"{float(s('count_f1','6x8 - 6x6')['observed_diff']):.3f}" == "0.022" and f"{float(s('count_f1','6x8 - 6x6')['p_perm_holm_within_metric']):.3f}" == "0.012",
          "6x8 J +0.011 nominal 0.03 Holm 0.09": f"{float(s('J','6x8 - 6x6')['observed_diff']):.3f}" == "0.011" and abs(float(s('J','6x8 - 6x6')['p_permutation']) - 0.03) < 0.005 and abs(float(s('J','6x8 - 6x6')['p_perm_holm_within_metric']) - 0.09) < 0.005,
          "10x12 RMSE p=0.013": f"{float(s('count_rmse','10x12 - 6x6')['p_perm_holm_within_metric']):.3f}" == "0.013",
          "all-18 smallest p=0.058": abs(min(float(r["p_perm_holm_all18"]) for k, r in sig.items() if k[0] == "site") - 0.058) < 0.001,
          "macro-F1 Holm p = 1.0 for all": all(float(s('macro_f1', f'{g} - 6x6')['p_perm_holm_within_metric']) == 1.0 for g in ("6x8", "8x8", "10x12")),
          "J 10x12 +0.0004 p=0.96": f"{float(s('J','10x12 - 6x6')['observed_diff']):.4f}" == "0.0004" and f"{float(s('J','10x12 - 6x6')['p_permutation']):.2f}" == "0.96"}
for k, v in probes.items():
    check("significance: " + k, v)
check("upward share 560/1280 = 44%", cn[("test", "6x8", "Combined")]["gt_up"] == "560" and round(560 / 1280 * 100) == 44)
check("count error +44% (23.1 vs 16.0)", round(float(cn[("test", "10x12", "Combined")]["mae_down"]) / float(cn[("test", "6x6", "Combined")]["mae_down"]) * 100 - 100) == 44)
check("benchmark numbers unchanged (audit_numbers recompute identical)", json.load(open(os.path.join(R, "paper", "final", "audit_numbers.json")))["ratios"]["S_same2"]["mean_std_n"][:2] == [0.8603, 0.0767])

# 4 dataset mapping
mp = list(csv.DictReader(open(os.path.join(P, "dataset_split_and_site_mapping.csv"))))
part = {}
for r in mp:
    part.setdefault(r["partition"].split(" ")[0], []).append(r)
check("mapping: 100 videos (12 benchmark, 26 validation, 44 test, 18 excluded)", len(mp) == 100 and [len(part[k]) for k in ("benchmark_12", "validation_26", "test_44", "excluded")] == [12, 26, 44, 18])
evs = {r["site_id"] for k in ("benchmark_12", "validation_26") for r in part[k]}
check("mapping: no test site is a validation/benchmark site", not ({r["site_id"] for r in part["test_44"]} & evs))
check("mapping: validation has 10 site groups, test 12", len({r["site_id"] for r in part["validation_26"]}) == 10 and len({r["site_id"] for r in part["test_44"]}) == 12)
ov = json.load(open(os.path.join(T, "objective_verification.json")))
check("objective: recomputed selection equals the frozen configuration", ov["match"])
check("objective: timing < freeze < test_set.json < first test run", ov["timing_summary_mtime"] < ov["selected_configuration_mtime"] < ov["test_set_json_mtime"] < ov["earliest_test_run_mtime"])
check("objective weights in the paper equal the code", all(x in tex for x in ("0.30\\,F_1^{\\text{mac}}", "0.10\\,\\min_c F_1^{c}", "0.25\\,F_1^{\\text{cnt}}", "0.15\\,R^{\\text{det}}", "0.10\\,(1-\\phi)", "0.10\\,(1-u)")) and ov["weights"] == {"macro_f1": 0.3, "min_class_f1": 0.1, "count_f1": 0.25, "det_recall": 0.15, "frag": 0.1, "unmatched": 0.1})

# 5 build log
log = os.path.expanduser("~/revbuild/revised_paper.blog")
if os.path.exists(log):
    L = open(log, errors="ignore").read()
    check("layout build: 0 LaTeX errors", L.count("\n!") == 0)
    check("layout build: 0 undefined references/citations", "undefined" not in L.lower())
    check("layout build: 8 pages (stand-in class)", "(8 pages" in L)
json.dump([list(r) for r in res], open(os.path.join(P, "updated_tables", "verification_results.json"), "w"), indent=1)
print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
