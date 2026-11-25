"""
audit_numbers.py -- recompute every number quoted in paper/final/paper_final.tex
from the repository's benchmark artifacts. Run from the repository root:

    python paper/final/audit_numbers.py

Writes paper/final/audit_numbers.json (machine-readable) and prints a report.
Aggregation follows the paper: metrics are computed per video, then
summarized as mean and sample standard deviation over videos.
"""
import csv, json, os, statistics as st, pstats

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = os.path.join(R, "benchmark_results_v2"); V1 = os.path.join(R, "benchmark_results")
f = lambda x: float(x)
def ms(v): v = [x for x in v]; return [round(st.mean(v), 4), round(st.stdev(v), 4), len(v)]
out = {}
rows = list(csv.DictReader(open(os.path.join(V2, "consolidated_results.csv"))))
sp = list(csv.DictReader(open(os.path.join(V2, "speedup_by_video.csv"))))
eq = list(csv.DictReader(open(os.path.join(V2, "equivalence.csv"))))
amd = list(csv.DictReader(open(os.path.join(V2, "amdahl_by_video.csv"))))
sysinfo = json.load(open(os.path.join(V2, "system_info.json")))
out["system"] = {k: sysinfo[k] for k in ("cpu_model", "physical_cores", "logical_processors", "platform", "python_version", "opencv_version", "numpy_version")}

# workloads
wl = {"A": ("sequential", "Lane1(ROI1)"), "B": ("parallel_1lane", "Lane1"), "C": ("serial_2lane", "Lane1"), "D": ("parallel", "Lane1")}
out["workloads"] = {}
for k, (impl, lane) in wl.items():
    s = [r for r in rows if r["implementation"] == impl and r["lane"] == lane]
    out["workloads"][k] = {m: ms([f(r[m]) for r in s]) for m in ("mean_frame_time_ms", "mean_frame_io_ms", "fps", "peak_memory_mb", "cpu_utilization_of_one_core")}
out["peak_mem_delta_D_minus_A_mb"] = round(out["workloads"]["D"]["peak_memory_mb"][0] - out["workloads"]["A"]["peak_memory_mb"][0], 2)
out["peak_mem_delta_pct"] = round(100 * out["peak_mem_delta_D_minus_A_mb"] / out["workloads"]["A"]["peak_memory_mb"][0], 2)

# ratios
def col(name): return [f(r[name]) for r in sp]
ratios = {
    "S_naive": col("naive_speedup"), "S_norm": col("normalized_speedup"),
    "S_norm_pf": col("normalized_speedup_per_frame"),
    "S_amort": [2 * f(r["T_A_s"]) / f(r["T_C_s"]) for r in sp],
    "S_amort_pf": [2 * f(r["frame_ms_A"]) / f(r["frame_ms_C"]) for r in sp],
    "S_same2": col("same_workload_speedup_2lane"), "S_same2_pf": col("same_workload_speedup_2lane_per_frame"),
    "S_same1": col("same_workload_speedup_1lane"), "S_same1_pf": col("same_workload_speedup_1lane_per_frame"),
}
out["ratios"] = {k: {"mean_std_n": ms(v), "min": round(min(v), 3), "max": round(max(v), 3), "n_above_1": sum(x > 1 for x in v)} for k, v in ratios.items()}
s2 = out["ratios"]["S_same2"]["mean_std_n"][0]
out["efficiency_same2"] = {n: round(s2 / n, 3) for n in (2, 4, 8)}

# density relation
dens = [f(r["density"]) for r in sp]
def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
out["pearson_density_Snorm"] = round(pearson(dens, ratios["S_norm"]), 3)
out["Snorm_by_tier"] = {t: ms([f(r["normalized_speedup"]) for r in sp if r["tier"] == t]) for t in ("light", "medium", "heavy")}
out["density_range"] = [round(min(dens), 2), round(max(dens), 2)]
out["seq_loop_ms_by_tier"] = {t: round(st.mean(f(r["mean_frame_time_ms"]) for r in rows if r["implementation"] == "sequential" and r["tier"] == t), 2) for t in ("light", "medium", "heavy")}
out["frames_total"] = sum(int(r["frames"]) for r in rows if r["implementation"] == "sequential")

# tracking share from profiles
sel = json.load(open(os.path.join(V2, "selection_report.json")))["selection"]
share = []
for e in sel:
    stem = e["video"].replace("\\", "/").split("/")[-1][:-4]
    s = pstats.Stats(os.path.join(V2, e["category"], stem, "sequential", "profiled", "profile_seq.prof"))
    share.append((e["density"], 100 * sum(v[3] for k, v in s.stats.items() if k[2] == "process_tracking_counting") / s.total_tt))
share.sort()
out["tracking_share_pct"] = {"min": round(min(x[1] for x in share), 1), "max": round(max(x[1] for x in share), 1), "lightest_video": round(share[0][1], 1), "densest_video": round(share[-1][1], 1)}

# Amdahl, MVI_39311
s = pstats.Stats(os.path.join(V2, "Sunny", "MVI_39311", "sequential", "profiled", "profile_seq.prof")); T = s.total_tt
nat7 = ["dilate", "GaussianBlur", "'read' of 'cv2.VideoCapture", "cvtColor", "absdiff", "split", "'write' of 'cv2.VideoWriter"]
native = sum(v[2] for k, v in s.stats.items() if k[0] == "~" and any(n in k[2] for n in nat7))
narrow = sum(v[3] for k, v in s.stats.items() if k[2] in ("process_grid_channel", "apply_histogram_equalization") and k[0].endswith("seq.py"))
dil = [v for k, v in s.stats.items() if k[0] == "~" and "dilate" in k[2]][0]
A_ = lambda p, n: 1 / ((1 - p) + p / n)
pn, pb = narrow / T, native / T
out["amdahl_MVI_39311"] = {"total_s": round(T, 2), "native7_s": round(native, 2), "native7_pct": round(100 * pb, 1),
    "p_narrow": round(pn, 3), "p_broad": round(pb, 3),
    "S_narrow": {n: round(A_(pn, n), 2) for n in (2, 4, 8)}, "S_broad": {n: round(A_(pb, n), 2) for n in (2, 4, 8)},
    "S_narrow_inf": round(1 / (1 - pn), 2), "S_broad_inf": round(1 / (1 - pb), 2),
    "dilate_us_per_call": round(1e6 * dil[2] / dil[1], 1)}
out["p_narrow_12"] = ms([f(r["p_narrow"]) for r in amd]); out["p_broad_12"] = ms([f(r["p_broad"]) for r in amd])
out["p_narrow_12_range"] = [round(min(f(r["p_narrow"]) for r in amd), 3), round(max(f(r["p_narrow"]) for r in amd), 3)]

# equivalence
out["equivalence"] = {"identical": sum(r["identical"] == "True" for r in eq), "total": len(eq),
                      "checks": sorted(set(r["check"] for r in eq))}

# accuracy (sequential == parallel Lane 1)
seq = [r for r in rows if r["implementation"] == "sequential"]
out["per_class_mean_over_videos"] = {c: {m: ms([f(r[f"{c.lower()}_{m}"]) for r in seq]) for m in ("precision", "recall", "f1")} for c in ("Car", "Van", "Bus")}
out["videos_with_zero_f1"] = {c: sum(f(r[f"{c.lower()}_f1"]) == 0 for r in seq) for c in ("Car", "Van", "Bus")}
groups = {g: [r for r in seq if r["category"] == g] for g in ("Cloudy", "Night", "Rainy", "Sunny")}
groups.update({t: [r for r in seq if r["tier"] == t] for t in ("light", "medium", "heavy")})
out["bus_f1_zero_groups"] = [g for g, s in groups.items() if all(f(r["bus_f1"]) == 0 for r in s)]
# pooled ground-truth vehicle counts for context
out["unique_gt_vehicles_total"] = sum(int(r["unique_gt_vehicles"]) for r in seq)

# v1 consistency
v1 = list(csv.DictReader(open(os.path.join(V1, "speedup_by_video.csv"))))
out["v1_S_norm"] = ms([f(r["normalized_speedup"]) for r in v1])
v1rows = list(csv.DictReader(open(os.path.join(V1, "consolidated_results.csv"))))
out["v1_seq_loop_ms"] = ms([f(r["mean_frame_time_ms"]) for r in v1rows if r["implementation"] == "sequential"])
v1f1 = {r["video"]: (r["car_f1"], r["van_f1"], r["bus_f1"]) for r in v1rows if r["implementation"] == "sequential"}
out["v1_v2_f1_identical"] = all(v1f1[r["video"]] == (r["car_f1"], r["van_f1"], r["bus_f1"]) for r in seq)

# pooled class composition (context for the per-class table)
import glob
pres = {"Car": 0, "Van": 0, "Bus": 0}; gt = dict(pres); pr = dict(pres)
for pth in glob.glob(os.path.join(V2, "*", "*", "sequential", "clean", "eval_metrics.json")):
    e = json.load(open(pth)); cm = e["confusion_matrix_vehicle"]
    for i, c in enumerate(e["confusion_matrix_vehicle_labels"]):
        rsum = sum(cm[i]); gt[c] += rsum; pr[c] += sum(r[i] for r in cm); pres[c] += rsum > 0
out["videos_with_class_in_matched_gt"] = pres
out["matched_gt_per_class_pooled"] = gt
out["predicted_per_class_pooled"] = pr

json.dump(out, open(os.path.join(R, "paper", "final", "audit_numbers.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
