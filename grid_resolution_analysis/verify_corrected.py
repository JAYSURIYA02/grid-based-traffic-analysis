"""verify_corrected.py -- Part 10 checks for the corrected analysis. Prints PASS/FAIL per check."""
import csv, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
ok_all = True
def check(name, cond, info=""):
    global ok_all; ok_all &= bool(cond); print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""))

# 1. originals unchanged relative to the commit that produced them
orig = ["seq.py", "rparallel_he3.py", "app.py", "user_input_data.json",
        "grid_resolution_analysis/results.csv", "grid_resolution_analysis/analysis_report.md",
        "grid_resolution_analysis/summary.json", "grid_resolution_analysis/tuning_6x6.json",
        "grid_resolution_analysis/tuning_8x8.json", "grid_resolution_analysis/tuning_10x12.json",
        "grid_resolution_analysis/classification_comparison.png", "grid_resolution_analysis/macro_f1_comparison.png",
        "grid_resolution_analysis/performance_comparison.png", "grid_resolution_analysis/threshold_tuning_comparison.png"]
d = subprocess.run(["git", "-c", "core.autocrlf=true", "diff", "--stat", "b7c3756", "--"] + orig, cwd=REPO, capture_output=True, text=True).stdout.strip()
check("original code, config and grid-experiment outputs unchanged since b7c3756", d == "", d)
d2 = subprocess.run(["git", "-c", "core.autocrlf=true", "diff", "--stat", "HEAD", "--", "benchmark_results", "benchmark_results_v2"], cwd=REPO, capture_output=True, text=True).stdout.strip()
check("benchmark_results / benchmark_results_v2 unchanged", d2 == "", d2)

# 2. same protocol for every configuration
cfgs = set()
for root, _, files in os.walk(os.path.join(HERE, "runs")):
    if "user_input_data.json" in files and "meta.json" in files:
        u = json.load(open(os.path.join(root, "user_input_data.json"))); m = json.load(open(os.path.join(root, "meta.json")))
        cfgs.add((u["color_channel"], json.dumps({k: v["roi"] for k, v in m["lanes"].items()}, sort_keys=True), m.get("variant", "original")))
check("all baseline runs share colour channel, ROIs and clustering", len(cfgs) == 1, str(cfgs))

# 3. CV never tunes on its validation fold
site = {v: k for k, vs in json.load(open(os.path.join(HERE, "site_groups_manual.json")))["groups"].items() for v in vs}
rows = list(csv.DictReader(open(os.path.join(HERE, "cross_validation_results.csv"))))
vids = sorted(os.listdir(os.path.join(HERE, "runs", "validation", "6x6")))
leak = []
for r in rows:
    held = set(r["held_out_videos"].split()); train = [v for v in vids if site[v] != r["held_out_site"]]
    if held & set(train) or int(r["n_train"]) != len(train) or any(site[v] != r["held_out_site"] for v in held):
        leak.append(r["fold"])
check("CV: held-out site group never in its training set; folds are whole site groups", not leak, str(leak))
check("CV: every validation video held out exactly once", sorted(v for r in rows if r["grid"] == "6x6" for v in r["held_out_videos"].split()) == vids)
bench = {"MVI_39761","MVI_40774","MVI_63525","MVI_20033","MVI_39771","MVI_40773","MVI_40141","MVI_20011","MVI_40963","MVI_40772","MVI_40131","MVI_39311"}
test = set(json.load(open(os.path.join(HERE, "test_set.json")))["videos"])
check("validation, benchmark and test video sets are disjoint", not (set(vids) & bench) and not (set(vids) & test) and not (bench & test))
check("no test video shares a site group with validation/benchmark", not (test & set(site)))
sel = json.load(open(os.path.join(HERE, "selected_configuration.json")))
check("selected_configuration.json frozen before test_set.json was written",
      os.path.getmtime(os.path.join(HERE, "selected_configuration.json")) < os.path.getmtime(os.path.join(HERE, "test_set.json")))

# 4. counting: replay verified, no double counting
cr = list(csv.DictReader(open(os.path.join(HERE, "counting_results.csv"))))
bad = [r for r in cr if int(r["tp"]) + int(r["duplicate"]) + int(r["false"]) != int(r["pred"]) or int(r["tp"]) + int(r["missed"]) != int(r["gt"])]
check("counting: tp+dup+false = pred and tp+missed = gt for every row", not bad, str(len(bad)))
comb = {}
for r in cr:
    comb.setdefault((r["variant"], r["set"], r["grid"], r["thresholds"], r["video"], r["gt_def"]), {})[r["scope"]] = r
bad2 = [k for k, v in comb.items() if int(v["Combined"]["tp"]) > int(v["Combined"]["gt"])
        or int(v["Combined"]["pred"]) != int(v["Lane1"]["pred"]) + int(v["Lane2"]["pred"])
        or int(v["Combined"]["gt"]) != int(v["Lane1"]["gt"]) + int(v["Lane2"]["gt"])]
check("combined lanes: each GT event matched at most once; combined = Lane1 + Lane2 events", not bad2, str(bad2[:3]))
fl = open(os.path.join(HERE, "failures.log")).read() if os.path.exists(os.path.join(HERE, "failures.log")) else ""
check("no logged run or replay failures", fl.strip() == "", fl[:300])

# 5. report numbers match CSV / JSON
rep = open(os.path.join(HERE, "updated_analysis_report.md"), encoding="utf-8").read()
fn = json.load(open(os.path.join(HERE, "final_numbers.json")))
probes = {f"{fn['test_classification']['6x8|frozen']['macro_f1']:.3f}": "test 6x8 frozen macro-F1",
          f"{fn['test_classification']['10x12|frozen']['macro_f1']:.3f}": "test 10x12 frozen macro-F1",
          f"{fn['counting']['original|test|10x12|original|down|Combined']['MAE']:.2f}": "test 10x12 MAE",
          f"{fn['counting']['original|test|6x8|original|down|Combined']['event_f1']:.3f}": "test 6x8 count F1",
          sel["selected_grid"]: "selected grid"}
miss = [v for k, v in probes.items() if k not in rep]
check("report contains the CSV/JSON values it cites (spot check)", not miss, str(miss))
print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
