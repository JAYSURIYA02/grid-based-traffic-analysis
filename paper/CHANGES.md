# paper.tex — changes from the previous draft (2026-11-25)

`paper_previous.tex` is the draft as submitted for review; `paper.tex` is the
corrected version. Every number below was recomputed from
`benchmark_results/` (the 12-video Windows run) unless marked **TBD**.

## Corrections of fact
| Where | Before | After | Evidence |
|---|---|---|---|
| IV-A | Ryzen 5 7235HS "6 cores / 12 threads" | 4 cores / 8 threads; `_CELL_POOL` = 32 workers | `Win32_Processor` on the benchmark machine; AMD spec |
| Table V | seq 7.99±0.24 ms, 92.5±4.6 FPS; par 14.91±0.86 ms, 58.2±2.9 FPS | seq 8.03±0.29 ms, 92.4±4.5 FPS; par 14.92±0.84 ms, 58.2±3.5 FPS; peak memory column (par 313.0±3.3 MB, seq not recorded) + footnote on the two time bases | `consolidated_results.csv`, per-run JSON |
| Table VIII (Amdahl) | S(N=6) column | S(8), S(4), S(2); broad row renamed "7 native calls"; both measured figures | 4C/8T; arithmetic |
| V-C | `findContours` "below the top-15 cutoff" | it is #9 (0.45 s); remainder itemised (1.71 s top-15 + ≈1.38 s below cutoff) | `profile_seq_tottime.txt` (MVI_39311) |
| Conclusion | Bus F1 = 0 in "three of the seven" groupings | four (Night, Sunny, Light, Medium) | Tables III–IV |
| III-E5, V-F | seq.py speed division "divide-by-zero"; default 0 in `user_input_data.json` | yields `inf` (NumPy); default in `user_input_data.example.json`; guard now in seq.py | code, commit a59f530 |
| III-F, Table II | flow normalised by "8 vehicles/s" | 8 crossings per 2-s window; 0.30/0.60 warm-up thresholds; 2-s track eviction | `seq.py` |
| III-B | min. area "unchanged" | ≥ → > | baseline commit 3d2d1a7 |
| V-F | "60/100/120-level thresholds" | 60 and 100 (120 is dead code) | `seq.py` |
| IV-D, V-E, V-F | "both scripts export memory", schema "mirrors exactly" | v1: only parallel memory; asymmetric timers; both fixed in shipped scripts | per-run JSON; commit 888e3ae |

## Claims removed or reframed
- **Density mechanism** (abstract, V-B, V-E, conclusion): the claim that a
  growing pure-Python tracking stage lowers speedup at high density is
  removed. Profiles show `process_tracking_counting` at 1.8–3.8 % of run time
  on every video with no density trend; r(density, S_norm) = −0.46 is
  reported as an observation; tiers are within-category.
- **Speedup reporting**: three figures throughout — S_naive 0.631±0.042,
  S_norm 1.261±0.084 (throughput ratio, end-to-end), S_norm,pf 1.079±0.064;
  explicit warning that S_norm includes amortisation of shared per-frame
  work and is not a threading speedup.
- **Amdahl**: compared at N=2 (broad 1.50 vs narrow 1.24 vs measured 1.26 /
  1.08), wording toned down ("consistent with", not "evidence"); 12-video
  p range added (0.378±0.017).
- **~3.3×**: now "an earlier, unarchived measurement", not used as a result.
- **4/12, 5/12 pre-fix match rates**: labelled as from working notes.
- E(N) now reported (0.63 / 0.32 / 0.16 for N = 2 / 4 / 8, indicative).
- Same-workload baseline: described as implemented (`BTP_EXECUTOR=serial`,
  `BTP_LANES=1`) and verified equivalent; **timings TBD** (Table VII) until
  the orchestrator is re-run on the benchmark machine.
- Script attribution: `analyze_results.py` → `orchestrate_benchmark.py`.

## Added
- IV-B: 960×540, 25 fps, processed frame counts (14 356 total), candidate
  pool sizes, "middle-ranked" selection, within-category tiers.
- IV-D: timing-artifact paragraph (`waitKey` 43.0 s / `imwrite` 7.6 s of
  83.5 s in the archived pre-fix profile) and instrumentation limitations.
- III-H: four evaluation details (t+1 GT, "others"→Car, ignored regions not
  masked, greedy matching).
- V-D: extended equivalence (B, C, D vs A; Lane 2 C vs D) — 48/48 identical
  in `verification/vm_linux_2vcpu_20261125/`.
- Section VI Reproducibility.
- Fig. 2 redrawn: no arrow crossings; HE submitted from the main thread to
  `_CELL_POOL`; serial full-frame preprocessing shown.

## Cross-references
Fixed `\ref{…}X` concatenations that rendered as III-BC, III-EE, III-GE,
III-GB, III-EC (new labels `sec:cellcontour`, `sec:speed`, `sec:dupsupp`,
`sec:pools`, `sec:speedup-limits`); `sec:discussion` (which pointed at all
of Section V) replaced by `sec:crosscheck` (V-D); wrong targets
("timing fix (Section IV)", "Section IV recommends averaging") fixed.
Compiled twice with an IEEEtran-numbering stand-in: 0 undefined
references, no "??".

## References
- [1] issue number added (no. 1). [3] vol. 30 added.
- [2] venue placeholder replaced by "unpublished manuscript, 2026" + **TBD**
  — the authors must confirm status/venue; the quoted sentence attributed
  to it must be checked against the manuscript.
- [7] URL/title updated to `c-api/threads.html`, "Thread states and the
  global interpreter lock".
- Uncitable "project report, Section 16.3" removed.

## Remaining TBD markers (search for `\TBD`)
Table VII (four-workload timings, S_same) and reference [2].

---

# Revision 2 (2026-11-25, afternoon) — v2 benchmark, citations, reference [2]

## New data: `benchmark_results_v2/` (run by the author on the benchmark machine)
Windows 11 (26300), AMD Ryzen 5 7235HS 4C/8T, Python 3.12.3, OpenCV 4.12.0,
NumPy 1.26.4 (recorded in `system_info.json`). 12 videos × 4 workloads,
0 failed runs, equivalence 60/60 identical, per-video F1 bit-identical to v1.

| Quantity | v2 (now in paper) |
|---|---|
| S_naive = T_A/T_D | 0.628 ± 0.114 |
| S_norm = 2T_A/T_D | 1.257 ± 0.228 (> 1 on 10/12) |
| S_norm,pf | 1.097 ± 0.187 |
| S_amort = 2T_A/T_C (two lanes, no threads) | 1.458 ± 0.216 |
| S_same(2 lanes) = T_C/T_D | **0.860 ± 0.077 (< 1 on 12/12)** |
| S_same(1 lane) = T_A/T_B | 0.900 ± 0.123 |
| E(2/4/8) = S_same/N | 0.43 / 0.22 / 0.11 |
| CPU (process CPU ÷ wall), seq.py | 4.20 ± 0.61 cores |
| Peak memory A / D | 302.5 / 310.0 MB (+2.5 %) |
| r(density, S_norm) | −0.45 |
| p_narrow / p_broad (MVI_39311) | 0.362 / 0.666; 12-video 0.375±0.019 / 0.667±0.011 |
| tracking share | 1.6–4.2 % |

## Consequences for the paper
- **Central claim changed**: threading does not reduce execution time on
  this machine; the normalized gain comes from sharing per-frame work
  between lanes (S_amort > S_norm). Abstract, V-B, V-C, new V-E
  ("Why threading does not help here"), conclusion rewritten.
- The earlier agreement between the narrow Amdahl ceiling (1.24) and
  S_norm (1.26) is now stated to be coincidental.
- Table V = four workloads (loop, I/O, FPS, peak memory, CPU); former
  TBD same-workload table merged into it; Table VI adds S_same column;
  Table VII (Amdahl) uses the v2 profile and measured S_same.
- v1 run summarized in IV-D ("Earlier benchmark run") for consistency.
- Threats: run-to-run variability documented (seq loop 8.03±0.29 ms v1 vs
  9.41±1.84 ms v2); spreads labelled as between-video variation.

## Baseline description corrected against the TPL2 paper
The TPL2 paper describes a lattice-run counting heuristic (Eqs. 7–8) and
count-based density, leaves tracking-based speed estimation to future
work, and its Table 1 reports only execution time and memory (26.55 %
efficiency gain, 3.72 % memory increase). Our paper previously said the
baseline has no counting/density at all; it now distinguishes the
baseline *paper* from the released baseline *code* (Intro, Table I,
III-A, III-C, III-E, III-F). "A recreated pool cannot hold state" narrowed
to "the baseline stores no persistent per-lane tracking state".

## Citations added (20, all DOIs verified against Crossref)
crouzil2016, lian2025, liu2025, zhai2023, chang2025, macko2025,
unzueta2012, abbas2021, chen2020, bewley2016, wojke2017, kamil2024,
sangsuwan2024, alhayanni2020, poolla2023, beyer2019, kalibera2013
(+ DOIs added to suzuki1985, wen2020, baseline).
Corrections vs. the Consensus list: Unzueta → 2012 (vol. 13 no. 2);
Poolla & Saxena → 2023 (vol. 96); Beyer et al. → 2019, vol. 21 no. 1;
Al-hayanni et al. → no. 4; "Chen and Wu-Sheng" → Y. Chen and W. Hu;
"Suzuki and Be" → Suzuki and Abe (Crossref record typo); Abbas et al.
page given as 838 only (end page not confirmed).

## Reference [2]
Now cited as an IEEE 2026 conference paper with ISBN 979-8-3195-2457-7
(from the PDF eXpress-certified camera-ready, 12 Aug 2026). The
conference name is not printed in the PDF and could not be found online:
**one \TBD remains — insert the conference name.** The quoted sentence
("reports execution time and memory consumption but does not provide
detection-accuracy metrics") is verbatim in that paper.

## Also addressed from the citation report
- "Identical accuracy proves equivalence" → "matching confusion matrices
  and F1 on all 12 tested videos … not a proof for every input".
- Speed estimation explicitly stated as not validated.
- Code-review findings consistently "all fixed".
