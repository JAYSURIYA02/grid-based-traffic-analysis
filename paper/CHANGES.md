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
