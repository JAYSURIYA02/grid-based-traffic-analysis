# Technical audit of the research paper (before revision)

**Audited version:** `paper/paper_final.tex` at commit fe76ed7, titled *"Lightweight Vehicle Classification, Tracking and Counting Using Grid-Based Motion Detection and Parallel Lane Processing"*. Its compiled layout preview was `paper/final/paper_final_layout_preview.pdf`.

**Sources checked against:**
- the production code: `seq.py`, `rparallel_he3.py`, `app.py`, `static/js/app.js`, `calibrate_pixels_per_meter.py`;
- the benchmark data: `benchmark_results_v2/`, `paper/final/audit_numbers.{py,json}` (recomputed: identical);
- the grid-resolution analysis: `grid_resolution_analysis/` (runs, cross-validation outputs, counting CSV, timing trials, site grouping, reports);
- the bibliography (Crossref, 26 Nov 2026).

This audit was completed before any scientific claim was rewritten.

## 1. Current paper structure

| Section | Content | Figures / tables |
|---|---|---|
| Abstract, Index Terms | | |
| I Introduction | Context, TPL2 baseline, 6 contributions | Table I (baseline vs. this paper) |
| II Related Work | Classical pipelines, [2], SORT/DeepSORT, speed, GIL, Amdahl | Eq. (1) Amdahl |
| III Methodology | Pipeline, motion detection, clustering and classification, tracking/counting/density, speed estimation, parallel implementation, evaluation | Fig. 1 ROI frame; Fig. 2 pipeline; Fig. 3 stages; Table II parameters; Eq. (2) speed; Fig. 4 speed (UI + overlay with speeds); Fig. 5 thread pools; Fig. 6 overlay output |
| IV Experimental Setup | Platform, 12 benchmark videos, four workloads | |
| V-A Accuracy and equivalence | 60/60 equivalence; 12-video per-class metrics | Fig. 7 F1 bars; Table III P/R/F1 |
| V-B Grid resolution | Site-grouped CV, 44 unseen-site test videos, classification, counting, fragmentation, selection | Fig. 8; Table IV |
| V-C Timing | Four workloads, ratios | Table V; Figs. 9–10 |
| V-D Profiling / Amdahl | | Fig. 11 |
| V-E Why threading does not help | | |
| V-F Threats to validity | | |
| VI Conclusion, References | 25 references | |

## 2. Main scientific contributions

1. A classification, tracking, counting and density layer, with an unvalidated speed estimator, on the TPL2 lattice front end. It is implemented identically in a sequential and a two-lane parallel implementation.
2. Persistent thread pools, per-lane state and a compute/draw split.
3. A cross-implementation check that found and fixed two preprocessing divergences.
4. A four-workload benchmark that separates lane count from threading.
5. A grid-resolution study: site-grouped CV, trajectory counting ground truth, and an unseen-site test set.

## 3. Claims supported by the implementation (verified in code)

| Claim | Evidence |
|---|---|
| Per-cell blur 5×5/σ=1, threshold 60, (cell_w, cell_h/2) dilation ×5, RETR_EXTERNAL, area > 100 px² | `rparallel_he3.py` `process_grid_channel` L144–151, `process_grid_cell` |
| 4-connected flood fill; span-based Bus/Van/Car rule (0.75 / 0.55 / 0.45) | `group_cells`, `cluster_to_object`, `classify_by_grid` |
| Label vote (cy/H)(1.15)^age; label scored after 5 frames; duplicate suppression IoU > 0.40 | `compute_tracking` STEP 4, `suppress_duplicate_tracks` |
| Counting: centroid moves from y < line to y ≥ line, age ≥ 2, 30-frame / 60 px cooldown; **downward only, in both lanes; no cross-lane de-duplication** | `compute_tracking` STEP 5 (L781–797) |
| Persistent `_CELL_POOL` / `_LANE_POOL`, per-lane `LaneTracker` | `rparallel_he3.py` |
| Speed: k = d/L (UI converts display to native coordinates), 3-cell-height displacement gate, v = (Δd/k)/(Δn/f)·3.6, mean of last 15 samples, magnitude only, no outlier rejection, f = `CAP_PROP_FPS`, disabled when k = 0 | `app.py` `/calibrate`; `static/js/app.js`; `seq.py` L571–598; `rparallel_he3.py` L660–695 |

## 4. Claims supported by experiments

| Claim | Evidence |
|---|---|
| 60/60 equivalence checks identical after the two fixes | `benchmark_results_v2/equivalence.csv` |
| Table V timing (A 9.41 ms / 81.1 FPS … D 17.46 ms / 50.1 FPS), CPU, memory | `audit_numbers.json` (recomputed from `benchmark_results_v2`; identical) |
| Ratios: S_same(2) = 0.860 ± 0.077, below 1 on 12/12 videos; S_norm = 1.257; S_amort = 1.458 | `audit_numbers.json` → `ratios` |
| Grid CV / test numbers (macro-F1, counts, MAE, fragmentation, FPS) | `grid_resolution_analysis/final_numbers.json`, `counting_results.csv`, `timing_trials_summary.json` |
| Fragmentation diagnosis (bridge −87–89 %, 8-connectivity −35–40 %) | `fragmentation_validation.csv` |

## 5. Overstated or insufficiently supported claims

1. **Significance of the 6×8 objective gain.** The old text said 6×8 "was the only grid whose objective improved significantly on 6×6".
   - That rested on a session-level percentile bootstrap without multiplicity correction.
   - With the camera site as the resampling unit and a paired permutation test, the gain is nominal p = 0.03, Holm-adjusted p = 0.09: **not significant**.
2. **Statistical claims did not name their metric.** "No grid differed significantly" (macro-F1) and "objective improved significantly" (J) referred to different metrics without saying so.
   - The review's statement that "10×12 significantly improves the combined objective" does **not** occur in the paper and is not supported. The 10×12 − 6×6 difference in J is +0.0004 (p = 0.96).
3. **"All tuned classifiers exceeded the always-Car baseline (0.40–0.43 vs 0.31)."** This is true for macro-F1 only:
   - weighted-F1 0.78–0.82 vs always-Car 0.80–0.81 (CI of the difference includes 0);
   - accuracy 0.77–0.84 vs 0.87.

   The claim is misleading without this context.
4. **The selection objective was not specified.** The paper listed the ingredients, but not the equation, weights, normalisation, guard or tie rule. A reader could not reproduce the choice of 6×8.
5. **The counting direction problem was described only partially.** In both the tuning videos and the test videos, the two ROIs carry mostly opposite traffic:
   - test Lane 1: 501 downward / 126 upward;
   - test Lane 2: 219 downward / 434 upward.

   The counter counts downward in both ROIs, so Lane 2's dominant direction is never counted. The old paper mentioned the 44 % omission but not that it is mainly Lane 2's main flow.
6. **Threads-only efficiency at N = 4 and 8.** E(4) and E(8) were obtained by dividing a ratio measured with two lane workers by 4 and 8. They are not measurements.
7. **The speed figure (Fig. 4)** showed unvalidated overlay speeds prominently in the method section. The caption warned about this, but an illustrative figure of unvalidated output does not belong in the main results.
8. **Naming conflict.** "12 test videos" for the benchmark set collides with the 44-video test set.
9. **Title.** "… Using Grid-Based Motion Detection and Parallel Lane Processing" can be read as claiming a speed-up from parallel processing, which the threads did not provide.

## 6. Numerical inconsistencies found

| Location | Paper | Source | Status |
|---|---|---|---|
| Table IV, 6×6 detection recall | 0.883 | 0.88246 (`final_numbers.json`) | **wrong rounding**; should be 0.882 (also in the earlier text "recall 0.883 to 0.908") |
| Grid text, 6×8 J significance | "significantly" | site-level permutation, Holm p = 0.09 | overstated (see §5.1) |
| All benchmark numbers (Tables III, V, ratios, Amdahl) | as printed | `audit_numbers.json` | consistent |
| Grid table counts, MAE, FPS, fragmentation | as printed | CSV/JSON | consistent |

## 7. Missing methodological details

- **Counting ground truth:** how crossing events are derived from trajectories, the lane assignment, the matching window, and the definitions of false, duplicate and missed counts and of directional omissions.
- **Counting metrics:** precision, recall, RMSE and MAPE were not reported; lanes were not reported separately.
- **Statistics:** resampling unit, test, number of comparisons, correction.
- **Site grouping:** visual criteria, uncertain merges, and the test-set site groups. The mapping was not provided.
- **Classification:** confusion matrix, per-class support, weighted-F1, Car↔Van confusion rates.
- **Track association failures:** whether missed counts come from undetected vehicles or from detected vehicles whose tracks never crossed the line.
- **Platform:** that the grid study ran on a different machine and OpenCV version from the benchmark.

## 8. References requiring verification

All 22 DOIs were re-resolved through the Crossref API on 26 Nov 2026. Authors, titles, venues, volumes, issues and pages match the bibliography. The second author of [8] is correctly "Gajdošech" (Crossref family name).

Items for author confirmation:
- **[13] Cheung & Kamath:** Crossref gives only the start page 881. The end page 892 cannot be confirmed from Crossref.
- **[20] Amdahl:** Crossref has no volume number. "vol. 30" (AFIPS Conf. Proc.) is conventional but not in the metadata.
- **[24], [25] Python documentation:** no DOI; access date given.
- **Muthuraja & Sankaranarayanan conference version:** still unresolved (venue not identifiable). Only the preprint [2] is cited.

Citations and entries match: every citation resolves, and no entry is unused (scripted check). The related-work comparison with [2] (71.03 % micro-F1, 1.34× multi-process) is explicitly marked as not directly comparable. That is fair.

## 9. Recommended corrections (all carried out in `revised_paper.tex`)

1. **Statistics.** Report paired permutation tests at site level with Holm correction, naming the metric of every claim. Withdraw the 6×8 J significance claim. State the all-18-test correction.
2. **Objective.** Give J as an equation with weights, constraints and the tie rule. State that it was fixed before cross-validation and test results existed (verified from timestamps and by recomputing the frozen selection).
3. **Counting.** Add a lane-by-lane table: false (with opposite-direction share), duplicate, missed (with not-detected share), precision, recall, F1, MAE and RMSE. Describe the trajectory-based ground truth exactly. Disclose that Lane 2's main direction is never counted.
4. **Classification.** Add the 6×8 confusion matrix with support, per-class P/R/F1, weighted-F1 and confusion rates. Compare with always-Car in both macro- and weighted-F1. Do not call it a three-class classifier.
5. **Dataset.** Describe the site grouping criteria. Provide the full mapping (`dataset_split_and_site_mapping.csv`) as supplementary material.
6. **Speed.** Remove Fig. 4 from the main text (keep it as supplementary material). State the magnitude-only estimate, the lack of outlier handling and that it was not evaluated.
7. **Benchmark.** Remove E(4) and E(8). State explicitly that the threading result is specific to this code and machine.
8. **Positioning.** Present the paper as an experimental study. Revise the title, abstract, discussion and conclusion accordingly.
9. **Naming and numbers.** Fix the "12 test videos" naming and the 0.883 → 0.882 rounding.
