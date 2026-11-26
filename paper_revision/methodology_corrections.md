# Methodology corrections

Each item states what was wrong or missing, what was done, and where the evidence is. No experiment was re-run with different settings, and no production code, historical result or original paper file was changed. All new numbers come from existing run logs through new analysis scripts in `paper_revision/scripts/`.

## 1. Statistical reporting (Part 2)

**Old procedure.**
- Paired percentile bootstrap of pooled metrics, resampling the 25 recording sessions (`grid_resolution_analysis/final_analysis.py:boot`), 2000 resamples.
- Four metrics × three grids, reported without multiplicity correction.
- The session is not the camera site: several sessions show the same scene, so session resampling understates between-site variance.

**New procedure** (`revision_analysis.py stats` → `updated_tables/significance_tests_test_set.csv`):
- **Pairing.** All grids run on the same 44 videos. Differences are always grid − 6×6 on identical videos.
- **Resampling unit.** The camera-site group, 12 groups on the test set. This is the primary analysis. Session level (25 units) is kept as a sensitivity analysis.
- **Intervals.** 95 % percentile bootstrap (4000 site resamples). Bonferroni-widened intervals for 3 comparisons are in the CSV.
- **Test.** Two-sided paired permutation: in each permutation, the two grids' results are swapped within a random subset of sites and the pooled metric is recomputed. The p-value is (count of |Δ*| ≥ |Δ| + 1) / (B + 1), with B = 4000. A permutation test is used because the pooled metrics (macro-F1, J, event-F1) are ratios with no simple variance formula.
- **Multiplicity.** Holm correction within each metric family (3 comparisons against the baseline) is the primary correction. Holm over all 18 tests is also reported.
- **Metrics tested.** Macro-F1, J, counting event-F1, detection recall, count MAE, count RMSE.

**Results (site level):**

| Metric | 6×8 − 6×6 | 8×8 − 6×6 | 10×12 − 6×6 |
|---|---|---|---|
| Macro-F1 | +0.010, p_Holm 1.0 | −0.003, 1.0 | +0.025, 1.0 |
| J | +0.011 [0.003, 0.022], p 0.030, **p_Holm 0.089** | +0.007, 1.0 | +0.0004, 1.0 |
| Count F1 (down) | **+0.022 [0.012, 0.030], p_Holm 0.012** | +0.028, 0.30 | +0.024, 0.30 |
| Detection recall | +0.010, 0.62 | +0.012, 0.62 | +0.025, 0.62 |
| Count MAE | −1.16, 0.23 | +0.61, 0.23 | **+7.07 [5.40, 8.45], p_Holm 0.0096** |
| Count RMSE | −1.91, 0.076 | −0.04, 0.92 | **+6.24 [4.74, 7.90], p_Holm 0.013** |

Over all 18 tests, the smallest Holm-adjusted p is 0.058 (10×12 MAE). The paper states this.

**Three kinds of difference, kept apart in the revised text:**
- **Numerical improvement:** a difference in point estimates. Example: 10×12 has the highest macro-F1, 0.430.
- **Statistical significance:** the corrected test. Only the 6×8 count-F1 gain and the 10×12 MAE/RMSE increase qualify, and only within their metric family.
- **Practical significance:** macro-F1 differences ≤ 0.025 and count-F1 +0.022 are small. +7 count errors per video (+44 %) is large.

**Withdrawn claim:** "6×8 is the only grid whose objective improved significantly". The gain is nominal only.

## 2. Grid-selection objective (Part 3)

The objective was taken from the implementation (`grid_resolution_analysis/cross_validation.py`, `W`, `pooled()`, `select()`, `timing_guard()`):

J(g) = 0.30·F1_macro + 0.10·min_c F1_c + 0.25·F1_count + 0.15·R_det + 0.10·(1 − φ) + 0.10·(1 − u)

**Terms.** Each term is in [0, 1] and pooled over the scored videos. There is no further normalisation, and the weights sum to 1.
- F1_macro: vehicle-level macro-F1 with the grid's thresholds.
- min_c F1_c: the weakest class F1.
- F1_count: combined event-F1 against downward trajectory crossings.
- R_det: matched GT vehicles / GT vehicles.
- φ: GT observations whose box contains two or more detection centres / GT observations.
- u: predicted tracks matching no GT vehicle / predicted tracks.

**Thresholds θ_g.** The argmax of pooled training macro-F1 over 810 candidates. Ties go to the |van_row − 0.45| minimum, then list order.

**Eligibility.** Median FPS ≥ 25 in `timing_trials_summary.json` and peak memory ≤ 1.5 × the 6×6 memory. All six grids were eligible.

**Tie rule.** Among eligible grids within 0.005 of the best J, choose the one with the fewest cells.

**Verification** (`updated_tables/objective_verification.json`):
- Recomputing the selection from the cached validation data gives 6×8 with exactly the frozen thresholds.
- File timestamps show the order: timing trials → frozen configuration → `test_set.json` → first test run.
- `cross_validation.py` was not modified after the CV run (the commit contains the version that produced the files).
- **Disclosure:** the weights reflect the author's judgement. They were fixed before any CV or test result, but after the exploratory benchmark analysis had been seen.

The paper now contains the equation (Eq. 3), every variable, the guard and the tie rule.

## 3. Counting analysis (Part 4)

Implementation: `revision_analysis.py counting`, reusing `grid_resolution_analysis/counting_evaluation.py` → `updated_tables/counting_by_lane_and_error_type.csv`.

**Ground-truth events.**
- For each UA-DETRAC target, take the box centre in consecutive annotated frames.
- An event occurs when the centre crosses y = 272, the same line the counter uses. Direction is down if prev < 272 ≤ cur, up if prev ≥ 272 > cur.
- The lane is the ROI whose x-range contains the centre. Events in the 12 px gap between ROIs or outside both ROIs are excluded and reported (48 on the test set).
- Total annotated vehicles are **not** used as a count.

**Predicted events.**
- These are the counter's own events, replayed exactly from the logs. The replay equals `lane_vehicle_counts` in 100 % of runs.
- Predictions in ignored regions are excluded, identically for every grid.

**Matching** (identical for all grids, predictions and GT):
- |Δt| ≤ 25 frames and |Δx| ≤ 80 px; only downward GT can be matched.
- One-to-one greedy by cost, with an identity bonus.
- Combined scope: both ROIs compete for one GT list, so a vehicle is never credited twice.

**Error types (newly separated):**

| Error type | Definition |
|---|---|
| False count | Unmatched prediction. Sub-counts: caused by an upward-moving vehicle (opposite direction), and overlapping no annotated vehicle (spurious cluster). |
| Duplicate | Unmatched prediction whose vehicle or compatible GT event was already matched. |
| Missed count | Unmatched downward GT event. Split into "not detected near the line" and "detected but no track crossed" (track association failure). |
| Directional omission | Upward GT events. The counter cannot count them. |
| Tracking fragmentation | Tracks per GT vehicle and the share of vehicles with an identity switch (`grid_resolution_analysis/fragmentation_test.csv`). |

**Metrics.**
- Event precision TP/P, recall TP/G↓ and F1.
- Per-video MAE, RMSE and MAPE (MAPE only for G > 0; n reported).
- The same metrics against all-direction GT are in the CSV.

**Lane verification.**
- **Do the lanes count the correct direction? No.** The ROIs carry mostly opposite directions:

  | Set | Lane 1 down / up | Lane 2 down / up |
  |---|---|---|
  | Tuning | 668 / 3 | 282 / 457 |
  | Test | 501 / 126 | 219 / 434 |

  The counter counts downward in both ROIs, so it never counts Lane 2's main direction. This is reported as a limitation. The production code was not changed.
- **Cross-ROI double counting.** Production has no cross-lane de-duplication. The evaluation prevents double credit (combined TP ≤ GT; checked for every row).
- **Same rules for predictions and GT:** yes (same line, same window, same lane assignment rule).

## 4. Dataset split and site grouping (Part 5)

`paper_revision/dataset_split_and_site_mapping.csv` covers all 100 videos. For each it gives: site ID, partition, frames, annotation file, annotated frames, weather and camera attributes, session prefix, automatic cluster ID and the grouping uncertainty.

**Tuning and benchmark sites (12 groups)** come from `grid_resolution_analysis/site_groups_manual.json`.

**Test sites (12 groups, T01–T12)** were defined in this revision for the site-level statistics, by the same visual procedure:
- Criteria: carriageway geometry, medians, fences and hedges, buildings, signboards, bus shelters, overpass pillars.
- Doubtful pairs were merged.
- This was done before the site-level tests were computed.

**Verified:**
- No video of a tuning site is in another fold (leave-one-site-out).
- No test site coincides with a tuning or benchmark site.
- 18 look-alike videos were excluded.
- No test data entered selection or tuning (timestamps; the selection is reproduced without test data).

**Uncertain assignments are listed per video.** The main ones:
- T05: diagonal fenced road, possibly several cameras.
- T06: 39931's hedge strip resembles tuning site L.
- The A / B / C merges among the tuning sites.

**Limitation (stated in the paper):** sites are inferred visually, not from metadata, so camera independence is plausible but unproven.

## 5. Classification evaluation (Part 6)

`updated_tables/classification_test_set.csv` gives, for each grid with tuned and original thresholds:
- the full confusion matrix and per-class support;
- precision, recall and F1 per class;
- macro- and weighted-F1 and accuracy;
- all six confusion rates;
- the always-Car baseline (macro, weighted, accuracy);
- site-bootstrap CIs of the difference from always-Car.

**Main facts for the selected 6×8:**
- Macro-F1 0.416 vs 0.310 (Δ CI [0.060, 0.152]).
- Weighted-F1 0.799 vs 0.805 (CI [−0.022, 0.022]); accuracy 0.803 vs 0.867.
- Van recall 0.168, Van→Car 81.8 %; Bus recall 0.149, Bus→Car 63.2 %.

**Conclusion.** The tuned classifier is better than always-Car only on the class-balanced metric. The paper now says it is not a reliable three-class classifier, explains why, and lists the improvements as future work (none implemented).

**Note:** support differs slightly between grids (1862–1915 matched vehicles), because detection determines which GT vehicles are matched.

## 6. Speed estimation (Part 7)

The code was re-checked:
- calibration k = d/L, with display-to-native coordinate conversion;
- f from `CAP_PROP_FPS`;
- Euclidean centroid displacement accumulated to ≥ 3 cell heights;
- v = (Δd/k)/(Δn/f)·3.6 km/h;
- mean of the last 15 samples;
- magnitude only, no direction;
- no outlier rejection beyond the gate;
- disabled when k = 0 (all experiments).

Status in the revised paper:

| Aspect | Status |
|---|---|
| Implemented | Yes; described |
| Qualitative demonstration | Mentioned; screenshots moved to `supplementary/fig_speed.jpg` |
| Quantitative validation | None; stated explicitly in the abstract, method and threats |

Fig. 4 was removed from the main text.

## 7. Benchmark section (Part 8)

All numbers were re-verified against `benchmark_results_v2` (the recomputed `audit_numbers.json` is identical).

- **Equivalent workloads.** All four workloads use identical preprocessing and outputs (60/60 equivalence).
- **Methodology.** One run per workload and video, the same timer boundaries, profiler off. This is stated, together with the run-to-run caveat.
- **Corrections:**
  - E(4) and E(8) removed (not measured);
  - explicit statement that the result is specific to this implementation and machine;
  - the shared-decode explanation, native OpenCV threading (4.2 cores used by `seq.py`), the GIL, and the distinction between Amdahl estimates and measurements were all kept.

## 8. Figures and tables (Part 9)

- All data figures were regenerated from the same CSVs with 7–8 pt fonts (previously 6–7.5 pt): `figure_sources/make_figures_revised.py`. The data CSVs are byte-identical to the originals.
- The Amdahl legend no longer overlaps a curve.
- New 2×2 grid figure from the revision tables: `figure_sources/make_grid_figures.py`.
- New tables: grid summary (with weighted-F1 and RMSE), the 6×8 confusion matrix with per-class metrics, and the counting-by-lane table.
- Removed from the main text (kept in `supplementary/`):
  - Fig. 4 (speed: illustrative, unvalidated);
  - Fig. 6 (output overlay: illustrative);
  - Fig. 7 (12-video F1 bars: same numbers as Table III).

## 9. Bibliography (Part 10)

All 22 DOIs were re-checked against Crossref on 26 Nov 2026, and all match. Items for author confirmation are in `technical_audit.md` §8. A BibTeX file is not provided because the paper uses an embedded `thebibliography`, which was kept unchanged except as needed by the revision. No entry was added or removed.
