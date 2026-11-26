# Final verification report: `paper_revision/revised_paper.tex`

The automated checks are in `paper_revision/scripts/verify_revision.py`, with results in `updated_tables/verification_results.json`. **All 37 checks pass.**

## 1. Build

| Item | Result |
|---|---|
| Template | **The real IEEEtran class is not available.** CTAN and GitHub are blocked by the network policy, and the class is not installed. `revised_paper.pdf` is a **layout preview** built with a stand-in class (`scripts/IEEEtran_standin_preview_only.cls`: 10 pt Times, IEEE letter margins, footnote-size captions, Roman section numbers). **It is not an IEEE-compliant build.** |
| LaTeX errors | 0 (pdfLaTeX, two passes) |
| Undefined references / citations | 0 |
| Overfull boxes | One 2.8 pt overfull inside a TikZ node of the thread-pool diagram (`draw_tracking_results` in a fixed-width node; already present in the original). No table overflow. |
| Pages (stand-in) | 8. The original `paper_final.tex` gives 8 pages plus 118 words with the same stand-in, so the revision is not longer. **Confirm the page count in Overleaf with IEEEtran.** |
| Page numbering | Continuous 1–8 (the stand-in prints page numbers; IEEEtran conference mode does not) |
| Figures | 7 figures. Data figures are vector PDF; the one photo (Fig. 1) is a 960×540 video frame, about 280 dpi at column width. Fonts are 7–8 pt at column width. All were rendered and inspected; no overlapping labels. |
| Equations | (1) Amdahl, (2) speed, (3) objective J. Each was checked against the code (`cross_validation.py`, `seq.py`/`rparallel_he3.py`). |
| Numbering | Tables I–VII and Figs. 1–8 are referenced in order; every `\ref` resolves |

## 2. Originals unchanged

`git diff fe76ed7` is empty for all of the following:
- `seq.py`, `rparallel_he3.py`, `app.py`, `user_input_data.json`;
- `paper/paper_final.tex`, `paper/paper.tex`, `paper/paper_8p.tex`, `paper/figures/`;
- `benchmark_results/`, `benchmark_results_v2/`;
- `grid_resolution_analysis/{results.csv, analysis_report.md, updated_analysis_report.md, counting_results.csv, cross_validation_results.csv, selected_configuration.json, runs/}`.

## 3. Numerical claims vs sources (scripted)

| Paper element | Checked against | Status |
|---|---|---|
| Table IV (all 32 numbers) | `classification_test_set.csv`, `final_numbers.json`, `counting_by_lane_and_error_type.csv` (TP/P/G recomputed from raw counts), `timing_trials_summary.json` | Pass |
| Table V (confusion matrix, 9 P/R/F1 values) | `classification_test_set.csv` | Pass |
| Table VI (all 144 numbers) | `counting_by_lane_and_error_type.csv` | Pass |
| p-values and intervals in the text (MAE, RMSE, count F1, J, macro-F1, all-18 minimum) | `significance_tests_test_set.csv` | Pass |
| 44 % upward crossings; +44 % count error | counting CSV | Pass |
| Benchmark numbers (Tables III, VII, ratios, Amdahl) | `audit_numbers.json` recomputed from `benchmark_results_v2` (identical) | Pass |
| Objective weights in Eq. (3) | `cross_validation.py` | Pass |
| Selection reproducible (6×8, frozen thresholds) | `objective_verification.json` | Pass |
| Timeline: timing → freeze → test-set definition → test runs | file timestamps | Pass |
| Mapping: 12 / 26 / 44 / 18 videos; 10 tuning-site groups, 12 test-site groups; no shared sites | `dataset_split_and_site_mapping.csv` | Pass |

Selected text numbers were also checked by hand against `objective_verification.json`, `cross_validation_oof_summary.csv` and the CSVs above:
- J per grid;
- the OOF range 0.518–0.538;
- 8 of 10 folds choosing 6×8;
- 701 of 976;
- 365 of 378;
- 2.5–3.1 tracks per vehicle and 69–75 % with an ID switch;
- Van 7 % and Bus 6 %.

## 4. Claim-to-evidence table

| Paper claim | Evidence file | Verification status |
|---|---|---|
| Sequential and parallel implementations agree exactly on 12 benchmark videos | `benchmark_results_v2/equivalence.csv` | **Verified** (60/60) |
| Threading slowed the same two-lane workload (0.860 ± 0.077; slower on 12/12) | `benchmark_results_v2`, `audit_numbers.json` | **Verified** (one run per workload; one machine) |
| Two-lane throughput gain comes from shared decode and preprocessing (1.46× unthreaded vs 1.26× threaded) | `audit_numbers.json` | **Verified** (measurement); the mechanism is an interpretation supported by the four-workload design |
| Grid resolution does not change macro-F1 significantly after re-tuning | `significance_tests_test_set.csv` | **Verified** (all Holm p = 1.0) |
| 6×8 significantly improves the selection objective | same | **Not supported → withdrawn** (Holm p = 0.09) |
| 10×12 improves the selection objective | same | **Not supported** (Δ = +0.0004, p = 0.96); not claimed |
| 10×12 increases count MAE/RMSE vs 6×6 | same; counting CSV | **Verified** within the metric family (Holm p = 0.010 / 0.013); not significant over all 18 tests (p = 0.058), as stated |
| 6×8 raises counting F1 by 0.022 | same | **Verified** within the family (Holm p = 0.012); small effect |
| Counting performance (Table VI, error decomposition) | `counting_by_lane_and_error_type.csv`; production totals in `rpar_frame_metrics.json` | **Verified** (replay equals production counts in 100 % of runs) |
| Counter never counts Lane 2's main direction | counting CSV (Lane 2: 434 up vs 219 down on test) and `rparallel_he3.py` STEP 5 | **Verified** |
| Classification performance (Table V) | `classification_test_set.csv` (pooled confusion matrix) | **Verified** |
| Classifier beats always-Car in macro-F1 but not weighted-F1 | same (site-bootstrap CIs) | **Verified** |
| Fragmentation mainly from empty interior cells (bridge −87–89 %, 8-conn −35–40 %) | `grid_resolution_analysis/fragmentation_validation.csv` | **Verified** (diagnostic variants on tuning videos) |
| Site-independent evaluation | `dataset_split_and_site_mapping.csv`, `site_groups_manual.json`, contact sheets | **Partially verified.** Site groups are disjoint by construction and the timeline is clean, but the sites are inferred visually (no metadata), so camera independence is plausible, not proven. Stated as a limitation. |
| Speed estimation accuracy | none (no speed ground truth) | **Not validated.** The paper states it is not evaluated. |

## 5. Conclusions vs results

Every sentence of the revised abstract and conclusion was compared with the table above.
- No conclusion claims speed accuracy, three-class classification quality, a general threading result, or a universally best grid.
- Interpretations are labelled "our interpretation".

## 6. Remaining limitations

- **No IEEEtran build.** The page count and final layout must be confirmed in Overleaf.
- **Unproven camera independence.** Visual site inference cannot prove it.
- **Single machine, single run.** The benchmark has one run per workload on one machine.
- **Different platform for the grid study.** It ran on a VM (OpenCV 5.0).
- **Judgement-based weights.** The objective weights were fixed in advance, but by judgement.
- **Fixed ROIs.** They are the same for every scene. The counter is downward-only, so the main direction of Lane 2 is never counted.
- **References for author confirmation:**
  - the end page of [13];
  - the volume of [20];
  - the conference version of the Muthuraja & Sankaranarayanan paper.
