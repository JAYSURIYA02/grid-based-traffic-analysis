# Grid resolution sensitivity: corrected methodology and counting evaluation

This report supersedes the conclusions of `analysis_report.md` (kept unchanged as the record of the first, retrospective analysis). Why a correction was needed is set out in `methodology_audit.md`. In short:
- the grid had been chosen by looking at the 12 benchmark videos;
- 8 of those 12 share camera sites with the validation videos used for tuning;
- counting had never been compared with ground truth.

All numbers below come from:
- `cross_validation_results.csv`
- `cross_validation_oof_summary.csv`
- `counting_results.csv`
- `fragmentation_validation.csv`
- `timing_trials_summary.json`
- `final_numbers.json`

## 1. Protocol

| Item | Setting |
|---|---|
| Pipeline | Unchanged production code (`rparallel_he3.py`, serial executor, 2 lanes, V channel, fixed ROIs, 4-connected clustering). The experimental variants `conn8` and `bridge` are run separately and are never used for selection. |
| Grids | 4×6, 6×6 (baseline), 6×8, 8×8, 8×10, 10×12 (rows × cols) |
| Selection data | Only the 26 validation videos, in 10 camera-site groups (`site_groups_manual.json`) |
| Cross-validation | Leave one site group out. In each fold, thresholds are tuned per grid on the training videos (810 candidates, pooled macro-F1). The grid is then selected on the training videos by the predefined objective J. Results are evaluated on the held-out site and pooled out-of-fold (OOF). |
| Objective J | 0.30 macro-F1 + 0.10 min class F1 + 0.25 counting event-F1 + 0.15 detection recall + 0.10 (1 − fragmentation) + 0.10 (1 − unmatched-track rate) |
| Guard | A grid must reach median ≥ 25 FPS in the timing trials and use at most 1.5× the 6×6 memory (all grids passed) |
| Tie rule | Any grid within 0.005 of the best J counts as tied; the one with fewer cells wins |
| Frozen config | `selected_configuration.json`, written before `test_set.json` existed |
| Test set | 44 UA-DETRAC videos (67,206 frames, 25 recording sessions). No earlier file references them, and their scenes do not resemble any of the 38 eval sites. Evaluated once, with frozen settings. |
| Benchmark set | The 12 original videos, reported as **retrospective / previously examined** only |
| Counting GT | Line crossings of annotated trajectories (box centre crossing y = 272), assigned to a lane by x |
| Counting replay | Production count events replayed from the logs. 100% of runs reproduce the production `lane_vehicle_counts` exactly. |
| Event matching | One-to-one, \|Δt\| ≤ 25 frames, \|Δx\| ≤ 80 px |
| Timing | 3 serial trials × 3 videos × 6 grids on the Linux VM (2 vCPU). Not comparable with the Windows numbers in the paper. |

Two choices for the counting GT matter for interpreting the results:
- **Downward GT ("down").** Only downward crossings, which are the only direction the counter can count. This GT is used in J and for MAE.
- **All crossings ("all").** Both directions. This shows the design limitation.

## 2. Cross-validated grid selection (validation only)

**Out-of-fold results, 26 validation videos, 10 site folds, thresholds tuned within each fold**

| Grid | J | macro-F1 (tuned) | macro-F1 (original thr.) | Count F1 (down) | Count F1 (all) | Det. recall | Fragmentation | Unmatched |
|---|---|---|---|---|---|---|---|---|
| 4×6 | 0.518 | 0.336 | 0.122 | 0.503 | 0.404 | 0.871 | 0.46% | 0.456 |
| 6×6 | 0.525 | 0.328 | 0.181 | 0.533 | 0.432 | 0.884 | 1.06% | 0.440 |
| 6×8 | 0.524 | 0.303 | 0.187 | 0.570 | 0.460 | 0.897 | 1.61% | 0.429 |
| 8×8 | 0.538 | 0.346 | 0.241 | 0.548 | 0.446 | 0.909 | 2.77% | 0.432 |
| 8×10 | 0.529 | 0.325 | 0.240 | 0.577 | 0.473 | 0.915 | 3.61% | 0.462 |
| 10×12 | 0.532 | 0.337 | 0.310 | 0.579 | 0.477 | 0.916 | 7.14% | 0.482 |
| Selection procedure (per fold) | 0.516 | 0.312 | none | 0.537 | 0.435 | 0.890 | 2.02% | 0.444 |

The always-Car baseline on the same pooled matrices is about 0.31 macro-F1.

**Selection is unstable.**
- **Per fold:** 6×8 was chosen in 8 of the 10 folds, 8×8 in one and 6×6 in one.
- **On all 26 videos:** J was 8×8 0.546, 10×12 0.543, 6×8 0.543, 8×10 0.541, 6×6 0.537 and 4×6 0.530. By the predefined tie rule, the cheapest grid within 0.005 of the best won.
- **Frozen configuration:** **6×8**, with thresholds bus_col 0.90, bus_both 0.75, van_row 0.70 and Van aspect ≤ 2.5:

```json
{"selected_grid": "6x8", "thresholds": {"bus_col": 0.9, "bus_both": 0.75, "van_row": 0.7, "van_max_aspect": 2.5}}
```

**The J differences between grids (≤ 0.02) are smaller than the fold-to-fold variation.** The selection procedure's own OOF score (0.516) is lower than every fixed grid's, which is a sign of selection noise. With the original thresholds, macro-F1 rises steadily with resolution, from 0.122 to 0.310. That trend disappears once thresholds are tuned per grid: OOF 0.30–0.35 for every grid, in no monotonic order. Most of the original "resolution effect" on classification was therefore a calibration effect: the fixed normalised-span thresholds suit fine grids better.

## 3. Site-independent test set (44 videos, frozen settings)

Each grid uses its own thresholds tuned on all 26 validation videos (`frozen_thresholds_by_grid.json`). The original thresholds are shown for reference. The always-Car macro-F1 is 0.309–0.310 for every grid.

| Grid | macro-F1 (frozen thr.) | Car / Van / Bus F1 | macro-F1 (original thr.) | Det. recall | Fragmentation | Merge | Unmatched | Count F1 (down) | Count F1 (all) | MAE (down) | J |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6×6 | 0.405 | 0.878 / 0.155 / 0.183 | 0.322 | 0.883 | 1.50% | 9.4% | 0.456 | 0.314 | 0.248 | 16.02 | 0.501 |
| **6×8 (selected)** | **0.416** | 0.895 / 0.148 / 0.204 | 0.339 | 0.892 | 2.33% | 7.8% | 0.430 | 0.336 | 0.263 | 14.86 | 0.512 |
| 8×8 | 0.402 | 0.913 / 0.135 / 0.160 | 0.371 | 0.894 | 3.78% | 6.6% | 0.422 | 0.342 | 0.271 | 16.64 | 0.508 |
| 10×12 | 0.430 | 0.912 / 0.079 / 0.299 | 0.418 | 0.908 | 9.49% | 4.0% | 0.467 | 0.337 | 0.275 | 23.09 | 0.501 |

**Paired bootstrap, difference from 6×6.** Resampling unit: recording session, n = 25; 2000 resamples; 95% intervals.

| Comparison (frozen thresholds) | macro-F1 | Count F1 (down) | Detection recall | J |
|---|---|---|---|---|
| 6×8 − 6×6 | −0.013 … +0.039 | **+0.006 … +0.038** | **+0.001 … +0.020** | **+0.001 … +0.025** |
| 8×8 − 6×6 | −0.038 … +0.036 | **+0.004 … +0.053** | −0.001 … +0.023 | −0.011 … +0.026 |
| 10×12 − 6×6 | −0.018 … +0.072 | −0.008 … +0.052 | **+0.007 … +0.046** | −0.017 … +0.019 |
| 10×12 − 8×8 | −0.020 … +0.080 | −0.034 … +0.020 | not computed | −0.025 … +0.016 |

With the original thresholds, 10×12 − 6×6 macro-F1 is +0.056 … +0.132 and 8×8 − 6×6 is +0.028 … +0.070. Both are significant, but only because 6×6 is badly calibrated by those thresholds.

What the test set shows:
- **The frozen 6×8 generalises to unseen sites as well as any grid.** Its J is the highest of the four, and it is the only grid whose J gain over 6×6 is significant.
- **No grid's tuned macro-F1 is significantly different from 6×6.** All tuned classifiers beat always-Car on this set (0.40–0.43 vs 0.31), but Van F1 stays at 0.08–0.16. Bus F1 varies the most between grids (0.16–0.30).
- **Finer grids find slightly more vehicles but produce more spurious counts.** Detection recall rises with resolution: 0.883 → 0.908, significant for 10×12. Meanwhile, predicted counts grow faster than correct counts.

## 4. Counting (ground truth from trajectories)

**Combined lanes, original thresholds** (counting does not depend on the class thresholds). Each cell shows GT / predicted / correct / false / duplicate / missed counts. Missed counts include the upward crossings for "all".

| Set | Grid | GT all (up) | Pred | TP | False | of which opposite-direction | Dup | Missed (all) | Event F1 down / all | MAE down (Lane1 / Lane2 / combined) |
|---|---|---|---|---|---|---|---|---|---|---|
| Validation (26) | 6×6 | 1410 (460) | 1000 | 520 | 428 | 134 | 52 | 890 | 0.533 / 0.432 | 12.88 / 9.50 / 13.92 |
| | 6×8 | 1410 | 988 | 552 | 405 | 115 | 31 | 858 | 0.570 / 0.460 | 12.23 / 8.77 / 13.15 |
| | 8×8 | 1410 | 1045 | 547 | 456 | 130 | 42 | 863 | 0.548 / 0.446 | 12.15 / 9.73 / 14.65 |
| | 10×12 | 1410 | 1217 | 627 | 533 | 160 | 57 | 783 | 0.579 / 0.477 | 12.73 / 11.08 / 16.27 |
| Test (44) | 6×6 | 1280 (560) | 1377 | 329 | 974 | 172 | 74 | 951 | 0.314 / 0.248 | 11.39 / 10.36 / 16.02 |
| | 6×8 | 1280 | 1318 | 342 | 915 | 182 | 61 | 938 | 0.336 / 0.263 | 10.30 / 10.30 / 14.86 |
| | 8×8 | 1280 | 1428 | 367 | 1007 | 212 | 54 | 913 | 0.342 / 0.271 | 10.50 / 11.41 / 16.64 |
| | 10×12 | 1280 | 1728 | 413 | 1219 | 238 | 96 | 867 | 0.337 / 0.275 | 12.91 / 13.86 / 23.09 |
| Benchmark (12, retrospective) | 6×6 | 396 (129) | 314 | 153 | 145 | 35 | 16 | 243 | 0.527 / 0.431 | 4.58 / 6.67 / 6.58 |
| | 6×8 | 396 | 299 | 162 | 126 | 36 | 11 | 234 | 0.572 / 0.466 | 3.92 / 6.75 / 6.83 |
| | 8×8 | 396 | 316 | 158 | 146 | 46 | 12 | 238 | 0.542 / 0.444 | 3.67 / 6.92 / 7.08 |
| | 10×12 | 396 | 361 | 173 | 168 | 50 | 20 | 223 | 0.551 / 0.457 | 4.92 / 7.42 / 9.67 |

MAPE (per-video mean, GT > 0) is large because many videos have few downward crossings. On the test set it is 2.86 / 2.71 / 2.90 / 3.49 for 6×6 / 6×8 / 8×8 / 10×12. All values are in `final_numbers.json`.

The count accuracy 1 − |P − G| / G is reported in the CSV but is not used for conclusions: false and missed counts can cancel. On the test set, 6×6 has a pooled count accuracy of 0.92 against all crossings while getting only 26% of crossings right. The event-level F1 is the measure used.

**Answers from the counting data:**
- **Higher resolution does not damage the correct counts.** TP rises: test 329 → 413, validation 520 → 627.
- **10×12 produces many more false and duplicate counts.** Test predictions are 1728 against 720 downward GT, and MAE rises from 14.9 (6×8) to 23.1. Against downward GT, 10×12 has the highest count MAE of all tested grids on every set.
- **8×8 sits in between.** Its count F1 is like 10×12's, its MAE is close to 6×6, and its MAE is higher than 6×8's.
- **The largest counting error is independent of the grid.**
  - Every upward vehicle is missed: 44% of test crossings, 33% of validation crossings.
  - Some upward vehicles generate false downward counts through centroid jitter: 18–21% of test false counts.
  - Test scenes (mostly intersections, with the fixed ROIs) have far more false counts than the validation scenes.
- **No vehicle is counted twice across lanes in the evaluation.** In the combined scope both lanes compete for a single GT list. Production does count some vehicles in both lanes; these appear as duplicates.

**Per-class counting (test, frozen thresholds, downward GT):**
- GT: 635 Car, 50 Van, 35 Bus.
- Correctly counted with the correct class: Car 254–341, Van 2–7, Bus 0 for every grid.
- Per-class counts are therefore only meaningful for Car.

## 5. Why fragmentation rises with resolution

**Detection stage.**
- A cell is hot whenever any pixel passes the threshold. The 100 px² area filter never takes effect at any grid (section 7 of the audit).
- Cells shrink from 73×35 px (4×6) to 36×14 px (10×12).
- Frame differencing marks mainly a vehicle's edges, so at small cell sizes the interior of a vehicle often contains empty cells.

**Clustering stage.** The variants were run on the 26 validation videos with original thresholds.

| Grid / variant | Fragmentation | Merge | Det. recall | Tracks per GT vehicle | GT vehicles with ID switch | macro-F1 (orig. thr.) | Count F1 down |
|---|---|---|---|---|---|---|---|
| 8×8 original | 2.77% | 7.6% | 0.909 | 2.27 | 69% | 0.241 | 0.548 |
| 8×8 conn8 | 1.80% | 9.1% | 0.902 | 2.08 | 65% | 0.226 | 0.558 |
| 8×8 bridge | 0.31% | 12.4% | 0.882 | 1.73 | 54% | 0.190 | 0.530 |
| 10×12 original | 7.14% | 4.6% | 0.916 | 2.58 | 73% | 0.310 | 0.579 |
| 10×12 conn8 | 4.31% | 5.9% | 0.909 | 2.28 | 67% | 0.277 | 0.591 |
| 10×12 bridge | 0.91% | 8.8% | 0.897 | 1.84 | 54% | 0.223 | 0.559 |

Fragment pairs at 10×12 (original) break down as follows:
- 3,021 of 6,891 are separated vertically by empty rows;
- 399 touch only at a corner;
- 607 are diagonally separated.

**Diagnosis:**
- **Main cause: one-row gaps inside a vehicle.** Fragmentation comes from the detection stage (sparse hot cells) and is exposed by the clustering rule. Bridging a single empty cell vertically removes 87–89% of it.
- **Secondary cause: diagonal-only adjacency.** 8-connectivity removes 35–40%.
- **Tracking adds identity fragmentation at every resolution.** Even 4×6 averages 2.04 tracks per GT vehicle, with 66% of vehicles having an ID switch. Finer grids increase this to 2.58 tracks and 73%.
- **Removing fragmentation does not improve classification.** Both variants increase merging and lower macro-F1 with the original thresholds. Counting changes only marginally (conn8 +0.01 count F1, bridge −0.02).
- **The variants are diagnostics, not recommended fixes.** They were not tuned, cross-validated or tested.

## 6. Performance (VM, serial, 3 trials × 3 videos)

| Grid | Median FPS (min–max) | Median peak memory |
|---|---|---|
| 4×6 | 98.3 (96.6–106.4) | 169.4 MB |
| 6×6 | 97.2 (91.2–100.3) | 171.1 MB |
| 6×8 | 91.9 (89.9–93.2) | 172.3 MB |
| 8×8 | 86.3 (83.9–91.9) | 174.8 MB |
| 8×10 | 84.3 (75.5–84.5) | 177.2 MB |
| 10×12 | 81.4 (78.4–81.6) | 182.7 MB |

Throughput falls by about 16% from 6×6 to 10×12, and memory rises by about 7%. All grids run at more than three times real time on this VM. These figures must not be compared with the Windows benchmark-machine figures.

## 7. Conclusions

1. **Corrected issue.** Grid choice and threshold "testing" no longer touch any video that shares a site with the tuning data. Selection is cross-validated by site, and the reported test results come from 44 unseen-site videos with frozen settings.
2. **Selected configuration: 6×8** with tuned thresholds, by a predefined rule. The margin over 8×8 and 10×12 is within noise.
3. **8×8 vs 10×12.** On the test set they are not significantly different in macro-F1, count F1 or J.
   - 10×12 has higher recall and lower merging.
   - 10×12 has 2.5× the fragmentation, about 20% more predicted counts and a higher count MAE (23.1 vs 16.6).
4. **Higher resolution does not lose correct counts but over-counts.** It increases false and duplicate counts, so count error grows at 10×12.
5. **Classification vs counting trade-off.** Finer grids help classification only under the original thresholds. Once thresholds are tuned per grid, the classification advantage vanishes, while the counting cost of fine grids remains.
6. **The classifier and the counter are the limiting factors, not the grid.**
   - The classifier stays close to the always-Car baseline on validation and recognises few Vans and Buses.
   - The counter misses all upward traffic.
7. **What is still needed.** The 44-video test set is site-independent but comes from the same dataset and the same fixed ROIs. An external dataset is still needed for claims beyond UA-DETRAC.
