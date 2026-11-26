# Grid Resolution Sensitivity Analysis for Geometry-Based Vehicle Classification

All numbers come from `results.csv`, `summary.json` and `tuning_*.json`, which
`experiment.py` produces. Nothing below is estimated by hand.

## 1. What was tested, and how

### Pipeline facts that shape the experiment

| Aspect | Where / how |
|---|---|
| Grid configuration | `user_input_data.json` → `grids.rows`, `grids.cols` (read by `seq.py` and `rparallel_he3.py`) |
| Cell size | `cell_w = 441 // cols`, `cell_h = 141 // rows`. Resulting sizes in px: 4×6 → 73×35, 6×6 → 73×23, 6×8 → 55×23, 8×8 → 55×17, 8×10 → 44×17, 10×12 → 36×14 |
| Morphology | per-cell 5×5 blur, threshold 60, dilation kernel (cell_w, cell_h/2) × 5 (scales with the cell); minimum contour area **fixed** at 100 px² |
| Clustering | 4-connected flood fill over hot cells (diagonal-only neighbours do **not** join) |
| Object | bounding box = union of member cells; `col_span`, `row_span` = distinct columns/rows; `cell_count` |
| Classifier | **only** normalized spans: Bus if `col_span/cols > 0.75` or both > 0.55; Van if `row_span/rows > 0.45`; else Car. Per-frame labels are aggregated per track by a weighted vote (weight (cy/H)·1.15^age); a track is scored after 5 frames |
| Evaluation | existing vehicle-level protocol: each scored track box is matched to the highest-IoU ground-truth box (IoU > 0.1) at frame t+1, then a majority label is taken per ground-truth ID. A matched DETRAC "others" vehicle counts as Car |

### Experimental design

* **Grids.** 4×6, **6×6 (current project baseline)**, 6×8, 8×8, 8×10 and 10×12.
* **Held fixed.** Only the grid changed: ROIs, videos, preprocessing, thresholds, tracking and evaluation are identical across grids.
* **Videos.** The 12 benchmark videos (Cloudy/Night/Rainy/Sunny × light/medium/heavy), with the same UA-DETRAC XML for every grid.
* **Execution.**
  * Each run used the shipped `rparallel_he3.py` in its unthreaded two-lane mode (`BTP_EXECUTOR=serial`). Its Lane 1 is bit-identical to `seq.py` (verified in the paper's benchmark), and Lane 2 is evaluated too.
  * `experiment.py` imports the script as a module, disables video writing, and logs per-frame track features. **The project's scripts were not modified.**
* **Replay verification.**
  * An offline replay of label voting and of `evaluate_lane` reproduces the original evaluation for **72/72** runs, including every logged per-frame label.
  * The vectorized replay used for threshold search equals the exact replay for 72/72 runs.
* **What the metrics count.** Detection failures (missed GT vehicles, unmatched predicted tracks) are reported separately from classification errors (the confusion matrix of matched vehicles).
* **Combined lanes.** A ground-truth vehicle that appears in both ROIs is counted once. 57 such vehicles occur, so the combined total is 470 GT vehicles, not 224 + 303 = 527.
* **Aggregation.** "Pooled" means confusion matrices summed over videos. Macro-F1 is the unweighted mean of the Car, Van and Bus F1.
* **Uncertainty.** 95% confidence intervals come from a paired bootstrap over videos (2000 resamples).
* **Reference point.** For context, a trivial rule that labels every matched vehicle **Car** is reported. On these videos it scores a macro-F1 of 0.313, because about 89% of matched vehicles are cars.

### Threshold tuning (experiment 2)

* **Data split.** Thresholds were tuned on the **26 other annotated videos** in `DETRAC_VIDEOS_eval` (validation). The 12 benchmark videos were used only for testing the selected thresholds.
* **Grids tuned.** Two grids:
  * the baseline, 6×6;
  * the best grid, 10×12.

  8×8 was also tuned, as the best cost/accuracy compromise.
* **Rule families searched.**
  * **A:** the original rule structure with new thresholds (330 candidates).
  * **B:** family A plus a maximum box aspect ratio for Van (480 candidates).
* **Objective.** Pooled macro-F1 on the validation set.

### Performance measurement

Timing and memory come from the same runs, on a 2-vCPU Linux VM (OpenCV 5.0) with video writing disabled. They are valid **between grids**, but not comparable with the paper's Windows numbers.

## 2. Results — experiment 1 (grid only, original thresholds)

Both lanes combined, pooled over the 12 videos:

| Grid | Macro-F1 | Car P / R / F1 | Van P / R / F1 | Bus P / R / F1 | Acc.* | Car→Van | Correct | Misclass. | Missed GT | Unmatched det. |
|---|---|---|---|---|---|---|---|---|---|---|
| 4×6 | 0.158 | 0.91 / 0.16 / 0.28 | 0.07 / 0.72 / 0.13 | 0.04 / 0.15 / 0.06 | 0.20 | 65.8% | 86 | 343 | 41 | 443 |
| **6×6** | **0.208** | 0.89 / 0.28 / 0.43 | 0.08 / 0.62 / 0.13 | 0.04 / 0.15 / 0.06 | 0.30 | 54.6% | 131 | 308 | 31 | 477 |
| 6×8 | 0.218 | 0.87 / 0.29 / 0.44 | 0.07 / 0.66 / 0.13 | 0.06 / 0.15 / 0.08 | 0.31 | 59.2% | 137 | 304 | 29 | 465 |
| 8×8 | 0.255 | 0.86 / 0.40 / 0.54 | 0.08 / 0.55 / 0.13 | 0.06 / 0.14 / 0.09 | 0.40 | 49.2% | 175 | 267 | 28 | 495 |
| 8×10 | 0.266 | 0.88 / 0.40 / 0.55 | 0.08 / 0.62 / 0.14 | 0.09 / 0.15 / 0.11 | 0.40 | 52.7% | 176 | 264 | 30 | 569 |
| 10×12 | **0.279** | 0.89 / 0.51 / 0.64 | 0.08 / 0.55 / 0.15 | 0.04 / 0.05 / 0.05 | 0.49 | **44.1%** | 211 | 221 | 38 | 757 |

\*Accuracy is measured among matched vehicles.

There are 470 GT vehicles in total. Detected (matched) vehicles range from 429 to 442 across grids, a detection recall of 0.91–0.94. Confusion matrices for each grid are in `confusion_matrices/`.

**Change in macro-F1 relative to 6×6** (paired bootstrap 95% CI):

| Grid | 95% CI of change | Videos better / same / worse |
|---|---|---|
| 4×6 | −0.086 to −0.025 | 2 / 1 / 9 |
| 6×8 | −0.019 to +0.042 | 6 / 1 / 5 |
| 8×8 | +0.013 to +0.078 | 9 / 1 / 2 |
| 8×10 | +0.013 to +0.104 | 10 / 1 / 1 |
| 10×12 | +0.030 to +0.120 | 11 / 0 / 1 |

**10×12 relative to 8×8:** −0.013 to +0.066 (not distinguishable).

**Every grid relative to "always Car":** every configuration scores below the trivial rule. Even for 10×12 the CI of the difference is −0.083 to +0.006.

**Replication on the 26 independent validation videos** (original thresholds):

| Grid | Macro-F1 |
|---|---|
| 6×6 | 0.181 |
| 8×8 | 0.241 |
| 10×12 | 0.310 |

This is the same ordering as on the benchmark videos.

### Lane-wise results (pooled macro-F1, original thresholds)

| Grid | Lane 1 | Lane 2 | Combined |
|---|---|---|---|
| 4×6 | 0.147 | 0.177 | 0.158 |
| 6×6 | 0.222 | 0.223 | 0.208 |
| 6×8 | 0.238 | 0.230 | 0.218 |
| 8×8 | 0.314 | 0.252 | 0.255 |
| 8×10 | 0.326 | 0.253 | 0.266 |
| 10×12 | 0.327 | 0.255 | 0.279 |

Lane 2 Bus F1 is 0.000 at every grid. Lane 2 stops improving after 8×8, while Lane 1 keeps improving.

### Why classification changes: cluster diagnostics

All rates are per frame, against GT at t+1, over all 12 videos and both lanes:

| Grid | Merge rate¹ | Fragmentation rate² | Mean IoU (matched boxes) | Median box w / GT w | Median box h / GT h | Car frames with norm. row span > 0.45 | AUC row span: Van vs Car | AUC col span: Bus vs rest |
|---|---|---|---|---|---|---|---|---|
| 4×6 | 10.2% | 1.0% | 0.284 | 1.68 | 0.96 | **80.1%** | 0.573 | 0.508 |
| 6×6 | 8.3% | 1.7% | 0.291 | 1.68 | 0.85 | 65.9% | 0.566 | 0.514 |
| 6×8 | 6.8% | 2.1% | 0.317 | 1.47 | 0.85 | 65.1% | 0.571 | 0.533 |
| 8×8 | 5.5% | 3.5% | 0.319 | 1.47 | 0.75 | 56.0% | 0.565 | 0.535 |
| 8×10 | 4.4% | 4.3% | 0.336 | 1.33 | 0.75 | 55.4% | 0.568 | 0.564 |
| 10×12 | 3.0% | **7.4%** | 0.347 | 1.25 | 0.69 | 47.4% | 0.557 | 0.602 |

¹Predicted boxes containing two or more GT centres. ²GT boxes containing two or more predicted-box centres.

Mean normalized spans of matched observations at 10×12 (mean ± std):

| Class | Normalized row span | Normalized column span |
|---|---|---|
| Car | 0.47 ± 0.26 | 0.28 ± 0.14 |
| Van | 0.52 ± 0.28 | 0.30 ± 0.14 |
| Bus | 0.54 ± 0.30 | 0.36 ± 0.21 |

Car and Van differ by about 0.06 in mean row span, against a within-class spread of about 0.27. Other grids show the same picture (see `summary.json → diagnostics.feature_means`).

## 3. Results — experiment 2 (threshold tuning; tuned on 26 validation videos, tested on the 12 benchmark videos)

| Grid | Thresholds | Validation macro-F1 | **Test macro-F1** | Test Car / Van / Bus F1 | Test Car→Van | Test Lane 1 / Lane 2 |
|---|---|---|---|---|---|---|
| 6×6 | original (0.75 / 0.55 / 0.45) | 0.181 | 0.208 | 0.43 / 0.13 / 0.06 | 54.6% | 0.222 / 0.223 |
| 6×6 | tuned A: bus_col 0.90, bus_both off, van_row 0.70 | 0.358 | **0.309** | 0.80 / 0.13 / 0.00 | 24.9% | 0.297 / 0.324 |
| 8×8 | original | 0.241 | 0.255 | 0.54 / 0.13 / 0.09 | 49.2% | 0.314 / 0.252 |
| 8×8 | tuned A: bus_col 0.90, bus_both 0.75, van_row 0.80 | 0.366 | **0.323** | 0.87 / 0.10 / 0.00 | 12.0% | 0.349 / 0.349 |
| 10×12 | original | 0.310 | 0.279 | 0.64 / 0.15 / 0.05 | 44.1% | 0.327 / 0.255 |
| 10×12 | tuned A: bus_col 0.60, bus_both 0.55, van_row 0.70 | 0.361 | **0.364** | 0.87 / 0.18 / 0.05 | **11.0%** | 0.385 / 0.363 |

**Family B** (adds a box-aspect condition for Van) never beat family A on validation by a meaningful margin. Its selected rules gave the same test macro-F1 (6×6: 0.310; 8×8 and 10×12: 0.323 and 0.364).

**Test-set 95% CIs** (paired bootstrap over the 12 videos):

| Grid | Tuned − original | Tuned − "always Car" |
|---|---|---|
| 6×6 | +0.043 to +0.173 | −0.034 to +0.024 |
| 8×8 | +0.005 to +0.148 | −0.018 to +0.039 |
| 10×12 | +0.028 to +0.130 | −0.011 to +0.084 |

Validation and test macro-F1 agree closely for 10×12 (0.361 vs 0.364), so there is no sign of overfitting to the validation videos. The tuned rules nevertheless work mainly by **raising the Van threshold (0.70–0.80), i.e. labelling far more vehicles as Car**, and they reach the level of the trivial majority rule rather than clearly exceeding it.

## 4. Performance

Unthreaded two-lane workload, 12 videos, 14,356 frame pairs, 2-vCPU VM:

| Grid | Total time (s) | FPS | Mean processing loop (ms/frame) | Peak memory (MB) |
|---|---|---|---|---|
| 4×6 | 160.6 | 89.4 | 10.29 | 174.0 |
| 6×6 | 158.7 | 90.5 | 10.28 | 177.7 |
| 6×8 | 165.1 | 87.0 | 10.67 | 180.7 |
| 8×8 | 169.8 | 84.6 | 11.06 | 185.3 |
| 8×10 | 186.2 | 77.1 | 12.26 | 190.1 |
| 10×12 | 179.2 | 80.1 | 11.82 | 200.5 |

Relative to 6×6, 10×12 costs about 13% more time (about 15% per frame) and 23 MB more peak memory. 8×8 costs about 7% more time. The 8×10 value above 10×12 is within run-to-run noise of a single run per video.

## 5. Answers to the research questions

**1. Which grid configuration gives the best vehicle classification performance?**
10×12. With the original thresholds it reaches macro-F1 0.279, Car F1 0.644 and the lowest Car→Van rate (44%). With tuned thresholds it reaches 0.364 on the held-out videos.

**2. Does increasing the number of rows and columns consistently improve classification?**
Mostly, but not strictly:

* Combined macro-F1 rises at every step in the tested order. The 6×8 gain is not significant, and the trend is repeated on the 26 validation videos.
* Per class, it is not consistent: only Car improves steadily. Van F1 stays at 0.13–0.15, and Bus F1 peaks at 8×10 (0.11) before falling to 0.05 at 10×12.

**3. Is there a point beyond which more resolution adds little?**
Yes, around 8×8:

* 8×8, 8×10 and 10×12 all beat 6×6 significantly.
* 10×12 is not significantly better than 8×8 (CI −0.013 to +0.066).
* Lane 2 is flat after 8×8 (0.252 → 0.255).

**4. Which configuration produces the least Car→Van misclassification?**
* With the original thresholds: 10×12 (44.1%, against 54.6% at 6×6 and 65.8% at 4×6).
* With tuned thresholds: 10×12 (11.0%), with 8×8 close behind (12.0%).

The mechanism is visible in the diagnostics. Coarse rows inflate the normalized row span of cars: at 4×6, 80% of car observations exceed the 0.45 Van threshold, against 47% at 10×12.

**5. Does grid resolution affect Car, Van and Bus differently?**
Yes:

* **Car** gains strongly (F1 0.28 → 0.64), because fewer cars cross the Van threshold.
* **Van** is unchanged (low precision of about 0.08 at every grid).
* **Bus** stays near zero and becomes worse at 10×12, because fragmentation (7.4%) splits long vehicles.

Row and column spans barely separate the classes at any grid: AUC 0.56–0.57 for Van vs Car, and 0.51–0.60 for Bus vs rest.

**6. Are the results consistent across both lanes?**
In direction, yes: both lanes improve from 4×6 to 8×8, and both improve with tuning. In size, no:

* Lane 1 benefits more (0.222 → 0.327).
* Lane 2 plateaus at about 0.25 after 8×8.
* Lane 2 never classifies a bus correctly.

**7. Does the best configuration have acceptable execution time?**
Yes in relative terms. 10×12 costs about 13–15% more time and 13% more memory than 6×6, and still runs at about 80 FPS in this VM.

The more important cost is **fragmentation**: unmatched detections rise from 477 to 757. Counting accuracy was not evaluated here and may suffer.

**8. Does threshold tuning improve the best grid configuration?**
Yes against the original thresholds: 10×12 rises from 0.279 to 0.364 on held-out videos (CI +0.028 to +0.130), and Car→Van falls from 44% to 11%.

But the improvement comes mostly from labelling more vehicles as Car. The tuned classifier is **not significantly better than labelling everything Car** (CI −0.011 to +0.084), and Van and Bus remain largely unrecognized (F1 0.18 and 0.05).

**9. What grid configuration should be recommended for future experiments, and why?**
**8×8** as the default:

* It captures the significant gain over 6×6 (+0.013 to +0.078).
* It is statistically indistinguishable from 10×12.
* It has about half the fragmentation of 10×12 (3.5% vs 7.4%) and fewer unmatched detections (495 vs 757), which matters for counting.
* It costs about 7% more time than 6×6.

Use **10×12** only if classification is the sole objective, and after checking counting accuracy. **Do not expect any grid to fix the classifier.** The evidence points to the features (normalized spans) as the limiting factor, not the grid. Better features are needed: e.g. calibrated physical length via the speed module's pixel-per-meter scale with perspective correction, temporal shape statistics, or appearance.

## 6. Final research-oriented summary

* **Baseline (6×6, original thresholds, 12 videos, both lanes):**
  * macro-F1 0.208;
  * Car / Van / Bus F1 0.43 / 0.13 / 0.06;
  * Car→Van 54.6%.
* **Best grid (10×12):**
  * macro-F1 0.279, a gain of +0.071 (95% CI +0.030 to +0.120);
  * Car F1 +0.22;
  * Van F1 +0.01;
  * Bus F1 −0.02.
* **8×8:** +0.047 (CI +0.013 to +0.078).
* **Lanes:**
  * Lane 1: 0.222 → 0.327.
  * Lane 2: 0.223 → 0.255 (Bus F1 0 throughout).
* **With tuned thresholds** (validation → test): 10×12 reaches 0.364, and Car→Van falls to 11%.
* **Overhead of 10×12 vs 6×6:**
  * about 13% time;
  * about 13% memory;
  * 59% more unmatched detections.
* **Limitations:**
  * Van (29) and Bus (20) are rare among matched test vehicles, so their F1 values are noisy.
  * Each grid ran once per video, on a VM.
  * Matching is greedy, with IoU > 0.1 and the t+1 offset; "others" counts as Car.
  * Counting accuracy and per-frame detection quality beyond the diagnostics above were not evaluated.
  * Grids above 10×12 were not tested, so the plateau for Lane 1 is not established.
  * Validation and test videos share some camera sites (UA-DETRAC reuses locations), so the held-out test is independent at the video level, not the site level.

**Is this strong enough for a paper subsection titled "Grid Resolution Sensitivity Analysis for Geometry-Based Vehicle Classification"?**
Yes, as an honest sensitivity and negative-result analysis. The supported claims are:

1. Finer grids significantly improve macro-F1 over the 6×6 configuration, with gains levelling off from 8×8.
2. The gain comes almost entirely from fewer Car→Van errors, through a smaller normalized row span for cars.
3. Fragmentation rises with resolution.
4. Neither grid resolution nor validated threshold tuning makes the span-based classifier reliably better than a majority-class rule, because span features barely separate the classes (AUC ≤ 0.60).

It does **not** support a claim that higher resolution produces a good vehicle classifier.

## 7. Reproduce

From the repository root (Python with OpenCV, NumPy, Matplotlib, psutil):

```
python grid_resolution_analysis/experiment.py run --set benchmark --resume
python grid_resolution_analysis/experiment.py run --set validation --grids 6x6,8x8,10x12 --resume
python grid_resolution_analysis/experiment.py tune --grids 6x6
python grid_resolution_analysis/experiment.py tune --grids 8x8
python grid_resolution_analysis/experiment.py tune --grids 10x12
python grid_resolution_analysis/experiment.py analyze
```

**Outputs:**

* `results.csv`: every grid × {Lane1, Lane2, Combined} with all metrics, counts, timing and diagnostics.
* `confusion_matrices/`: CSV and PNG per grid.
* `classification_comparison.png`, `macro_f1_comparison.png`, `performance_comparison.png`, `threshold_tuning_comparison.png`.
* `summary.json` and `tuning_<grid>.json`.
* `runs/`: per-run logs, track logs, original evaluations and metrics.

Failures: none (72 benchmark runs and 78 validation runs completed).
