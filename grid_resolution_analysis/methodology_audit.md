# Methodology audit: grid-resolution experiment (commit b7c3756)

This audit was written before the corrected experiments were run. It records what the first grid-resolution analysis actually did, where its methodology was weak, and what the correction changes. All statements were checked against the files named in each item.

## 1. Which videos were used

**Benchmark set (12 videos).**
- Source: `benchmark_results/selection_report.json`, three videos per weather folder of `DETRAC_VIDEOS_eval`, chosen by density tier.
- Videos: MVI_39761, 39771, 40131, 40141, 40772, 40773, 40774, 40963, 63525, 20011, 20033, 39311.

**Validation set (26 videos).**
- Definition: `experiment.py video_sets()` takes every other video in `DETRAC_VIDEOS_eval` that has an XML.
- Videos: MVI_20012, 20032, 20034, 20035, 20051, 20052, 20061–20065, 39031, 39211, 39401, 39501, 39511, 39851, 40191, 40192, 40201, 40204, 40771, 40775, 40962, 63521, 63544.

**File-level overlap: none.** No video is in both sets.

## 2. Camera and site sharing

The two sets are disjoint by filename but are not independent by camera site.
- **No site metadata in the annotations.** UA-DETRAC XMLs contain only `sequence_attribute` (camera_state, sence_weather) and `ignored_region` boxes.
- **Sites were inferred visually.** The method was:
  1. compute median-background thumbnails of all 100 videos (`site_grouping.py`);
  2. view them as a contact sheet (`site_grouping_all_sheet.png`, `site_grouping/zoom_eval.png`);
  3. cross-check with an automatic gradient-correlation clustering (`site_groups.json`).
- **Rule for doubtful cases: merge.** Over-merging only costs folds, while under-merging leaks scene information. The resulting 12 groups are listed in `site_groups_manual.json`.

| Site group | Benchmark videos | Validation videos |
|---|---|---|
| A, bus-stop boulevard | 20011, 20033 | 20012, 20032, 20034, 20035, 20051, 20052, 20061–20065 (11) |
| C, wet post-divider road | 40131, 63525 | 63521, 63544 |
| D, barrier highway at dusk | 40963 | 40962 |
| E, night intersection | 40772, 40773, 40774 | 40771, 40775 |
| B, hedged divider road | 39761, 39771, 40141 | none |
| H, plaza / roundabout | 39311 | none |
| F, G, I, J, K, L | none | 39851; 39031, 39211; 39401; 39501, 39511; 40191, 40192; 40201, 40204 |

**Finding: 8 of the 12 benchmark videos share a camera site with validation videos, and 16 of the 26 validation videos share a site with the benchmark.** The earlier "tuned on validation, tested on benchmark" figures are therefore not site-independent.

## 3. Prior use of the videos

**All 38 eval videos were examined before the grid experiment:**
- in the multi-grid batch evaluation (`evaluation/per_video_results.csv`, 2×3 / 6×6 / 10×16);
- in `heatmaps/`.

**The 12 benchmark videos were examined in several places:**
- `benchmark_results/` and `benchmark_results_v2/`;
- the paper drafts;
- the first grid experiment.

**The 62 other UA-DETRAC videos (in `DETRAC_VIDEOS` only) appear to be unused.** A recursive grep over all *.csv, *.json, *.md, *.txt, *.py, *.tex and *.log files outside the video and annotation folders found no reference to any of them.

The repository cannot show whether someone watched these 62 videos, so that remains unverifiable.

## 4. How the grid and the thresholds were selected (b7c3756)

**Grid selection used the benchmark videos.**
- Six grids were run on the 12 benchmark videos.
- 10×12 was named "the best grid" from those results (`analysis_report.md`, Q1).
- The grids to tune (6×6, 10×12, plus 8×8 as the cost/accuracy compromise) were picked after seeing benchmark results.
- The recommendation of 8×8 (Q9) also rests on benchmark numbers.

**Threshold tuning used the 26 validation videos.**
- Method: grid search of 810 candidates, maximising pooled macro-F1.
- Evaluation: once on the 12 benchmark videos.
- Problem: those 12 videos share sites with validation (section 2) and had already guided the grid choice.

**Consequence.** The benchmark figures are retrospective, previously examined results. They are not an untouched test. Examples:
- 10×12 tuned macro-F1 of 0.364;
- the +0.013 to +0.078 gain for 8×8.

**No cross-validation.** Only one validation/test partition was used, so the estimates have no measure of their variance.

## 5. Counting evaluation

**There was no ground-truth counting evaluation.**
- `rparallel_he3.py` writes `lane_vehicle_counts` per run.
- Nothing compared those counts with the annotations.
- The earlier report said: "Counting accuracy was not evaluated here and may suffer."

**Production counting rule** (`LaneTracker.compute_tracking`, STEP 5):
- A track is counted once when its centroid moves from above to on or below `count_line_y = roi_y + roi_h//2`, with age ≥ 2.
- A crossing is suppressed if it falls within 30 frames and 60 px of an earlier crossing in the same lane.
- Tracks suppressed as duplicates are marked counted.

## 6. Lane directions

**Only downward crossings are counted, in both lanes.**
- Vehicles moving up the image never trigger a count. In the UA-DETRAC views these are vehicles driving away from the camera.
- The two ROIs are fixed for every video: Lane 1 x 47–488, Lane 2 x 500–941, both y 202–343. They do not follow each scene's real carriageways.
- The trajectory GT built in this audit shows how much this matters. Upward crossings are a large share of all line crossings in the ROIs:

| Set | Upward crossings | Share |
|---|---|---|
| Validation | 460 of 1410 | 33% |
| Benchmark | 129 of 396 | 33% |
| Test | 560 of 1280 | 44% |

Those vehicles cannot be counted by any grid.

**Lane-overlap risk.** A vehicle straddling the 12 px gap between the ROIs can be tracked in both lanes. The production code has no cross-lane de-duplication of counts. The combined-lane evaluation must therefore not count such a vehicle twice.

## 7. Other observations relevant to the correction

- **Timing came from single runs.** Earlier grid timings were one run per grid and video, with no repetition. All grid-experiment runs used the Linux VM; none used the Windows benchmark machine. The corrected runs launch two workers at a time for throughput, so their run times are not used. Dedicated serial, repeated timing trials are needed.
- **The fixed 100 px² minimum contour area never takes effect.** The per-cell dilation (kernel = cell width × half cell height, 5 iterations) grows any single pixel above threshold into a blob that fills the cell. Contour areas are 455 px² at 10×12 and 2448 px² at 4×6. A cell is "hot" whenever any pixel passes the threshold (verified numerically in `fragmentation_detection_stage.json`).

## 8. Execution plan (as carried out)

1. **Complete the validation runs.** Run all six grids on the 26 validation videos; 4×6, 6×8 and 8×10 were missing.
2. **Grouped cross-validation (`cross_validation.py`).**
   - Leave one site group out (10 folds) on validation only.
   - In each fold, tune the thresholds per grid on the training folds and select the grid with a predefined multi-metric objective J (classification, counting, detection, fragmentation and unmatched detections), with an FPS and memory guard.
   - Evaluate on the held-out site, then pool the out-of-fold results.
3. **Freeze the configuration.** Run the selection on all 26 validation videos and save it as `selected_configuration.json`.
4. **Build a site-independent test set** (`test_set.json`).
   - Start from the 62 unused videos.
   - Remove 18 whose scenes resemble any of the 38 eval sites, leaving 44 videos.
   - Run these only after the freeze, with no tuning on them.
5. **Counting ground truth (`counting_evaluation.py`).**
   - Build GT events from trajectory crossings of the same line.
   - Replay the production count events exactly from the logs, verified against `lane_vehicle_counts`.
   - Match events per lane and combined (deduplicated), then compute MAE, MAPE, event F1, and false, missed and duplicate counts, overall and per class.
6. **Fragmentation diagnosis (`fragmentation_analysis.py`).** Run separately as experimental variants: 8-connected clustering (`conn8`) and vertical gap-bridging (`bridge`). Also analyse ID switches.
7. **Repeated serial timing trials** (`timing_trials.py`): 3 trials × 3 videos × 6 grids, with the grid order rotated.
8. **Report and verify.** Write the updated report, the reproduction commands and the paper subsection, then run the verification checks (`verify_corrected.py`).
