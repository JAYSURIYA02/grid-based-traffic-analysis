# FINAL_AUDIT.md — `paper/paper_final.tex`

Date: 2026-11-25. Branch: `paper-reproducibility-fixes` (not pushed).
Preserved unchanged: `paper/paper.tex` (extended version), `paper/paper_8p.tex`
(the 6-page version this refines), `paper/paper_previous.tex`, and every
earlier figure in git history. The user's compiled 6-page PDF was not
supplied to this session, so the refinement starts from `paper_8p.tex`, its
source.

## 1. Changes made

| # | Original issue | Modification | Reason | Verification |
|---|---|---|---|---|
| 1 | Ref. [2] had a placeholder: "Proc. 2026 IEEE Conf. [TBD]" | Ref. [2] now cites the Research Square preprint by the same authors, *Real-time traffic density estimation using scene-adaptive parallel grid analysis*, doi:10.21203/rs.3.rs-9946108/v1 (Jun. 2026) | The supplied PDF is this preprint. It refers to "the conference version" but names no conference, and its Crossref record has no relation entries either. The preprint supports every statement the paper makes about [2]: 60 sequences, Hungarian matching, 71.03 % micro-F1, and the baseline reporting only timing and memory | Crossref record checked; text checked against the PDF |
| 2 | Related work said [2] "does not address the parallel implementation" | Now says [2]'s multi-process engine reaches 1.34× over sequential, with IPC serialization and sequential tracking binding; notes this is not directly comparable to our thread-level result | The preprint adds exactly this; omitting it would misrepresent [2] | Preprint p. 1 (abstract) and §3.7 |
| 3 | Speed estimation was described in one sentence; calibration was not described | New subsection III-E: two-point calibration (k = d/L, display→native coordinates in the web UI; CLI averaging with a coefficient-of-variation > 15 % warning), 3-cell-height displacement gate, Eq. (2), **15-sample moving average**, single scale with no perspective correction, inactive in benchmarks (k = 0), qualitative run shows implausible values | Requested by the author; every statement checked against `app.py`, `static/js/app.js`, `calibrate_pixels_per_meter.py`, `seq.py`, `rparallel_he3.py` | See §3 below |
| 4 | — | New Fig. 4 (speed module): (a) the author's calibration-UI screenshot (6 m, 8.721 px/m); (b) frame 847 of the author's screen recording of a calibrated run | Shows the implemented module from real project output. The caption states the speeds are unvalidated and that other frames show implausible values | Source images in `paper/assets/` (pixel-identical to the uploads); built by `make_figures.py` |
| 5 | Classification results were given only as a grouped-bar chart | New Table IV: per-class precision/recall/F1, mean ± std over the 12 videos (the paper's existing aggregation), plus the number of videos whose matched ground truth contains the class | Requested; makes the numbers readable without misleading. No new averaging method was introduced | `paper/final/audit_numbers.json` → `per_class_mean_over_videos` |
| 6 | Weak accuracy was explained only qualitatively | Added pooled class composition: 179 of 203 matched GT vehicles are cars, but 127 are labelled Van (a systematic Car→Van bias); Van/Bus occur in only 7/5 videos; Bus F1 = 0 in 10/12 videos | Explains *why* accuracy is low, from the data | `matched_gt_per_class_pooled`, `predicted_per_class_pooled`, `videos_with_zero_f1` |
| 7 | "A previous draft found …" (draft history inside the paper) | Rephrased: the narrow ceiling *happens* to lie near S_norm (1.22 vs 1.26); the proximity is coincidental | Draft history does not belong in a submitted paper | — |
| 8 | Mixed terms ("parallel version", "versions") | Unified on "sequential implementation", "two-lane parallel implementation", "threaded/unthreaded workload", "hot cells", "ROI-frame throughput", "vehicle-level" metrics | Consistency (prompt §7) | grep: no "version" left |
| 9 | Abstract, contributions and conclusion listed speed as an ordinary contribution | Now "calibration-based speed estimation (implemented but not validated)"; the conclusion states the three ratios and their interpretation explicitly | Prompt §3A and §5 | — |
| 10 | Pipeline diagram 9 pt wider than the text block | Wrapped in `\resizebox{\textwidth}`; speed box labelled "(unvalid.)" | Column overflow | 0 overfull boxes |
| 11 | Two charts had 5.8–6 pt text; vague axis titles | Ratios legend and workload tick labels raised to 6.5 pt; axis titles "time per frame (ms)", "ratio (dimensionless)", "mean annotated vehicles per frame" | Print legibility | Rendered and inspected |
| 12 | No plotting data shipped | `paper/figures/data/fig_{f1,workloads,ratios,amdahl}.csv` written by `make_figures.py` | Prompt §10B | — |
| 13 | Refs [Amdahl], [Cheung] had no DOI; online refs had no access date | DOIs 10.1145/1465482.1465560 and 10.1117/12.526886 added (Crossref); "accessed Nov. 25, 2026" added | IEEE style; verifiability | Crossref |
| 14 | `\TBD` macro defined | Removed | No placeholders remain | grep: 0 |

## 2. Reference audit

* 25 references; every one cited; every citation resolves (scripted check).
* DOIs verified against Crossref in this session or the previous one: all 22
  that have DOIs. The two Python documentation pages have no DOI (URL and
  access date given).
* Corrections made earlier and still in force: Unzueta 2012 (vol. 13, no. 2);
  Poolla & Saxena 2023; Beyer et al. 2019, vol. 21, no. 1; Al-hayanni et al.
  no. 4; Chen & **Hu**; Suzuki & **Abe**; Abbas et al. page 838 only.
* **Unresolved:** the IEEE 2026 conference version of Muthuraja &
  Sankaranarayanan (ISBN 979-8-3195-2457-7, "Robust road traffic density
  estimation using adaptive scene analysis and occupancy inference"): its
  conference name and pages are not printed in the camera-ready PDF and could
  not be found online. It is therefore **not cited**; if the authors confirm the
  venue, it can be added as a further reference.
* Note: the preprint describes the TPL2 baseline as using Python's
  `multiprocessing`. The TPL2 paper itself states `ThreadPoolExecutor`, so the
  paper follows the primary source.

## 3. Scientific claims verified against code and data

| Claim | Evidence |
|---|---|
| Calibration: k = pixel distance / real distance; UI clicks scaled to native resolution | `app.py` `/calibrate` (k = `pixel_distance / real_world_distance_m`); `static/js/app.js` L216–223 (`naturalWidth / clientWidth`) |
| CLI averages several pairs; warns if CV > 15 % | `calibrate_pixels_per_meter.py` L184, L196 |
| Speed = (Δd_px / k) / (Δn / f) · 3.6; gate = 3 × cell height | `seq.py` L579–589, L781–782; `rparallel_he3.py` L689 |
| Smoothing = moving average of the last 15 samples (**not** exponential) | `seq.py` L590–594 (`pop(0)` above 15, `np.mean`); same in `rparallel_he3.py` L690–694. The only exponential weighting is the label vote (1.15^age) |
| Speed inactive in benchmarks | every run's `user_input_data.json`: `pixels_per_meter = 0`; guard `PIXELS_PER_METER > 0` |
| Implausible speeds in the qualitative run | author's recording: frame 758 "427.3 km/h", frame 933 "180.0 km/h", frame 890 ≈ 4xx km/h, alongside 42–43 km/h |
| 60/60 equivalence checks identical | `benchmark_results_v2/equivalence.csv` (12 videos × 5 checks, incl. profiled pass) |
| Per-video F1 identical between v1 and v2 runs | `v1_v2_f1_identical: true` |
| Agreement is on the tested videos only | stated explicitly in §V-A |

## 4. Numerical consistency

`paper/final/audit_numbers.py` recomputes every quoted value from
`benchmark_results_v2/` (and v1 where cited). An automated cross-check found
**60/60 expected values present** in `paper_final.tex`, covering:

* Table V (all 20 cells);
* the 7 ratios with their spreads, ranges and counts;
* E(N);
* r;
* tier values and the tracking share;
* the Amdahl breakdown and ceilings;
* equivalence;
* frames and density range;
* v1 values;
* Table IV (9 cells);
* class composition.

No stale values were found ("1.24×", "0.391", "15.64", "6 cores", "TBD" are all absent).

Checks on the equations:

* **Amdahl at N = 2:** p = 0.362 gives 1/(0.638 + 0.181) = 1.22, and p = 0.666 gives 1.50.
* **E(N) = 0.860/N:** this gives 0.43, 0.215 and 0.108, which the paper prints as 0.43, 0.22 and 0.11.
* **Means and standard deviations:** sample std (n − 1) throughout.

No benchmark value was changed.

## 5. Figures and tables

10 figures: ROI frame, pipeline, processing stages, **speed module (new)**,
dispatch, benchmark overlay, F1, workloads, ratios, Amdahl. 4 tables:
baseline vs ours, parameters, **per-class P/R/F1 (new)**, workloads.

All are referenced in the text. All charts are built from repository data by
`paper/make_figures.py`, with their CSVs in `paper/figures/data/`. The images
are real outputs:

* DETRAC frames;
* the benchmark-run overlay;
* the author's calibration screenshot and screen recording.

## 6. Remaining limitations (kept in the paper)

* Single camera geometry with hand-tuned thresholds.
* Footprint-only classification, with a Car→Van bias and Bus F1 = 0 in most videos.
* Speed unvalidated: single scale, no perspective correction, implausible values observed.
* One laptop and one run per workload; the spreads are between-video variation.
* Pre-fix match rates (4/12, 5/12) come from development logs.
* Tracks younger than 5 frames are never scored.

## 7. Unresolved bibliographic details

Only the IEEE conference version of [2] (see §2). Nothing in the paper depends on it.

## 8. Page count

* **Stand-in layout:** 6 pages (letter paper, Times 10 pt, two columns, IEEE margins).
* **Previous version:** the stand-in gave 5 pages and the author's IEEEtran build gave 6.
* **Expected for this version:** about 7 pages in IEEEtran, which is under the 8-page limit.

## 9. IEEE compilation status

`IEEEtran.cls` is not installed in this environment and could not be downloaded:

* CTAN and GitHub are blocked by the network policy;
* a request to attach a GitHub repository for it was denied.

The final IEEEtran build must therefore be done on Overleaf or a local TeX
distribution, using `paper_final_overleaf.zip`.

The stand-in build (`paper/final/paper_final_layout_preview.pdf`) compiled
twice with:

* 0 errors;
* 0 undefined or multiply defined references;
* 0 overfull boxes;
* no "??".

The preview is not the camera-ready PDF.
