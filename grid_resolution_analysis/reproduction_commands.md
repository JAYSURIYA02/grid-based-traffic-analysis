# Reproduction commands (corrected grid-resolution analysis)

Run every command from the repository root. Each step is resumable: rerun with `--resume`, or simply rerun, because results are cached. Run and replay failures are appended to `grid_resolution_analysis/failures.log`.

**Environment of the reported run:**
- Linux VM, 2 vCPU
- Python 3.10.12
- OpenCV 5.0.0
- NumPy 2.2.6
- SciPy not required

The production files `seq.py` and `rparallel_he3.py` are only imported; they are never modified.

```bash
# 0. Site grouping (median backgrounds, contact sheet, automatic clustering)
python grid_resolution_analysis/site_grouping.py bg            # repeat until "done"
python grid_resolution_analysis/site_grouping.py save 0.4      # site_groups.json + site_contact_sheet.png
#    The groups used for CV are the reviewed, conservative ones in site_groups_manual.json

# 1. Baseline runs: all six grids on the 26 validation videos
#    (benchmark runs already exist from b7c3756)
python grid_resolution_analysis/experiment.py run --set validation --grids 4x6,6x6,6x8,8x8,8x10,10x12 --resume --jobs 2 --budget 140

# 2. Timing trials: serial, 3 trials x 3 videos x 6 grids, nothing else running
python grid_resolution_analysis/timing_trials.py run --budget 150   # repeat until "all trials done"
python grid_resolution_analysis/timing_trials.py summary            # timing_trials.csv, timing_trials_summary.json

# 3. Cross-validation on validation only, then freeze the configuration
python grid_resolution_analysis/cross_validation.py prepare --set validation
python grid_resolution_analysis/cross_validation.py cv
#    -> cross_validation_results.csv, cross_validation_oof_summary.csv, selected_configuration.json
python -c "import json;c=json.load(open('grid_resolution_analysis/selected_configuration.json'));json.dump({g:v['thresholds'] for g,v in c['validation_scores_all_grids'].items()},open('grid_resolution_analysis/frozen_thresholds_by_grid.json','w'),indent=1)"

# 4. Site-independent test set: test_set.json was written only after step 3
python grid_resolution_analysis/experiment.py run --set test --grids 6x8,6x6,8x8,10x12 --resume --jobs 2 --budget 140
python grid_resolution_analysis/cross_validation.py prepare --set test --grids 6x6,6x8,8x8,10x12
python grid_resolution_analysis/cross_validation.py prepare --set benchmark --grids 6x6,6x8,8x8,10x12

# 5. Counting evaluation (identical rules for every grid)
G6=4x6,6x6,6x8,8x8,8x10,10x12; O=grid_resolution_analysis/counting_results.csv; T=grid_resolution_analysis/frozen_thresholds_by_grid.json
python grid_resolution_analysis/counting_evaluation.py --set validation --grids $G6 --out $O
python grid_resolution_analysis/counting_evaluation.py --set benchmark  --grids $G6 --out $O --append
python grid_resolution_analysis/counting_evaluation.py --set test --grids 6x6,6x8,8x8,10x12 --out $O --append
python grid_resolution_analysis/counting_evaluation.py --set benchmark --grids 6x6,6x8,8x8,10x12 --thresholds $T --out $O --append
python grid_resolution_analysis/counting_evaluation.py --set test --grids 6x6,6x8,8x8,10x12 --thresholds $T --out $O --append

# 6. Fragmentation diagnosis: experimental variants run separately (runs_variants/)
python grid_resolution_analysis/experiment.py run --set validation --variant conn8  --grids 10x12,8x8 --resume --jobs 2 --budget 140
python grid_resolution_analysis/experiment.py run --set validation --variant bridge --grids 10x12,8x8 --resume --jobs 2 --budget 140
python grid_resolution_analysis/counting_evaluation.py --set validation --variant conn8  --grids 8x8,10x12 --out $O --append
python grid_resolution_analysis/counting_evaluation.py --set validation --variant bridge --grids 8x8,10x12 --out $O --append
python grid_resolution_analysis/fragmentation_analysis.py --set validation   # fragmentation_validation.csv, fragmentation_detection_stage.json

# 7. Aggregates, figures, verification
python grid_resolution_analysis/final_analysis.py      # final_numbers.json, grid_tradeoff_comparison.png, counting_comparison.png
python grid_resolution_analysis/verify_corrected.py    # Part-10 checks
```

**Notes:**
- `--jobs 2` runs two videos at once for throughput only. Performance figures come from step 2 alone.
- `--budget N` stops launching new videos after N seconds. It was used because each remote shell call is limited to about 3 minutes.
- The original first-round analysis can still be regenerated with the earlier commands in the header of `experiment.py`. Its outputs (`results.csv`, `analysis_report.md`, `summary.json`, `tuning_*.json`, the four original PNGs) are not touched by the corrected pipeline.
