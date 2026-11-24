# archive/pre-fix/ — superseded benchmark artifacts (do not cite)

Moved here on 2026-11-25 from their original locations. Original files are
preserved byte-for-byte; only their location changed.

| File(s) | Original location | Why it is archived |
|---|---|---|
| `benchmarking/benchmark_results.{csv,txt}`, `benchmarking/benchmark_runner.py` | `benchmarking/` | Single video (MVI_63521, 6×6), 3 trials. Reports seq/par = 0.893× ("sequential is faster"), but is internally inconsistent: the three trials report identical mean totals (80.06 s, ±0.00) while their recorded wall times are ~111–112 s, which suggests a stale metrics file was re-read on every trial. Predates the `waitKey`/JPEG timing fix and the preprocessing fixes. |
| `profiling/*` | `profiling/` | Single-run profiles from Aug–Sep 2026. The sequential and parallel runs used **different videos** (seq_frame_metrics: MVI_63521; rpar_frame_metrics: MVI_40191) and a **10×10** grid, so they are not comparable with each other or with the paper (6×6). `profile_seq_tottime.txt` shows the timing artifact that the paper describes: `waitKey` 43.0 s and `imwrite` 7.6 s of an 83.5 s run. |
| `evaluation/results_table.tex`, `evaluation/analyze_results.py` | `evaluation/` | `analyze_results.py` compared one seq run with one parallel run from `profiling/`; `results_table.tex` is its output from the mismatched runs above. Superseded by `benchmarking/orchestrate_benchmark.py`. (Paths are relative, so the script still runs against `archive/pre-fix/profiling/`.) |

The "~3.3×" speedup mentioned in earlier drafts of the paper is not
reproduced by any file here; the paper now describes it only as an
earlier, unarchived measurement.
