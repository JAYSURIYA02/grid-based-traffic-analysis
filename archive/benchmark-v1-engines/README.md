# archive/benchmark-v1-engines/ — scripts that produced `benchmark_results/`

`benchmark_results/` (12 UA-DETRAC videos, Nov 2026, AMD Ryzen 5 7235HS,
Windows 11) — the data behind every table in the current paper — was
produced by `orchestrate_benchmark.py` running these two copies:

- `seq_bench.py` — `seq.py` + `BTP_*` switches (display/profile/frame-dump),
  the `pixels_per_meter > 0` guard and `eval_metrics.json` export.
- `rparallel_he3_bench.py` — `rparallel_he3.py` + the 7×7 pre-difference
  blur, channel-aware histogram equalisation, `BTP_*` switches and metrics
  export.

Since commit a59f530 the shipped `seq.py` and `rparallel_he3.py` contain all
of this (detection/tracking/evaluation code identical to these copies), and
since 888e3ae they additionally have symmetric per-frame timing, sequential
memory export, hardware/software provenance and the 1-lane / serial
workload modes. New benchmarks run the shipped files directly.

Known limitations of the v1 instrumentation (documented in the paper,
Sec. IV-D): the sequential per-frame timer stopped before
`VideoWriter.write` while the parallel one stopped after it (and after the
preview call); `seq_frame_metrics.json` had no `memory_usages_mb`.

`seq_bench.diff` / `rparallel_he3_bench.diff` are the original diffs and are
out of date (e.g. `seq_bench.diff` does not show the `waitKey`/JPEG gating).
`rparallel_he3_instrumented.py` is an earlier intermediate copy.
