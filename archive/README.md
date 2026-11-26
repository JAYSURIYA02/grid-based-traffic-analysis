# archive/

Nothing in this folder is used by the current code, benchmark, or paper.
It is kept, unmodified, so that earlier results remain inspectable.

| Folder | What it is |
|---|---|
| `pre-fix/` | Benchmark/profiling artifacts produced **before** the timing-instrumentation fix and the two preprocessing fixes. They conflict with the paper and must not be quoted as current results. |
| `benchmark-v1-engines/` | The instrumented script copies that produced `benchmark_results/` (the data behind the current paper tables). Their detection/tracking/evaluation logic is now part of the shipped `seq.py` / `rparallel_he3.py`. |

See the README in each sub-folder.
