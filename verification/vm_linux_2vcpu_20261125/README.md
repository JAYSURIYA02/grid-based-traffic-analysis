# Verification run — Linux VM, 2 vCPU (NOT the paper's benchmark machine)

**Purpose:** check, on the shipped `seq.py` / `rparallel_he3.py` (commit
888e3ae) and orchestrator, that (1) all four workloads run end-to-end on the
paper's 12 videos and (2) the equivalence claims hold. **Do not use these
timings in the paper.** The machine is a 2-vCPU Linux VM (Python 3.10.12,
OpenCV 5.0.0) on the same AMD Ryzen 5 7235HS host, not the Windows 11 /
Python 3.12 / OpenCV 4.12 / 8-logical-CPU configuration of the paper; see
`system_info.json`.

Command (run in <3-minute chunks, hence `--resume`; profiled passes skipped):

    BTP_OUTPUT_ROOT=verification/vm_linux_2vcpu_20261125 \
      python benchmarking/orchestrate_benchmark.py --no-profile --resume

## Results

- Runs: 48/48 completed (12 videos × 4 workloads, clean pass).
- **Equivalence: 48/48 checks identical** (`equivalence.csv`):
  Lane 1 of B (threaded, 1 lane), C (serial, 2 lanes) and D (threaded,
  2 lanes) is bit-identical to `seq.py` (A) on every video — vehicle and
  grid confusion matrices, ID-count summary, per-class P/R/F1 — and Lane 2
  is identical between C and D. B-Lane1 = D-Lane1 also shows Lane 2 does
  not perturb Lane 1. Absolute F1 differs from the paper's tables because
  OpenCV differs (5.0 vs 4.12); equivalence is a within-environment check.

## Timing observation (environment-specific; to be checked on the benchmark machine)

On this 2-vCPU VM the threaded 2-lane run was **slower** than the same
2-lane `LaneTracker` code with no threads (same-workload ratio
T_C/T_D = 0.72 ± 0.06; per-frame 0.63 ± 0.06), yet the workload-normalized
end-to-end ratio 2·T_A/T_D was still 1.20 ± 0.14. Here, then, the
normalized gain comes from amortising once-per-frame work (decode, video
write, full-frame preprocessing) over two lanes, not from threading. Whether
this also holds with 8 logical CPUs must be measured with the full
orchestrator run on the Windows machine (Table VII of the paper).
