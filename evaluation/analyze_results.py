"""
analyze_results.py
-------------------
Run this AFTER you have generated both:
  - seq_frame_metrics.json           (from seq.py)
  - rpar_frame_metrics.json          (from rparallel_he3_instrumented.py)

It prints a comparison summary and writes results_table.tex, a ready-to-paste
LaTeX table for the paper's Results section.

IMPORTANT — read this before trusting the numbers:
seq.py processes ONE ROI/lane per frame. rparallel_he3_instrumented.py
processes TWO ROIs/lanes per frame. That means total_time_seconds alone is
NOT a fair "speedup" ratio -- part of any slowdown in the parallel run is
simply double the workload, not a parallelization failure, and part of any
speedup could be hiding a bad *per-ROI* result. This script reports BOTH:
  1. naive_speedup      = seq_total_time / par_total_time      (workload-confounded, reported for completeness only)
  2. throughput_seq/par = (frames * lanes) / total_time         ("ROI-frames processed per second")
  3. normalized_speedup = throughput_par / throughput_seq       (fair, workload-normalized figure)

For the cleanest possible comparison, also build a "sequential dual-lane"
baseline (same 2-ROI workload as the parallel version, but processed with
plain loops instead of ThreadPoolExecutor) and diff that against the
parallel run instead of against seq.py. This script will use
baseline_frame_metrics.json for that comparison automatically if present.
"""
import json
import os
import sys
import statistics as stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROFILING_DIR = os.path.join(_HERE, "..", "profiling")

SEQ_FILE      = os.path.join(_PROFILING_DIR, "seq_frame_metrics.json")
PAR_FILE      = os.path.join(_PROFILING_DIR, "rpar_frame_metrics.json")
BASELINE_FILE = os.path.join(_PROFILING_DIR, "baseline_frame_metrics.json")  # optional, see docstring


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def summarize(name, m):
    if m is None:
        return None
    ft = m.get("frame_times_seconds", [])
    lanes = m.get("lanes", 1)
    frames = m["frames"]
    total_time = m["total_time_seconds"]
    throughput = (frames * lanes) / total_time if total_time > 0 else float("nan")
    summary = {
        "name": name,
        "implementation": m.get("implementation", name),
        "frames": frames,
        "lanes": lanes,
        "total_time_seconds": total_time,
        "mean_frame_time_ms": (stats.mean(ft) * 1000) if ft else float("nan"),
        "median_frame_time_ms": (stats.median(ft) * 1000) if ft else float("nan"),
        "stdev_frame_time_ms": (stats.pstdev(ft) * 1000) if len(ft) > 1 else 0.0,
        "fps": frames / total_time if total_time > 0 else float("nan"),
        "roi_frame_throughput": throughput,  # ROI-frames / second, workload-normalized
        "peak_memory_mb": max(m.get("memory_usages_mb", [0])) if m.get("memory_usages_mb") else None,
    }
    return summary


def print_summary(s):
    print(f"\n--- {s['name']} ({s['implementation']}) ---")
    print(f"  Frames processed        : {s['frames']}")
    print(f"  Lanes/ROIs per frame    : {s['lanes']}")
    print(f"  Total time (s)          : {s['total_time_seconds']:.3f}")
    print(f"  Mean frame time (ms)    : {s['mean_frame_time_ms']:.3f}")
    print(f"  Median frame time (ms)  : {s['median_frame_time_ms']:.3f}")
    print(f"  Stdev frame time (ms)   : {s['stdev_frame_time_ms']:.3f}")
    print(f"  Overall FPS             : {s['fps']:.3f}")
    print(f"  ROI-frame throughput    : {s['roi_frame_throughput']:.3f} ROI-frames/sec")
    if s["peak_memory_mb"] is not None:
        print(f"  Peak memory (MB)        : {s['peak_memory_mb']:.1f}")


def main():
    seq = summarize("Sequential (seq.py)", load(SEQ_FILE))
    par = summarize("Parallel (rparallel_he3_instrumented.py)", load(PAR_FILE))
    baseline = summarize("Sequential dual-lane baseline", load(BASELINE_FILE))

    if seq is None or par is None:
        print("Missing one or both benchmark files. Run seq.py and "
              "rparallel_he3_instrumented.py first, then re-run this script "
              "from the directory containing their output JSON files.")
        return

    print_summary(seq)
    print_summary(par)
    if baseline is not None:
        print_summary(baseline)

    naive_speedup = seq["total_time_seconds"] / par["total_time_seconds"]
    normalized_speedup = par["roi_frame_throughput"] / seq["roi_frame_throughput"]
    cores = os.cpu_count() or 1
    efficiency = normalized_speedup / cores

    print("\n=== Comparison ===")
    print(f"Naive speedup (seq total / par total, workload-CONFOUNDED): {naive_speedup:.3f}x")
    print(f"Normalized speedup (ROI-frame throughput, apples-to-apples): {normalized_speedup:.3f}x")
    print(f"Logical CPU cores available: {cores}")
    print(f"Parallel efficiency (normalized speedup / cores): {efficiency:.3f}  ({efficiency*100:.1f}%)")

    if baseline is not None:
        fair_speedup = baseline["total_time_seconds"] / par["total_time_seconds"]
        fair_efficiency = fair_speedup / cores
        print("\n--- Preferred comparison (dual-lane baseline vs. parallel; SAME workload) ---")
        print(f"Speedup: {fair_speedup:.3f}x   Efficiency: {fair_efficiency:.3f} ({fair_efficiency*100:.1f}%)")
    else:
        print("\n[NOTE] baseline_frame_metrics.json not found — only the workload-normalized "
              "throughput comparison above is apples-to-apples. Consider building the "
              "sequential dual-lane baseline described in the paper's Experimental Setup "
              "section for a cleaner, workload-matched speedup number.")

    # ---- Emit a ready-to-paste LaTeX table -------------------------------
    rows = [seq, par] + ([baseline] if baseline is not None else [])
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-frame timing comparison across implementations.}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Implementation} & \textbf{Lanes} & \textbf{Mean (ms)} & \textbf{FPS} & \textbf{Total (s)} \\")
    lines.append(r"\hline")
    for r in rows:
        lines.append(
            f"{r['implementation']} & {r['lanes']} & "
            f"{r['mean_frame_time_ms']:.2f} & {r['fps']:.2f} & "
            f"{r['total_time_seconds']:.2f} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open("results_table.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n[OK] Wrote results_table.tex — paste this into Section XII (Results) of paper.tex.")


if __name__ == "__main__":
    main()