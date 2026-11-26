"""
check_equivalence.py -- exact cross-implementation equivalence check.

Compares two eval_metrics.json payloads (one lane each) field by field:
vehicle (3x3) and grid (4x4) confusion matrices, ID-based counting summary,
and per-class precision/recall/F1. Equality is EXACT (no tolerance): two
deterministic implementations of the same algorithm on the same input must
agree bit-for-bit, so any difference indicates a real divergence.

Usage (single pair of run directories):
    python benchmarking/check_equivalence.py <runA_dir> <laneA> <runB_dir> <laneB>
where lane is "Lane1"/"Lane2" for rparallel_he3.py runs and "-" for seq.py runs
(seq.py writes a single-lane payload, not a {lane: payload} dict).
"""
import json
import sys
from pathlib import Path

COMPARED_FIELDS = [
    "confusion_matrix_vehicle",
    "confusion_matrix_grid",
    "unique_gt_vehicles",
    "unique_predicted_tracks",
    "matched_gt_vehicles",
    "unmatched_gt_vehicles",
    "matched_predicted_tracks",
    "unmatched_predicted_tracks",
    "metrics",
]


def load_lane_payload(run_dir, lane):
    path = Path(run_dir) / "eval_metrics.json"
    with open(path) as f:
        data = json.load(f)
    if lane in (None, "-", ""):
        return data
    return data.get(lane)


def compare_payloads(a, b):
    """Return (equal: bool, differing_fields: list[str])."""
    if a is None or b is None:
        return False, ["<missing payload>"]
    diffs = [k for k in COMPARED_FIELDS if a.get(k) != b.get(k)]
    return (not diffs), diffs


def main(argv):
    if len(argv) != 5:
        print(__doc__)
        return 2
    _, run_a, lane_a, run_b, lane_b = argv
    a = load_lane_payload(run_a, lane_a)
    b = load_lane_payload(run_b, lane_b)
    equal, diffs = compare_payloads(a, b)
    print("IDENTICAL" if equal else f"DIFFERENT in: {', '.join(diffs)}")
    return 0 if equal else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
