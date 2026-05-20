#!/usr/bin/env python3
"""Compare two lidar_proc_summary.parquet trees for the MapMOS Step-1 gate.

Exits 0 iff every chunk's summary parquet matches on the columns that
the static/dynamic regression contract cares about (plan §1).

Usage:
    python3 scripts/compare_summaries.py <baseline_root> <candidate_root>

Where each <root> is a `data/artifacts/raw/<bag_id>` directory containing
`chunks/<chunk_id>/lidar_proc_summary.parquet` files.

The script uses pyarrow (a project dep, available in the lidar_preprocessing
container) instead of pandas so it works in both places. If you prefer
running on the host, `pip install pyarrow` and you're set.
"""

from __future__ import annotations

import glob
import os
import sys

import pyarrow.parquet as pq

# Columns the Step-1 regression invariant requires to match. n_sweeps_valid
# is included so a chunk that silently dropped sweeps after enabling MapMOS
# also trips the comparison.
_COLS = ["chunk_id", "n_sweeps_valid", "n_points_static", "n_points_dynamic"]


def _load(path: str) -> list[tuple]:
    """Return a sorted-by-chunk_id list of (chunk_id, n_valid, n_static, n_dyn) tuples."""
    table = pq.read_table(path, columns=_COLS)
    cols = {name: table.column(name).to_pylist() for name in _COLS}
    rows = list(
        zip(
            cols["chunk_id"],
            cols["n_sweeps_valid"],
            cols["n_points_static"],
            cols["n_points_dynamic"],
        )
    )
    return sorted(rows, key=lambda r: r[0])


def _fmt(rows: list[tuple]) -> str:
    header = "  ".join(_COLS)
    body = "\n".join("  ".join(str(v) for v in r) for r in rows)
    return f"{header}\n{body}"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    baseline_root, candidate_root = argv[1], argv[2]
    pattern = f"{baseline_root}/chunks/*/lidar_proc_summary.parquet"
    bases = sorted(glob.glob(pattern))
    if not bases:
        print(f"no summary parquets under {pattern}", file=sys.stderr)
        return 2

    mismatches: list[tuple[str, list[tuple], list[tuple]]] = []
    for base in bases:
        rel = os.path.relpath(base, baseline_root)
        cand = os.path.join(candidate_root, rel)
        if not os.path.exists(cand):
            print(f"MISSING {rel} in candidate", file=sys.stderr)
            mismatches.append((rel, _load(base), []))
            continue
        a = _load(base)
        b = _load(cand)
        if a != b:
            mismatches.append((rel, a, b))

    if mismatches:
        for rel, a, b in mismatches:
            print(f"MISMATCH {rel}")
            print("baseline:")
            print(_fmt(a))
            print("candidate:")
            print(_fmt(b))
            print()
        return 1
    print(f"OK — all summaries match across {len(bases)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
