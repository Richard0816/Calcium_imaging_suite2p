"""Filter ``event_count_per_window.csv`` rows by user-selected clusters.

For each recording listed in the selection table, keep only the ROIs that
belong to the listed clusters. Cluster ROI lists live at
``<detection_root>/<rec>/final/suite2p/plane0/r0p7_filtered_cluster_results/C{n}_rois.npy``
and contain global ROI indices that match the ``roi`` column produced by
``count_events_per_window.py``.

Selection table format (CSV or TSV; header row optional):
    slice                              clusters_kept
    E054_1-20_Abeta_9DIV_15            C2+C4
    E054_1-100_Abeta_9DIV_18           C1
    ...

Usage:
    python filter_event_counts_by_cluster.py \\
        event_count_per_window.csv selection.csv filtered.csv
    python filter_event_counts_by_cluster.py \\
        event_count_per_window.csv selection.csv filtered.csv \\
        --detection-root E:\\sparse_plus_cellpose --prefix r0p7_filtered_
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DETECTION_ROOT = Path(r"E:\sparse_plus_cellpose")
DEFAULT_PREFIX = "r0p7_filtered_"
_SPLIT = re.compile(r"[\t,]+")
_CLUSTER_TOKEN = re.compile(r"C\d+", re.IGNORECASE)


def parse_selection(path: Path) -> dict[str, list[str]]:
    """Returns {recording_id: [cluster_token, ...]} from a CSV/TSV file.

    Cluster columns may use any of: ``C2+C4``, ``C2,C4``, ``C2 C4``.
    Tokens are uppercased so ``c1`` == ``C1``.
    """
    sel: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip().lstrip("﻿")
            if not line:
                continue
            cells = [c.strip() for c in _SPLIT.split(line) if c.strip()]
            if len(cells) < 2:
                continue
            rec_id, rest = cells[0], " ".join(cells[1:])
            tokens = _CLUSTER_TOKEN.findall(rest)
            if not tokens:
                if i == 0:
                    continue  # header row
                print(f"[filter] WARN row {i} ({rec_id}): no Cn tokens in {rest!r}")
                continue
            sel[rec_id] = [t.upper() for t in tokens]
    return sel


def cluster_dir(detection_root: Path, rec_id: str, prefix: str) -> Path:
    return (detection_root / rec_id / "final" / "suite2p" / "plane0"
            / f"{prefix}cluster_results")


def load_cluster_rois(cl_dir: Path, tokens: list[str]) -> np.ndarray:
    """Union of global ROI indices across the listed cluster files."""
    out: list[int] = []
    for tok in tokens:
        path = cl_dir / f"{tok}_rois.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        rois = np.load(path).astype(int).ravel().tolist()
        out.extend(rois)
    return np.unique(np.asarray(out, dtype=int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("counts_csv", help="event_count_per_window.csv")
    ap.add_argument("selection_csv",
                    help="CSV/TSV: rec_id, clusters (e.g. C2+C4)")
    ap.add_argument("output_csv", help="filtered output CSV path")
    ap.add_argument("--detection-root", type=Path,
                    default=DEFAULT_DETECTION_ROOT)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX,
                    help=f"cluster-results folder prefix (default {DEFAULT_PREFIX})")
    args = ap.parse_args()

    selection = parse_selection(Path(args.selection_csv))
    if not selection:
        raise SystemExit(f"no selections parsed from {args.selection_csv}")

    counts = pd.read_csv(args.counts_csv)
    if "recording_id" not in counts.columns or "roi" not in counts.columns:
        raise SystemExit(
            f"{args.counts_csv} must have 'recording_id' and 'roi' columns; "
            f"got {list(counts.columns)}"
        )

    print(f"[filter] {len(counts):,} input rows across "
          f"{counts['recording_id'].nunique()} recordings")
    print(f"[filter] {len(selection)} recordings in selection")

    kept_parts: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for rec_id, tokens in selection.items():
        sub = counts[counts["recording_id"] == rec_id]
        if sub.empty:
            print(f"[filter] SKIP {rec_id}: not found in counts CSV")
            continue
        cl_dir = cluster_dir(args.detection_root, rec_id, args.prefix)
        try:
            kept_rois = load_cluster_rois(cl_dir, tokens)
        except FileNotFoundError as ex:
            print(f"[filter] SKIP {rec_id}: missing cluster file {ex}")
            continue

        kept = sub[sub["roi"].isin(kept_rois)].copy()
        kept.insert(2, "clusters_kept", "+".join(tokens))
        kept_parts.append(kept)

        summary_rows.append({
            "recording_id": rec_id,
            "clusters_kept": "+".join(tokens),
            "rois_in_clusters": int(len(kept_rois)),
            "rois_in_counts": int(len(sub)),
            "rows_kept": int(len(kept)),
        })
        print(f"[filter] {rec_id} ({'+'.join(tokens)}): "
              f"{len(kept)}/{len(sub)} rows kept "
              f"(cluster ROIs={len(kept_rois)})")

    if not kept_parts:
        raise SystemExit("no recordings produced filtered rows")

    result = pd.concat(kept_parts, ignore_index=True)
    out_path = Path(args.output_csv)
    result.to_csv(out_path, index=False)
    print(f"[filter] wrote {out_path} ({len(result):,} rows)")

    sidecar = out_path.with_name(out_path.stem + "_summary.csv")
    pd.DataFrame(summary_rows).to_csv(sidecar, index=False)
    print(f"[filter] wrote {sidecar}")


if __name__ == "__main__":
    main()
