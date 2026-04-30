"""Event frequency (events/second) per ROI per window, restricted to
user-selected clusters.

Combines the work of two earlier scripts:
  * ``count_events_per_window.py`` — runs hysteresis onset detection on the
    filtered dF/F derivative memmap and counts onsets in [start, end] frame
    windows.
  * ``filter_event_counts_by_cluster.py`` — keeps only ROIs that belong to
    the listed clusters for each recording.

The output reports **frequency in Hz** rather than raw counts:
    freq = onsets_in_window / window_duration_seconds
where window_duration_seconds = (end_frame - start_frame + 1) / fps
(inclusive bounds match count_events_per_window).

Inputs
------
windows.csv: rec_id, w1_start, w1_end, w2_start, w2_end, ...
selection.csv: rec_id, clusters (e.g. "C2+C4")

Per-recording cluster ROIs are read from
``<detection_root>/<rec>/final/suite2p/plane0/r0p7_filtered_cluster_results/Cn_rois.npy``
and their union is intersected with the detected ROIs (kept-mask indices).

Usage:
    python event_frequency_per_window.py windows.csv selection.csv freq.csv
    python event_frequency_per_window.py windows.csv selection.csv freq.csv \\
        --detection-root E:\\sparse_plus_cellpose --fps 15.07 \\
        --z-enter 3.5 --z-exit 1.5 --min-sep-s 0.1
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from count_events_per_window import (
    DEFAULT_DETECTION_ROOT,
    DEFAULT_FPS,
    DEFAULT_MIN_SEP_S,
    DEFAULT_PREFIX,
    DEFAULT_Z_ENTER,
    DEFAULT_Z_EXIT,
    detect_onsets,
    parse_windows_csv,
)
from filter_event_counts_by_cluster import (
    cluster_dir,
    load_cluster_rois,
    parse_selection,
)


def frequency_for_recording(onsets_by_roi: list[np.ndarray],
                            original_idx: np.ndarray,
                            kept_rois: np.ndarray,
                            windows: list[tuple[int, int]],
                            fps: float) -> pd.DataFrame:
    """Per-ROI events/second in each window, restricted to ``kept_rois``.

    ``original_idx[j]`` is the global ROI id for column j in the memmap.
    Returns one row per ROI in the intersection of ``original_idx`` and
    ``kept_rois``, with one ``freq_window_i_hz`` column per window.
    """
    keep_mask = np.isin(original_idx, kept_rois)
    keep_positions = np.flatnonzero(keep_mask)
    rec_rois = original_idx[keep_positions]

    df = pd.DataFrame({"roi": rec_rois})
    for i, (s, e) in enumerate(windows, start=1):
        lo, hi = (s, e) if s <= e else (e, s)
        duration_s = (hi - lo + 1) / fps
        freqs = np.empty(keep_positions.size, dtype=float)
        for k, j in enumerate(keep_positions):
            ts = onsets_by_roi[j]
            count = 0 if ts.size == 0 else int(np.sum((ts >= lo) & (ts <= hi)))
            freqs[k] = count / duration_s if duration_s > 0 else np.nan
        df[f"freq_window_{i}_hz"] = freqs
        df.attrs.setdefault("durations_s", []).append(duration_s)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows_csv", help="CSV: rec_id, w1_start, w1_end, ...")
    ap.add_argument("selection_csv",
                    help="CSV/TSV: rec_id, clusters (e.g. C2+C4)")
    ap.add_argument("output_csv", help="output CSV path")
    ap.add_argument("--detection-root", type=Path,
                    default=DEFAULT_DETECTION_ROOT)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX,
                    help=f"memmap/cluster-folder prefix (default {DEFAULT_PREFIX})")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--z-enter", type=float, default=DEFAULT_Z_ENTER)
    ap.add_argument("--z-exit", type=float, default=DEFAULT_Z_EXIT)
    ap.add_argument("--min-sep-s", type=float, default=DEFAULT_MIN_SEP_S)
    args = ap.parse_args()

    jobs = parse_windows_csv(Path(args.windows_csv))
    if not jobs:
        raise SystemExit(f"no rows parsed from {args.windows_csv}")
    selection = parse_selection(Path(args.selection_csv))
    if not selection:
        raise SystemExit(f"no selections parsed from {args.selection_csv}")

    max_w = max(len(w) for _, w in jobs)
    print(f"[freq] {len(jobs)} recordings (windows); {len(selection)} in "
          f"cluster selection; up to {max_w} windows each "
          f"(z_enter={args.z_enter}, z_exit={args.z_exit}, "
          f"min_sep_s={args.min_sep_s}, fps={args.fps})")

    parts: list[pd.DataFrame] = []
    windows_log: list[dict] = []
    summary_rows: list[dict] = []

    for rec_id, wins in jobs:
        if rec_id not in selection:
            print(f"[freq] SKIP {rec_id}: not in selection CSV")
            continue
        tokens = selection[rec_id]

        plane0 = args.detection_root / rec_id / "final" / "suite2p" / "plane0"
        if not plane0.is_dir():
            print(f"[freq] SKIP {rec_id}: missing {plane0}")
            continue

        cl_dir = cluster_dir(args.detection_root, rec_id, args.prefix)
        try:
            kept_rois = load_cluster_rois(cl_dir, tokens)
        except FileNotFoundError as ex:
            print(f"[freq] SKIP {rec_id}: missing cluster file {ex}")
            continue

        try:
            onsets_by_roi, original_idx = detect_onsets(
                plane0, prefix=args.prefix, fps=args.fps,
                z_enter=args.z_enter, z_exit=args.z_exit,
                min_sep_s=args.min_sep_s,
            )
        except FileNotFoundError as ex:
            print(f"[freq] SKIP {rec_id}: {ex}")
            continue

        df = frequency_for_recording(
            onsets_by_roi, original_idx, kept_rois, wins, fps=args.fps,
        )
        if df.empty:
            print(f"[freq] {rec_id}: 0 ROIs after intersecting clusters "
                  f"({'+'.join(tokens)}) with detection set; skipping")
            continue

        df.insert(0, "recording_id", rec_id)
        df.insert(2, "clusters_kept", "+".join(tokens))
        for i in range(len(wins) + 1, max_w + 1):
            df[f"freq_window_{i}_hz"] = ""
        parts.append(df)

        durations = df.attrs.get("durations_s", [])
        for i, ((s, e), dur) in enumerate(zip(wins, durations), start=1):
            windows_log.append({
                "recording_id": rec_id,
                "window": i,
                "start_frame": s, "end_frame": e,
                "duration_s": dur,
            })
            mean_hz = float(df[f"freq_window_{i}_hz"].mean())
            summary_rows.append({
                "recording_id": rec_id,
                "clusters_kept": "+".join(tokens),
                "window": i,
                "duration_s": dur,
                "n_rois": int(len(df)),
                "mean_freq_hz": mean_hz,
                "total_events": int(round(mean_hz * dur * len(df))),
            })

        print(f"[freq] {rec_id} ({'+'.join(tokens)}): "
              f"{len(df)} ROIs x {len(wins)} windows")

    if not parts:
        raise SystemExit("no recordings produced output")

    cols = ["recording_id", "roi", "clusters_kept"] + [
        f"freq_window_{i}_hz" for i in range(1, max_w + 1)
    ]
    result = pd.concat(parts, ignore_index=True)[cols]
    out_path = Path(args.output_csv)
    result.to_csv(out_path, index=False)
    print(f"[freq] wrote {out_path} ({len(result):,} rows)")

    win_sidecar = out_path.with_name(out_path.stem + "_windows.csv")
    pd.DataFrame(windows_log).to_csv(win_sidecar, index=False)
    print(f"[freq] wrote {win_sidecar}")

    sum_sidecar = out_path.with_name(out_path.stem + "_summary.csv")
    pd.DataFrame(summary_rows).to_csv(sum_sidecar, index=False)
    print(f"[freq] wrote {sum_sidecar}")


if __name__ == "__main__":
    main()
