"""Count per-ROI events in user-specified frame windows.

For each recording, opens the filtered dF/F derivative memmap
(``r0p7_filtered_dff_dt.memmap.float32``) at
``<detection_root>/<rec>/final/suite2p/plane0/`` and runs the same
event detection used elsewhere in the pipeline:

    z = mad_z(dt[:, roi])              # robust z per ROI
    onsets = hysteresis_onsets(z, z_hi=z_enter, z_lo=z_exit, ...)

then counts onset *frames* per ROI inside each user-supplied
[start_frame, end_frame] window (inclusive).

Windows CSV format (header row optional):
    rec_id, w1_start, w1_end, w2_start, w2_end, ...
All bounds are integer frame indices (0-based). Pairs after column 0
are (start, end). Each row may have a different number of windows;
missing windows are blank.

Usage:
    python count_events_per_window.py windows.csv counts.csv
    python count_events_per_window.py windows.csv counts.csv \
        --detection-root E:\\sparse_plus_cellpose --fps 15.07 \
        --z-enter 3.5 --z-exit 1.5 --min-sep-s 0.1
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

import utils

DEFAULT_DETECTION_ROOT = Path(r"E:\sparse_plus_cellpose")
DEFAULT_PREFIX = "r0p7_filtered_"
DEFAULT_FPS = 15.07
DEFAULT_Z_ENTER = 3.5
DEFAULT_Z_EXIT = 1.5
DEFAULT_MIN_SEP_S = 0.1


def parse_windows_csv(path: Path) -> list[tuple[str, list[tuple[int, int]]]]:
    rows: list[tuple[str, list[tuple[int, int]]]] = []
    with open(path, newline="") as f:
        for i, raw in enumerate(csv.reader(f)):
            cells = [c.strip() for c in raw if c.strip() != ""]
            if not cells:
                continue
            rec_id, vals = cells[0], cells[1:]
            try:
                nums = [int(float(v)) for v in vals]
            except ValueError:
                if i == 0:
                    continue  # header row
                raise ValueError(
                    f"row {i} ({rec_id}): non-numeric window value in {vals}")
            if len(nums) % 2:
                raise ValueError(
                    f"row {i} ({rec_id}): odd number of window bounds "
                    f"({len(nums)}); expected start/end pairs")
            wins = list(zip(nums[0::2], nums[1::2]))
            rows.append((rec_id, wins))
    return rows


def detect_onsets(plane0: Path, prefix: str, fps: float,
                  z_enter: float, z_exit: float,
                  min_sep_s: float
                  ) -> tuple[list[np.ndarray], np.ndarray]:
    """Returns (onsets_by_roi, original_roi_indices).

    Onsets are returned as integer frame indices (0-based).
    ``original_roi_indices[j]`` is the suite2p stat.npy index of the j-th
    column in the filtered memmap (i.e. the position in the unfiltered
    ROI list). ``fps`` is still required because hysteresis_onsets uses
    it to convert ``min_sep_s`` to a frame gap."""
    _, _, dt, T, N = utils.s2p_open_memmaps(plane0, prefix=prefix)

    # Recover the keep-mask used to build the filtered memmap so we can
    # map filtered column j -> original suite2p ROI index.
    F_path = plane0 / "F.npy"
    if F_path.exists():
        n_total = int(np.load(F_path, mmap_mode="r").shape[0])
    else:
        n_total = N
    keep_mask = utils._load_keep_mask(plane0, n_total)
    original_idx = np.flatnonzero(keep_mask).astype(int)
    if original_idx.size != N:
        raise ValueError(
            f"keep-mask kept {original_idx.size} ROIs but memmap has {N}"
        )

    onsets_by_roi: list[np.ndarray] = []
    for j in range(N):
        z, _, _ = utils.mad_z(np.asarray(dt[:, j]))
        on = utils.hysteresis_onsets(
            z, z_hi=z_enter, z_lo=z_exit, fps=fps, min_sep_s=min_sep_s,
        )
        onsets_by_roi.append(np.asarray(on, dtype=np.int64))
    return onsets_by_roi, original_idx


def count_for_recording(onsets_by_roi: list[np.ndarray],
                        original_idx: np.ndarray,
                        windows: list[tuple[int, int]]) -> pd.DataFrame:
    n_rois = len(onsets_by_roi)
    out = pd.DataFrame({"roi": original_idx})
    for i, (s, e) in enumerate(windows, start=1):
        lo, hi = (s, e) if s <= e else (e, s)
        counts = np.empty(n_rois, dtype=int)
        for j, ts in enumerate(onsets_by_roi):
            if ts.size == 0:
                counts[j] = 0
            else:
                counts[j] = int(np.sum((ts >= lo) & (ts <= hi)))
        out[f"count_window_{i}"] = counts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows_csv", help="CSV: rec_id, w1_start, w1_end, ...")
    ap.add_argument("output_csv", help="output CSV path")
    ap.add_argument("--detection-root", type=Path,
                    default=DEFAULT_DETECTION_ROOT)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX,
                    help=f"memmap prefix (default {DEFAULT_PREFIX})")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--z-enter", type=float, default=DEFAULT_Z_ENTER)
    ap.add_argument("--z-exit", type=float, default=DEFAULT_Z_EXIT)
    ap.add_argument("--min-sep-s", type=float, default=DEFAULT_MIN_SEP_S)
    args = ap.parse_args()

    jobs = parse_windows_csv(Path(args.windows_csv))
    if not jobs:
        raise SystemExit(f"no rows parsed from {args.windows_csv}")
    max_w = max(len(w) for _, w in jobs)
    print(f"[counts] {len(jobs)} recordings; up to {max_w} windows each "
          f"(z_enter={args.z_enter}, z_exit={args.z_exit}, "
          f"min_sep_s={args.min_sep_s}, fps={args.fps})")

    parts: list[pd.DataFrame] = []
    windows_log: list[dict] = []
    for rec_id, wins in jobs:
        plane0 = args.detection_root / rec_id / "final" / "suite2p" / "plane0"
        if not plane0.is_dir():
            print(f"[counts] SKIP {rec_id}: missing {plane0}")
            continue
        try:
            onsets_by_roi, original_idx = detect_onsets(
                plane0, prefix=args.prefix, fps=args.fps,
                z_enter=args.z_enter, z_exit=args.z_exit,
                min_sep_s=args.min_sep_s,
            )
        except FileNotFoundError as ex:
            print(f"[counts] SKIP {rec_id}: {ex}")
            continue
        df = count_for_recording(onsets_by_roi, original_idx, wins)
        df.insert(0, "recording_id", rec_id)
        for i in range(len(wins) + 1, max_w + 1):
            df[f"count_window_{i}"] = ""
        parts.append(df)
        for i, (s, e) in enumerate(wins, start=1):
            windows_log.append({"recording_id": rec_id,
                                "window": i,
                                "start_frame": s, "end_frame": e})
        print(f"[counts] {rec_id}: {len(df)} ROIs x {len(wins)} windows "
              f"(total events={sum(int(t.size) for t in onsets_by_roi)})")

    if not parts:
        raise SystemExit("no recordings produced counts")

    cols = ["recording_id", "roi"] + [
        f"count_window_{i}" for i in range(1, max_w + 1)
    ]
    result = pd.concat(parts, ignore_index=True)[cols]
    out_path = Path(args.output_csv)
    result.to_csv(out_path, index=False)
    print(f"[counts] wrote {out_path} ({len(result)} rows)")

    sidecar = out_path.with_name(out_path.stem + "_windows.csv")
    pd.DataFrame(windows_log).to_csv(sidecar, index=False)
    print(f"[counts] wrote {sidecar}")


if __name__ == "__main__":
    main()
