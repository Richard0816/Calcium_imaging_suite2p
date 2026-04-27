"""
ROI Manual Curation GUI
-----------------------
Displays, for each ROI across a list of recordings:
    [ mean image ] [ max proj + ROI ] [ ROI footprint ] [ dF/F trace ]

ROIs are ranked by the trained cell filter's predicted_cell_prob.npy.

Three ordering modes (SORT_MODE):
    "uncertainty"  - lowest |p - 0.5| first (ambiguous cases first; best for
                     active learning / finding model disagreements)
    "pooled"       - highest p across the whole dataset first
    "per_recording" - finish one recording at a time, highest p first

User presses '1' = cell, '0' = not a cell. Results saved to F:\\roi_curation.csv
as: recording_ID, ROI_number, user_defined_cell

Keybinds:
    1          : label as cell
    0          : label as not a cell
    Left arrow : undo last label (removes last row from CSV)
    Esc / q    : quit (progress is already saved)

Resume-safe: re-running skips any (recording_ID, ROI_number) pairs already in
the output CSV.
"""

from __future__ import annotations

import csv
import sys
import tkinter as tk
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Make utils importable. Adjust if utils.py lives elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils  # noqa: E402


# ---------------------------- CONFIG ----------------------------

ROOTS: List[str] = [
    r"G:\sparse_plus_cellpose\2025-11-05-0001\final"
]

OUTPUT_CSV = Path(r"F:\roi_curation.csv")
DFF_PREFIX = "r0p7_"
FPS_FALLBACK = 15.07

# The trained cell filter writes these to each suite2p/plane0/ folder.
SCORES_FILENAME = "predicted_cell_prob.npy"

# "uncertainty"   = rank by |p - 0.5| ascending (most ambiguous first).
#                   Best for active learning against the trained model.
# "pooled"        = rank by p descending across all recordings.
# "per_recording" = finish one recording at a time, p descending within each.
SORT_MODE = "uncertainty"

# ---------------------------------------------------------------


def load_recording(root: Path):
    """Load everything needed for one recording."""
    plane0 = root / "suite2p" / "plane0"

    stat = np.load(plane0 / "stat.npy", allow_pickle=True)
    ops = np.load(plane0 / "ops.npy", allow_pickle=True).item()

    # Use the raw mean image (not meanImgE), since meanImgE is high-pass
    # filtered and visually overlaps with the max projection.
    mean_img = np.asarray(ops["meanImg"])

    # Max projection (falls back to mean if not present).
    # Suite2p's ops['max_proj'] is cropped to ops['yrange'] x ops['xrange'];
    # pad it back into a full-frame image so ROI coords (full-frame) align.
    max_img = ops.get("max_proj", None)
    if max_img is None:
        max_img = ops.get("maxImg", None)
    if max_img is None:
        max_img = mean_img
    max_img = np.asarray(max_img)

    if max_img.shape != mean_img.shape:
        full = np.zeros_like(mean_img, dtype=max_img.dtype)
        yr = ops.get("yrange", (0, max_img.shape[0]))
        xr = ops.get("xrange", (0, max_img.shape[1]))
        y0, y1 = int(yr[0]), int(yr[1])
        x0, x1 = int(xr[0]), int(xr[1])
        # Guard against any mismatch between yrange/xrange and max_proj shape.
        h = min(y1 - y0, max_img.shape[0])
        w = min(x1 - x0, max_img.shape[1])
        full[y0:y0 + h, x0:x0 + w] = max_img[:h, :w]
        max_img = full

    # dF/F memmap
    dff, _, _, T, N = utils.s2p_open_memmaps(plane0, prefix=DFF_PREFIX)

    # Cell filter probabilities live in suite2p/plane0/predicted_cell_prob.npy
    scores_path = plane0 / SCORES_FILENAME
    if not scores_path.exists():
        # legacy fallback: roi_scores.npy at the recording root
        legacy = root / "roi_scores.npy"
        if legacy.exists():
            scores_path = legacy
        else:
            raise FileNotFoundError(
                f"No {SCORES_FILENAME} in {plane0} (and no roi_scores.npy at {root}). "
                "Run cellfilter.predict first."
            )
    scores = np.load(scores_path, allow_pickle=False)

    if len(scores) != len(stat):
        raise ValueError(
            f"{root}: scores has {len(scores)} entries but stat has {len(stat)}"
        )

    # fps
    try:
        fps = utils.get_fps_from_notes(str(plane0), default_fps=FPS_FALLBACK)
    except Exception:
        fps = FPS_FALLBACK

    return {
        "root": root,
        "recording_id": root.name,
        "plane0": plane0,
        "stat": stat,
        "mean_img": mean_img,
        "max_img": max_img,
        "dff": dff,
        "T": T,
        "N": N,
        "scores": scores,
        "fps": fps,
    }


def _find_scores_path(root: Path) -> Path | None:
    """Return path to the score file for a recording, or None if missing."""
    p = root / "suite2p" / "plane0" / SCORES_FILENAME
    if p.exists():
        return p
    # legacy fallback
    legacy = root / "roi_scores.npy"
    if legacy.exists():
        return legacy
    legacy2 = root / "suite2p" / "plane0" / "roi_scores.npy"
    if legacy2.exists():
        return legacy2
    return None


def build_work_queue(roots: List[Path], done: set) -> List[Tuple[int, int, float]]:
    """
    Return list of (rec_idx, roi_idx, score) ordered by SORT_MODE.
    Skips any (recording_id, roi_idx) already labeled in `done`.
    Missing recordings / missing score files are warned and skipped.

    `score` is the predicted_cell_prob value as-is (not transformed to
    uncertainty); the uncertainty sort applies an |p-0.5| key only for
    ordering, so the status bar still shows the actual probability.
    """
    entries = []
    for ri, root in enumerate(roots):
        rec_id = root.name
        if not root.exists():
            print(f"[skip] root missing: {root}")
            continue
        scores_path = _find_scores_path(root)
        if scores_path is None:
            print(f"[skip] no {SCORES_FILENAME} (or legacy roi_scores.npy) for {rec_id}")
            continue
        try:
            scores = np.load(scores_path, allow_pickle=False)
        except Exception as ex:
            print(f"[skip] failed to load scores for {rec_id}: {ex}")
            continue

        per_rec = []
        for roi_idx, s in enumerate(scores):
            if (rec_id, int(roi_idx)) in done:
                continue
            per_rec.append((ri, int(roi_idx), float(s)))

        if SORT_MODE == "per_recording":
            per_rec.sort(key=lambda x: x[2], reverse=True)
        elif SORT_MODE == "uncertainty":
            # most ambiguous first, within each recording's contribution to
            # the list (pooled uncertainty sort happens at the end)
            pass
        entries.extend(per_rec)

    if SORT_MODE == "pooled":
        entries.sort(key=lambda x: x[2], reverse=True)
    elif SORT_MODE == "uncertainty":
        entries.sort(key=lambda x: abs(x[2] - 0.5))
    return entries


def load_done_set(csv_path: Path) -> set:
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add((row["recording_ID"], int(row["ROI_number"])))
            except (KeyError, ValueError):
                continue
    return done


def ensure_csv_header(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["recording_ID", "ROI_number", "user_defined_cell"])


def append_label(csv_path: Path, rec_id: str, roi: int, label: int):
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([rec_id, roi, label])


def remove_last_row(csv_path: Path) -> Tuple[str, int, int] | None:
    """Remove the last data row from the CSV. Returns the removed row or None."""
    if not csv_path.exists():
        return None
    with open(csv_path, "r", newline="") as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return None  # only header, nothing to undo
    last = lines[-1].strip()
    if not last:
        # skip trailing empty line
        lines = lines[:-1]
        if len(lines) <= 1:
            return None
        last = lines[-1].strip()
    parts = next(csv.reader([last]))
    try:
        removed = (parts[0], int(parts[1]), int(parts[2]))
    except (IndexError, ValueError):
        return None
    with open(csv_path, "w", newline="") as f:
        f.writelines(lines[:-1])
    return removed


# ---------------------------- GUI ----------------------------


class CurationApp:
    def __init__(self, roots: List[Path], csv_path: Path):
        self.csv_path = csv_path
        ensure_csv_header(csv_path)

        self.roots = roots
        self.done = load_done_set(csv_path)
        self.queue = build_work_queue(roots, self.done)

        def _count(r: Path) -> int:
            p = _find_scores_path(r)
            if p is None:
                return 0
            try:
                return len(np.load(p, allow_pickle=False))
            except Exception:
                return 0

        self.total_all = sum(_count(r) for r in roots)

        if not self.queue:
            print("Nothing to curate. All ROIs already labeled.")
            sys.exit(0)

        # Cache loaded recordings lazily
        self._cache = {}

        # Track the most recently labeled entry so Undo can put it back at the
        # front of the queue
        self.last_labeled: Tuple[int, int, float] | None = None

        self.cursor = 0  # index into self.queue

        # --- Tk setup ---
        self.root = tk.Tk()
        self.root.title("ROI Curation")
        self.root.geometry("1700x800")

        # Figure: 2 rows, 4 cols. Top row = 4 image panels.
        # Bottom row = ΔF/F trace spanning all 4 columns.
        self.fig = plt.figure(figsize=(17, 7.5))
        gs = self.fig.add_gridspec(2, 4, height_ratios=[3, 1.5])
        self.img_axes = [self.fig.add_subplot(gs[0, c]) for c in range(4)]
        self.trace_ax = self.fig.add_subplot(gs[1, :])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar()
        status = tk.Label(
            self.root, textvariable=self.status_var, anchor="w",
            font=("Arial", 11),
        )
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=4)
        tk.Button(btn_frame, text="Not a cell (0)", width=15,
                  command=lambda: self.label(0)).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Cell (1)", width=15,
                  command=lambda: self.label(1)).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Undo (←)", width=10,
                  command=self.undo).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="Quit (Esc)", width=10,
                  command=self.quit).pack(side=tk.RIGHT, padx=4)

        # Keybinds
        self.root.bind("1", lambda e: self.label(1))
        self.root.bind("0", lambda e: self.label(0))
        self.root.bind("<Left>", lambda e: self.undo())
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("q", lambda e: self.quit())

        self.show_current()

    # ------------- data access -------------

    def get_recording(self, rec_idx: int):
        if rec_idx not in self._cache:
            self._cache[rec_idx] = load_recording(self.roots[rec_idx])
        return self._cache[rec_idx]

    # ------------- display -------------

    def show_current(self):
        if self.cursor >= len(self.queue):
            self.status_var.set("All done.")
            for ax in self.img_axes:
                ax.clear()
            self.trace_ax.clear()
            self.canvas.draw_idle()
            return

        rec_idx, roi_idx, score = self.queue[self.cursor]
        rec = self.get_recording(rec_idx)
        rec_id = rec["recording_id"]
        stat = rec["stat"]
        mean_img = rec["mean_img"]
        max_img = rec["max_img"]
        dff = rec["dff"]
        T = rec["T"]
        fps = rec["fps"]

        roi_stat = stat[roi_idx]
        xpix = roi_stat["xpix"]
        ypix = roi_stat["ypix"]

        for ax in self.img_axes:
            ax.clear()
        self.trace_ax.clear()

        x0, x1 = int(xpix.min()), int(xpix.max())
        y0, y1 = int(ypix.min()), int(ypix.max())
        pad = 8

        def _locator_box(ax):
            ax.add_patch(
                plt.Rectangle(
                    (x0 - pad, y0 - pad),
                    (x1 - x0) + 2 * pad,
                    (y1 - y0) + 2 * pad,
                    fill=False, edgecolor="yellow", linewidth=1.2,
                )
            )

        # -- Panel 0: mean image (locator box only) --
        ax0 = self.img_axes[0]
        vmin, vmax = np.percentile(mean_img, [1, 99])
        ax0.imshow(mean_img, cmap="gray", vmin=vmin, vmax=vmax)
        _locator_box(ax0)
        ax0.set_title(f"Mean image  ({rec_id})")
        ax0.set_xticks([]); ax0.set_yticks([])

        # -- Panel 1: max projection (locator box only, no ROI overlay) --
        ax1 = self.img_axes[1]
        vmin_m, vmax_m = np.percentile(max_img, [1, 99])
        ax1.imshow(max_img, cmap="gray", vmin=vmin_m, vmax=vmax_m)
        _locator_box(ax1)
        ax1.set_title("Max projection")
        ax1.set_xticks([]); ax1.set_yticks([])

        # -- Panel 2: max projection with ROI overlay --
        ax2 = self.img_axes[2]
        ax2.imshow(max_img, cmap="gray", vmin=vmin_m, vmax=vmax_m)
        ax2.scatter(xpix, ypix, s=2, c="red", alpha=0.6, edgecolors="none")
        _locator_box(ax2)
        ax2.set_title("Max projection + ROI")
        ax2.set_xticks([]); ax2.set_yticks([])

        # -- Panel 3: ROI footprint --
        ax3 = self.img_axes[3]
        ax3.scatter(xpix, ypix, s=4, c="black")
        ax3.invert_yaxis()
        ax3.set_aspect("equal")
        ax3.set_title(f"ROI footprint  (#{roi_idx})")
        ax3.set_xticks([]); ax3.set_yticks([])

        # -- Trace panel (spans bottom row) --
        trace = np.asarray(dff[:, roi_idx])
        time = np.arange(T) / fps
        self.trace_ax.plot(time, trace, lw=0.8)
        self.trace_ax.set_title(f"ΔF/F   p(cell)={score:.3f}")
        self.trace_ax.set_xlabel("Time (s)")
        self.trace_ax.margins(x=0)

        self.fig.tight_layout()
        self.canvas.draw_idle()

        labeled = len(self.done)
        remaining = len(self.queue) - self.cursor
        self.status_var.set(
            f"Labeled: {labeled} / {self.total_all}    "
            f"Remaining in queue: {remaining}    "
            f"Current: {rec_id}  ROI {roi_idx}  p(cell)={score:.3f}    "
            f"[1]=cell  [0]=not a cell  [←]=undo  [Esc]=quit"
        )

    # ------------- actions -------------

    def label(self, value: int):
        if self.cursor >= len(self.queue):
            return
        rec_idx, roi_idx, score = self.queue[self.cursor]
        rec_id = self.roots[rec_idx].name
        append_label(self.csv_path, rec_id, roi_idx, value)
        self.done.add((rec_id, roi_idx))
        self.last_labeled = (rec_idx, roi_idx, score)
        self.cursor += 1
        self.show_current()

    def undo(self):
        removed = remove_last_row(self.csv_path)
        if removed is None:
            self.status_var.set("Nothing to undo.")
            return
        rec_id, roi_idx, _label = removed
        self.done.discard((rec_id, roi_idx))

        # Step cursor back and verify the current entry matches what we removed.
        if self.cursor > 0:
            self.cursor -= 1
        # If the current queue entry doesn't match (e.g. user restarted since),
        # fall back to requeuing the removed item at the current position.
        if self.cursor < len(self.queue):
            ri, roi, _ = self.queue[self.cursor]
            if self.roots[ri].name != rec_id or roi != roi_idx:
                # Find rec_idx for removed recording
                rec_idx = next(
                    (i for i, r in enumerate(self.roots) if r.name == rec_id),
                    None,
                )
                if rec_idx is not None:
                    self.queue.insert(self.cursor, (rec_idx, roi_idx, 0.0))
        self.show_current()

    def quit(self):
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    roots = [Path(r) for r in ROOTS]
    for r in roots:
        if not r.exists():
            print(f"WARNING: root does not exist: {r}")
    app = CurationApp(roots, OUTPUT_CSV)
    app.run()


if __name__ == "__main__":
    main()