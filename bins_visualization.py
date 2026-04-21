from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import utils


@dataclass
class AlignedCoactivationFigureConfig:
    root: Path
    prefix: str = "r0p7_filtered_"
    fps: Optional[float] = None

    # Event detection
    z_enter: float = 3.5
    z_exit: float = 1.5
    min_sep_s: float = 0.3

    # Coactivation binning
    bin_sec: float = 0.5
    frac_required: float = 0.8

    # ROI selection
    use_cell_mask: bool = True
    cell_mask_path: Optional[Path] = None

    # Figure behavior
    max_bins_to_plot: int = 12
    figsize_per_row: float = 2.2
    save_path: Optional[Path] = None


def load_dt_memmap(root: Path, prefix: str):
    """
    Uses the same memmap loader style as your current pipeline.
    """
    _, _, dt, T, N = utils.s2p_open_memmaps(root, prefix=prefix)
    return np.asarray(dt), T, N


def load_roi_mask(cfg: AlignedCoactivationFigureConfig, n_rois: int) -> np.ndarray:
    if not cfg.use_cell_mask:
        return np.ones(n_rois, dtype=bool)

    candidates = []
    if cfg.cell_mask_path is not None:
        candidates.append(cfg.cell_mask_path)
    candidates.append(cfg.root / "r0p7_cell_mask_bool.npy")

    for path in candidates:
        if path.exists():
            mask = np.load(path, allow_pickle=False).astype(bool).ravel()
            if mask.size != n_rois:
                raise ValueError(f"Mask length {mask.size} does not match ROI count {n_rois}")
            return mask

    print("[INFO] No cell mask found, using all ROIs.")
    return np.ones(n_rois, dtype=bool)


def detect_onsets_by_roi(
    dt: np.ndarray,
    fps: float,
    z_enter: float,
    z_exit: float,
    min_sep_s: float,
) -> list[np.ndarray]:
    """
    Same ROI wise event detection logic you already use.
    """
    onsets_by_roi = []
    for i in range(dt.shape[1]):
        z, _, _ = utils.mad_z(dt[:, i])
        on = utils.hysteresis_onsets(
            z,
            z_hi=z_enter,
            z_lo=z_exit,
            fps=fps,
            min_sep_s=min_sep_s,
        )
        onsets_by_roi.append(np.asarray(on, dtype=np.int64) / float(fps))
    return onsets_by_roi


def build_bin_edges(T: int, fps: float, bin_sec: float) -> np.ndarray:
    total_sec = T / float(fps)
    n_bins = int(np.ceil(total_sec / float(bin_sec)))
    return np.linspace(0.0, n_bins * float(bin_sec), n_bins + 1)


def activation_matrix(onsets_by_roi: list[np.ndarray], edges: np.ndarray):
    """
    Same structure as your current coactivation code.
    A[i, b] is True if ROI i has an onset in bin b.
    first_time[i, b] stores the first onset time in that bin.
    """
    N = len(onsets_by_roi)
    B = len(edges) - 1

    A = np.zeros((N, B), dtype=bool)
    first_time = np.full((N, B), np.nan, dtype=float)

    for i, ts in enumerate(onsets_by_roi):
        if ts.size == 0:
            continue

        bins = np.searchsorted(edges, ts, side="right") - 1
        valid = (bins >= 0) & (bins < B)
        if not np.any(valid):
            continue

        ubins = np.unique(bins[valid])
        A[i, ubins] = True

        for b in ubins:
            in_bin = valid & (bins == b)
            first_time[i, b] = np.min(ts[in_bin])

    return A, first_time


def select_active_bins(A: np.ndarray, frac_required: float) -> tuple[np.ndarray, np.ndarray]:
    active_counts = A.sum(axis=0)
    n_rois = A.shape[0]
    min_count = int(np.ceil(float(frac_required) * n_rois))
    keep_bins = np.where(active_counts >= min_count)[0]
    return keep_bins, active_counts


def extract_aligned_bin_events(
    keep_bins: np.ndarray,
    A: np.ndarray,
    first_time: np.ndarray,
    edges: np.ndarray,
):
    """
    For each active bin, shift event times so earliest onset is zero.
    """
    out = []

    for b in keep_bins:
        sel = A[:, b] & ~np.isnan(first_time[:, b])
        if not np.any(sel):
            continue

        roi_idx = np.where(sel)[0]
        abs_times = first_time[roi_idx, b]
        earliest = float(np.min(abs_times))
        rel_times = abs_times - earliest

        order = np.argsort(rel_times, kind="mergesort")

        out.append(
            {
                "bin_index": int(b),
                "bin_start_s": float(edges[b]),
                "bin_end_s": float(edges[b + 1]),
                "n_active": int(sel.sum()),
                "roi_idx": roi_idx[order],
                "abs_times": abs_times[order],
                "rel_times": rel_times[order],
                "earliest_time": earliest,
            }
        )

    return out


def make_aligned_coactivation_figure(cfg: AlignedCoactivationFigureConfig):
    dt, T, N = load_dt_memmap(cfg.root, cfg.prefix)

    fps = cfg.fps
    if fps is None:
        recording_root = cfg.root.parent.parent.parent
        fps = float(utils.get_fps_from_notes(str(recording_root)))

    mask = load_roi_mask(cfg, N)
    keep_roi_idx = np.where(mask)[0]

    dt_use = dt[:, keep_roi_idx]

    onsets_by_roi = detect_onsets_by_roi(
        dt=dt_use,
        fps=fps,
        z_enter=cfg.z_enter,
        z_exit=cfg.z_exit,
        min_sep_s=cfg.min_sep_s,
    )

    edges = build_bin_edges(T, fps, cfg.bin_sec)
    A, first_time = activation_matrix(onsets_by_roi, edges)
    keep_bins, active_counts = select_active_bins(A, cfg.frac_required)

    aligned_bins = extract_aligned_bin_events(keep_bins, A, first_time, edges)

    if len(aligned_bins) == 0:
        raise RuntimeError("No active bins found with the current parameters.")

    aligned_bins = aligned_bins[: cfg.max_bins_to_plot]

    n_rows = len(aligned_bins) + 1
    fig = plt.figure(figsize=(14, max(4, cfg.figsize_per_row * n_rows)))
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[2.2, 1.0], hspace=0.8, wspace=0.3)

    # Panel A, overview of coactivation counts across bins
    ax0 = fig.add_subplot(gs[0, :])
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    threshold_count = int(np.ceil(cfg.frac_required * A.shape[0]))

    ax0.plot(bin_centers, active_counts, linewidth=1.5)
    ax0.axhline(threshold_count, linestyle="--", linewidth=1.0, label=f"threshold = {threshold_count} ROIs")
    ax0.scatter(
        bin_centers[keep_bins],
        active_counts[keep_bins],
        s=18,
        zorder=3,
        label="active bins",
    )
    ax0.set_title("Coactivation bins detected from percentage of active cells")
    ax0.set_xlabel("Time in recording, s")
    ax0.set_ylabel("Active ROIs per bin")
    ax0.legend(frameon=False)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # Per bin aligned rasters plus lag histograms
    for r, bin_info in enumerate(aligned_bins, start=1):
        ax_raster = fig.add_subplot(gs[r, 0])
        ax_hist = fig.add_subplot(gs[r, 1])

        y = np.arange(len(bin_info["roi_idx"]))
        x = bin_info["rel_times"]

        ax_raster.scatter(x, y, s=18)
        for xi, yi in zip(x, y):
            ax_raster.plot([0, xi], [yi, yi], linewidth=0.8, alpha=0.5)

        ax_raster.axvline(0, linestyle="--", linewidth=1.0)
        ax_raster.set_title(
            f"Bin {bin_info['bin_index']}   "
            f"{bin_info['bin_start_s']:.2f} to {bin_info['bin_end_s']:.2f} s   "
            f"n={bin_info['n_active']}"
        )
        ax_raster.set_xlabel("Aligned onset time, s from earliest ROI")
        ax_raster.set_ylabel("Active ROI rank")
        ax_raster.spines["top"].set_visible(False)
        ax_raster.spines["right"].set_visible(False)

        ax_hist.hist(bin_info["rel_times"], bins=20)
        ax_hist.axvline(0, linestyle="--", linewidth=1.0)
        ax_hist.set_xlabel("Relative lag, s")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Lag distribution")
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)

    fig.suptitle(
        "Bin wise coactivation events aligned to earliest detected activation",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout()

    if cfg.save_path is not None:
        cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {cfg.save_path}")

    return fig, {
        "edges": edges,
        "A": A,
        "first_time": first_time,
        "keep_bins": keep_bins,
        "active_counts": active_counts,
        "aligned_bins": aligned_bins,
        "kept_roi_original_indices": keep_roi_idx,
        "fps": fps,
    }


if __name__ == "__main__":
    cfg = AlignedCoactivationFigureConfig(
        root=Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0"),
        prefix="r0p7_filtered_",
        fps=None,
        z_enter=3.5,
        z_exit=1.5,
        min_sep_s=0.3,
        bin_sec=0.5,
        frac_required=0.25,
        use_cell_mask=False,
        max_bins_to_plot=10,
        save_path=None,
    )

    fig, results = make_aligned_coactivation_figure(cfg)
    plt.show()