from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import utils
import numpy as np
from scipy.signal import find_peaks, peak_widths

def detect_density_peaks(
    centers,
    smooth,
    counts=None,
    min_prominence=0.02,
    min_width_bins=2,
    min_distance_bins=3,
):
    peaks, props = find_peaks(
        smooth,
        prominence=min_prominence,
        width=min_width_bins,
        distance=min_distance_bins,
    )

    widths, width_heights, left_ips, right_ips = peak_widths(
        smooth,
        peaks,
        rel_height=0.5,
    )

    results = []
    for i, p in enumerate(peaks):
        left_idx = max(0, int(np.floor(left_ips[i])))
        right_idx = min(len(centers) - 1, int(np.ceil(right_ips[i])))

        area_smooth = np.trapz(smooth[left_idx:right_idx + 1], centers[left_idx:right_idx + 1])

        area_counts = None
        if counts is not None:
            area_counts = np.trapz(counts[left_idx:right_idx + 1], centers[left_idx:right_idx + 1])

        results.append({
            "peak_idx": int(p),
            "peak_time_s": float(centers[p]),
            "peak_height": float(smooth[p]),
            "prominence": float(props["prominences"][i]),
            "width_bins": float(widths[i]),
            "left_boundary_s": float(centers[left_idx]),
            "right_boundary_s": float(centers[right_idx]),
            "area_smooth": float(area_smooth),
            "area_counts": None if area_counts is None else float(area_counts),
        })

    return results, peaks, props

@dataclass
class OnsetDensityConfig:
    root: Path
    prefix: str = "r0p7_"
    fps: Optional[float] = None

    # Event detection
    z_enter: float = 3.5
    z_exit: float = 1.5
    min_sep_s: float = 0.1

    # Optional crop
    t_start_s: float = 0.0
    t_end_s: Optional[float] = None

    # Density settings
    bin_sec: float = 0.5
    smooth_sigma_bins: float = 2.0
    normalize_by_num_rois: bool = True

    # Output
    save_path: Optional[Path] = None
    show: bool = True


def extract_onsets_by_roi(
    dt: np.ndarray,
    fps: float,
    z_enter: float,
    z_exit: float,
    min_sep_s: float,
    t_start_frame: int = 0,
) -> list[np.ndarray]:
    """
    Return one onset-time array in seconds per ROI.
    dt is expected to be shape (T, N).
    """
    if dt.ndim != 2:
        raise ValueError(f"dt must be 2D with shape (T, N). Got {dt.shape}")

    onsets_sec_by_roi: list[np.ndarray] = []

    for roi in range(dt.shape[1]):
        x = np.asarray(dt[:, roi], dtype=np.float32)
        z, _, _ = utils.mad_z(x)
        onsets = utils.hysteresis_onsets(
            z,
            z_enter,
            z_exit,
            fps,
            min_sep_s=min_sep_s,
        )
        onsets_sec = onsets.astype(np.float64) / float(fps)
        onsets_sec += float(t_start_frame) / float(fps)
        onsets_sec_by_roi.append(onsets_sec)

    return onsets_sec_by_roi


def flatten_onsets(onsets_by_roi: list[np.ndarray]) -> np.ndarray:
    """
    Flatten list of onset arrays into one 1D vector of onset times in seconds.
    """
    nonempty = [x for x in onsets_by_roi if x.size > 0]
    if not nonempty:
        return np.array([], dtype=np.float64)
    return np.concatenate(nonempty).astype(np.float64, copy=False)


def build_density(
    flat_onsets_sec: np.ndarray,
    duration_s: float,
    bin_sec: float,
    smooth_sigma_bins: float,
    n_rois: int,
    normalize_by_num_rois: bool,
):
    """
    Build histogram density and smoothed density on a common time axis.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive.")
    if bin_sec <= 0:
        raise ValueError("bin_sec must be positive.")

    edges = np.arange(0.0, duration_s + bin_sec, bin_sec, dtype=np.float64)
    if edges[-1] < duration_s:
        edges = np.append(edges, duration_s)

    counts, edges = np.histogram(flat_onsets_sec, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if normalize_by_num_rois and n_rois > 0:
        counts_for_plot = counts.astype(np.float64) / float(n_rois)
        ylabel = f"Onsets per {bin_sec:g}s bin per ROI"
    else:
        counts_for_plot = counts.astype(np.float64)
        ylabel = f"Onsets per {bin_sec:g}s bin"

    smooth = gaussian_filter1d(counts_for_plot, sigma=smooth_sigma_bins, mode="nearest")

    return centers, counts_for_plot, smooth, ylabel


def plot_onset_density(
    centers: np.ndarray,
    counts: np.ndarray,
    smooth: np.ndarray,
    ylabel: str,
    cfg: OnsetDensityConfig,
    n_rois: int,
    n_events: int,
    duration_s: float,
    flat_onsets_sec: np.ndarray,
    peaks,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4.5))

    width = np.median(np.diff(centers)) if centers.size > 1 else cfg.bin_sec
    #ax.eventplot(
    #    [flat_onsets_sec],
    #    lineoffsets=np.max(counts) * 1.05 if counts.size else 1.0,
    #    linelengths=np.max(counts) * 0.08 if counts.size else 0.1,
    #)
    ax.bar(centers, counts, width=width, alpha=0.35, align="center", label="Binned counts")
    ax.plot(centers, smooth, linewidth=2.0, label="Smoothed density")
    ax.plot(centers[peaks], smooth[peaks], "o")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Flattened onset density across all ROIs\n"
        f"N ROIs = {n_rois}, total onsets = {n_events}, duration = {duration_s:.1f}s"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def run_onset_density(cfg: OnsetDensityConfig):
    root = Path(cfg.root)

    dff, low, dt, T, N = utils.s2p_open_memmaps(root, prefix=cfg.prefix)

    fps = float(cfg.fps) if cfg.fps is not None else 15.0

    start_frame = max(0, int(round(cfg.t_start_s * fps)))
    end_frame = T if cfg.t_end_s is None else min(T, int(round(cfg.t_end_s * fps)))

    if end_frame <= start_frame:
        raise ValueError("t_end_s must be greater than t_start_s.")

    dt_crop = np.asarray(dt[start_frame:end_frame, :], dtype=np.float32)
    duration_s = float(end_frame - start_frame) / fps

    onsets_by_roi = extract_onsets_by_roi(
        dt=dt_crop,
        fps=fps,
        z_enter=cfg.z_enter,
        z_exit=cfg.z_exit,
        min_sep_s=cfg.min_sep_s,
        t_start_frame=start_frame,
    )

    flat_onsets_sec = flatten_onsets(onsets_by_roi)

    centers, counts, smooth, ylabel = build_density(
        flat_onsets_sec=flat_onsets_sec,
        duration_s=duration_s,
        bin_sec=cfg.bin_sec,
        smooth_sigma_bins=cfg.smooth_sigma_bins,
        n_rois=N,
        normalize_by_num_rois=cfg.normalize_by_num_rois,
    )
    peak_table, peaks, props = detect_density_peaks(
        centers=centers,
        smooth=smooth,
        counts=counts,
        min_prominence=0.007,
        min_width_bins=2,
        min_distance_bins=3,
    )

    fig = plot_onset_density(
        centers=centers,
        counts=counts,
        smooth=smooth,
        ylabel=ylabel,
        cfg=cfg,
        n_rois=N,
        n_events=int(flat_onsets_sec.size),
        duration_s=duration_s,
        flat_onsets_sec=flat_onsets_sec,
        peaks=peaks
    )

    if cfg.save_path is not None:
        cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {cfg.save_path}")

        out_csv = cfg.save_path.with_suffix(".csv")
        np.savetxt(
            out_csv,
            np.column_stack([centers, counts, smooth]),
            delimiter=",",
            header="time_s,binned_density,smoothed_density",
            comments="",
        )
        print(f"Saved density values to: {out_csv}")

    if cfg.show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "onsets_by_roi": onsets_by_roi,
        "flat_onsets_sec": flat_onsets_sec,
        "time_centers_s": centers,
        "binned_density": counts,
        "smoothed_density": smooth,
    }


if __name__ == "__main__":
    cfg = OnsetDensityConfig(
        root=Path(r"F:\data\2p_shifted\Hip\2024-06-04_00001\suite2p\plane0"),
        prefix="r0p7_filtered_",
        fps=15.0,
        z_enter=3.5,
        z_exit=1.5,
        min_sep_s=0.1,
        t_start_s=0.0,
        t_end_s=None,
        bin_sec=0.05,
        smooth_sigma_bins=2.0,
        normalize_by_num_rois=True,
        save_path=None,
        show=True,
    )
    run_onset_density(cfg)