from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import utils


@dataclass
class OnsetDensityConfig:
    root: Path
    prefix: str = "r0p7_"
    fps: Optional[float] = None

    # Event detection on ROI traces
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

    # Peak detection on smoothed density
    peak_min_prominence: float = 0.007
    peak_min_width_bins: float = 2.0
    peak_min_distance_bins: int = 3

    # Local baseline for drifting floor
    baseline_window_s: float = 3.0
    baseline_percentile: float = 20.0

    # Boundary finding on detrended trace
    boundary_enter_fraction: float = 0.15
    boundary_exit_fraction: float = 0.05
    min_boundary_residual: float = 0.0

    # Plot
    plot_peak_apices: bool = False
    plot_peak_boundaries: bool = True
    plot_boundary_lines: bool = True
    plot_baseline: bool = True
    plot_detrended: bool = False

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


def rolling_percentile_baseline(
    y: np.ndarray,
    window_bins: int,
    percentile: float,
) -> np.ndarray:
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if window_bins < 1:
        raise ValueError("window_bins must be >= 1")

    half = window_bins // 2
    baseline = np.empty_like(y, dtype=np.float64)

    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        baseline[i] = np.percentile(y[lo:hi], percentile)

    return baseline


def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def find_peak_boundaries_drift_aware(
    detrended: np.ndarray,
    peak_idx: int,
    peak_height: float,
    enter_fraction: float,
    exit_fraction: float,
    min_boundary_residual: float = 0.0,
) -> tuple[int, int, float, float]:
    enter_thr = max(min_boundary_residual, enter_fraction * peak_height)
    exit_thr = max(min_boundary_residual, exit_fraction * peak_height)

    left = peak_idx
    while left > 0 and detrended[left] > enter_thr:
        left -= 1

    right = peak_idx
    while right < len(detrended) - 1 and detrended[right] > exit_thr:
        right += 1

    return left, right, enter_thr, exit_thr


def detect_density_peaks(
    centers: np.ndarray,
    smooth: np.ndarray,
    counts: Optional[np.ndarray] = None,
    min_prominence: float = 0.02,
    min_width_bins: float = 2.0,
    min_distance_bins: int = 3,
    baseline_window_s: float = 3.0,
    bin_sec: float = 0.05,
    baseline_percentile: float = 20.0,
    boundary_enter_fraction: float = 0.15,
    boundary_exit_fraction: float = 0.05,
    min_boundary_residual: float = 0.0,
):
    if smooth.ndim != 1 or centers.ndim != 1:
        raise ValueError("smooth and centers must both be 1D")
    if smooth.size != centers.size:
        raise ValueError("smooth and centers must have the same length")

    if smooth.size == 0:
        return [], np.array([], dtype=int), {}, np.array([]), np.array([])

    peaks, props = find_peaks(
        smooth,
        prominence=min_prominence,
        width=min_width_bins,
        distance=min_distance_bins,
    )

    window_bins = max(3, int(round(baseline_window_s / bin_sec)))
    if window_bins % 2 == 0:
        window_bins += 1

    baseline = rolling_percentile_baseline(
        smooth,
        window_bins=window_bins,
        percentile=baseline_percentile,
    )
    detrended = smooth - baseline
    detrended[detrended < 0] = 0.0

    if min_boundary_residual <= 0:
        residual_mad = mad(detrended)
        min_boundary_residual = 0.25 * residual_mad

    results = []

    for i, p in enumerate(peaks):
        p = int(p)
        peak_height_raw = float(smooth[p])
        peak_height_detrended = float(detrended[p])
        prominence = float(props["prominences"][i])

        if peak_height_detrended <= 0:
            continue

        left_idx, right_idx, enter_thr, exit_thr = find_peak_boundaries_drift_aware(
            detrended=detrended,
            peak_idx=p,
            peak_height=peak_height_detrended,
            enter_fraction=boundary_enter_fraction,
            exit_fraction=boundary_exit_fraction,
            min_boundary_residual=min_boundary_residual,
        )

        left_idx = max(0, min(left_idx, p))
        right_idx = min(len(smooth) - 1, max(right_idx, p))

        area_smooth = np.trapz(
            smooth[left_idx:right_idx + 1],
            centers[left_idx:right_idx + 1],
        )
        area_detrended = np.trapz(
            detrended[left_idx:right_idx + 1],
            centers[left_idx:right_idx + 1],
        )

        area_counts = None
        if counts is not None:
            area_counts = np.trapz(
                counts[left_idx:right_idx + 1],
                centers[left_idx:right_idx + 1],
            )

        results.append({
            "peak_idx": p,
            "peak_time_s": float(centers[p]),
            "peak_height_raw": peak_height_raw,
            "peak_height_detrended": peak_height_detrended,
            "prominence": prominence,
            "baseline_at_peak": float(baseline[p]),
            "left_boundary_idx": int(left_idx),
            "right_boundary_idx": int(right_idx),
            "left_boundary_s": float(centers[left_idx]),
            "right_boundary_s": float(centers[right_idx]),
            "enter_threshold": float(enter_thr),
            "exit_threshold": float(exit_thr),
            "area_smooth": float(area_smooth),
            "area_detrended": float(area_detrended),
            "area_counts": np.nan if area_counts is None else float(area_counts),
        })

    return results, peaks, props, baseline, detrended


def plot_onset_density(
    centers: np.ndarray,
    counts: np.ndarray,
    smooth: np.ndarray,
    baseline: np.ndarray,
    detrended: np.ndarray,
    ylabel: str,
    cfg: OnsetDensityConfig,
    n_rois: int,
    n_events: int,
    duration_s: float,
    peak_table: list[dict],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4.5))

    width = np.median(np.diff(centers)) if centers.size > 1 else cfg.bin_sec

    ax.bar(
        centers,
        counts,
        width=width,
        alpha=0.35,
        align="center",
        label="Binned counts",
    )
    ax.plot(centers, smooth, linewidth=2.0, label="Smoothed density")

    if cfg.plot_baseline:
        ax.plot(centers, baseline, linewidth=1.5, alpha=0.9, label="Local baseline")

    if cfg.plot_peak_boundaries and peak_table:
        left_times = np.array([row["left_boundary_s"] for row in peak_table], dtype=float)
        right_times = np.array([row["right_boundary_s"] for row in peak_table], dtype=float)
        left_vals = np.array([smooth[row["left_boundary_idx"]] for row in peak_table], dtype=float)
        right_vals = np.array([smooth[row["right_boundary_idx"]] for row in peak_table], dtype=float)

        ax.plot(left_times, left_vals, linestyle="None", marker="o", markersize=5, label="Peak start")
        ax.plot(right_times, right_vals, linestyle="None", marker="o", markersize=5, label="Peak end")

    if cfg.plot_boundary_lines:
        for row in peak_table:
            ax.hlines(
                y=row["baseline_at_peak"] + row["enter_threshold"],
                xmin=row["left_boundary_s"],
                xmax=row["peak_time_s"],
                linewidth=2,
                alpha=0.8,
            )
            ax.hlines(
                y=row["baseline_at_peak"] + row["exit_threshold"],
                xmin=row["peak_time_s"],
                xmax=row["right_boundary_s"],
                linewidth=2,
                alpha=0.8,
            )

    if cfg.plot_peak_apices and peak_table:
        peak_times = np.array([row["peak_time_s"] for row in peak_table], dtype=float)
        peak_vals = np.array([row["peak_height_raw"] for row in peak_table], dtype=float)
        ax.plot(peak_times, peak_vals, linestyle="None", marker="o", markersize=4, label="Peak apex")

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

    if cfg.plot_detrended:
        fig2, ax2 = plt.subplots(figsize=(12, 3.5))
        ax2.plot(centers, detrended, linewidth=2.0)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Detrended density")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        fig2.tight_layout()

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

    peak_table, peaks, props, baseline, detrended = detect_density_peaks(
        centers=centers,
        smooth=smooth,
        counts=counts,
        min_prominence=cfg.peak_min_prominence,
        min_width_bins=cfg.peak_min_width_bins,
        min_distance_bins=cfg.peak_min_distance_bins,
        baseline_window_s=cfg.baseline_window_s,
        bin_sec=cfg.bin_sec,
        baseline_percentile=cfg.baseline_percentile,
        boundary_enter_fraction=cfg.boundary_enter_fraction,
        boundary_exit_fraction=cfg.boundary_exit_fraction,
        min_boundary_residual=cfg.min_boundary_residual,
    )

    fig = plot_onset_density(
        centers=centers,
        counts=counts,
        smooth=smooth,
        baseline=baseline,
        detrended=detrended,
        ylabel=ylabel,
        cfg=cfg,
        n_rois=N,
        n_events=int(flat_onsets_sec.size),
        duration_s=duration_s,
        peak_table=peak_table,
    )

    if cfg.save_path is not None:
        cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {cfg.save_path}")

        density_csv = cfg.save_path.with_suffix(".csv")
        np.savetxt(
            density_csv,
            np.column_stack([centers, counts, smooth, baseline, detrended]),
            delimiter=",",
            header="time_s,binned_density,smoothed_density,baseline,detrended_density",
            comments="",
        )
        print(f"Saved density values to: {density_csv}")

        peaks_csv = cfg.save_path.with_name(cfg.save_path.stem + "_peaks.csv")
        if peak_table:
            columns = [
                "peak_idx",
                "peak_time_s",
                "peak_height_raw",
                "peak_height_detrended",
                "prominence",
                "baseline_at_peak",
                "left_boundary_idx",
                "right_boundary_idx",
                "left_boundary_s",
                "right_boundary_s",
                "enter_threshold",
                "exit_threshold",
                "area_smooth",
                "area_detrended",
                "area_counts",
            ]
            peak_array = np.array(
                [[row[col] for col in columns] for row in peak_table],
                dtype=float,
            )
            np.savetxt(
                peaks_csv,
                peak_array,
                delimiter=",",
                header=",".join(columns),
                comments="",
            )
            print(f"Saved peak table to: {peaks_csv}")

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
        "baseline": baseline,
        "detrended_density": detrended,
        "peak_table": peak_table,
        "peaks": peaks,
        "props": props,
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
        peak_min_prominence=0.007,
        peak_min_width_bins=2.0,
        peak_min_distance_bins=3,
        baseline_window_s=3.0,
        baseline_percentile=20.0,
        boundary_enter_fraction=0.15,
        boundary_exit_fraction=0.05,
        min_boundary_residual=0.0,
        plot_peak_apices=False,
        plot_peak_boundaries=True,
        plot_boundary_lines=True,
        plot_baseline=True,
        plot_detrended=False,
        save_path=None,
        show=True,
    )
    run_onset_density(cfg)