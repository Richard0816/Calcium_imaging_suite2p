"""
Event-boundary module that operates on pre-detected peaks.

Peak detection is performed by `event_detection.detect_density_peaks`
(scipy.signal.find_peaks with prominence / width / distance gates). This
module takes that output and produces per-event (start, peak, end) times
using a baseline-walk + Gaussian-fit approach with the "whichever comes
first" rule to choose the tighter boundary on each side.

Pipeline
--------
1. Compute a rolling-percentile baseline on the smoothed density and estimate
   noise from residuals around the baseline.
2. For each peak (passed in from event_detection), walk outward until the
   density drops below baseline + k * noise. That's the outer safety bracket.
3. Fit a Gaussian (moment matching on density - baseline) inside that bracket
   and read off symmetric quantile-based boundaries.
4. For each side, keep whichever boundary is tighter (gaussian or baseline-walk).
5. Merge overlapping windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


# ---------- config / output ----------

@dataclass
class EventBoundaryConfig:
    # Baseline / noise estimation
    baseline_mode: str = "rolling"         # "rolling" or "global"
    baseline_percentile: float = 5.0
    baseline_window_s: float = 30.0        # only used for rolling
    noise_quiet_percentile: float = 40.0
    noise_mad_factor: float = 1.4826

    # Boundary walking
    end_threshold_k: float = 2.0           # end = baseline + k * noise
    max_event_duration_s: float = 10.0     # hard cap on one-sided walk
    merge_gap_s: float = 0.0               # merge events closer than this

    # Gaussian-fit refinement
    use_gaussian_boundary: bool = True
    gaussian_quantile: float = 0.99        # one-sided quantile; 0.99 -> ~2.326 sigma
    gaussian_fit_pad_s: float = 0.5
    gaussian_min_sigma_s: float = 0.05

    # Output
    save_csv: Optional[Path] = None
    save_fig: Optional[Path] = None
    show: bool = True


@dataclass
class EventTable:
    start_s: np.ndarray
    peak_s: np.ndarray
    end_s: np.ndarray
    peak_height: np.ndarray
    prominence: np.ndarray
    duration_s: np.ndarray
    baseline_trace: np.ndarray
    end_threshold_trace: np.ndarray
    baseline_noise: float
    mu_s: np.ndarray                    # fitted mean (NaN if gaussian disabled)
    sigma_s: np.ndarray                 # fitted sigma
    boundary_source_left: np.ndarray    # "gaussian" or "baseline"
    boundary_source_right: np.ndarray   # "gaussian" or "baseline"

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "start_s", "peak_s", "end_s",
            "peak_height", "prominence", "duration_s",
            "baseline_trace", "end_threshold_trace", "baseline_noise",
            "mu_s", "sigma_s",
            "boundary_source_left", "boundary_source_right",
        )}


# ---------- baseline / noise ----------

def estimate_global_baseline(density: np.ndarray, baseline_percentile: float) -> np.ndarray:
    if density.size == 0:
        return np.zeros_like(density)
    b = float(np.percentile(density, baseline_percentile))
    return np.full_like(density, b, dtype=np.float64)


def estimate_rolling_baseline(
    density: np.ndarray,
    fps_density: float,
    baseline_window_s: float,
    baseline_percentile: float,
) -> np.ndarray:
    from scipy.ndimage import percentile_filter
    if density.size == 0:
        return np.zeros_like(density)
    win = max(3, int(round(baseline_window_s * fps_density)) | 1)
    win = min(win, density.size if density.size % 2 == 1 else density.size - 1)
    if win < 3:
        return estimate_global_baseline(density, baseline_percentile)
    return percentile_filter(
        density.astype(np.float64),
        size=win,
        percentile=baseline_percentile,
        mode="reflect",
    )


def estimate_noise_from_quiet(
    density: np.ndarray,
    baseline_trace: np.ndarray,
    quiet_percentile: float = 40.0,
    noise_mad_factor: float = 1.4826,
) -> float:
    if density.size == 0:
        return 0.0
    resid = density - baseline_trace
    cutoff = float(np.percentile(resid, quiet_percentile))
    quiet_mask = resid <= cutoff
    if quiet_mask.sum() < 32:
        quiet_mask = np.ones_like(resid, dtype=bool)
    sample = resid[quiet_mask]
    med = np.median(sample)
    mad = np.median(np.abs(sample - med)) + 1e-12
    return float(noise_mad_factor * mad)


# ---------- boundary walking ----------

def walk_boundary(
    density: np.ndarray,
    peak_idx: int,
    end_threshold_trace: np.ndarray,
    direction: int,
    max_steps: int,
) -> int:
    n = density.size
    i = peak_idx
    steps = 0
    while 0 <= i + direction < n and steps < max_steps:
        nxt = i + direction
        if density[nxt] <= end_threshold_trace[nxt]:
            return i
        i = nxt
        steps += 1
    return i


# ---------- Gaussian fit ----------

def _gaussian_z(q: float) -> float:
    from scipy.special import erfinv
    if not (0.5 < q < 1.0):
        raise ValueError("gaussian_quantile must be in (0.5, 1.0)")
    return float(np.sqrt(2.0) * erfinv(2.0 * q - 1.0))


def fit_gaussian_to_peak(
    time_s: np.ndarray,
    density: np.ndarray,
    baseline_trace: np.ndarray,
    peak_idx: int,
    left_idx: int,
    right_idx: int,
    pad_samples: int,
    min_sigma_s: float,
) -> tuple[float, float]:
    """
    Moment match a Gaussian to (density - baseline) inside a padded window.
    Baseline subtraction is essential or sigma blows up.
    """
    n = density.size
    L = max(0, left_idx - pad_samples)
    R = min(n - 1, right_idx + pad_samples)
    if R <= L:
        return float(time_s[peak_idx]), float(min_sigma_s)

    t_win = time_s[L:R + 1].astype(np.float64)
    d_win = density[L:R + 1].astype(np.float64)
    b_win = baseline_trace[L:R + 1].astype(np.float64)
    w = np.clip(d_win - b_win, 0.0, None)

    total = w.sum()
    if total <= 0:
        return float(time_s[peak_idx]), float(min_sigma_s)

    mu = float((t_win * w).sum() / total)
    var = float(((t_win - mu) ** 2 * w).sum() / total)
    sigma = max(float(np.sqrt(max(var, 0.0))), float(min_sigma_s))
    return mu, sigma


# ---------- main entry point ----------

def detect_boundaries_from_peaks(
    time_s: np.ndarray,
    smooth_density: np.ndarray,
    peak_indices: Sequence[int],
    cfg: EventBoundaryConfig,
) -> EventTable:
    """
    Given a smoothed density and pre-detected peak indices (from e.g.
    event_detection.detect_density_peaks), produce per-event boundaries.
    """
    time_s = np.asarray(time_s, dtype=np.float64)
    smooth = np.asarray(smooth_density, dtype=np.float64)
    peaks = np.asarray(peak_indices, dtype=np.int64)

    if time_s.shape != smooth.shape:
        raise ValueError(
            f"time_s and smooth_density must have the same shape; "
            f"got {time_s.shape} vs {smooth.shape}"
        )
    if time_s.size < 3:
        raise ValueError("Need at least 3 samples to detect boundaries.")

    dt = float(np.median(np.diff(time_s)))
    if dt <= 0:
        raise ValueError("time_s must be strictly increasing.")
    fps_density = 1.0 / dt

    # 1. baseline + noise
    if cfg.baseline_mode == "global":
        baseline_trace = estimate_global_baseline(smooth, cfg.baseline_percentile)
    elif cfg.baseline_mode == "rolling":
        baseline_trace = estimate_rolling_baseline(
            smooth, fps_density, cfg.baseline_window_s, cfg.baseline_percentile,
        )
    else:
        raise ValueError(f"Unknown baseline_mode: {cfg.baseline_mode}")

    noise = estimate_noise_from_quiet(
        smooth, baseline_trace,
        quiet_percentile=cfg.noise_quiet_percentile,
        noise_mad_factor=cfg.noise_mad_factor,
    )
    end_threshold_trace = baseline_trace + cfg.end_threshold_k * noise

    if peaks.size == 0:
        empty_f = np.array([], dtype=np.float64)
        empty_o = np.array([], dtype=object)
        return EventTable(
            start_s=empty_f, peak_s=empty_f, end_s=empty_f,
            peak_height=empty_f, prominence=empty_f, duration_s=empty_f,
            baseline_trace=baseline_trace,
            end_threshold_trace=end_threshold_trace,
            baseline_noise=noise,
            mu_s=empty_f, sigma_s=empty_f,
            boundary_source_left=empty_o, boundary_source_right=empty_o,
        )

    order = np.argsort(peaks)
    peaks = peaks[order]

    # 2. baseline walk for each peak
    max_steps = max(1, int(round(cfg.max_event_duration_s / dt)))
    start_idx = np.empty(peaks.size, dtype=np.int64)
    end_idx = np.empty(peaks.size, dtype=np.int64)
    for i, p in enumerate(peaks):
        start_idx[i] = walk_boundary(smooth, int(p), end_threshold_trace, -1, max_steps)
        end_idx[i] = walk_boundary(smooth, int(p), end_threshold_trace, +1, max_steps)

    # 3. merge overlapping / touching events
    merge_gap_samples = max(0, int(round(cfg.merge_gap_s / dt)))
    keep = np.ones(peaks.size, dtype=bool)
    for i in range(1, peaks.size):
        j = i - 1
        while j >= 0 and not keep[j]:
            j -= 1
        if j < 0:
            continue
        if start_idx[i] - end_idx[j] <= merge_gap_samples:
            if smooth[peaks[i]] > smooth[peaks[j]]:
                peaks[j] = peaks[i]
            end_idx[j] = max(end_idx[j], end_idx[i])
            start_idx[j] = min(start_idx[j], start_idx[i])
            keep[i] = False

    peaks = peaks[keep]
    start_idx = start_idx[keep]
    end_idx = end_idx[keep]
    n_events = peaks.size

    # 4. Gaussian-fit refinement (optional)
    mu_s = np.full(n_events, np.nan, dtype=np.float64)
    sigma_s = np.full(n_events, np.nan, dtype=np.float64)
    src_left = np.empty(n_events, dtype=object)
    src_right = np.empty(n_events, dtype=object)

    base_start_s = time_s[start_idx].astype(np.float64)
    base_end_s = time_s[end_idx].astype(np.float64)

    if cfg.use_gaussian_boundary:
        z = _gaussian_z(cfg.gaussian_quantile)
        pad_samples = max(0, int(round(cfg.gaussian_fit_pad_s / dt)))

        start_s_out = np.empty(n_events, dtype=np.float64)
        end_s_out = np.empty(n_events, dtype=np.float64)

        for i in range(n_events):
            mu, sig = fit_gaussian_to_peak(
                time_s=time_s,
                density=smooth,
                baseline_trace=baseline_trace,
                peak_idx=int(peaks[i]),
                left_idx=int(start_idx[i]),
                right_idx=int(end_idx[i]),
                pad_samples=pad_samples,
                min_sigma_s=cfg.gaussian_min_sigma_s,
            )
            mu_s[i] = mu
            sigma_s[i] = sig

            g_start = mu - z * sig
            g_end = mu + z * sig

            # Whichever comes first: tighter on each side
            chosen_start = max(g_start, base_start_s[i])
            chosen_end = min(g_end, base_end_s[i])

            # Never cross the peak
            peak_t = float(time_s[peaks[i]])
            chosen_start = min(chosen_start, peak_t)
            chosen_end = max(chosen_end, peak_t)

            start_s_out[i] = chosen_start
            end_s_out[i] = chosen_end
            src_left[i] = "gaussian" if g_start >= base_start_s[i] else "baseline"
            src_right[i] = "gaussian" if g_end <= base_end_s[i] else "baseline"
    else:
        start_s_out = base_start_s
        end_s_out = base_end_s
        src_left[:] = "baseline"
        src_right[:] = "baseline"

    peak_height = smooth[peaks]
    prominences_final = peak_height - end_threshold_trace[peaks]

    return EventTable(
        start_s=start_s_out,
        peak_s=time_s[peaks].astype(np.float64),
        end_s=end_s_out,
        peak_height=peak_height,
        prominence=prominences_final,
        duration_s=end_s_out - start_s_out,
        baseline_trace=baseline_trace,
        end_threshold_trace=end_threshold_trace,
        baseline_noise=noise,
        mu_s=mu_s,
        sigma_s=sigma_s,
        boundary_source_left=src_left,
        boundary_source_right=src_right,
    )


# ---------- plotting ----------

def plot_events_with_counts(
    time_s: np.ndarray,
    smooth_density: np.ndarray,
    binned_counts: np.ndarray,
    events: EventTable,
    ylabel: str = "Onset density (per bin per ROI)",
    title: str = "Detected events on onset density",
) -> plt.Figure:
    """
    Figure: binned counts (bars) + smoothed density (line) + baseline +
    end threshold + shaded event windows + peak markers.
    """
    fig, ax = plt.subplots(figsize=(14, 4.5))

    bin_width = float(np.median(np.diff(time_s))) if time_s.size > 1 else 1.0

    # Binned counts as translucent bars
    ax.bar(
        time_s, binned_counts,
        width=bin_width, alpha=0.35, align="center",
        color="C0", edgecolor="none", label="Binned counts",
    )

    # Smoothed density
    ax.plot(time_s, smooth_density, linewidth=1.5, color="C0",
            label="Smoothed density")

    # Baseline and end threshold
    ax.plot(time_s, events.baseline_trace, color="gray", linestyle=":",
            linewidth=1.0, label="Baseline")
    ax.plot(time_s, events.end_threshold_trace, color="black", linestyle="--",
            linewidth=1.0, label="End threshold")

    # Shade each event window
    for s, e in zip(events.start_s, events.end_s):
        ax.axvspan(s, e, color="C1", alpha=0.20)

    # Peak markers
    ax.plot(events.peak_s, events.peak_height, "v", color="C3", markersize=6,
            label=f"Peaks (n={events.peak_s.size})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


# ---------- convenience runner ----------

def run_pipeline(
    onset_cfg,
    boundary_cfg: EventBoundaryConfig,
    peak_kwargs: Optional[dict] = None,
) -> tuple[EventTable, dict]:
    """
    Run the full pipeline:
        1. event_detection.run_onset_density   -> density + smoothed + binned
        2. event_detection.detect_density_peaks -> peaks
        3. detect_boundaries_from_peaks        -> EventTable
        4. plot_events_with_counts             -> figure
    """
    import event_detection as ed
    from dataclasses import replace

    quiet_cfg = replace(onset_cfg, show=False, save_path=None)
    density_result = ed.run_onset_density(quiet_cfg)

    centers = density_result["time_centers_s"]
    smooth = density_result["smoothed_density"]
    counts = density_result["binned_density"]

    if peak_kwargs is None:
        peak_kwargs = dict(min_prominence=0.007, min_width_bins=2, min_distance_bins=3)

    _peak_table, peak_indices, _props = ed.detect_density_peaks(
        centers=centers,
        smooth=smooth,
        counts=counts,
        **peak_kwargs,
    )

    events = detect_boundaries_from_peaks(
        time_s=centers,
        smooth_density=smooth,
        peak_indices=peak_indices,
        cfg=boundary_cfg,
    )

    if boundary_cfg.save_csv is not None:
        boundary_cfg.save_csv.parent.mkdir(parents=True, exist_ok=True)
        header = "start_s,peak_s,end_s,peak_height,prominence,duration_s,mu_s,sigma_s"
        data = np.column_stack([
            events.start_s, events.peak_s, events.end_s,
            events.peak_height, events.prominence, events.duration_s,
            events.mu_s, events.sigma_s,
        ])
        np.savetxt(boundary_cfg.save_csv, data, delimiter=",",
                   header=header, comments="")
        print(f"Saved events to: {boundary_cfg.save_csv}")

    fig = plot_events_with_counts(
        time_s=centers,
        smooth_density=smooth,
        binned_counts=counts,
        events=events,
        title=f"Detected events on onset density (n={events.peak_s.size})",
    )

    if boundary_cfg.save_fig is not None:
        boundary_cfg.save_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(boundary_cfg.save_fig, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {boundary_cfg.save_fig}")

    if boundary_cfg.show:
        plt.show()
    else:
        plt.close(fig)

    return events, density_result


# ---------- entry point ----------

if __name__ == "__main__":
    import event_detection as ed

    onset_cfg = ed.OnsetDensityConfig(
        root=Path(r"D:\sparse_plus_cellpose\2024-11-20_00003\final\suite2p\plane0"),
        prefix="r0p7_",
        fps=15.0,
        z_enter=3.5,
        z_exit=1.5,
        min_sep_s=0.1,
        t_start_s=0.0,
        t_end_s=None,
        bin_sec=0.05,
        smooth_sigma_bins=2.0,
        normalize_by_num_rois=True,
    )

    boundary_cfg = EventBoundaryConfig(
        # Baseline
        baseline_mode="rolling",
        baseline_percentile=5.0,
        baseline_window_s=3.0,
        noise_quiet_percentile=40.0,
        # Boundaries
        end_threshold_k=1.0,
        max_event_duration_s=10.0,
        merge_gap_s=0.0,
        # Gaussian-fit refinement
        use_gaussian_boundary=True,
        gaussian_quantile=0.999,
        gaussian_fit_pad_s=0.5,
        gaussian_min_sigma_s=0.05,
        save_csv=None,
        save_fig=None,
        show=True,
    )

    # Peak detection knobs forwarded to event_detection.detect_density_peaks
    peak_kwargs = dict(
        min_prominence=0.007,
        min_width_bins=2,
        min_distance_bins=3,
    )

    events, _ = run_pipeline(onset_cfg, boundary_cfg, peak_kwargs=peak_kwargs)

    print(f"\nDetected {events.peak_s.size} events")
    print(f"Baseline (median) = {np.median(events.baseline_trace):.5f}, "
          f"noise = {events.baseline_noise:.5f}")
    print(f"End threshold (median) = {np.median(events.end_threshold_trace):.5f}")
    if events.peak_s.size > 0:
        print(f"Median duration = {np.median(events.duration_s):.2f}s")
        print(f"Median sigma = {np.nanmedian(events.sigma_s):.3f}s")
        g_l = int((events.boundary_source_left == 'gaussian').sum())
        g_r = int((events.boundary_source_right == 'gaussian').sum())
        n = events.peak_s.size
        print(f"Gaussian chose boundary: left {g_l}/{n}, right {g_r}/{n}")