"""
Event detection on the flattened onset-density trace.

Replaces the per-bin "percentage activated" approach. Given a smoothed onset
density d(t), this module identifies discrete population events, each described
by (start_time, peak_time, end_time).

Pipeline
--------
1. Compute a robust baseline and noise estimate on the smoothed density, using
   a low percentile (not the mean) so that the peaks themselves do not inflate
   the baseline.
2. Run scipy.signal.find_peaks with a minimum prominence and a minimum peak
   separation (refractory-like period) to get candidate event peaks.
3. For each peak, walk outward left and right until the smoothed density
   drops below an end-threshold (baseline + k * baseline_noise). This gives
   per-event (start, end) times that adapt to the local baseline.
4. Merge any overlapping or touching events (e.g. doublets that share a
   boundary) into a single event with multiple peaks if desired, or keep them
   separate.

The module is designed to be used after event_detection.run_onset_density()
which already returns the smoothed density on a common time axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import utils
import event_detection as ed


@dataclass
class EventBoundaryConfig:
    # Peak detection on the smoothed density
    min_prominence: Optional[float] = None
        # If None, auto-set to max(k_prom * baseline_noise, min_prominence_abs).
    k_prominence: float = 8.0
        # Prominence threshold in units of robust noise of the density baseline.
    min_prominence_abs: Optional[float] = None
        # Hard floor on prominence (density units). Prevents degradation when
        # the noise estimate is inflated by dense event activity. If None,
        # auto-set to 0.5 * (median peak-region estimate). Set explicitly to
        # something like 0.003 if autoselection misbehaves.
    prominence_wlen_s: float = 8.0
        # Width of the local window (seconds) used by find_peaks to compute
        # prominence. Keeps prominence measured LOCALLY so a peak in the dense
        # middle region isn't compared against the quietest dip 200s away.
    min_peak_distance_s: float = 1.0
        # Minimum separation between detected event peaks.
    min_peak_height: Optional[float] = None
        # Optional hard floor on peak height (in density units). Usually not
        # needed — prominence is the primary gate.

    # Baseline / noise estimation
    baseline_mode: str = "rolling"
        # "global" = single percentile across whole trace (old behaviour).
        # "rolling" = time-local percentile baseline; much more robust when
        # event density varies across the recording (your case).
    baseline_percentile: float = 5.0
        # Baseline = this percentile of the smoothed density. Low value (5)
        # targets the truly quiet inter-event floor, not the typical value.
    baseline_window_s: float = 30.0
        # Window for rolling-baseline mode. Should be long enough to contain
        # several events (so the low percentile lands in quiet time) but short
        # enough to track genuine drift. 20–60s is reasonable.
    noise_quiet_percentile: float = 40.0
        # Noise MAD is computed from residuals (density - baseline_trace) on
        # the bottom `quiet_percentile` of those residuals -- i.e. the samples
        # sitting near baseline. This captures the visible wiggle around
        # baseline rather than the extreme low tail, which collapses after
        # smoothing.
    noise_mad_factor: float = 1.4826
        # MAD -> sigma conversion for Gaussian-distributed baseline noise.

    # Boundary walking
    end_threshold_k: float = 2.0
        # Event ends when density drops below baseline + k * baseline_noise.
        # With rolling baseline, this uses the local baseline at each peak.
    max_event_duration_s: float = 10.0
        # Hard cap on one-sided walk distance from peak; prevents runaway
        # boundaries when two events are very close and never cross threshold.
    merge_gap_s: float = 0.0
        # If two events are closer than this, merge them into one.
        # 0 = only merge when they actually touch/overlap.

    # Gaussian-fit boundary refinement
    use_gaussian_boundary: bool = True
        # If True, after baseline-walking we fit a Gaussian to each peak and
        # take boundaries = min(gaussian_quantile, baseline_walk) -- whichever
        # is tighter. If False, use baseline-walk only.
    gaussian_quantile: float = 0.99
        # Right-tail quantile for the end time; symmetric for the start.
        # 0.99 -> mu +/- 2.326 * sigma, 0.975 -> +/- 1.96, 0.95 -> +/- 1.645
    gaussian_fit_pad_s: float = 0.5
        # Pad the baseline-walk window by this much on each side when fitting,
        # to capture tails that the baseline walk cut off early.
    gaussian_min_sigma_s: float = 0.05
        # Hard floor on fitted sigma to avoid degenerate single-bin fits.

    # Output
    save_csv: Optional[Path] = None
    save_fig: Optional[Path] = None
    show: bool = True


@dataclass
class EventTable:
    start_s: np.ndarray          # (n_events,)
    peak_s: np.ndarray           # (n_events,)
    end_s: np.ndarray            # (n_events,)
    peak_height: np.ndarray      # (n_events,) density at peak
    prominence: np.ndarray       # (n_events,)
    duration_s: np.ndarray       # (n_events,) = end_s - start_s
    baseline_trace: np.ndarray   # (n_time,) baseline used (constant or rolling)
    end_threshold_trace: np.ndarray  # (n_time,) end-threshold used
    baseline_noise: float        # scalar noise used
    # Gaussian-fit diagnostics (NaN-filled if use_gaussian_boundary is False)
    mu_s: np.ndarray             # (n_events,) fitted mean time
    sigma_s: np.ndarray          # (n_events,) fitted sigma
    # Per-event boundary source: "gaussian" (tighter), "baseline", or "equal"
    boundary_source_left: np.ndarray   # (n_events,) object dtype
    boundary_source_right: np.ndarray  # (n_events,) object dtype

    def as_dict(self) -> dict:
        return {
            "start_s": self.start_s,
            "peak_s": self.peak_s,
            "end_s": self.end_s,
            "peak_height": self.peak_height,
            "prominence": self.prominence,
            "duration_s": self.duration_s,
            "baseline_trace": self.baseline_trace,
            "end_threshold_trace": self.end_threshold_trace,
            "baseline_noise": self.baseline_noise,
            "mu_s": self.mu_s,
            "sigma_s": self.sigma_s,
        }


# ---------- core routines ----------

def estimate_noise_from_quiet(
    density: np.ndarray,
    baseline_trace: np.ndarray,
    quiet_percentile: float = 40.0,
    noise_mad_factor: float = 1.4826,
) -> float:
    """
    Estimate baseline noise from residuals around the baseline.

    We take samples where (density - baseline) is small (bottom `quiet_percentile`
    of the residual distribution), and compute MAD on that set. This gives us
    the visible wiggle around baseline, not the width of the extreme low tail
    (which collapses after smoothing).
    """
    if density.size == 0:
        return 0.0
    resid = density - baseline_trace
    # only positive residuals matter -- density rarely dips below baseline in
    # a non-negative density, but we keep both sides for robustness
    cutoff = float(np.percentile(resid, quiet_percentile))
    quiet_mask = resid <= cutoff
    if quiet_mask.sum() < 32:
        quiet_mask = np.ones_like(resid, dtype=bool)
    sample = resid[quiet_mask]
    med = np.median(sample)
    mad = np.median(np.abs(sample - med)) + 1e-12
    return float(noise_mad_factor * mad)


def estimate_global_baseline(
    density: np.ndarray,
    baseline_percentile: float = 5.0,
) -> np.ndarray:
    """
    Return a constant baseline trace (same length as density).
    """
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
    """
    Rolling percentile baseline.

    fps_density is samples-per-second of the density trace (1 / bin_sec).
    Uses scipy.ndimage.percentile_filter with a centered window; falls back to
    reflect mode so the edges don't get starved.
    """
    from scipy.ndimage import percentile_filter
    if density.size == 0:
        return np.zeros_like(density)
    win = max(3, int(round(baseline_window_s * fps_density)) | 1)  # odd
    win = min(win, density.size if density.size % 2 == 1 else density.size - 1)
    if win < 3:
        return estimate_global_baseline(density, baseline_percentile)
    return percentile_filter(
        density.astype(np.float64),
        size=win,
        percentile=baseline_percentile,
        mode="reflect",
    )


def walk_boundary(
    density: np.ndarray,
    peak_idx: int,
    end_threshold: np.ndarray,
    direction: int,
    max_steps: int,
) -> int:
    """
    Walk outward from peak_idx in `direction` (+1 right, -1 left) until the
    density drops to or below end_threshold[i], or we exceed max_steps.

    end_threshold is a per-sample array so rolling baselines work naturally.
    Returns the last index where density was still above threshold on the
    peak side.
    """
    n = density.size
    i = peak_idx
    steps = 0
    while 0 <= i + direction < n and steps < max_steps:
        nxt = i + direction
        if density[nxt] <= end_threshold[nxt]:
            return i
        i = nxt
        steps += 1
    return i


# Z-score cutoff for one-sided Gaussian quantile q (q in (0.5, 1.0))
# mu + z(q) * sigma is the upper quantile; mu - z(q) * sigma is the lower.
def _gaussian_z(q: float) -> float:
    """One-sided Gaussian quantile z-score. q in (0.5, 1.0)."""
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
    Fit a Gaussian to the density above baseline inside a padded window
    [left_idx - pad, right_idx + pad] around peak_idx using moment matching.

    Weights are max(0, density - baseline_trace). Subtracting baseline is
    critical: otherwise the flat floor dominates the variance calculation and
    sigma blows up.

    Returns (mu_s, sigma_s). If the fit degenerates (no mass, zero variance),
    falls back to (time_s[peak_idx], min_sigma_s).
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
    sigma = float(np.sqrt(max(var, 0.0)))
    sigma = max(sigma, float(min_sigma_s))
    return mu, sigma


def detect_events_on_density(
    time_s: np.ndarray,
    density: np.ndarray,
    cfg: EventBoundaryConfig,
) -> EventTable:
    """
    Detect events on a (pre-smoothed) density trace and return an EventTable.
    """
    if time_s.shape != density.shape:
        raise ValueError(
            f"time_s and density must have the same shape; "
            f"got {time_s.shape} vs {density.shape}"
        )
    if time_s.size < 3:
        raise ValueError("Need at least 3 samples to detect events.")

    dt = float(np.median(np.diff(time_s)))
    if dt <= 0:
        raise ValueError("time_s must be strictly increasing.")
    fps_density = 1.0 / dt

    density = density.astype(np.float64, copy=False)

    # 1. baseline (constant or rolling) + noise from quiet samples only
    if cfg.baseline_mode == "global":
        baseline_trace = estimate_global_baseline(density, cfg.baseline_percentile)
    elif cfg.baseline_mode == "rolling":
        baseline_trace = estimate_rolling_baseline(
            density,
            fps_density=fps_density,
            baseline_window_s=cfg.baseline_window_s,
            baseline_percentile=cfg.baseline_percentile,
        )
    else:
        raise ValueError(f"Unknown baseline_mode: {cfg.baseline_mode}")

    noise = estimate_noise_from_quiet(
        density,
        baseline_trace=baseline_trace,
        quiet_percentile=cfg.noise_quiet_percentile,
        noise_mad_factor=cfg.noise_mad_factor,
    )

    end_threshold_trace = baseline_trace + cfg.end_threshold_k * noise

    # 2. peak detection on smoothed density with LOCAL prominence window
    if cfg.min_prominence is not None:
        prominence = cfg.min_prominence
    else:
        prom_from_noise = cfg.k_prominence * max(noise, 1e-12)
        prom_floor = (
            cfg.min_prominence_abs
            if cfg.min_prominence_abs is not None
            else 0.0
        )
        prominence = max(prom_from_noise, prom_floor)

    distance_samples = max(1, int(round(cfg.min_peak_distance_s / dt)))
    wlen_samples = max(3, int(round(cfg.prominence_wlen_s / dt)) | 1)

    peaks, _props = find_peaks(
        density,
        prominence=prominence,
        distance=distance_samples,
        height=cfg.min_peak_height,
        wlen=wlen_samples,
    )

    if peaks.size == 0:
        return EventTable(
            start_s=np.array([]), peak_s=np.array([]), end_s=np.array([]),
            peak_height=np.array([]), prominence=np.array([]),
            duration_s=np.array([]),
            baseline_trace=baseline_trace,
            end_threshold_trace=end_threshold_trace,
            baseline_noise=noise,
            mu_s=np.array([]), sigma_s=np.array([]),
            boundary_source_left=np.array([], dtype=object),
            boundary_source_right=np.array([], dtype=object),
        )

    max_steps = max(1, int(round(cfg.max_event_duration_s / dt)))

    # 3. walk boundaries for each peak (per-sample end threshold)
    start_idx = np.empty(peaks.size, dtype=np.int64)
    end_idx = np.empty(peaks.size, dtype=np.int64)
    for i, p in enumerate(peaks):
        left = walk_boundary(density, p, end_threshold_trace, direction=-1, max_steps=max_steps)
        right = walk_boundary(density, p, end_threshold_trace, direction=+1, max_steps=max_steps)
        start_idx[i] = left
        end_idx[i] = right

    # 4. merge overlapping / touching events
    merge_gap_samples = max(0, int(round(cfg.merge_gap_s / dt)))
    keep = np.ones(peaks.size, dtype=bool)
    for i in range(1, peaks.size):
        # if the previous event's end touches or overlaps this event's start
        # (within merge_gap), fold this event into the previous one and
        # keep whichever peak is taller.
        j = i - 1
        while j >= 0 and not keep[j]:
            j -= 1
        if j < 0:
            continue
        if start_idx[i] - end_idx[j] <= merge_gap_samples:
            # extend previous
            if density[peaks[i]] > density[peaks[j]]:
                # prefer taller peak as representative
                peaks[j] = peaks[i]
            end_idx[j] = max(end_idx[j], end_idx[i])
            start_idx[j] = min(start_idx[j], start_idx[i])
            keep[i] = False

    peaks = peaks[keep]
    start_idx = start_idx[keep]
    end_idx = end_idx[keep]

    # 5. Gaussian-fit boundary refinement (optional)
    n_events = peaks.size
    mu_s = np.full(n_events, np.nan, dtype=np.float64)
    sigma_s = np.full(n_events, np.nan, dtype=np.float64)
    src_left = np.empty(n_events, dtype=object)
    src_right = np.empty(n_events, dtype=object)

    # Baseline-walk boundaries in seconds -- our "slow" safety clamp
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
                density=density,
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

            # "Whichever comes first": tighter on each side
            # right side: min of gaussian_end and baseline_end
            # left  side: max of gaussian_start and baseline_start
            chosen_start = max(g_start, base_start_s[i])
            chosen_end = min(g_end, base_end_s[i])

            # Defensive: never cross the peak
            peak_t = float(time_s[peaks[i]])
            chosen_start = min(chosen_start, peak_t)
            chosen_end = max(chosen_end, peak_t)

            start_s_out[i] = chosen_start
            end_s_out[i] = chosen_end

            # tag source for diagnostics
            src_left[i] = "gaussian" if g_start >= base_start_s[i] else "baseline"
            src_right[i] = "gaussian" if g_end <= base_end_s[i] else "baseline"
    else:
        start_s_out = base_start_s
        end_s_out = base_end_s
        src_left[:] = "baseline"
        src_right[:] = "baseline"

    peak_height = density[peaks]
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

def plot_detected_events(
    time_s: np.ndarray,
    density: np.ndarray,
    events: EventTable,
    title: str = "Detected events on onset density",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(time_s, density, linewidth=1.2, color="C0", label="Smoothed density")

    ax.plot(time_s, events.baseline_trace, color="gray", linestyle=":",
            linewidth=1.0, label="Baseline")
    ax.plot(time_s, events.end_threshold_trace, color="black", linestyle="--",
            linewidth=1.0, label="End threshold")

    # shade each event span
    for s, e in zip(events.start_s, events.end_s):
        ax.axvspan(s, e, color="C1", alpha=0.20)

    # mark peaks
    ax.plot(events.peak_s, events.peak_height, "v", color="C3", markersize=6,
            label=f"Peaks (n={events.peak_s.size})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Onset density (per bin per ROI)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


# ---------- convenience runner that glues everything together ----------

def run_event_boundaries(
    onset_cfg: ed.OnsetDensityConfig,
    boundary_cfg: EventBoundaryConfig,
) -> tuple[EventTable, dict]:
    """
    Run the full pipeline: extract onsets -> build density -> detect events.
    Returns (events, onset_density_result_dict).
    """
    # reuse existing density builder — turn off its plot/save, we do our own
    onset_cfg = _copy_cfg_no_show(onset_cfg)
    density_result = ed.run_onset_density(onset_cfg)

    events = detect_events_on_density(
        time_s=density_result["time_centers_s"],
        density=density_result["smoothed_density"],
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

    fig = plot_detected_events(
        time_s=density_result["time_centers_s"],
        density=density_result["smoothed_density"],
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


def _copy_cfg_no_show(cfg: ed.OnsetDensityConfig) -> ed.OnsetDensityConfig:
    """Return a copy of the onset cfg with show=False and save_path=None."""
    from dataclasses import replace
    return replace(cfg, show=False, save_path=None)


# ---------- entry point ----------

if __name__ == "__main__":
    onset_cfg = ed.OnsetDensityConfig(
        root=Path(r"F:\data\2p_shifted\Hip\2024-06-04_00001\suite2p\plane0"),
        prefix="r0p7_filtered_",
        fps=15.0,
        z_enter=3.5,
        z_exit=1.5,
        min_sep_s=0,
        t_start_s=0.0,
        t_end_s=None,
        bin_sec=0.2,
        smooth_sigma_bins=3.0,
        normalize_by_num_rois=True,
    )

    boundary_cfg = EventBoundaryConfig(
        # Peak detection
        min_prominence=None,          # auto via k_prominence * noise
        k_prominence=10.0,
        min_prominence_abs=0.003,     # absolute safety floor in density units
        prominence_wlen_s=6.0,        # local window for prominence
        min_peak_distance_s=0.5,
        min_peak_height=None,
        # Baseline
        baseline_mode="rolling",      # "rolling" or "global"
        baseline_percentile=5.0,
        baseline_window_s=3.0,
        noise_quiet_percentile=40.0,
        # Boundaries
        end_threshold_k=1.0,
        max_event_duration_s=10.0,
        merge_gap_s=0.1,
        # Gaussian-fit refinement
        use_gaussian_boundary=False,
        gaussian_quantile=0.95,       # one-sided; 0.99 -> mu +/- 2.326 * sigma
        gaussian_fit_pad_s=0.5,
        gaussian_min_sigma_s=0.05,
        save_csv=None,
        save_fig=None,
        show=True,
    )

    events, _ = run_event_boundaries(onset_cfg, boundary_cfg)

    print(f"Detected {events.peak_s.size} events")
    print(f"Baseline (median) = {np.median(events.baseline_trace):.5f}, "
          f"noise = {events.baseline_noise:.5f}")
    print(f"End threshold (median) = {np.median(events.end_threshold_trace):.5f}")
    if events.peak_s.size > 0:
        print(f"Median duration = {np.median(events.duration_s):.2f}s")
        print(f"Median peak height = {np.median(events.peak_height):.5f}")