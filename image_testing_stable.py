import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
import utils


# --------------------------- Config ---------------------------

@dataclass(frozen=True)
class Config:
    root: Path                 # Path to Suite2p outputs
    fps: float = 30.0          # Imaging frame rate (Hz)
    roi: int = 10              # ROI index to visualize
    t_max: Union[float, None] = None               # Seconds to plot (None = full trace)
    z_enter: float = 3.5       # z-score threshold to detect event onset
    z_exit: float = 2.0        # z-score threshold to detect event offset (hysteresis)
    save_path: Optional[Path] = None  # If provided, save the figure here (PNG)


# --------------------------- I/O & Loading ---------------------------
def load_suite2p_raw(root: Path, roi: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Load F and Fneu and return ROI traces in time-major form along with T and N.
    """
    F, Fneu, num_frames, num_rois, time_major = utils.s2p_load_raw(root)

    if F.shape != Fneu.shape:
        raise ValueError(f"F and Fneu shapes differ: {F.shape} vs {Fneu.shape}")

    if roi < 0 or roi >= num_rois:
        raise IndexError(f"ROI index {roi} out of bounds for N={num_rois}")

    if time_major:
        raw_trace = F[:, roi]
        neu_trace = Fneu[:, roi]
    else:
        raw_trace = F[roi, :]
        neu_trace = Fneu[roi, :]
        # Convert to time-major explicitly for downstream consistency
        raw_trace = np.asarray(raw_trace)
        neu_trace = np.asarray(neu_trace)

    if raw_trace.shape[0] != num_frames:
        # Ensure time-major (num_frames,)
        raw_trace = raw_trace.reshape(-1)
        neu_trace = neu_trace.reshape(-1)

    return raw_trace, neu_trace, num_frames, num_rois


def load_processed_traces(root: Path, T: int, N: int, roi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ΔF/F, low-pass ΔF/F, and derivative (all memmaps: (T, N)) and return
    1D ROI vectors (T,).
    """
    dff, low, dt, _, _ = utils.s2p_open_memmaps(root, prefix="r0p7_")

    dff_trace = np.asarray(dff[:, roi])
    low_trace = np.asarray(low[:, roi])
    dt_trace  = np.asarray(dt[:, roi])
    return dff_trace, low_trace, dt_trace


# --------------------------- Analytics ---------------------------

def robust_z_scores(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Robust z using MAD. Returns (z, median, MAD).
    utils.mad_z(x) is expected to do this; wrapped for clarity & single import site.
    """
    z, med, mad = utils.mad_z(x)
    return z, float(med), float(mad)


def detect_onsets_hysteresis(z: np.ndarray, z_enter: float, z_exit: float, fps: float) -> np.ndarray:
    """
    Wrapper for utils.hysteresis_onsets. Returns event times in seconds.
    """
    onsets_idx: Sequence[int] = utils.hysteresis_onsets(z, z_enter, z_exit, fps)
    return np.asarray(onsets_idx, dtype=np.int64) / fps


# --------------------------- Plotting ---------------------------
def plot_all(
    time: np.ndarray,
    raw_trace: np.ndarray,
    neu_trace: np.ndarray,
    dff_trace: np.ndarray,
    low_trace: np.ndarray,
    dt_trace: np.ndarray,
    event_times: np.ndarray,
    med: float,
    mad: float,
    z_enter: float,
    z_exit: float,
    t_max: Optional[float],
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Create the 4-panel figure and return the Figure object.
    """
    idx = utils.build_time_mask(time, t_max)

    fig = plt.figure(figsize=(12, 9))

    # Panel 1: Raw + neuropil fluorescence
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(time[idx], raw_trace[idx], label="F raw")
    ax1.plot(time[idx], neu_trace[idx], label="F neuropil", alpha=0.7)
    ax1.set_ylabel("Fluorescence")
    ax1.legend(loc="upper right")

    # Panel 2: ΔF/F
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax2.plot(time[idx], dff_trace[idx], label="ΔF/F", color="black")
    ax2.set_ylabel("ΔF/F")
    ax2.legend(loc="upper right")

    # Panel 3: Low-pass ΔF/F
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax3.plot(time[idx], low_trace[idx], label="Low-pass ΔF/F", color='green')
    ax3.set_ylabel("Filtered")
    ax3.legend(loc="upper right")

    # Panel 4: Derivative + thresholds + onsets
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)
    ax4.plot(time[idx], dt_trace[idx], label="Derivative", color='red')

    # Convert z thresholds back to derivative units for display
    # (MAD scaled by 1.4826 approximates σ if underlying data is normal)
    thr_enter_val = med + (1.4826 * mad) * z_enter
    thr_exit_val  = med + (1.4826 * mad) * z_exit
    ax4.axhline(thr_enter_val, linestyle="--", linewidth=1, label=f"Enter z={z_enter:.1f}")
    ax4.axhline(thr_exit_val,  linestyle=":",  linewidth=1, label=f"Exit z={z_exit:.1f}")

    # Event lines
    if event_times.size:
        for t0 in event_times:
            if t_max is None or t0 < t_max:
                ax4.axvline(t0, linestyle="-", linewidth=0.8, alpha=0.6)

    ax4.set_ylabel("d/dt (a.u./s)")
    ax4.set_xlabel("Time (s)")
    ax4.legend(loc="upper right")

    if title:
        fig.suptitle(title, y=0.995)

    fig.tight_layout()
    return fig


# --------------------------- Pipeline ---------------------------

def run(cfg: Config) -> None:
    # Load raw traces
    raw_trace, neu_trace, num_frames, num_rois = load_suite2p_raw(cfg.root, cfg.roi)

    # Processed traces (num_frames, num_rois) memmaps -> 1D (T,) for ROI
    dff_trace, low_trace, dt_trace = load_processed_traces(cfg.root, num_frames, num_rois, cfg.roi)

    # Robust z on derivative
    z, med, mad = robust_z_scores(dt_trace)

    # Hysteresis detection -> times (s)
    event_times = detect_onsets_hysteresis(z, cfg.z_enter, cfg.z_exit, cfg.fps)

    # Time axis
    time = np.arange(num_frames, dtype=float) / cfg.fps

    # Plot
    title = f"ROI {cfg.roi} • fps={cfg.fps:.1f} • onsets={len(event_times)}"
    fig = plot_all(
        time, raw_trace, neu_trace, dff_trace, low_trace, dt_trace,
        event_times, med, mad, cfg.z_enter, cfg.z_exit, cfg.t_max, title=title
    )

    # Save or show
    if cfg.save_path:
        cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {cfg.save_path}")
        plt.close(fig)
    else:
        plt.show()

    # Summary
    print(f"ROI {cfg.roi}: {len(event_times)} events")
    if event_times.size:
        preview = ", ".join(f"{t:.2f}s" for t in event_times[:10])
        print("Event times (first 10):", preview)


# --------------------------- CLI ---------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Visualize Suite2p traces and MAD-hysteresis events for one ROI.")
    p.add_argument("--root", required=True, type=Path, help="Path to Suite2p plane directory (contains F.npy, Fneu.npy, *.memmap.*).")
    p.add_argument("--fps", type=float, default=30.0, help="Imaging frame rate (Hz).")
    p.add_argument("--roi", type=int, default=10, help="ROI index to visualize.")
    p.add_argument("--t_max", type=float, default=None, help="Seconds to plot (None = full trace).")
    p.add_argument("--z_enter", type=float, default=3.5, help="z threshold to enter (onset).")
    p.add_argument("--z_exit", type=float, default=2.0, help="z threshold to exit (offset).")
    p.add_argument("--save", type=Path, default=None, help="Optional output PNG path to save the figure.")
    args = p.parse_args()

    return Config(
        root=args.root,
        fps=args.fps,
        roi=args.roi,
        t_max=args.t_max,
        z_enter=args.z_enter,
        z_exit=args.z_exit,
        save_path=args.save,
    )

# todo fast ROI loop and different batch z_entre/z_exit
if __name__ == "__main__":
    cfg = Config(
    root=Path(r'F:\data\2p_shifted\Hip\2024-06-04_00010\suite2p\plane0'),
    roi=61,
    fps=30.0,
    z_enter=3.5,
    z_exit=2.0,
    t_max=None,
    save_path=None
    )
    run(cfg)
