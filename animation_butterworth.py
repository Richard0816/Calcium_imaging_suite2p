from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.signal import butter, sosfilt

import utils


# --------------------------------------------------
# Config, matches your ExampleROI
# --------------------------------------------------
ROOT = Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0")
ROI = 21
CROP_START_S = 200.0
CROP_END_S = 450.0

FPS = 15.0
NEUROPIL_R = 0.7

FILTER_ORDER = 2

# --- Build cutoff sequence (reverse + pause at 1 Hz) ---
forward = np.linspace(8, 1, 120)


# Find closest index to 1 Hz
idx_1hz = np.argmin(np.abs(forward-1))

# Insert pause frames (0.3 s)
fps_anim = 24  # GIF playback FPS
pause_frames = int(round(1 * fps_anim))

cutoffs = list(forward)

# duplicate the 1 Hz frame
cutoffs[idx_1hz:idx_1hz] = [cutoffs[idx_1hz]] * pause_frames


SAVE_PATH = ROOT / f"roi_{ROI:03d}_butterworth_cutoff_sweep.mp4"
GIF_FALLBACK_PATH = ROOT / f"roi_{ROI:03d}_butterworth_cutoff_sweep.gif"

LINEWIDTH_RAW = 0.9
LINEWIDTH_FILT = 1.5


# --------------------------------------------------
# Loading
# --------------------------------------------------
def load_roi_raw_and_neuropil(root: Path, roi: int):
    F, Fneu, num_frames, num_rois, time_major = utils.s2p_load_raw(root)

    if roi < 0 or roi >= num_rois:
        raise IndexError(f"ROI {roi} out of bounds for N={num_rois}")

    if time_major:
        raw = np.asarray(F[:, roi], dtype=np.float32).reshape(-1)
        neu = np.asarray(Fneu[:, roi], dtype=np.float32).reshape(-1)
    else:
        raw = np.asarray(F[roi, :], dtype=np.float32).reshape(-1)
        neu = np.asarray(Fneu[roi, :], dtype=np.float32).reshape(-1)

    return raw, neu


def neuropil_correct(raw: np.ndarray, neu: np.ndarray, r: float = 0.7) -> np.ndarray:
    return raw - r * neu


def build_time_mask(n_frames: int, fps: float, start_s: float, end_s: float) -> np.ndarray:
    t = np.arange(n_frames, dtype=np.float32) / float(fps)
    return (t >= start_s) & (t <= end_s)


# --------------------------------------------------
# Filtering
# --------------------------------------------------
def causal_butterworth_lowpass(x: np.ndarray, fps: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    nyq = fps / 2.0
    cutoff_hz = float(np.clip(cutoff_hz, 1e-4, 0.95 * nyq))
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")

    y = sosfilt(sos, x)
    return np.asarray(y, dtype=np.float32)


# --------------------------------------------------
# Animation
# --------------------------------------------------
def make_animation():
    raw, neu = load_roi_raw_and_neuropil(ROOT, ROI)
    corrected = neuropil_correct(raw, neu, r=NEUROPIL_R)

    mask = build_time_mask(len(corrected), FPS, CROP_START_S, CROP_END_S)
    t = np.arange(len(corrected), dtype=np.float32) / float(FPS)

    t_crop = t[mask]
    corr_crop = corrected[mask]

    if t_crop.size == 0:
        raise ValueError("Crop window produced an empty trace.")

    # --- Single clean panel ---
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

    # faint input
    ax.plot(
        t_crop,
        corr_crop,
        linewidth=1.0,
        alpha=0.25,
        label="input"
    )

    # animated filtered line
    filt_line, = ax.plot([], [], linewidth=1.8, label="filtered")

    # styling
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F")
    ax.legend(frameon=False)

    # remove clutter
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # stable limits
    y_min = float(np.nanmin(corr_crop))
    y_max = float(np.nanmax(corr_crop))
    y_pad = 0.08 * (y_max - y_min + 1e-9)

    ax.set_xlim(float(t_crop[0]), float(t_crop[-1]))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    title = ax.set_title("")

    def init():
        filt_line.set_data([], [])
        title.set_text("Butterworth filter")
        return filt_line, title

    def update(frame_idx):
        cutoff = float(cutoffs[frame_idx])

        filtered = causal_butterworth_lowpass(
            corr_crop,
            FPS,
            cutoff_hz=cutoff,
            order=FILTER_ORDER
        )

        filt_line.set_data(t_crop, filtered)
        title.set_text(f"cutoff = {cutoff:.2f} Hz")

        return filt_line, title

    anim = FuncAnimation(
        fig,
        update,
        frames=len(cutoffs),
        interval=1000 / fps_anim,
        blit=False
    )

    return fig, anim

def save_animation(anim, mp4_path: Path, gif_path: Path):
    from matplotlib.animation import PillowWriter

    gif_path = ROOT / f"roi_{ROI:03d}_butterworth_reverse_pause.gif"

    writer = PillowWriter(fps=fps_anim)
    anim.save(str(gif_path), writer=writer)

    print(f"Saved GIF: {gif_path}")

if __name__ == "__main__":
    fig, anim = make_animation()
    save_animation(anim, SAVE_PATH, GIF_FALLBACK_PATH)