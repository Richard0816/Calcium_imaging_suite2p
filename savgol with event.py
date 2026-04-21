from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import savgol_filter

import utils


# --------------------------------------------------
# Config
# --------------------------------------------------
ROOT = Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0")
ROI = 21
CROP_START_S = 295.0
CROP_END_S = 310.0

FPS = 15.0
NEUROPIL_R = 0.7

SG_WIN_MS = 333
SG_POLY = 3

Z_ENTER = 3.5
Z_EXIT = 2.0
MIN_SEP_S = 0.0

ANIM_FPS = 12
STEP_FRAMES = 1

GIF_PATH = ROOT / f"roi_{ROI:03d}_savgol_derivative_hysteresis.gif"


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
# Savitzky-Golay derivative
# --------------------------------------------------
def sg_first_derivative_1d(x: np.ndarray, fps: float, win_ms: float = 333, poly: int = 3):
    x = np.asarray(x, dtype=np.float32)
    n = x.size

    win = max(3, int((win_ms / 1000.0) * fps) | 1)
    if win >= n:
        win = max(3, n if n % 2 == 1 else n - 1)

    if win < 3 or n < 3:
        g = np.empty_like(x)
        g[0] = 0.0
        g[1:] = (x[1:] - x[:-1]) * fps
        return g, win

    deriv = savgol_filter(
        x,
        window_length=win,
        polyorder=poly,
        deriv=1,
        delta=1.0 / fps
    ).astype(np.float32)

    return deriv, win


# --------------------------------------------------
# Hysteresis on robust z-score
# --------------------------------------------------
def compute_hysteresis_state(z: np.ndarray, z_enter: float, z_exit: float) -> np.ndarray:
    """
    Boolean state over time.
    True means currently inside an event.
    """
    z = np.asarray(z, dtype=np.float32)
    state = np.zeros(z.shape[0], dtype=bool)

    active = False
    for i, val in enumerate(z):
        if not active:
            if val >= z_enter:
                active = True
        else:
            if val <= z_exit:
                active = False
        state[i] = active

    return state


def onset_mask_from_indices(n: int, onset_idx: np.ndarray) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    onset_idx = np.asarray(onset_idx, dtype=int)
    onset_idx = onset_idx[(onset_idx >= 0) & (onset_idx < n)]
    mask[onset_idx] = True
    return mask


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

    deriv_crop, win_len = sg_first_derivative_1d(
        corr_crop,
        fps=FPS,
        win_ms=SG_WIN_MS,
        poly=SG_POLY
    )

    z_crop, med, mad = utils.mad_z(deriv_crop)
    z_crop = np.asarray(z_crop, dtype=np.float32)

    onset_idx = utils.hysteresis_onsets(
        z_crop,
        z_hi=Z_ENTER,
        z_lo=Z_EXIT,
        fps=FPS,
        min_sep_s=MIN_SEP_S
    )
    onset_bool = onset_mask_from_indices(len(z_crop), onset_idx)
    event_state = compute_hysteresis_state(z_crop, Z_ENTER, Z_EXIT)

    half_win = win_len // 2

    fig, (ax0, ax1) = plt.subplots(
        2, 1,
        figsize=(12, 6),
        sharex=True,
        constrained_layout=True
    )

    # Top panel, corrected trace with rolling window
    ax0.plot(t_crop, corr_crop, linewidth=1.0, alpha=0.22, color="black")
    trace_marker, = ax0.plot([], [], marker="o", linestyle="None", markersize=4, color="black")
    tangent_line, = ax0.plot([], [], linewidth=1.6, color="tab:blue")
    window_patch = ax0.axvspan(t_crop[0], t_crop[0], alpha=0.10, color="tab:blue")

    ax0.set_ylabel("ΔF/F")
    ax0.set_title("Savitzky-Golay rolling window and local derivative")

    # Bottom panel, derivative with hysteresis display
    ax1.plot(t_crop, deriv_crop, linewidth=1.0, alpha=0.14, color="black")
    deriv_line, = ax1.plot([], [], linewidth=1.8, color="tab:blue")
    deriv_marker, = ax1.plot([], [], marker="o", linestyle="None", markersize=4, color="tab:blue")

    zero_line = ax1.axhline(0, linewidth=2.0, color="0.55", alpha=0.9)
    enter_line = ax1.axhline(
        med + (1.4826 * mad) * Z_ENTER,
        linewidth=1.0,
        linestyle="--",
        color="0.65"
    )
    exit_line = ax1.axhline(
        med + (1.4826 * mad) * Z_EXIT,
        linewidth=1.0,
        linestyle=":",
        color="0.65"
    )

    onset_vlines = []

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("d/dt")
    ax1.set_title("Derivative with robust hysteresis onset detection")

    for ax in (ax0, ax1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Stable limits
    y0_min = float(np.nanmin(corr_crop))
    y0_max = float(np.nanmax(corr_crop))
    y0_pad = 0.08 * (y0_max - y0_min + 1e-9)
    ax0.set_xlim(float(t_crop[0]), float(t_crop[-1]))
    ax0.set_ylim(y0_min - y0_pad, y0_max + y0_pad)

    y1_abs = float(np.nanmax(np.abs(deriv_crop))) if deriv_crop.size else 1.0
    y1_lim = 1.08 * max(y1_abs, 1e-6)
    ax1.set_ylim(-y1_lim, y1_lim)

    title = fig.suptitle("", fontsize=12)

    current_patch = {"artist": window_patch}
    frame_indices = np.arange(half_win, len(t_crop) - half_win, STEP_FRAMES, dtype=int)
    if frame_indices.size == 0:
        frame_indices = np.arange(len(t_crop), dtype=int)

    def init():
        deriv_line.set_data([], [])
        tangent_line.set_data([], [])
        trace_marker.set_data([], [])
        deriv_marker.set_data([], [])
        title.set_text("Savitzky-Golay derivative with hysteresis thresholding")
        return deriv_line, tangent_line, trace_marker, deriv_marker, current_patch["artist"], zero_line, title

    def update(k):
        nonlocal onset_vlines

        i = int(frame_indices[k])

        left = max(0, i - half_win)
        right = min(len(t_crop), i + half_win + 1)

        t_win = t_crop[left:right]
        t0 = float(t_crop[i])
        y0 = float(corr_crop[i])
        d0 = float(deriv_crop[i])

        # Rolling window patch
        current_patch["artist"].remove()
        current_patch["artist"] = ax0.axvspan(
            float(t_crop[left]),
            float(t_crop[right - 1]),
            alpha=0.12
        )

        # Local derivative line
        dt_local = t_win - t0
        y_tangent = y0 + d0 * dt_local
        tangent_line.set_data(t_win, y_tangent)
        trace_marker.set_data([t0], [y0])

        # Progressive derivative
        deriv_line.set_data(t_crop[:i + 1], deriv_crop[:i + 1])
        deriv_marker.set_data([t0], [d0])

        # Onset markers up to current time
        for artist in onset_vlines:
            artist.remove()
        onset_vlines = []

        shown_onsets = np.where(onset_bool[:i + 1])[0]
        for j in shown_onsets:
            onset_vlines.append(
                ax1.axvline(
                    float(t_crop[j]),
                    linewidth=0.9,
                    color="tab:red",
                    alpha=0.5
                )
            )

        # Color the zero line when event is active
        if event_state[i]:
            zero_line.set_color("tab:red")
            zero_line.set_alpha(1.0)
        else:
            zero_line.set_color("0.55")
            zero_line.set_alpha(0.9)

        state_txt = "EVENT" if event_state[i] else "baseline"
        title.set_text(
            f"Savitzky-Golay derivative, window={win_len} frames, poly={SG_POLY}, "
            f"z={z_crop[i]:.2f}, state={state_txt}, t={t0:.2f}s"
        )

        artists = [
            deriv_line,
            tangent_line,
            trace_marker,
            deriv_marker,
            current_patch["artist"],
            zero_line,
            title,
        ]
        artists.extend(onset_vlines)
        return tuple(artists)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=1000 / ANIM_FPS,
        blit=False
    )

    return fig, anim


def save_animation(anim):
    writer = PillowWriter(fps=ANIM_FPS)
    anim.save(str(GIF_PATH), writer=writer)
    print(f"Saved GIF: {GIF_PATH}")


if __name__ == "__main__":
    fig, anim = make_animation()
    save_animation(anim)