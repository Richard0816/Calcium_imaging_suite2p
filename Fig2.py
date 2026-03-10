import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Sequence

import utils


@dataclass(frozen=True)
class ExampleROI:
    root: Path
    roi: int
    nth_spike: int
    crop_start_s: float
    crop_end_s: float


@dataclass(frozen=True)
class MultiPanelConfig:
    examples: list[ExampleROI]
    fps: float = 15.0
    z_enter: float = 3.5
    z_exit: float = 2.0
    zoom_window_s: float = 12.0
    t_max: Optional[float] = None
    memmap_prefix: str = "r0p7_"
    save_path: Optional[Path] = None


def load_suite2p_raw(root: Path, roi: int):
    F, Fneu, num_frames, num_rois, time_major = utils.s2p_load_raw(root)

    if F.shape != Fneu.shape:
        raise ValueError(f"F and Fneu shapes differ: {F.shape} vs {Fneu.shape}")

    if roi < 0 or roi >= num_rois:
        raise IndexError(f"ROI index {roi} out of bounds for N={num_rois}")

    if time_major:
        raw_trace = np.asarray(F[:, roi])
        neu_trace = np.asarray(Fneu[:, roi])
    else:
        raw_trace = np.asarray(F[roi, :])
        neu_trace = np.asarray(Fneu[roi, :])

    raw_trace = raw_trace.reshape(-1)
    neu_trace = neu_trace.reshape(-1)

    return raw_trace, neu_trace, num_frames, num_rois


def load_processed_traces(root: Path, roi: int, prefix: str = "r0p7_"):
    dff, low, dt, T, N = utils.s2p_open_memmaps(root, prefix=prefix)
    dff_trace = np.asarray(dff[:, roi]).reshape(-1)
    low_trace = np.asarray(low[:, roi]).reshape(-1)
    dt_trace = np.asarray(dt[:, roi]).reshape(-1)
    return dff_trace, low_trace, dt_trace, T, N


def robust_z_scores(x: np.ndarray):
    z, med, mad = utils.mad_z(x)
    return np.asarray(z), float(med), float(mad)


def detect_onsets_hysteresis(z: np.ndarray, z_enter: float, z_exit: float, fps: float):
    onsets_idx: Sequence[int] = utils.hysteresis_onsets(z, z_enter, z_exit, fps)
    return np.asarray(onsets_idx, dtype=np.int64)


def load_stat_and_ops(root: Path):
    stat = np.load(root / "stat.npy", allow_pickle=True)
    ops = np.load(root / "ops.npy", allow_pickle=True).item()
    return stat, ops


def build_roi_overlay_image(root: Path, roi: int):
    stat, ops = load_stat_and_ops(root)

    # Prefer Suite2p max projection if available
    if "max_proj" in ops:
        img = ops["max_proj"]
    elif "meanImgE" in ops:
        img = ops["meanImgE"]
    elif "meanImg" in ops:
        img = ops["meanImg"]
    else:
        raise KeyError(f"No max_proj, meanImgE or meanImg found in ops.npy for {root}")

    roi_stat = stat[roi]
    ypix = roi_stat["ypix"]
    xpix = roi_stat["xpix"]

    return img, xpix, ypix


def get_zoom_window(trace: np.ndarray, event_idx: int, fps: float, window_s: float):
    half_w = int((window_s * fps) / 2)
    start = max(0, event_idx - half_w)
    end = min(len(trace), event_idx + half_w)
    return start, end


def add_roi_overlay(ax, img, xpix, ypix, label):
    ax.imshow(img, cmap="gray")
    ax.scatter(xpix, ypix, s=2, c="red", alpha=0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=10)


def make_crop_mask(time: np.ndarray, start_s: float, end_s: float, t_max: Optional[float] = None):
    if end_s <= start_s:
        raise ValueError(f"crop_end_s must be greater than crop_start_s. Got {start_s} to {end_s}")

    mask = (time >= start_s) & (time <= end_s)

    if t_max is not None:
        mask &= (time <= t_max)

    return mask


def plot_top_full_trace(ax, example: ExampleROI, fps: float, memmap_prefix: str):
    raw_trace, neu_trace, num_frames, _ = load_suite2p_raw(example.root, example.roi)

    time = np.arange(num_frames) / fps

    ax.plot(time, raw_trace, lw=1.0, color="tab:blue", label="Raw fluorescence")
    ax.plot(time, neu_trace, lw=1.0, color="tab:orange", alpha=0.7, label="Neuropil")

    ax.axvspan(
        example.crop_start_s,
        example.crop_end_s,
        color="tab:blue",
        alpha=0.15,
    )

    ax.set_ylabel("Fluorescence")
    ax.set_xlabel("Time (s)")
    ax.legend(frameon=False, fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_example_row(
    axes_row,
    root: Path,
    roi: int,
    nth_spike: int,
    crop_start_s: float,
    crop_end_s: float,
    fps: float,
    z_enter: float,
    z_exit: float,
    zoom_window_s: float,
    t_max: Optional[float],
    memmap_prefix: str,
):
    raw_trace, neu_trace, num_frames, _ = load_suite2p_raw(root, roi)
    dff_trace, low_trace, dt_trace, _, _ = load_processed_traces(root, roi, prefix=memmap_prefix)

    z, med, mad = robust_z_scores(dt_trace)
    onset_frames = detect_onsets_hysteresis(z, z_enter, z_exit, fps)

    time = np.arange(num_frames) / fps
    idx = make_crop_mask(time, crop_start_s, crop_end_s, t_max=t_max)

    img, xpix, ypix = build_roi_overlay_image(root, roi)

    ax0, ax1, ax2, ax3, ax4, ax5 = axes_row

    # Column 1: FOV
    add_roi_overlay(
        ax0,
        img,
        xpix,
        ypix,
        f"{root.parent.parent.name}\nROI {roi}"
    )

    # Column 2: raw, cropped
    ax1.plot(time[idx], raw_trace[idx], lw=1.0, color="tab:blue", label="F raw")
    ax1.plot(time[idx], neu_trace[idx], lw=1.0, color="tab:orange", alpha=0.8, label="F neuropil")
    ax1.set_xlim(crop_start_s, crop_end_s)

    # Column 3: dF/F, cropped
    ax2.plot(time[idx], dff_trace[idx], lw=1.0, color="black", label="ΔF/F")
    ax2.set_xlim(crop_start_s, crop_end_s)

    # Column 4: filtered, cropped
    ax3.plot(time[idx], low_trace[idx], lw=1.0, color="green", label="Low pass ΔF/F")
    ax3.set_xlim(crop_start_s, crop_end_s)

    # Column 5: zoomed filtered at nth event
    if len(onset_frames) > nth_spike:
        center = onset_frames[nth_spike]
        start, end = get_zoom_window(low_trace, center, fps, zoom_window_s)
        tz = np.arange(start, end) / fps
        ax4.plot(tz, low_trace[start:end], lw=1.2, color="green")
        ax4.axvline(center / fps, color="tab:blue", lw=1.0, alpha=0.8)
        ax4.set_title(f"Zoomed event", fontsize=10)
    else:
        ax4.text(
            0.5,
            0.5,
            f"Event {nth_spike + 1}\nnot available",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=9
        )

    # Column 6: derivative, cropped
    thr_enter_val = med + (1.4826 * mad) * z_enter
    thr_exit_val = med + (1.4826 * mad) * z_exit

    ax5.plot(time[idx], dt_trace[idx], lw=1.0, color="red", label="Derivative")
    ax5.axhline(thr_enter_val, linestyle="--", linewidth=1, color="tab:blue", label=f"Enter z={z_enter:.1f}")
    ax5.axhline(thr_exit_val, linestyle=":", linewidth=1, color="tab:blue", label=f"Exit z={z_exit:.1f}")
    ax5.set_xlim(crop_start_s, crop_end_s)

    for f in onset_frames:
        t0 = f / fps
        if crop_start_s <= t0 <= crop_end_s:
            ax5.axvline(t0, linestyle="-", linewidth=0.7, alpha=0.5, color="tab:blue")

    # Formatting
    for ax in (ax1, ax2, ax3, ax4, ax5):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return len(onset_frames)


def make_multipanel_figure(cfg: MultiPanelConfig):
    n_rows = len(cfg.examples)

    fig = plt.figure(figsize=(24, 3.4 + 4.2 * n_rows))
    gs = fig.add_gridspec(
        nrows=n_rows + 1,
        ncols=6,
        height_ratios=[1.2] + [4.0] * n_rows
    )

    # Top full trace panel spanning columns 2 to 6
    ax_top = fig.add_subplot(gs[0, 1:6])
    plot_top_full_trace(
        ax=ax_top,
        example=cfg.examples[0],
        fps=cfg.fps,
        memmap_prefix=cfg.memmap_prefix,
    )
    ax_top.set_xlabel("Time (s)")

    # Empty top-left cell for alignment
    ax_empty = fig.add_subplot(gs[0, 0])
    ax_empty.axis("off")

    # Build row axes
    axes = []
    for r in range(n_rows):
        row_axes = [fig.add_subplot(gs[r + 1, c]) for c in range(6)]
        axes.append(row_axes)
    axes = np.array(axes, dtype=object)

    col_titles = [
        "Field of view",
        "Raw fluorescence",
        "ΔF/F",
        "Low pass filtered",
        "Zoomed filtered event",
        "Derivative and event detection",
    ]

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=12)

    onset_counts = []

    for r, ex in enumerate(cfg.examples):
        n_onsets = plot_example_row(
            axes_row=axes[r],
            root=ex.root,
            roi=ex.roi,
            nth_spike=ex.nth_spike,
            crop_start_s=ex.crop_start_s,
            crop_end_s=ex.crop_end_s,
            fps=cfg.fps,
            z_enter=cfg.z_enter,
            z_exit=cfg.z_exit,
            zoom_window_s=cfg.zoom_window_s,
            t_max=cfg.t_max,
            memmap_prefix=cfg.memmap_prefix,
        )
        onset_counts.append(n_onsets)

        axes[r, 0].set_ylabel(f"Example {r + 1}", fontsize=11)

        if r == 0:
            axes[r, 1].legend(frameon=False, fontsize=8, loc="upper right")
            axes[r, 5].legend(frameon=False, fontsize=8, loc="upper right")

    # Shared axis labeling
    for r in range(n_rows):
        axes[r, 1].set_ylabel("Fluorescence")
        axes[r, 2].set_ylabel("ΔF/F")
        axes[r, 3].set_ylabel("Filtered")
        axes[r, 4].set_ylabel("Filtered")
        axes[r, 5].set_ylabel("d/dt")

    for c in range(1, 6):
        axes[-1, c].set_xlabel("Time (s)")

    fig.suptitle("ROI preprocessing and event detection workflow", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    if cfg.save_path is not None:
        cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {cfg.save_path}")

    return fig, onset_counts


if __name__ == "__main__":
    examples = [
        ExampleROI(
            root=Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0"),
            roi=21,
            nth_spike=0,
            crop_start_s=200.0,
            crop_end_s=450.0,
        ),
        ExampleROI(
            root=Path(r"F:\data\2p_shifted\Hip\2024-10-30_00003\suite2p\plane0"),
            roi=2,
            nth_spike=1,
            crop_start_s=350.0,
            crop_end_s=650.0,
        ),
        ExampleROI(
            root=Path(r"F:\data\2p_shifted\Hip\2024-06-03_00009\suite2p\plane0"),
            roi=2,
            nth_spike=3,
            crop_start_s=700.0,
            crop_end_s=900.0,
        ),
    ]

    cfg = MultiPanelConfig(
        examples=examples,
        fps=15.0,
        z_enter=3.5,
        z_exit=2.0,
        zoom_window_s=12.0,
        t_max=None,
        memmap_prefix="r0p7_",
        save_path=None,
    )

    fig, onset_counts = make_multipanel_figure(cfg)
    plt.show()

    for ex, n in zip(cfg.examples, onset_counts):
        print(
            f"{ex.root.parent.parent.name} | ROI {ex.roi} | "
            f"detected onsets = {n} | crop = [{ex.crop_start_s}, {ex.crop_end_s}] s"
        )