from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import utils


@dataclass
class Figure6Recording:
    root: Path
    label: str
    plot_seconds: Optional[float] = None


@dataclass
class Figure6Config:
    recordings: list[Figure6Recording]
    prefix: str = "r0p7_filtered_"
    time_cols_target: int = 1200
    z_enter: float = 3.5
    z_exit: float = 1.5
    min_separation_s: float = 0.1
    save_path: Optional[Path] = None


def _load_imaging_data(root: Path, prefix: str):
    dff, lowpass, derivative, num_frames, num_rois = utils.s2p_open_memmaps(str(root), prefix=prefix)
    return num_rois, num_frames, lowpass, derivative


def _process_roi(
    lowpass_roi: np.ndarray,
    derivative_roi: np.ndarray,
    fps: float,
    z_enter: float,
    z_exit: float,
    min_separation_s: float,
    downsample_factor: int,
    num_cols: int,
):
    z_scores, _, _ = utils.mad_z(np.asarray(derivative_roi, dtype=np.float32))
    onsets = utils.hysteresis_onsets(
        z_scores,
        z_enter,
        z_exit,
        fps,
        min_sep_s=min_separation_s
    )
    event_count = int(onsets.size)

    lowpass_roi = np.asarray(lowpass_roi, dtype=np.float32)

    if downsample_factor > 1:
        trimmed = lowpass_roi[:num_cols * downsample_factor].reshape(num_cols, downsample_factor)
        lowpass_downsampled = trimmed.mean(axis=1)

        event_raster_row = np.zeros(num_cols, dtype=np.uint8)
        if onsets.size:
            bins = (onsets // downsample_factor).clip(0, num_cols - 1)
            event_raster_row[np.unique(bins)] = 1
    else:
        lowpass_downsampled = lowpass_roi[:num_cols]
        event_raster_row = np.zeros(num_cols, dtype=np.uint8)
        valid_onsets = onsets[onsets < num_cols]
        event_raster_row[valid_onsets] = 1

    p1 = np.percentile(lowpass_downsampled, 1)
    p99 = np.percentile(lowpass_downsampled, 99)

    if p99 <= p1:
        heatmap_row = np.zeros_like(lowpass_downsampled, dtype=np.uint8)
    else:
        norm = np.clip((lowpass_downsampled - p1) / (p99 - p1), 0, 1)
        heatmap_row = (norm * 255.0 + 0.5).astype(np.uint8)

    return heatmap_row, event_raster_row, event_count


def _build_summary_for_recording(rec: Figure6Recording, cfg: Figure6Config):
    fps = utils.get_fps_from_notes(str(rec.root))
    num_rois, num_frames, lowpass, derivative = _load_imaging_data(rec.root, cfg.prefix)

    if rec.plot_seconds is not None:
        num_frames_cropped = min(num_frames, int(rec.plot_seconds * fps))
        time_slice = slice(0, num_frames_cropped)
    else:
        num_frames_cropped = num_frames
        time_slice = slice(None)

    downsample_factor = max(1, num_frames_cropped // cfg.time_cols_target)
    num_cols = num_frames_cropped // downsample_factor

    heatmap = np.zeros((num_rois, num_cols), dtype=np.uint8)
    event_raster = np.zeros((num_rois, num_cols), dtype=np.uint8)
    event_counts = np.zeros(num_rois, dtype=int)

    for roi_idx in range(num_rois):
        lowpass_roi = lowpass[time_slice, roi_idx]
        derivative_roi = derivative[time_slice, roi_idx]

        heat_row, raster_row, count = _process_roi(
            lowpass_roi=lowpass_roi,
            derivative_roi=derivative_roi,
            fps=fps,
            z_enter=cfg.z_enter,
            z_exit=cfg.z_exit,
            min_separation_s=cfg.min_separation_s,
            downsample_factor=downsample_factor,
            num_cols=num_cols,
        )

        heatmap[roi_idx, :] = heat_row
        event_raster[roi_idx, :] = raster_row
        event_counts[roi_idx] = count

    sort_order = np.argsort(-event_counts)

    return {
        "label": rec.label,
        "fps": fps,
        "num_rois": num_rois,
        "num_frames_cropped": num_frames_cropped,
        "num_cols": num_cols,
        "downsample_factor": downsample_factor,
        "duration_s": num_frames_cropped / fps,
        "heatmap_sorted": heatmap[sort_order],
        "event_raster_sorted": event_raster[sort_order],
    }


def make_figure6(cfg: Figure6Config):
    if len(cfg.recordings) != 3:
        raise ValueError("Figure 6 expects exactly 3 recordings.")

    summaries = [_build_summary_for_recording(rec, cfg) for rec in cfg.recordings]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(16, 14),
        squeeze=False
    )

    col_titles = ["Heatmap", "Event raster"]
    panel_labels = [["A", "B"], ["C", "D"], ["E", "F"]]

    for r, summary in enumerate(summaries):
        heat = summary["heatmap_sorted"]
        raster = summary["event_raster_sorted"]
        duration_s = summary["duration_s"]
        n_rois = summary["num_rois"]
        label = summary["label"]

        ax_heat = axes[r, 0]
        ax_raster = axes[r, 1]

        im = ax_heat.imshow(
            heat,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            extent=[0, duration_s, n_rois, 0],
        )
        ax_heat.set_title(f"{panel_labels[r][0]}  {label} heatmap", fontsize=12)
        ax_heat.set_ylabel("ROIs")
        ax_heat.set_xlabel("Time (s)")

        ax_raster.imshow(
            raster,
            aspect="auto",
            interpolation="nearest",
            cmap="Greys",
            extent=[0, duration_s, n_rois, 0],
        )
        ax_raster.set_title(f"{panel_labels[r][1]}  {label} event raster", fontsize=12)
        ax_raster.set_ylabel("ROIs")
        ax_raster.set_xlabel("Time (s)")

        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.set_label("Relative ΔF/F")

    fig.suptitle("Population activity heatmaps and event rasters across representative recordings", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if cfg.save_path is not None:
        cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {cfg.save_path}")

    return fig


if __name__ == "__main__":
    cfg = Figure6Config(
        recordings=[
            Figure6Recording(
                root=Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0"),
                label="Recording 1"
            ),
            Figure6Recording(
                root=Path(r"F:\data\2p_shifted\Hip\2024-07-01_00001\suite2p\plane0"),
                label="Recording 2"
            ),
            Figure6Recording(
                root=Path(r"F:\data\2p_shifted\Cx\2024-07-02_00012\suite2p\plane0"),
                label="Recording 3"
            ),
        ],
        prefix="r0p7_",
        time_cols_target=1200,
        z_enter=3.5,
        z_exit=1.5,
        min_separation_s=0.1,
        save_path=None,
    )

    fig = make_figure6(cfg)
    plt.show()