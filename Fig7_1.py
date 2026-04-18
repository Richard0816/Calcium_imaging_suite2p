
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import utils


@dataclass
class ClusterSpec:
    label: str
    roi_file: Path


@dataclass
class RecordingSpec:
    label: str
    recording_root: Path
    clusters: list[ClusterSpec]
    prefix: str = "r0p7_"


@dataclass
class Figure7Config:
    recordings: list[RecordingSpec]
    time_cols_target: int = 1200
    z_enter: float = 3.5
    z_exit: float = 1.5
    min_separation_s: float = 0.1
    plot_seconds: Optional[float] = None
    save_path: Optional[Path] = None


def _load_recording_data(root: Path, prefix: str):
    dff, lowpass, derivative, num_frames, num_rois = utils.s2p_open_memmaps(str(root), prefix=prefix)
    fps = utils.get_fps_from_notes(str(root))
    return dff, lowpass, derivative, num_frames, num_rois, fps


def _load_cluster_rois(roi_file: Path, max_roi: int):
    rois = np.load(roi_file)
    rois = np.asarray(rois).astype(int).ravel()
    rois = rois[(rois >= 0) & (rois < max_roi)]
    return rois


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
        min_sep_s=min_separation_s,
    )
    event_count = int(onsets.size)

    lowpass_roi = np.asarray(lowpass_roi, dtype=np.float32)

    if downsample_factor > 1:
        trimmed = lowpass_roi[: num_cols * downsample_factor].reshape(num_cols, downsample_factor)
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


def _build_cluster_summary(cluster_rois, lowpass, derivative, fps, cfg: Figure7Config, num_frames: int):
    if cfg.plot_seconds is not None:
        num_frames_cropped = min(num_frames, int(cfg.plot_seconds * fps))
        time_slice = slice(0, num_frames_cropped)
    else:
        num_frames_cropped = num_frames
        time_slice = slice(None)

    downsample_factor = max(1, num_frames_cropped // cfg.time_cols_target)
    num_cols = max(1, num_frames_cropped // downsample_factor)

    heatmap = np.zeros((len(cluster_rois), num_cols), dtype=np.uint8)
    raster = np.zeros((len(cluster_rois), num_cols), dtype=np.uint8)
    event_counts = np.zeros(len(cluster_rois), dtype=int)

    for i, roi_idx in enumerate(cluster_rois):
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
        heatmap[i, :] = heat_row
        raster[i, :] = raster_row
        event_counts[i] = count

    sort_order = np.argsort(-event_counts)

    return {
        "heatmap": heatmap[sort_order],
        "raster": raster[sort_order],
        "duration_s": num_frames_cropped / fps,
        "n_rois": len(cluster_rois),
    }


def make_figure7(cfg: Figure7Config):
    if len(cfg.recordings) != 2:
        raise ValueError("Figure 7 is set up for exactly two recordings, for example one hippocampal and one cortical recording.")

    row_specs = []
    for rec in cfg.recordings:
        _, lowpass, derivative, num_frames, num_rois, fps = _load_recording_data(rec.recording_root, rec.prefix)
        for cluster in rec.clusters:
            rois = _load_cluster_rois(cluster.roi_file, num_rois)
            summary = _build_cluster_summary(rois, lowpass, derivative, fps, cfg, num_frames)
            row_specs.append({
                "recording_label": rec.label,
                "cluster_label": cluster.label,
                "summary": summary,
            })

    n_rows = len(row_specs)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=2,
        figsize=(16, 3.0 * n_rows),
        squeeze=False,
    )

    panel_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    current_recording = None

    for r, row in enumerate(row_specs):
        summary = row["summary"]
        recording_label = row["recording_label"]
        cluster_label = row["cluster_label"]

        ax_heat = axes[r, 0]
        ax_raster = axes[r, 1]

        heat = summary["heatmap"]
        raster = summary["raster"]
        duration_s = summary["duration_s"]
        n_rois_cluster = summary["n_rois"]

        im = ax_heat.imshow(
            heat,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            extent=[0, duration_s, n_rois_cluster, 0],
        )
        ax_raster.imshow(
            raster,
            aspect="auto",
            interpolation="nearest",
            cmap="Greys",
            extent=[0, duration_s, n_rois_cluster, 0],
        )

        title_prefix_heat = panel_letters[2 * r]
        title_prefix_raster = panel_letters[2 * r + 1]
        title_text = f"{recording_label}, {cluster_label}"

        ax_heat.set_title(f"{title_prefix_heat}  {title_text} heatmap", fontsize=12)
        ax_raster.set_title(f"{title_prefix_raster}  {title_text} event raster", fontsize=12)

        ax_heat.set_ylabel("ROIs")
        ax_raster.set_ylabel("ROIs")
        ax_heat.set_xlabel("Time (s)")
        ax_raster.set_xlabel("Time (s)")

        if current_recording != recording_label:
            ax_heat.text(
                -0.16,
                1.08,
                recording_label,
                transform=ax_heat.transAxes,
                fontsize=14,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
            current_recording = recording_label

        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.set_label("Relative ΔF/F")

    fig.suptitle("Cluster segregated heatmaps and event rasters across two recordings", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if cfg.save_path is not None:
        cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {cfg.save_path}")

    return fig


if __name__ == "__main__":
    cfg = Figure7Config(
        recordings=[
            RecordingSpec(
                label="Patient 37",
                recording_root=Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0"),
                prefix="r0p7_filtered_",
                clusters=[
                    ClusterSpec(label="C1", roi_file=Path(
                        r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0\r0p7_filtered_cluster_results\C1_rois.npy")),
                    ClusterSpec(label="C2", roi_file=Path(
                        r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0\r0p7_filtered_cluster_results\C2_rois.npy")),
                    ClusterSpec(label="C3", roi_file=Path(
                        r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0\r0p7_filtered_cluster_results\C3_rois.npy")),
                    ClusterSpec(label="C4", roi_file=Path(
                        r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0\r0p7_filtered_cluster_results\C4_rois.npy")),
                ],
            ),
            RecordingSpec(
                label="Patient 35",
                recording_root=Path(r"F:\data\2p_shifted\Hip\2024-06-04_00009\suite2p\plane0"),
                prefix="r0p7_filtered_",
                clusters=[
                    ClusterSpec(label="C1", roi_file=Path(r"F:\data\2p_shifted\Hip\2024-06-04_00009\suite2p\plane0\r0p7_filtered_cluster_results\C1_rois.npy")),
                    ClusterSpec(label="C2", roi_file=Path(r"F:\data\2p_shifted\Hip\2024-06-04_00009\suite2p\plane0\r0p7_filtered_cluster_results\C2_rois.npy")),
                    ClusterSpec(label="C3", roi_file=Path(r"F:\data\2p_shifted\Hip\2024-06-04_00009\suite2p\plane0\r0p7_filtered_cluster_results\C3_rois.npy")),
                    ClusterSpec(label="C4", roi_file=Path(r"F:\data\2p_shifted\Hip\2024-06-04_00009\suite2p\plane0\r0p7_filtered_cluster_results\C4_rois.npy")),
                ],
            ),
        ],
        time_cols_target=1200,
        z_enter=3.5,
        z_exit=1.5,
        min_separation_s=0.1,
        plot_seconds=None,
        save_path=None,
    )

    fig = make_figure7(cfg)
    plt.show()
