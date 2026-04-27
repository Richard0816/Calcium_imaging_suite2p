from dataclasses import dataclass
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import utils


@dataclass
class ImagingConfig:
    """Configuration parameters for imaging analysis."""
    prefix: str = 'r0p7_'  # matches memmaps written by analyze_output_gpu (all ROIs)
    cell_mask_name: str = 'predicted_cell_mask.npy'  # cellfilter output; None/missing => all ROIs
    plot_seconds: float = None  # None = full recording; or e.g., 300 for first 5 minutes
    time_cols_target: int = 1200  # target width (columns) for heatmaps; code downsamples time to ~this many bins

    # Event detection
    z_enter: float = 3.5  # robust z threshold for entering an “event”
    z_exit: float = 1.5 # lower threshold to exit (hysteresis)
    min_separation_s: float = 0.1  # merge onsets closer than this (sec) into a single event for counts

    # Small multiples
    top_k: int = 96  # number of ROIs to plot per page (line plots)
    grid_rows: int = 12  # 12x8 = 96 panels
    grid_cols: int = 8 # 12x8 = 96 panels


def _load_imaging_data(root: str, prefix: str):
    """Load Suite2p data and processed memmaps.

    Returns:
        tuple: (num_rois, num_frames, lowpass_memmap, derivative_memmap)
    """
    _, lowpass, derivative, num_frames, num_rois = (None, None, None, None, None)
    dff, lowpass, derivative, num_frames, num_rois = utils.s2p_open_memmaps(root, prefix=prefix)
    return num_rois, num_frames, lowpass, derivative


def _process_roi(roi_index: int, lowpass_data: np.ndarray, derivative_data: np.ndarray,
                 config: ImagingConfig, downsample_factor: int, num_cols: int, fps: float = 15.0):
    """Process a single ROI to generate heatmap row, event raster, and event count.

    Returns:
        tuple: (heatmap_row, event_raster_row, event_count)
    """
    lowpass_roi = np.asarray(lowpass_data, dtype=np.float32)
    derivative_roi = np.asarray(derivative_data, dtype=np.float32)

    # Detect events using hysteresis on derivative z-scores
    z_scores, median, mad = utils.mad_z(derivative_roi)
    onsets = utils.hysteresis_onsets(
        z_scores, config.z_enter, config.z_exit, fps, min_sep_s=config.min_separation_s
    )
    event_count = onsets.size

    # Downsample time dimension
    if downsample_factor > 1:
        trimmed = lowpass_roi[:num_cols * downsample_factor].reshape(num_cols, downsample_factor)
        lowpass_downsampled = trimmed.mean(axis=1)
        event_raster = np.zeros(num_cols, dtype=np.uint8)
        if onsets.size:
            bins = (onsets // downsample_factor).clip(0, num_cols - 1)
            event_raster[np.unique(bins)] = 1
    else:
        lowpass_downsampled = lowpass_roi
        event_raster = np.isin(np.arange(lowpass_roi.size), onsets).astype(np.uint8)

    # Robust scaling to 0-255 for heatmap visualization
    percentile_low = np.percentile(lowpass_downsampled, 1)
    percentile_high = np.percentile(lowpass_downsampled, 99)

    if percentile_high <= percentile_low:
        heatmap_row = np.zeros_like(lowpass_downsampled, dtype=np.uint8)
    else:
        normalized = np.clip(
            (lowpass_downsampled - percentile_low) / (percentile_high - percentile_low), 0, 1
        )
        heatmap_row = (normalized * 255.0 + 0.5).astype(np.uint8)

    return heatmap_row, event_raster, event_count


def _build_summaries(kept_indices: np.ndarray, lowpass: np.ndarray, derivative: np.ndarray,
                     time_slice: slice, config: ImagingConfig, downsample_factor: int, num_cols: int, fps: float = 15.0):
    """Build heatmap matrix, event raster, and event counts for ROIs in `kept_indices`.

    Returns:
        tuple: (heatmap, event_raster, event_counts, sorted_order)
            arrays indexed by position in `kept_indices` (len = n_kept).
    """
    n_kept = kept_indices.size
    heatmap = np.zeros((n_kept, num_cols), dtype=np.uint8)
    event_raster = np.zeros((n_kept, num_cols), dtype=np.uint8)
    event_counts = np.zeros(n_kept, dtype=int)

    for out_idx, roi_index in enumerate(kept_indices):
        lowpass_slice = lowpass[time_slice, roi_index]
        derivative_slice = derivative[time_slice, roi_index]

        heatmap_row, event_row, count = _process_roi(
            int(roi_index), lowpass_slice, derivative_slice,
            config, downsample_factor, num_cols, fps=fps
        )

        heatmap[out_idx, :] = heatmap_row
        event_raster[out_idx, :] = event_row
        event_counts[out_idx] = count

    sorted_order = np.argsort(-event_counts)

    return heatmap, event_raster, event_counts, sorted_order


def _save_heatmap(heatmap_sorted: np.ndarray, root: str, config: ImagingConfig,
                  num_rois: int, num_cols: int, downsample_factor: int, sample_name: str, fps: float = 15.0):
    """Generate and save the global heatmap visualization."""
    plt.figure(figsize=(14, 10))
    plt.imshow(heatmap_sorted, aspect='auto', interpolation='nearest')
    plt.title(
        f'Low-pass ΔF/F (sorted by event count)  N={num_rois}, '
        f'width~{num_cols} bins (~{num_cols * downsample_factor / fps:.1f}s), '
        f'sample ({sample_name})'
    )
    plt.xlabel('Time (downsampled bins)')
    plt.ylabel('ROIs (most active at top)')
    colorbar = plt.colorbar()
    colorbar.set_label('Relative intensity (robust 1–99% scaled)')
    plt.tight_layout()
    plt.savefig(os.path.join(root, f'{config.prefix}overview_heatmap.png'), dpi=200)
    plt.close()


def _save_event_raster(event_raster_sorted: np.ndarray, root: str, config: ImagingConfig,
                       num_rois: int, sample_name: str):
    """Generate and save the event raster visualization."""
    plt.figure(figsize=(14, 10))
    plt.imshow(event_raster_sorted, aspect='auto', interpolation='nearest', cmap='Greys')
    plt.title(
        f'Event raster (hysteresis z_enter={config.z_enter}, z_exit={config.z_exit})  '
        f'N={num_rois}, sample ({sample_name})'
    )
    plt.xlabel('Time (downsampled bins)')
    plt.ylabel('ROIs (most active at top)')
    plt.tight_layout()
    plt.savefig(os.path.join(root, f'{config.prefix}event_raster.png'), dpi=200)
    plt.close()


def _save_small_multiples(lowpass: np.ndarray, kept_indices: np.ndarray,
                          sorted_order: np.ndarray, event_counts: np.ndarray,
                          root: str, config: ImagingConfig, num_rois: int, num_frames_cropped: int, fps: float):
    """Generate and save small multiples line plots for top active ROIs."""
    time_axis = np.arange(num_frames_cropped) / fps
    max_pages = 5
    num_pages = math.ceil(min(num_rois, max_pages * config.top_k) / config.top_k)

    for page_num in range(num_pages):
        start_idx = page_num * config.top_k
        end_idx = min(num_rois, start_idx + config.top_k)
        if start_idx >= end_idx:
            break

        order_slice = sorted_order[start_idx:end_idx]
        fig, axes = plt.subplots(config.grid_rows, config.grid_cols, figsize=(16, 18), sharex=True)
        axes = np.array(axes).reshape(-1)

        for panel_idx, kept_pos in enumerate(order_slice):
            real_roi = int(kept_indices[kept_pos])
            ax = axes[panel_idx]
            trace = np.asarray(lowpass[:num_frames_cropped, real_roi], dtype=np.float32)

            y_low, y_high = np.percentile(trace, [1, 99])
            ax.plot(time_axis, trace, linewidth=0.8)
            ax.set_ylim(y_low, y_high if y_high > y_low else y_low + 1e-3)
            ax.set_title(
                f'ROI {real_roi} (#{start_idx + panel_idx + 1})  events={event_counts[kept_pos]}',
                fontsize=9
            )
            ax.grid(True, alpha=0.15)

        # Hide unused panels on last page
        for panel_idx in range(end_idx - start_idx, config.grid_rows * config.grid_cols):
            axes[panel_idx].axis('off')

        fig.suptitle(
            f'Low-pass ΔF/F small multiples — page {page_num + 1}/{num_pages} '
            f'(ROIs {start_idx}–{end_idx - 1})',
            fontsize=14
        )
        fig.text(0.5, 0.04, 'Time (s)', ha='center')
        fig.text(0.06, 0.5, 'ΔF/F (robust scale)', va='center', rotation='vertical')
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.97])

        output_path = os.path.join(root, f'{config.prefix}small_multiples_p{page_num + 1}.png')
        plt.savefig(output_path, dpi=160)
        plt.close()
        print("Saved", output_path)


def run_full_imaging_on_folder(folder_name: str):
    """Run complete imaging analysis pipeline on a Suite2p folder."""
    start_time = time.time()

    # Initialize configuration
    config = ImagingConfig()
    root = os.path.join(folder_name, "suite2p\\plane0\\") # Path to a single Suite2p plane folder
    sample_name = root.split("\\")[-4]  # Human-readable sample name from path
    print(f'Processing {sample_name}')
    fps = utils.get_fps_from_notes(root)
    # Load data
    num_rois, num_frames, lowpass, derivative = _load_imaging_data(root, config.prefix)

    # Apply predicted cell mask (cellfilter output); fallback to all ROIs if missing.
    mask_path = os.path.join(root, config.cell_mask_name) if config.cell_mask_name else None
    if mask_path and os.path.exists(mask_path):
        cell_mask = np.load(mask_path).astype(bool)
        if cell_mask.size != num_rois:
            raise ValueError(
                f"predicted_cell_mask size {cell_mask.size} != num_rois {num_rois} "
                f"(stale mask? rerun analyze_output_gpu.)"
            )
        kept_indices = np.flatnonzero(cell_mask).astype(np.int64)
        print(f"[mask] using {kept_indices.size}/{num_rois} cell-filter-positive ROIs")
    else:
        kept_indices = np.arange(num_rois, dtype=np.int64)
        print(f"[mask] {config.cell_mask_name} not found at {root}; using all {num_rois} ROIs")

    n_kept = kept_indices.size

    # Determine time cropping and downsampling
    if config.plot_seconds is not None:
        num_frames_cropped = min(num_frames, int(config.plot_seconds * fps))
        time_slice = slice(0, num_frames_cropped)
    else:
        num_frames_cropped = num_frames
        time_slice = slice(None)

    downsample_factor = max(1, num_frames_cropped // config.time_cols_target)
    num_cols = num_frames_cropped // downsample_factor

    # Build summaries (only for kept ROIs)
    heatmap, event_raster, event_counts, sorted_order = _build_summaries(
        kept_indices, lowpass, derivative, time_slice, config, downsample_factor, num_cols, fps
    )
    print("Summaries built: heatmap matrix =", heatmap.shape, "event raster =", event_raster.shape)

    heatmap_sorted = heatmap[sorted_order]
    event_raster_sorted = event_raster[sorted_order]

    _save_heatmap(heatmap_sorted, root, config, n_kept, num_cols, downsample_factor, sample_name, fps=fps)
    _save_event_raster(event_raster_sorted, root, config, n_kept, sample_name)
    _save_small_multiples(lowpass, kept_indices, sorted_order, event_counts, root, config, n_kept, num_frames_cropped, fps)

    print("Saved:",
          os.path.join(root, f'{config.prefix}overview_heatmap.png'),
          os.path.join(root, f'{config.prefix}event_raster.png'),
          "and small-multiples pages.")
    print(f"Completed in {time.time() - start_time} seconds.")


def run():
    utils.run_on_folders('D:\\data\\2p_shifted\\', run_full_imaging_on_folder)


# ================== RUN IT ==================
if __name__ == "__main__":
    run_full_imaging_on_folder(r'F:\data\2p_shifted\Hip\2024-06-04_00009')
    #utils.log("raster_and_heatmaps_plots.log", run_full_imaging_on_folder(r'F:\data\2p_shifted\2024-06-05_00007'))

