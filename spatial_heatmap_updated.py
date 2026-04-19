"""
Spatial heatmap generation from Suite2p processed data.

Public API (two functions):

    view_roi_features(folder_name, cell_mask_path=None, ...)
        Paint per-ROI scalar features (event_rate, mean_dff, peak_dz) as
        spatial maps. One figure per feature, with and without the cell mask.

    coactivation_maps(folder_name, cell_mask_path=None,
                      propagation_vectors=False, ...)
        Detect population events from the flattened onset density and save
        per-event activation-order maps. If propagation_vectors=True, also
        overlays a start->end arrow on each order map and writes a CSV of
        propagation vectors (speed, angle, dt per event).

Cell filtering is no longer handled here. Provide a pre-computed cell mask
via `cell_mask_path` (e.g. r0p7_cell_mask_bool.npy, predicted_cell_mask.npy).
If None, all ROIs are used.
"""

import csv
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, Union

import utils


# ---- Colormap for coactivation order maps ----
CYAN_TO_RED = mpl.colors.LinearSegmentedColormap.from_list(
    "cyan_to_red", ["#00FFFF", "#0000FF", "#FF0000"], N=256
).copy()
CYAN_TO_RED.set_bad(color="#808080")


# =============================================================================
# PUBLIC API
# =============================================================================

def view_roi_features(
    folder_name: str,
    cell_mask_path: Optional[str] = None,
    prefix: str = "r0p7_",
    fps: Optional[float] = None,
    z_enter: float = 3.5,
    z_exit: float = 1.5,
    min_sep_s: float = 0.3,
    bin_seconds: Optional[float] = None,
    cmap: str = "magma",
    metrics: tuple = ("event_rate", "mean_dff", "peak_dz"),
):
    """
    Save spatial heatmaps of per-ROI scalar features.

    For each metric in `metrics`, saves two images:
      * "<metric>.png"               -- all ROIs painted
      * "<metric>_masked.png"        -- only ROIs where cell_mask is True
                                        (omitted if cell_mask_path is None)

    If `bin_seconds` is set, additional time-binned maps are saved.

    Parameters
    ----------
    folder_name : str
        Recording folder (the one containing suite2p/plane0).
    cell_mask_path : str, optional
        Path to a .npy boolean mask (shape = n_rois). Passed through to the
        masked-map variants. Common choices: 'r0p7_cell_mask_bool.npy',
        'predicted_cell_mask.npy'. If None, masked variants are skipped.
    prefix : str
        Memmap filename prefix (must match preprocessing).
    fps : float, optional
        Frame rate. If None, read from notes via utils.get_fps_from_notes.
    z_enter, z_exit, min_sep_s : float
        Onset-detection parameters (used when computing event_rate / peak_dz).
    bin_seconds : float, optional
        If set, also save maps for each time bin of this length (seconds).
    cmap : str
        Matplotlib colormap for scalar feature maps.
    metrics : tuple
        Which metrics to paint. Subset of ('event_rate', 'mean_dff', 'peak_dz').
    """
    if fps is None:
        fps = utils.get_fps_from_notes(folder_name)

    config = _Config(folder_name, prefix, fps, z_enter, z_exit, min_sep_s)
    data = _load_suite2p_data(config)
    cell_mask = _load_cell_mask(cell_mask_path, n_rois=data["N"])

    for metric in metrics:
        _save_feature_map(data, config, metric=metric, cell_mask=cell_mask,
                          t_slice=None, bin_index=None, cmap=cmap)

        if bin_seconds is not None and bin_seconds > 0:
            _save_time_binned_feature_maps(
                data, config, metric=metric, cell_mask=cell_mask,
                bin_seconds=bin_seconds, cmap=cmap,
            )


def coactivation_maps(
    folder_name: str,
    cell_mask_path: Optional[str] = None,
    propagation_vectors: bool = False,
    prefix: str = "r0p7_",
    fps: Optional[float] = None,
    z_enter: float = 3.5,
    z_exit: float = 1.5,
    min_sep_s: float = 0.3,
    # event-detection overrides (forwarded to utils.EventDetectionParams)
    bin_sec: float = 0.05,
    smooth_sigma_bins: float = 2.0,
    min_prominence: float = 0.007,
    min_width_bins: float = 2.0,
    min_distance_bins: float = 3.0,
    # violin plot
    violin_min_events: int = 5,
    violin_top_n: int = 60,
    cmap=CYAN_TO_RED,
):
    """
    Detect population events and save per-event activation-order heatmaps.

    For each detected event:
      * paints ROI-wise activation order (earliest=cyan, latest=red)
      * saves to <root>/<prefix>coact_order_bin_cells/<prefix>coact_order_bin####_cells.png

    If propagation_vectors=True, additionally:
      * overlays a start->end propagation arrow on each order map
      * saves to <root>/<prefix>coact_propagation_vectors/
      * writes a CSV summary of per-event speed/angle/dt/distance

    Always writes:
      * <root>/<prefix>coactivation_summary.csv  (per-ROI timing per event)
      * <root>/<prefix>coactivation_roi_relative_lag_violin.png

    Parameters
    ----------
    folder_name : str
        Recording folder containing suite2p/plane0.
    cell_mask_path : str, optional
        Path to a .npy boolean mask (shape = n_rois). If None, all ROIs are used.
    propagation_vectors : bool
        If True, also compute and save per-event propagation arrows + CSV.
    prefix : str
        Memmap filename prefix.
    fps : float, optional
        Frame rate. If None, read from notes.
    z_enter, z_exit, min_sep_s : float
        Per-ROI onset-detection thresholds.
    bin_sec, smooth_sigma_bins, min_prominence, min_width_bins, min_distance_bins :
        Event-detection parameters, forwarded to utils.EventDetectionParams.
    violin_min_events : int
        ROIs with fewer than this many event participations are omitted from
        the lag-distribution violin plot.
    violin_top_n : int
        Cap on ROIs shown in the violin plot (sorted by median lag, earliest
        first).
    cmap : matplotlib colormap
        Used for order-rank visualisation.
    """
    if fps is None:
        fps = utils.get_fps_from_notes(folder_name)

    config = _Config(folder_name, prefix, fps, z_enter, z_exit, min_sep_s)
    data = _load_suite2p_data(config)

    cell_mask = _load_cell_mask(cell_mask_path, n_rois=data["N"])
    if cell_mask is None:
        cell_mask = np.ones(data["N"], dtype=bool)
    idx_keep = np.where(cell_mask)[0]
    stat_filtered = [data["stat"][i] for i in idx_keep]

    print(f"[CoAct] Using {cell_mask.sum()} / {cell_mask.size} ROIs "
          f"({'all' if cell_mask_path is None else cell_mask_path})")

    # FOV in microns (for propagation vector arrow overlay)
    zoom = utils.get_zoom_from_notes(folder_name)
    zoom = float(zoom) if zoom else 1.0
    fov_um_x = 3080.90169 / zoom
    fov_um_y = 3560.14057 / zoom

    # 1. Per-ROI onset times (restricted to cell_mask)
    onsets_all = _event_onsets_by_roi(data, config, t_slice=None)
    onsets = [onsets_all[i] for i in idx_keep]

    # 2. Population events via density-based detection
    ev_params = utils.EventDetectionParams(
        bin_sec=bin_sec,
        smooth_sigma_bins=smooth_sigma_bins,
        min_prominence=min_prominence,
        min_width_bins=min_width_bins,
        min_distance_bins=min_distance_bins,
    )
    event_windows, A, first_time = utils.detect_event_windows(
        onsets_by_roi=onsets, T=data["T"], fps=config.fps,
        params=ev_params, return_diagnostics=False,
    )
    n_events = event_windows.shape[0]
    if n_events == 0:
        print("[CoAct] No events detected; nothing to save.")
        return

    print(f"[CoAct] Detected {n_events} population events.")

    # 3. Per-ROI timing CSV
    _write_coactivation_summary_csv(
        config=config,
        event_windows=event_windows,
        A=A, first_time=first_time,
        idx_keep=idx_keep,
    )

    # 4. Per-event spatial order maps (+ optional propagation arrows)
    Ly, Lx = data["Ly"], data["Lx"]

    prop_rows = []
    order_dir = os.path.join(config.root, f"{prefix}coact_order_bin_cells")
    os.makedirs(order_dir, exist_ok=True)

    prop_dir = None
    if propagation_vectors:
        prop_dir = os.path.join(config.root, f"{prefix}coact_propagation_vectors")
        os.makedirs(prop_dir, exist_ok=True)

    for ev_idx in range(n_events):
        order_rank = _order_map_for_event(first_time[:, ev_idx], A[:, ev_idx])
        spatial_order = _paint_order_map(order_rank, stat_filtered, Ly, Lx)

        t0 = float(event_windows[ev_idx, 0])
        t1 = float(event_windows[ev_idx, 1])
        n_active = int(A[:, ev_idx].sum())
        frac = n_active / A.shape[0] if A.shape[0] > 0 else 0.0

        base_title = (
            f"Activation order (event {ev_idx}: {t0:.2f}–{t1:.2f}s)\n"
            f"active={n_active}/{A.shape[0]} ({100 * frac:.1f}%)"
        )

        # Always save the plain order map
        order_out = os.path.join(order_dir, f"{prefix}coact_order_bin{ev_idx:04d}_cells.png")
        _show_spatial(spatial_order, base_title, Lx, Ly, stat_filtered,
                      pix_to_um=data["pix_to_um"], cmap=cmap, outpath=order_out)

        # Optionally save the arrow-overlay variant + record for CSV
        if propagation_vectors:
            prop = _compute_propagation_vector_for_event(
                first_time[:, ev_idx], A[:, ev_idx], stat_filtered, fps=config.fps,
            )
            if prop is None:
                continue

            start_px, end_px, dt_sec, n_first, n_last = prop

            # pixel -> um (anisotropic)
            um_per_px_x = float(fov_um_x) / float(Lx)
            um_per_px_y = float(fov_um_y) / float(Ly)
            start_um = np.array([start_px[0] * um_per_px_x, start_px[1] * um_per_px_y])
            end_um = np.array([end_px[0] * um_per_px_x, end_px[1] * um_per_px_y])

            # Arrow points from earliest-firing centroid (blue) to latest-firing
            # centroid (red) -- the direction of propagation through time.
            vec_um = end_um - start_um

            dist_um = float(np.hypot(vec_um[0], vec_um[1]))
            speed = dist_um / dt_sec if (dt_sec is not None and dt_sec > 0) else 0.0
            angle_deg = (float(np.degrees(np.arctan2(vec_um[1], vec_um[0])))
                         if dist_um > 0 else 0.0)

            arrow_title = (
                f"Propagation vector (event {ev_idx}: {t0:.2f}–{t1:.2f}s)\n"
                f"speed={speed:.1f} µm/s, angle={angle_deg:.1f}°, dt={dt_sec:.3f}s "
                f"(firstROIs={n_first}, lastROIs={n_last})"
            )
            arrow_out = os.path.join(
                prop_dir, f"{prefix}coact_propagation_bin{ev_idx:04d}.png"
            )
            _show_spatial_with_arrow_um(
                img=spatial_order, title=arrow_title,
                fov_um_x=fov_um_x, fov_um_y=fov_um_y,
                arrow_start_um=start_um, arrow_vec_um=vec_um,
                stat_filtered=stat_filtered, outpath=arrow_out, cmap=cmap,
            )

            prop_rows.append({
                "event_index": int(ev_idx),
                "event_start_s": t0,
                "event_end_s": t1,
                "zoom": float(zoom),
                "fov_um_x": float(fov_um_x),
                "fov_um_y": float(fov_um_y),
                "start_x_um": float(start_um[0]),
                "start_y_um": float(start_um[1]),
                "end_x_um": float(end_um[0]),
                "end_y_um": float(end_um[1]),
                "dx_um": float(vec_um[0]),
                "dy_um": float(vec_um[1]),
                "dt_s": float(dt_sec),
                "distance_um": float(dist_um),
                "speed_um_per_s": float(speed),
                "angle_deg": float(angle_deg),
                "n_first_frame_rois": int(n_first),
                "n_last_frame_rois": int(n_last),
            })

    print(f"[CoAct] Saved {n_events} order maps → {order_dir}")

    # 5. Propagation CSV (if propagation_vectors requested and any rows)
    if propagation_vectors:
        prop_csv = os.path.join(config.root, f"{prefix}coactivation_propagation.csv")
        if prop_rows:
            with open(prop_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(prop_rows[0].keys()))
                w.writeheader()
                w.writerows(prop_rows)
            print(f"[CoAct] Saved propagation CSV → {prop_csv}")
        else:
            print("[CoAct] No valid propagation vectors computed.")

    # 6. Per-ROI relative-lag violin plot
    _save_relative_lag_violin(
        config=config,
        event_windows=event_windows,
        A=A, first_time=first_time,
        idx_keep=idx_keep,
        min_events=violin_min_events,
        top_n=violin_top_n,
    )


# =============================================================================
# Internals: config + data loading + cell mask
# =============================================================================

class _Config:
    """Minimal config holder (folder + onset detection parameters)."""

    def __init__(self, folder_name, prefix, fps, z_enter, z_exit, min_sep_s):
        self.folder_name = folder_name
        self.prefix = prefix
        self.fps = fps
        self.z_enter = z_enter
        self.z_exit = z_exit
        self.min_sep_s = min_sep_s
        self.root = os.path.join(folder_name, "suite2p", "plane0") + os.sep
        self.sample_name = os.path.basename(folder_name.rstrip("\\/"))


def _load_suite2p_data(config):
    """Load Suite2p metadata + dF/F memmaps."""
    ops = np.load(os.path.join(config.root, "ops.npy"), allow_pickle=True).item()
    stat = np.load(os.path.join(config.root, "stat.npy"), allow_pickle=True)

    Ly, Lx = ops["Ly"], ops["Lx"]
    pix_to_um = ops.get("pix_to_um", None)

    low = np.memmap(
        os.path.join(config.root, f"{config.prefix}dff_lowpass.memmap.float32"),
        dtype="float32", mode="r",
    )
    dt = np.memmap(
        os.path.join(config.root, f"{config.prefix}dff_dt.memmap.float32"),
        dtype="float32", mode="r",
    )

    N = len(stat)
    T = low.size // N
    low = low.reshape(T, N)
    dt = dt.reshape(T, N)

    return {
        "stat": stat,
        "Ly": Ly, "Lx": Lx,
        "pix_to_um": pix_to_um,
        "low": low, "dt": dt,
        "T": T, "N": N,
    }


def _load_cell_mask(cell_mask_path: Optional[str], n_rois: int) -> Optional[np.ndarray]:
    """
    Load a cell mask from a .npy file. Returns a boolean array of length n_rois,
    or None if cell_mask_path is None.

    Raises if the file is missing or has the wrong shape.
    """
    if cell_mask_path is None:
        return None
    if not os.path.exists(cell_mask_path):
        raise FileNotFoundError(f"Cell mask not found: {cell_mask_path}")
    mask = np.load(cell_mask_path)
    if mask.ndim != 1 or mask.size != n_rois:
        raise ValueError(
            f"Cell mask shape mismatch: file has {mask.shape}, expected ({n_rois},). "
            f"Path: {cell_mask_path}"
        )
    return mask.astype(bool)


# =============================================================================
# Internals: feature-map painting
# =============================================================================

def _save_feature_map(data, config, metric, cell_mask, t_slice=None,
                      bin_index=None, cmap="magma"):
    """Compute a scalar feature per ROI, paint it, save to disk."""
    signals = {"low": data["low"], "dt": data["dt"]}
    time_slice = t_slice if t_slice is not None else slice(None)

    vals = utils.roi_metric(
        signals, which=metric, t_slice=time_slice,
        fps=config.fps, z_enter=config.z_enter,
        z_exit=config.z_exit, min_sep_s=config.min_sep_s,
    )
    spatial = utils.paint_spatial(vals, data["stat"], data["Ly"], data["Lx"])

    out_dir = os.path.join(config.root, f"{config.prefix}spatial_{metric}")
    os.makedirs(out_dir, exist_ok=True)

    base_title = _metric_title(metric, config)
    if bin_index is None:
        out_path = os.path.join(out_dir, f"{config.prefix}spatial_{metric}")
        title = base_title
    else:
        t0, t1 = time_slice.start, time_slice.stop
        out_path = os.path.join(
            out_dir, f"{config.prefix}spatial_{metric}_bin{bin_index:03d}"
        )
        title = (f"{base_title}\n"
                 f"Window {bin_index}: {t0 / config.fps:.1f}–{t1 / config.fps:.1f} s")

    # All-ROI map
    _show_spatial(spatial, title, data["Lx"], data["Ly"], data["stat"],
                  pix_to_um=data["pix_to_um"], cmap=cmap, outpath=out_path + ".png")

    # Masked-only map
    if cell_mask is not None:
        idx_keep = np.where(cell_mask)[0]
        if idx_keep.size > 0:
            vals_filtered = vals[idx_keep]
            stat_filtered = [data["stat"][i] for i in idx_keep]
            spatial_filtered = utils.paint_spatial(
                vals_filtered, stat_filtered, data["Ly"], data["Lx"]
            )
            _show_spatial(
                spatial_filtered, title + " (cell mask)",
                data["Lx"], data["Ly"], stat_filtered,
                pix_to_um=data["pix_to_um"], cmap=cmap,
                outpath=out_path + "_masked.png",
            )
        else:
            print(f"[view_roi_features] Cell mask empty; skipping masked map for {metric}.")


def _save_time_binned_feature_maps(data, config, metric, cell_mask, bin_seconds, cmap):
    """Save per-bin feature maps."""
    T = data["T"]
    Tbin = int(bin_seconds * config.fps)
    n_bins = int(np.ceil(T / Tbin))

    for b in range(n_bins):
        t0 = b * Tbin
        t1 = min(T, (b + 1) * Tbin)
        if t1 - t0 < max(5, int(0.2 * Tbin)):
            continue
        _save_feature_map(
            data, config, metric=metric, cell_mask=cell_mask,
            t_slice=slice(t0, t1), bin_index=b + 1, cmap=cmap,
        )


def _metric_title(metric, config):
    titles = {
        "event_rate": (
            f"Event rate (events/min) — z_enter={config.z_enter}, "
            f"z_exit={config.z_exit} ({config.sample_name})"
        ),
        "mean_dff": f"Mean ΔF/F (low-pass) ({config.sample_name})",
        "peak_dz": f"Peak derivative z (robust) ({config.sample_name})",
    }
    return titles.get(metric, f"{metric} ({config.sample_name})")


# =============================================================================
# Internals: onset extraction (per-ROI) + event ordering
# =============================================================================

def _event_onsets_by_roi(data, config, t_slice=None):
    """Per-ROI onset times (seconds) from MAD-z + hysteresis on dt."""
    T = data["T"]
    fps = config.fps
    dt = data["dt"]

    if t_slice is None:
        t0, t1 = 0, T
    else:
        t0 = 0 if t_slice.start is None else int(t_slice.start)
        t1 = T if t_slice.stop is None else int(t_slice.stop)

    onsets_sec = []
    for i in range(dt.shape[1]):
        x = dt[t0:t1, i]
        z, _, _ = utils.mad_z(x)
        idxs = utils.hysteresis_onsets(z, config.z_enter, config.z_exit, fps)
        onsets_sec.append(np.asarray(idxs, dtype=np.int64) / fps + (t0 / fps))
    return onsets_sec


def _order_map_for_event(first_time_col, active_mask_col):
    """Rank 1..K for active ROIs in this event by first onset time. NaN for inactives."""
    order_rank = np.full_like(first_time_col, np.nan, dtype=float)
    sel = active_mask_col & ~np.isnan(first_time_col)
    if not np.any(sel):
        return order_rank
    times = first_time_col[sel]
    idx_sorted = np.argsort(times, kind="mergesort")
    ranks = np.empty_like(idx_sorted, dtype=float)
    ranks[idx_sorted] = np.arange(1, idx_sorted.size + 1, dtype=float)
    order_rank[np.where(sel)[0]] = ranks
    return order_rank


def _paint_order_map(order_rank, stat, Ly, Lx):
    """
    Paint ranks onto the FOV, normalised 0..1 so that earliest=0 and latest=1.
    With the CYAN_TO_RED colormap (cyan -> blue -> red), this gives
    earliest=cyan/blue, latest=red.
    """
    vals = order_rank.astype(float).copy()
    if np.all(np.isnan(vals)):
        img = utils.paint_spatial(np.full_like(order_rank, np.nan, dtype=float), stat, Ly, Lx)
        coverage = utils.paint_spatial(np.ones(len(stat), dtype=float), stat, Ly, Lx)
        img[coverage == 0] = np.nan
        return img

    # order_rank is 1..K with 1 = earliest. Normalise so earliest -> 0, latest -> 1.
    maxr = np.nanmax(vals)
    if maxr > 1:
        vals = (vals - 1.0) / (maxr - 1.0)
    else:
        # single active ROI: map to 0 (treat as earliest)
        vals = np.where(np.isnan(vals), np.nan, 0.0)

    img = utils.paint_spatial(vals, stat, Ly, Lx)
    coverage = utils.paint_spatial(np.ones(len(stat), dtype=float), stat, Ly, Lx)
    img[coverage == 0] = np.nan
    return img


# =============================================================================
# Internals: CSV outputs + violin plot
# =============================================================================

def _write_coactivation_summary_csv(config, event_windows, A, first_time, idx_keep):
    """Per-ROI timing row per event: roi_index, event times, onset time, relative lag, rank."""
    summary_csv = os.path.join(config.root, f"{config.prefix}coactivation_summary.csv")
    n_events = event_windows.shape[0]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "roi_index_filtered", "roi_index_original",
            "event_index", "event_start_s", "event_end_s",
            "first_onset_s", "relative_lag_s", "order_rank",
        ])
        writer.writeheader()

        for ev_idx in range(n_events):
            order_rank = _order_map_for_event(first_time[:, ev_idx], A[:, ev_idx])
            active_times = first_time[A[:, ev_idx] & ~np.isnan(first_time[:, ev_idx]), ev_idx]
            if active_times.size == 0:
                continue
            earliest = float(np.min(active_times))
            t0 = float(event_windows[ev_idx, 0])
            t1 = float(event_windows[ev_idx, 1])

            for roi_local_idx in np.where(A[:, ev_idx])[0]:
                ft = first_time[roi_local_idx, ev_idx]
                if np.isnan(ft):
                    continue
                writer.writerow({
                    "roi_index_filtered": int(roi_local_idx),
                    "roi_index_original": int(idx_keep[roi_local_idx]),
                    "event_index": int(ev_idx),
                    "event_start_s": t0,
                    "event_end_s": t1,
                    "first_onset_s": float(ft),
                    "relative_lag_s": float(ft) - earliest,
                    "order_rank": float(order_rank[roi_local_idx]),
                })

    print(f"[CoAct] Saved coactivation summary CSV → {summary_csv}")


def _save_relative_lag_violin(config, event_windows, A, first_time, idx_keep,
                              min_events, top_n):
    """Per-ROI relative-lag distributions as violin plot, sorted by median lag."""
    n_rois = A.shape[0]
    roi_rel_lags = [[] for _ in range(n_rois)]

    for ev_idx in range(event_windows.shape[0]):
        sel = A[:, ev_idx] & ~np.isnan(first_time[:, ev_idx])
        if not np.any(sel):
            continue
        earliest = float(np.min(first_time[sel, ev_idx]))
        for i in np.where(sel)[0]:
            roi_rel_lags[i].append(float(first_time[i, ev_idx]) - earliest)

    valid = [(i, np.asarray(lags, float))
             for i, lags in enumerate(roi_rel_lags) if len(lags) >= min_events]
    if len(valid) == 0:
        print("[CoAct] No ROIs have enough events for violin plot.")
        return

    valid.sort(key=lambda t: np.median(t[1]))
    valid = valid[:top_n]

    roi_idx = [i for i, _ in valid]
    data_lags = [lags for _, lags in valid]
    medians = np.array([np.median(l) for l in data_lags])
    labels = [str(int(idx_keep[i])) for i in roi_idx]

    plt.figure(figsize=(max(12, 0.22 * len(data_lags)), 6))
    plt.violinplot(data_lags, showmeans=False, showmedians=True,
                   showextrema=False, widths=0.9)
    x = np.arange(1, len(data_lags) + 1)
    plt.scatter(x, medians, s=10)
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Relative lag (s) = ROI first onset − earliest onset in event")
    plt.xlabel("ROI (original index), sorted by median relative lag")
    plt.title(f"Per-ROI relative lag distributions "
              f"(top {len(data_lags)} ROIs, min_events={min_events})")
    # upper limit from longest event window
    max_dur = float(np.max(event_windows[:, 1] - event_windows[:, 0]))
    plt.ylim(0, max_dur)
    plt.tight_layout()

    out_png = os.path.join(
        config.root, f"{config.prefix}coactivation_roi_relative_lag_violin.png"
    )
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[CoAct] Saved violin plot → {out_png}")


# =============================================================================
# Internals: propagation vector per event
# =============================================================================

def _roi_centroids_xy(stat_list):
    xs = np.array([float(np.median(s["xpix"])) for s in stat_list], dtype=float)
    ys = np.array([float(np.median(s["ypix"])) for s in stat_list], dtype=float)
    return xs, ys


def _compute_propagation_vector_for_event(first_time_col, active_mask_col,
                                          stat_filtered, fps):
    """
    Start = mean centroid of ROIs active on the earliest onset frame.
    End   = mean centroid of ROIs active on the latest onset frame.
    Returns (start_px, end_px, dt_sec, n_first, n_last) or None.
    """
    sel = active_mask_col & ~np.isnan(first_time_col)
    if not np.any(sel):
        return None

    times = first_time_col[sel].astype(float)
    roi_idx = np.where(sel)[0]

    t_first = float(np.min(times))
    t_last = float(np.max(times))

    tol = 0.5 / float(fps)
    first_rois = roi_idx[np.abs(first_time_col[roi_idx] - t_first) <= tol]
    last_rois = roi_idx[np.abs(first_time_col[roi_idx] - t_last) <= tol]

    if first_rois.size == 0:
        first_rois = np.array([roi_idx[np.argmin(first_time_col[roi_idx])]], dtype=int)
    if last_rois.size == 0:
        last_rois = np.array([roi_idx[np.argmax(first_time_col[roi_idx])]], dtype=int)

    xs, ys = _roi_centroids_xy(stat_filtered)
    start = np.array([xs[first_rois].mean(), ys[first_rois].mean()], dtype=float)
    end = np.array([xs[last_rois].mean(), ys[last_rois].mean()], dtype=float)

    return start, end, (t_last - t_first), int(first_rois.size), int(last_rois.size)


# =============================================================================
# Internals: plotting
# =============================================================================

def _show_spatial(img, title, Lx, Ly, stat, pix_to_um=None, cmap="magma", outpath=None):
    """Display/save a scalar spatial map with optional um axes."""
    extent = None
    xlabel, ylabel = "X (pixels)", "Y (pixels)"
    if pix_to_um is not None:
        extent = [0, Lx * pix_to_um, 0, Ly * pix_to_um]
        xlabel, ylabel = "X (µm)", "Y (µm)"

    plt.figure(figsize=(8, 7))
    im = plt.imshow(img, origin="lower", cmap=cmap, extent=extent, aspect="equal")
    plt.colorbar(im, label=title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
        print("Saved", outpath)
    else:
        plt.show()


def _add_scale_bar_um(ax, fov_um_x, fov_um_y, bar_um=200.0, pad_frac=0.05,
                      lw=3.5, color="white", fontsize=10):
    """Horizontal scale bar in µm coordinates."""
    x_start = fov_um_x * (1.0 - pad_frac) - bar_um
    x_end = x_start + bar_um
    y_bar = fov_um_y * pad_frac
    y_text = y_bar + fov_um_y * 0.03
    ax.plot([x_start, x_end], [y_bar, y_bar], color=color, lw=lw, solid_capstyle="butt")
    ax.text((x_start + x_end) / 2.0, y_text, f"{int(bar_um)} µm",
            color=color, fontsize=fontsize, ha="center", va="bottom")


def _show_spatial_with_arrow_um(img, title, fov_um_x, fov_um_y, arrow_start_um,
                                arrow_vec_um, stat_filtered, outpath, cmap=CYAN_TO_RED):
    """Spatial image with propagation arrow and scale bar."""
    extent = [0, float(fov_um_x), 0, float(fov_um_y)]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(img, origin="lower", cmap=cmap, extent=extent, aspect="equal")

    # ROI centroid dots
    xs_px, ys_px = _roi_centroids_xy(stat_filtered)
    Ly, Lx = img.shape[0], img.shape[1]
    um_per_px_x = float(fov_um_x) / float(Lx)
    um_per_px_y = float(fov_um_y) / float(Ly)
    ax.scatter(xs_px * um_per_px_x, ys_px * um_per_px_y,
               s=4, c="white", alpha=0.35, linewidths=0)

    sx, sy = float(arrow_start_um[0]), float(arrow_start_um[1])
    vx, vy = float(arrow_vec_um[0]), float(arrow_vec_um[1])
    if np.isfinite(vx) and np.isfinite(vy) and (abs(vx) + abs(vy)) > 1e-6:
        ax.arrow(
            sx, sy, vx, vy,
            length_includes_head=True,
            head_width=0.03 * max(fov_um_x, fov_um_y),
            head_length=0.04 * max(fov_um_x, fov_um_y),
            linewidth=2.0, color="white",
        )

    _add_scale_bar_um(ax, fov_um_x=fov_um_x, fov_um_y=fov_um_y,
                      bar_um=200.0, pad_frac=0.05, lw=4.0, color="white", fontsize=10)

    plt.colorbar(im, ax=ax, label=title)
    ax.set_title(title)
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print("Saved", outpath)


# =============================================================================
# ARCHIVED: legacy cell-scoring pipeline (no longer used)
# =============================================================================
"""
The code below is retained for reference only. Cell filtering has moved out
of this file -- pass a pre-computed cell mask path (e.g.
r0p7_cell_mask_bool.npy, predicted_cell_mask.npy) to view_roi_features /
coactivation_maps instead.

Archived:
    - edge_mask_from_stat       -- edge-buffer mask from ROI pixel bounds
    - _safe_div                 -- zero-safe division
    - compute_cell_scores       -- logistic scoring on event_rate + peak_dz + area
    - soft_cell_mask            -- threshold / top-k selection with fallbacks
    - plot_leadlag_split_spatial_from_csv  -- lead/lag split map from old CSV

\"\"\"
def edge_mask_from_stat(stat, Lx, Ly, edge_buffer_px=10, rule='centroid'):
    if rule == 'centroid':
        xs = np.array([np.mean(s['xpix']) for s in stat], dtype=float)
        ys = np.array([np.mean(s['ypix']) for s in stat], dtype=float)
        inside = (
            (xs > edge_buffer_px) & (xs < (Lx - edge_buffer_px)) &
            (ys > edge_buffer_px) & (ys < (Ly - edge_buffer_px))
        )
    elif rule == 'bbox':
        xmins = np.array([s['xpix'].min() for s in stat])
        xmaxs = np.array([s['xpix'].max() for s in stat])
        ymins = np.array([s['ypix'].min() for s in stat])
        ymaxs = np.array([s['ypix'].max() for s in stat])
        inside = (
            (xmins > edge_buffer_px) & (xmaxs < (Lx - edge_buffer_px)) &
            (ymins > edge_buffer_px) & (ymaxs < (Ly - edge_buffer_px))
        )
    else:
        raise ValueError(\"rule must be 'centroid' or 'bbox'\")
    return inside.astype(bool)


def _safe_div(x, d):
    d = float(d) if d else 1.0
    return x / d


def compute_cell_scores(data, config,
                        w_er=1.0, w_pz=1.0, w_area=0.5,
                        scale_er=1.0, scale_pz=3.0, scale_area=50.0,
                        bias=-2.0, t_slice=None,
                        edge_buffer_px=6, edge_rule='centroid', save_masks=True):
    stat = data['stat']
    Lx, Ly = data['Lx'], data['Ly']
    signals = {'low': data['low'], 'dt': data['dt']}
    time_slice = t_slice if t_slice is not None else slice(None)

    event_rate = utils.roi_metric(signals, which='event_rate', t_slice=time_slice,
                                  fps=config.fps, z_enter=config.z_enter,
                                  z_exit=config.z_exit, min_sep_s=config.min_sep_s)
    peak_dz = utils.roi_metric(signals, which='peak_dz', t_slice=time_slice,
                               fps=config.fps, z_enter=config.z_enter,
                               z_exit=config.z_exit, min_sep_s=config.min_sep_s)
    pixel_area = np.array([s['npix'] for s in data['stat']], dtype=float)

    x_er = event_rate / (scale_er if scale_er else 1.0)
    x_pz = peak_dz / (scale_pz if scale_pz else 1.0)
    x_area = pixel_area / (scale_area if scale_area else 1.0)
    lin = bias + w_er * x_er + w_pz * x_pz + w_area * x_area
    scores = 1.0 / (1.0 + np.exp(-lin))

    mask_inside = edge_mask_from_stat(stat, Lx, Ly,
                                      edge_buffer_px=edge_buffer_px, rule=edge_rule)
    scores = scores.copy()
    scores[~mask_inside] = 0.0
    if save_masks:
        np.save(os.path.join(config.folder_name,
                             f'roi_mask_inside_{edge_buffer_px}px.npy'), mask_inside)
        np.save(os.path.join(config.folder_name, 'roi_scores.npy'), scores)
    return scores


def soft_cell_mask(scores, score_threshold=0.5, top_k_pct=None):
    if top_k_pct is not None:
        k = max(1, int(np.ceil(scores.size * (top_k_pct / 100.0))))
        thresh = np.partition(scores, -k)[-k]
        mask = scores >= thresh
    else:
        mask = scores >= score_threshold

    if mask.sum() < 0.02 * scores.size:
        valid = scores > 0
        if valid.sum() >= 10:
            mu = scores[valid].mean()
            sigma = scores[valid].std()
            tail_thresh = mu + 1.0 * sigma
            mask_alt = scores >= tail_thresh
            if mask_alt.sum() > mask.sum():
                print(f'[SpatialHeatmap] Falling back to tail threshold {tail_thresh:.2f}')
                mask = mask_alt
    if mask.sum() > 1000:
        mask = scores >= 0.68
    return mask


def plot_leadlag_split_spatial_from_csv(folder_name, csv_path=None, prefix='r0p7_',
                                        summary='median', min_events=5, percentile=50.0):
    config = _Config(folder_name, prefix, 15.0, 3.5, 1.5, 0.3)
    data = _load_suite2p_data(config)
    stat = data['stat']
    Ly, Lx = data['Ly'], data['Lx']
    if csv_path is None:
        csv_path = os.path.join(config.root, f'{prefix}coactivation_summary.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV not found: {csv_path}')
    roi_to_lags = {}
    with open(csv_path, 'r', newline='') as f:
        header = f.readline().strip().split(',')
        col = {name: i for i, name in enumerate(header)}
        if 'roi_index_original' not in col or 'relative_lag_s' not in col:
            raise ValueError('CSV must contain: roi_index_original, relative_lag_s')
        for line in f:
            parts = line.strip().split(',')
            if len(parts) <= max(col['roi_index_original'], col['relative_lag_s']):
                continue
            try:
                roi = int(float(parts[col['roi_index_original']]))
                lag = float(parts[col['relative_lag_s']])
            except ValueError:
                continue
            if np.isnan(lag):
                continue
            roi_to_lags.setdefault(roi, []).append(lag)
    roi_ids, roi_summ, roi_n = [], [], []
    for roi, lags in roi_to_lags.items():
        if len(lags) < int(min_events):
            continue
        lags = np.asarray(lags, dtype=float)
        s = float(np.mean(lags)) if summary == 'mean' else float(np.median(lags))
        roi_ids.append(int(roi)); roi_summ.append(s); roi_n.append(int(len(lags)))
    if len(roi_ids) == 0:
        print('[LeadLag] No ROIs met min_events; nothing to plot.')
        return
    roi_summ = np.asarray(roi_summ, dtype=float)
    split = float(np.percentile(roi_summ, percentile))
    classes = np.full(data['N'], np.nan, dtype=float)
    for roi, s in zip(roi_ids, roi_summ):
        if 0 <= roi < data['N']:
            classes[roi] = 0.0 if s <= split else 1.0
    img = utils.paint_spatial(classes, stat, Ly, Lx)
    coverage = utils.paint_spatial(np.ones(len(stat), dtype=float), stat, Ly, Lx)
    img[coverage == 0] = np.nan
    cmap = ListedColormap(['blue', 'red'])
    cmap.set_bad(color=(0.15, 0.15, 0.15, 1.0))
    out_dir = os.path.join(config.root, f'{prefix}coact_leadlag')
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir,
        f'{prefix}coact_leadlag_{summary}_p{int(percentile)}_min{int(min_events)}.png')
    title = (f'Lead/Lag split by {summary} relative lag '
             f'(p{percentile:.0f}={split:.3f}s)\\n'
             f'blue=early, red=late, min_events={min_events}')
    _show_spatial(img, title, Lx, Ly, stat,
                  pix_to_um=data['pix_to_um'], cmap=cmap, outpath=out_png)
"""


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    root = r"F:\data\2p_shifted\Cx\2024-07-01_00018"

    # Example: look at ROI features (skipping masked variants)
    # view_roi_features(
    #     folder_name=root,
    #     cell_mask_path=None,   # or "F:\...\r0p7_cell_mask_bool.npy"
    #     prefix="r0p7_",
    # )

    # Example: coactivation maps with propagation arrows
    cell_mask_path = os.path.join(
        root, "suite2p", "plane0", "r0p7_cell_mask_bool.npy"
    )
    coactivation_maps(
        folder_name=root,
        cell_mask_path=cell_mask_path,
        propagation_vectors=True,
        prefix="r0p7_",
    )