import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import utils
from typing import Union
import matplotlib as mpl
import sys

# --- Custom cyan→blue→red map with grey background ---
CYAN_TO_RED = mpl.colors.LinearSegmentedColormap.from_list(
    "cyan_to_red", ["#00FFFF", "#0000FF", "#FF0000"], N=256
).copy()  # make mutable copy

# Set NaN / "bad" values to neutral grey
CYAN_TO_RED.set_bad(color="#808080")

# ============================= CO-ACTIVATION BINNING & ORDER MAPS =============================

def _event_onsets_by_roi(data, config, t_slice=None):
    """
    Return list of onset-time arrays (seconds) per ROI using MAD-z + hysteresis on dt.
    """
    T = data['T']
    fps = config.fps
    dt = data['dt']

    if t_slice is None:
        t0, t1 = 0, T
    else:
        t0 = 0 if t_slice.start is None else int(t_slice.start)
        t1 = T if t_slice.stop is None else int(t_slice.stop)

    onsets_sec = []
    for i in range(dt.shape[1]):
        x = dt[t0:t1, i]
        z, _, _ = utils.mad_z(x)  # robust z
        idxs = utils.hysteresis_onsets(z, config.z_enter, config.z_exit, fps)  # onset indices (relative to slice)
        onsets_sec.append(np.asarray(idxs, dtype=np.int64) / fps + (t0 / fps))
    return onsets_sec


def _bin_edges_and_indexer(T, fps, bin_sec):
    """
    Prepare bin edges in seconds and a helper that maps onset times->bin index.
    """
    total_sec = T / fps
    n_bins = int(np.ceil(total_sec / bin_sec))
    edges = np.linspace(0.0, n_bins * bin_sec, n_bins + 1)
    return edges


def _activation_matrix(onsets_by_roi, edges):
    """
    Build (N, B) boolean matrix where entry [i, b] is True if ROI i has any onset within bin b.
    """
    N = len(onsets_by_roi)
    B = len(edges) - 1
    A = np.zeros((N, B), dtype=bool)
    # We'll also store first-onset time within each bin for later ordering
    first_time = np.full((N, B), np.nan, dtype=float)

    for i, ts in enumerate(onsets_by_roi):
        if ts.size == 0:
            continue
        # digitize on [edges[b], edges[b+1])  (right=False behavior)
        bins = np.searchsorted(edges, ts, side='right') - 1
        # keep only valid bins
        valid = (bins >= 0) & (bins < B)
        if not np.any(valid):
            continue
        ubins = np.unique(bins[valid])
        A[i, ubins] = True
        # first-onset per bin
        for b in ubins:
            mask_b = (bins == b) & valid
            if np.any(mask_b):
                first_time[i, b] = np.min(ts[mask_b])
    return A, first_time


import numpy as np

def _select_high_coactivation_bins(
    A,
    frac_required=0.8,
    min_count=None,
    target_min=20,
    target_max=50,
    max_iters=300,
    relax_factor=0.7,
    tighten_factor=1.1,
):
    """
    Return indices of bins where the count of active ROIs exceeds a threshold.

    Behavior:
    - If min_count is provided, it is used as the initial threshold (integer).
    - Otherwise threshold is ceil(frac_required * N).
    - Iteratively relax/tighten threshold to try to get keep_bins in [target_min, target_max].
    - Never recurses; bounded by max_iters.
    - Safe fallbacks when recordings have too few / zero coactivation bins.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"A must be 2D (N, B). Got {A.shape}")
    N, B = A.shape

    active_counts = A.sum(axis=0)

    # If there are no bins or no activity at all, return empty safely.
    if B == 0 or active_counts.max(initial=0) == 0:
        return np.array([], dtype=int), active_counts

    # Initial threshold
    if min_count is None:
        # clamp frac_required to [0, 1] for safety
        frac = float(frac_required)
        frac = max(0.0, min(1.0, frac))
        thresh = int(np.ceil(frac * N))
    else:
        thresh = int(min_count)

    # Clamp threshold to valid range
    thresh = max(1, min(N, thresh))

    def bins_for_threshold(t: int):
        return np.where(active_counts >= t)[0]

    keep_bins = bins_for_threshold(thresh)

    # If total bins are small, just return what we can without trying to hit 20–50.
    if B < target_min:
        return keep_bins, active_counts

    # Iterate to get within desired range
    for _ in range(max_iters):
        k = keep_bins.size
        if target_min <= k <= target_max:
            return keep_bins, active_counts

        if k < target_min:
            # Too few bins: relax threshold (make it easier)
            # Adjust frac_required or directly reduce thresh; here we directly reduce thresh.
            new_thresh = max(1, int(np.floor(thresh * relax_factor)))
            if new_thresh == thresh:
                new_thresh = max(1, thresh - 1)
            thresh = new_thresh

        else:  # k > target_max
            # Too many bins: tighten threshold (make it harder)
            new_thresh = min(N, int(np.ceil(thresh * tighten_factor)))
            if new_thresh == thresh:
                new_thresh = min(N, thresh + 1)
            thresh = new_thresh

        keep_bins = bins_for_threshold(thresh)

        # If threshold hits bounds and we still can't satisfy, stop.
        if thresh == 1 and keep_bins.size < target_min:
            break
        if thresh == N and keep_bins.size > target_max:
            break

    # ----- Fallbacks if we couldn't hit the target window -----
    # If we have *some* bins but fewer than target_min, keep them.
    if 0 < keep_bins.size < target_min:
        return keep_bins, active_counts

    # If we still have too many bins, choose top bins by coactivation count.
    if keep_bins.size > target_max:
        # pick bins with highest active_counts
        order = np.argsort(active_counts)[::-1]
        top = order[:target_max]
        top.sort()
        return top.astype(int), active_counts

    # If we ended up with none, return empty safely.
    return np.array([], dtype=int), active_counts



def _order_map_for_bin(first_time_col, active_mask_col):
    """
    For one bin: produce a ranking (1..K) for active ROIs by their first onset time in that bin.
    Returns vector 'order_rank' (N,) with NaN for inactives, 1 for earliest, etc.
    """
    order_rank = np.full_like(first_time_col, np.nan, dtype=float)
    # Only consider those with a valid time and active flag
    sel = active_mask_col & ~np.isnan(first_time_col)
    if not np.any(sel):
        return order_rank
    times = first_time_col[sel]
    # argsort times ascending -> ranks 1..K
    idx_sorted = np.argsort(times, kind='mergesort')  # stable
    ranks = np.empty_like(idx_sorted, dtype=float)
    ranks[idx_sorted] = np.arange(1, idx_sorted.size + 1, dtype=float)
    # place back
    order_rank[np.where(sel)[0]] = ranks
    return order_rank


def _paint_order_map(order_rank, stat, Ly, Lx):
    vals = order_rank.astype(float).copy()

    # if everything is NaN, return a fully-NaN image (grey background)
    if np.all(np.isnan(vals)):
        img = utils.paint_spatial(np.full_like(order_rank, np.nan, dtype=float), stat, Ly, Lx)
        coverage = utils.paint_spatial(np.ones(len(stat), dtype=float), stat, Ly, Lx)
        img[coverage == 0] = np.nan
        return img

    # map ranks → 0..1; earliest=1, latest→0 (you invert later if you want cyan→red)
    maxr = np.nanmax(vals)
    inv = (maxr - vals + 1.0)
    inv[np.isnan(vals)] = np.nan
    vals = inv / maxr

    # paint & set true background to NaN
    img = utils.paint_spatial(vals, stat, Ly, Lx)
    coverage = utils.paint_spatial(np.ones(len(stat), dtype=float), stat, Ly, Lx)
    img[coverage == 0] = np.nan
    return img



def coactivation_order_heatmaps(
    folder_name,
    prefix='r0p7_',
    fps=15.0,
    z_enter=3.5,
    z_exit=1.5,
    min_sep_s=0.3,
    bin_sec=0.5,
    frac_required=0.8,
    # filtering (scores)
    w_er=1.0, w_pz=1.0, w_area=0.5,
    scale_er=1.0, scale_pz=3.0, scale_area=50.0,
    bias=-2.0,
    score_threshold=0.5,
    top_k_pct=None,
    # I/O
    cmap='viridis'
):
    """
    1) Computes weighted cell scores -> mask
    2) Finds time bins (bin_sec) where >= frac_required of filtered cells activate
    3) For each such bin, saves a spatial heatmap colored by activation order within that bin
    """
    # Load data
    config = SpatialHeatmapConfig(folder_name, metric='event_rate', prefix=prefix,
                                  fps=fps, z_enter=z_enter, z_exit=z_exit,
                                  min_sep_s=min_sep_s, bin_seconds=None)
    data = _load_suite2p_data(config)

    # --- FOV in µm from zoom ---
    zoom = utils.get_zoom_from_notes(folder_name)  # same pattern as Hz; uses .lower() columns
    zoom = float(zoom) if zoom else 1.0

    fov_um_x = 3080.90169 / zoom
    fov_um_y = 3560.14057 / zoom

    # --- filter to "cells" via scores ---
    if os.path.exists(os.path.join(folder_name, 'roi_scores.npy')):
        scores = np.load(os.path.join(folder_name, 'roi_scores.npy'))
    else:
        scores = compute_cell_scores(
            data, config,
            w_er=w_er, w_pz=w_pz, w_area=w_area,
            scale_er=scale_er, scale_pz=scale_pz, scale_area=scale_area,
            bias=bias
        )

    cell_mask = soft_cell_mask(scores, score_threshold=score_threshold, top_k_pct=top_k_pct)
    print(f"[CoAct] Using {cell_mask.sum()} / {len(cell_mask)} ROIs after filter.")

    # --- event onsets per ROI (whole recording) ---
    onsets = _event_onsets_by_roi(data, config, t_slice=None)

    # keep only filtered ROIs
    onsets = [onsets[i] for i in np.where(cell_mask)[0]]

    # --- binning and co-activation selection ---
    edges = _bin_edges_and_indexer(data['T'], config.fps, bin_sec)
    A, first_time = _activation_matrix(onsets, edges)
    keep_bins, active_counts = _select_high_coactivation_bins(A, frac_required=frac_required)

    if keep_bins.size == 0:
        print("[CoAct] No bins met the co-activation threshold.")
        return

    Ly, Lx = data['Ly'], data['Lx']
    stat_all = data['stat']
    # Build a "filtered" stat for paint_spatial: keep same indexing (we painted with per-ROI arrays aligned to stat).
    # We will create an array of length N_all with NaN for non-kept ROIs, values only for kept.
    N_all = data['N']
    idx_keep = np.where(cell_mask)[0]

    # --- For each selected bin: create order map and save ---
    stat_filtered = [data['stat'][i] for i in idx_keep]  # only cell ROIs
    # ---- Save coactivation per-ROI timing summary ----
    summary_csv = os.path.join(config.root, f"{config.prefix}coactivation_summary.csv")

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "roi_index_filtered",
                "roi_index_original",
                "bin_index",
                "bin_start_s",
                "bin_end_s",
                "first_onset_s",
                "relative_lag_s",
                "order_rank",
            ],
        )
        writer.writeheader()

        for b in keep_bins:
            # ranks within this bin (1=earliest)
            order_rank = _order_map_for_bin(first_time[:, b], A[:, b])

            # earliest onset among active ROIs in this bin
            active_times = first_time[A[:, b] & ~np.isnan(first_time[:, b]), b]
            if active_times.size == 0:
                continue
            earliest_time = float(np.min(active_times))

            t0 = float(edges[b])
            t1 = float(edges[b + 1])

            # write one row per active ROI
            active_rois = np.where(A[:, b])[0]
            for roi_local_idx in active_rois:
                ft = first_time[roi_local_idx, b]
                if np.isnan(ft):
                    continue

                writer.writerow(
                    {
                        "roi_index_filtered": int(roi_local_idx),
                        "roi_index_original": int(idx_keep[roi_local_idx]),
                        "bin_index": int(b),
                        "bin_start_s": t0,
                        "bin_end_s": t1,
                        "first_onset_s": float(ft),
                        "relative_lag_s": float(ft) - earliest_time,
                        "order_rank": float(order_rank[roi_local_idx]),
                    }
                )

    print(f"[CoAct] Saved coactivation summary CSV → {summary_csv}")

    # ---- propagation vector outputs ----
    prop_dir = os.path.join(config.root, f"{config.prefix}coact_propagation_vectors")
    if not os.path.exists(prop_dir):
        os.makedirs(prop_dir)

    prop_csv = os.path.join(config.root, f"{config.prefix}coactivation_propagation.csv")
    prop_rows = []

    for b in keep_bins:
        order_rank_filtered = _order_map_for_bin(first_time[:, b], A[:, b])

        # Only keep the filtered cells (no NaN overlay for non-cells)
        spatial_order = _paint_order_map(order_rank_filtered, stat_filtered, Ly, Lx)
        # ---- propagation vector (start/end from earliest/latest onset frame ROIs) ----
        prop = _compute_propagation_vector_for_bin(first_time[:, b], A[:, b], stat_filtered, fps=config.fps)

        if prop is not None:
            start_px, end_px, dt_sec, n_first, n_last = prop

            # convert px -> µm (anisotropic)
            um_per_px_x = float(fov_um_x) / float(Lx)
            um_per_px_y = float(fov_um_y) / float(Ly)

            start_um = np.array([start_px[0] * um_per_px_x, start_px[1] * um_per_px_y], dtype=float)
            end_um   = np.array([end_px[0]   * um_per_px_x, end_px[1]   * um_per_px_y], dtype=float)

            # Arrow was reversed
            tmp = start_um
            start_um = end_um
            end_um = tmp

            vec_um   = end_um - start_um

            dist_um = float(np.hypot(vec_um[0], vec_um[1]))
            speed = dist_um / dt_sec if (dt_sec is not None and dt_sec > 0) else 0.0
            angle_deg = float(np.degrees(np.arctan2(vec_um[1], vec_um[0]))) if dist_um > 0 else 0.0

            # Save arrow overlay image (separate folder)
            out_arrow = os.path.join(prop_dir, f"{config.prefix}coact_propagation_bin{b:04d}.png")
            title_arrow = (
                f"Propagation vector (bin {b}: {t0:.2f}–{t1:.2f}s)\n"
                f"speed={speed:.1f} µm/s, angle={angle_deg:.1f}°, dt={dt_sec:.3f}s "
                f"(firstROIs={n_first}, lastROIs={n_last})"
            )

            _show_spatial_with_arrow_um(
                spatial_order,
                title_arrow,
                fov_um_x=fov_um_x,
                fov_um_y=fov_um_y,
                arrow_start_um=start_um,
                arrow_vec_um=vec_um,
                stat_filtered=stat_filtered,
                outpath=out_arrow,
                cmap=CYAN_TO_RED,
            )

            # record CSV row
            prop_rows.append(
                {
                    "bin_index": int(b),
                    "bin_start_s": float(t0),
                    "bin_end_s": float(t1),
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
                }
            )

        t0 = edges[b]
        t1 = edges[b + 1]
        frac = active_counts[b] / A.shape[0]
        title = (f"Activation order in bin {b} ({t0:.2f}–{t1:.2f}s)\n"
                 f"active={active_counts[b]}/{A.shape[0]} ({100 * frac:.1f}%)")

        new_root = os.path.join(config.root, f"{config.prefix}coact_order_bin_cells")
        if not os.path.exists(new_root):
            os.makedirs(new_root)
        out = os.path.join(new_root, f"{config.prefix}coact_order_bin{b:04d}_cells.png")
        show_spatial(spatial_order, title, Lx, Ly, stat_filtered,
                     pix_to_um=data['pix_to_um'], cmap=CYAN_TO_RED, outpath=out)

    print(f"[CoAct] Saved {keep_bins.size} co-activation order maps.")
    # ---- write propagation CSV ----
    if len(prop_rows) > 0:
        import csv as _csv
        with open(prop_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(prop_rows[0].keys()))
            w.writeheader()
            w.writerows(prop_rows)
        print(f"[CoAct] Saved propagation CSV → {prop_csv}")
    else:
        print("[CoAct] No propagation vectors computed (no valid bins/ROIs).")



    # ---- collect relative lags per ROI ----
    roi_rel_lags = [[] for _ in range(A.shape[0])]  # filtered ROI indexing

    for b in keep_bins:
        sel = A[:, b] & ~np.isnan(first_time[:, b])
        if not np.any(sel):
            continue
        earliest = float(np.min(first_time[sel, b]))
        for i in np.where(sel)[0]:
            roi_rel_lags[i].append(float(first_time[i, b]) - earliest)

    # Keep only ROIs that have at least a few events
    min_events = 5
    valid = [(i, np.asarray(lags, float)) for i, lags in enumerate(roi_rel_lags) if len(lags) >= min_events]
    if len(valid) == 0:
        print("[CoAct] No ROIs have enough events for violin plot.")
    else:
        # Sort by median lag (initiators left)
        valid.sort(key=lambda t: np.median(t[1]))

        # Optional: limit number of ROIs shown (otherwise x-axis becomes unreadable)
        top_n = 60  # adjust; 40–80 is usually sane
        valid = valid[:top_n]

        roi_idx = [i for i, _ in valid]
        data = [lags for _, lags in valid]
        medians = np.array([np.median(l) for l in data])
        n_events = np.array([len(l) for l in data])

        # Labels = original ROI indices (so you can map back)
        labels = [str(int(idx_keep[i])) for i in roi_idx]

        # ---- plot ----
        plt.figure(figsize=(max(12, 0.22 * len(data)), 6))

        parts = plt.violinplot(
            data,
            showmeans=False,
            showmedians=True,
            showextrema=False,
            widths=0.9,
        )

        # Overlay median points (helps readability)
        x = np.arange(1, len(data) + 1)
        plt.scatter(x, medians, s=10)

        plt.xticks(x, labels, rotation=90)
        plt.ylabel("Relative lag (s) = ROI first onset - earliest onset in bin")
        plt.xlabel("ROI (original index), sorted by median relative lag")
        plt.title(f"Per-ROI relative lag distributions (top {len(data)} ROIs, min_events={min_events})")
        plt.ylim(0, float(bin_sec))  # relative lag should live in [0, bin_sec] typically
        plt.tight_layout()

        out_png = os.path.join(config.root, f"{config.prefix}coactivation_roi_relative_lag_violin.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[CoAct] Saved violin plot → {out_png}")


# ---- Cell masking ----

def edge_mask_from_stat(stat, Lx, Ly, edge_buffer_px=10, rule="centroid"):
    """
    True = ROI is safely inside the FOV (not near edges).
    rule='centroid' uses the mean x/y of ROI pixels.
    rule='bbox' excludes if any ROI pixel falls within edge_buffer of a border.
    """
    if rule == "centroid":
        xs = np.array([np.mean(s['xpix']) for s in stat], dtype=float)
        ys = np.array([np.mean(s['ypix']) for s in stat], dtype=float)
        inside = (
            (xs > edge_buffer_px) & (xs < (Lx - edge_buffer_px)) &
            (ys > edge_buffer_px) & (ys < (Ly - edge_buffer_px))
        )
    elif rule == "bbox":
        xmins = np.array([s['xpix'].min() for s in stat])
        xmaxs = np.array([s['xpix'].max() for s in stat])
        ymins = np.array([s['ypix'].min() for s in stat])
        ymaxs = np.array([s['ypix'].max() for s in stat])
        inside = (
            (xmins > edge_buffer_px) & (xmaxs < (Lx - edge_buffer_px)) &
            (ymins > edge_buffer_px) & (ymaxs < (Ly - edge_buffer_px))
        )
    else:
        raise ValueError("rule must be 'centroid' or 'bbox'")
    return inside.astype(bool)

def _safe_div(x, d):
    d = float(d) if d else 1.0
    return x / d

def compute_cell_scores(data, config,
                        w_er=1.0, w_pz=1.0, w_area=0.5,
                        scale_er=1.0, scale_pz=3.0, scale_area=50.0,
                        bias=-2.0,
                        t_slice=None,
                        edge_buffer_px=6,  # <<< NEW
                        edge_rule="centroid",  # <<< NEW ('centroid' or 'bbox')
                        save_masks=True  # <<< optional
                        ):
    """
    Returns an array (N,) of cell probabilities for each ROI.
    """
    stat = data['stat']
    Lx, Ly = data['Lx'], data['Ly']
    signals = {'low': data['low'], 'dt': data['dt']}
    time_slice = t_slice if t_slice is not None else slice(None)

    event_rate = utils.roi_metric(
        signals, which='event_rate', t_slice=time_slice,
        fps=config.fps, z_enter=config.z_enter, z_exit=config.z_exit,
        min_sep_s=config.min_sep_s
    )  # typically events/min

    peak_dz = utils.roi_metric(
        signals, which='peak_dz', t_slice=time_slice,
        fps=config.fps, z_enter=config.z_enter, z_exit=config.z_exit,
        min_sep_s=config.min_sep_s
    )

    pixel_area = np.array([s['npix'] for s in data['stat']], dtype=float)

    # Vectorized logistic scoring
    x_er   = event_rate / (scale_er if scale_er else 1.0)
    x_pz   = peak_dz    / (scale_pz if scale_pz else 1.0)
    x_area = pixel_area / (scale_area if scale_area else 1.0)

    lin = bias + w_er * x_er + w_pz * x_pz + w_area * x_area
    scores = 1.0 / (1.0 + np.exp(-lin))

    mask_inside = edge_mask_from_stat(stat, Lx, Ly,
                                      edge_buffer_px=edge_buffer_px,
                                      rule=edge_rule)
    scores = scores.copy()
    scores[~mask_inside] = 0.0

    if save_masks:
        np.save(os.path.join(config.folder_name, f'roi_mask_inside_{edge_buffer_px}px.npy'), mask_inside)
        np.save(os.path.join(config.folder_name, 'roi_scores.npy'), scores)

    return scores

def soft_cell_mask(scores, score_threshold=0.5, top_k_pct=None):
    """
    Convert probabilities into a boolean mask.
    If top_k_pct is set (e.g., 20 for top 20%), it overrides score_threshold.
    """

    """zero_frac = np.mean(scores == 0)
    if zero_frac >= 0.1:
        return scores != 0""" #only include this code block if 0 is invalid
    # if we have wayyyy to many little cells lower the threshold

    #if (scores >= score_threshold).sum() < scores.size * 0.1:
    #    k = int(np.ceil(0.1 * scores.size))
    #    thresh = np.partition(scores, -k)[-k]
    #    return scores >= thresh
    if top_k_pct is not None:
        k = max(1, int(np.ceil(scores.size * (top_k_pct / 100.0))))
        thresh = np.partition(scores, -k)[-k]
        mask = scores >= thresh
    else:
        mask = scores >= score_threshold

    # fall back if too little cells
    if mask.sum() < 0.02 * scores.size:
        valid = scores > 0  # ignore structural zeros
        if valid.sum() >= 10:
            mu = scores[valid].mean()
            sigma = scores[valid].std()
            tail_thresh = mu + 1.0 * sigma
            mask_alt = scores >= tail_thresh
            if mask_alt.sum() > mask.sum():
                print(f"[SpatialHeatmap] Falling back to tail threshold {tail_thresh:.2f}")
                mask = mask_alt

    if mask.sum() > 1000: # default
        mask = scores >= 0.68

    return mask

# ---- Helpers ----
def _roi_centroids_xy(stat_list):
    """Return centroid (x, y) in pixel coordinates for each ROI in stat_list."""
    xs = np.array([float(np.median(s["xpix"])) for s in stat_list], dtype=float)
    ys = np.array([float(np.median(s["ypix"])) for s in stat_list], dtype=float)
    return xs, ys

def _compute_propagation_vector_for_bin(first_time_col, active_mask_col, stat_filtered, fps):
    """
    Compute a single propagation vector for a bin:
      - start = mean centroid of ROIs active on the earliest onset frame
      - end   = mean centroid of ROIs active on the latest onset frame
      - velocity = distance(start->end) / (t_last - t_first)

    Returns:
      start_xy_px (2,), end_xy_px (2,), dt_sec, n_first, n_last
    """
    sel = active_mask_col & ~np.isnan(first_time_col)
    if not np.any(sel):
        return None

    times = first_time_col[sel].astype(float)
    roi_idx = np.where(sel)[0]

    t_first = float(np.min(times))
    t_last  = float(np.max(times))

    # define "same frame" tolerance as ±0.5 frame
    tol = 0.5 / float(fps)

    first_rois = roi_idx[np.abs(first_time_col[roi_idx] - t_first) <= tol]
    last_rois  = roi_idx[np.abs(first_time_col[roi_idx] - t_last)  <= tol]

    xs, ys = _roi_centroids_xy(stat_filtered)

    # Fallbacks (in case tolerance yields none, though it usually won’t)
    if first_rois.size == 0:
        first_rois = np.array([roi_idx[np.argmin(first_time_col[roi_idx])]], dtype=int)
    if last_rois.size == 0:
        last_rois = np.array([roi_idx[np.argmax(first_time_col[roi_idx])]], dtype=int)

    start = np.array([xs[first_rois].mean(), ys[first_rois].mean()], dtype=float)
    end   = np.array([xs[last_rois].mean(),  ys[last_rois].mean()],  dtype=float)

    return start, end, (t_last - t_first), int(first_rois.size), int(last_rois.size)

def _show_spatial_with_arrow_um(
    img,
    title,
    fov_um_x,
    fov_um_y,
    arrow_start_um,
    arrow_vec_um,
    stat_filtered,
    outpath,
    cmap=CYAN_TO_RED,
):
    """
    Save spatial image with an arrow overlay in µm coordinates.
    """
    extent = [0, float(fov_um_x), 0, float(fov_um_y)]

    plt.figure(figsize=(8, 7))
    im = plt.imshow(img, origin="lower", cmap=cmap, extent=extent, aspect="equal")

    # ROI centroid overlay in µm
    xs_px, ys_px = _roi_centroids_xy(stat_filtered)
    # convert px -> µm using fov / dims inferred from image shape
    Ly, Lx = img.shape[0], img.shape[1]
    um_per_px_x = float(fov_um_x) / float(Lx)
    um_per_px_y = float(fov_um_y) / float(Ly)
    xs_um = xs_px * um_per_px_x
    ys_um = ys_px * um_per_px_y
    plt.scatter(xs_um, ys_um, s=4, c="white", alpha=0.35, linewidths=0)

    # Arrow
    sx, sy = float(arrow_start_um[0]), float(arrow_start_um[1])
    vx, vy = float(arrow_vec_um[0]), float(arrow_vec_um[1])

    # Draw arrow only if non-trivial
    if np.isfinite(vx) and np.isfinite(vy) and (abs(vx) + abs(vy)) > 1e-6:
        plt.arrow(
            sx, sy, vx, vy,
            length_includes_head=True,
            head_width=0.03 * max(fov_um_x, fov_um_y),
            head_length=0.04 * max(fov_um_x, fov_um_y),
            linewidth=2.0,
            color="white",
        )

    plt.colorbar(im, label=title)
    plt.title(title)
    plt.xlabel("X (µm)")
    plt.ylabel("Y (µm)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print("Saved", outpath)

def show_spatial(img, title, Lx, Ly, stat, pix_to_um=None, cmap='magma', outpath=None, ):
    """
    Display/save a spatial scalar map with optional µm axes and ROI centroid overlay.
    """
    extent = None
    xlabel, ylabel = 'X (pixels)', 'Y (pixels)'
    if pix_to_um is not None:
        extent = [0, Lx * pix_to_um, 0, Ly * pix_to_um]
        xlabel, ylabel = 'X (µm)', 'Y (µm)'

    plt.figure(figsize=(8, 7))
    im = plt.imshow(img, origin='lower', cmap=cmap, extent=extent, aspect='equal')
    # Light overlay of ROI centroids (helps sanity-check registration)
    xs = [np.median(s['xpix']) for s in stat]
    ys = [np.median(s['ypix']) for s in stat]
    if pix_to_um is not None:
        xs = np.array(xs) * pix_to_um
        ys = np.array(ys) * pix_to_um
    #plt.scatter(xs, ys, s=4, c='white', alpha=0.35, linewidths=0)
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

class SpatialHeatmapConfig:
    """Configuration parameters for spatial heatmap generation."""

    def __init__(self, folder_name, metric='event_rate', prefix='r0p7_',
                 fps=15.0, z_enter=3.5, z_exit=1.5, min_sep_s=0.3, bin_seconds=None):
        self.folder_name = folder_name
        self.metric = metric
        self.prefix = prefix
        self.fps = fps
        self.z_enter = z_enter
        self.z_exit = z_exit
        self.min_sep_s = min_sep_s
        self.bin_seconds = bin_seconds

        # Derived paths
        self.root = os.path.join(folder_name, "suite2p\\plane0\\")
        self.sample_name = folder_name.split("\\")[-1] if "\\" in folder_name else folder_name.split("/")[-1]

    def get_metric_title(self):
        """Generate title based on metric type."""
        titles = {
            'event_rate': f'Event rate (events/min) — z_enter={self.z_enter}, z_exit={self.z_exit} ({self.sample_name})',
            'mean_dff': f'Mean ΔF/F (low-pass) ({self.sample_name})',
            'peak_dz': f'Peak derivative z (robust) ({self.sample_name})'
        }
        return titles[self.metric]


def _load_suite2p_data(config):
    """Load Suite2p metadata and processed signals."""
    ops = np.load(os.path.join(config.root, 'ops.npy'), allow_pickle=True).item()
    stat = np.load(os.path.join(config.root, 'stat.npy'), allow_pickle=True)

    Ly, Lx = ops['Ly'], ops['Lx']
    pix_to_um = ops.get('pix_to_um', None)

    # Load memmaps
    low = np.memmap(os.path.join(config.root, f'{config.prefix}dff_lowpass.memmap.float32'),
                    dtype='float32', mode='r')
    dt = np.memmap(os.path.join(config.root, f'{config.prefix}dff_dt.memmap.float32'),
                   dtype='float32', mode='r')

    # Reshape to (T, N)
    N = len(stat)
    T = low.size // N
    low = low.reshape(T, N)
    dt = dt.reshape(T, N)

    return {
        'stat': stat,
        'Ly': Ly,
        'Lx': Lx,
        'pix_to_um': pix_to_um,
        'low': low,
        'dt': dt,
        'T': T,
        'N': N
    }


def _compute_and_save_spatial_map(data, config, t_slice=None, bin_index=None,
                                  scores: Union[np.ndarray, None] =None,
                                  score_threshold: float = 0.5,
                                  top_k_pct: Union[float, None] =None):
    """Compute metric values and generate spatial heatmap."""
    signals = {'low': data['low'], 'dt': data['dt']}
    time_slice = t_slice if t_slice is not None else slice(None)

    vals = utils.roi_metric(signals, which=config.metric, t_slice=time_slice,
                      fps=config.fps, z_enter=config.z_enter,
                      z_exit=config.z_exit, min_sep_s=config.min_sep_s)

    spatial = utils.paint_spatial(vals, data['stat'], data['Ly'], data['Lx'])

    new_root = os.path.join(config.root, f'{config.prefix}spatial_{config.metric}')
    if not os.path.exists(new_root):
        os.makedirs(new_root)
    # Generate output path and title
    if bin_index is None:
        out = os.path.join(new_root, f'{config.prefix}spatial_{config.metric}')
        title = config.get_metric_title()
    else:
        out = os.path.join(new_root, f'{config.prefix}spatial_{config.metric}_bin{bin_index:03d}')
        t0, t1 = t_slice.start, t_slice.stop
        title = f'{config.get_metric_title()}\nWindow {bin_index}: {t0 / config.fps:.1f}–{t1 / config.fps:.1f} s'
    
    show_spatial(spatial, title, data['Lx'], data['Ly'], data['stat'],
                 pix_to_um=data['pix_to_um'], cmap='magma', outpath=out)

    # 2) Probability-driven maps
    if scores is not None:
        # ROI-wise -> pixel map of probabilities
        spatial_prob = utils.paint_spatial(scores, data['stat'], data['Ly'], data['Lx'])
        show_spatial(spatial_prob, "Cell-likeness probability", data['Lx'], data['Ly'], data['stat'],
                     pix_to_um=data['pix_to_um'], cmap='magma', outpath=out + '_prob.png')

        # Soft mask from scores
        mask = soft_cell_mask(scores, score_threshold=score_threshold, top_k_pct=top_k_pct)

        n_total = scores.size
        n_pass = int(mask.sum())
        print(f"[SpatialHeatmap] Retained {n_pass} / {n_total} ROIs (≥ score {score_threshold:.2f})")

        # Masked metric
        vals_masked = np.where(mask, vals, np.nan)
        spatial_masked = utils.paint_spatial(vals_masked, data['stat'], data['Ly'], data['Lx'])
        show_spatial(spatial_masked, title + " (prob-masked)", data['Lx'], data['Ly'], data['stat'],
                     pix_to_um=data['pix_to_um'], cmap='magma', outpath=out + '_probmasked.png')

        idx_keep = np.where(mask)[0]
        if idx_keep.size > 0:
            vals_filtered = vals[idx_keep]
            stat_filtered = [data['stat'][i] for i in idx_keep]
            spatial_filtered = utils.paint_spatial(vals_filtered, stat_filtered, data['Ly'], data['Lx'])
            show_spatial(spatial_filtered, title + " (filtered only)", data['Lx'], data['Ly'], stat_filtered,
                         pix_to_um=data['pix_to_um'], cmap='magma', outpath=out + '_probmask_cells_only.png')
        else:
            print("[spatial_heatmap] No ROIs passed the filter — skipping filtered-only map.")

def _generate_time_binned_maps(data, config):
    """Generate time-binned spatial heatmaps."""
    T = data['T']
    Tbin = int(config.bin_seconds * config.fps)
    n_bins = int(np.ceil(T / Tbin))

    for b in range(n_bins):
        t0 = b * Tbin
        t1 = min(T, (b + 1) * Tbin)

        # Skip tiny tail windows to avoid noisy/empty maps
        if t1 - t0 < max(5, int(0.2 * Tbin)):
            continue

        _compute_and_save_spatial_map(data, config,
                                      t_slice=slice(t0, t1),
                                      bin_index=b + 1)


def run_spatial_heatmap(folder_name, metric='event_rate', prefix='r0p7_',
                        fps=15.0, z_enter=3.5, z_exit=1.5, min_sep_s=0.3, bin_seconds=None,
                        # scoring params
                        w_er=1.0, w_pz=1.0, w_area=0.5,
                        scale_er=1.0, scale_pz=3.0, scale_area=50.0,
                        bias=-2.0,
                        score_threshold=0.5, top_k_pct=None):
    """
    Generate spatial heatmaps of calcium imaging metrics.

    :param folder_name: Folder to run heatmap on
    :param metric: Metric to display on the spatial heatmap (one scalar per ROI)
        options: 'event_rate', 'mean_dff', 'peak_dz'
    :param prefix: Must match your preprocessed memmap filename prefix
    :param fps: Frame rate (Hz)
    :param z_enter: Event detection entry threshold
    :param z_exit: Event detection exit threshold
    :param min_sep_s: Merge onsets that are < n seconds apart
    :param bin_seconds: Make per-time-bin maps (in seconds). Set None to skip binning.
        e.g., 60 for per-minute maps; or None for whole recording
    :return: None
    """
    config = SpatialHeatmapConfig(folder_name, metric, prefix, fps,
                                  z_enter, z_exit, min_sep_s, bin_seconds)

    data = _load_suite2p_data(config)
    if os.path.exists(os.path.join(folder_name, 'roi_scores.npy')):
        scores = np.load(os.path.join(folder_name, 'roi_scores.npy'))
    else:
        # Global scores over whole recording (you can also recompute per bin)
        scores = compute_cell_scores(
            data, config,
            w_er=w_er, w_pz=w_pz, w_area=w_area,
            scale_er=scale_er, scale_pz=scale_pz, scale_area=scale_area,
            bias=bias,
            t_slice=None
        )

    # Generate whole-recording map
    _compute_and_save_spatial_map(
        data, config,
        scores=scores,
        score_threshold=score_threshold,
        top_k_pct=top_k_pct
    )

    # Generate optional time-binned maps
    if config.bin_seconds is not None and config.bin_seconds > 0:
        _generate_time_binned_maps(data, config)

def run(file_name):
    weights = [2.3662, 1.0454, 1.1252, 0.2987]  # (bias, er, pz, area)
    sd_mu = [4.079, 11.24, 41.178]
    sd_sd = [1.146, 4.214, 37.065]
    thres = 0.68
    bias = float(
        weights[0]
        - (weights[1] * sd_mu[0] / sd_sd[0])
        - (weights[2] * sd_mu[1] / sd_sd[1])
        - (weights[3] * sd_mu[2] / sd_sd[2])
    )
    fps = utils.get_fps_from_notes(file_name)
    run_spatial_heatmap(
        file_name,
        metric='event_rate',
        fps=fps, z_enter=3.5, z_exit=1.5, min_sep_s=0.3,
        # scoring: emphasize peak_dz slightly, normalize by typical ranges
        w_er=weights[1], w_pz=weights[2], w_area=weights[3],
        scale_er=float(sd_sd[0]),  # ~1 event/min considered “unit”
        scale_pz=float(sd_sd[1]),  # z≈5 as a “unit bump”
        scale_area=float(sd_sd[2]),  # 10 px as a “unit area”
        bias=bias,  # stricter overall
        score_threshold=thres,  # classify as cell if P>=0.5
        top_k_pct=None  # or set e.g. 25 for top-25% only
    )
def plot_leadlag_split_spatial_from_csv(
    folder_name: str,
    csv_path: str = None,
    prefix: str = "r0p7_",
    summary: str = "median",     # "median" or "mean"
    min_events: int = 5,
    percentile: float = 50.0
):
    """
    Read a coactivation CSV (assumed to have columns: roi_index_original, relative_lag_s),
    compute per-ROI summary lag, split by percentile (default 50th), and plot spatial map:
      early = blue, late = red.

    Saves a PNG into suite2p/plane0/ under a coact_leadlag folder.
    """
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Load suite2p metadata (stat, Ly/Lx, pix_to_um) using existing config/loader
    config = SpatialHeatmapConfig(folder_name, metric="event_rate", prefix=prefix)
    data = _load_suite2p_data(config)
    stat = data["stat"]
    Ly, Lx = data["Ly"], data["Lx"]

    # CSV path default
    if csv_path is None:
        csv_path = os.path.join(config.root, f"{prefix}coactivation_summary.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read CSV (no pandas required)
    # Expect at least: roi_index_original, relative_lag_s
    roi_to_lags = {}
    with open(csv_path, "r", newline="") as f:
        header = f.readline().strip().split(",")
        col = {name: i for i, name in enumerate(header)}

        if "roi_index_original" not in col or "relative_lag_s" not in col:
            raise ValueError(
                "CSV must contain columns: roi_index_original, relative_lag_s "
                f"(found: {list(col.keys())})"
            )

        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= max(col["roi_index_original"], col["relative_lag_s"]):
                continue
            try:
                roi = int(float(parts[col["roi_index_original"]]))
                lag = float(parts[col["relative_lag_s"]])
            except ValueError:
                continue
            if np.isnan(lag):
                continue
            roi_to_lags.setdefault(roi, []).append(lag)

    # Compute per-ROI summary lag
    roi_ids = []
    roi_summ = []
    roi_n = []
    for roi, lags in roi_to_lags.items():
        if len(lags) < int(min_events):
            continue
        lags = np.asarray(lags, dtype=float)
        if summary == "mean":
            s = float(np.mean(lags))
        else:
            s = float(np.median(lags))
        roi_ids.append(int(roi))
        roi_summ.append(s)
        roi_n.append(int(len(lags)))

    if len(roi_ids) == 0:
        print("[LeadLag] No ROIs met min_events; nothing to plot.")
        return

    roi_summ = np.asarray(roi_summ, dtype=float)
    split = float(np.percentile(roi_summ, percentile))

    # Build per-ROI class array over ALL ROIs (length N_all)
    # class: 0=early (blue), 1=late (red), NaN = no data/background
    N_all = data["N"]
    classes = np.full(N_all, np.nan, dtype=float)

    for roi, s in zip(roi_ids, roi_summ):
        if 0 <= roi < N_all:
            classes[roi] = 0.0 if s <= split else 1.0

    # Paint to spatial image
    img = utils.paint_spatial(classes, stat, Ly, Lx)
    coverage = utils.paint_spatial(np.ones(len(stat), dtype=float), stat, Ly, Lx)
    img[coverage == 0] = np.nan

    # Colormap: 0->blue, 1->red
    cmap = ListedColormap(["blue", "red"])
    cmap.set_bad(color=(0.15, 0.15, 0.15, 1.0))  # background (NaN)

    # Save
    out_dir = os.path.join(config.root, f"{prefix}coact_leadlag")
    os.makedirs(out_dir, exist_ok=True)

    out_png = os.path.join(
        out_dir,
        f"{prefix}coact_leadlag_{summary}_p{int(percentile)}_min{int(min_events)}.png"
    )

    title = (
        f"Lead/Lag split by {summary} relative lag (p{percentile:.0f}={split:.3f}s)\n"
        f"blue=early (≤ split), red=late (> split), min_events={min_events}"
    )

    show_spatial(img, title, Lx, Ly, stat, pix_to_um=data["pix_to_um"], cmap=cmap, outpath=out_png)

if __name__ == "__main__":
    # Co-activation with your current scoring params
    weights = [2.3662, 1.0454, 1.1252, 0.2987]  # (bias, er, pz, area)
    sd_mu = [4.079, 11.24, 41.178]
    sd_sd = [1.146, 4.214, 37.065]
    thres = 0.15
    bias = float(
        weights[0]
        - (weights[1] * sd_mu[0] / sd_sd[0])
        - (weights[2] * sd_mu[1] / sd_sd[1])
        - (weights[3] * sd_mu[2] / sd_sd[2])
    )
    root = r'F:\data\2p_shifted\Hip\2024-06-04_00001'
    #run(root)
    fps = utils.get_fps_from_notes(root)
    coactivation_order_heatmaps(
        folder_name=root,
        prefix='r0p7_',
        fps=fps, z_enter=3.5, z_exit=1.5, min_sep_s=0.3,
        bin_sec=0.5,  # 0.5 s bin size
        frac_required=0.8,  # at least 80% of filtered cells active
        # weighted filter (use your fitted values if you have them)
        w_er=weights[1], w_pz=weights[2], w_area=weights[3],
        scale_er=float(sd_sd[0]),  # ~1 event/min considered “unit”
        scale_pz=float(sd_sd[1]),  # z≈5 as a “unit bump”
        scale_area=float(sd_sd[2]),  # 10 px as a “unit area”
        bias=bias,  # stricter overall
        score_threshold=thres,  # classify as cell if P>=0.5,  # from your fitted model; or use top_k_pct
        top_k_pct=None,
        cmap='viridis'  # any matplotlib cmap
    )

    plot_leadlag_split_spatial_from_csv(
        folder_name=root,
        prefix="r0p7_",
        summary="mean",
        min_events=5,
        percentile=25.0
    )

    #utils.log(
    #    "cell_detection.log",
    #    utils.run_on_folders(r'F:\data\2p_shifted',run)
    #)
#