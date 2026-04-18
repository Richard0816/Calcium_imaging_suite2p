import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib as mpl
import utils

# --- Custom cyan→blue→red map with grey background ---
CYAN_TO_RED = mpl.colors.LinearSegmentedColormap.from_list(
    "cyan_to_red", ["#00FFFF", "#0000FF", "#FF0000"], N=256
).copy()  # make mutable copy

# Set NaN / "bad" values to neutral grey
CYAN_TO_RED.set_bad(color="#808080")
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



def _bin_edges_and_indexer(T, fps, bin_sec):
    """
    Prepare bin edges in seconds and a helper that maps onset times->bin index.
    """
    total_sec = T / fps
    n_bins = int(np.ceil(total_sec / bin_sec))
    edges = np.linspace(0.0, n_bins * bin_sec, n_bins + 1)
    return edges


def _draw_scale_bar_right(ax, fov_um_x, fov_um_y, bar_um=200.0, color="white", lw=4, fontsize=10):
    """
    Minimal 200 µm scale bar placed at bottom right in µm coordinates.
    """
    pad_x = 0.05 * fov_um_x
    pad_y = 0.06 * fov_um_y

    x1 = fov_um_x - pad_x
    x0 = x1 - bar_um
    y = pad_y

    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt")
    ax.text(
        (x0 + x1) / 2,
        y + 0.025 * fov_um_y,
        f"{int(bar_um)} µm",
        color=color,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )


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

def _show_spatial_with_arrow_um_minimal(
    ax,
    img,
    fov_um_x,
    fov_um_y,
    arrow_start_um,
    arrow_vec_um,
    bin_start_s,
    speed_um_per_s,
    angle_deg,
    cmap=CYAN_TO_RED,
    panel_label=None,
):
    """
    Minimal spatial panel.
    Shows only:
      - image
      - arrow
      - 200 µm scale bar
      - small text: t, speed, angle
    """
    extent = [0, float(fov_um_x), 0, float(fov_um_y)]

    ax.imshow(img, origin="lower", cmap=cmap, extent=extent, aspect="equal")

    sx, sy = float(arrow_start_um[0]), float(arrow_start_um[1])
    vx, vy = float(arrow_vec_um[0]), float(arrow_vec_um[1])

    if np.isfinite(vx) and np.isfinite(vy) and (abs(vx) + abs(vy)) > 1e-6:
        ax.arrow(
            sx, sy, vx, vy,
            length_includes_head=True,
            head_width=0.03 * max(fov_um_x, fov_um_y),
            head_length=0.04 * max(fov_um_x, fov_um_y),
            linewidth=2.2,
            color="white",
        )

    _draw_scale_bar_right(
        ax,
        fov_um_x=fov_um_x,
        fov_um_y=fov_um_y,
        bar_um=200.0,
        color="white",
        lw=4,
        fontsize=10,
    )

    info = f"t = {bin_start_s:.2f} s\nspeed = {speed_um_per_s:.1f} µm/s\nangle = {angle_deg:.1f}°"
    ax.text(
        0.03, 0.97, info,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=3),
    )

    if panel_label is not None:
        ax.text(
            0.02, 1.02, panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="black",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_combined_propagation_figure(
    folder_name,
    selected_bins,
    prefix="r0p7_",
    fps=None,
    save_name=None,
):
    """
    Build one figure with:
      - 4 manually selected propagation bins in a 2x2 layout
      - 1 regression panel
      - 1 shared colorbar on the far right
    """
    if len(selected_bins) != 4:
        raise ValueError("selected_bins must contain exactly 4 bin indices.")

    if fps is None:
        fps = utils.get_fps_from_notes(folder_name)

    config = SpatialHeatmapConfig(
        folder_name,
        metric="event_rate",
        prefix=prefix,
        fps=fps,
        z_enter=3.5,
        z_exit=1.5,
        min_sep_s=0.3,
    )
    data = _load_suite2p_data(config)

    zoom = utils.get_zoom_from_notes(folder_name)
    zoom = float(zoom) if zoom else 1.0
    fov_um_x = 3080.90169 / zoom
    fov_um_y = 3560.14057 / zoom

    weights = [2.3662, 1.0454, 1.1252, 0.2987]
    sd_mu = [4.079, 11.24, 41.178]
    sd_sd = [1.146, 4.214, 37.065]
    thres = 0.15
    bias = float(
        weights[0]
        - (weights[1] * sd_mu[0] / sd_sd[0])
        - (weights[2] * sd_mu[1] / sd_sd[1])
        - (weights[3] * sd_mu[2] / sd_sd[2])
    )

    roi_scores_path = os.path.join(folder_name, "roi_scores.npy")
    if os.path.exists(roi_scores_path):
        scores = np.load(roi_scores_path)
    else:
        scores = compute_cell_scores(
            data,
            config,
            w_er=weights[1],
            w_pz=weights[2],
            w_area=weights[3],
            scale_er=float(sd_sd[0]),
            scale_pz=float(sd_sd[1]),
            scale_area=float(sd_sd[2]),
            bias=bias,
        )

    cell_mask = soft_cell_mask(scores, score_threshold=thres, top_k_pct=None)
    idx_keep = np.where(cell_mask)[0]
    stat_filtered = [data["stat"][i] for i in idx_keep]

    onsets = _event_onsets_by_roi(data, config, t_slice=None)
    onsets = [onsets[i] for i in idx_keep]

    bin_sec = 0.5
    edges = _bin_edges_and_indexer(data["T"], config.fps, bin_sec)
    A, first_time = _activation_matrix(onsets, edges)

    prop_csv = os.path.join(config.root, f"{prefix}coactivation_propagation.csv")
    if not os.path.exists(prop_csv):
        raise FileNotFoundError(
            f"Missing propagation CSV: {prop_csv}\nRun coactivation_order_heatmaps first."
        )
    prop_df = pd.read_csv(prop_csv).set_index("bin_index")

    # layout
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[1, 1, 0.06],
        height_ratios=[1, 1, 0.9],
        wspace=0.12,
        hspace=0.18,
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, 0:2])
    cax = fig.add_subplot(gs[0:2, 2])

    axes = [axA, axB, axC, axD]
    panel_labels = ["A", "B", "C", "D"]

    last_im = None

    for ax, b, label in zip(axes, selected_bins, panel_labels):
        if b not in prop_df.index:
            raise ValueError(f"Bin {b} not found in propagation CSV.")

        order_rank_filtered = _order_map_for_bin(first_time[:, b], A[:, b])
        spatial_order = _paint_order_map(
            order_rank_filtered,
            stat_filtered,
            data["Ly"],
            data["Lx"]
        )

        row = prop_df.loc[b]
        arrow_start_um = np.array([row["start_x_um"], row["start_y_um"]], dtype=float)
        arrow_vec_um = np.array([row["dx_um"], row["dy_um"]], dtype=float)

        extent = [0, float(fov_um_x), 0, float(fov_um_y)]
        last_im = ax.imshow(
            spatial_order,
            origin="lower",
            cmap=CYAN_TO_RED,
            extent=extent,
            aspect="equal",
            vmin=0,
            vmax=1,
        )

        sx, sy = float(arrow_start_um[0]), float(arrow_start_um[1])
        vx, vy = float(arrow_vec_um[0]), float(arrow_vec_um[1])

        if np.isfinite(vx) and np.isfinite(vy) and (abs(vx) + abs(vy)) > 1e-6:
            ax.arrow(
                sx, sy, vx, vy,
                length_includes_head=True,
                head_width=0.03 * max(fov_um_x, fov_um_y),
                head_length=0.04 * max(fov_um_x, fov_um_y),
                linewidth=2.2,
                color="white",
            )

        _draw_scale_bar_right(
            ax,
            fov_um_x=fov_um_x,
            fov_um_y=fov_um_y,
            bar_um=200.0,
            color="white",
            lw=4,
            fontsize=10,
        )

        info = (
            f"time = {float(row['first_onset_s']):.2f} s\n"
            f"speed = {float(row['speed_um_per_s']):.1f} µm/s\n"
            f"angle = {float(row['angle_deg']):.1f}°"
        )
        ax.text(
            0.03, 0.97,
            info,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=3),
        )

        ax.text(
            0.02, 1.02, label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="black",
        )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # regression panel
    x = prop_df["bin_start_s"].to_numpy()
    y = prop_df["angle_deg"].to_numpy()

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r2 = r_value ** 2

    axE.scatter(x, y, alpha=0.7, s=30)
    x_fit = np.linspace(np.min(x), np.max(x), 200)
    y_fit = slope * x_fit + intercept
    axE.plot(x_fit, y_fit, linewidth=2)

    axE.set_xlabel("Event time (s)")
    axE.set_ylabel("Propagation angle (deg)")

    axE.text(
        0.02, 1.02, "E",
        transform=axE.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="black",
    )

    axE.text(
        0.03, 0.97,
        f"$R^2$ = {r2:.3f}\nslope = {slope:.3f}\np = {p_value:.3g}",
        transform=axE.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3),
    )

    # shared gradient bar on right
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Relative recruitment order", rotation=90)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Early", "Late"])

    if save_name is None:
        bin_tag = "_".join(str(int(b)) for b in selected_bins)
        save_name = os.path.join(
            config.root,
            f"{prefix}combined_propagation_figure_bins_{bin_tag}.png"
        )

    fig.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved combined figure to: {save_name}")

# example use
if __name__ == "__main__":
    root = r"F:\data\2p_shifted\Cx\2024-07-01_00018"
    make_combined_propagation_figure(
        folder_name=root,
        selected_bins=[623, 807, 909, 1130],   # replace with your 4 chosen bins
        prefix="r0p7_",
    )