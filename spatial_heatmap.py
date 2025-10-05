import os
import numpy as np
import matplotlib.pyplot as plt
import utils
from typing import Union

# ---- Cell masking ----
def _safe_div(x, d):
    d = float(d) if d else 1.0
    return x / d

def compute_cell_scores(data, config,
                        w_er=1.0, w_pz=1.0, w_area=0.5,
                        scale_er=1.0, scale_pz=3.0, scale_area=50.0,
                        bias=-2.0,
                        t_slice=None):
    """
    Returns an array (N,) of cell probabilities for each ROI.
    """
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
    return scores

def soft_cell_mask(scores, score_threshold=0.5, top_k_pct=None):
    """
    Convert probabilities into a boolean mask.
    If top_k_pct is set (e.g., 20 for top 20%), it overrides score_threshold.
    """
    if top_k_pct is not None:
        k = max(1, int(np.ceil(scores.size * (top_k_pct / 100.0))))
        thresh = np.partition(scores, -k)[-k]  # kth largest as cutoff
        return scores >= thresh
    return scores >= score_threshold

# ---- Helpers ----
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
    plt.scatter(xs, ys, s=4, c='white', alpha=0.35, linewidths=0)
    plt.colorbar(im, label=title)
    plt.title(title)
    plt.xlabel(xlabel);
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
                 fps=30.0, z_enter=3.5, z_exit=1.5, min_sep_s=0.3, bin_seconds=None):
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

    # Generate output path and title
    if bin_index is None:
        out = os.path.join(config.root, f'{config.prefix}spatial_{config.metric}.png')
        title = config.get_metric_title()
    else:
        out = os.path.join(config.root, f'{config.prefix}spatial_{config.metric}_bin{bin_index:03d}.png')
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

        # Masked metric
        vals_masked = np.where(mask, vals, np.nan)
        spatial_masked = utils.paint_spatial(vals_masked, data['stat'], data['Ly'], data['Lx'])
        show_spatial(spatial_masked, title + " (prob-masked)", data['Lx'], data['Ly'], data['stat'],
                     pix_to_um=data['pix_to_um'], cmap='magma', outpath=out + '_probmask.png')

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
                        fps=30.0, z_enter=3.5, z_exit=1.5, min_sep_s=0.3, bin_seconds=None,
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

run_spatial_heatmap(
    r'F:\data\2p_shifted\2024-07-01_00018',
    metric='event_rate',
    fps=30.0, z_enter=3.5, z_exit=1.5, min_sep_s=0.3,
    # scoring: emphasize peak_dz slightly, normalize by typical ranges
    w_er=0.7934, w_pz=2.1061, w_area=0.255,
    scale_er=1.146,      # ~1 event/min considered “unit”
    scale_pz=4.214,      # z≈5 as a “unit bump”
    scale_area=37.065,   # 10 px as a “unit area”
    bias=-6.955536301665436,            # stricter overall
    score_threshold=0.5,  # classify as cell if P>=0.5
    top_k_pct=None        # or set e.g. 25 for top-25% only
)
