import os
import numpy as np
import matplotlib.pyplot as plt
import utils

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
        self.sample_name = folder_name.split("\\")[-4] if "\\" in folder_name else folder_name.split("/")[-4]

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


def _compute_and_save_spatial_map(data, config, t_slice=None, bin_index=None):
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
                        fps=30.0, z_enter=3.5, z_exit=1.5, min_sep_s=0.3, bin_seconds=None):
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

    # Generate whole-recording map
    _compute_and_save_spatial_map(data, config)

    # Generate optional time-binned maps
    if config.bin_seconds is not None and config.bin_seconds > 0:
        _generate_time_binned_maps(data, config)

run_spatial_heatmap('D:\\data\\2p_shifted\\2024-08-20_00002\\', 'event_rate')