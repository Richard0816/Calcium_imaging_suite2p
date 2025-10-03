import os
import numpy as np
import matplotlib.pyplot as plt
import utils

# ---- Helpers ----
def roi_metric(values, which='event_rate', t_slice=slice(None), fps=30.0, z_enter=3.5, z_exit=1.5, min_sep_s=0.3):
    """
    Computes a region of interest (ROI) metric based on the specified type.

    This function processes time-series data to compute metrics such as 
    event rates, mean delta F/F (dff), or peak robust z-scores for the provided
    regions of interest. The chosen metric depends on the `which`"""
    
    # Extract the low-pass filtered data (delta F/F) for the selected time slice
    # Shape: (Tsel, N) where Tsel = number of time frames, N = number of ROIs
    lp = values['low'][t_slice]  # (Tsel, N)
    
    # Extract the detrended data for the selected time slice
    # This is used for z-score calculations and event detection
    dd = values['dt'][t_slice]  # (Tsel, N)
    
    # Get the number of time frames in the selected slice
    Tsel = lp.shape[0]
    
    # Initialize output array with zeros, one value per ROI
    # dtype=float32 for memory efficiency
    out = np.zeros(lp.shape[1], dtype=np.float32)
    
    # Branch 1: Calculate mean delta F/F across time for each ROI
    if which == 'mean_dff':
        # Compute mean across time axis (axis=0), ignoring NaN values
        # Convert to float32 for consistency
        out = np.nanmean(lp, axis=0).astype(np.float32)
    
    # Branch 2: Calculate peak robust z-score for each ROI
    elif which == 'peak_dz':
        # Peak robust z per ROI (loop over columns for per-ROI MAD)
        
        # Create empty array to store z-scores with same shape as detrended data
        z = np.empty_like(dd, dtype=np.float32)
        
        # Loop through each ROI (column)
        for j in range(dd.shape[1]):
            # Calculate robust z-score using MAD for this ROI's time series
            # mad_z returns (z-score, median, mad); we only need the z-score
            zj, _, _ = utils.mad_z(dd[:, j])
            
            # Store z-scores for this ROI in the z array
            z[:, j] = zj
        
        # Find maximum z-score across time for each ROI
        # This represents the peak activity level
        out = np.nanmax(z, axis=0).astype(np.float32)
    
    # Branch 3: Calculate event rate (events per minute) for each ROI
    elif which == 'event_rate':
        # Count onsets per ROI and divide by duration (min) → events/min
        
        # Initialize array to count number of events detected in each ROI
        counts = np.zeros(dd.shape[1], dtype=np.int32)
        
        # Loop through each ROI to detect and count events
        for j in range(dd.shape[1]):
            # Calculate robust z-score for this ROI's time series
            zj, _, _ = utils.mad_z(dd[:, j])
            
            # Detect event onsets using hysteresis thresholding
            # Events start when z >= z_enter and end when z <= z_exit
            # Returns array of frame indices where events begin
            on = utils.hysteresis_onsets(zj, z_enter, z_exit, fps, min_sep_s=min_sep_s)
            
            # Count the number of detected events (size of onset array)
            counts[j] = on.size
        
        # Convert total time frames to minutes: frames / (frames/sec) / (sec/min)
        duration_min = Tsel / fps / 60.0
        
        # Calculate event rate: events / minutes
        # Use max() to avoid division by zero (minimum denominator = 1e-9)
        out = (counts / max(duration_min, 1e-9)).astype(np.float32)
    else:
        raise ValueError("metric must be one of: 'event_rate', 'mean_dff', 'peak_dz'")

    return out

def paint_spatial(values_per_roi, stat_list, Ly, Lx):
    """
    Paint per-ROI scalar values onto the imaging plane using ROI masks.
    Uses 'lam' weights for soft assignment; normalizes by accumulated weight.
    Returns (Ly, Lx) float32 image.
    """
    # Initialize the output image array with zeros, shape matches imaging plane dimensions
    img = np.zeros((Ly, Lx), dtype=np.float32)

    # Initialize weight accumulator array to track total lambda weights per pixel
    w = np.zeros((Ly, Lx), dtype=np.float32)

    # Iterate through each ROI and its corresponding statistics dictionary
    for j, s in enumerate(stat_list):
        # Get the scalar value (metric) for the current ROI
        v = values_per_roi[j]

        # Extract y and x-coordinates of all pixels belonging to this ROI
        ypix = s['ypix']
        xpix = s['xpix']

        # Extract lambda weights (pixel-wise contribution strengths) and convert to float32
        lam = s['lam'].astype(np.float32)

        # Add weighted ROI value to image: each pixel gets v * its lambda weight
        img[ypix, xpix] += v * lam

        # Accumulate the lambda weights at each pixel (for normalization)
        w[ypix, xpix] += lam

    # Create boolean mask identifying pixels with non-zero accumulated weights
    m = w > 0

    # Normalize weighted values by dividing by accumulated weights (only where w > 0)
    img[m] /= w[m]

    return img

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

    vals = roi_metric(signals, which=config.metric, t_slice=time_slice,
                      fps=config.fps, z_enter=config.z_enter,
                      z_exit=config.z_exit, min_sep_s=config.min_sep_s)

    spatial = paint_spatial(vals, data['stat'], data['Ly'], data['Lx'])

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

run_spatial_heatmap('D:\\data\\2p_shifted\\2024-07-01_00018\\', 'event_rate')