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
    img = np.zeros((Ly, Lx), dtype=np.float32)
    w = np.zeros((Ly, Lx), dtype=np.float32)
    for j, s in enumerate(stat_list):
        v = values_per_roi[j]
        ypix = s['ypix']
        xpix = s['xpix']
        lam = s['lam'].astype(np.float32)
        img[ypix, xpix] += v * lam
        w[ypix, xpix] += lam
    m = w > 0
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

def run_spatial_heatmap(folder_name, metric='event_rate', prefix='r0p7_', fps=30.0, z_enter=3.5, z_exit=1.5, min_sep_s=0.3, bin_seconds=None):
    """
    :param folder_name: Folder to run heatmap on
    :param metric: Metric to display on the spatial heatmap (one scalar per ROI)
        options: 'event_rate', 'mean_dff', 'peak_dz'
    :param prefix: Must match your preprocessed memmap filename prefix
    :param fps: Frame rate (Hz)
    :param z_enter: Event detection thresholds
    :param z_exit: Event detection thresholds
    :param min_sep_s: # Merge onsets that are < n seconds apart
    :param bin_seconds: make per-time-bin maps (in seconds). Set None to skip binning.
        # e.g., 60 for per-minute maps; or None for whole recording
    :return: None
    """
    # ------------- CONFIG -------------
    root = os.path.join(folder_name, "suite2p\\plane0\\")  # Path to a single Suite2p plane folder
    sample_name = root.split("\\")[-4]  # Human-readable sample name from path

    # ---- Load Suite2p metadata ----
    ops = np.load(os.path.join(root, 'ops.npy'), allow_pickle=True).item()
    stat = np.load(os.path.join(root, 'stat.npy'), allow_pickle=True)

    Ly, Lx = ops['Ly'], ops['Lx']  # Image dimensions
    pix_to_um = ops.get('pix_to_um', None)  # Pixel→µm scale if available

    # ---- Load processed signals (time-major T x N) ----
    # We only need low-pass ΔF/F and derivative for the supported metrics
    low = np.memmap(os.path.join(root, f'{prefix}dff_lowpass.memmap.float32'),
                    dtype='float32', mode='r')
    dt = np.memmap(os.path.join(root, f'{prefix}dff_dt.memmap.float32'),
                   dtype='float32', mode='r')

    # Infer T (frames) and N (ROIs) from file sizes; reshape memmaps to (T, N)
    N = len(stat)
    T = low.size // N
    low = low.reshape(T, N)
    dt = dt.reshape(T, N)

    # ---- Whole-recording map ----
    vals = roi_metric({'low': low, 'dt': dt}, which=metric, t_slice=slice(None),
                      fps=fps, z_enter=z_enter, z_exit=z_exit, min_sep_s=min_sep_s)
    spatial = paint_spatial(vals, stat, Ly, Lx)

    ttl = {
        'event_rate': f'Event rate (events/min) — z_enter={z_enter}, z_exit={z_exit} ({sample_name})',
        'mean_dff': f'Mean ΔF/F (low-pass) ({sample_name})',
        'peak_dz': f'Peak derivative z (robust) ({sample_name})'
    }[metric]
    out = os.path.join(root, f'{prefix}spatial_{metric}.png')
    show_spatial(spatial, ttl, Lx, Ly, stat, pix_to_um=pix_to_um, cmap='magma', outpath=out)

    # ---- Optional: time-binned maps (e.g., per minute) ----
    if bin_seconds is not None and bin_seconds > 0:
        Tbin = int(bin_seconds * fps)
        n_bins = int(np.ceil(T / Tbin))
        for b in range(n_bins):
            t0 = b * Tbin
            t1 = min(T, (b + 1) * Tbin)
            # Skip tiny tail windows to avoid noisy/empty maps
            if t1 - t0 < max(5, int(0.2 * Tbin)):
                continue
            vals_b = roi_metric({'low': low, 'dt': dt}, which=metric, t_slice=slice(None),
                      fps=fps, z_enter=z_enter, z_exit=z_exit, min_sep_s=min_sep_s)
            spatial_b = paint_spatial(vals_b, stat, Ly, Lx)
            out_b = os.path.join(root, f'{prefix}spatial_{metric}_bin{b + 1:03d}.png')
            title_b = f'{ttl}\nWindow {b + 1}: {t0 / fps:.1f}–{t1 / fps:.1f} s'
            show_spatial(spatial_b, title_b, Lx, Ly, stat, pix_to_um=pix_to_um, cmap='magma', outpath=out_b)

run_spatial_heatmap('D:\\data\\2p_shifted\\2024-07-01_00018\\', 'event_rate')