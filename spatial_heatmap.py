import os
import numpy as np
import matplotlib.pyplot as plt

# ------------- CONFIG -------------
root   = r'D:\data\2p_shifted\2024-11-05_00007\suite2p\plane0\\'
prefix = 'r0p7_'   # must match your processed files
fps    = 30.0

# Metric to display on the spatial heatmap:
metric = 'event_rate'   # options: 'event_rate', 'mean_dff', 'peak_dz'

# Event detection thresholds (used if metric involves events)
z_enter = 3.5
z_exit  = 1.5
min_sep_s = 0.3  # merge onsets within 300 ms

# Optional: make per-time-bin maps (in seconds). Set None to skip.
bin_seconds = None  # e.g., 60 for per-minute maps; or None for whole recording
# ----------------------------------

# ---- Load Suite2p metadata ----
ops  = np.load(os.path.join(root, 'ops.npy'), allow_pickle=True).item()
stat = np.load(os.path.join(root, 'stat.npy'), allow_pickle=True)

Ly, Lx = ops['Ly'], ops['Lx']
pix_to_um = ops.get('pix_to_um', None)  # might be None

# ---- Load processed signals (time-major T x N) ----
# We only need low-pass ΔF/F and d/dt for metrics
low = np.memmap(os.path.join(root, f'{prefix}dff_lowpass.memmap.float32'),
                dtype='float32', mode='r')
dt  = np.memmap(os.path.join(root, f'{prefix}dff_dt.memmap.float32'),
                dtype='float32', mode='r')

# infer T and N robustly from stat length
N = len(stat)
T = low.size // N
low = low.reshape(T, N)
dt  = dt.reshape(T, N)

# ---- Helpers ----
def mad_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad), med, mad

def hysteresis_onsets(z, z_hi, z_lo, fps, min_sep_s=0.0):
    above_hi = z >= z_hi
    onsets = []
    active = False
    for i in range(z.size):
        if not active and above_hi[i]:
            active = True
            onsets.append(i)
        elif active and z[i] <= z_lo:
            active = False
    if not onsets:
        return np.array([], dtype=int)
    onsets = np.array(onsets, dtype=int)
    if min_sep_s > 0:
        min_sep = int(min_sep_s * fps)
        merged = [onsets[0]]
        for k in onsets[1:]:
            if k - merged[-1] >= min_sep:
                merged.append(k)
        onsets = np.asarray(merged, dtype=int)
    return onsets

def roi_metric(values, which='event_rate', t_slice=slice(None)):
    """
    Compute one scalar per ROI for a chosen metric.
    values: dict with arrays per ROI/time (expects 'low' and 'dt').
    which: 'event_rate', 'mean_dff', 'peak_dz'
    """
    lp = values['low'][t_slice]  # (Tsel, N)
    dd = values['dt'][t_slice]   # (Tsel, N)
    Tsel = lp.shape[0]
    out = np.zeros(lp.shape[1], dtype=np.float32)

    if which == 'mean_dff':
        out = np.nanmean(lp, axis=0).astype(np.float32)

    elif which == 'peak_dz':
        # peak robust z of derivative per ROI
        z = np.empty_like(dd, dtype=np.float32)
        for j in range(dd.shape[1]):
            zj, _, _ = mad_z(dd[:, j])
            z[:, j] = zj
        out = np.nanmax(z, axis=0).astype(np.float32)

    elif which == 'event_rate':
        counts = np.zeros(dd.shape[1], dtype=np.int32)
        for j in range(dd.shape[1]):
            zj, _, _ = mad_z(dd[:, j])
            on = hysteresis_onsets(zj, z_enter, z_exit, fps, min_sep_s=min_sep_s)
            counts[j] = on.size
        duration_min = Tsel / fps / 60.0
        out = (counts / max(duration_min, 1e-9)).astype(np.float32)  # events per minute
    else:
        raise ValueError("metric must be one of: 'event_rate', 'mean_dff', 'peak_dz'")

    return out

def paint_spatial(values_per_roi, stat_list, Ly, Lx):
    """
    Paint per-ROI scalar values onto the imaging plane using ROI masks.
    Weighted by 'lam' (pixel membership). Returns (Ly, Lx) float32 image.
    """
    img = np.zeros((Ly, Lx), dtype=np.float32)
    w   = np.zeros((Ly, Lx), dtype=np.float32)
    for j, s in enumerate(stat_list):
        v = values_per_roi[j]
        ypix = s['ypix']
        xpix = s['xpix']
        lam  = s['lam'].astype(np.float32)
        # accumulate
        img[ypix, xpix] += v * lam
        w[ypix, xpix]   += lam
    # normalize where we have weight
    m = w > 0
    img[m] /= w[m]
    return img

def show_spatial(img, title, pix_to_um=None, cmap='magma', outpath=None):
    extent = None
    xlabel, ylabel = 'X (pixels)', 'Y (pixels)'
    if pix_to_um is not None:
        extent = [0, Lx*pix_to_um, 0, Ly*pix_to_um]
        xlabel, ylabel = 'X (µm)', 'Y (µm)'

    plt.figure(figsize=(8, 7))
    im = plt.imshow(img, origin='lower', cmap=cmap, extent=extent, aspect='equal')
    xs = [np.median(s['xpix']) for s in stat]
    ys = [np.median(s['ypix']) for s in stat]
    if pix_to_um is not None:
        xs = np.array(xs) * pix_to_um
        ys = np.array(ys) * pix_to_um
    plt.scatter(xs, ys, s=4, c='white', alpha=0.35, linewidths=0)
    plt.colorbar(im, label=title)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
        print("Saved", outpath)
    else:
        plt.show()

# ---- Whole-recording map ----
vals = roi_metric({'low': low, 'dt': dt}, which=metric, t_slice=slice(None))
spatial = paint_spatial(vals, stat, Ly, Lx)

ttl = {
    'event_rate': f'Event rate (events/min) — z_enter={z_enter}, z_exit={z_exit}',
    'mean_dff':   'Mean ΔF/F (low-pass)',
    'peak_dz':    'Peak derivative z (robust)'
}[metric]
out = os.path.join(root, f'{prefix}spatial_{metric}.png')
show_spatial(spatial, ttl, pix_to_um=pix_to_um, cmap='magma', outpath=out)

# ---- Optional: time-binned maps (e.g., per minute) ----
if bin_seconds is not None and bin_seconds > 0:
    Tbin = int(bin_seconds * fps)
    n_bins = int(np.ceil(T / Tbin))
    for b in range(n_bins):
        t0 = b * Tbin
        t1 = min(T, (b+1) * Tbin)
        if t1 - t0 < max(5, int(0.2 * Tbin)):  # skip tiny tail
            continue
        vals_b = roi_metric({'low': low[t0:t1], 'dt': dt[t0:t1]},
                            which=metric, t_slice=slice(None))
        spatial_b = paint_spatial(vals_b, stat, Ly, Lx)
        out_b = os.path.join(root, f'{prefix}spatial_{metric}_bin{b+1:03d}.png')
        title_b = f'{ttl}\nWindow {b+1}: {t0/fps:.1f}–{t1/fps:.1f} s'
        show_spatial(spatial_b, title_b, pix_to_um=pix_to_um, cmap='magma', outpath=out_b)
