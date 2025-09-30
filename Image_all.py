import os, math
import numpy as np
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
root = r'D:\data\2p_shifted\2024-11-20_00004\suite2p\plane0\\'   # Path to a single Suite2p plane folder
sample_name = root.split("\\")[-5]                                # Human-readable sample name from path
fps = 30.0                                                        # Imaging frame rate (Hz)
prefix = 'r0p7_'     # matches your processing run (prefix used by your saved memmaps)
plot_seconds = None  # None = full recording; or e.g., 300 for first 5 minutes
time_cols_target = 1200  # target width (columns) for heatmaps; code downsamples time to ~this many bins

# Event detection (on derivative)
z_enter = 3.5        # robust z threshold for entering an “event”
z_exit  = 1.5        # lower threshold to exit (hysteresis)
min_separation_s = 0.1    # merge onsets closer than this (sec) into a single event for counts

# Small multiples
top_k = 96                 # number of ROIs to plot per page (line plots)
grid_rows, grid_cols = 12, 8  # 12x8 = 96 panels
# ------------------------------------------

# ---- Load Suite2p shapes & memmaps (time-major) ----
F = np.load(os.path.join(root, 'F.npy'), allow_pickle=True)
if F.shape[0] < F.shape[1]:  # nROIs x T layout
    nROIs, T = F.shape
else:                        # T x nROIs layout
    T, nROIs = F.shape

# Low-pass ΔF/F and derivative (memmaps saved earlier; shape forced to (T, N))
dff = np.memmap(os.path.join(root, f'{prefix}dff.memmap.float32'), dtype='float32', mode='r', shape=(T, nROIs))
low = np.memmap(os.path.join(root, f'{prefix}dff_lowpass.memmap.float32'), dtype='float32', mode='r', shape=(T, nROIs))
dt  = np.memmap(os.path.join(root, f'{prefix}dff_dt.memmap.float32'), dtype='float32', mode='r', shape=(T, nROIs))

# Optional time crop (e.g., to first N seconds)
if plot_seconds is not None:
    Tcrop = min(T, int(plot_seconds * fps))
    t_slice = slice(0, Tcrop)
else:
    Tcrop = T
    t_slice = slice(None)

# Downsample factor to reach ~time_cols_target columns in heatmaps
ds = max(1, Tcrop // time_cols_target)
cols = Tcrop // ds

# ----------- helper: robust z + hysteresis -----------
def mad_z(x):
    """
    Robust z-score using MAD (per ROI), with 1.4826 factor to approximate σ for normal data.
    Returns z, and the median/MAD for optional inverse transforms.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad), med, mad

def hysteresis_onsets(z, z_hi, z_lo, fps, min_sep_s=0.0):
    """
    Detect onset indices with hysteresis:
      - Start when z >= z_hi and not active
      - Stop when z <= z_lo
    Optionally merge onsets that are < min_sep_s apart.
    """
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

# ----------- Build global summaries (streaming) -----------
# Allocate outputs:
#   heat: robust-scaled low-pass ΔF/F per ROI (downsampled in time) → 0..255 uint8
#   erast: binary event raster per ROI/time-bin (0/1)
#   evt_counts: number of events per ROI (after hysteresis & merging)
heat = np.zeros((nROIs, cols), dtype=np.uint8)
erast = np.zeros((nROIs, cols), dtype=np.uint8)
evt_counts = np.zeros(nROIs, dtype=int)

for j in range(nROIs):
    # Pull ROI j slices (views; cheap even for memmaps)
    lp = np.asarray(low[t_slice, j], dtype=np.float32)
    dd = np.asarray(dt[t_slice, j], dtype=np.float32)

    # Robust z on derivative + hysteresis event detection
    z, med, mad = mad_z(dd)
    onsets = hysteresis_onsets(z, z_enter, z_exit, fps, min_sep_s=min_separation_s)
    evt_counts[j] = onsets.size

    # Downsample in time by mean; event bins flagged if any onset falls inside the bin
    if ds > 1:
        trimmed = lp[:cols*ds].reshape(cols, ds)
        lp_ds = trimmed.mean(axis=1)
        er = np.zeros(cols, dtype=np.uint8)
        if onsets.size:
            bins = (onsets // ds).clip(0, cols-1)
            er[np.unique(bins)] = 1
    else:
        lp_ds = lp
        er = (np.isin(np.arange(lp.size), onsets)).astype(np.uint8)

    # Per-ROI robust scaling (1–99th percentiles) to 0..255 for heatmap
    lo = np.percentile(lp_ds, 1)
    hi = np.percentile(lp_ds, 99)
    if hi <= lo:
        scaled = np.zeros_like(lp_ds, dtype=np.uint8)
    else:
        x = np.clip((lp_ds - lo) / (hi - lo), 0, 1)
        scaled = (x * 255.0 + 0.5).astype(np.uint8)

    heat[j, :]  = scaled
    erast[j, :] = er

print("Summaries built: heatmap matrix =", heat.shape, "event raster =", erast.shape)

# ------------- Sort ROIs by activity ----------------
order = np.argsort(-evt_counts)  # sort by descending event counts
heat_sorted  = heat[order]
erast_sorted = erast[order]

# ------------- Plot: Global heatmap -----------------
plt.figure(figsize=(14, 10))
plt.imshow(heat_sorted, aspect='auto', interpolation='nearest')
plt.title(f'Low-pass ΔF/F (sorted by event count)  N={nROIs}, width~{cols} bins (~{cols*ds/fps:.1f}s), sample ({sample_name})')
plt.xlabel('Time (downsampled bins)')
plt.ylabel('ROIs (most active at top)')
# Colourbar expresses robust 1–99% scaling (relative)
cbar = plt.colorbar()
cbar.set_label('Relative intensity (robust 1–99% scaled)')
plt.tight_layout()
plt.savefig(os.path.join(root, f'{prefix}overview_heatmap.png'), dpi=200)
plt.close()

# ------------- Plot: Event raster -------------------
plt.figure(figsize=(14, 10))
plt.imshow(erast_sorted, aspect='auto', interpolation='nearest', cmap='Greys')
plt.title(f'Event raster (hysteresis z_enter={z_enter}, z_exit={z_exit})  N={nROIs}, sample ({sample_name})')
plt.xlabel('Time (downsampled bins)')
plt.ylabel('ROIs (most active at top)')
plt.tight_layout()
plt.savefig(os.path.join(root, f'{prefix}event_raster.png'), dpi=200)
plt.close()

# ------------- Small multiples for top-K (paged) -------------
# Plot ΔF/F low-pass lines for the most active ROIs, K per page (first up to ~5 pages)
time = (np.arange(Tcrop) / fps)
pages = math.ceil(min(nROIs, 5*top_k) / top_k)  # cap to first 5 pages to keep file count manageable
for p in range(pages):
    start = p * top_k
    end   = min(nROIs, start + top_k)
    if start >= end: break
    ids = order[start:end]

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, 18), sharex=True)
    axes = np.array(axes).reshape(-1)
    for k, roi in enumerate(ids):
        ax = axes[k]
        y = np.asarray(low[t_slice, roi], dtype=np.float32)
        # Robust per-panel y-limits for readability across heterogeneous ROIs
        lo, hi = np.percentile(y, [1, 99])
        ax.plot(time, y, linewidth=0.8)
        ax.set_ylim(lo, hi if hi > lo else lo + 1e-3)
        ax.set_title(f'ROI {roi} (#{start+k+1})  events={evt_counts[roi]}', fontsize=9)
        ax.grid(True, alpha=0.15)
    # Hide unused panels on last page
    for k in range(end-start, grid_rows*grid_cols):
        axes[k].axis('off')

    fig.suptitle(f'Low-pass ΔF/F small multiples — page {p+1}/{pages} (ROIs {start}–{end-1})', fontsize=14)
    fig.text(0.5, 0.04, 'Time (s)', ha='center')
    fig.text(0.06, 0.5, 'ΔF/F (robust scale)', va='center', rotation='vertical')
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.97])
    out = os.path.join(root, f'{prefix}small_multiples_p{p+1}.png')
    plt.savefig(out, dpi=160)
    plt.close()
    print("Saved", out)

print("Saved:",
      os.path.join(root, f'{prefix}overview_heatmap.png'),
      os.path.join(root, f'{prefix}event_raster.png'),
      "and small-multiples pages.")
