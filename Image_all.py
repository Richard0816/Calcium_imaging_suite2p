import os, math
import numpy as np
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
root = r'D:\data\2p_shifted\2024-11-20_00004\suite2p\plane0\\'
sample_name = root.split("\\")[-5]
fps = 30.0
prefix = 'r0p7_'     # matches your processing run
plot_seconds = None  # None = full; or e.g. 300 for first 5 min
time_cols_target = 1200  # target width for heatmaps (auto downsample)

# Event detection (on derivative)
z_enter = 3.5
z_exit  = 1.5
min_separation_s = 0.1    # merge events within 100 ms for counts

# Small multiples
top_k = 96                 # how many ROIs to show per page
grid_rows, grid_cols = 12, 8  # 12x8 = 96
# ------------------------------------------

# ---- Load Suite2p shapes & memmaps (time-major) ----
F = np.load(os.path.join(root, 'F.npy'), allow_pickle=True)
if F.shape[0] < F.shape[1]:  # nROIs x T
    nROIs, T = F.shape
else:                        # T x nROIs
    T, nROIs = F.shape

dff = np.memmap(os.path.join(root, f'{prefix}dff.memmap.float32'), dtype='float32', mode='r', shape=(T, nROIs))
low = np.memmap(os.path.join(root, f'{prefix}dff_lowpass.memmap.float32'), dtype='float32', mode='r', shape=(T, nROIs))
dt  = np.memmap(os.path.join(root, f'{prefix}dff_dt.memmap.float32'), dtype='float32', mode='r', shape=(T, nROIs))

# Optional time crop
if plot_seconds is not None:
    Tcrop = min(T, int(plot_seconds * fps))
    t_slice = slice(0, Tcrop)
else:
    Tcrop = T
    t_slice = slice(None)

# Downsample factor for heatmaps
ds = max(1, Tcrop // time_cols_target)
cols = Tcrop // ds

# ----------- helper: robust z + hysteresis -----------
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

# ----------- Build global summaries (streaming) -----------
# 1) activity metric per ROI (event count) and
# 2) heatmap-friendly normalized arrays (uint8) for low-pass dF/F and binary events

heat = np.zeros((nROIs, cols), dtype=np.uint8)     # for low-pass dF/F (0-255 after robust norm)
erast = np.zeros((nROIs, cols), dtype=np.uint8)    # binary event raster
evt_counts = np.zeros(nROIs, dtype=int)

for j in range(nROIs):
    # pull ROI j slice (Tcrop,) as np array (memmap views are cheap)
    lp = np.asarray(low[t_slice, j], dtype=np.float32)
    dd = np.asarray(dt[t_slice, j], dtype=np.float32)

    # robust z on derivative
    z, med, mad = mad_z(dd)
    onsets = hysteresis_onsets(z, z_enter, z_exit, fps, min_sep_s=min_separation_s)
    evt_counts[j] = onsets.size

    # downsample in time
    if ds > 1:
        # reshape to (cols, ds) then mean
        trimmed = lp[:cols*ds].reshape(cols, ds)
        lp_ds = trimmed.mean(axis=1)
        # event raster: mark any detected onset bin as 1
        er = np.zeros(cols, dtype=np.uint8)
        if onsets.size:
            bins = (onsets // ds).clip(0, cols-1)
            er[np.unique(bins)] = 1
    else:
        lp_ds = lp
        er = (np.isin(np.arange(lp.size), onsets)).astype(np.uint8)

    # robust scaling for heatmap (per ROI)
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
order = np.argsort(-evt_counts)  # descending by event count
heat_sorted  = heat[order]
erast_sorted = erast[order]

# ------------- Plot: Global heatmap -----------------
plt.figure(figsize=(14, 10))
plt.imshow(heat_sorted, aspect='auto', interpolation='nearest')
plt.title(f'Low-pass ΔF/F (sorted by event count)  N={nROIs}, width~{cols} bins (~{cols*ds/fps:.1f}s), sample ({sample_name})')
plt.xlabel('Time (downsampled bins)')
plt.ylabel('ROIs (most active at top)')
# optional colourbar in 0-1 scale
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
# we’ll plot low-pass ΔF/F as lines, K per page
time = (np.arange(Tcrop) / fps)
pages = math.ceil(min(nROIs, 5*top_k) / top_k)  # cap to first 5 pages by default; tweak as you like
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
        # robust y-lims per panel for readability
        lo, hi = np.percentile(y, [1, 99])
        ax.plot(time, y, linewidth=0.8)
        ax.set_ylim(lo, hi if hi > lo else lo + 1e-3)
        ax.set_title(f'ROI {roi} (#{start+k+1})  events={evt_counts[roi]}', fontsize=9)
        ax.grid(True, alpha=0.15)
    # hide unused panels
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
