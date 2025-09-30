import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os, imageio.v2 as imageio

# ---------------- CONFIG ----------------
root   = r'D:\data\2p_shifted\2024-11-05_00007\suite2p\plane0\\'
prefix = 'r0p7_'      # matches your processed files
fps    = 30.0

# Choose signal to drive the vectors: 'low' (ΔF/F low-pass) or 'dt' (derivative)
signal_type = 'dt'     # 'dt' is usually best for propagation

# Time binning for smoother vectors (in seconds)
bin_sec = 0.5          # set to 0 for frame-by-frame
# Neighborhood for vector computation
k_neighbors = 12       # number of nearest neighbors per ROI (try 8–20)
radius_px   = None     # or set a radius in pixels; if set, overrides k_neighbors

# Vector scaling for quiver
quiver_scale = 0.1     # larger → shorter arrows on plot (matplotlib quiver convention)
arrow_alpha  = 0.8

# Optionally subset time (in seconds) to keep figures manageable
t_start_s, t_end_s = 0, None
# ---------------------------------------

# ---- Load Suite2p metadata ----
ops  = np.load(os.path.join(root, 'ops.npy'), allow_pickle=True).item()
stat = np.load(os.path.join(root, 'stat.npy'), allow_pickle=True)
Ly, Lx = ops['Ly'], ops['Lx']
pix_to_um = ops.get('pix_to_um', None)

# ROI centroids (float arrays)
xy = np.column_stack([
    np.array([np.mean(s['xpix']) for s in stat], dtype=np.float32),
    np.array([np.mean(s['ypix']) for s in stat], dtype=np.float32)
])

# ---- Load signals (time-major T x N) ----
low = np.memmap(os.path.join(root, f'{prefix}dff_lowpass.memmap.float32'),
                dtype='float32', mode='r')
dt  = np.memmap(os.path.join(root, f'{prefix}dff_dt.memmap.float32'),
                dtype='float32', mode='r')

N = len(stat)
T = low.size // N
low = low.reshape(T, N)
dt  = dt.reshape(T, N)

# Pick which signal to use
X = dt if signal_type == 'dt' else low   # shape (T, N)

# Optional time cropping
t0 = int(fps * (t_start_s or 0))
t1 = T if t_end_s is None else min(T, int(fps * t_end_s))
X = X[t0:t1]
Tsel = X.shape[0]
time = np.arange(t0, t0+Tsel) / fps

# Temporal binning
if bin_sec and bin_sec > 0:
    B = int(max(1, round(bin_sec * fps)))
else:
    B = 1

nbins = int(np.ceil(Tsel / B))

# Build neighbor graph
tree = cKDTree(xy)
if radius_px is not None:
    # variable number of neighbors per ROI within radius
    neighbor_lists = tree.query_ball_point(xy, r=radius_px)
else:
    # fixed k nearest neighbors (excluding self)
    dists, idxs = tree.query(xy, k=k_neighbors+1)  # includes self at col 0
    neighbor_lists = [list(ids[1:]) for ids in idxs]  # drop self

# Helper: robust scale within each bin to avoid global amplitude bias
def robust_norm(v):
    lo, hi = np.percentile(v, (5, 95))
    if hi <= lo:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)

# Compute vectors per bin
out_dir = os.path.join(root, f'{prefix}vectors_{signal_type}')
os.makedirs(out_dir, exist_ok=True)

"""for b in range(nbins):
    a = b * B
    z = min(Tsel, (b+1) * B)
    Xb = X[a:z]                                  # (B, N) or (frames in last bin, N)

    # Activity change in this bin: compare mean of last half vs first half
    if Xb.shape[0] == 1:
        delta = Xb[0]
    else:
        h = Xb.shape[0] // 2
        delta = Xb[h:].mean(axis=0) - Xb[:h].mean(axis=0)  # (N,)

    # Optional: normalize deltas robustly to reduce hot-ROI bias
    dnorm = robust_norm(delta.astype(np.float32))

    # Build vectors at each ROI:
    # For ROI i, compare to neighbors j; accumulate vectors pointing toward neighbors with larger increase.
    U = np.zeros(N, dtype=np.float32)  # x-component
    V = np.zeros(N, dtype=np.float32)  # y-component
    for i in range(N):
        nbrs = neighbor_lists[i]
        if len(nbrs) == 0:
            continue
        vi = dnorm[i]
        xi, yi = xy[i]

        # differences to neighbors (only positive “pulls”)
        pulls = []
        for j in nbrs:
            w = dnorm[j] - vi
            if w > 0:  # neighbor is rising more
                dx = xy[j,0] - xi
                dy = xy[j,1] - yi
                norm = np.hypot(dx, dy) + 1e-6
                pulls.append((w, dx/norm, dy/norm))
        if pulls:
            wsum = sum(w for w,_,_ in pulls)
            ux = sum(w*cx for w,cx,_ in pulls) / (wsum + 1e-9)
            vy = sum(w*cy for w,_,cy in pulls) / (wsum + 1e-9)
            U[i] = ux
            V[i] = vy

    # Optional smoothing of vectors by averaging with neighbors (one pass)
    U_s = np.copy(U)
    V_s = np.copy(V)
    for i in range(N):
        nbrs = neighbor_lists[i]
        if not nbrs:
            continue
        U_s[i] = (U[i] + U[nbrs].mean()) / 2.0
        V_s[i] = (V[i] + V[nbrs].mean()) / 2.0

    # Plot quiver on top of an intensity background (mean low-pass in this bin)
    bg = low[t0+a:t0+z].mean(axis=0)  # (N,)
    # paint background to image (quick centroid scatter approximation)
    bg_img = np.zeros((Ly, Lx), dtype=np.float32)
    counts = np.zeros((Ly, Lx), dtype=np.float32)
    for j, s in enumerate(stat):
        ypix, xpix = s['ypix'], s['xpix']
        lam = s['lam'].astype(np.float32)
        val = bg[j]
        bg_img[ypix, xpix] += val * lam
        counts[ypix, xpix] += lam
    m = counts > 0
    bg_img[m] /= counts[m]
    # robust background scaling
    lo, hi = np.percentile(bg_img[m], (2, 98))
    bg_img = np.clip((bg_img - lo) / (hi - lo + 1e-9), 0, 1)

    # figure
    plt.figure(figsize=(8, 7))
    extent = None
    xlabel, ylabel = 'X (px)', 'Y (px)'
    if pix_to_um is not None:
        extent = [0, Lx*pix_to_um, 0, Ly*pix_to_um]
        xlabel, ylabel = 'X (µm)', 'Y (µm)'

    plt.imshow(bg_img, origin='lower', cmap='gray', extent=extent, aspect='equal', alpha=0.9)

    # Quiver input positions
    Xpos = xy[:,0].copy()
    Ypos = xy[:,1].copy()
    Uplot = U_s
    Vplot = V_s

    if pix_to_um is not None:
        Xpos *= pix_to_um; Ypos *= pix_to_um

    plt.quiver(Xpos, Ypos, Uplot, Vplot,
               angles='xy', scale_units='xy', scale=quiver_scale,
               color='tab:cyan', alpha=arrow_alpha, width=0.003)

    t0b = time[a]; t1b = time[z-1]
    plt.title(f'Propagation vectors ({signal_type})  {t0b:.2f}–{t1b:.2f} s')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    out = os.path.join(out_dir, f'vectors_{signal_type}_bin{b+1:04d}.png')
    plt.savefig(out, dpi=200)
    plt.close()
    print('Saved', out)"""

frame_dir = os.path.join(root, f'{prefix}vectors_dt')  # folder where you saved PNGs
out_path  = os.path.join(root, f'{prefix}vector_spread.mp4')

frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
fps_out = 10   # output video frame rate

# ---- for imageio v2 ----
writer = imageio.get_writer(out_path, format='FFMPEG', mode='I', fps=fps_out)
for fn in frames:
    img = imageio.imread(os.path.join(frame_dir, fn))
    writer.append_data(img)
writer.close()

print("Saved video:", out_path)