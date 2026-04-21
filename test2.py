import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

spks_path = Path(r"F:\data\2p_shifted\Hip\2024-06-04_00001\suite2p\plane0\spks.npy")
spks = np.load(spks_path, allow_pickle=False)

# Ensure 2D
if spks.ndim == 1:
    spks = spks.reshape(1, -1)
elif spks.ndim != 2:
    raise ValueError(f"Expected spks to be 2D [n_rois, T]. Got shape {spks.shape}")

n_rois, T = spks.shape

# Select ROI index 0 (change this index if you want another ROI)
roi_index = 0
if roi_index >= n_rois:
    raise ValueError(f"ROI index {roi_index} out of bounds for {n_rois} ROIs.")

roi_spks = spks[roi_index]
events = roi_spks > 0

# Downsample for readability if long
target_T = 1300
ds = max(1, int(np.ceil(T / target_T)))

if ds > 1:
    new_T = int(np.ceil(T / ds))
    events_ds = np.zeros(new_T, dtype=bool)
    for j in range(new_T):
        s = j * ds
        e = min((j + 1) * ds, T)
        events_ds[j] = events[s:e].any()
    events_plot = events_ds
    x_label = f"Time (downsampled bins, ds={ds})"
else:
    events_plot = events
    x_label = "Time (frames/bins)"

t_idx = np.where(events_plot)[0]

plt.figure(figsize=(16, 3), dpi=150)
plt.scatter(t_idx, np.zeros_like(t_idx), s=5, marker=".")
plt.yticks([])
plt.xlabel(x_label)
plt.title(f"Event raster for ROI {roi_index}")
plt.tight_layout()

out_path = Path(r"F:\data\2p_shifted\Hip\2024-06-04_00001\suite2p\plane0\spks_single_roi_raster.png")
plt.savefig(out_path)
plt.close()

str(out_path)
