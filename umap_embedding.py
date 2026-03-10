from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import utils

def load_dff(root: Path, prefix: str):
    plane_dir = root / "suite2p" / "plane0"
    dff, _, _, _, _ = utils.s2p_open_memmaps(plane_dir, prefix)
    # Use asarray to preserve memmap views when possible and avoid
    # unintentionally materializing the entire array into memory.
    return np.asarray(dff)

def load_clusters(root: Path, prefix: str, cluster_folder: str | None = None):
    base_dir = root / "suite2p" / "plane0" / f"{prefix}cluster_results"
    if cluster_folder is not None:
        base_dir = base_dir / cluster_folder

    roi_files = [
        f for f in sorted(base_dir.glob("*_rois.npy"))
        if "manual_combined" not in f.stem.lower() and "c4" not in f.stem.lower()
    ]
    if len(roi_files) < 2:
        raise ValueError(f"Need at least two *_rois.npy files in {base_dir}")

    return {f.stem.replace("_rois", ""): np.load(f) for f in roi_files}

root = Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018")

prefix_clusters = "r0p7_filtered_"
prefix_dff = "r0p7_"  # important, match ROI index space used to create clusters

X = load_dff(root, prefix_dff)  # (n_rois, n_time)
X = X.T
clusters = load_clusters(root, prefix_clusters)

n_rois = X.shape[0]
cluster_names = list(clusters.keys())
cluster_labels = np.full(n_rois, -1, dtype=int)

for i, name in enumerate(cluster_names):
    roi_idx = clusters[name].astype(int)
    if np.any(roi_idx >= n_rois):
        raise ValueError(
            f"{name} has ROI index >= {n_rois}. Your dff prefix does not match cluster ROI index space."
        )
    cluster_labels[roi_idx] = i

print("Assigned ROIs:", int(np.sum(cluster_labels != -1)))
print("Unassigned ROIs:", int(np.sum(cluster_labels == -1)))

# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# UMAP
try:
    import umap
except ImportError as e:
    raise ImportError("Install umap-learn: pip install umap-learn") from e

reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    metric="euclidean",
    random_state=0
)
emb = reducer.fit_transform(X_scaled)  # (n_rois, 2)

color_map = {
    "C1": "orange",
    "C2": "green",
    "C3": "red",
}

plt.figure()
for i, name in enumerate(cluster_names):
    idx = cluster_labels == i
    plt.scatter(emb[idx, 0], emb[idx, 1], s=10, alpha=0.75, label=name, color=color_map[name])

idx = cluster_labels == -1
if np.any(idx):
    plt.scatter(emb[idx, 0], emb[idx, 1], s=8, alpha=0.2, label="unassigned")

plt.title("UMAP of ROIs colored by cluster")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.legend(markerscale=2, fontsize=8)
plt.show()
