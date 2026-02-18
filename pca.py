import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np
import utils

def load_dff(root: Path, prefix: str):
    plane_dir = root / "suite2p" / "plane0"
    dff, _, _, _, _ = utils.s2p_open_memmaps(plane_dir, prefix)
    print("Loaded dff shape:", dff.shape)
    return np.array(dff)

def load_clusters(root: Path, prefix: str, cluster_folder):
    base_dir = root / "suite2p" / "plane0" / f"{prefix}cluster_results"
    if cluster_folder is not None:
        base_dir = base_dir / cluster_folder

    roi_files = [
        f for f in sorted(base_dir.glob("*_rois.npy"))
        if "manual_combined" not in f.stem.lower() and 'c4'not in f.stem.lower()
    ]
    print(roi_files)
    if len(roi_files) < 2:
        raise ValueError(f"Need at least two *_rois.npy files in {base_dir}")

    clusters = {f.stem.replace("_rois", ""): np.load(f) for f in roi_files}
    return clusters

root = Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018")
prefix = "r0p7_filtered_"

# Load data
X = load_dff(root, prefix)  # (n_rois, n_time)
print(X)
n_rois = X.shape[0]

# Load clusters and build labels
clusters = load_clusters(root, prefix, cluster_folder=None)
print(clusters)
cluster_names = list(clusters.keys())
cluster_labels = np.full(n_rois, -1, dtype=int)

for i, name in enumerate(cluster_names):
    roi_idx = clusters[name].astype(int)
    roi_idx = roi_idx[(roi_idx >= 0) & (roi_idx < n_rois)]
    cluster_labels[roi_idx] = i

if np.any(cluster_labels == -1):
    print("Warning: some ROIs were not assigned to any cluster (label = -1).")

# ROI PCA so each ROI is a point in PC space
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # (n_rois, n_time)

pca = PCA(n_components=3)
roi_scores = pca.fit_transform(X_scaled)  # (n_rois, 3)

print("Explained variance ratio (PC1..PC3):", pca.explained_variance_ratio_)

# Plot ROIs in PC space, colored by cluster label
plt.figure()

for i, name in enumerate(cluster_names):
    idx = cluster_labels == i
    plt.scatter(
        roi_scores[idx, 0],
        roi_scores[idx, 1],
        label=name,
        alpha=0.75,
        s=12
    )

# Optional, show unassigned ROIs in gray
#idx = cluster_labels == -1
if np.any(idx):
    plt.scatter(
        roi_scores[idx, 0],
        roi_scores[idx, 1],
        label="unassigned",
        alpha=0.3,
        s=10
    )

plt.xlabel("PC1 (ROI scores)")
plt.ylabel("PC3 (ROI scores)")
plt.title("ROI PCA, colored by cluster")
plt.legend(markerscale=2, fontsize=8)
plt.show()
