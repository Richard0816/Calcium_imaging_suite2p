import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import utils  # must be accessible (same folder or PYTHONPATH)


def load_dff(root: Path, prefix: str = "r0p7_"):
    """Load ΔF/F traces from Suite2p memmaps."""
    dff, _, _ = utils.s2p_open_memmaps(root, prefix=prefix)[:3]
    if dff.ndim != 2:
        raise ValueError(f"Expected (T, N) array, got {dff.shape}")
    print(f"Loaded ΔF/F: {dff.shape[0]} frames × {dff.shape[1]} ROIs")
    return dff


def run_clustering(dff: np.ndarray, method: str = "ward", metric: str = "euclidean"):
    """
    Perform hierarchical clustering on ROI activity.
    Returns linkage matrix and sorted index order.
    """
    # Normalize each ROI (zero mean, unit variance)
    dff_z = (dff - np.mean(dff, axis=0)) / (np.std(dff, axis=0) + 1e-8)

    # Pairwise distances (ROIs × ROIs)
    dist_matrix = pdist(dff_z.T, metric=metric)

    # Hierarchical linkage
    Z = linkage(dist_matrix, method=method)
    return Z


def plot_dendrogram_heatmap(dff: np.ndarray, Z, save_dir: Path, fps: float = 30.0):
    """Plot clustered heatmap with dendrogram and save to file."""
    save_dir.mkdir(parents=True, exist_ok=True)
    num_frames, num_rois = dff.shape

    # Cluster order from linkage
    dendro = dendrogram(Z, no_plot=True)
    order = dendro["leaves"]
    dff_sorted = dff[:, order]

    # Time axis
    time = np.arange(num_frames) / fps

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        dff_sorted.T,
        cmap="magma",
        cbar_kws={"label": "ΔF/F"},
        xticklabels=False,
        yticklabels=False,
    )
    plt.title("Hierarchical Clustering of ROI ΔF/F Traces")
    plt.xlabel("Time (s)")
    plt.ylabel("ROIs (clustered order)")
    plt.tight_layout()

    heatmap_path = save_dir / "cluster_heatmap.png"
    plt.savefig(heatmap_path, dpi=200)
    plt.close()
    print(f"Saved heatmap: {heatmap_path}")

    # Plot dendrogram separately
    plt.figure(figsize=(10, 5))
    dendrogram(Z, color_threshold=0.7 * max(Z[:, 2]))
    plt.title("ROI Hierarchical Clustering Dendrogram")
    plt.xlabel("ROIs")
    plt.ylabel("Linkage distance")
    plt.tight_layout()

    dendro_path = save_dir / "dendrogram.png"
    plt.show()
    plt.savefig(dendro_path, dpi=200)
    plt.close()
    print(f"Saved dendrogram: {dendro_path}")

    return order


def main(root: Path, fps: float = 30.0, prefix: str = "r0p7_", method: str = "ward", metric: str = "euclidean"):
    """Main entry: cluster all ROI ΔF/F traces and save plots."""
    save_dir = root / "cluster_results"
    dff = load_dff(root, prefix=prefix)
    Z = run_clustering(dff, method=method, metric=metric)
    order = plot_dendrogram_heatmap(dff, Z, save_dir, fps=fps)

    # Save sorted ROI order
    np.save(save_dir / "cluster_order.npy", np.array(order, dtype=int))
    print(f"Saved cluster order to {save_dir / 'cluster_order.npy'}")
    print("Clustering complete.")


if __name__ == "__main__":
    root = Path(r'F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0')
    fps = 30.0
    prefix = 'r0p7_filtered_'
    method = 'ward'
    metric = 'euclidean'
    main(root, fps, prefix, method, metric)
