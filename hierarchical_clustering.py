from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

import spatial_heatmap
import utils



def load_dff(root: Path, prefix: str = "r0p7_"):
    dff, _, _ = utils.s2p_open_memmaps(root, prefix=prefix)[:3]
    if dff.ndim != 2:
        raise ValueError(f"Expected (T, N) array, got {dff.shape}")
    print(f"Loaded ΔF/F: {dff.shape[0]} frames × {dff.shape[1]} ROIs")
    return dff


def run_clustering(dff: np.ndarray, method: str = "ward", metric: str = "euclidean"):
    dff_z = (dff - np.mean(dff, axis=0)) / (np.std(dff, axis=0) + 1e-8)
    dist_matrix = pdist(dff_z.T, metric=metric)
    Z = linkage(dist_matrix, method=method)
    return Z


def plot_dendrogram_heatmap(dff: np.ndarray, Z, save_dir: Path, fps: float = 30.0):
    save_dir.mkdir(parents=True, exist_ok=True)
    num_frames, num_rois = dff.shape
    dendro = dendrogram(Z, no_plot=True)
    order = dendro["leaves"]
    dff_sorted = dff[:, order]

    plt.figure(figsize=(12, 6))
    sns.heatmap(dff_sorted.T, cmap="magma", cbar_kws={"label": "ΔF/F"}, xticklabels=False, yticklabels=False)
    plt.title("Hierarchical Clustering of ROI ΔF/F Traces")
    plt.xlabel("Time (s)")
    plt.ylabel("ROIs (clustered order)")
    plt.tight_layout()

    heatmap_path = save_dir / "cluster_heatmap.png"
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    dendrogram(Z, color_threshold=0.7 * max(Z[:, 2]))
    plt.title("ROI Hierarchical Clustering Dendrogram")
    plt.xlabel("ROIs")
    plt.ylabel("Linkage distance")
    plt.tight_layout()

    dendro_path = save_dir / "dendrogram.png"
    plt.savefig(dendro_path, dpi=200)
    plt.close()

    return order


def plot_spatial_from_labels(root: Path, cmap,labels_file: str = "cluster_labels.npy"):
    """Load precomputed cluster labels and color ROIs accordingly."""
    import matplotlib as mpl

    labels_path = root / "cluster_results" /labels_file
    if not labels_path.exists():
        raise FileNotFoundError(f"{labels_path} not found")

    cluster_labels = np.load(labels_path)
    ops = np.load(root / "ops.npy", allow_pickle=True).item()
    stat = np.load(root / "stat.npy", allow_pickle=True)
    Ly, Lx = ops["Ly"], ops["Lx"]
    print(f"Loaded spatial map from {labels_path}")

    # Paint cluster IDs onto spatial map
    cluster_labels_full = cluster_labels.astype(float)
    cluster_labels_full[np.isnan(cluster_labels_full)] = np.nan
    img = utils.paint_spatial(cluster_labels_full, stat, Ly, Lx)

    out_path = root / "cluster_results" / "spatial_from_labels.png"

    spatial_heatmap.show_spatial(
        img,
        title="Spatial Map from cluster_labels.npy",
        Lx=Lx,
        Ly=Ly,
        stat=stat,
        pix_to_um=ops.get("pix_to_um", None),
        cmap=cmap,
        outpath=out_path,
    )
    print(f"Saved: {out_path}")

def cmap_from_link_colors(link_colors):
    """
    Create a matplotlib colormap from a list of dendrogram link colors,
    where the first color is neutral grey instead of the first link color.
    """
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap

    # Deduplicate while preserving order
    unique_colors = []
    for c in link_colors:
        if c not in unique_colors:
            unique_colors.append(c)

    # Insert grey at the beginning
    grey = "#808080"
    colors = [grey] + unique_colors

    # Build colormap
    cmap = ListedColormap(colors, name="dendro_cmap")
    return cmap

def main(root: Path, fps: float = 30.0, prefix: str = "r0p7_", method: str = "ward", metric: str = "euclidean"):
    save_dir = root / "cluster_results"
    dff = load_dff(root, prefix=prefix)
    Z = run_clustering(dff, method=method, metric=metric)
    order = plot_dendrogram_heatmap(dff, Z, save_dir, fps=fps)

    np.save(save_dir / "cluster_order.npy", np.array(order, dtype=int))
    print(f"Saved cluster order to {save_dir / 'cluster_order.npy'}")

    r = dendrogram(Z, no_plot=True, color_threshold=0.7 * max(Z[:, 2]))
    link_colors = r['color_list']
    custom_cmap = cmap_from_link_colors(link_colors)
    plot_spatial_from_labels(root, custom_cmap)


if __name__ == "__main__":
    root = Path(r'F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0')
    fps = 30.0
    prefix = 'r0p7_'
    method = 'ward'
    metric = 'euclidean'
    main(root, fps, prefix, method, metric)

