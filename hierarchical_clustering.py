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


def plot_spatial_from_labels(root: Path, order, link_colors, prefix: str = "r0p7_"):
    """Color ROIs spatially by dendrogram leaf colors corresponding to their order."""
    import matplotlib as mpl
    import numpy as np

    ops = np.load(root / "ops.npy", allow_pickle=True).item()
    stat = np.load(root / "stat.npy", allow_pickle=True)
    Ly, Lx = ops["Ly"], ops["Lx"]

    mask_path = root / "blank.npy" # manually define the mask as a .npy file

    # --- Apply ROI mask if filtered ---
    if "filtered" in prefix.split("_"):
        mask_path = root / "r0p7_cell_mask_bool.npy"
    if mask_path.exists():
        mask = np.load(mask_path)
        stat = [s for s, keep in zip(stat, mask) if keep]
        print(f"Applied cell mask: {mask.sum()} / {len(mask)} ROIs kept.")
    else:
        print(f"Warning: {mask_path} not found; skipping mask application.")

    # --- Build per-ROI RGB array ---
    roi_rgb = np.zeros((len(stat), 3))
    for i, roi_idx in enumerate(order):
        color = link_colors[i]
        roi_rgb[roi_idx, :] = mpl.colors.to_rgb(color)

    # --- Paint each color channel separately ---
    R = utils.paint_spatial(roi_rgb[:, 0], stat, Ly, Lx)
    G = utils.paint_spatial(roi_rgb[:, 1], stat, Ly, Lx)
    B = utils.paint_spatial(roi_rgb[:, 2], stat, Ly, Lx)

    # Stack to RGB image
    img = np.dstack([R, G, B])
    coverage = utils.paint_spatial(np.ones(len(stat)), stat, Ly, Lx)
    img[coverage == 0] = np.nan  # transparent background

    out_path = root / f"{prefix}cluster_results" / "spatial_dendrogram_colored_rois.png"

    spatial_heatmap.show_spatial(
        img,
        title="Spatial map colored by dendrogram ROI colors",
        Lx=Lx,
        Ly=Ly,
        stat=stat,
        pix_to_um=ops.get("pix_to_um", None),
        cmap=None,
        outpath=out_path,
    )
    print(f"Saved: {out_path}")


def main(root: Path, fps: float = 30.0, prefix: str = "r0p7_", method: str = "ward", metric: str = "euclidean"):
    save_dir = root / f"{prefix}cluster_results"
    dff = load_dff(root, prefix=prefix)
    Z = run_clustering(dff, method=method, metric=metric)
    order = plot_dendrogram_heatmap(dff, Z, save_dir, fps=fps)
    np.save(save_dir / "cluster_order.npy", np.array(order, dtype=int))
    print(f"Saved cluster order to {save_dir / 'cluster_order.npy'}")

    r = dendrogram(Z, no_plot=True, color_threshold=0.7 * max(Z[:, 2]))
    link_colors = r['leaves_color_list']

    plot_spatial_from_labels(root, order, link_colors, prefix=prefix)
    print(
        f"Saved spatial map colored by dendrogram ROI colors to {save_dir / 'spatial_dendrogram_colored_rois.png'}"
    )


if __name__ == "__main__":
    root = Path(r'F:\data\2p_shifted\Cx\2024-11-05_00007\suite2p\plane0')
    fps = 30.0
    prefix = 'r0p7_'
    method = 'ward'
    metric = 'euclidean'
    main(root, fps, prefix, method, metric)

