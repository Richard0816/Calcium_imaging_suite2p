from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

import spatial_heatmap
import utils

def count_leaf_color_groups(Z, color_threshold: float) -> int:
    """How many dendrogram leaf-color groups would we get at this threshold fraction?"""
    r = dendrogram(Z, no_plot=True, color_threshold=color_threshold * np.max(Z[:, 2]))
    return len(set(r["leaves_color_list"]))


def export_rois_by_leaf_color(root: Path, Z, color_threshold: float = 0.7, prefix: str = "r0p7_"):
    """
    Export lists of ROI indices grouped by dendrogram leaf color.
    Each color (leaf cluster) is saved as <color>_rois.npy.
    """
    from scipy.cluster.hierarchy import dendrogram
    import re

    # Get dendrogram info
    r = dendrogram(Z, no_plot=True, color_threshold=color_threshold * max(Z[:, 2]))
    leaves = r['leaves']
    leaf_colors = r['leaves_color_list']

    # Group ROI indices by color
    color_groups = {}
    for roi, color in zip(leaves, leaf_colors):
        color_groups.setdefault(color, []).append(roi)

    # Save each color group as separate .npy file
    save_dir = root / f"{prefix}cluster_results"
    save_dir.mkdir(parents=True, exist_ok=True)

    for color, roi_list in color_groups.items():
        # sanitize color name (e.g., 'C0', '#FF8800') → 'C0' or 'FF8800'
        color_name = re.sub(r'[^A-Za-z0-9]+', '_', color)
        path = save_dir / f"{color_name}_rois.npy"
        np.save(path, np.array(roi_list, dtype=int))
        print(f"Saved {path} with {len(roi_list)} ROIs.")

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


def plot_dendrogram_heatmap(dff: np.ndarray, Z, save_dir: Path, fps: float = 30.0, color_threshold: float = 0.7):
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

    T = color_threshold * np.max(Z[:, 2])

    dendrogram(Z, color_threshold=T)

    # draw the cut line
    plt.axhline(T, linestyle="--", linewidth=2)
    plt.text(
        0.99, T, f" cut @ {T:.3g}  ({color_threshold:.2f}×max)",
        transform=plt.gca().get_yaxis_transform(),  # x in axes coords, y in data coords
        ha="right", va="bottom"
    )
    plt.title("ROI Hierarchical Clustering Dendrogram")
    plt.xlabel("ROIs")
    plt.ylabel("Linkage distance")
    plt.tight_layout()

    dendro_path = save_dir / "dendrogram.png"
    plt.savefig(dendro_path, dpi=200)
    plt.close()

    return order


def plot_spatial_from_labels(root: Path, order, link_colors, labels_file: str = None, prefix: str = "r0p7_", directory_extension: str = "combined_cluster"):
    """Color ROIs spatially by dendrogram leaf colors corresponding to their order."""
    import matplotlib as mpl
    import numpy as np


    ops = np.load(root / "ops.npy", allow_pickle=True).item()
    stat = np.load(root / "stat.npy", allow_pickle=True)
    Ly, Lx = ops["Ly"], ops["Lx"]


    # --- Determine which ROIs to use ---
    if labels_file is not None:
        labels_path = root / f"{prefix}cluster_results" / directory_extension / labels_file
        if not labels_path.exists():
            raise FileNotFoundError(f"{labels_path} not found")
        cluster_labels = np.load(labels_path)
        print(f"Loaded {len(cluster_labels)} ROI indices from {labels_file}")
        stat = [stat[i] for i in cluster_labels]
        used_indices = cluster_labels
    elif "filtered" in prefix.split("_"):
        mask_path = root / "r0p7_cell_mask_bool.npy"
        if mask_path.exists():
            mask = np.load(mask_path)
            stat = [s for s, keep in zip(stat, mask) if keep]
            used_indices = np.where(mask)[0]
            print(f"Applied cell mask: {mask.sum()} / {len(mask)} ROIs kept.")
        else:
            used_indices = np.arange(len(stat))
            print(f"Warning: {mask_path} not found; skipping mask application.")
    else:
        used_indices = np.arange(len(stat))

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

    if labels_file is not None:
        out_path = root / f"{prefix}cluster_results" / directory_extension / f"spatial_dendrogram_colored_rois.png"
    else:
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
    # --- Automatically choose a color_threshold that yields ~4–5 groups ---
    target_counts = {4, 5}
    start = 0.90
    stop = 0.05
    step = 0.01

    chosen = start
    chosen_n = count_leaf_color_groups(Z, chosen)

    for ct in np.arange(start, stop - 1e-9, -step):
        n_groups = count_leaf_color_groups(Z, float(ct))
        if n_groups in target_counts:
            chosen = float(ct)
            chosen_n = n_groups
            break

    color_threshold = chosen
    print(f"[dendrogram] auto color_threshold={color_threshold:.2f} → {chosen_n} groups")

    order = plot_dendrogram_heatmap(dff, Z, save_dir, fps=fps, color_threshold=color_threshold)
    np.save(save_dir / "cluster_order.npy", np.array(order, dtype=int))
    print(f"Saved cluster order to {save_dir / 'cluster_order.npy'}")

    r = dendrogram(Z, no_plot=True, color_threshold=color_threshold * max(Z[:, 2]))
    link_colors = r['leaves_color_list']

    plot_spatial_from_labels(root, order, link_colors, prefix=prefix)
    print(
        f"Saved spatial map colored by dendrogram ROI colors to {save_dir / 'spatial_dendrogram_colored_rois.png'}"
    )

    export_rois_by_leaf_color(root, Z, color_threshold=color_threshold, prefix=prefix)
    print("Saved ROI lists by dendrogram leaf color.")

def main_from_existing_clustering(root: Path,
                                  roi_files: list[str],
                                  cluster_folder: str = None,
                                  fps: float = 30.0,
                                  prefix: str = "r0p7_",
                                  method: str = "ward",
                                  metric: str = "euclidean"):
    """
    Manual re-clustering utility that replaces main_selected_npy.
    - Lets user manually select ROI .npy files (C1_rois.npy, C2_rois.npy, etc.) from any folder.
    - Combines all selected ROI files into one clustering.
    - Saves results, new spatial maps, and exports per-color ROI subsets.
    """
    from scipy.cluster.hierarchy import dendrogram

    # Determine working directory
    if cluster_folder is not None:
        save_dir = root / f"{prefix}cluster_results" / cluster_folder
    else:
        save_dir = root / f"{prefix}cluster_results"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Manually selected ROI files ---
    combined_rois = []
    for roi_file in roi_files:
        # Allow selecting from parent or subfolders
        roi_path = Path(roi_file)
        if not roi_path.exists():
            potential = save_dir / roi_file
            if potential.exists():
                roi_path = potential
            else:
                print(f"⚠️ Skipping {roi_file} (not found)")
                continue

        rois = np.load(roi_path)
        combined_rois.extend(rois.tolist())
        print(f"Loaded {len(rois)} ROIs from {roi_path}")

    if not combined_rois:
        raise ValueError("No valid ROI indices found across selected files.")

    combined_rois = np.unique(combined_rois)
    print(f"Total combined ROIs for manual re-clustering: {len(combined_rois)}")

    # --- Load ΔF/F traces ---
    dff = load_dff(root, prefix=prefix)
    if np.max(combined_rois) >= dff.shape[1]:
        raise ValueError(f"ROI indices exceed available ROIs ({dff.shape[1]})")

    dff_subset = dff[:, combined_rois]
    print(f"Subset shape for manual re-clustering: {dff_subset.shape}")

    # --- Run clustering ---
    Z = run_clustering(dff_subset, method=method, metric=metric)
    order = plot_dendrogram_heatmap(dff_subset, Z, save_dir, fps=fps, color_threshold=0.9)

    np.save(save_dir / "manual_combined_rois.npy", combined_rois)
    np.save(save_dir / "manual_order.npy", np.array(order, dtype=int))
    np.save(save_dir / "manual_linkage.npy", Z)
    print(f"Saved manual re-clustering results to {save_dir}")

    # --- Generate dendrogram colors and spatial map ---

    r = dendrogram(Z, no_plot=True, color_threshold=0.9 * max(Z[:, 2]))
    link_colors = r["leaves_color_list"]

    plot_spatial_from_labels(root, order, link_colors,
                             labels_file="manual_combined_rois.npy",
                             prefix=prefix,
                             directory_extension=(cluster_folder or "manual_recluster"))

    # --- Export color-based ROI lists ---
    print("Exporting new ROI groups by leaf color...")
    leaves = r['leaves']
    leaf_colors = r['leaves_color_list']
    color_groups = {}
    for roi_local, color in zip(leaves, leaf_colors):
        roi_global = combined_rois[roi_local]
        color_groups.setdefault(color, []).append(int(roi_global))

    for color, roi_list in color_groups.items():
        color_name = color.replace('#', '').replace('C', 'C')
        path = save_dir / f"{color_name}_rois.npy"
        np.save(path, np.array(roi_list, dtype=int))
        print(f"Saved {path} with {len(roi_list)} ROIs.")

    print(f"✅ Exported {len(color_groups)} color-based ROI subsets in {save_dir}.")
    print(f"✅ Manual re-clustering complete — results saved and ready for reuse.")


if __name__ == "__main__":
    root = Path(r'F:\data\2p_shifted\Hip\2024-06-03_00009\suite2p\plane0')
    fps = 30.0
    prefix = 'r0p7_filtered_'
    method = 'ward'
    metric = 'euclidean'
    #main_selected_npy(root, ['C2_rois.npy'], fps, prefix, method, metric)

    # Manually selected ROI subsets — these can be in parent or subfolders
    roi_files = [
        r"F:\data\2p_shifted\Hip\2024-06-03_00009\suite2p\plane0\r0p7_filtered_cluster_results\C1_rois.npy"
    ]

    # Optional: specify a target folder name for new clustering outputs
    cluster_folder = r"C1_recluster"

    # Run manual re-clustering on these selected ROI sets
    #main_from_existing_clustering(root=root, roi_files=roi_files, cluster_folder=cluster_folder, fps=30.0, prefix=prefix, method=method, metric=metric)

    main(root, fps, prefix, method, metric)

