"""
Hierarchical clustering with a user-selectable colormap / palette.

Designed as a backend for GUI integration: each step is a small function
that accepts parameters and returns plain data, so a GUI layer can call
them independently (load -> cluster -> plot) without re-running work.

The `palette` argument drives BOTH the dendrogram branch colors and the
spatial ROI map (they stay in sync). The palette is sampled across its
full range so that the first and last clusters (furthest apart in the
dendrogram) land on opposite ends of the colormap and the rest are
spread in between. The heatmap uses its own `heatmap_cmap`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import (
    linkage,
    dendrogram,
    fcluster,
    set_link_color_palette,
)
from scipy.spatial.distance import pdist

import utils


# Handy presets a GUI dropdown can offer directly.
CATEGORICAL_PALETTES = [
    "tab10", "tab20", "Set1", "Set2", "Set3",
    "Paired", "Accent", "Pastel1", "Pastel2", "Dark2",
]
CONTINUOUS_PALETTES = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "coolwarm", "RdBu", "RdYlBu", "Spectral",
    "turbo", "rainbow", "hsv",
]
AVAILABLE_PALETTES = CATEGORICAL_PALETTES + CONTINUOUS_PALETTES


PaletteLike = Union[str, Iterable[str], mpl.colors.Colormap]


def resolve_palette(palette: PaletteLike, n_colors: int = 10) -> list[str]:
    """
    Normalize a palette spec into a list of hex color strings spanning
    the full range of the colormap.

    Accepts:
      - matplotlib colormap name (e.g. "viridis", "tab10")
      - a matplotlib Colormap instance
      - an iterable of color strings (hex, named, RGB tuples)

    For continuous colormaps, `n_colors` samples are drawn evenly from 0
    to 1, so the first sample lands on one end of the cmap and the last
    on the other. For qualitative colormaps, colors are taken in order
    and spread out across the available entries.
    """
    if not isinstance(palette, (str, mpl.colors.Colormap)):
        colors = [mpl.colors.to_hex(c) for c in palette]
        if n_colors <= 0:
            return colors
        if n_colors <= len(colors):
            # Spread selections evenly across the provided list, so a
            # small cluster count still ends up using endpoints.
            idx = np.linspace(0, len(colors) - 1, n_colors).round().astype(int)
            return [colors[i] for i in idx]
        return colors

    cmap = mpl.colormaps.get_cmap(palette) if isinstance(palette, str) else palette
    n_colors = max(1, int(n_colors))

    # Qualitative colormaps expose a discrete .colors list.
    discrete = getattr(cmap, "colors", None)
    if discrete is not None and len(discrete) <= 20:
        colors = list(discrete)
        if n_colors <= len(colors):
            idx = np.linspace(0, len(colors) - 1, n_colors).round().astype(int)
            return [mpl.colors.to_hex(colors[i]) for i in idx]
        return [mpl.colors.to_hex(c) for c in colors]

    # Continuous: sample n_colors evenly across [0, 1].
    if n_colors == 1:
        return [mpl.colors.to_hex(cmap(0.5))]
    return [
        mpl.colors.to_hex(cmap(i / (n_colors - 1)))
        for i in range(n_colors)
    ]


def _resolve_pix_to_um(root: Path, ops: dict) -> Optional[float]:
    """
    Same pix_to_um resolution used elsewhere in the codebase:
    prefer ops['pix_to_um'] when present; otherwise derive from the
    recording's zoom notes (matches Fig1.get_pix_to_um_from_zoom).
    """
    px = ops.get("pix_to_um", None)
    if px is not None:
        return float(px)
    try:
        folder_for_notes = str(Path(root).parent.parent.parent)
        zoom = utils.get_zoom_from_notes(folder_for_notes)
        zoom = float(zoom) if zoom else 1.0
        fov_um_x = 3080.90169 / zoom
        return float(fov_um_x) / float(ops["Lx"])
    except Exception:
        return None


def _load_filter_mask(root: Path) -> Optional[np.ndarray]:
    """Cell-filter keep mask: predicted_cell_mask.npy, then iscell.npy fallback."""
    pred_path = root / "predicted_cell_mask.npy"
    if pred_path.exists():
        return np.load(pred_path).astype(bool)
    iscell_path = root / "iscell.npy"
    if iscell_path.exists():
        ic = np.load(iscell_path)
        return ((ic[:, 0] > 0) if ic.ndim == 2 else (ic > 0)).astype(bool)
    return None


def load_dff(root: Path, prefix: str = "r0p7_filtered_") -> np.ndarray:
    """Open ``<prefix>dff.memmap.float32`` against the cell-filter mask.

    For a "filtered" prefix the memmap was written at the size of the cell-
    filter keep mask (predicted_cell_mask.npy preferred, iscell.npy fallback),
    so we apply that exact mask here. For an unfiltered prefix we use the
    full Suite2p ROI count.
    """
    root = Path(root)
    F = np.load(root / "F.npy", mmap_mode="r")
    N_total, T = F.shape
    is_filtered = "filtered" in prefix.split("_")

    if is_filtered:
        mask = _load_filter_mask(root)
        if mask is None:
            raise FileNotFoundError(
                f"{root}: prefix {prefix!r} requires a cell-filter mask "
                "(predicted_cell_mask.npy or iscell.npy).")
        if mask.size != N_total:
            raise ValueError(
                f"{root}: cell-filter mask length {mask.size} does not "
                f"match F.npy ROI count {N_total}.")
        N_kept = int(mask.sum())
    else:
        N_kept = N_total

    dff_path = root / f"{prefix}dff.memmap.float32"
    if not dff_path.exists():
        raise FileNotFoundError(f"Missing dF/F memmap: {dff_path}")
    dff = np.memmap(dff_path, dtype="float32", mode="r",
                    shape=(T, N_kept))
    return np.asarray(dff)


def run_clustering(
    dff: np.ndarray,
    method: str = "ward",
    metric: str = "euclidean",
) -> np.ndarray:
    """Z-score per ROI, then linkage on the ROI-by-ROI distance matrix."""
    dff_z = (dff - np.mean(dff, axis=0)) / (np.std(dff, axis=0) + 1e-8)
    dist_matrix = pdist(dff_z.T, metric=metric)
    return linkage(dist_matrix, method=method)


def count_leaf_color_groups(Z: np.ndarray, color_threshold: float) -> int:
    r = dendrogram(Z, no_plot=True, color_threshold=color_threshold * np.max(Z[:, 2]))
    return len(set(r["leaves_color_list"]))


def count_clusters(Z: np.ndarray, color_threshold: float) -> int:
    """Number of distinct clusters below `color_threshold * max(linkage)`."""
    T = color_threshold * np.max(Z[:, 2])
    labels = fcluster(Z, t=T, criterion="distance")
    return int(len(np.unique(labels)))


def auto_choose_threshold(
    Z: np.ndarray,
    target_counts: Iterable[int] = (4, 5),
    start: float = 0.90,
    stop: float = 0.05,
    step: float = 0.01,
) -> float:
    """Sweep the cut fraction downward until group count lands in target_counts."""
    target_counts = set(target_counts)
    for ct in np.arange(start, stop - 1e-9, -step):
        if count_leaf_color_groups(Z, float(ct)) in target_counts:
            return float(ct)
    return start


def plot_dendrogram(
    Z: np.ndarray,
    save_path: Path,
    color_threshold: float,
    palette: PaletteLike = "tab10",
    above_threshold_color: str = "gray",
    n_palette_colors: Optional[int] = None,
) -> list[str]:
    """
    Draw the dendrogram using `palette` for the below-cut branches.

    If `n_palette_colors` is None, it defaults to the actual number of
    below-threshold clusters so the palette spans end-to-end (first
    cluster on one side of the cmap, last cluster on the other).
    """
    if n_palette_colors is None:
        n_palette_colors = max(1, count_clusters(Z, color_threshold))

    colors = resolve_palette(palette, n_colors=n_palette_colors)
    set_link_color_palette(colors)

    T = color_threshold * np.max(Z[:, 2])
    plt.figure(figsize=(10, 5))
    dendrogram(Z, color_threshold=T, above_threshold_color=above_threshold_color)
    plt.axhline(T, linestyle="--", linewidth=2)
    plt.text(
        0.99, T, f" cut @ {T:.3g}  ({color_threshold:.2f}×max)",
        transform=plt.gca().get_yaxis_transform(),
        ha="right", va="bottom",
    )
    plt.title("ROI Hierarchical Clustering Dendrogram")
    plt.xlabel("ROIs")
    plt.ylabel("Linkage distance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    return colors


def plot_heatmap(
    dff: np.ndarray,
    order: np.ndarray,
    save_path: Path,
    heatmap_cmap: str = "magma",
) -> None:
    dff_sorted = dff[:, order]
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        dff_sorted.T,
        cmap=heatmap_cmap,
        cbar_kws={"label": "ΔF/F"},
        xticklabels=False,
        yticklabels=False,
    )
    plt.title("Hierarchical Clustering of ROI ΔF/F Traces")
    plt.xlabel("Time (s)")
    plt.ylabel("ROIs (clustered order)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _stat_for_prefix(root: Path, prefix: str) -> tuple[list, Optional[np.ndarray]]:
    """
    Restrict ``stat.npy`` to the cell-filter keep mask when ``prefix`` signals
    a filtered set (e.g. 'r0p7_filtered_') so the i-th stat entry matches the
    i-th column of the filtered ΔF/F memmap. Mask source matches ``load_dff``:
    predicted_cell_mask.npy first, iscell.npy as fallback.
    """
    root = Path(root)
    stat = list(np.load(root / "stat.npy", allow_pickle=True))
    if "filtered" in prefix.split("_"):
        mask = _load_filter_mask(root)
        if mask is not None:
            used = np.where(mask)[0]
            return [stat[i] for i in used], used
        print(f"⚠️ no cell-filter mask in {root}; "
              "painting against unfiltered stat.npy.")
    return stat, None


def plot_spatial(
    root: Path,
    order: Iterable[int],
    link_colors: Iterable[str],
    save_path: Path,
    used_indices: Optional[np.ndarray] = None,
    prefix: str = "r0p7_filtered_",
    title: str = "Spatial map colored by dendrogram ROI colors",
) -> None:
    """
    Paint ROIs on the FOV using the same colors scipy gave the dendrogram
    leaves. Axes are in µm when a pix_to_um conversion is available
    (ops or zoom notes). No colorbar — the image is RGB by construction.

    If `used_indices` is provided, stat is restricted to those rows.
    Otherwise, if `prefix` indicates a filtered ROI set, stat is restricted
    via the cell mask so `order` aligns with clustering indices.
    """
    root = Path(root)
    ops = np.load(root / "ops.npy", allow_pickle=True).item()
    Ly, Lx = ops["Ly"], ops["Lx"]

    if used_indices is not None:
        full_stat = list(np.load(root / "stat.npy", allow_pickle=True))
        stat = [full_stat[i] for i in used_indices]
    else:
        stat, _ = _stat_for_prefix(root, prefix)

    link_colors = list(link_colors)
    roi_rgb = np.zeros((len(stat), 3))
    for i, roi_idx in enumerate(order):
        roi_rgb[roi_idx, :] = mpl.colors.to_rgb(link_colors[i])

    R = utils.paint_spatial(roi_rgb[:, 0], stat, Ly, Lx)
    G = utils.paint_spatial(roi_rgb[:, 1], stat, Ly, Lx)
    B = utils.paint_spatial(roi_rgb[:, 2], stat, Ly, Lx)
    img = np.dstack([R, G, B])

    coverage = utils.paint_spatial(np.ones(len(stat)), stat, Ly, Lx)
    img[coverage == 0] = np.nan

    pix_to_um = _resolve_pix_to_um(root, ops)
    if pix_to_um is not None:
        extent = [0, Lx * pix_to_um, 0, Ly * pix_to_um]
        xlabel, ylabel = "X (µm)", "Y (µm)"
    else:
        extent = None
        xlabel, ylabel = "X (pixels)", "Y (pixels)"

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(img, origin="lower", extent=extent, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print("Saved", save_path)


def cluster_and_plot(
    root: Path,
    prefix: str = "r0p7_filtered_",
    method: str = "ward",
    metric: str = "euclidean",
    palette: PaletteLike = "tab10",
    heatmap_cmap: str = "magma",
    above_threshold_color: str = "gray",
    color_threshold: Optional[float] = None,
    target_counts: Iterable[int] = (4, 5),
    save_dir: Optional[Path] = None,
    save_format: Union[str, Iterable[str]] = "png",
) -> dict:
    """
    End-to-end: load ΔF/F -> cluster -> write dendrogram, heatmap, spatial map.

    `save_format` controls the output extension(s). Pass "png" (default),
    "svg", "pdf", or an iterable like ("png", "svg") to save in multiple
    formats. matplotlib's savefig picks the renderer from the extension.

    Returns a dict with the linkage matrix, leaf order, chosen threshold,
    leaf colors, and the output directory — everything a GUI needs to
    display results or re-render with a different palette.
    """
    save_dir = (
        Path(save_dir)
        if save_dir
        else Path(root) / f"{prefix}cluster_results" / "colourblind_safe"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    formats = (save_format,) if isinstance(save_format, str) else tuple(save_format)
    formats = tuple(f.lstrip(".").lower() for f in formats)

    dff = load_dff(Path(root), prefix=prefix)
    Z = run_clustering(dff, method=method, metric=metric)

    if color_threshold is None:
        color_threshold = auto_choose_threshold(Z, target_counts=target_counts)

    order = np.asarray(dendrogram(Z, no_plot=True)["leaves"], dtype=int)

    # Sample the palette with exactly n_clusters entries so the first
    # cluster lands on one end of the colormap, the last on the other,
    # and everything else is evenly spread in between. Apply BEFORE both
    # the plotted dendrogram and the no_plot extraction so leaves_color_list
    # reflects the user's choice for the spatial map too.
    n_clusters = max(1, count_clusters(Z, color_threshold))
    palette_colors = resolve_palette(palette, n_colors=n_clusters)
    set_link_color_palette(palette_colors)

    T = color_threshold * np.max(Z[:, 2])
    link_colors = dendrogram(Z, no_plot=True, color_threshold=T)["leaves_color_list"]

    for ext in formats:
        plot_heatmap(
            dff, order, save_dir / f"cluster_heatmap.{ext}",
            heatmap_cmap=heatmap_cmap,
        )
        plot_dendrogram(
            Z,
            save_dir / f"dendrogram.{ext}",
            color_threshold=color_threshold,
            palette=palette,
            above_threshold_color=above_threshold_color,
            n_palette_colors=n_clusters,
        )
        plot_spatial(
            Path(root),
            order,
            link_colors,
            save_path=save_dir / f"spatial_dendrogram_colored_rois.{ext}",
            prefix=prefix,
        )

    np.save(save_dir / "cluster_order.npy", order)
    np.save(save_dir / "linkage.npy", Z)

    return {
        "linkage": Z,
        "order": order,
        "color_threshold": color_threshold,
        "n_clusters": n_clusters,
        "link_colors": link_colors,
        "palette_colors": palette_colors,
        "save_dir": save_dir,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Hierarchical clustering with a user-selectable colormap."
    )
    p.add_argument("root", type=Path, help="Suite2p plane0 directory")
    p.add_argument("--prefix", default="r0p7_filtered_")
    p.add_argument("--method", default="ward")
    p.add_argument("--metric", default="euclidean")
    p.add_argument(
        "--palette",
        default="tab10",
        help=f"Colormap/palette for dendrogram branches + spatial map. "
             f"One of: {', '.join(AVAILABLE_PALETTES)} (or any mpl cmap name).",
    )
    p.add_argument("--heatmap-cmap", default="magma", help="Colormap for the ΔF/F heatmap.")
    p.add_argument(
        "--color-threshold",
        type=float,
        default=None,
        help="Cut fraction (0-1) of max linkage. Auto-selected if omitted.",
    )
    p.add_argument("--above-threshold-color", default="gray")
    p.add_argument(
        "--save-format",
        nargs="+",
        default=["png"],
        help="Output format(s). One or more of: png, svg, pdf, eps, tif. "
             "Pass multiple to save every plot in each format (e.g. "
             "`--save-format png svg`).",
    )
    args = p.parse_args()

    result = cluster_and_plot(
        root=args.root,
        prefix=args.prefix,
        method=args.method,
        metric=args.metric,
        palette=args.palette,
        heatmap_cmap=args.heatmap_cmap,
        color_threshold=args.color_threshold,
        above_threshold_color=args.above_threshold_color,
        save_format=args.save_format,
    )
    print(f"Done. Results saved to {result['save_dir']}")
