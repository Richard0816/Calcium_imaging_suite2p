import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from Fig2 import plot_top_full_trace, ExampleROI

@dataclass(frozen=True)
class Figure1Config:
    root: Path
    filtered_mask_path: Optional[Path] = None
    save_path: Optional[Path] = None
    title: Optional[str] = "Figure 1. ROI detection and cell filtering"
    show_roi_numbers: bool = False
    edge_color_all: str = "deepskyblue"
    edge_color_keep: str = "crimson"
    lw_all: float = 0.35
    lw_keep: float = 0.55
    alpha_all: float = 0.45
    alpha_keep: float = 0.75


def load_mean_image(root: Path) -> np.ndarray:
    mean_img_path = root / "meanImg.npy"
    ops_path = root / "ops.npy"

    if mean_img_path.exists():
        mean_img = np.load(mean_img_path, allow_pickle=True)
        return np.asarray(mean_img)

    if ops_path.exists():
        ops = np.load(ops_path, allow_pickle=True).item()
        if "meanImg" in ops:
            return np.asarray(ops["meanImg"])

    raise FileNotFoundError("Could not find meanImg.npy or meanImg inside ops.npy")

def load_ops(root: Path) -> dict:
    ops_path = root / "ops.npy"
    if not ops_path.exists():
        raise FileNotFoundError(f"Missing ops.npy at {ops_path}")
    return np.load(ops_path, allow_pickle=True).item()


def get_pix_to_um_from_zoom(root: Path, mean_img: np.ndarray) -> float:
    """
    Match the zoom -> FOV conversion used in spatial_heatmap.py.
    Returns an isotropic µm per pixel value for plotting a scale bar.
    """
    folder_for_notes = str(root.parent.parent.parent)  # recording folder above suite2p/plane0
    zoom = utils.get_zoom_from_notes(folder_for_notes)
    zoom = float(zoom) if zoom else 1.0

    fov_um_x = 3080.90169 / zoom
    # spatial_heatmap.py also defines fov_um_y, but for a horizontal scale bar
    # we only need the x scaling
    Lx = mean_img.shape[1]

    return float(fov_um_x) / float(Lx)


def add_scale_bar(
    ax,
    pix_to_um: float,
    bar_um: float = 100.0,
    pad_frac: float = 0.05,
    lw: float = 4.0,
    color: str = "white",
    fontsize: int = 10,
):
    """
    Draw a horizontal scale bar in the lower right of an image axis.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    width = abs(x1 - x0)
    height = abs(y1 - y0)

    bar_px = bar_um / pix_to_um

    x_start = min(x0, x1) + width * (1 - pad_frac) - bar_px
    x_end = x_start + bar_px

    if y1 < y0:
        # image coordinates after imshow usually have inverted y
        y_bar = y0 - height * pad_frac
        y_text = y_bar - height * 0.03
        y_bar = y_bar + 10
        print(y_bar, y_text)
    else:
        y_bar = y0 + height * pad_frac
        y_text = y_bar + height * 0.03

    ax.plot([x_start, x_end], [y_bar, y_bar], color=color, lw=lw, solid_capstyle="butt")
    ax.text(
        (x_start + x_end) / 2,
        y_text,
        f"{int(bar_um)} µm",
        color=color,
        fontsize=fontsize,
        ha="center",
        va="top" if y1 < y0 else "bottom",
    )
def load_stat(root: Path) -> List[dict]:
    stat_path = root / "stat.npy"
    if not stat_path.exists():
        raise FileNotFoundError(f"Missing stat.npy at {stat_path}")
    stat = np.load(stat_path, allow_pickle=True)
    return list(stat)


def load_filtered_mask(root: Path, filtered_mask_path: Optional[Path]) -> np.ndarray:
    candidates = []

    if filtered_mask_path is not None:
        candidates.append(filtered_mask_path)

    candidates.extend([
        root / "r0p7_cell_mask_bool.npy",
        root / "cell_mask_bool.npy",
        root / "iscell.npy",
    ])

    for path in candidates:
        if path.exists():
            arr = np.load(path, allow_pickle=True)

            # Handle Suite2p iscell.npy format: first column is boolean label
            if path.name == "iscell.npy" and arr.ndim == 2 and arr.shape[1] >= 1:
                return arr[:, 0].astype(bool)

            arr = np.asarray(arr).squeeze()
            if arr.dtype != bool:
                arr = arr.astype(bool)
            return arr

    raise FileNotFoundError(
        "Could not find a filtered ROI mask. Pass --filtered_mask explicitly or place "
        "r0p7_cell_mask_bool.npy in the Suite2p folder."
    )


def roi_polygon_from_stat(s: dict) -> Optional[np.ndarray]:
    ypix = np.asarray(s.get("ypix", []))
    xpix = np.asarray(s.get("xpix", []))

    if len(xpix) < 3 or len(ypix) < 3:
        return None

    pts = np.column_stack([xpix, ypix])

    # A simple pixel cloud outline. For dense Suite2p masks this is often enough.
    # If you want a true contour later, you can replace this with a convex hull.
    return pts


def build_roi_patches(stat: List[dict], mask: Optional[np.ndarray] = None) -> List[Polygon]:
    patches = []

    for i, s in enumerate(stat):
        if mask is not None and not mask[i]:
            continue

        poly_pts = roi_polygon_from_stat(s)
        if poly_pts is None:
            continue

        patches.append(Polygon(poly_pts, closed=False, fill=False))

    return patches


def add_roi_centroid_labels(ax, stat: List[dict], mask: Optional[np.ndarray] = None, color: str = "white"):
    for i, s in enumerate(stat):
        if mask is not None and not mask[i]:
            continue

        ypix = np.asarray(s.get("ypix", []))
        xpix = np.asarray(s.get("xpix", []))
        if len(xpix) == 0 or len(ypix) == 0:
            continue

        cx = float(np.mean(xpix))
        cy = float(np.mean(ypix))
        ax.text(cx, cy, str(i), color=color, fontsize=5, ha="center", va="center")


def make_figure_1(cfg: Figure1Config):
    mean_img = load_mean_image(cfg.root)
    ops = load_ops(cfg.root)
    stat = load_stat(cfg.root)
    keep_mask = load_filtered_mask(cfg.root, cfg.filtered_mask_path)

    pix_to_um = ops.get("pix_to_um", None)
    if pix_to_um is None:
        pix_to_um = get_pix_to_um_from_zoom(cfg.root, mean_img)

    if len(keep_mask) != len(stat):
        raise ValueError(f"Filtered mask length {len(keep_mask)} does not match number of ROIs {len(stat)}")

    n_all = len(stat)
    n_keep = int(np.sum(keep_mask))
    n_removed = n_all - n_keep

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        gridspec_kw={"width_ratios": [15, 15, 70]},
        constrained_layout=True
    )

    # Panel A
    ax = axes[0]
    ax.imshow(mean_img, cmap="gray")
    add_scale_bar(ax, pix_to_um=pix_to_um, bar_um=200, pad_frac=0.1, lw=1.5, color="white", fontsize=8)
    ax.set_title("A. Raw mean image")
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel B
    ax = axes[1]
    ax.imshow(mean_img, cmap="gray")
    patches_all = build_roi_patches(stat)
    pc_all = PatchCollection(
        patches_all,
        match_original=False,
        facecolor="none",
        edgecolor=cfg.edge_color_all,
        linewidth=cfg.lw_all,
        alpha=cfg.alpha_all
    )
    ax.add_collection(pc_all)
    if cfg.show_roi_numbers:
        add_roi_centroid_labels(ax, stat, color="white")
    ax.set_title(f"B. All Suite2p ROIs\nn = {n_all}")
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel C
    ax = axes[2]

    example = ExampleROI(
            root=Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0"),
            roi=21,
            nth_spike=0,
            crop_start_s=0,
            crop_end_s=0,
        )

    plot_top_full_trace(
        ax=ax,
        example=example,
        fps=15.0,
        memmap_prefix="r0p7_",
    )

    ax.set_title("C. Example fluorescence trace")
    """# Panel C
    ax = axes[2]
    ax.imshow(mean_img, cmap="gray")
    patches_keep = build_roi_patches(stat, keep_mask)
    pc_keep = PatchCollection(
        patches_keep,
        match_original=False,
        facecolor="none",
        edgecolor=cfg.edge_color_keep,
        linewidth=cfg.lw_keep,
        alpha=cfg.alpha_keep
    )
    ax.add_collection(pc_keep)
    if cfg.show_roi_numbers:
        add_roi_centroid_labels(ax, stat, mask=keep_mask, color="white")
    ax.set_title(f"C. Filtered ROIs after cell scoring\nkept = {n_keep}, removed = {n_removed}")
    ax.set_xticks([])
    ax.set_yticks([])"""

    #if cfg.title:
    #    fig.suptitle(cfg.title, fontsize=14)

    return fig


def run(cfg: Figure1Config):
    fig = make_figure_1(cfg)

    if cfg.save_path is not None:
        cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {cfg.save_path}")
        plt.close(fig)
    else:
        plt.show()


def parse_args() -> Figure1Config:
    p = argparse.ArgumentParser(description="Make Figure 1: mean image with Suite2p ROI overlays.")
    p.add_argument("--root", required=True, type=Path, help="Path to suite2p/plane0 folder")
    p.add_argument("--filtered_mask", type=Path, default=None, help="Optional path to filtered ROI boolean mask")
    p.add_argument("--save", type=Path, default=None, help="Optional output image path")
    p.add_argument("--show_roi_numbers", action="store_true", help="Label ROI centroids with ROI indices")
    args = p.parse_args()

    return Figure1Config(
        root=args.root,
        filtered_mask_path=args.filtered_mask,
        save_path=args.save,
        show_roi_numbers=args.show_roi_numbers,
    )


if __name__ == "__main__":
    root = Path(r"F:\data\2p_shifted\Hip\2024-06-04_00010\suite2p\plane0")
    cfg = Figure1Config(
        root=root,
        filtered_mask_path=root / r"0p7_cell_mask_bool.npy",   # or Path(r"...\r0p7_cell_mask_bool.npy")
        save_path=None,
        show_roi_numbers=False,
    )
    run(cfg)