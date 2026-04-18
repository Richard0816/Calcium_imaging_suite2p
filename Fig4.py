import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import utils


def robust_zscore_1d(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-12:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def hysteresis_onsets(z, z_enter=3.5, z_exit=2.0, min_sep_frames=1):
    z = np.asarray(z, dtype=float)

    onsets = []
    active = False
    last_onset = -10**12

    for i, val in enumerate(z):
        if not active:
            if val >= z_enter and (i - last_onset) >= min_sep_frames:
                onsets.append(i)
                last_onset = i
                active = True
        else:
            if val < z_exit:
                active = False

    return np.asarray(onsets, dtype=int)


def compute_event_rate_and_peak_dz(
    deriv_memmap,
    fps,
    z_enter=3.5,
    z_exit=2.0,
    min_sep_s=0.25,
):
    deriv_memmap = np.asarray(deriv_memmap)
    n_frames, n_rois = deriv_memmap.shape

    duration_min = n_frames / fps / 60.0
    min_sep_frames = max(1, int(round(min_sep_s * fps)))

    event_rate = np.zeros(n_rois, dtype=np.float32)
    peak_dz = np.zeros(n_rois, dtype=np.float32)

    for roi in range(n_rois):
        dz = np.asarray(deriv_memmap[:, roi], dtype=float)
        z = robust_zscore_1d(dz)

        onsets = hysteresis_onsets(
            z,
            z_enter=z_enter,
            z_exit=z_exit,
            min_sep_frames=min_sep_frames
        )

        event_rate[roi] = len(onsets) / duration_min if duration_min > 0 else 0.0
        peak_dz[roi] = np.max(z) if z.size else 0.0

    return event_rate, peak_dz


def main():
    recording_root = Path(r"F:\data\2p_shifted\Hip\2024-06-04_00010")
    plane0 = recording_root / "suite2p" / "plane0"

    score_path = recording_root / "roi_scores.npy"
    mask_path = plane0 / "r0p7_cell_mask_bool.npy"
    stat_path = plane0 / "stat.npy"
    ops_path = plane0 / "ops.npy"
    # open saved memmaps
    dff_mm, low_mm, deriv_mm, n_frames, n_rois = utils.s2p_open_memmaps(
        plane0,
        prefix="r0p7_"
    )

    fps = 15.0

    event_rate, peak_dz = compute_event_rate_and_peak_dz(
        deriv_memmap=deriv_mm,
        fps=fps,
        z_enter=3.5,
        z_exit=2.0,
        min_sep_s=0.25,
    )
    scores = np.load(score_path).astype(float)
    keep_mask = np.load(mask_path).astype(bool)
    stat = np.load(stat_path, allow_pickle=True).tolist()
    ops = np.load(ops_path, allow_pickle=True).item()

    Ly = int(ops["Ly"])
    Lx = int(ops["Lx"])

    # -------------------------------------------------
    # Per ROI features
    # -------------------------------------------------
    # area from Suite2p stat
    area = np.array([s.get("npix", len(s["ypix"])) for s in stat], dtype=float)

    # event_rate and peak_z should come from your pipeline outputs.

    if not (len(scores) == len(area) == len(event_rate) == len(peak_dz) == len(stat)):
        raise ValueError("Feature arrays do not all match the number of ROIs.")

    # infer effective threshold from kept ROIs
    threshold = float(np.min(scores[keep_mask])) if np.any(keep_mask) else None

    # -------------------------------------------------
    # Spatial probability map using your existing pipeline function
    # -------------------------------------------------
    spatial_prob = utils.paint_spatial(scores, stat, Ly, Lx)

    # -------------------------------------------------
    # Figure
    # -------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axA, axB, axC, axD, axE, axF = axes.ravel()

    # A. Event rate vs peak derivative z score
    sc1 = axA.scatter(
        event_rate,
        peak_dz,
        c=scores,
        cmap="viridis",
        s=18,
        alpha=0.85,
        vmin=0,
        vmax=1
    )
    cbar1 = fig.colorbar(sc1, ax=axA, fraction=0.046, pad=0.04)
    cbar1.set_label("Cell-likeness score")
    axA.set_xlabel("Event rate")
    axA.set_ylabel("Peak derivative z score")
    axA.set_title("A. Event rate vs peak derivative z score")

    # B. Area vs event rate
    sc2 = axB.scatter(
        area,
        event_rate,
        c=scores,
        cmap="viridis",
        s=18,
        alpha=0.85,
        vmin=0,
        vmax=1
    )
    cbar2 = fig.colorbar(sc2, ax=axB, fraction=0.046, pad=0.04)
    cbar2.set_label("Cell-likeness score")
    axB.set_xlabel("ROI area (pixels)")
    axB.set_ylabel("Event rate")
    axB.set_title("B. ROI area vs event rate")

    # C. Histogram of scores
    axC.hist(scores, bins=40)
    if threshold is not None:
        axC.axvline(threshold, linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.3f}")
        axC.legend(frameon=False)
    axC.set_xlabel("Cell-likeness score")
    axC.set_ylabel("Count")
    axC.set_title("C. Distribution of cell scores")

    # D. Spatial map of cell likelihood
    im = axD.imshow(spatial_prob, cmap="magma", vmin=0, vmax=1)
    cbar3 = fig.colorbar(im, ax=axD, fraction=0.046, pad=0.04)
    cbar3.set_label("Cell-likeness score")
    axD.set_title("D. Spatial map of cell likelihood")
    axD.set_xticks([])
    axD.set_yticks([])

    kept = int(np.sum(keep_mask))
    total = len(keep_mask)
    #fig.suptitle(
    #    f"Cell scoring and filtering, kept {kept} of {total} ROIs",
    #    fontsize=14
    #)
    # --------------------------------
    # Load Suite2p data for ROI panels
    # --------------------------------
    plane0 = recording_root / "suite2p" / "plane0"


    ops_path = plane0 / "ops.npy"


    mean_img = np.load(ops_path, allow_pickle=True).item()["meanImg"]
    stat = np.load(plane0 / "stat.npy", allow_pickle=True)
    keep_mask = np.load(plane0 / "r0p7_cell_mask_bool.npy").astype(bool)

    # helper to draw ROI outlines
    def draw_rois(ax, stat, mask=None, color="deepskyblue", lw=0.35, alpha=0.5):
        patches = []
        for i, s in enumerate(stat):
            if mask is not None and not mask[i]:
                continue

            ypix = np.asarray(s["ypix"])
            xpix = np.asarray(s["xpix"])

            if len(xpix) < 3:
                continue

            pts = np.column_stack([xpix, ypix])
            patches.append(Polygon(pts, closed=False, fill=False))

        pc = PatchCollection(
            patches,
            match_original=False,
            facecolor="none",
            edgecolor=color,
            linewidth=lw,
            alpha=alpha
        )
        ax.add_collection(pc)

    axE.imshow(mean_img, cmap="gray")
    draw_rois(axE, stat, mask=None, color="deepskyblue", lw=0.35, alpha=0.45)

    axE.set_title("E. All Suite2p ROIs")
    axE.set_xticks([])
    axE.set_yticks([])
    axF.imshow(mean_img, cmap="gray")
    draw_rois(axF, stat, mask=keep_mask, color="crimson", lw=0.5, alpha=0.75)

    axF.set_title(f"F. Filtered ROIs kept: {kept} of {total} ROIs")
    axF.set_xticks([])
    axF.set_yticks([])
    plt.tight_layout()

    plt.show()



if __name__ == "__main__":
    main()