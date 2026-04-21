import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import utils

def plot_roi_shape_and_trace(root: Path, roi: int, prefix="r0p7_"):
    """
    Plot:
    - Left: ROI spatial footprint (from stat.npy)
    - Right: ΔF/F trace
    """

    # --- Load stat ---
    stat = np.load(root / "stat.npy", allow_pickle=True)

    if roi >= len(stat):
        raise ValueError(f"ROI {roi} out of bounds (N={len(stat)})")

    roi_stat = stat[roi]
    xpix = roi_stat["xpix"]
    ypix = roi_stat["ypix"]

    # --- Load ΔF/F ---
    dff, _, _, T, N = utils.s2p_open_memmaps(root, prefix=prefix)

    trace = np.asarray(dff[:, roi])

    # --- Time axis ---
    fps = 15.0  # adjust if needed
    time = np.arange(T) / fps

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2))

    # ROI shape
    ax1.scatter(xpix, ypix, s=1)
    ax1.invert_yaxis()
    ax1.set_aspect("equal")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # ΔF/F trace
    ax2.plot(time, trace, lw=1)
    ax2.set_title("ΔF/F trace")
    ax2.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    root = Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0")
    for i in range(200):
        plot_roi_shape_and_trace(root, roi=190+i)
        print(f"ROI {190+i} plotted.")