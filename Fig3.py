import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import percentile_filter


def neuropil_correct(F, Fneu, r=0.7):
    return F - r * Fneu


def rolling_baseline_dff(trace, fps, window_s=45, percentile=10):
    window = max(3, int(window_s * fps))
    if window % 2 == 0:
        window += 1

    F0_t = percentile_filter(trace, percentile=percentile, size=window, mode="nearest")
    epsilon = max(np.percentile(F0_t, 1), 1e-9)
    dff = (trace - F0_t) / epsilon
    return dff, F0_t, epsilon


def fixed_baseline_dff(trace, fps, baseline_s=120):
    n0 = min(len(trace), int(baseline_s * fps))
    if n0 < 1:
        raise ValueError("Baseline window is too short.")
    F0 = np.mean(trace[:n0])
    F0 = max(F0, 1e-9)
    dff = (trace - F0) / F0
    return dff, F0


def load_roi_trace_from_suite2p(root, roi_idx):
    root = Path(root)
    F = np.load(root / "F.npy")
    Fneu = np.load(root / "Fneu.npy")

    raw = F[roi_idx]
    neu = Fneu[roi_idx]
    corrected = neuropil_correct(raw, neu, r=0.7)
    return raw, neu, corrected


def make_baseline_comparison_figure(
    rec1_root,
    rec1_roi,
    rec2_root,
    rec2_roi,
    fps,
    save_path="baseline_comparison_two_recordings.png",
    t_max_s=None,
):
    raw1, neu1, corr1 = load_roi_trace_from_suite2p(rec1_root, rec1_roi)
    raw2, neu2, corr2 = load_roi_trace_from_suite2p(rec2_root, rec2_roi)

    dff_roll_1, F0t_1, eps1 = rolling_baseline_dff(corr1, fps=fps)
    dff_fix_1, F01 = fixed_baseline_dff(corr1, fps=fps)

    dff_roll_2, F0t_2, eps2 = rolling_baseline_dff(corr2, fps=fps)
    dff_fix_2, F02 = fixed_baseline_dff(corr2, fps=fps)

    t1 = np.arange(len(corr1)) / fps
    t2 = np.arange(len(corr2)) / fps

    if t_max_s is not None:
        n1 = min(len(t1), int(t_max_s * fps))
        n2 = min(len(t2), int(t_max_s * fps))
    else:
        n1 = len(t1)
        n2 = len(t2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # identical scale across all panels
    all_traces = np.concatenate([
        dff_roll_1,
        dff_fix_1,
        dff_roll_2,
        dff_fix_2
    ])

    ymin, ymax = np.min(all_traces), np.max(all_traces)
    pad = 0.05 * (ymax - ymin)


    # Recording 1, rolling
    ax = axes[0, 0]
    ax.plot(t1[:n1], dff_roll_1[:n1], lw=1.0)
    ax.set_title(f"Recording 1, ROI {rec1_roi}\nRolling baseline")
    ax.set_ylabel("ΔF/F")

    # Recording 1, fixed
    ax = axes[0, 1]
    ax.plot(t1[:n1], dff_fix_1[:n1], lw=1.0)
    ax.set_title(f"Recording 1, ROI {rec1_roi}\nFirst 2 min baseline")

    # Recording 2, rolling
    ax = axes[1, 0]
    ax.plot(t2[:n2], dff_roll_2[:n2], lw=1.0)
    ax.set_title(f"Recording 2, ROI {rec2_roi}\nRolling baseline")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F")

    # Recording 2, fixed
    ax = axes[1, 1]
    ax.plot(t2[:n2], dff_fix_2[:n2], lw=1.0)
    ax.set_title(f"Recording 2, ROI {rec2_roi}\nFirst 2 min baseline")
    ax.set_xlabel("Time (s)")

    for ax in axes.ravel():
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Comparison of rolling and fixed baseline ΔF/F normalization", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved to {save_path}")
    print(f"Recording 1 epsilon: {eps1:.6f}, fixed F0: {F01:.6f}")
    print(f"Recording 2 epsilon: {eps2:.6f}, fixed F0: {F02:.6f}")


if __name__ == "__main__":
    make_baseline_comparison_figure(
        rec1_root=r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0",
        rec1_roi=16,
        rec2_root=r"F:\data\2p_shifted\Hip\2024-10-30_00005\suite2p\plane0",
        rec2_roi=10,
        fps=15.0,
        save_path="baseline_comparison_two_recordings.png",
        t_max_s=600,
    )