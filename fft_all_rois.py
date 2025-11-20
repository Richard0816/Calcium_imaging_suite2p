from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import math
import utils  # must be importable (same folder or PYTHONPATH)
import os

def compute_fft(signal: np.ndarray, fps: float):
    """Compute FFT and return frequencies and power."""
    N = len(signal)
    yf = fft.rfft(signal)
    xf = fft.rfftfreq(N, 1 / fps)
    power = np.abs(yf) ** 2
    return xf, power


def plot_fft_grid(
    dff: np.ndarray,
    fps: float,
    save_dir: Path,
    rois_per_fig: int = 96,
    freq_max: float = 15.0
):
    """
    Plot FFT power spectra for all ROIs in grids of rois_per_fig per figure.
    """
    num_frames, num_rois = dff.shape
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving FFT plots to {save_dir} ...")

    num_batches = math.ceil(num_rois / rois_per_fig)

    for batch_idx in range(num_batches):
        start = batch_idx * rois_per_fig
        end = min((batch_idx + 1) * rois_per_fig, num_rois)
        batch_rois = range(start, end)
        n_rois = end - start

        ncols = int(math.ceil(np.sqrt(rois_per_fig)))
        nrows = int(math.ceil(n_rois / ncols))

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(ncols * 2.2, nrows * 1.8),
            sharex=True, sharey=True
        )
        axes = np.array(axes).ravel()

        for i, roi in enumerate(batch_rois):
            signal = np.asarray(dff[:, roi])
            xf, power = compute_fft(signal, fps)
            sorted_y_data = sorted(power[3:])
            second_largest_y = sorted_y_data[-1] if len(sorted_y_data) > 0 else 1

            axes[i].plot(xf, power, lw=0.8)
            axes[i].set_xlim(0, freq_max)
            axes[i].set_ylim(0, second_largest_y * 1.1)
            axes[i].set_title(f"ROI {roi}", fontsize=7)
            axes[i].tick_params(axis='both', which='both', labelsize=6, length=2)

        # Turn off unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        fig.suptitle(f"FFT Power Spectra (ROIs {start}-{end - 1})", y=0.92)
        fig.text(0.5, 0.04, "Frequency (Hz)", ha="center")
        fig.text(0.04, 0.5, "Power", va="center", rotation="vertical")
        fig.tight_layout(rect=[0.05, 0.05, 1, 0.9])

        out_path = save_dir / f"fft_batch_{batch_idx:03d}.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")


def main(root: Path, fps: float = 30.0, prefix: str = "r0p7_", rois_per_fig: int = 96, freq_max: float = 15.0):
    """Main entry: load ΔF/F traces and save FFT plots."""
    dff, _, _ = utils.s2p_open_memmaps(root, prefix=prefix)[:3]
    if dff.ndim != 2:
        raise ValueError(f"Expected dff to be 2D (T, N), got {dff.shape}")

    print(f"Loaded ΔF/F: {dff.shape[0]} frames × {dff.shape[1]} ROIs")
    save_dir = root / "fft_grids"

    if save_dir.exists():
        if any(save_dir.iterdir()):
            print(f"Warning: {save_dir} is not empty; skipping FFT generation for this directory")
            return

    plot_fft_grid(dff, fps, save_dir, rois_per_fig=rois_per_fig, freq_max=freq_max)
    print("All FFT grids saved.")

def run_on_folder(root: str):
    root = Path(os.path.join(root, "suite2p\\plane0\\"))
    fps = 30.0
    prefix = "r0p7_"
    freq_max = 15.0
    rois_per_fig = 60
    main(root, fps, prefix, rois_per_fig, freq_max)

if __name__ == "__main__":
    utils.run_on_folders("F:\\data\\2p_shifted\\Hip\\", run_on_folder, "fft_output.log")
