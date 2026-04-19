

"""import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and will be used.")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU.")

"""
"""import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import utils
# --- Config ---
root   = r'F:\data\2p_shifted\2024-11-05_00007\suite2p\plane0'
prefix = 'r0p7_'
fps    = 30.0
roi    = 126
t_max  = None         # seconds to plot (None = full)
z_thr  = 3.5        # z-score (MAD) threshold for valid peaks
prominence_min = 0.05   # base prominence for find_peaks

# --- Load data ---
stat = np.load(os.path.join(root, 'stat.npy'), allow_pickle=True)
N = len(stat)
low = np.memmap(os.path.join(root, f'{prefix}dff_lowpass.memmap.float32'),
                dtype='float32', mode='r')
T = low.size // N
low = low.reshape(T, N)

sig = low[:, roi]
time = np.arange(T) / fps
if t_max is not None:
    idx = time < t_max
    sig = sig[idx]; time = time[idx]

# --- Robust z-score (MAD) ---
z, med, mad = utils.mad_z(sig)

# --- Find peaks (basic prominence, then filter by z-score) ---
peaks, props = find_peaks(sig, prominence=prominence_min)
# keep only peaks whose z >= threshold
keep = z[peaks] >= z_thr
peaks = peaks[keep]
for k in props: props[k] = props[k][keep]

# --- FWHM (full width at half prominence ≈ FWHM) ---
res = peak_widths(sig, peaks, rel_height=0.5)
width_samples = res[0]; left_ips, right_ips = res[2], res[3]
fwhm_sec = width_samples / fps
peak_times = time[peaks]
peak_amps  = sig[peaks]

# --- Print summary ---
print(f"ROI {roi}: {len(peaks)} peaks pass z≥{z_thr}")
for i, (t, amp, w) in enumerate(zip(peak_times, peak_amps, fwhm_sec)):
    print(f"{i:3d}: t={t:7.3f}s  amp={amp:7.4f}  FWHM≈{w:6.3f}s")

# --- Plot ---
plt.figure(figsize=(12,5))
plt.plot(time, sig, label='ΔF/F (low-pass)')
plt.axhline(med + z_thr*1.4826*mad, color='r', ls='--', label=f'z={z_thr}')
plt.axhline(med, color='k', ls=':', lw=0.5)

for i, p in enumerate(peaks):
    w = width_samples[i]
    li = left_ips[i]/fps; ri = right_ips[i]/fps
    hh = sig[p] - 0.5*props["prominences"][i]
    plt.hlines(hh, li, ri, colors='g', linestyles='--', lw=1)
    plt.vlines(time[p], hh, sig[p], colors='g', lw=1)

plt.legend(); plt.xlabel('Time (s)'); plt.ylabel('ΔF/F')
plt.title(f'ROI {roi} peaks (z ≥ {z_thr}) with FWHM lines')
plt.tight_layout(); plt.show()
"""

"""import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
def plot_fft(signal_data, sampling_rate):
    N = len(signal_data)  # Number of samples
    yf = fft.rfft(signal_data)  # Compute the FFT
    xf = fft.rfftfreq(N, 1 / sampling_rate)  # Get the corresponding frequencies
    power_spectrum = np.abs(yf)**2 # Power spectrum
    # Or, for Power Spectral Density (PSD) using Welch's method:
    #from scipy.signal import welch
    #frequencies_welch, psd_welch = welch(signal_data, fs=sampling_rate, nperseg=256)
    sorted_y_data = sorted(power_spectrum[3:])
    second_largest_y = sorted_y_data[-1]

    plt.figure(figsize=(10, 5))
    plt.plot(xf, power_spectrum)
    plt.ylim(0, second_largest_y * 1.1)

    plt.title('Power Spectrum of the Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)
    plt.show()"""
"""
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt

"""

'''import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import utils
def add_corner_scalebar(ax, x_len, y_len, x_label="1 s", y_label="0.02 ΔF/F"):
    """
    Draws an L-shaped scalebar in bottom-left corner.
    x_len, y_len are in data units.
    """

    # get limits
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # anchor in bottom-left
    x_start = x0 + 0.1 * (x1 - x0)
    y_start = y0 + 0.5 * (y1 - y0)

    # horizontal bar
    ax.plot([x_start, x_start + x_len], [y_start, y_start],
            color="black", lw=2)

    # vertical bar
    ax.plot([x_start, x_start], [y_start, y_start + y_len],
            color="black", lw=2)

    # labels
    ax.text(x_start + x_len / 2, y_start - 0.05 * (y1 - y0),
            x_label, ha="center", va="top", fontsize=10)

    ax.text(x_start - 0.02 * (x1 - x0), y_start + y_len / 2,
            y_label, ha="right", va="center", fontsize=10, rotation=90)

def plot_F_and_Fneu(root: Path, roi: int, fps=15.0):
    # --- Load raw signals ---
    F, Fneu, T, N, time_major = utils.s2p_load_raw(root)

    # --- Extract trace ---
    if time_major:
        f_trace = F[:, roi]
        fneu_trace = Fneu[:, roi]
    else:
        f_trace = F[roi, :]
        fneu_trace = Fneu[roi, :]

    # --- Time axis ---
    time = np.arange(len(f_trace)) / fps

    # --- Plot ---
    plt.figure(figsize=(4, 3))
    ax = plt.gca()

    ax.plot(time, f_trace, lw=1)
    ax.plot(time, fneu_trace, lw=1, alpha=0.7)

    # remove all axes
    ax.axis("off")

    # add scalebar
    add_corner_scalebar(
        ax,
        x_len=60.0,  # 1 second
        y_len=1000,  # 0.02 ΔF/F (adjust to your data scale)
        x_label="30 s",
        y_label="100 FU"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    root = Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0")
    plot_F_and_Fneu(root, roi=21)'''

from adaptive_detection import visualize_audit, AdaptiveConfig

config = AdaptiveConfig(
    soma_diameter_px=24.0,       # tune to your typical soma size
    blob_min_contrast=0.1,
    blob_min_area_px=25,
    blob_max_area_px=400,
)
result = visualize_audit(
    r'F:\data\2p_shifted\Cx\2024-08-21_00002\suite2p\plane0',
    config,
    outpath='audit.png',
)
print(f"Detected {result['n_rois']} ROIs, found {result['n_residual_blobs']} missed candidates")