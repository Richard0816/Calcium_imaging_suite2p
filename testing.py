"""import numpy as np
from pathlib import Path
import suite2p
output_ops = np.load('D:\\data\\2p_shifted\\2024-06-03_00001\\suite2p\\plane0\\spks.npy', allow_pickle=True)
print(output_ops)

f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename='D:\\data\\2p_shifted\\2024-06-03_00001\\suite2p\\plane0\\data.bin', n_frames = f_raw.shape[0])
print(f_reg)
#f_cells = np.load(Path(output_ops['D:\\data\\2p_shifted\\2024-06-03_00001\\suite2p\\plane0\\']).joinpath('F.npy'))
"""

"""import psutil

# Get virtual memory statistics
virtual_memory = psutil.virtual_memory()

# Total physical memory
total_memory_gb = round(virtual_memory.total / (1024**3), 2)
print(f"Total Physical Memory: {total_memory_gb} GB")

# Available memory
available_memory_gb = round(virtual_memory.available / (1024**3), 2)
print(f"Available Memory: {available_memory_gb} GB")"""
"""
import numpy as np
import matplotlib.pyplot as plt

#root = 'D:\\data\\2p_shifted\\2024-06-03_00001\\suite2p\\plane0\\'
#root = 'D:\\data\\2p_shifted\\2024-07-01_00019\\suite2p\\plane0\\'
#root = 'D:\\data\\2p_shifted\\2024-06-03_00004\\suite2p\\plane0\\'
root = 'D:\\data\\2p_shifted\\2024-11-05_00007\\suite2p\\plane0\\'

F = np.load(root+'F.npy', allow_pickle=True)
Fneu = np.load(root+'Fneu.npy', allow_pickle=True)
spks = np.load(root+'spks.npy', allow_pickle=True)
stat = np.load(root+'stat.npy', allow_pickle=True)
ops = np.load(root+'ops.npy', allow_pickle=True)
ops = ops.item()
iscell = np.load(root+'iscell.npy', allow_pickle=True)


print(len(F[0]))


def derv_plot(lst):
    final_lst =[]
    for i in range(len(lst)-2):
        final_lst.append(lst[i+2]-lst[i])
    return final_lst


def temp(iscell):
    for i in range(len(iscell)):
        x = iscell[i].tolist()
        if x[0] == 1 and float(x[1]) > 0.9:
            return i
print(temp(iscell))

F_naught = np.mean(Fneu[0:2000])

#n=3
#n=18
#n=2
n=126

plt.plot(Fneu[n]/F_naught)
#plt.plot(F[3])
plt.plot(derv_plot(Fneu[n]/F_naught))
plt.show()
"""
"""import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and will be used.")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU.")"""

'''import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
import os

fps = 30.0
root = 'D:\\data\\2p_shifted\\2024-06-03_00001\\suite2p\\plane0\\'
F_cell = np.load(os.path.join(root, 'F.npy'), allow_pickle=True)
F_neuropil = np.load(os.path.join(root, 'Fneu.npy'), allow_pickle=True)

import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import butter, sosfiltfilt, savgol_filter

def robust_df_over_f(F, win_sec=45, perc=10, fps=30.0):
    """Rolling percentile baseline without as_strided."""
    win = max(3, int(win_sec * fps) | 1)  # odd, >=3
    F = np.asarray(F, dtype=np.float64)
    # handle NaNs by temporary interpolation
    F_finite = np.isfinite(F)
    if not F_finite.all():
        F = np.interp(np.arange(len(F)), np.flatnonzero(F_finite), F[F_finite])
    F0 = percentile_filter(F, size=win, percentile=perc, mode='nearest')
    eps = np.nanpercentile(F0, 1) if np.isfinite(F0).any() else 1.0
    dff = (F - F0) / max(eps, 1e-9)
    return dff

def safe_lowpass(x, fps, cutoff_hz=5.0, order=3):
    """Stable SOS low-pass with guards for NaNs, short traces, and pad issues."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 10:
        return x.copy()  # too short to filter meaningfully

    # sanitize cutoff (must be < Nyquist)
    nyq = fps / 2.0
    cutoff = min(max(1e-6, cutoff_hz), nyq * 0.95)

    # fill NaNs/Infs
    finite = np.isfinite(x)
    if not finite.all():
        x = np.interp(np.arange(n), np.flatnonzero(finite), x[finite])

    sos = butter(order, cutoff / nyq, btype='low', output='sos')

    # choose a safe padlen (SciPy default can exceed length for short traces)
    default_padlen = 3 * (2 * order)  # approx for SOS
    padlen = min(default_padlen, max(1, n // 2 - 1))

    try:
        y = sosfiltfilt(sos, x, padtype='odd', padlen=padlen)
    except Exception:
        # fall back to zero-padding; if that fails, return unfiltered
        try:
            y = sosfiltfilt(sos, x, padtype='constant', padlen=padlen)
        except Exception:
            y = x.copy()
    return y

def sg_first_derivative(x, fps, win_ms=333, poly=3):
    """Savitzky–Golay smoothed first derivative (robust and simple)."""
    win = max(3, int((win_ms/1000.0)*fps) | 1)  # odd, >=3
    win = min(win, len(x) - (1 - len(x) % 2))   # don't exceed length
    if win < 3:
        return np.gradient(x) * fps
    return savgol_filter(x, window_length=win, polyorder=poly, deriv=1, delta=1.0/fps)

Fcorr = F_cell - 0.7 * F_neuropil            # or your fitted neuropil coeff
dff   = robust_df_over_f(Fcorr, win_sec=45, perc=10, fps=fps)
dff_f = safe_lowpass(dff, fps=fps, cutoff_hz=5.0, order=3)
d_dt  = sg_first_derivative(dff_f, fps=fps, win_ms=333, poly=3)

print(d_dt)'''