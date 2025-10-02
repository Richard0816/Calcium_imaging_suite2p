import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import butter, sosfilt, savgol_filter
import os
import time
import utils

# ---------- per-trace helpers (1D) ----------
def robust_df_over_f_1d(F, win_sec=45, perc=10, fps=30.0):
    """Rolling percentile baseline on a 1D array, low-RAM."""
    F = np.asarray(F, dtype=np.float32)
    n = F.size

    # interp over non-finite values if any
    finite = np.isfinite(F)
    if not finite.all():
        F = np.interp(np.arange(n), np.flatnonzero(finite), F[finite]).astype(np.float32)

    win = max(3, int(win_sec * fps) | 1)  # odd
    win = min(win, n if n % 2 == 1 else n - 1)  # cannot exceed length & must be odd
    if win < 3:  # too short to filter meaningfully
        F0 = np.full_like(F, np.nanpercentile(F, perc))
    else:
        F0 = percentile_filter(F, size=win, percentile=perc, mode='nearest').astype(np.float32)

    eps = np.nanpercentile(F0, 1) if np.isfinite(F0).any() else 1.0
    eps = max(eps, 1e-9)
    dff = (F - F0) / eps
    return dff

def lowpass_causal_1d(x, fps, cutoff_hz=5.0, order=2, zi=None, sos=None):
    """Causal SOS low-pass (no filtfilt copies). Returns y, zf, sos."""
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n < 3:
        return x.copy(), zi, sos

    nyq = fps / 2.0
    cutoff = min(max(1e-4, cutoff_hz), 0.95 * nyq)
    if sos is None:
        sos = butter(order, cutoff / nyq, btype='low', output='sos')

    if zi is None:
        # zi shape: (sections, 2). Init with first sample to avoid transient.
        zi = np.zeros((sos.shape[0], 2), dtype=np.float32)
        zi[:, 0] = x[0]
        zi[:, 1] = x[0]

    y, zf = sosfilt(sos, x, zi=zi)
    return y.astype(np.float32), zf.astype(np.float32), sos

def sg_first_derivative_1d(x, fps, win_ms=333, poly=3):
    """Savitzky–Golay smoothed first derivative on 1D."""
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    win = max(3, int((win_ms / 1000.0) * fps) | 1)  # odd
    if win >= n:
        win = max(3, (n - (1 - n % 2)))  # largest valid odd <= n
    if win < 3 or n < 3:
        # fallback simple gradient
        g = np.empty_like(x)
        g[0] = 0.0
        g[1:] = (x[1:] - x[:-1]) * fps
        return g
    return savgol_filter(x, window_length=win, polyorder=poly, deriv=1, delta=1.0 / fps).astype(np.float32)


def custom_lowpass_cutoff(cutoffs, aav_info_csv, file_name):
    """
    :param cutoffs: dictionary containing cutoff values
    :param aav_info_csv: name of the file we are looking to analyse
    :param file_name: This is information taken from the human_SLE_2p_meta.xlsx file, saved as a csv for easy use
        will always look for the columns of "AAV" and "video" to determine the file name and appropriate video used
    :return: float value for our Hz value
    """
    # look into utils.py to get full information
    cutoff_hz = utils.file_name_to_aav_to_dictionary_lookup(file_name, aav_info_csv, cutoffs)

    return cutoff_hz


# ---------- batch processing over Suite2p matrices ----------
def process_suite2p_traces(
    F_cell, F_neuropil, fps,
    r=0.7,
    batch_size=256,
    win_sec=45, perc=10,
    cutoff_hz=5.0, sg_win_ms=333, sg_poly=3,
    out_dir=None, prefix=''
):
    """
    F_cell, F_neuropil: arrays from Suite2p (nROIs, T) or (T, nROIs) — auto-handled.
    Writes memmap outputs to disk to avoid RAM blowups.
    """
    # Ensure float32, and get to shape (T, N) time-major for cache efficiency
    F_cell = np.asarray(F_cell, dtype=np.float32, order='C')
    F_neuropil = np.asarray(F_neuropil, dtype=np.float32, order='C')

    if F_cell.ndim != 2 or F_neuropil.ndim != 2:
        raise ValueError("F and Fneu must be 2D: (nROIs, T) or (T, nROIs).")

    # detect orientation (Suite2p is nROIs x T). Convert to T x N.
    if F_cell.shape[0] == F_neuropil.shape[0] and F_cell.shape[0] < F_cell.shape[1]:
        # Likely nROIs x T -> transpose to T x N
        F_cell = F_cell.T
        F_neuropil = F_neuropil.T
    elif F_cell.shape[1] == F_neuropil.shape[1] and F_cell.shape[0] > F_cell.shape[1]:
        # Likely T x nROIs already
        pass
    else:
        raise ValueError("F and Fneu shapes do not align.")

    T, N = F_cell.shape
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    # Prepare memmaps for outputs (time-major for easy incremental writes)
    dff_path = os.path.join(out_dir, f"{prefix}dff.memmap.float32")
    low_path = os.path.join(out_dir, f"{prefix}dff_lowpass.memmap.float32")
    dt_path  = os.path.join(out_dir, f"{prefix}dff_dt.memmap.float32")

    dff_mm = np.memmap(dff_path, mode='w+', dtype='float32', shape=(T, N))
    low_mm = np.memmap(low_path, mode='w+', dtype='float32', shape=(T, N))
    dt_mm  = np.memmap(dt_path,  mode='w+', dtype='float32', shape=(T, N))

    # Precompute SOS once (reused for all cells)
    from scipy.signal import butter
    nyq = fps / 2.0
    cutoff = min(max(1e-4, cutoff_hz), 0.95 * nyq)
    sos = butter(2, cutoff / nyq, btype='low', output='sos').astype(np.float64)  # filter coeffs as float64, cheap

    # Process in batches of cells
    for j0 in range(0, N, batch_size):
        start_time = time.time()
        j1 = min(N, j0 + batch_size)
        # neuropil subtraction (vectorized)
        Fc_batch = (F_cell[:, j0:j1] - r * F_neuropil[:, j0:j1]).astype(np.float32)

        # per-cell operations
        for j in range(j1 - j0):
            trace = Fc_batch[:, j]

            # ΔF/F
            dff = robust_df_over_f_1d(trace, win_sec=win_sec, perc=perc, fps=fps)
            dff_mm[:, j0 + j] = dff  # write to disk

            # low-pass causal
            lp, _, _ = lowpass_causal_1d(dff, fps=fps, cutoff_hz=cutoff_hz, order=2, zi=None, sos=sos)
            low_mm[:, j0 + j] = lp

            # derivative
            dd = sg_first_derivative_1d(lp, fps=fps, win_ms=sg_win_ms, poly=sg_poly)
            dt_mm[:, j0 + j] = dd

        # flush batch to disk
        dff_mm.flush(); low_mm.flush(); dt_mm.flush()
        print(f"Processed cells {j0}–{j1-1} / {N-1} in {time.time() - start_time} seconds.")


    # Return paths to results; close memmaps (let GC handle) or del explicitly
    del dff_mm, low_mm, dt_mm
    return dff_path, low_path, dt_path

def run_analysis_on_folder(folder_name: str):
    start_time = time.time()
    fps = 30.0
    root = os.path.join(folder_name, "suite2p\\plane0\\")
    sample_name = root.split("\\")[-4]  # Human-readable sample name from path

    # Load Suite2p outputs
    F_cell = np.load(os.path.join(root, 'F.npy'), allow_pickle=True)
    F_neu = np.load(os.path.join(root, 'Fneu.npy'), allow_pickle=True)

    # Where to write outputs
    out_dir = root  # save alongside Suite2p
    # Optional: a prefix for filenames so you can run variants without clobbering
    prefix = 'r0p7_'  # e.g., indicates r=0.7

    print(f'Processing {sample_name}')

    cutoffs = {
        "6f": 5.0,
        "6m": 5.0,
        "6s": 5.0,
        "8m": 3.0
    }

    cutoff_hz = custom_lowpass_cutoff(cutoffs, "human_SLE_2p_meta", sample_name)
    print(f'cutoff_hz: {cutoff_hz}')

    dff_path, low_path, dt_path = process_suite2p_traces(
        F_cell, F_neu, fps,
        r=0.7,
        batch_size=2500,
        win_sec=45, perc=10,
        cutoff_hz=cutoff_hz, sg_win_ms=400, sg_poly=1,
        out_dir=out_dir, prefix=prefix
    )

    print("Wrote:")
    print(" dF/F       ->", dff_path)
    print(" low-pass   ->", low_path)
    print(" d/dt       ->", dt_path)
    print(f'Total time {time.time() - start_time} seconds.')


def run():
    utils.run_on_folders('D:\\data\\2p_shifted\\', run_analysis_on_folder)


# ================== RUN IT ==================
if __name__ == "__main__":
    utils.log("fluorescence_analysis.log", run)


