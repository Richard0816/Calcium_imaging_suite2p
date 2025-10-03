import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import butter, sosfilt, savgol_filter
import os
import time
import utils

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


# ---------- batch processing over Suite2p matrices ----------
def _normalize_and_transpose_arrays(F_cell, F_neuropil):
    """
    Normalize arrays to float32 and transpose to time-major format (T, N).

    Returns:
        tuple: (F_cell, F_neuropil, num_timepoints, num_rois)
    """
    F_cell = np.asarray(F_cell, dtype=np.float32, order='C')
    F_neuropil = np.asarray(F_neuropil, dtype=np.float32, order='C')

    if F_cell.ndim != 2 or F_neuropil.ndim != 2:
        raise ValueError("F and Fneu must be 2D: (nROIs, T) or (T, nROIs).")

    # Detect orientation (Suite2p is nROIs x T). Convert to T x N.
    if F_cell.shape[0] == F_neuropil.shape[0] and F_cell.shape[0] < F_cell.shape[1]:
        # Likely nROIs x T -> transpose to T x N
        F_cell = F_cell.T
        F_neuropil = F_neuropil.T
    elif F_cell.shape[1] == F_neuropil.shape[1] and F_cell.shape[0] > F_cell.shape[1]:
        # Already T x nROIs format
        pass
    else:
        raise ValueError("F and Fneu shapes do not align.")

    num_timepoints, num_rois = F_cell.shape
    return F_cell, F_neuropil, num_timepoints, num_rois


def _setup_output_memmaps(out_dir, prefix, num_timepoints, num_rois):
    """
    Create output directory and memory-mapped arrays for results.

    Returns:
        tuple: (dff_memmap, lowpass_memmap, derivative_memmap, file_paths)
    """
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    dff_path = os.path.join(out_dir, f"{prefix}dff.memmap.float32")
    lowpass_path = os.path.join(out_dir, f"{prefix}dff_lowpass.memmap.float32")
    derivative_path = os.path.join(out_dir, f"{prefix}dff_dt.memmap.float32")

    dff_memmap = np.memmap(dff_path, mode='w+', dtype='float32', shape=(num_timepoints, num_rois))
    lowpass_memmap = np.memmap(lowpass_path, mode='w+', dtype='float32', shape=(num_timepoints, num_rois))
    derivative_memmap = np.memmap(derivative_path, mode='w+', dtype='float32', shape=(num_timepoints, num_rois))

    file_paths = (dff_path, lowpass_path, derivative_path)
    return dff_memmap, lowpass_memmap, derivative_memmap, file_paths


def _compute_filter_coefficients(fps, cutoff_hz, filter_order=2):
    """
    Compute Butterworth filter SOS coefficients for low-pass filtering.

    Returns:
        ndarray: Second-order sections representation of the filter
    """
    from scipy.signal import butter

    nyquist_freq = fps / 2.0
    normalized_cutoff = min(max(1e-4, cutoff_hz), 0.95 * nyquist_freq)
    sos = butter(filter_order, normalized_cutoff / nyquist_freq, btype='low', output='sos')
    return sos.astype(np.float64)


def _process_single_cell(trace, fps, win_sec, perc, cutoff_hz, sg_win_ms, sg_poly, sos):
    """
    Process a single cell trace through the complete analysis pipeline.

    Returns:
        tuple: (dff, lowpass_filtered, derivative)
    """
    # Compute ΔF/F
    dff = robust_df_over_f_1d(trace, win_sec=win_sec, perc=perc, fps=fps)

    # Apply low-pass causal filter
    lowpass_filtered, _, _ = lowpass_causal_1d(dff, fps=fps, cutoff_hz=cutoff_hz,
                                               order=2, zi=None, sos=sos)

    # Compute Savitzky-Golay first derivative
    derivative = sg_first_derivative_1d(lowpass_filtered, fps=fps,
                                        win_ms=sg_win_ms, poly=sg_poly)

    return dff, lowpass_filtered, derivative


def _process_cell_batch(F_cell_batch, F_neuropil_batch, neuropil_coefficient,
                        cell_start_idx, fps, win_sec, perc, cutoff_hz,
                        sg_win_ms, sg_poly, sos,
                        dff_memmap, lowpass_memmap, derivative_memmap):
    """
    Process a batch of cells with neuropil subtraction and write results to disk.
    """
    # Neuropil subtraction (vectorized)
    corrected_batch = (F_cell_batch - neuropil_coefficient * F_neuropil_batch).astype(np.float32)

    num_cells_in_batch = corrected_batch.shape[1]

    # Process each cell in the batch
    for cell_idx in range(num_cells_in_batch):
        trace = corrected_batch[:, cell_idx]
        global_cell_idx = cell_start_idx + cell_idx

        dff, lowpass_filtered, derivative = _process_single_cell(
            trace, fps, win_sec, perc, cutoff_hz, sg_win_ms, sg_poly, sos
        )

        # Write results to disk
        dff_memmap[:, global_cell_idx] = dff
        lowpass_memmap[:, global_cell_idx] = lowpass_filtered
        derivative_memmap[:, global_cell_idx] = derivative

    # Flush batch to disk
    dff_memmap.flush()
    lowpass_memmap.flush()
    derivative_memmap.flush()


def process_suite2p_traces(
        F_cell, F_neuropil, fps,
        r=0.7,
        batch_size=256,
        win_sec=45, perc=10,
        cutoff_hz=5.0, sg_win_ms=333, sg_poly=3,
        out_dir=None, prefix=''
):
    """
    Process Suite2p fluorescence traces through neuropil correction, ΔF/F computation,
    low-pass filtering, and derivative calculation.

    F_cell, F_neuropil: arrays from Suite2p (nROIs, T) or (T, nROIs) — auto-handled.
    Writes memmap outputs to disk to avoid RAM blowups.

    Returns:
        tuple: (dff_path, lowpass_path, derivative_path)
    """
    # Step 1: Normalize and transpose arrays to time-major format
    F_cell, F_neuropil, num_timepoints, num_rois = _normalize_and_transpose_arrays(
        F_cell, F_neuropil
    )

    # Step 2: Set up output memory-mapped arrays
    dff_memmap, lowpass_memmap, derivative_memmap, file_paths = _setup_output_memmaps(
        out_dir, prefix, num_timepoints, num_rois
    )

    # Step 3: Precompute filter coefficients (reused for all cells)
    sos = _compute_filter_coefficients(fps, cutoff_hz, filter_order=2)

    # Step 4: Process cells in batches
    for batch_start_idx in range(0, num_rois, batch_size):
        batch_start_time = time.time()
        batch_end_idx = min(num_rois, batch_start_idx + batch_size)

        F_cell_batch = F_cell[:, batch_start_idx:batch_end_idx]
        F_neuropil_batch = F_neuropil[:, batch_start_idx:batch_end_idx]

        _process_cell_batch(
            F_cell_batch, F_neuropil_batch, r, batch_start_idx,
            fps, win_sec, perc, cutoff_hz, sg_win_ms, sg_poly, sos,
            dff_memmap, lowpass_memmap, derivative_memmap
        )

        batch_duration = time.time() - batch_start_time
        print(f"Processed cells {batch_start_idx}–{batch_end_idx - 1} / {num_rois - 1} "
              f"in {batch_duration:.2f} seconds.")

    # Step 5: Clean up and return file paths
    del dff_memmap, lowpass_memmap, derivative_memmap

    dff_path, lowpass_path, derivative_path = file_paths
    return dff_path, lowpass_path, derivative_path


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

    cutoff_hz = custom_lowpass_cutoff(cutoffs, "human_SLE_2p_meta.csv", sample_name)
    print(f'cutoff_hz: {cutoff_hz}')

    batch_size = utils.change_batch_according_to_free_ram()*20

    dff_path, low_path, dt_path = process_suite2p_traces(
        F_cell, F_neu, fps,
        r=0.7,
        batch_size=batch_size,
        win_sec=45, perc=10,
        cutoff_hz=cutoff_hz, sg_win_ms=333, sg_poly=2,
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


