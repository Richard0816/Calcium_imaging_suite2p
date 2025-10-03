import numpy as np
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

    num_frames, num_rois, time_major = utils.s2p_infer_orientation(F_cell)

    if not time_major:
        F_cell = F_cell.T
        F_neuropil = F_neuropil.T
    # (sanity) make sure shapes still align
    if F_cell.shape != F_neuropil.shape or F_cell.shape != (num_frames, num_rois):
        raise ValueError("F and Fneu shapes do not align after orientation handling.")

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

    sos_local = sos

    # Process each cell in the batch
    for cell_idx in range(num_cells_in_batch):
        trace = corrected_batch[:, cell_idx]
        global_cell_idx = cell_start_idx + cell_idx

        # 1) ΔF/F (robust)
        dff = utils.robust_df_over_f_1d(trace, win_sec=win_sec, perc=perc, fps=fps)

        # 2) Low-pass (build SOS once, then reuse)
        lowpass_filtered, _, sos_local = utils.lowpass_causal_1d(
                    dff, fps=fps, cutoff_hz=cutoff_hz, order=2, zi=None, sos=sos_local)

        # 3) SG first derivative
        derivative = utils.sg_first_derivative_1d(lowpass_filtered, fps=fps, win_ms=sg_win_ms, poly=sg_poly)


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

    # Step 3: initialized SOS filter coefficients will be calculated when first cell run
    sos = None

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


