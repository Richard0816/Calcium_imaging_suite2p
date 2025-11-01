import numpy as np
import os
import time
import utils
from spatial_heatmap import compute_cell_scores, soft_cell_mask, SpatialHeatmapConfig, _load_suite2p_data

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

def filter_rois_by_cell_score(root, fps=30.0, z_enter=3.5, z_exit=1.5, min_sep_s=0.3,
                              w_er=1.0, w_pz=1.2, w_area=0.4,
                              scale_er=1.0, scale_pz=3.0, scale_area=50.0,
                              bias=-2.0, score_threshold=0.5, top_k_pct=None):
    """
    Compute weighted cell scores for all ROIs and return a boolean mask
    selecting only those that pass the threshold or top-k%.
    """
    cfg = SpatialHeatmapConfig(
        folder_name=root.replace("suite2p\\plane0\\", ""),
        fps=fps, z_enter=z_enter, z_exit=z_exit, min_sep_s=min_sep_s,
        bin_seconds=None
    )
    data = _load_suite2p_data(cfg)
    scores = compute_cell_scores(
        data, cfg,
        w_er=w_er, w_pz=w_pz, w_area=w_area,
        scale_er=scale_er, scale_pz=scale_pz, scale_area=scale_area,
        bias=bias
    )
    mask = soft_cell_mask(scores, score_threshold=score_threshold, top_k_pct=top_k_pct)
    n_pass = int(mask.sum())
    print(f"[Filter] Retained {n_pass}/{len(mask)} ROIs ({100*n_pass/len(mask):.1f}%)")
    np.save(os.path.join(root, "r0p7_cell_mask_bool.npy"), mask)
    return mask

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

    weights = [2.3662, 1.0454, 1.1252, 0.2987]  # (bias, er, pz, area)
    sd_mu = [4.079, 11.24, 41.178]
    sd_sd = [1.146, 4.214, 37.065]
    thres = 0.68
    bias = float(
        weights[0]
        - (weights[1] * sd_mu[0] / sd_sd[0])
        - (weights[2] * sd_mu[1] / sd_sd[1])
        - (weights[3] * sd_mu[2] / sd_sd[2])
    )

    mask = filter_rois_by_cell_score(
        root,
        fps=fps, z_enter=3.5, z_exit=1.5, min_sep_s=0.3,
        w_er=weights[1], w_pz=weights[2], w_area=weights[3],
        scale_er=float(sd_sd[0]),  # ~1 event/min considered “unit”
        scale_pz=float(sd_sd[1]),  # z≈5 as a “unit bump”
        scale_area=float(sd_sd[2]),  # 10 px as a “unit area”
        bias=bias,  # stricter overall
        score_threshold=thres,  # classify as cell if P>=0.5
        top_k_pct=None  # or set e.g. 25 for top-25% only
    )
    # Apply mask to F and Fneu
    F_cell = F_cell[mask, :]
    F_neu = F_neu[mask, :]
    print(f"[Filter] After applying mask: {F_cell.shape[0]} ROIs retained.")

    # Where to write outputs
    out_dir = root  # save alongside Suite2p
    # Optional: a prefix for filenames so you can run variants without clobbering
    prefix = 'r0p7_filtered_'  # e.g., indicates r=0.7

    print(f'Processing {sample_name}')

    cutoffs = {
        "6f": 5.0,
        "6m": 5.0,
        "6s": 5.0,
        "8m": 3.0
    }

    cutoff_hz = custom_lowpass_cutoff(cutoffs, "human_SLE_2p_meta.csv", sample_name)
    print(f'cutoff_hz: {1.0}')

    batch_size = utils.change_batch_according_to_free_ram()*20

    dff_path, low_path, dt_path = process_suite2p_traces(
        F_cell, F_neu, fps,
        r=0.7,
        batch_size=batch_size,
        win_sec=45, perc=10,
        cutoff_hz=1.0, sg_win_ms=333, sg_poly=2,
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
    run_analysis_on_folder(r'F:\data\2p_shifted\Hip\2024-11-05_00003')
    #utils.log("fluorescence_analysis.log", run)


