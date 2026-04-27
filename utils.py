from __future__ import annotations

import os
import io
import contextlib
import sys
import numpy as np
import re
import pandas as pd
import psutil
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.signal import butter, sosfilt, savgol_filter, find_peaks
from scipy.special import erfinv
from pathlib import Path
from typing import Tuple, Optional, Sequence, Union
from dataclasses import dataclass, field

import matplotlib.pyplot as plt


# ----------- Mass function deployment + logging functionality -----------
class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)


def change_batch_according_to_free_ram() -> int:
    """
    Not totally necessary but changes batch size according to how much memory we have
    :return: The updated ops data after appropriate adjustment has been made
    """
    # calculate the current available memory
    available_mem = round(psutil.virtual_memory().available / (1024 ** 3), 1)

    # ensures that the minimum batch size is 100, even if we are running super low on memory
    if available_mem <= 13.5:
        return 100

    # calculates the batch size based on a linear relationship between memory and run
    else:
        return int(
            20 * available_mem - 170)  # calculated using the two point form of the linear eqn (16, 150), (200, 4000)


def _peek_tiff_frame_shape(tiff_folder: str):
    """Return (Ly, Lx) from the first TIFF in the folder, or None if unavailable."""
    try:
        import tifffile
    except ImportError:
        return None
    try:
        for name in sorted(os.listdir(tiff_folder)):
            if name.lower().endswith(('.tif', '.tiff')):
                with tifffile.TiffFile(os.path.join(tiff_folder, name)) as tf:
                    page = tf.pages[0]
                    shape = page.shape
                    if len(shape) >= 2:
                        return int(shape[-2]), int(shape[-1])
                break
    except Exception:
        return None
    return None


def change_nbinned_according_to_free_ram(tiff_folder: str,
                                         ram_fraction: float = 0.35,
                                         default: int = 1500,
                                         floor: int = 500,
                                         ceiling: int = 5000,
                                         peak_multiplier: float = 3.0) -> int:
    """
    Cap Suite2p's ``nbinned`` so sparsery's peak intermediate fits in RAM.

    In practice sparsery's peak footprint is larger than a single
    ``(nbinned, Ly, Lx) float32`` array: ``neuropil_subtraction`` allocates
    ``np.zeros_like(mov)`` (×2), plus suite2p holds other intermediates and
    numpy allocator overhead. We model the peak as
    ``peak_multiplier * nbinned * Ly * Lx * 4`` bytes (default ~3×) and
    budget ``ram_fraction`` of currently available RAM for it.

    Falls back to ``default`` if the TIFF frame shape can't be read.
    """
    available_bytes = psutil.virtual_memory().available
    budget = available_bytes * ram_fraction

    shape = _peek_tiff_frame_shape(tiff_folder)
    if shape is None:
        print(f"[nbinned] could not peek TIFF shape in {tiff_folder}; "
              f"using default={default}")
        return default

    Ly, Lx = shape
    bytes_per_nbinned = Ly * Lx * 4 * peak_multiplier
    if bytes_per_nbinned <= 0:
        return default

    nbinned = int(budget // bytes_per_nbinned)
    capped = max(floor, min(ceiling, nbinned))
    avail_gb = available_bytes / (1024 ** 3)
    print(f"[nbinned] avail={avail_gb:.1f}GB frame={Ly}x{Lx} "
          f"raw_nbinned={nbinned} -> {capped} "
          f"(peak~{capped * Ly * Lx * 4 * peak_multiplier / 1024 ** 3:.2f}GB)")
    return capped


def run_on_folders(parent_folder: str, func, log_filename, addon_vals: list = None, leaf_folders_only=False) -> None:
    """
    Run a custom function on every subfolder inside a parent folder (non-recursive).

    :param leaf_folders_only: only look for folders that contain no subfolders False by default
    :param func: the function that you want to run
    :param addon_vals: A list of values that are needed (optional, will pass None if none given)
    :param parent_folder: Path to the parent folder.

    then
    Logs the output of a given function to a specified log file as well as the standard
    output. It intercepts the output of both standard output and standard error streams
    and writes them to the specified log file whilst still displaying them in the console.

    The function uses a helper class `Tee` and `contextlib.redirect_stdout` and
    `contextlib.redirect_stderr` to handle output redirection.
    :param log_filename: The path to the log file where the output should be stored.
    """

    for entry in os.scandir(parent_folder):
        logfile = open(log_filename, "a", encoding="utf-8")
        tee = Tee(sys.__stdout__, logfile)

        # running in here just to store the output in the logfile
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            if entry.is_dir():
                has_subfolders = any(sub.is_dir() for sub in os.scandir(entry.path))
                if leaf_folders_only and has_subfolders:  # if we are looking for only folders with no subfolders and we find a subfolder, skip this entry
                    continue

                if addon_vals:  # if we have values to pass, pass the values
                    func(entry.path, addon_vals)
                else:  # if there's nothing just pass the path
                    func(entry.path)
        logfile.close()


'''def log(log_filename: str, run_function) -> None:
    """
    Logs the output of a given function to a specified log file as well as the standard
    output. It intercepts the output of both standard output and standard error streams
    and writes them to the specified log file whilst still displaying them in the console.

    The function uses a helper class `Tee` and `contextlib.redirect_stdout` and
    `contextlib.redirect_stderr` to handle output redirection.

    :param log_filename: The path to the log file where the output should be stored.
    :param run_function: The function to be executed, whose output is to be logged.
    :return: None
    """
    logfile = open(log_filename, "a")
    tee = Tee(sys.__stdout__, logfile)

    # running in here just to store the output in the logfile
    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
        run_function()

    logfile.close()'''


# ---- Suite2p I/O + orientation ----
def s2p_infer_orientation(F: np.ndarray) -> tuple[int, int, bool]:
    """
    Suite2p's F.npy / Fneu.npy are canonically shaped (N_ROIs, T). Trust that
    layout. Returns (num_frames, num_rois, time_major) where time_major=False.

    (The previous n_rows<n_cols heuristic silently broke when N_ROIs >= T.)
    """
    if F.ndim != 2:
        raise ValueError(f"Expected 2D array, got {F.shape}")
    n_rois, n_frames = F.shape
    return n_frames, n_rois, False


def s2p_load_raw(root: Union[str, Path]) -> tuple[np.ndarray, np.ndarray, int, int, bool]:
    """
    Load F.npy and Fneu.npy, and return (F, Fneu, num_frames, num_rois, time_major).
    """
    root = Path(root)
    F = np.load(root / "F.npy", allow_pickle=False)
    Fneu = np.load(root / "Fneu.npy", allow_pickle=False)
    if F.shape != Fneu.shape:
        raise ValueError(f"F and Fneu shapes differ: {F.shape} vs {Fneu.shape}")
    num_frames, num_rois, time_major = s2p_infer_orientation(F)
    return F, Fneu, num_frames, num_rois, time_major


def s2p_open_memmaps(root: Union[str, Path], prefix: str = "r0p7_") -> tuple[np.memmap, np.memmap, np.memmap, int, int]:
    """
    Open ΔF/F, low-pass ΔF/F, and derivative Suite2p memmaps with a given prefix.
    Returns (dff, low, dt, T, N) with shape (T, N).
    """
    root = Path(root)
    # We infer num_frames, num_rois from one of the files by opening in read-only "r" mode with a guess.
    # To avoid loading whole arrays, we require caller to give num_frames, num_rois; but many scripts
    # already know num_frames, num_rois from raw F. When they don't, we can do a small dance:
    # Here, we first read F to get num_frames, num_rois which is robust.
    F, _, num_frames, num_rois, _ = s2p_load_raw(root)

    if prefix.split("_")[-2] == "filtered":
        mask = np.load(root / "r0p7_cell_mask_bool.npy", allow_pickle=False)
        F = F[mask, :]
        num_rois = mask.sum()

    dff = np.memmap(root / f"{prefix}dff.memmap.float32", dtype="float32", mode="r", shape=(num_frames, num_rois))
    low = np.memmap(root / f"{prefix}dff_lowpass.memmap.float32", dtype="float32", mode="r",
                    shape=(num_frames, num_rois))
    dt = np.memmap(root / f"{prefix}dff_dt.memmap.float32", dtype="float32", mode="r", shape=(num_frames, num_rois))
    return dff, low, dt, num_frames, num_rois


# ----------- Signal Processing -----------
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


def first_n_min_df_over_f_1d(F, baseline_min=2.0, perc=10, fps=30.0):
    """Constant baseline taken from the first ``baseline_min`` minutes.

    F0 = percentile(F[:N], perc) where N = baseline_min * 60 * fps,
    then dF/F = (F - F0) / max(F0, eps). Use this when the recording
    has a clean pre-stimulus window and a rolling baseline would smear
    long sustained events into the baseline itself.
    """
    F = np.asarray(F, dtype=np.float32)
    n = F.size

    finite = np.isfinite(F)
    if not finite.all():
        F = np.interp(np.arange(n), np.flatnonzero(finite),
                      F[finite]).astype(np.float32)

    n_baseline = max(1, int(round(float(baseline_min) * 60.0 * float(fps))))
    n_baseline = min(n_baseline, n)
    F0_scalar = float(np.nanpercentile(F[:n_baseline], perc))
    eps = max(F0_scalar, 1e-9)
    return ((F - F0_scalar) / eps).astype(np.float32)


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


def mad_z(x):
    """
    Robust z-score using MAD (per ROI), with 1.4826 factor to approximate σ for normal data. (stolen from Stern et al. 2024)
    Returns z, and the median/MAD for optional inverse transforms.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad), med, mad


def hysteresis_onsets(z, z_hi, z_lo, fps, min_sep_s=0.0):
    """
    Hysteresis onset detection on robust z:
      - enter when z >= z_hi; exit when z <= z_lo
      - merge onsets closer than min_sep_s (sec)
    Returns onset frame indices (np.int32).
    """
    above_hi = z >= z_hi
    onsets = []
    active = False
    for i in range(z.size):
        if not active and above_hi[i]:
            active = True
            onsets.append(i)
        elif active and z[i] <= z_lo:
            active = False
    if not onsets:
        return np.array([], dtype=int)
    onsets = np.array(onsets, dtype=int)
    if min_sep_s > 0:
        min_sep = int(min_sep_s * fps)
        merged = [onsets[0]]
        for k in onsets[1:]:
            if k - merged[-1] >= min_sep:
                merged.append(k)
        onsets = np.asarray(merged, dtype=int)
    return onsets


# ----------- Analysis -----------
def roi_metric(values, which='event_rate', t_slice=slice(None), fps=30.0, z_enter=3.5, z_exit=1.5, min_sep_s=0.3):
    """
    Computes a region of interest (ROI) metric based on the specified type.

    This function processes time-series data to compute metrics such as
    event rates, mean delta F/F (dff), or peak robust z-scores for the provided
    regions of interest. The chosen metric depends on the `which`"""

    # Extract the low-pass filtered data (delta F/F) for the selected time slice
    # Shape: (Tsel, N) where Tsel = number of time frames, N = number of ROIs
    lp = values['low'][t_slice]  # (Tsel, N)

    # Extract the detrended data for the selected time slice
    # This is used for z-score calculations and event detection
    dd = values['dt'][t_slice]  # (Tsel, N)

    # Get the number of time frames in the selected slice
    Tsel = lp.shape[0]

    # Initialize output array with zeros, one value per ROI
    # dtype=float32 for memory efficiency
    out = np.zeros(lp.shape[1], dtype=np.float32)

    # Branch 1: Calculate mean delta F/F across time for each ROI
    if which == 'mean_dff':
        # Compute mean across time axis (axis=0), ignoring NaN values
        # Convert to float32 for consistency
        out = np.nanmean(lp, axis=0).astype(np.float32)

    # Branch 2: Calculate peak robust z-score for each ROI
    elif which == 'peak_dz':
        # Peak robust z per ROI (loop over columns for per-ROI MAD)

        # Create empty array to store z-scores with same shape as detrended data
        z = np.empty_like(dd, dtype=np.float32)

        # Loop through each ROI (column)
        for j in range(dd.shape[1]):
            # Calculate robust z-score using MAD for this ROI's time series
            # mad_z returns (z-score, median, mad); we only need the z-score
            zj, _, _ = mad_z(dd[:, j])

            # Store z-scores for this ROI in the z array
            z[:, j] = zj

        # Find maximum z-score across time for each ROI
        # This represents the peak activity level
        out = np.nanmax(z, axis=0).astype(np.float32)

    # Branch 3: Calculate event rate (events per minute) for each ROI
    elif which == 'event_rate':
        # Count onsets per ROI and divide by duration (min) → events/min

        # Initialize array to count number of events detected in each ROI
        counts = np.zeros(dd.shape[1], dtype=np.int32)

        # Loop through each ROI to detect and count events
        for j in range(dd.shape[1]):
            # Calculate robust z-score for this ROI's time series
            zj, _, _ = mad_z(dd[:, j])

            # Detect event onsets using hysteresis thresholding
            # Events start when z >= z_enter and end when z <= z_exit
            # Returns array of frame indices where events begin
            on = hysteresis_onsets(zj, z_enter, z_exit, fps, min_sep_s=min_sep_s)

            # Count the number of detected events (size of onset array)
            counts[j] = on.size

        # Convert total time frames to minutes: frames / (frames/sec) / (sec/min)
        duration_min = Tsel / fps / 60.0

        # Calculate event rate: events / minutes
        # Use max() to avoid division by zero (minimum denominator = 1e-9)
        out = (counts / max(duration_min, 1e-9)).astype(np.float32)
    else:
        raise ValueError("metric must be one of: 'event_rate', 'mean_dff', 'peak_dz'")

    return out


def paint_spatial(values_per_roi, stat_list, Ly, Lx):
    """
    Paint per-ROI scalar values onto the imaging plane using ROI masks.
    Uses 'lam' weights for soft assignment; normalizes by accumulated weight.
    Returns (Ly, Lx) float32 image.
    """
    # Initialize the output image array with zeros, shape matches imaging plane dimensions
    img = np.zeros((Ly, Lx), dtype=np.float32)

    # Initialize weight accumulator array to track total lambda weights per pixel
    w = np.zeros((Ly, Lx), dtype=np.float32)

    # Iterate through each ROI and its corresponding statistics dictionary
    for j, s in enumerate(stat_list):
        # Get the scalar value (metric) for the current ROI
        v = values_per_roi[j]

        # Extract y and x-coordinates of all pixels belonging to this ROI
        ypix = s['ypix']
        xpix = s['xpix']

        # Extract lambda weights (pixel-wise contribution strengths) and convert to float32
        lam = s['lam'].astype(np.float32)

        # Add weighted ROI value to image: each pixel gets v * its lambda weight
        img[ypix, xpix] += v * lam

        # Accumulate the lambda weights at each pixel (for normalization)
        w[ypix, xpix] += lam

    # Create boolean mask identifying pixels with non-zero accumulated weights
    m = w > 0

    # Normalize weighted values by dividing by accumulated weights (only where w > 0)
    img[m] /= w[m]

    return img


def build_time_mask(time: np.ndarray, t_max: Union[float, None]) -> Union[np.ndarray, slice]:
    """
    Return a boolean mask for time < t_max, or slice(None) if t_max is None.
    """
    if t_max is None:
        return slice(None)
    return np.asarray(time) < float(t_max)


def aav_cleanup_and_dictionary_lookup(aav: str, dic: dict) -> float:
    """
    :param dic: a dictionary containing aav information as keys and a translation as items
    :param aav: The aav used in the experiment (string pulled from metadata)
    :return: float of the recommended dictionary value to use based on aav information
    """
    # drop any usage of "rg"
    aav = aav.replace("rg", "")

    # split by -, _, or + (list output)
    components = re.split(r"[-_+]", aav)

    # make both the keys and list case-insensitive, and then match list and keys to find 6f, 6m, 8m etc
    dict_lower = {k.lower(): v for k, v in dic.items()}
    list_lower = {item.lower() for item in components}

    # Find intersection (common keys)
    common = dict_lower.keys() & list_lower

    # Return value for the first common key
    return dict_lower[next(iter(common))]


def get_row_number_csv_module(csv_filename: str, header_name: str, target_element: str) -> int:
    """
    Find the row number of a given target element under a specific header column in a CSV file.
    The comparison is done by splitting strings into lists of integers (delimiters: '-' and '_')
    and checking if the lists are identical.

    :param csv_filename: Path to the CSV file.
    :param header_name: The name of the column header to search under.
    :param target_element: The element to look for in the specified column.
    :return: The row number (1-based, not counting the header) if found, otherwise None.
    """
    try:
        # Read only the target column from the CSV file for efficiency
        col = pd.read_csv(csv_filename, usecols=[header_name])
    except ValueError:
        # If the header doesn't exist, return None
        print(f"Error: Header '{header_name}' not found in the CSV file.")

        return None

    # Helper to convert string into list of integers split by "-" or "_"
    def to_int_list(s: str):
        return [int(x) for x in re.split(r"[-_]", str(s)) if x.isdigit()]

    # Convert target element into integer list
    target_list = to_int_list(target_element)

    # Go through column and find first matching row
    for idx, val in col[header_name].items():
        if to_int_list(val) == target_list:
            return idx + 1  # 1-based row number

    return None


def file_name_to_aav_to_dictionary_lookup(file_name, aav_info_csv, dic):
    """
    Look up a given file name in the CSV file and perform operations to retrieve
    a dictionary value using corresponding AAV information.

    :param file_name: The name of the file to look up in the CSV file
    :type file_name: str
    :param aav_info_csv: The path to the CSV file containing AAV information
        and video details
    :type aav_info_csv: str
    :param dic: The dictionary to perform a lookup using the cleaned AAV value
    :type dic: dict
    :return: The retrieved value from the dictionary after performing the lookup
    :rtype: Any
    """
    # look up the file name in aav_info under "video"
    # get the row number of file name
    row_num = get_row_number_csv_module(aav_info_csv, 'video', file_name)

    # get aav information
    col = pd.read_csv(aav_info_csv, usecols=["AAV"])  # Read only the needed column
    element = col["AAV"].iloc[row_num - 1]  # row_num is 1-based
    element = str(element)

    # get the dictionary value
    dictionary_value = aav_cleanup_and_dictionary_lookup(element, dic)

    return dictionary_value


RECORDING_DIR_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d+")


def _find_recording_root(path: Path) -> Path | None:
    """
    Walk upward until we find a folder named YYYY-MM-DD_#####.
    """
    for p in [path] + list(path.parents):
        if RECORDING_DIR_RE.fullmatch(p.name):
            return p
    return None


def get_fps_from_notes(
        path: str,
        notes_root: str = r"F:\notes_recordings",
        default_fps: float = 15.07,
) -> float:
    """
    Resolve FPS for a recording, given ANY path inside that recording.
    Safe fallback to default_fps if metadata is missing.
    """
    try:
        path = Path(path)

        rec_root = _find_recording_root(path)
        if rec_root is None:
            return default_fps

        date_str, rec_str = rec_root.name.split("_", 1)
        target = f"{date_str}-{rec_str}"

        notes_root = Path(notes_root)
        notes_path = notes_root / f"{date_str}.xlsx"

        if not notes_path.exists():
            candidates = sorted(notes_root.glob(f"*{date_str}*.xlsx"))
            if not candidates:
                return default_fps
            notes_path = candidates[0]

        df = pd.read_excel(notes_path, sheet_name="2P settings")
        df.columns = [str(c).strip() for c in df.columns]
        cols = {c.lower(): c for c in df.columns}

        if "filename" not in cols or "rate (hz)" not in cols:
            return default_fps

        fn_col = cols["filename"]
        rate_col = cols["rate (hz)"]

        fn = df[fn_col].astype(str).str.strip()

        hits = df.loc[fn == target]

        if hits.empty:
            hits = df.loc[
                fn.str.endswith(f"-{rec_str}", na=False) |
                (fn == rec_str)
                ]

        if hits.empty:
            return default_fps

        rate_val = hits.iloc[0][rate_col]
        if pd.isna(rate_val):
            return default_fps

        return float(rate_val)

    except Exception:
        return default_fps


def get_zoom_from_notes(
        path: str,
        notes_root: str = r"F:\notes_recordings",
        default_zoom: float = 1.0,
) -> float:
    """
    Resolve zoom for a recording, given ANY path inside that recording.
    Safe fallback to default_zoom if metadata is missing.
    """
    try:
        path = Path(path)

        rec_root = _find_recording_root(path)
        if rec_root is None:
            return default_zoom

        date_str, rec_str = rec_root.name.split("_", 1)
        target = f"{date_str}-{rec_str}"

        notes_root = Path(notes_root)
        notes_path = notes_root / f"{date_str}.xlsx"

        if not notes_path.exists():
            candidates = sorted(notes_root.glob(f"*{date_str}*.xlsx"))
            if not candidates:
                return default_zoom
            notes_path = candidates[0]

        df = pd.read_excel(notes_path, sheet_name="2P settings")
        df.columns = [str(c).strip() for c in df.columns]
        cols = {c.lower(): c for c in df.columns}

        # IMPORTANT: use .lower() match for "zoom"
        if "filename" not in cols or "zoom" not in cols:
            return default_zoom

        fn_col = cols["filename"]
        zoom_col = cols["zoom"]

        fn = df[fn_col].astype(str).str.strip()
        hits = df.loc[fn == target]

        if hits.empty:
            hits = df.loc[
                fn.str.endswith(f"-{rec_str}", na=False) |
                (fn == rec_str)
                ]

        if hits.empty:
            return default_zoom

        zval = hits.iloc[0][zoom_col]
        if pd.isna(zval):
            return default_zoom

        return float(zval)

    except Exception:
        return default_zoom


# -------- EVENT DETECTION  --------
# ---------- config ----------

@dataclass
class EventDetectionParams:
    """
    All parameters for density-based event detection and boundary refinement.
    Sensible defaults are chosen for 2P calcium imaging at 15Hz.
    """
    # Density construction
    bin_sec: float = 0.05
    smooth_sigma_bins: float = 2.0
    normalize_by_num_rois: bool = True

    # Peak detection on smoothed density (scipy.signal.find_peaks)
    min_prominence: float = 0.007
    min_width_bins: float = 2.0
    min_distance_bins: float = 3.0

    # Baseline / noise (for boundary walking)
    baseline_mode: str = "rolling"  # "rolling" or "global"
    baseline_percentile: float = 5.0
    baseline_window_s: float = 30.0
    noise_quiet_percentile: float = 40.0
    noise_mad_factor: float = 1.4826

    # Boundary walking
    end_threshold_k: float = 2.0  # end = baseline + k * noise
    max_event_duration_s: float = 10.0

    # Overlap merging
    merge_gap_s: float = 0.0  # merge events closer than this

    # Gaussian-fit refinement (whichever comes first wrt baseline walk)
    use_gaussian_boundary: bool = True
    gaussian_quantile: float = 0.99
    gaussian_fit_pad_s: float = 0.5
    gaussian_min_sigma_s: float = 0.05


# ---------- top-level entry point ----------

def detect_event_windows(
        onsets_by_roi: Sequence[np.ndarray],
        T: int,
        fps: float,
        params: Optional[EventDetectionParams] = None,
        return_diagnostics: bool = False,
):
    """
    Detect population events from per-ROI onset times and return event windows
    alongside per-event activation matrices ready to substitute for the old
    (A, first_time, keep_bins) triplet.

    Parameters
    ----------
    onsets_by_roi : list of 1D arrays
        Per-ROI onset times in SECONDS (already in recording time, not frames).
        Equivalent to the output of _event_onsets_by_roi / extract_onsets_by_roi.
        Length = number of ROIs (post cell-mask filtering, if any).
    T : int
        Number of frames in the recording.
    fps : float
        Sampling rate in Hz.
    params : EventDetectionParams, optional
        Detection / boundary parameters. Uses defaults if None.
    return_diagnostics : bool
        If True, also returns a dict with baseline, noise, peak heights,
        gaussian fits, etc. -- useful for plotting and debugging.

    Returns
    -------
    event_windows : (n_events, 2) float array
        Per-event [start_s, end_s] in seconds.
    A : (N_rois, n_events) bool array
        A[i, e] = True iff ROI i had >= 1 onset inside event e's window.
    first_time : (N_rois, n_events) float array
        first_time[i, e] = time (s) of ROI i's earliest onset inside event e,
        NaN if inactive.
    diagnostics : dict (only if return_diagnostics=True)
        Contains: time_centers_s, binned_density, smoothed_density,
        baseline_trace, end_threshold_trace, baseline_noise,
        peak_s, peak_height, mu_s, sigma_s,
        boundary_source_left, boundary_source_right, prominence, duration_s.
    """
    if params is None:
        params = EventDetectionParams()

    # 1) Flat onset times across all ROIs -> density
    duration_s = float(T) / float(fps)
    centers, counts, smooth = _build_density(
        onsets_by_roi=onsets_by_roi,
        duration_s=duration_s,
        bin_sec=params.bin_sec,
        smooth_sigma_bins=params.smooth_sigma_bins,
        n_rois=len(onsets_by_roi),
        normalize_by_num_rois=params.normalize_by_num_rois,
    )

    # 2) Peak detection
    peak_indices = _detect_density_peaks(
        smooth=smooth,
        min_prominence=params.min_prominence,
        min_width_bins=params.min_width_bins,
        min_distance_bins=params.min_distance_bins,
    )

    # 3) Per-event boundaries
    boundaries = _boundaries_from_peaks(
        time_s=centers,
        smooth=smooth,
        peak_indices=peak_indices,
        params=params,
    )

    event_windows = np.column_stack([boundaries["start_s"], boundaries["end_s"]])

    # 4) Activation matrix + first_time per event (drop-in for _activation_matrix)
    A, first_time = _activation_matrix_from_windows(onsets_by_roi, event_windows)

    if not return_diagnostics:
        return event_windows, A, first_time

    diagnostics = {
        "time_centers_s": centers,
        "binned_density": counts,
        "smoothed_density": smooth,
        "baseline_trace": boundaries["baseline_trace"],
        "end_threshold_trace": boundaries["end_threshold_trace"],
        "baseline_noise": boundaries["baseline_noise"],
        "peak_s": boundaries["peak_s"],
        "peak_height": boundaries["peak_height"],
        "mu_s": boundaries["mu_s"],
        "sigma_s": boundaries["sigma_s"],
        "boundary_source_left": boundaries["boundary_source_left"],
        "boundary_source_right": boundaries["boundary_source_right"],
        "prominence": boundaries["prominence"],
        "duration_s": boundaries["duration_s"],
    }
    return event_windows, A, first_time, diagnostics


# ---------- plotting helper ----------

def plot_event_detection(
        diagnostics: dict,
        ylabel: str = "Onset density (per bin per ROI)",
        title: str = "Detected events on onset density",
        ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Figure: binned counts (bars) + smoothed density (line) + baseline +
    end threshold + shaded event windows + peak markers.
    Takes the `diagnostics` dict returned by detect_event_windows(..., return_diagnostics=True).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4.5))
    else:
        fig = ax.figure

    centers = diagnostics["time_centers_s"]
    smooth = diagnostics["smoothed_density"]
    counts = diagnostics["binned_density"]

    bin_width = float(np.median(np.diff(centers))) if centers.size > 1 else 1.0

    ax.bar(centers, counts, width=bin_width, alpha=0.35, align="center",
           color="C0", edgecolor="none", label="Binned counts")
    ax.plot(centers, smooth, linewidth=1.5, color="C0", label="Smoothed density")
    ax.plot(centers, diagnostics["baseline_trace"], color="gray", linestyle=":",
            linewidth=1.0, label="Baseline")
    ax.plot(centers, diagnostics["end_threshold_trace"], color="black",
            linestyle="--", linewidth=1.0, label="End threshold")

    peak_s = diagnostics["peak_s"]
    peak_h = diagnostics["peak_height"]
    ax.plot(peak_s, peak_h, "v", color="C3", markersize=6,
            label=f"Peaks (n={peak_s.size})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def shade_event_windows(ax: plt.Axes, event_windows: np.ndarray, color="C1", alpha=0.20):
    """Overlay shaded event windows onto an existing axis."""
    for s, e in event_windows:
        ax.axvspan(s, e, color=color, alpha=alpha)


# =====================================================================
# Internals
# =====================================================================

def _build_density(
        onsets_by_roi: Sequence[np.ndarray],
        duration_s: float,
        bin_sec: float,
        smooth_sigma_bins: float,
        n_rois: int,
        normalize_by_num_rois: bool,
):
    """Flatten onsets, histogram them, and Gaussian-smooth."""
    nonempty = [np.asarray(x, dtype=np.float64) for x in onsets_by_roi if len(x) > 0]
    flat = np.concatenate(nonempty) if nonempty else np.array([], dtype=np.float64)

    edges = np.arange(0.0, duration_s + bin_sec, bin_sec, dtype=np.float64)
    if edges[-1] < duration_s:
        edges = np.append(edges, duration_s)

    counts, edges = np.histogram(flat, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    counts = counts.astype(np.float64)
    if normalize_by_num_rois and n_rois > 0:
        counts = counts / float(n_rois)

    smooth = gaussian_filter1d(counts, sigma=smooth_sigma_bins, mode="nearest")
    return centers, counts, smooth


def _detect_density_peaks(
        smooth: np.ndarray,
        min_prominence: float,
        min_width_bins: float,
        min_distance_bins: float,
) -> np.ndarray:
    """Run scipy find_peaks with the same gates as event_detection.detect_density_peaks."""
    peaks, _props = find_peaks(
        smooth,
        prominence=min_prominence,
        width=min_width_bins,
        distance=max(1, int(round(min_distance_bins))),
    )
    return peaks


# ---- baseline / noise ----

def _estimate_global_baseline(density: np.ndarray, baseline_percentile: float) -> np.ndarray:
    if density.size == 0:
        return np.zeros_like(density)
    b = float(np.percentile(density, baseline_percentile))
    return np.full_like(density, b, dtype=np.float64)


def _estimate_rolling_baseline(
        density: np.ndarray,
        fps_density: float,
        baseline_window_s: float,
        baseline_percentile: float,
) -> np.ndarray:
    if density.size == 0:
        return np.zeros_like(density)
    win = max(3, int(round(baseline_window_s * fps_density)) | 1)
    win = min(win, density.size if density.size % 2 == 1 else density.size - 1)
    if win < 3:
        return _estimate_global_baseline(density, baseline_percentile)
    return percentile_filter(
        density.astype(np.float64),
        size=win,
        percentile=baseline_percentile,
        mode="reflect",
    )


def _estimate_noise_from_quiet(
        density: np.ndarray,
        baseline_trace: np.ndarray,
        quiet_percentile: float = 40.0,
        noise_mad_factor: float = 1.4826,
) -> float:
    if density.size == 0:
        return 0.0
    resid = density - baseline_trace
    cutoff = float(np.percentile(resid, quiet_percentile))
    quiet_mask = resid <= cutoff
    if quiet_mask.sum() < 32:
        quiet_mask = np.ones_like(resid, dtype=bool)
    sample = resid[quiet_mask]
    med = np.median(sample)
    mad = np.median(np.abs(sample - med)) + 1e-12
    return float(noise_mad_factor * mad)


# ---- boundary walk ----

def _walk_boundary(
        density: np.ndarray,
        peak_idx: int,
        end_threshold_trace: np.ndarray,
        direction: int,
        max_steps: int,
) -> int:
    n = density.size
    i = peak_idx
    steps = 0
    while 0 <= i + direction < n and steps < max_steps:
        nxt = i + direction
        if density[nxt] <= end_threshold_trace[nxt]:
            return i
        i = nxt
        steps += 1
    return i


# ---- Gaussian fit ----

def _gaussian_z(q: float) -> float:
    if not (0.5 < q < 1.0):
        raise ValueError("gaussian_quantile must be in (0.5, 1.0)")
    return float(np.sqrt(2.0) * erfinv(2.0 * q - 1.0))


def _fit_gaussian_to_peak(
        time_s: np.ndarray,
        density: np.ndarray,
        baseline_trace: np.ndarray,
        peak_idx: int,
        left_idx: int,
        right_idx: int,
        pad_samples: int,
        min_sigma_s: float,
) -> tuple[float, float]:
    n = density.size
    L = max(0, left_idx - pad_samples)
    R = min(n - 1, right_idx + pad_samples)
    if R <= L:
        return float(time_s[peak_idx]), float(min_sigma_s)

    t_win = time_s[L:R + 1].astype(np.float64)
    d_win = density[L:R + 1].astype(np.float64)
    b_win = baseline_trace[L:R + 1].astype(np.float64)
    w = np.clip(d_win - b_win, 0.0, None)

    total = w.sum()
    if total <= 0:
        return float(time_s[peak_idx]), float(min_sigma_s)

    mu = float((t_win * w).sum() / total)
    var = float(((t_win - mu) ** 2 * w).sum() / total)
    sigma = max(float(np.sqrt(max(var, 0.0))), float(min_sigma_s))
    return mu, sigma


# ---- peaks -> boundaries ----

def _boundaries_from_peaks(
        time_s: np.ndarray,
        smooth: np.ndarray,
        peak_indices: np.ndarray,
        params: EventDetectionParams,
) -> dict:
    """Baseline walk + Gaussian fit + merge. Returns a dict of per-event arrays."""
    time_s = np.asarray(time_s, dtype=np.float64)
    smooth = np.asarray(smooth, dtype=np.float64)
    peaks = np.asarray(peak_indices, dtype=np.int64)

    dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else 1.0
    fps_density = 1.0 / max(dt, 1e-12)

    # baseline + noise
    if params.baseline_mode == "global":
        baseline_trace = _estimate_global_baseline(smooth, params.baseline_percentile)
    else:
        baseline_trace = _estimate_rolling_baseline(
            smooth, fps_density, params.baseline_window_s, params.baseline_percentile,
        )
    noise = _estimate_noise_from_quiet(
        smooth, baseline_trace,
        quiet_percentile=params.noise_quiet_percentile,
        noise_mad_factor=params.noise_mad_factor,
    )
    end_threshold_trace = baseline_trace + params.end_threshold_k * noise

    empty_f = np.array([], dtype=np.float64)
    empty_o = np.array([], dtype=object)
    if peaks.size == 0:
        return dict(
            start_s=empty_f, peak_s=empty_f, end_s=empty_f,
            peak_height=empty_f, prominence=empty_f, duration_s=empty_f,
            mu_s=empty_f, sigma_s=empty_f,
            boundary_source_left=empty_o, boundary_source_right=empty_o,
            baseline_trace=baseline_trace, end_threshold_trace=end_threshold_trace,
            baseline_noise=noise,
        )

    peaks = np.sort(peaks)

    # baseline walk
    max_steps = max(1, int(round(params.max_event_duration_s / dt)))
    start_idx = np.empty(peaks.size, dtype=np.int64)
    end_idx = np.empty(peaks.size, dtype=np.int64)
    for i, p in enumerate(peaks):
        start_idx[i] = _walk_boundary(smooth, int(p), end_threshold_trace, -1, max_steps)
        end_idx[i] = _walk_boundary(smooth, int(p), end_threshold_trace, +1, max_steps)

    # merge overlapping / touching events
    merge_gap_samples = max(0, int(round(params.merge_gap_s / dt)))
    keep = np.ones(peaks.size, dtype=bool)
    for i in range(1, peaks.size):
        j = i - 1
        while j >= 0 and not keep[j]:
            j -= 1
        if j < 0:
            continue
        if start_idx[i] - end_idx[j] <= merge_gap_samples:
            if smooth[peaks[i]] > smooth[peaks[j]]:
                peaks[j] = peaks[i]
            end_idx[j] = max(end_idx[j], end_idx[i])
            start_idx[j] = min(start_idx[j], start_idx[i])
            keep[i] = False
    peaks = peaks[keep]
    start_idx = start_idx[keep]
    end_idx = end_idx[keep]
    n_events = peaks.size

    # Gaussian-fit refinement
    mu_s = np.full(n_events, np.nan, dtype=np.float64)
    sigma_s = np.full(n_events, np.nan, dtype=np.float64)
    src_left = np.empty(n_events, dtype=object)
    src_right = np.empty(n_events, dtype=object)

    base_start_s = time_s[start_idx].astype(np.float64)
    base_end_s = time_s[end_idx].astype(np.float64)

    if params.use_gaussian_boundary and n_events > 0:
        z = _gaussian_z(params.gaussian_quantile)
        pad_samples = max(0, int(round(params.gaussian_fit_pad_s / dt)))
        start_s_out = np.empty(n_events, dtype=np.float64)
        end_s_out = np.empty(n_events, dtype=np.float64)
        for i in range(n_events):
            mu, sig = _fit_gaussian_to_peak(
                time_s=time_s, density=smooth, baseline_trace=baseline_trace,
                peak_idx=int(peaks[i]), left_idx=int(start_idx[i]), right_idx=int(end_idx[i]),
                pad_samples=pad_samples, min_sigma_s=params.gaussian_min_sigma_s,
            )
            mu_s[i] = mu
            sigma_s[i] = sig
            g_start = mu - z * sig
            g_end = mu + z * sig
            # Whichever comes first
            chosen_start = max(g_start, base_start_s[i])
            chosen_end = min(g_end, base_end_s[i])
            # Never cross the peak
            peak_t = float(time_s[peaks[i]])
            chosen_start = min(chosen_start, peak_t)
            chosen_end = max(chosen_end, peak_t)
            start_s_out[i] = chosen_start
            end_s_out[i] = chosen_end
            src_left[i] = "gaussian" if g_start >= base_start_s[i] else "baseline"
            src_right[i] = "gaussian" if g_end <= base_end_s[i] else "baseline"
    else:
        start_s_out = base_start_s
        end_s_out = base_end_s
        src_left[:] = "baseline"
        src_right[:] = "baseline"

    peak_height = smooth[peaks]
    prominences_final = peak_height - end_threshold_trace[peaks]

    return dict(
        start_s=start_s_out,
        peak_s=time_s[peaks].astype(np.float64),
        end_s=end_s_out,
        peak_height=peak_height,
        prominence=prominences_final,
        duration_s=end_s_out - start_s_out,
        mu_s=mu_s, sigma_s=sigma_s,
        boundary_source_left=src_left,
        boundary_source_right=src_right,
        baseline_trace=baseline_trace,
        end_threshold_trace=end_threshold_trace,
        baseline_noise=noise,
    )


# ---- activation matrix within event windows ----

def _activation_matrix_from_windows(
        onsets_by_roi: Sequence[np.ndarray],
        event_windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop-in for the old _activation_matrix, but using variable-width event
    windows instead of fixed bins.

    Returns
    -------
    A : (N, E) bool     -- any onset in [start_s, end_s] ?
    first_time : (N, E) float -- earliest onset time (s) in window, NaN if inactive
    """
    N = len(onsets_by_roi)
    E = event_windows.shape[0]
    A = np.zeros((N, E), dtype=bool)
    first_time = np.full((N, E), np.nan, dtype=np.float64)

    if E == 0:
        return A, first_time

    starts = event_windows[:, 0].astype(np.float64)
    ends = event_windows[:, 1].astype(np.float64)

    for i, ts in enumerate(onsets_by_roi):
        ts = np.asarray(ts, dtype=np.float64)
        if ts.size == 0:
            continue
        # For each event window, find the earliest onset inside [start, end]
        # Vectorised: broadcast onsets against window boundaries (E can be large,
        # but typically 20-200 events so (E x ts.size) is fine).
        # If N*E*ts is ever too large we can switch to searchsorted per ROI.
        inside = (ts[None, :] >= starts[:, None]) & (ts[None, :] <= ends[:, None])  # (E, n_onsets)
        any_inside = inside.any(axis=1)
        A[i, any_inside] = True
        for e in np.where(any_inside)[0]:
            first_time[i, e] = float(ts[inside[e]].min())

    return A, first_time
