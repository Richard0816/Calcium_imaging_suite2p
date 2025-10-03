import os
import io
import contextlib
import sys
import numpy as np
import re
import pandas as pd
import psutil
from scipy.ndimage import percentile_filter
from scipy.signal import butter, sosfilt, savgol_filter
from pathlib import Path
from typing import Tuple, Optional, Sequence, Union


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
    available_mem = round(psutil.virtual_memory().available/ (1024 ** 3), 1)

    # ensures that the minimum batch size is 100, even if we are running super low on memory
    if available_mem <= 13.5:
        return 100

    # calculates the batch size based on a linear relationship between memory and run
    else:
        return int(20 * available_mem - 170) # calculated using the two point form of the linear eqn (16, 150), (200, 4000)

def run_on_folders(parent_folder: str, func, addon_vals: list = None, leaf_folders_only=False) -> None:
    """
    Run a custom function on every subfolder inside a parent folder (non-recursive).

    :param leaf_folders_only: only look for folders that contain no subfolders False by default
    :param func: the function that you want to run
    :param addon_vals: A list of values that are needed (optional, will pass None if none given)
    :param parent_folder: Path to the parent folder.
    """
    for entry in os.scandir(parent_folder):
        if entry.is_dir():
            has_subfolders = any(sub.is_dir() for sub in os.scandir(entry.path))
            if leaf_folders_only and has_subfolders: # if we are looking for only folders with no subfolders and we find a subfolder, skip this entry
                continue

            if addon_vals: # if we have values to pass, pass the values
                func(entry.path, addon_vals)
            else: # if there's nothing just pass the path
                func(entry.path)

def log(log_filename: str, run_function) -> None:
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

    logfile.close()

# ---- Suite2p I/O + orientation ----
def s2p_infer_orientation(F: np.ndarray) -> tuple[int, int, bool]:
    """
    Infer whether Suite2p arrays are shaped (N_ROIs, T) or (T, N_ROIs).
    Returns (T, N, time_major) where time_major=True means (T, N).
    """
    if F.ndim != 2:
        raise ValueError(f"Expected 2D array, got {F.shape}")
    n0, n1 = F.shape
    if n0 < n1:
        # (N, T)
        num_frames, num_rois, time_major = n1, n0, False
    else:
        # (T, N)
        num_frames, num_rois, time_major = n0, n1, True
    return num_frames, num_rois, time_major

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

    dff = np.memmap(root / f"{prefix}dff.memmap.float32", dtype="float32", mode="r", shape=(num_frames, num_rois))
    low = np.memmap(root / f"{prefix}dff_lowpass.memmap.float32", dtype="float32", mode="r", shape=(num_frames, num_rois))
    dt  = np.memmap(root / f"{prefix}dff_dt.memmap.float32", dtype="float32", mode="r", shape=(num_frames, num_rois))
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
#box car filter
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


# ----------- Data processing -----------
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


# k nearest neighbor
# umap
# PCA
