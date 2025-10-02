import os
import io
import contextlib
import sys
import numpy as np
import re
import pandas as pd
import psutil


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
    logfile = open(log_filename, "a")
    tee = Tee(sys.__stdout__, logfile)

    # running in here just to store the output in the logfile
    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
        run_function()

    logfile.close()

# ----------- Stats -----------
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
    :param file_name:
    :param aav_info_csv:
    :param dic:
    :return:
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