import suite2p
import numpy as np
import csv
import pandas as pd
import re
import psutil
import os
from pathlib import Path
import io
import sys
import contextlib
import logging

class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)


def run():
    tau_vals = {
        "6f": 0.7,
        "6m": 1.0,
        "6s": 1.3,
        "8m": 0.137
    }


    def aav_cleanup_and_tau_lookup(aav: str, tau_vals: dict) -> float:
        """
        :param aav: The aav used in the experiment (string pulled from meta data)
        :return: float of the recommended tau value to use based on aav information
        """
        # drop any usage of "rg"
        aav = aav.replace("rg", "")

        # split by -, _, or + (list output)
        components = re.split(r"[-_+]", aav)

        # make both the keys and list case insensitive, and then match list and keys to find 6f, 6m, 8m etc
        dict_lower = {k.lower(): v for k, v in tau_vals.items()}
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


    def change_tau_according_to_calmod(ops: dict, tau_vals: dict, aav_info_csv: str, file_name: str) -> dict:
        """
        Takes the existing ops.npy file information and modifies the Tau value according to the aav that was used
        :param file_name: name of the file we are looking to analyise
        :param ops: OPS file, this is a dictionary
        :param tau_vals: A translation of the calmod used and the appropriate tau value to apply
        :param aav_info_csv: This is information taken from the human_SLE_2p_meta.xlsx file, saved as a csv for easy use
            will always look for the columns of "AAV" and "video" to determine the file name and appropriate video used
        :return: The updated ops data after appropriate adjustment has been made
        """
        # look up the file name in aav_info under "video"
        # get the row number of file name
        row_num = get_row_number_csv_module(aav_info_csv, 'video', file_name)

        # get aav information
        col = pd.read_csv(aav_info_csv, usecols=["AAV"])  # Read only the needed column
        element = col["AAV"].iloc[row_num - 1]  # row_num is 1-based
        element = str(element)

        # interpret aav information 6f, 8m etc & define what is the necessary tau value
        tau = aav_cleanup_and_tau_lookup(element, tau_vals)

        # apply tau value to ops file
        ops['tau'] = tau

        return ops


    def change_batch_according_to_free_ram(ops: dict) -> dict:
        """
        Not totally necessary but changes batch size according to how much memory we have
        :param ops: OPS file, this is a dictionary
        :return: The updated ops data after appropriate adjustment has been made
        """
        virtual_memory = psutil.virtual_memory()
        available_mem = round(virtual_memory.available / (1024 ** 3), 1)
        if available_mem <= 13.5:
            ops["batch_size"] = 100
            return ops

        else:
            ops["batch_size"] = int(20 * available_mem - 170)
            return ops


    path_to_ops = "D:\\suite2p_2p_ops_240621.npy"

    ops = np.load(path_to_ops, allow_pickle=True).item()


    def run_suite2p_on_folder(folder_name: str) -> None:
        print(f'Running on {folder_name}')
        change_tau_according_to_calmod(ops, tau_vals, "human_SLE_2p_meta.csv", folder_name.split("\\")[-1])
        change_batch_according_to_free_ram(ops)
        db = {
            'data_path': [folder_name]
        }

        output_ops = suite2p.run_s2p(ops, db)
        print(set(output_ops.keys()).difference(ops.keys()))


    def run_on_folders(parent_folder: str, run_suite2p_on_folder):
        """
        Run a custom function on every subfolder inside a parent folder (non-recursive).

        :param parent_folder: Path to the parent folder.
        :param func: Function to run on each subfolder. It should accept the folder path as argument.
        """
        for entry in os.scandir(parent_folder):
            if entry.is_dir():
                has_subfolders = any(sub.is_dir() for sub in os.scandir(entry.path))
                if not has_subfolders:
                    run_suite2p_on_folder(entry.path)



    run_on_folders('D:\\data\\2p_shifted\\', run_suite2p_on_folder)

logfile = open("suite2p_raw_output.log", "a")
tee = Tee(sys.__stdout__, logfile)

with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
    run()

logfile.close()
