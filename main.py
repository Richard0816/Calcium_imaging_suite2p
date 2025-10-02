import suite2p
import numpy as np
import psutil
import utils


def change_tau_according_to_GCaMP(ops: dict, tau_vals: dict, aav_info_csv: str, file_name: str) -> dict:
    """
    Takes the existing ops.npy file information and modifies the Tau value according to the aav that was used
    :param file_name: name of the file we are looking to analyse
    :param ops: OPS file, this is a dictionary
    :param tau_vals: A translation of the GCaMP used and the appropriate tau value to apply
    :param aav_info_csv: This is information taken from the human_SLE_2p_meta.xlsx file, saved as a csv for easy use
        will always look for the columns of "AAV" and "video" to determine the file name and appropriate video used
    :return: The updated ops data after appropriate adjustment has been made
    """
    # look into utils.py to get full information
    tau = utils.file_name_to_aav_to_dictionary_lookup(file_name, aav_info_csv, tau_vals)

    # apply tau value to ops file
    ops['tau'] = tau

    return ops


def change_batch_according_to_free_ram(ops: dict) -> dict:
    """
    Not totally necessary but changes batch size according to how much memory we have
    :param ops: OPS file, this is a dictionary
    :return: The updated ops data after appropriate adjustment has been made
    """
    # define the memory
    virtual_memory = psutil.virtual_memory()

    # calculate the current available memory
    available_mem = round(virtual_memory.available / (1024 ** 3), 1)

    # ensures that the minimum batch size is 100, even if we are running super low on memory
    if available_mem <= 13.5:
        ops["batch_size"] = 100
        return ops

    # calculates the batch size based on a linear relationship between memory and run
    else:
        ops["batch_size"] = int(20 * available_mem - 170) # calculated using the two point form of the linear eqn (16, 150), (200, 4000)
        return ops


def run_suite2p_on_folder(folder_name: str, addon_vals: list) -> None:
    """
    :param folder_name:
    :param addon_vals: A list of values that are needed in this case [ops, tau_vals]
        :param ops: OPS file, this is a dictionary
        :param tau_vals: A translation of the GCaMP used and the appropriate tau value to apply
    :return: None, just runs Suite2p
    """
    print(f'Running on {folder_name}')
    ops, tau_vals = addon_vals

    # Changing the ops file
    change_tau_according_to_GCaMP(ops, tau_vals, "human_SLE_2p_meta.csv", folder_name.split("\\")[-1])
    change_batch_according_to_free_ram(ops) # can delete if you are happy with the batch_size defined in the ops.npy

    # defining the folder
    db = {
        'data_path': [folder_name]
    }

    # running suite2p on the modified ops and data_path
    output_ops = suite2p.run_s2p(ops, db)

    print(set(output_ops.keys()).difference(ops.keys()))


def run():
    # taken from suite2p documentation https://suite2p.readthedocs.io/en/latest/settings.html
    tau_vals = {
        "6f": 0.7,
        "6m": 1.0,
        "6s": 1.3,
        "8m": 0.137  # empirically defined
    }

    path_to_ops = "D:\\suite2p_2p_ops_240621.npy"

    ops = np.load(path_to_ops, allow_pickle=True).item()

    utils.run_on_folders('D:\\data\\2p_shifted\\', run_suite2p_on_folder, [ops, tau_vals], True)

if __name__ == '__main__':
    utils.log("suite2p_raw_output.log", run())
