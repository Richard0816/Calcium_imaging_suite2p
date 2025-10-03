import tifffile
import os
import numpy as np

unshifted_dir = "D:\\data\\2p_to-be-shifted"
shifted_dir = "D:\\data\\2p_shifted"

def shift_file(filename, unshifted_dir, shifted_dir):
    """
    Takes a file and shifts the data such that the 2p data minimum is 0 and saves it to the shifted dir
    Each file will be in a seperate folder according to the file name
    :param filename: name of the file that
    :return: None
    """

    #Define what the final fold name should be
    folder_name = filename[:-4]

    #Create the directory and change current working dir
    os.chdir(shifted_dir)
    os.mkdir(folder_name)
    os.chdir(folder_name)

    print(f'Opening {filename}')
    #Raw unshifted data
    twoP_data = tifffile.imread(os.path.join(unshifted_dir, filename))


    twoP_data = twoP_data.astype(np.int32)


    twoP_data += np.abs(twoP_data.min())

    assert (twoP_data.max() < 65535)
    twoP_data = twoP_data.astype(np.uint16)

    new_file = "shifted_" + filename.split("\\")[-1]
    new_filename = os.path.join(os.getcwd(), new_file)

    print("Saving the shifted data " + new_filename)
    tifffile.imwrite(new_filename, twoP_data)
    print("Done.")


def file_names(folder_path):
    """
    Prints the names of all files within a specified folder.

    Args:
        folder_path (str): The path to the folder.
    """
    try:
        # Get a list of all entries (files and directories) in the folder
        entries = os.listdir(folder_path)
        filtered_entries = []

        # Filter out directories and print only file names
        found_files = False
        for entry in entries:
            full_path = os.path.join(folder_path, entry)
            if os.path.isfile(full_path):
                filtered_entries.append(entry)
                found_files = True

        if not found_files:
            print("No files found in this directory.")

        return filtered_entries

    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


lst = file_names(unshifted_dir)
for file in lst:
    shift_file(file, unshifted_dir, shifted_dir)
