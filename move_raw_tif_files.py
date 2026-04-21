import os
import shutil
import utils

def move_one_project(directory):
    source_file_path = os.path.join(directory, directory.split("\\")[-1] + ".tif")
    if not os.path.exists(source_file_path):
        source_file_path = os.path.join(directory, "shifted_"+directory.split("\\")[-1] + ".tif")
        if not os.path.exists(source_file_path):
            return None
    destination_directory = os.path.join("D:\\", directory.split("\\")[-1])

    os.makedirs(destination_directory, exist_ok=True)

    shutil.move(source_file_path, destination_directory)
    print(f"Moved {source_file_path} to {destination_directory}")
    return None


if __name__ == "__main__":
    utils.run_on_folders("F:\\data\\2p_shifted\\", move_one_project)
