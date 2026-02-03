from typing import Union, Tuple, Any

import utils
import hierarchical_clustering
import crosscorrelation
import analyze_output
import spatial_heatmap
import image_all
import glob
from pathlib import Path
import os

import io
import sys
import contextlib

import sys
import logging
from logging.handlers import SMTPHandler
import traceback
import smtplib

# --- Email Configuration ---
# Use environment variables for sensitive info (e.g., passwords) for security
SMTP_SERVER = "smt.gmail.com"  # e.g., smtp.gmail.com, SMTP.office365.com
SMTP_PORT = 587  # typically 587 for TLS, 465 for SSL
SENDER_EMAIL = "richard.script.use@gmail.com"
RECIPIENT_EMAIL = "richardjiang2004@gmail.com"
EMAIL_PASSWORD = "wewi0816" # Or an app-specific password

# --- Setup Logging with SMTPHandler ---
# Create a logger
error_logger = logging.getLogger(__name__)
error_logger.setLevel(logging.ERROR)

# Create the SMTP handler
try:
    smtp_handler = SMTPHandler(
        mailhost=(SMTP_SERVER, SMTP_PORT),
        fromaddr=SENDER_EMAIL,
        toaddrs=[RECIPIENT_EMAIL],
        subject="CRITICAL Error in Python Script",
        credentials=(SENDER_EMAIL, EMAIL_PASSWORD),
        secure=() # Use secure=() for STARTTLS (port 587)
    )
    smtp_handler.setLevel(logging.ERROR)
    error_logger.addHandler(smtp_handler)
except smtplib.SMTPException as e:
    print(f"Failed to set up SMTP handler: {e}")

# --- Global Exception Handler Function ---
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """
    Logs unhandled exceptions and sends an email alert.
    """
    # Log the exception with full traceback
    error_logger.error("An unhandled exception occurred:", 
                       exc_info=(exc_type, exc_value, exc_traceback))
    
    # Optionally, print to console as well (default behavior)
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Set the custom handler as the global exception hook
sys.excepthook = global_exception_handler

# --- Example Usage (will trigger the handler) ---
if __name__ == "__main__":
    print("Script starting...")
    # Simulate an error
    1 / 0 
    print("Script finished (this line won't be reached if error occurs)")



class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        # Write to all streams; if a stream can't encode a char (e.g., ≥), replace it.
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except UnicodeEncodeError:
                # Fallback: write bytes with replacement for unencodable characters
                enc = getattr(s, "encoding", None) or "utf-8"
                if hasattr(s, "buffer"):
                    s.buffer.write(data.encode(enc, errors="replace"))
                    s.flush()
                else:
                    # Last resort: replace problematic chars and retry
                    s.write(data.encode(enc, errors="replace").decode(enc, errors="replace"))
                    s.flush()
        return len(data)


def run_with_logging(logfile_name: str, func, *args, **kwargs) -> None:
    """
    Logs the output of a given function to a specified log file as well as the standard
    output. It intercepts the output of both standard output and standard error streams
    and writes them to the specified log file whilst still displaying them in the console.

    The function uses a helper class `Tee` and `contextlib.redirect_stdout` and
    `contextlib.redirect_stderr` to handle output redirection.

    :param logfile_name: The path to the log file where the output should be stored.
    :param func: The function to be executed, whose output is to be logged.
    :return: None
    """
    # Force log file to be UTF-8 so it can always store characters like ≥.
    with open(logfile_name, "a", encoding="utf-8", errors="replace") as logfile:
        tee = Tee(sys.__stdout__, logfile)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            func(*args, **kwargs)


def need_to_run_analysis_py(folder_name: str, override: bool = False) -> Union[tuple[bool, str], tuple[bool, None]]:
    """
    :param override: If need to override the decision making
    :param folder_name: Current folder name
    :return: Boolean if we have dff.memmap.float32 etc
    """
    if override:
        return True, None

    working_directory = Path(folder_name + r'\suite2p\plane0')

    required_files_suffix = [  # r0p7 is an arbitrary choice and is subject to change
        "_filtered_dff.memmap.float32",
        "_filtered_dff_dt.memmap.float32",
        "_filtered_dff_lowpass.memmap.float32"
    ]

    # Collect all filenames
    files = {p.name for p in working_directory.iterdir() if p.is_file()}

    for file_name in files:
        for suffix in required_files_suffix:
            if file_name.endswith(suffix):
                prefix = file_name.removesuffix(suffix)
                # update required_files_suffix to just required files eg. r0p7_filtered_dff.memmap.float32
                if all(f'{prefix}{suf}' in files for suf in required_files_suffix):
                    return False, prefix

    return True, None


def need_to_run_spatial_heatmap(folder_name: str) -> bool:
    """
    :param folder_name: Current folder name
    :return:
    """
    return True


def need_to_run_image_all_py(folder_name: str) -> bool:
    """
    :param folder_name: Current folder name
    :return:
    """
    return True


def need_to_run_hierarchial_cluster(folder_name: str, override: bool = False) -> bool:
    """
    :param override: If need to override the decision making
    :param folder_name: Current folder name
    :return:
    """
    if override:
        return True

    exists = any(
        os.path.isdir(p)
        for p in glob.glob(os.path.join(folder_name, "*_filtered_cluster_results"))
    )

    return not exists


def need_to_run_crosscorrelation(folder_name: str) -> bool:
    """
    :param folder_name: Current folder name
    :return:
    """
    return True


def main(folder_name: str):
    # Check if we need to analyze
    need_to_run_analysis_py_truth, prefix = need_to_run_analysis_py(folder_name, override=True)
    if need_to_run_analysis_py_truth:
        run_with_logging(
            "fluorescence_analysis.log",
            analyze_output.run_analysis_on_folder,
            folder_name
        )

    # Check if we need to run imaging (spatial heatmap)
    if need_to_run_spatial_heatmap(folder_name):
        run_with_logging(
            "raster_and_heatmaps_plots.log",
            spatial_heatmap.run_spatial_heatmap,
            folder_name,
            score_threshold=0.15  # classify as cell if P>=0.5
        )
        run_with_logging(
            "raster_and_heatmaps_plots.log",
            spatial_heatmap.coactivation_order_heatmaps,
            folder_name,
            score_threshold=0.15
        )

    # Check if we need to run image_all.py
    if need_to_run_image_all_py(folder_name):
        run_with_logging(
            "image_all.log",
            image_all.run_full_imaging_on_folder,
            folder_name
        )

    # Check if we need to run hierarchical clustering
    if need_to_run_hierarchial_cluster(folder_name, override=True):
        params = dict(
            root=Path(folder_name + r'\suite2p\plane0'),
            fps=utils.get_fps_from_notes(folder_name),
            prefix="r0p7_filtered_",
            method="ward",
            metric="euclidean",
        )
        run_with_logging(
            "hierarchical_clustering.log",
            hierarchical_clustering.main,
            **params
        )

        params = dict(
            root=Path(folder_name + r'\suite2p\plane0'),
            fps=utils.get_fps_from_notes(folder_name),
            prefix="r0p7_filtered_",
            cluster_folder="",
            max_lag_seconds=5.0,
            cpu_fallback=True,
            zero_lag=True,
            zero_lag_only=False,
        )

        run_with_logging(
            "crosscorrelation.log",
            crosscorrelation.run_cluster_cross_correlations_gpu,
            **params
        )

        params = dict(
            root=Path(folder_name + r'\suite2p\plane0'),
            prefix="r0p7_filtered_",
            fps=utils.get_fps_from_notes(folder_name),
            n_surrogates=5000,
            min_shift_s=1,
            max_shift_s=500,
            shift_cluster="B",  # shift C2, leave C1 fixed
            two_sided=False,  # synchrony: usually one-sided
            seed=0,
            use_gpu=True,
            fdr_alpha=0.05,
            save_pairwise_csv=True
        )
        run_with_logging(
            "crosscorrelation.log",
            crosscorrelation.run_clusterpair_zero_lag_shift_surrogate_stats,
            **params,
        )

    return None


if __name__ == '__main__':
    #print(need_to_run_analysis_py(r'D:\data\2p_shifted\Hip\2024-06-03_00003'))
    for entry in os.scandir(r'E:\data\2p_shifted\Cx'):
        # running in here just to store the output in the logfile
        if entry.is_dir():
            main(entry.path)
    for entry in os.scandir(r'E:\data\2p_shifted\Hip'):
        # running in here just to store the output in the logfile
        if entry.is_dir():
            main(entry.path)
    #main(r'E:\data\2p_shifted\Cx\2024-08-20_00001')
