from datetime import datetime
import re
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
SMTP_SERVER = "smtp.gmail.com"  # e.g., smtp.gmail.com, SMTP.office365.com
SMTP_PORT = 587  # typically 587 for TLS, 465 for SSL
SENDER_EMAIL = "richard.script.use@gmail.com"
RECIPIENT_EMAIL = "richardjiang2004@gmail.com"
EMAIL_PASSWORD = "uhau dvea emsk bair" # Or an app-specific password

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

# --- Add this helper somewhere near your email config (top of file) ---

def send_email(subject: str, body: str) -> None:
    """
    Send a plaintext email via SMTP (STARTTLS).
    Uses the global SMTP_* / SENDER_EMAIL / RECIPIENT_EMAIL / EMAIL_PASSWORD config.
    """
    msg = (
        f"From: {SENDER_EMAIL}\r\n"
        f"To: {RECIPIENT_EMAIL}\r\n"
        f"Subject: {subject}\r\n"
        f"\r\n"
        f"{body}\r\n"
    )

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, [RECIPIENT_EMAIL], msg.encode("utf-8", errors="replace"))


# Set the custom handler as the global exception hook
sys.excepthook = global_exception_handler

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

def folder_in_all_logs(folder_name: str, log_files: list[str]) -> bool:
    """
    Returns True if folder_name appears in every log file.
    If a log file does not exist, treat as not completed.
    """
    for log in log_files:
        if not os.path.exists(log):
            return False
        try:
            with open(log, "r", encoding="utf-8", errors="ignore") as f:
                if folder_name not in f.read():
                    return False
        except Exception:
            return False
    return True

def count_cluster_roi_files(base_dir: Path) -> int:
    """
    Counts how many cluster ROI files exist (e.g. *_rois.npy) in base_dir.
    """
    if not base_dir.exists():
        return 0
    return len(list(base_dir.glob("*_rois.npy")))

def main(folder_name: str):
    ALL_LOGS = [
        "fluorescence_analysis.log",
        "raster_and_heatmaps_plots.log",
        "image_all.log",
        "hierarchical_clustering.log",
        "crosscorrelation.log",
    ]

    if folder_in_all_logs(folder_name, ALL_LOGS):
        print(f"[SKIP] {folder_name} already present in all logs.")
        return None
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

        cluster_dir = Path(folder_name) / "suite2p" / "plane0" / "r0p7_filtered_cluster_results"
        n_clusters = count_cluster_roi_files(cluster_dir)

        if n_clusters < 2:
            run_with_logging("crosscorrelation.log", print,
                f"[SKIP] cross-correlation for {folder_name}: \n "
                f"found {n_clusters} '*_rois.npy' files in {cluster_dir} (need >= 2)."
            )
        else:
            params = dict(
                root=Path(folder_name + r'\suite2p\plane0'),
                prefix="r0p7_filtered_",
                fps=utils.get_fps_from_notes(folder_name),
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
                shift_cluster="B",
                two_sided=False,
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

def entries_to_run_to_list(entries: list) -> list:
    """
    :param entries: "entries_to_run"
    :return: List of lists where each string is broken down into list of ints
    """
    final = []
    for e in entries:
        final.append(string_to_int_list(e))
    return final


def string_to_int_list(string: str) -> list:
    temp = []
    delimiters_pattern = r'-|_'
    lst = re.split(delimiters_pattern, string)
    for i in lst:
        temp.append(int(i))
    return temp

if __name__ == '__main__':
    #print(need_to_run_analysis_py(r'D:\data\2p_shifted\Hip\2024-06-03_00003'))
    entries_to_run = [
        "2024-06-03-00004", "2024-06-03-00007", "2024-06-03-00009", "2024-06-04-00001", "2024-06-04-00002",
        "2024-06-04-00006", "2024-06-04-00009", "2024-06-04-00010", "2024-06-05-00001", "2024-06-05-00006",
        "2024-06-05-00007", "2024-07-01-00001", "2024-07-01-00002", "2024-07-01-00005", "2024-07-01-00006",
        "2024-07-01-00012", "2024-07-01-00016", "2024-07-01-00017", "2024-07-01-00018", "2024-07-01-00019",
        "2024-07-02-00001", "2024-07-02-00002", "2024-07-02-00005", "2024-07-02-00006", "2024-07-02-00007",
        "2024-07-02-00008", "2024-07-02-00012", "2024-07-02-00013", "2024-08-21-0003", "2024-08-22-0001",
        "2024-08-22-0003", "2024-08-22-0004", "2024-10-30-0003", "2024-10-30-0005", "2024-10-30-0010",
        "2024-10-30-0012", "2024-10-31-0001", "2024-10-31-0005", "2024-11-04-0003", "2024-11-04-0004",
        "2024-11-04-0010", "2024-11-05-0001", "2024-11-05-0004", "2024-11-05-0007", "2024-11-18-0003",
        "2024-11-18-0005", "2024-11-18-0008", "2024-11-20-0001"
    ]

    entries_to_run = entries_to_run_to_list(entries_to_run)

    for entry in os.scandir(r'E:\data\2p_shifted\Cx'):
        # running in here just to store the output in the logfile

        if entry.is_dir():
            if string_to_int_list(entry.name) in entries_to_run:
                try:
                    send_email(
                        subject="Pipeline entry triggered",
                        body=entry.path
                    )
                except Exception as e:
                    # Don't crash the pipeline if email fails
                    run_with_logging("crosscorrelation.log", print, f"[WARN] Email failed for {entry.path}: {e}")

                cluster_dir = Path(entry.path) / "suite2p" / "plane0" / "r0p7_filtered_cluster_results"
                n_clusters = count_cluster_roi_files(cluster_dir)

                if n_clusters < 2:
                    run_with_logging("crosscorrelation.log", print,
                                     f"[SKIP] cross-correlation for {entry.path}: \n "
                                     f"found {n_clusters} '*_rois.npy' files in {cluster_dir} (need >= 2)."
                                     )
                else:
                    params = dict(
                        root=Path(entry.path + r'\suite2p\plane0'),
                        fps=utils.get_fps_from_notes(entry.path),
                        prefix="r0p7_filtered_",
                        cluster_folder="",
                        bin_sec=0.5,
                        frac_required=0.8,
                        use_gpu=True,
                        zero_lag_only=False,
                        top_k_lag=200,
                        max_lag_seconds=2,
                        max_cluster_size=500
                    )
                    run_with_logging(
                        "crosscorrelation.log",
                        crosscorrelation.run_crosscorr_per_coactivation_bin_fast,
                        **params,
                    )
    for entry in os.scandir(r'E:\data\2p_shifted\Hip'):
        # running in here just to store the output in the logfile
        if entry.is_dir():
            if string_to_int_list(entry.name) in entries_to_run:
                try:
                    send_email(
                        subject="Pipeline entry triggered",
                        body=entry.path
                    )
                except Exception as e:
                    # Don't crash the pipeline if email fails
                    run_with_logging("crosscorrelation.log", print, f"[WARN] Email failed for {entry.path}: {e}")

                cluster_dir = Path(entry.path) / "suite2p" / "plane0" / "r0p7_filtered_cluster_results"
                n_clusters = count_cluster_roi_files(cluster_dir)

                if n_clusters < 2:
                    run_with_logging("crosscorrelation.log", print,
                                     f"[SKIP] cross-correlation for {entry.path}: \n "
                                     f"found {n_clusters} '*_rois.npy' files in {cluster_dir} (need >= 2)."
                                     )
                else:
                    params = dict(
                        root=Path(entry.path + r'\suite2p\plane0'),
                        fps=utils.get_fps_from_notes(entry.path),
                        prefix="r0p7_filtered_",
                        cluster_folder="",
                        bin_sec=0.5,
                        frac_required=0.8,
                        use_gpu=True,
                        zero_lag_only=False,
                        top_k_lag=200,
                        max_lag_seconds=2,
                        max_cluster_size=500
                    )
                    run_with_logging(
                        "crosscorrelation.log",
                        crosscorrelation.run_crosscorr_per_coactivation_bin_fast,
                        **params,
                    )
    # ---- Pipeline finished ----
    try:
        send_email(
            subject="Pipeline completed",
            body=f"Pipeline finished at {datetime.now().isoformat()}"
        )
    except Exception as e:
        print(f"[WARN] Final completion email failed: {e}")

    #main(r'E:\data\2p_shifted\Cx\2024-08-20_00001')
