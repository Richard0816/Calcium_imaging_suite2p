import utils
import hierarchical_clustering
import crosscorrelation
import analyze_output
import spatial_heatmap
import image_all

from pathlib import Path
import os

import io
import sys
import contextlib

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


def need_to_run_analysis_py(folder_name: str) -> bool:
    """
    :param folder_name: Current folder name
    :return: Boolean if we have dff.memmap.float32 etc
    """

    return True


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

def need_to_run_hierarchial_cluster(folder_name: str) -> bool:
    """
    :param folder_name: Current folder name
    :return:
    """
    return True

def need_to_run_crosscorrelation(folder_name: str) -> bool:
    """
    :param folder_name: Current folder name
    :return:
    """
    return True


def main(folder_name: str):
    #Check if we need to analyze
    if need_to_run_analysis_py(folder_name):
        run_with_logging(
            "fluorescence_analysis_test.log",
            analyze_output.run_analysis_on_folder,
            folder_name
        )



    #Check if we need to run imaging (spatial heatmap)
    if need_to_run_spatial_heatmap(folder_name):
        run_with_logging(
            "raster_and_heatmaps_plots_test.log",
            spatial_heatmap.run_spatial_heatmap,
            folder_name
        )


    #Check if we need to run image_all.py
    if need_to_run_image_all_py(folder_name):
        run_with_logging(
            "image_all_test.log",
            image_all.run_full_imaging_on_folder,
            folder_name
        )

    #Check if we need to run hierarchical clustering
    if need_to_run_hierarchial_cluster(folder_name):
        params = dict(
            root=Path(folder_name+r'\suite2p\plane0'),
            fps=30.0,
            prefix="r0p7_filtered_",
            method="ward",
            metric="euclidean",
        )
        run_with_logging(
            "hierarchical_clustering_test.log",
            hierarchical_clustering.main,
            **params
        )


    #Check if we need to run cross correlations
    if need_to_run_crosscorrelation(folder_name):
        params = dict(
            root=Path(folder_name+r'\suite2p\plane0'),
            fps=30.0,
            prefix="r0p7_filtered_",
            cluster_folder="",
            max_lag_seconds=5.0,
            cpu_fallback=True,
            zero_lag=True,
            zero_lag_only=False,
        )

        run_with_logging(
            "crosscorrelation_test.log",
            crosscorrelation.run_cluster_cross_correlations_gpu,
            **params
        )





    return None





if __name__ == '__main__':
    main(r'F:\data\2p_shifted\Hip\2024-06-03_00009')


