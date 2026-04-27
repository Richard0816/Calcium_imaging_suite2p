"""
GPU-accelerated version of analyze_output.py.

Differences vs. CPU version:
  - Processes all ROIs (no heuristic pre-filter). The cell mask is produced
    afterwards by cellfilter.predict (predicted_cell_mask.npy).
  - F0 is the mean of the first `baseline_sec` seconds per ROI (scalar per ROI),
    not a rolling percentile. Assumes baseline window is event-free and bleaching
    is negligible over the recording.
  - Neuropil subtraction, dF/F, low-pass, and SG derivative run batched over all
    ROIs on GPU via CuPy.
  - Writes memmaps with prefix `r0p7_` so cellfilter.dataset._RecordingCache can
    find them (DFF_PREFIX in cellfilter.config).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

import utils
from analyze_output import (
    custom_lowpass_cutoff,
    _setup_output_memmaps,
)


def _to_time_major(F_cell: np.ndarray, F_neuropil: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Suite2p always saves F.npy / Fneu.npy as (N_ROIs, T). Transpose to
    (T, N) unconditionally. Do not trust the shape-heuristic in
    utils.s2p_infer_orientation — it misfires when N_ROIs > T."""
    F_cell = np.asarray(F_cell, dtype=np.float32, order='C')
    F_neuropil = np.asarray(F_neuropil, dtype=np.float32, order='C')
    if F_cell.ndim != 2 or F_neuropil.ndim != 2:
        raise ValueError("F and Fneu must be 2D")
    if F_cell.shape != F_neuropil.shape:
        raise ValueError(f"F and Fneu shapes differ: {F_cell.shape} vs {F_neuropil.shape}")

    # Suite2p convention: (N, T) -> transpose to (T, N)
    F_cell = F_cell.T.copy()
    F_neuropil = F_neuropil.T.copy()
    T, N = F_cell.shape
    return F_cell, F_neuropil, T, N

try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    _CUPY_OK = True
except ImportError:
    cp = None
    cp_signal = None
    _CUPY_OK = False
    print("[warn] CuPy not available; analyze_output_gpu cannot run on GPU.")


def _butter_sos(fps: float, cutoff_hz: float = 1.0, order: int = 2) -> np.ndarray:
    from scipy.signal import butter
    nyq = fps / 2.0
    cutoff = min(max(1e-4, cutoff_hz), 0.95 * nyq)
    return butter(order, cutoff / nyq, btype='low', output='sos').astype(np.float32)


def _sg_derivative_gpu(x, fps: float, win_ms: float, poly: int):
    """Batched Savitzky–Golay first derivative along axis 0 on GPU."""
    T = x.shape[0]
    win = max(3, int((win_ms / 1000.0) * fps) | 1)
    if win >= T:
        win = max(3, T - (1 - T % 2))

    if win < 3 or T < 3:
        d = cp.empty_like(x)
        d[0, :] = 0.0
        d[1:, :] = (x[1:, :] - x[:-1, :]) * fps
        return d

    if hasattr(cp_signal, 'savgol_filter'):
        return cp_signal.savgol_filter(
            x, window_length=win, polyorder=poly,
            deriv=1, delta=1.0 / fps, axis=0,
        ).astype(cp.float32)

    from scipy.signal import savgol_coeffs
    from cupyx.scipy.ndimage import convolve1d
    coeffs = savgol_coeffs(win, poly, deriv=1, delta=1.0 / fps, use='conv').astype(np.float32)
    return convolve1d(x, weights=cp.asarray(coeffs), axis=0, mode='nearest').astype(cp.float32)


def _gpu_process_all(
    F_cell: np.ndarray, F_neuropil: np.ndarray, r: float, fps: float,
    baseline_sec: float = 60.0,
    cutoff_hz: float = 1.0, order: int = 2,
    sg_win_ms: float = 333, sg_poly: int = 2,
    roi_chunk: int | None = None,
    pad: int = 64,
):
    """
    Run the full pipeline on GPU. Inputs are (T, N) float32 numpy arrays.
    Returns (dff, lowpass, derivative) as (T, N) float32 numpy arrays.
    """
    if not _CUPY_OK:
        raise RuntimeError("CuPy/cupyx not available; cannot run GPU analyze.")

    T, N = F_cell.shape
    if roi_chunk is None or roi_chunk <= 0:
        roi_chunk = N

    n_baseline = max(3, min(T, int(round(baseline_sec * fps))))
    sos_gpu = cp.asarray(_butter_sos(fps, cutoff_hz=cutoff_hz, order=order))

    dff_out = np.empty((T, N), dtype=np.float32)
    lp_out = np.empty((T, N), dtype=np.float32)
    dt_out = np.empty((T, N), dtype=np.float32)

    for start in range(0, N, roi_chunk):
        end = min(N, start + roi_chunk)

        Fc = cp.asarray(F_cell[:, start:end], dtype=cp.float32)
        Fn = cp.asarray(F_neuropil[:, start:end], dtype=cp.float32)

        F_corr = Fc - r * Fn
        F0 = cp.mean(F_corr[:n_baseline, :], axis=0, keepdims=True)
        F0_safe = cp.maximum(F0, 1e-6)
        dff = (F_corr - F0) / F0_safe

        pad_block = cp.broadcast_to(dff[:1, :], (pad, dff.shape[1]))
        dff_pad = cp.concatenate([pad_block, dff], axis=0)
        lp = cp_signal.sosfilt(sos_gpu, dff_pad, axis=0)[pad:, :]

        deriv = _sg_derivative_gpu(lp, fps=fps, win_ms=sg_win_ms, poly=sg_poly)

        dff_out[:, start:end] = cp.asnumpy(dff)
        lp_out[:, start:end] = cp.asnumpy(lp)
        dt_out[:, start:end] = cp.asnumpy(deriv)

        del Fc, Fn, F_corr, F0, F0_safe, dff, pad_block, dff_pad, lp, deriv
        cp.get_default_memory_pool().free_all_blocks()

    return dff_out, lp_out, dt_out


def process_suite2p_traces_gpu(
    F_cell, F_neuropil, fps,
    r: float = 0.7,
    baseline_sec: float = 60.0,
    cutoff_hz: float = 1.0, sg_win_ms: float = 333, sg_poly: int = 2,
    out_dir: str | None = None, prefix: str = '',
    roi_chunk: int | None = None,
):
    F_cell, F_neuropil, T, N = _to_time_major(F_cell, F_neuropil)

    dff_mm, lp_mm, dt_mm, paths = _setup_output_memmaps(out_dir, prefix, T, N)

    t0 = time.time()
    dff, lp, dt = _gpu_process_all(
        F_cell, F_neuropil, r=r, fps=fps,
        baseline_sec=baseline_sec,
        cutoff_hz=cutoff_hz, order=2,
        sg_win_ms=sg_win_ms, sg_poly=sg_poly,
        roi_chunk=roi_chunk,
    )
    print(f"[GPU] Processed {N} ROIs × {T} frames in {time.time() - t0:.2f}s")

    dff_mm[:] = dff
    lp_mm[:] = lp
    dt_mm[:] = dt
    dff_mm.flush(); lp_mm.flush(); dt_mm.flush()
    del dff_mm, lp_mm, dt_mm

    return paths


def _write_dff_csv(dff_memmap_path: str, T: int, N: int,
                   out_path: Path, row_chunk: int = 2048,
                   keep_idx: Optional[np.ndarray] = None) -> None:
    """Write the dF/F memmap as a gzip-compressed CSV (rows=frames, cols=ROIs).

    Written in row chunks so memory stays small. Column headers are `roi_<i>`
    where <i> is the original Suite2p ROI index (preserved even when
    filtering). Readable in R via `read.csv(gzfile('...csv.gz'))`.

    If `keep_idx` is given (int array of ROI indices to keep, in the original
    Suite2p index space), only those columns are written.
    """
    import gzip
    import io

    dff = np.memmap(dff_memmap_path, mode='r', dtype='float32', shape=(T, N))

    if keep_idx is None:
        cols = np.arange(N)
    else:
        cols = np.asarray(keep_idx, dtype=np.int64)
    header = ",".join(f"roi_{i}" for i in cols) + "\n"

    t0 = time.time()
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=5) as fh:
        fh.write(header)
        buf = io.StringIO()
        for start in range(0, T, row_chunk):
            end = min(T, start + row_chunk)
            chunk = np.asarray(dff[start:end, cols], dtype=np.float32)
            buf.seek(0); buf.truncate(0)
            np.savetxt(buf, chunk, fmt="%.6g", delimiter=",")
            fh.write(buf.getvalue())
    print(f"[csv] wrote {out_path.name} ({T}×{len(cols)}) "
          f"in {time.time() - t0:.1f}s")


def _run_cellfilter_predict(plane0: Path, ckpt_path: Path | None = None) -> None:
    """Run cellfilter.predict on the plane0 directory to produce
    predicted_cell_prob.npy and predicted_cell_mask.npy."""
    import torch
    from cellfilter import config as C
    from cellfilter.model import CellFilter
    from cellfilter.predict import predict_recording

    ckpt = Path(ckpt_path) if ckpt_path else (C.CHECKPOINT_DIR / "best.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"No cellfilter checkpoint at {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cellfilter] device={device} ckpt={ckpt}")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = CellFilter().to(device)
    model.load_state_dict(state["model"])
    model.eval()

    rec_id = plane0.parent.parent.name
    predict_recording(rec_id, model, device, plane0=plane0)


def run_analysis_on_folder_gpu(
    folder_name: str,
    baseline_sec: float = 60.0,
    prefix: str = "r0p7_",
    run_cellfilter: bool = True,
    ckpt_path: Optional[str] = None,
    write_dff_csv: bool = True,
):
    start_time = time.time()
    fps = utils.get_fps_from_notes(folder_name)
    plane0 = Path(folder_name) / "suite2p" / "plane0"
    sample_name = Path(folder_name).name

    F_cell = np.load(plane0 / 'F.npy', allow_pickle=True)
    F_neu = np.load(plane0 / 'Fneu.npy', allow_pickle=True)
    print(f"[load] {sample_name}: raw shape={F_cell.shape}  -> interpreting as (N_ROIs={F_cell.shape[0]}, T={F_cell.shape[1]})")

    print(f'Processing {sample_name} (GPU, baseline={baseline_sec:.1f}s, all ROIs)')

    #cutoffs = {"6f": 5.0, "6m": 5.0, "6s": 5.0, "8m": 3.0}
    #cutoff_hz = custom_lowpass_cutoff(cutoffs, "human_SLE_2p_meta.csv", sample_name)
    #print(f'cutoff_hz: {1.0}')

    dff_path, low_path, dt_path = process_suite2p_traces_gpu(
        F_cell, F_neu, fps,
        r=0.7, baseline_sec=baseline_sec,
        cutoff_hz=1.0, sg_win_ms=333, sg_poly=2,
        out_dir=str(plane0), prefix=prefix,
    )

    print("Wrote:")
    print(" dF/F       ->", dff_path)
    print(" low-pass   ->", low_path)
    print(" d/dt       ->", dt_path)

    # Run cellfilter FIRST so the CSV can be restricted to predicted cells.
    if run_cellfilter:
        print(f"[cellfilter] predicting cell mask for {sample_name}")
        try:
            _run_cellfilter_predict(plane0, ckpt_path=Path(ckpt_path) if ckpt_path else None)
        except Exception as ex:
            print(f"[cellfilter] FAILED: {ex}")
            raise

    if write_dff_csv:
        T, N = F_cell.shape[1], F_cell.shape[0]  # raw (N, T) -> T frames, N ROIs
        mask_path = plane0 / "predicted_cell_mask.npy"
        keep_idx = None
        if mask_path.exists():
            mask = np.load(mask_path).astype(bool)
            if mask.shape[0] != N:
                print(f"[csv] WARNING: mask len {mask.shape[0]} != N {N}; "
                      f"writing all ROIs")
            else:
                keep_idx = np.flatnonzero(mask)
                print(f"[csv] filtering to {keep_idx.size}/{N} predicted cells")
        else:
            print(f"[csv] no {mask_path.name}; writing all ROIs")
        csv_path = plane0 / f"{prefix}dff.csv.gz"
        _write_dff_csv(dff_path, T, N, csv_path, keep_idx=keep_idx)

    print(f'Total time {time.time() - start_time:.2f} seconds.')


if __name__ == "__main__":
    run_analysis_on_folder_gpu(r'F:\data\2p_shifted\Hip\2024-06-03_00009')
