"""End-to-end pipeline: shift -> sparse+cellpose detection -> cellfilter
prediction -> analyze_output -> image_all -> hierarchical clustering ->
spatial heatmaps (with and without propagation vectors).

Usage
-----
    # single TIFF
    python run_full_pipeline.py --tiff D:\\data\\2p_to-be-shifted\\2024-11-20_00003.tif

    # a folder of TIFFs (batch)
    python run_full_pipeline.py --tiff-dir D:\\data\\2p_to-be-shifted

    # grouped xlsx: rows sharing 'Identifyer' are treated as one continuous
    # recording (suite2p reads them all from <shifted_root>/<id>/ in order)
    python run_full_pipeline.py --group-xlsx D:\\manifests\\E056.xlsx

Directory layout produced per recording stem `<rec>`:
    <shifted_root>/<rec>/shifted_<rec>.tif                (single-TIFF case)
    <shifted_root>/<rec>/NNN_shifted_<orig>.tif           (--group-xlsx case)
    <detection_root>/<rec>/final/suite2p/plane0/{stat,ops,F,Fneu,spks,iscell}.npy
                                               /predicted_cell_{prob,mask}.npy
                                               /r0p7_dff.memmap.float32 (+ lowpass, dt)
                                               /*.png, cluster_results/, coact_*/
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import tifffile
import torch

WORKTREE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKTREE))


# ---------------------------------------------------------------------------
# Defaults (override via CLI)
# ---------------------------------------------------------------------------
DEFAULT_SHIFTED_ROOT   = Path(r"E:\data\2p_shifted")
DEFAULT_DETECTION_ROOT = Path(r"E:\sparse_plus_cellpose")
DEFAULT_OPS            = WORKTREE / "suite2p_2p_ops_240621.npy"
DEFAULT_PREFIX         = "r0p7_"
DEFAULT_FPS            = 15.07
DEFAULT_BASELINE_SEC   = 90.0


# ---------------------------------------------------------------------------
# 1. Shift (inlined from shifting.py to avoid module-level side effects)
# ---------------------------------------------------------------------------
def _scan_min_max(tiff_path: Path, label: str = "shift") -> tuple[int, int, int]:
    """Stream pass 1: returns (gmin, gmax, n_pages) for a single TIFF."""
    print(f"[{label}] scanning {tiff_path}")
    gmin = np.iinfo(np.int64).max
    gmax = np.iinfo(np.int64).min
    with tifffile.TiffFile(str(tiff_path)) as tf:
        n_pages = len(tf.pages)
        for i, page in enumerate(tf.pages):
            frame = page.asarray()
            pmin = int(frame.min()); pmax = int(frame.max())
            if pmin < gmin: gmin = pmin
            if pmax > gmax: gmax = pmax
            if (i + 1) % 2000 == 0:
                print(f"[{label}]   scan {i + 1}/{n_pages}  "
                      f"min={gmin} max={gmax}")
    return gmin, gmax, n_pages


def _write_shifted(src: Path, dst: Path, shift: int,
                   label: str = "shift") -> None:
    """Stream pass 2: write src as uint16 with `shift` added, BigTIFF output."""
    print(f"[{label}] writing {dst}")
    with tifffile.TiffFile(str(src)) as tf, \
         tifffile.TiffWriter(str(dst), bigtiff=True) as tw:
        n_pages = len(tf.pages)
        for i, page in enumerate(tf.pages):
            frame = page.asarray()
            if shift != 0:
                # upcast one frame at a time to avoid negative-wrap
                frame = (frame.astype(np.int32) + shift).astype(np.uint16)
            elif frame.dtype != np.uint16:
                frame = frame.astype(np.uint16)
            tw.write(frame, contiguous=True)
            if (i + 1) % 2000 == 0:
                print(f"[{label}]   write {i + 1}/{n_pages}")


def shift_tiff(tiff_path: Path, shifted_root: Path,
               stem: str | None = None) -> Path:
    """Shift a raw 2p TIFF so min=0, save as uint16. Returns the folder
    containing the shifted tif (suitable as TIFF_FOLDER for suite2p).

    If `stem` is given, output folder and filename use it instead of
    `tiff_path.stem` (useful when an external manifest names recordings
    differently than their filenames)."""
    stem = stem or tiff_path.stem
    out_dir = shifted_root / stem
    shifted_name = f"shifted_{stem}{tiff_path.suffix}"
    out_path = out_dir / shifted_name

    if out_path.exists():
        print(f"[shift] exists, skipping: {out_path}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    gmin, gmax, _ = _scan_min_max(tiff_path)
    shift = -gmin if gmin < 0 else 0
    shifted_max = gmax + shift
    print(f"[shift] global min={gmin} max={gmax}  -> shift+={shift}  "
          f"(shifted max={shifted_max})")
    if shifted_max > 65535:
        raise ValueError(f"shifted max {shifted_max} > 65535; "
                         f"uint16 overflow for {tiff_path}")
    _write_shifted(tiff_path, out_path, shift)
    return out_dir


def shift_tiff_group(tiff_paths: list[Path], shifted_root: Path,
                     stem: str) -> Path:
    """Shift a group of raw TIFFs that together form one continuous recording.

    Computes ONE global min across all inputs (so a single shift constant is
    applied), then writes each input as a separate uint16 BigTIFF into
    <shifted_root>/<stem>/. Output filenames are prefixed with a zero-padded
    index reflecting the input order so suite2p (which uses natsort on
    data_path) reads them as one concatenated movie. Returns the output dir
    (suitable as TIFF_FOLDER for suite2p)."""
    if not tiff_paths:
        raise ValueError("shift_tiff_group: empty tiff_paths")

    out_dir = shifted_root / stem
    pad = max(2, len(str(len(tiff_paths) - 1)))
    out_paths = [
        out_dir / f"{i:0{pad}d}_shifted_{src.stem}{src.suffix}"
        for i, src in enumerate(tiff_paths)
    ]

    if all(p.exists() for p in out_paths):
        print(f"[shift/group] {stem}: all {len(out_paths)} outputs exist, "
              f"skipping")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: find global min/max across ALL inputs so they share one shift.
    gmin = np.iinfo(np.int64).max
    gmax = np.iinfo(np.int64).min
    page_counts: list[int] = []
    for src in tiff_paths:
        smin, smax, n = _scan_min_max(src, label="shift/group")
        page_counts.append(n)
        if smin < gmin: gmin = smin
        if smax > gmax: gmax = smax
    shift = -gmin if gmin < 0 else 0
    shifted_max = gmax + shift
    print(f"[shift/group] {stem}: pages={sum(page_counts)} "
          f"({page_counts})  min={gmin} max={gmax}  -> shift+={shift}  "
          f"(shifted max={shifted_max})")
    if shifted_max > 65535:
        raise ValueError(f"shifted max {shifted_max} > 65535; "
                         f"uint16 overflow for group {stem}")

    # Pass 2: write each shifted output (skip individual files already present).
    for src, dst in zip(tiff_paths, out_paths):
        if dst.exists():
            print(f"[shift/group] exists, skipping: {dst.name}")
            continue
        _write_shifted(src, dst, shift, label="shift/group")
    return out_dir


# ---------------------------------------------------------------------------
# 2. Sparse + Cellpose detection
# ---------------------------------------------------------------------------
def run_detection(shifted_folder: Path, detection_root: Path,
                  ops_path: Path) -> Path:
    """Run sparse_plus_cellpose.main() with mutated module globals.
    Returns the `final` folder (which contains suite2p/plane0)."""
    stem = shifted_folder.name
    save_folder = detection_root / stem
    final_folder = save_folder / "final"

    if (final_folder / "suite2p" / "plane0" / "stat.npy").exists():
        print(f"[detect] exists, skipping: {final_folder}")
        return final_folder

    import sparse_plus_cellpose as spc
    spc.TIFF_FOLDER = str(shifted_folder)
    spc.SAVE_FOLDER = str(save_folder)
    spc.PATH_TO_OPS = str(ops_path)
    print(f"[detect] running sparse+cellpose -> {final_folder}")
    spc.main()
    return final_folder


# ---------------------------------------------------------------------------
# 2b. Normalize F / Fneu so neither has negative values
# ---------------------------------------------------------------------------
def normalize_f_fneu(plane0: Path) -> None:
    """Shift F.npy and Fneu.npy by a single shared constant so neither has
    negative values (mirrors the raw-movie shift in preprocessing). Backs up
    the originals to F_orig.npy / Fneu_orig.npy the first time the shift is
    applied; re-runs read from those backups so the shift is never compounded.
    """
    F_path    = plane0 / "F.npy"
    Fneu_path = plane0 / "Fneu.npy"
    F_orig    = plane0 / "F_orig.npy"
    Fneu_orig = plane0 / "Fneu_orig.npy"
    label = plane0.parent.parent.name

    if not F_path.exists() or not Fneu_path.exists():
        print(f"[normalize] skip {label}: F.npy or Fneu.npy missing")
        return

    # If backups exist, they are the source of truth so re-runs don't
    # compound the shift.
    src_F    = F_orig    if F_orig.exists()    else F_path
    src_Fneu = Fneu_orig if Fneu_orig.exists() else Fneu_path
    F    = np.load(src_F)
    Fneu = np.load(src_Fneu)

    fmin, fnmin = float(F.min()), float(Fneu.min())
    gmin = min(fmin, fnmin)
    if gmin >= 0:
        print(f"[normalize] {label}: F.min={fmin:.3f} Fneu.min={fnmin:.3f}; "
              f"already non-negative")
        return

    shift = -gmin
    print(f"[normalize] {label}: F.min={fmin:.3f} Fneu.min={fnmin:.3f} "
          f"-> shift+={shift:.3f}")

    # First-time modification: snapshot the originals before overwriting.
    if not F_orig.exists():
        np.save(F_orig, F)
    if not Fneu_orig.exists():
        np.save(Fneu_orig, Fneu)

    np.save(F_path,    F    + shift)
    np.save(Fneu_path, Fneu + shift)


# ---------------------------------------------------------------------------
# 3. cellfilter predict
# ---------------------------------------------------------------------------
def run_cellfilter(plane0: Path, ckpt_path: Path | None = None) -> None:
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


def make_filtered_memmaps(plane0: Path, prefix: str = DEFAULT_PREFIX,
                          chunk_frames: int = 4096) -> None:
    """Apply predicted_cell_mask.npy along the ROI axis to produce
    {prefix}filtered_{dff,dff_dt,dff_lowpass}.memmap.float32 in plane0.
    Also writes r0p7_cell_mask_bool.npy so utils.s2p_open_memmaps works
    with prefix='r0p7_filtered_'."""
    kinds = ("dff", "dff_dt", "dff_lowpass")
    dst_prefix = f"{prefix}filtered_"

    mask_path = plane0 / "predicted_cell_mask.npy"
    F_path = plane0 / "F.npy"
    if not mask_path.exists() or not F_path.exists():
        print(f"[filtered] skip {plane0}: missing mask or F.npy")
        return

    F_shape = np.load(F_path, mmap_mode="r").shape
    if len(F_shape) != 2:
        print(f"[filtered] skip {plane0}: F.npy unexpected shape {F_shape}")
        return
    N, T = int(F_shape[0]), int(F_shape[1])

    mask = np.load(mask_path).astype(bool).ravel()
    if mask.size != N:
        print(f"[filtered] skip {plane0}: mask len {mask.size} != N {N}")
        return
    K = int(mask.sum())
    if K == 0:
        print(f"[filtered] skip {plane0}: mask keeps zero ROIs")
        return

    src_paths = {k: plane0 / f"{prefix}{k}.memmap.float32" for k in kinds}
    missing = [p.name for p in src_paths.values() if not p.exists()]
    if missing:
        print(f"[filtered] skip {plane0}: missing source memmaps {missing}")
        return
    expected_bytes = T * N * 4
    for p in src_paths.values():
        if p.stat().st_size != expected_bytes:
            print(f"[filtered] skip {plane0}: {p.name} size "
                  f"{p.stat().st_size} != expected {expected_bytes}")
            return

    cols = np.flatnonzero(mask)
    dst_paths = {k: plane0 / f"{dst_prefix}{k}.memmap.float32" for k in kinds}
    out_bytes = T * K * 4
    if all(p.exists() and p.stat().st_size == out_bytes for p in dst_paths.values()):
        bool_path = plane0 / "r0p7_cell_mask_bool.npy"
        if not bool_path.exists():
            np.save(bool_path, mask)
        print(f"[filtered] {plane0.parent.parent.name}: already present (K={K})")
        return

    print(f"[filtered] {plane0.parent.parent.name}: T={T} N={N} K={K} -> writing")
    for k in kinds:
        src = np.memmap(src_paths[k], dtype="float32", mode="r", shape=(T, N))
        dst = np.memmap(dst_paths[k], dtype="float32", mode="w+", shape=(T, K))
        for start in range(0, T, chunk_frames):
            end = min(T, start + chunk_frames)
            dst[start:end, :] = src[start:end, :][:, cols]
        dst.flush()
        del dst, src
        print(f"           wrote {dst_paths[k].name} ({T}x{K})")
    np.save(plane0 / "r0p7_cell_mask_bool.npy", mask)


# ---------------------------------------------------------------------------
# 4-7. analyze, image, cluster, heatmaps
# ---------------------------------------------------------------------------
def run_analyze(final_folder: Path, use_gpu: bool = True,
                baseline_sec: float = DEFAULT_BASELINE_SEC,
                ckpt_path: Path | None = None) -> bool:
    """Returns True if cellfilter prediction was performed inside the analyze
    step (GPU path); False otherwise (caller should run it separately)."""
    if use_gpu:
        try:
            import analyze_output_gpu
            if analyze_output_gpu._CUPY_OK:
                print(f"[analyze/gpu] {final_folder} (baseline={baseline_sec:.1f}s)")
                analyze_output_gpu.run_analysis_on_folder_gpu(
                    str(final_folder),
                    baseline_sec=baseline_sec,
                    run_cellfilter=True,
                    ckpt_path=str(ckpt_path) if ckpt_path else None,
                )
                return True
            print("[analyze] CuPy unavailable; falling back to CPU.")
        except ImportError as ex:
            print(f"[analyze] GPU import failed ({ex}); falling back to CPU.")

    import analyze_output
    print(f"[analyze/cpu] {final_folder}")
    analyze_output.run_analysis_on_folder(str(final_folder))
    return False


def run_image_all(final_folder: Path) -> None:
    import image_all
    print(f"[image_all] {final_folder}")
    image_all.run_full_imaging_on_folder(str(final_folder))


def run_cluster(plane0: Path, fps: float, prefix: str) -> None:
    import hierarchical_clustering as hc
    print(f"[cluster] {plane0}")
    hc.main(root=plane0, fps=fps, prefix=prefix,
            method="ward", metric="euclidean")


def run_spatial_heatmaps(final_folder: Path, plane0: Path,
                         fps: float, prefix: str) -> None:
    import spatial_heatmap_updated as shu
    mask_path = plane0 / "predicted_cell_mask.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"predicted_cell_mask.npy missing at {mask_path}")

    print(f"[heatmap/vectors] {final_folder}")
    shu.coactivation_maps(
        folder_name=str(final_folder),
        cell_mask_path=str(mask_path),
        propagation_vectors=True,
        prefix=prefix,
        fps=fps,
    )


# ---------------------------------------------------------------------------
# Per-recording driver
# ---------------------------------------------------------------------------
def run_one(tiff_path: Path, shifted_root: Path, detection_root: Path,
            ops_path: Path, ckpt: Path | None, fps: float, prefix: str,
            stem: str | None = None,
            use_gpu_analyze: bool = True,
            baseline_sec: float = DEFAULT_BASELINE_SEC) -> None:
    t0 = time.time()
    label = stem or tiff_path.stem
    print(f"\n{'=' * 72}\n[pipeline] {label} ({tiff_path})\n{'=' * 72}")

    shifted_folder = shift_tiff(tiff_path, shifted_root, stem=stem)
    final_folder = run_detection(shifted_folder, detection_root, ops_path)
    plane0 = final_folder / "suite2p" / "plane0"
    normalize_f_fneu(plane0)

    cellfilter_done = run_analyze(
        final_folder, use_gpu=use_gpu_analyze,
        baseline_sec=baseline_sec, ckpt_path=ckpt,
    )
    if not cellfilter_done:
        run_cellfilter(plane0, ckpt_path=ckpt)
    make_filtered_memmaps(plane0, prefix=prefix)
    run_image_all(final_folder)
    run_cluster(plane0, fps=fps, prefix=prefix)
    run_spatial_heatmaps(final_folder, plane0, fps=fps, prefix=prefix)

    print(f"[pipeline] DONE {label} in {time.time() - t0:.1f}s")


def run_one_group(tiff_paths: list[Path], stem: str,
                  shifted_root: Path, detection_root: Path,
                  ops_path: Path, ckpt: Path | None, fps: float, prefix: str,
                  use_gpu_analyze: bool = True,
                  baseline_sec: float = DEFAULT_BASELINE_SEC) -> None:
    """Run the pipeline on a group of TIFFs treated as one continuous
    recording. Suite2p concatenates them via data_path (natsort on the
    zero-padded shifted filenames preserves input order)."""
    t0 = time.time()
    print(f"\n{'=' * 72}\n[pipeline/group] {stem}  ({len(tiff_paths)} TIFFs)")
    for p in tiff_paths:
        print(f"    {p}")
    print('=' * 72)

    shifted_folder = shift_tiff_group(tiff_paths, shifted_root, stem=stem)
    final_folder = run_detection(shifted_folder, detection_root, ops_path)
    plane0 = final_folder / "suite2p" / "plane0"
    normalize_f_fneu(plane0)

    cellfilter_done = run_analyze(
        final_folder, use_gpu=use_gpu_analyze,
        baseline_sec=baseline_sec, ckpt_path=ckpt,
    )
    if not cellfilter_done:
        run_cellfilter(plane0, ckpt_path=ckpt)
    make_filtered_memmaps(plane0, prefix=prefix)
    run_image_all(final_folder)
    run_cluster(plane0, fps=fps, prefix=prefix)
    run_spatial_heatmaps(final_folder, plane0, fps=fps, prefix=prefix)

    print(f"[pipeline/group] DONE {stem} in {time.time() - t0:.1f}s")


def run_one_from_cellfilter(rec_dir: Path, ckpt: Path | None,
                            fps: float, prefix: str) -> None:
    """Skip shift / detection / analyze. Re-run cellfilter onwards on an
    existing recording directory (must contain final/suite2p/plane0)."""
    t0 = time.time()
    label = rec_dir.name
    print(f"\n{'=' * 72}\n[pipeline/from-cellfilter] {label} ({rec_dir})\n{'=' * 72}")

    final_folder = rec_dir / "final"
    plane0 = final_folder / "suite2p" / "plane0"
    if not plane0.is_dir():
        raise FileNotFoundError(f"no plane0 at {plane0}")

    normalize_f_fneu(plane0)
    run_cellfilter(plane0, ckpt_path=ckpt)
    make_filtered_memmaps(plane0, prefix=prefix)
    run_image_all(final_folder)
    run_cluster(plane0, fps=fps, prefix=prefix)
    run_spatial_heatmaps(final_folder, plane0, fps=fps, prefix=prefix)

    print(f"[pipeline/from-cellfilter] DONE {label} in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Manifest loader (xlsx with Name / Location columns)
# ---------------------------------------------------------------------------
def load_manifest(xlsx_path: Path) -> list[tuple[str, Path]]:
    """Read an xlsx with columns 'Name' and 'Location'. Returns a list of
    (stem, tiff_path) tuples. Missing files are skipped with a warning."""
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    cols = {c.lower(): c for c in df.columns}
    if "name" not in cols or "location" not in cols:
        raise ValueError(f"{xlsx_path} must have 'Name' and 'Location' columns")
    name_col, loc_col = cols["name"], cols["location"]

    out: list[tuple[str, Path]] = []
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        loc = Path(str(row[loc_col]).strip())
        if not name or name.lower() == "nan":
            continue
        if not loc.exists():
            print(f"[manifest] MISSING, skipping: {name} -> {loc}")
            continue
        out.append((name, loc))
    print(f"[manifest] loaded {len(out)} recordings from {xlsx_path}")
    return out


def _trailing_index(p: Path) -> int:
    """Last run of digits in a filename stem. e.g. '2026-02-04_00015.tif' -> 15.
    Used to order multiple TIFFs of one identifier by acquisition sequence."""
    import re
    m = re.search(r"(\d+)(?!.*\d)", p.stem)
    if not m:
        raise ValueError(f"cannot extract trailing index from {p.name}")
    return int(m.group(1))


def load_group_manifest(xlsx_path: Path) -> list[tuple[str, list[Path]]]:
    """Read an xlsx with 'Identifyer' (or 'Identifier') and 'location' columns.

    Returns [(identifier, [tiff_paths sorted by trailing index]), ...] in the
    order identifiers first appear in the sheet. Missing files are dropped
    with a warning."""
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    cols = {c.lower(): c for c in df.columns}
    id_key = next((cols[k] for k in ("identifyer", "identifier") if k in cols),
                  None)
    if id_key is None or "location" not in cols:
        raise ValueError(f"{xlsx_path} must have 'Identifyer' (or 'Identifier') "
                         f"and 'location' columns; got {list(df.columns)}")
    loc_key = cols["location"]

    groups: dict[str, list[Path]] = {}
    order: list[str] = []
    for _, row in df.iterrows():
        ident = str(row[id_key]).strip()
        loc_raw = str(row[loc_key]).strip()
        if not ident or ident.lower() == "nan":
            continue
        if not loc_raw or loc_raw.lower() == "nan":
            continue
        loc = Path(loc_raw)
        if not loc.exists():
            print(f"[group-manifest] MISSING, skipping: {ident} -> {loc}")
            continue
        if ident not in groups:
            groups[ident] = []
            order.append(ident)
        groups[ident].append(loc)

    out: list[tuple[str, list[Path]]] = []
    for ident in order:
        paths = sorted(groups[ident], key=_trailing_index)
        out.append((ident, paths))
        idx_str = ", ".join(str(_trailing_index(p)) for p in paths)
        print(f"[group-manifest] {ident}: {len(paths)} TIFFs (indices: {idx_str})")
    print(f"[group-manifest] loaded {len(out)} identifiers from {xlsx_path}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tiff", type=str,
                     help="Single raw TIFF to process.")
    src.add_argument("--tiff-dir", type=str,
                     help="Directory of raw TIFFs to batch process.")
    src.add_argument("--xlsx", type=str,
                     help="Manifest xlsx with Name / Location columns.")
    src.add_argument("--group-xlsx", type=str,
                     help="Manifest xlsx with 'Identifyer' / 'location' columns. "
                          "Rows sharing an identifier are treated as one "
                          "continuous recording: each input TIFF is shifted "
                          "with a shared global-min, written into "
                          "<shifted-root>/<id>/ with a zero-padded order "
                          "prefix, and suite2p reads the folder as one "
                          "concatenated movie via natsort on data_path.")
    src.add_argument("--from-cellfilter", action="store_true",
                     help="Skip shift/detection/analyze. Iterate existing "
                          "recording dirs under --detection-root and re-run "
                          "cellfilter -> image_all -> cluster -> spatial_heatmaps.")
    ap.add_argument("--rec", type=str, default=None,
                    help="(only with --from-cellfilter) comma-separated "
                         "recording names to limit to; default = all subdirs "
                         "of --detection-root.")
    ap.add_argument("--shifted-root", type=str,
                    default=str(DEFAULT_SHIFTED_ROOT))
    ap.add_argument("--detection-root", type=str,
                    default=str(DEFAULT_DETECTION_ROOT))
    ap.add_argument("--ops", type=str, default=str(DEFAULT_OPS))
    ap.add_argument("--ckpt", type=str, default=None,
                    help="cellfilter checkpoint (defaults to config best.pt)")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    ap.add_argument("--no-gpu-analyze", action="store_true",
                    help="Force CPU analyze step (default: use GPU if CuPy available).")
    ap.add_argument("--baseline-sec", type=float, default=DEFAULT_BASELINE_SEC,
                    help=f"Baseline window (s) for F0 in GPU analyze (default {DEFAULT_BASELINE_SEC}).")
    args = ap.parse_args()

    shifted_root = Path(args.shifted_root)
    detection_root = Path(args.detection_root)
    ops_path = Path(args.ops)
    ckpt = Path(args.ckpt) if args.ckpt else None
    shifted_root.mkdir(parents=True, exist_ok=True)
    detection_root.mkdir(parents=True, exist_ok=True)

    if args.from_cellfilter:
        if args.rec:
            wanted = {n.strip() for n in args.rec.split(",") if n.strip()}
            rec_dirs = [detection_root / n for n in sorted(wanted)]
            missing = [str(p) for p in rec_dirs if not p.is_dir()]
            if missing:
                raise SystemExit(f"missing recording dirs: {missing}")
        else:
            rec_dirs = sorted(p for p in detection_root.iterdir() if p.is_dir())
        print(f"[pipeline/from-cellfilter] {len(rec_dirs)} recordings under {detection_root}")
        for rec_dir in rec_dirs:
            try:
                run_one_from_cellfilter(rec_dir, ckpt, args.fps, args.prefix)
            except Exception as ex:
                print(f"[pipeline] FAILED {rec_dir.name}: {ex}")
                import traceback
                traceback.print_exc()
        return

    if args.group_xlsx:
        groups = load_group_manifest(Path(args.group_xlsx))
        for ident, tiff_paths in groups:
            try:
                run_one_group(
                    tiff_paths, ident, shifted_root, detection_root, ops_path,
                    ckpt, args.fps, args.prefix,
                    use_gpu_analyze=not args.no_gpu_analyze,
                    baseline_sec=args.baseline_sec,
                )
            except Exception as ex:
                print(f"[pipeline] FAILED {ident}: {ex}")
                import traceback
                traceback.print_exc()
        return

    if args.tiff:
        jobs: list[tuple[str | None, Path]] = [(None, Path(args.tiff))]
    elif args.tiff_dir:
        d = Path(args.tiff_dir)
        tiffs = sorted([p for p in d.iterdir()
                        if p.is_file() and p.suffix.lower() in (".tif", ".tiff")])
        print(f"[pipeline] found {len(tiffs)} TIFFs in {d}")
        jobs = [(None, p) for p in tiffs]
    else:
        jobs = [(name, path) for name, path in load_manifest(Path(args.xlsx))]

    for stem, tp in jobs:
        try:
            run_one(tp, shifted_root, detection_root, ops_path,
                    ckpt, args.fps, args.prefix, stem=stem,
                    use_gpu_analyze=not args.no_gpu_analyze,
                    baseline_sec=args.baseline_sec)
        except Exception as ex:
            print(f"[pipeline] FAILED {stem or tp.name}: {ex}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
