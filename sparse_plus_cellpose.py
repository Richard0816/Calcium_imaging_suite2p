"""Sparsery + Cellpose detection (standalone).

Run sparsery once at a fixed threshold, run cellpose once on the mean
image, merge non-overlapping cellpose ROIs into the sparsery set, then
extract fluorescence + deconvolve + classify + save.

This module is self-contained — no dependency on adaptive_detection or
brute_force_ops. The previous version imported helpers from those two
files; they have been inlined here so the GUI can drive detection
without dragging in the adaptive search machinery.

Usage:
    # CLI (uses TIFF_FOLDER / SAVE_FOLDER constants below)
    python sparse_plus_cellpose.py

    # Programmatic
    from sparse_plus_cellpose import run
    final_plane0 = run(
        tiff_folder=...,
        save_folder=...,
        path_to_ops=...,
    )
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

WORKTREE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKTREE))

import suite2p
from suite2p.detection.stats import roi_stats
from suite2p.extraction import extract, dcnv
from suite2p import classification

import utils

try:
    from cellpose import models as _cp_models  # type: ignore
    _CELLPOSE_AVAILABLE = True
    _CELLPOSE_IMPORT_ERROR: Optional[str] = None
except Exception as _e:  # pragma: no cover
    _cp_models = None
    _CELLPOSE_AVAILABLE = False
    _CELLPOSE_IMPORT_ERROR = repr(_e)


# ============================================================================
# CONFIGURATION (defaults — used when run as a script)
# ============================================================================

TIFF_FOLDER = r'D:\2024-11-20_00003'
SAVE_FOLDER = r'D:\sparse_plus_cellpose\2024-11-20_00003'
PATH_TO_OPS = r'suite2p_2p_ops_240621.npy'

DEFAULT_SPARSERY_OPS = {
    'high_pass':          100,
    'preclassify':        0.0,
    'smooth_sigma':       1.0,
    'sparse_mode':        True,
    'spatial_scale':      0,      # 0 = suite2p auto
    'threshold_scaling':  0.85,
    'max_iterations':     1500,
}

DEFAULT_CELLPOSE_CFG = {
    'cellpose_model_type':         'cyto2',
    'cellpose_diameter':           0,      # 0 = auto
    'cellpose_flow_threshold':     0.8,
    'cellpose_cellprob_threshold': -1.0,
    'cellpose_channel_input':      'meanImg',
}

DEFAULT_TAU_VALS = {"6f": 0.7, "6m": 1.0, "6s": 1.3, "8m": 0.137}

HARD_CAP = 60000
CELLPOSE_MERGE_MAX_OVERLAP = 0.3


# ============================================================================
# Sparsery hard-cap (mid-detection abort via print monkey-patch)
# ============================================================================

class _RoiHardCapExceeded(Exception):
    """Raised from inside sparsery when ROI count crosses the cap."""
    def __init__(self, count: int, cap: int):
        super().__init__(
            f"sparsery produced {count} ROIs (>= cap={cap}); "
            f"threshold is almost certainly detecting noise"
        )
        self.count = count
        self.cap = cap


_SPARSERY_PROGRESS_RE = re.compile(r'^\s*(\d+)\s+ROIs')


def _install_sparsery_roi_cap(cap: int):
    import suite2p.detection.sparsedetect as _sd
    original_print = getattr(_sd, 'print', print)

    def _watching_print(*args, **kwargs):
        if args:
            msg = args[0] if isinstance(args[0], str) else str(args[0])
            m = _SPARSERY_PROGRESS_RE.match(msg)
            if m and int(m.group(1)) >= cap:
                raise _RoiHardCapExceeded(int(m.group(1)), cap)
        return original_print(*args, **kwargs)

    _sd.print = _watching_print
    return original_print


def _restore_sparsery_print(original_print):
    import suite2p.detection.sparsedetect as _sd
    if original_print is print:
        if 'print' in _sd.__dict__:
            del _sd.__dict__['print']
    else:
        _sd.print = original_print


# ============================================================================
# Helpers
# ============================================================================

def _link_or_copy(src, dst):
    """Hardlink (cheap) or copy on cross-device / permission error."""
    src = Path(src); dst = Path(dst)
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except (OSError, NotImplementedError):
        shutil.copy2(str(src), str(dst))


def build_roi_pixel_mask(stat, Ly: int, Lx: int) -> np.ndarray:
    """Binary mask of all pixels belonging to any detected ROI."""
    mask = np.zeros((Ly, Lx), dtype=bool)
    for s in stat:
        yp = np.asarray(s['ypix']); xp = np.asarray(s['xpix'])
        ok = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
        mask[yp[ok], xp[ok]] = True
    return mask


def _cellpose_masks_to_stat(masks: np.ndarray) -> list[dict]:
    """Convert a cellpose integer label mask into suite2p-style stat dicts."""
    stat: list[dict] = []
    if masks is None:
        return stat
    labels = np.unique(masks)
    labels = labels[labels > 0]
    for lbl in labels:
        ys, xs = np.where(masks == lbl)
        if ys.size == 0:
            continue
        npix = int(ys.size)
        stat.append({
            'ypix': ys.astype(np.int32),
            'xpix': xs.astype(np.int32),
            'lam':  np.ones(npix, dtype=np.float32) / float(npix),
            'med':  [float(np.median(ys)), float(np.median(xs))],
            'npix': npix,
            'radius': float(np.sqrt(npix / np.pi)),
        })
    return stat


# ============================================================================
# Ops loading
# ============================================================================

def load_base_ops(
    tiff_folder,
    path_to_ops: Optional[str] = None,
    aav_info_csv: Optional[str] = None,
    tau_vals: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """Load starting ops + apply per-recording tau, batch_size, nbinned, and
    pipeline-required overrides (sparse_mode, allow_overlap, etc.).
    """
    if path_to_ops and os.path.exists(path_to_ops):
        if verbose:
            print(f"Loading base ops from {path_to_ops}")
        ops = np.load(path_to_ops, allow_pickle=True).item()
    else:
        if verbose:
            print("No base ops provided, starting from suite2p defaults")
        ops = suite2p.default_ops()
        ops.update({
            'fs': 15.07, 'nchannels': 1, 'nplanes': 1,
            'high_pass': 100.0, 'smooth_sigma': 1.3,
        })

    pipeline_required = {
        'sparse_mode':   True,
        'spatial_scale': 0,         # 0 = suite2p auto
        'preclassify':   0.5,
        'allow_overlap': False,
    }
    if verbose:
        for k, v in pipeline_required.items():
            prev = ops.get(k, '<unset>')
            if prev != v:
                print(f"  enforcing ops['{k}'] = {v}  (was {prev!r})")
    ops.update(pipeline_required)

    if aav_info_csv and tau_vals and os.path.exists(aav_info_csv):
        try:
            file_name = os.path.basename(os.path.normpath(str(tiff_folder)))
            tau = utils.file_name_to_aav_to_dictionary_lookup(
                file_name, aav_info_csv, tau_vals)
            ops['tau'] = tau
            if verbose:
                print(f"Set tau={tau} via AAV lookup for {file_name}")
        except Exception as e:
            if verbose:
                print(f"tau lookup failed ({e}); keeping tau={ops.get('tau')}")

    ops['batch_size'] = utils.change_batch_according_to_free_ram()
    safe_nbinned = utils.change_nbinned_according_to_free_ram(str(tiff_folder))
    cur_nbinned = ops.get('nbinned', 0)
    if cur_nbinned <= 0:
        if verbose:
            print(f"Setting nbinned = {safe_nbinned}")
        ops['nbinned'] = safe_nbinned
    elif cur_nbinned > safe_nbinned:
        if verbose:
            print(f"Lowering nbinned {cur_nbinned} -> {safe_nbinned}")
        ops['nbinned'] = safe_nbinned

    if verbose:
        keys = ('sparse_mode', 'spatial_scale', 'preclassify', 'allow_overlap',
                'high_pass', 'smooth_sigma', 'tau', 'fs', 'nbinned',
                'batch_size')
        summary = ', '.join(f"{k}={ops.get(k)!r}" for k in keys)
        print(f"  effective ops: {summary}")
    return ops


# ============================================================================
# Shared registration + sparsery + cellpose
# ============================================================================

def _get_or_create_shared_registration(
    tiff_folder, save_folder, base_ops: dict, verbose: bool = True,
) -> Path:
    """Run suite2p register-only into ``{save_folder}/_shared_reg/`` and return
    its plane0. Subsequent passes hardlink ``data.bin`` from here so the TIFF
    is registered exactly once.
    """
    save_root = Path(save_folder)
    shared_dir = save_root / '_shared_reg'
    shared_plane0 = shared_dir / 'suite2p' / 'plane0'

    if (shared_plane0 / 'data.bin').exists() and (shared_plane0 / 'ops.npy').exists():
        if verbose:
            print(f"  > reusing shared registration at {shared_plane0}")
        return shared_plane0

    if verbose:
        print(f"  > running suite2p register-only -> {shared_dir}")

    reg_ops = dict(base_ops)
    reg_ops['save_path0'] = str(shared_dir)
    reg_ops['save_folder'] = 'suite2p'
    reg_ops['roidetect'] = False
    db = {'data_path': [str(tiff_folder)]}
    suite2p.run_s2p(ops=reg_ops, db=db)

    if not (shared_plane0 / 'data.bin').exists():
        raise RuntimeError(
            f"Registration completed but data.bin was not written to "
            f"{shared_plane0}. Check suite2p output for errors."
        )
    return shared_plane0


def run_sparsery_pass(
    save_dir, shared_plane0, ops: dict,
    verbose: bool = True, hard_cap: int = 0,
):
    """Detection-only sparsery pass that reuses an already-registered binary.

    Hardlinks ``data.bin`` from ``shared_plane0`` into ``{save_dir}/suite2p/plane0/``,
    writes a detection-only ops.npy with ``do_registration=0``, and invokes
    ``suite2p.run_plane`` directly.
    """
    plane0 = Path(save_dir) / 'suite2p' / 'plane0'
    plane0.mkdir(parents=True, exist_ok=True)
    _link_or_copy(Path(shared_plane0) / 'data.bin', plane0 / 'data.bin')

    reg_ops = np.load(Path(shared_plane0) / 'ops.npy',
                      allow_pickle=True).item()
    pass_ops = dict(reg_ops)
    detection_keys = (
        'threshold_scaling', 'max_iterations', 'spatial_scale',
        'sparse_mode', 'high_pass', 'smooth_sigma', 'preclassify',
        'allow_overlap', 'max_overlap', 'tau', 'fs', 'nbinned',
        'batch_size', 'roidetect',
    )
    for k in detection_keys:
        if k in ops:
            pass_ops[k] = ops[k]
    pass_ops['do_registration'] = 0
    pass_ops['save_path0'] = str(save_dir)
    pass_ops['save_folder'] = 'suite2p'
    pass_ops['reg_file'] = str(plane0 / 'data.bin')
    pass_ops['ops_path'] = str(plane0 / 'ops.npy')
    pass_ops.setdefault('roidetect', True)
    np.save(plane0 / 'ops.npy', pass_ops, allow_pickle=True)

    if verbose:
        print(f"  > sparsery: threshold_scaling={pass_ops.get('threshold_scaling')} "
              f"spatial_scale={pass_ops.get('spatial_scale')} "
              f"max_iterations={pass_ops.get('max_iterations')}")

    _orig = (_install_sparsery_roi_cap(hard_cap)
             if hard_cap and hard_cap > 0 else None)
    try:
        suite2p.run_plane(pass_ops, ops_path=str(plane0 / 'ops.npy'))
    except _RoiHardCapExceeded as e:
        if verbose:
            print(f"  > ABORT: {e} -- returning empty pass")
        ops_loaded = (np.load(plane0 / 'ops.npy',
                              allow_pickle=True).item()
                      if (plane0 / 'ops.npy').exists() else pass_ops)
        try:
            np.save(plane0 / 'stat.npy',
                    np.array([], dtype=object), allow_pickle=True)
            (plane0 / 'HARD_CAP_ABORTED.txt').write_text(
                f"Sparsery aborted at ~{e.count} ROIs (cap={e.cap}).\n")
        except Exception:
            pass
        return [], ops_loaded, plane0
    except ValueError as e:
        if 'no ROIs' in str(e) or 'ROIs were found' in str(e):
            if verbose:
                print("  > suite2p found 0 ROIs; returning empty pass")
            ops_loaded = (np.load(plane0 / 'ops.npy',
                                  allow_pickle=True).item()
                          if (plane0 / 'ops.npy').exists() else pass_ops)
            try:
                np.save(plane0 / 'stat.npy',
                        np.array([], dtype=object), allow_pickle=True)
            except Exception:
                pass
            return [], ops_loaded, plane0
        raise
    finally:
        if _orig is not None:
            _restore_sparsery_print(_orig)

    stat = list(np.load(plane0 / 'stat.npy', allow_pickle=True))
    ops_loaded = np.load(plane0 / 'ops.npy', allow_pickle=True).item()
    return stat, ops_loaded, plane0


def run_cellpose_pass(save_dir, shared_plane0, cfg: dict, verbose: bool = True):
    """Run Cellpose on the mean image from the shared registration.

    Returns ``(stat, ops_out, plane0)`` mirroring run_sparsery_pass.
    """
    if not _CELLPOSE_AVAILABLE:
        raise RuntimeError(f'cellpose not importable: {_CELLPOSE_IMPORT_ERROR}')

    plane0 = Path(save_dir) / 'suite2p' / 'plane0'
    plane0.mkdir(parents=True, exist_ok=True)

    reg_ops = np.load(Path(shared_plane0) / 'ops.npy',
                      allow_pickle=True).item()
    img_key = cfg.get('cellpose_channel_input', 'meanImg')
    if img_key not in reg_ops or reg_ops[img_key] is None:
        img_key = 'meanImg'
    img = np.asarray(reg_ops[img_key]).astype(np.float32)

    lo = float(np.quantile(img, 0.01))
    hi = float(np.quantile(img, 0.999))
    if hi <= lo:
        hi = float(img.max()); lo = float(img.min())
    span = max(hi - lo, 1e-6)
    img_norm = np.clip((img - lo) / span, 0.0, 1.0)

    model_type = cfg.get('cellpose_model_type', 'cyto')
    diameter = cfg.get('cellpose_diameter', 0) or None
    flow_threshold = float(cfg.get('cellpose_flow_threshold', 0.4))
    cellprob_threshold = float(cfg.get('cellpose_cellprob_threshold', 0.0))

    has_wrapper = hasattr(_cp_models, 'Cellpose')
    if verbose:
        api = '3x' if has_wrapper else '4x'
        print(f"  > cellpose ({api}): model={model_type}  diameter={diameter}  "
              f"flow={flow_threshold}  cellprob={cellprob_threshold}  "
              f"img={img_key} {img.shape}")

    t0 = time.time()
    if has_wrapper:
        model = _cp_models.Cellpose(model_type=model_type, gpu=True)
        eval_kwargs = dict(
            diameter=diameter, channels=[0, 0],
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
    else:
        model = _cp_models.CellposeModel(gpu=True)
        eval_kwargs = dict(
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )

    eval_out = model.eval(img_norm, **eval_kwargs)
    masks = eval_out[0]
    diams = eval_out[3] if len(eval_out) >= 4 else None
    if verbose:
        print(f"  > cellpose done in {time.time() - t0:.1f}s  "
              f"(est. diameter={diams})")

    stat = _cellpose_masks_to_stat(np.asarray(masks))

    ops_out = dict(reg_ops)
    ops_out['Ly'] = int(img.shape[0])
    ops_out['Lx'] = int(img.shape[1])
    ops_out['cellpose_config'] = {
        'model_type': model_type,
        'diameter_requested': cfg.get('cellpose_diameter', 0),
        'diameter_estimated': float(diams) if diams is not None else None,
        'flow_threshold': flow_threshold,
        'cellprob_threshold': cellprob_threshold,
        'channel_input': img_key,
    }
    np.save(plane0 / 'ops.npy', ops_out, allow_pickle=True)
    np.save(plane0 / 'stat.npy', np.array(stat, dtype=object),
            allow_pickle=True)
    np.save(plane0 / 'cellpose_masks.npy', np.asarray(masks))
    return stat, ops_out, plane0


# ============================================================================
# Merge + extract
# ============================================================================

def filter_non_overlapping_cellpose(
    sparsery_stat, cellpose_stat, Ly: int, Lx: int,
    max_overlap: float, verbose: bool = True,
):
    """Keep cellpose ROIs with < max_overlap fraction covered by sparsery."""
    sparsery_mask = build_roi_pixel_mask(sparsery_stat, Ly, Lx)
    kept, dropped = [], 0
    for s in cellpose_stat:
        yp = np.asarray(s['ypix']).astype(int)
        xp = np.asarray(s['xpix']).astype(int)
        ok = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
        yp = yp[ok]; xp = xp[ok]
        if yp.size == 0:
            dropped += 1
            continue
        overlap = float(sparsery_mask[yp, xp].sum()) / float(yp.size)
        if overlap < max_overlap:
            kept.append(s)
        else:
            dropped += 1
    if verbose:
        print(f"    cellpose merge: kept {len(kept)}/{len(cellpose_stat)} "
              f"ROIs (dropped {dropped} overlapping sparsery)")
    return kept


def merge_and_extract(
    sparsery_stat, cellpose_stat, shared_plane0, base_ops: dict,
    final_dir, max_overlap: float, verbose: bool = True,
):
    """Combine sparsery + non-overlapping cellpose ROIs, extract fluorescence,
    deconvolve, classify, and save the final suite2p-layout outputs into
    ``{final_dir}/suite2p/plane0/``.
    """
    final_plane0 = Path(final_dir) / 'suite2p' / 'plane0'
    final_plane0.mkdir(parents=True, exist_ok=True)

    _link_or_copy(Path(shared_plane0) / 'data.bin',
                  final_plane0 / 'data.bin')
    reg_ops = np.load(Path(shared_plane0) / 'ops.npy',
                      allow_pickle=True).item()
    Ly, Lx = int(reg_ops['Ly']), int(reg_ops['Lx'])

    cellpose_kept = filter_non_overlapping_cellpose(
        sparsery_stat, cellpose_stat, Ly, Lx,
        max_overlap=max_overlap, verbose=verbose,
    )

    combined_raw = []
    for s in sparsery_stat:
        combined_raw.append({
            'ypix': np.asarray(s['ypix']).astype(np.int32),
            'xpix': np.asarray(s['xpix']).astype(np.int32),
            'lam':  np.asarray(s['lam']).astype(np.float32),
            'med':  list(s.get('med', [int(np.median(s['ypix'])),
                                        int(np.median(s['xpix']))])),
            '_source': 'sparsery',
        })
    for s in cellpose_kept:
        combined_raw.append({
            'ypix': np.asarray(s['ypix']).astype(np.int32),
            'xpix': np.asarray(s['xpix']).astype(np.int32),
            'lam':  np.asarray(s['lam']).astype(np.float32),
            'med':  list(s.get('med', [int(np.median(s['ypix'])),
                                        int(np.median(s['xpix']))])),
            '_source': 'cellpose',
        })

    n_sp = len(sparsery_stat); n_cp = len(cellpose_kept)
    if verbose:
        print(f"    merged stat: {n_sp} sparsery + {n_cp} cellpose = "
              f"{len(combined_raw)} total ROIs")

    if not combined_raw:
        raise RuntimeError('merge produced 0 ROIs — nothing to extract')

    final_ops = dict(reg_ops)
    for k in ('tau', 'fs', 'neucoeff', 'baseline', 'win_baseline',
              'sig_baseline', 'prctile_baseline', 'batch_size'):
        if k in base_ops:
            final_ops[k] = base_ops[k]
    final_ops['save_path0'] = str(final_dir)
    final_ops['save_folder'] = 'suite2p'
    final_ops['reg_file']    = str(final_plane0 / 'data.bin')
    final_ops['ops_path']    = str(final_plane0 / 'ops.npy')
    final_ops['do_registration'] = 0
    final_ops['Ly'] = Ly
    final_ops['Lx'] = Lx

    median_diam = max(3, int(round(
        2.0 * float(np.sqrt(np.mean(
            [len(s['ypix']) for s in combined_raw]) / np.pi)),
    )))
    enriched = roi_stats(combined_raw, Ly, Lx, diameter=median_diam)
    for i, s in enumerate(enriched):
        if '_source' not in s:
            s['_source'] = combined_raw[i].get('_source', 'sparsery')
    stat_arr = np.array(enriched, dtype=object)

    if verbose:
        print(f"    extracting traces for {len(stat_arr)} ROIs")
    stat_arr, F, Fneu, F_chan2, Fneu_chan2 = extract.create_masks_and_extract(
        final_ops, stat_arr,
    )

    tau = float(final_ops.get('tau', 1.0))
    fs = float(final_ops.get('fs', 15.0))
    neucoeff = float(final_ops.get('neucoeff', 0.7))
    F_sub = F - neucoeff * Fneu
    F_pp = dcnv.preprocess(
        F=F_sub,
        baseline=final_ops.get('baseline', 'maximin'),
        win_baseline=float(final_ops.get('win_baseline', 60.0)),
        sig_baseline=float(final_ops.get('sig_baseline', 10.0)),
        fs=fs,
        prctile_baseline=float(final_ops.get('prctile_baseline', 8.0)),
    )
    spks = dcnv.oasis(
        F=F_pp, batch_size=int(final_ops.get('batch_size', 3000)),
        tau=tau, fs=fs,
    )

    try:
        iscell = classification.classify(
            stat=stat_arr, classfile=classification.builtin_classfile,
        )
    except Exception as e:
        if verbose:
            print(f"    classifier failed ({e}); marking all as cells")
        iscell = np.ones((len(stat_arr), 2), dtype=np.float32)

    np.save(final_plane0 / 'stat.npy',   stat_arr,  allow_pickle=True)
    np.save(final_plane0 / 'F.npy',      F)
    np.save(final_plane0 / 'Fneu.npy',   Fneu)
    np.save(final_plane0 / 'spks.npy',   spks)
    np.save(final_plane0 / 'iscell.npy', iscell)
    np.save(final_plane0 / 'ops.npy',    final_ops, allow_pickle=True)

    if verbose:
        n_accept = (int((iscell[:, 0] > 0).sum())
                    if iscell.ndim == 2 else int((iscell > 0).sum()))
        print(f"    final: {len(stat_arr)} ROIs "
              f"({n_accept} classified as cells)")

    return stat_arr, final_ops, final_plane0


# ============================================================================
# Public entry point
# ============================================================================

def run(
    tiff_folder,
    save_folder,
    path_to_ops: Optional[str] = None,
    sparsery_ops: Optional[dict] = None,
    cellpose_cfg: Optional[dict] = None,
    hard_cap: int = HARD_CAP,
    max_overlap: float = CELLPOSE_MERGE_MAX_OVERLAP,
    aav_info_csv: Optional[str] = None,
    tau_vals: Optional[dict] = None,
    verbose: bool = True,
) -> Path:
    """Sparsery + cellpose detection. Returns path to final ``suite2p/plane0``."""
    sparsery_ops = (dict(DEFAULT_SPARSERY_OPS) if sparsery_ops is None
                    else dict(sparsery_ops))
    cellpose_cfg = (dict(DEFAULT_CELLPOSE_CFG) if cellpose_cfg is None
                    else dict(cellpose_cfg))

    save_root = Path(save_folder)
    save_root.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f'[s+cp] loading base ops')
    base_ops = load_base_ops(
        tiff_folder, path_to_ops, aav_info_csv, tau_vals, verbose=verbose,
    )

    if verbose:
        print(f'[s+cp] shared registration')
    t0 = time.time()
    shared_plane0 = _get_or_create_shared_registration(
        tiff_folder, save_folder, base_ops, verbose=verbose,
    )
    if verbose:
        print(f'[s+cp] shared reg ready at {shared_plane0}  '
              f'({time.time() - t0:.1f}s)')

    if verbose:
        print(f'[s+cp] sparsery pass '
              f'(threshold_scaling={sparsery_ops["threshold_scaling"]})')
    sparsery_dir = save_root / 'sparsery_pass'
    ops = dict(base_ops)
    ops.update(sparsery_ops)
    ops['roidetect'] = True
    t_sp = time.time()
    sp_stat, sp_ops, sp_plane0 = run_sparsery_pass(
        sparsery_dir, shared_plane0, ops,
        verbose=verbose, hard_cap=(hard_cap or 0),
    )
    if verbose:
        print(f'[s+cp] sparsery: {len(sp_stat)} ROIs '
              f'({time.time() - t_sp:.1f}s)')

    if verbose:
        print(f'[s+cp] cellpose pass')
    cp_dir = save_root / 'cellpose_pass'
    t_cp = time.time()
    cp_stat, cp_ops, cp_plane0 = run_cellpose_pass(
        cp_dir, shared_plane0, cellpose_cfg, verbose=verbose,
    )
    if verbose:
        print(f'[s+cp] cellpose: {len(cp_stat)} ROIs '
              f'({time.time() - t_cp:.1f}s)')

    if verbose:
        print(f'[s+cp] merging + extracting')
    final_dir = save_root / 'final'
    stat_arr, final_ops, final_plane0 = merge_and_extract(
        sp_stat, cp_stat, shared_plane0, base_ops, final_dir,
        max_overlap=max_overlap, verbose=verbose,
    )

    if verbose:
        print(f'\n[s+cp] DONE.')
        print(f'  plane0:    {final_plane0}')
        print(f'  stat.npy:  {len(stat_arr)} ROIs')
        print(f'  F/Fneu/spks/iscell saved')
    return final_plane0


def main():
    run(
        TIFF_FOLDER, SAVE_FOLDER, PATH_TO_OPS,
        sparsery_ops=DEFAULT_SPARSERY_OPS,
        cellpose_cfg=DEFAULT_CELLPOSE_CFG,
        hard_cap=HARD_CAP,
        max_overlap=CELLPOSE_MERGE_MAX_OVERLAP,
        aav_info_csv='human_SLE_2p_meta.csv',
        tau_vals=DEFAULT_TAU_VALS,
        verbose=True,
    )


if __name__ == '__main__':
    main()
