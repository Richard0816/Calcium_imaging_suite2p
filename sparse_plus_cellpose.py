"""Sparsery + Cellpose detection (no adaptive threshold search).

Minimal pipeline — run sparsery once at a fixed threshold, run cellpose
once, merge non-overlapping cellpose ROIs into the sparsery set, then
extract fluorescence + deconvolve + classify + save.

Compared to ``final_adaptive_detection.py``, this script skips the
spatial-scale estimation and threshold binary search. Ops are whatever
you set at the top of this file.

Usage:
    python sparse_plus_cellpose.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKTREE))

import suite2p   # noqa: F401
from suite2p.detection.stats import roi_stats
from suite2p.extraction import extract, dcnv
from suite2p import classification

from adaptive_detection import (
    AdaptiveConfig,
    load_base_ops,
    _get_or_create_shared_registration,
    _link_or_copy,
    run_one_pass,
    build_roi_pixel_mask,
    visualize_audit,
)

from brute_force_ops import run_cellpose_pass


# ============================================================================
# CONFIGURATION
# ============================================================================

TIFF_FOLDER = r'D:\2024-11-20_00003'
SAVE_FOLDER = r'D:\sparse_plus_cellpose\2024-11-20_00003'
PATH_TO_OPS = r'suite2p_2p_ops_240621.npy'

# Fixed sparsery ops — no binary search, just one pass with these values.
SPARSERY_OPS = {
    'high_pass':          100,
    'preclassify':        0.0,
    'smooth_sigma':       1.0,
    'sparse_mode':        True,
    'spatial_scale':      0,      # 0 = auto
    'threshold_scaling':  0.85,
    'max_iterations':     1500,
}

# Cellpose config (pinned to the brute-force winner).
CELLPOSE_CFG = {
    'cellpose_model_type':         'cyto2',
    'cellpose_diameter':           0,      # 0 = auto
    'cellpose_flow_threshold':     0.8,
    'cellpose_cellprob_threshold': -1.0,
    'cellpose_channel_input':      'meanImg',
}

# Cap sparsery mid-detection if it floods past this many ROIs. None = no cap.
HARD_CAP = 60000

# A cellpose ROI is dropped if this fraction of its pixels is already
# covered by the union of sparsery ROIs.
CELLPOSE_MERGE_MAX_OVERLAP = 0.3


# ============================================================================
# Helpers
# ============================================================================

def filter_non_overlapping_cellpose(sparsery_stat, cellpose_stat,
                                    Ly: int, Lx: int,
                                    max_overlap: float,
                                    verbose: bool = True):
    """Keep cellpose ROIs with < ``max_overlap`` fraction covered by sparsery."""
    sparsery_mask = build_roi_pixel_mask(sparsery_stat, Ly, Lx)
    kept = []
    dropped = 0
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
        print(f"    cellpose merge: kept {len(kept)} / {len(cellpose_stat)} "
              f"ROIs (dropped {dropped} overlapping sparsery)")
    return kept


def merge_and_extract(sparsery_stat, cellpose_stat,
                      shared_plane0: Path, base_ops: dict,
                      final_dir: Path, max_overlap: float,
                      verbose: bool = True):
    """Combine sparsery + non-overlapping cellpose ROIs, then extract
    fluorescence → dcnv → classify → save into ``final_dir/suite2p/plane0/``.
    """
    final_plane0 = final_dir / 'suite2p' / 'plane0'
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

    n_sp = len(sparsery_stat)
    n_cp = len(cellpose_kept)
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
            [len(s['ypix']) for s in combined_raw]
        ) / np.pi)),
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
# Main
# ============================================================================

def main():
    save_root = Path(SAVE_FOLDER)
    save_root.mkdir(parents=True, exist_ok=True)

    cfg = AdaptiveConfig(
        tiff_folder=TIFF_FOLDER,
        save_folder=SAVE_FOLDER,
        path_to_ops=PATH_TO_OPS,
        auto_estimate_spatial_scale=False,
        generate_audit_pngs=False,
        augment_with_residual_blobs=False,
        use_cache=False,
        verbose=True,
    )

    print(f'[s+cp] loading base ops from {PATH_TO_OPS}')
    base_ops = load_base_ops(cfg)

    print(f'[s+cp] shared registration')
    t0 = time.time()
    shared_plane0 = _get_or_create_shared_registration(cfg, base_ops)
    print(f'[s+cp] shared reg ready at {shared_plane0}  '
          f'({time.time() - t0:.1f}s)')

    # ---- sparsery (one shot, fixed ops) ----
    print(f'[s+cp] sparsery pass '
          f'(threshold_scaling={SPARSERY_OPS["threshold_scaling"]})')
    sparsery_dir = save_root / 'sparsery_pass'
    ops = dict(base_ops)
    for k, v in SPARSERY_OPS.items():
        ops[k] = v
    ops['roidetect'] = True
    t_sp = time.time()
    sp_stat, sp_ops, sp_plane0 = run_one_pass(
        TIFF_FOLDER, sparsery_dir, ops,
        verbose=True, use_cache=False,
        shared_plane0=shared_plane0,
        hard_cap=HARD_CAP or 0,
    )
    print(f'[s+cp] sparsery: {len(sp_stat)} ROIs '
          f'({time.time() - t_sp:.1f}s)')

    # ---- cellpose ----
    print(f'[s+cp] cellpose pass')
    cp_dir = save_root / 'cellpose_pass'
    t_cp = time.time()
    cp_stat, cp_ops, cp_plane0 = run_cellpose_pass(
        cp_dir, shared_plane0, CELLPOSE_CFG, verbose=True,
    )
    print(f'[s+cp] cellpose: {len(cp_stat)} ROIs '
          f'({time.time() - t_cp:.1f}s)')

    # ---- merge + extract ----
    print(f'[s+cp] merging + extracting')
    final_dir = save_root / 'final'
    stat_arr, final_ops, final_plane0 = merge_and_extract(
        sp_stat, cp_stat, shared_plane0, base_ops, final_dir,
        max_overlap=CELLPOSE_MERGE_MAX_OVERLAP, verbose=True,
    )

    # ---- audit.png ----
    print(f'[s+cp] writing audit.png')
    try:
        visualize_audit(str(final_plane0), config=cfg,
                        outpath=str(final_plane0 / 'audit.png'),
                        title_prefix='sparse+cellpose')
    except Exception as e:
        print(f'[s+cp] audit failed: {e}')

    print(f'\n[s+cp] DONE. Outputs:')
    print(f'  plane0:     {final_plane0}')
    print(f'  stat.npy:   {len(stat_arr)} ROIs')
    print(f'  F/Fneu/spks/iscell all saved')
    print(f'  audit.png:  {final_plane0 / "audit.png"}')


if __name__ == '__main__':
    main()
