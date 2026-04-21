"""
Adaptive Suite2p detection pipeline.

Designed as a drop-in replacement for main.py that iteratively runs Suite2p
with progressively more aggressive detection parameters, auditing the mean
image after each pass to decide whether cells are still being missed.

User only needs to specify:
    - tiff_folder: where the raw TIFF stack lives
    - save_folder: where Suite2p output should be written
    - path_to_ops (optional): starting ops.npy to build from; if None, uses
      suite2p.default_ops() with sensible defaults for human 2p GCaMP data

The goal is to wrap this behind a minimal app where the user never touches
the Suite2p GUI or config files directly.
"""

from __future__ import annotations

import os
import shutil
import numpy as np
import suite2p
import utils
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AdaptiveConfig:
    """All parameters controlling the adaptive loop in one place."""

    # ---- Required paths ----
    tiff_folder: str = ""                         # folder containing TIFFs
    save_folder: str = ""                         # where per-pass output lives
    path_to_ops: Optional[str] = None             # optional starting ops.npy

    # ---- Sample metadata (used for tau lookup) ----
    aav_info_csv: str = "human_SLE_2p_meta.csv"
    tau_vals: dict = field(default_factory=lambda: {
        "6f": 0.7, "6m": 1.0, "6s": 1.3, "8m": 0.137,
    })

    # ---- Adaptive schedule (under-detection regime) ----
    # threshold_scaling and max_iterations values tried in order when the
    # initial pass looks reasonable but is missing cells
    threshold_schedule: tuple = (0.88, 0.60, 0.40)
    max_iter_schedule: tuple = (100, 200, 300)

    # ---- Fallback schedule (over-detection regime) ----
    # Used when the initial pass locks onto the wrong spatial scale and
    # produces a flood of tiny non-cell ROIs. Each entry is (spatial_scale,
    # threshold_scaling, max_overlap). Spatial scale 2 ~= 12px, 3 ~= 24px.
    spatial_scale_schedule: tuple = (
        (3, 1.2, 0.5),   # large cells, tight overlap
        (2, 1.0, 0.5),   # medium cells
        (3, 0.8, 0.5),   # large cells, more sensitive
    )

    # ---- Over-detection diagnosis (triggers fallback schedule) ----
    # If the first pass detects more ROIs than this, assume wrong spatial scale
    over_detection_roi_count: int = 1500
    # Or if the median ROI pixel area is below this, same conclusion
    over_detection_median_area_px: float = 20.0
    # Also require that accepted/total ratio from suite2p classifier is low,
    # since a huge number of real cells is possible in principle. If iscell
    # keeps fewer than this fraction, that confirms the ROIs are mostly junk.
    over_detection_iscell_ratio: float = 0.15

    # ---- Blob detection on residual image ----
    soma_diameter_px: float = 12.0
    soma_scale_tolerance: float = 0.5
    blob_min_contrast: float = 0.10
    blob_min_area_px: int = 25
    blob_max_area_px: int = 400

    # ---- Stopping criteria ----
    min_residual_blobs: int = 8          # stop if fewer missed cells than this
    min_new_rois_per_pass: int = 3       # stop if a pass adds fewer than this
    iou_dedup_threshold: float = 0.3     # ROI pairs above this IoU are merged

    # ---- Misc ----
    verbose: bool = True


# ============================================================================
# Ops loading / preparation
# ============================================================================

def load_base_ops(config: AdaptiveConfig) -> dict:
    """
    Load starting ops either from a user-supplied .npy or from Suite2p defaults.
    Apply sample-specific adjustments (tau via GCaMP lookup, batch size, etc.)
    following the same convention as main.py.
    """
    if config.path_to_ops is not None and os.path.exists(config.path_to_ops):
        if config.verbose:
            print(f"Loading base ops from {config.path_to_ops}")
        ops = np.load(config.path_to_ops, allow_pickle=True).item()
    else:
        if config.verbose:
            print("No base ops provided, starting from suite2p defaults")
        ops = suite2p.default_ops()
        # Sensible starting values for human slice 2p GCaMP data
        ops.update({
            'fs': 15.07,
            'nchannels': 1,
            'nplanes': 1,
            'sparse_mode': True,
            'spatial_scale': 0,   # auto-select initially
            'high_pass': 100.0,
            'nbinned': 15000,
            'smooth_sigma': 1.3,
            'preclassify': 0.0,
            'allow_overlap': False,
        })

    # Apply sample-specific tau if metadata is available
    file_name = os.path.basename(os.path.normpath(config.tiff_folder))
    if os.path.exists(config.aav_info_csv):
        try:
            tau = utils.file_name_to_aav_to_dictionary_lookup(
                file_name, config.aav_info_csv, config.tau_vals
            )
            ops['tau'] = tau
            if config.verbose:
                print(f"Set tau={tau} based on AAV lookup for {file_name}")
        except Exception as e:
            if config.verbose:
                print(f"tau lookup failed ({e}); keeping existing tau={ops.get('tau')}")

    # Dynamic batch size (same logic as main.py)
    ops['batch_size'] = utils.change_batch_according_to_free_ram()

    return ops


# ============================================================================
# ROI mask, residual image, blob detection
# ============================================================================

def build_roi_pixel_mask(stat, Ly: int, Lx: int) -> np.ndarray:
    """Binary mask of all pixels belonging to any detected ROI."""
    mask = np.zeros((Ly, Lx), dtype=bool)
    for s in stat:
        ypix = np.asarray(s['ypix'])
        xpix = np.asarray(s['xpix'])
        valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        mask[ypix[valid], xpix[valid]] = True
    return mask


def compute_residual_image(ops: dict, roi_mask: np.ndarray,
                           dilate_px: int = 2, prefer: str = 'meanImg'):
    """
    Residual brightness after subtracting detected ROIs from the mean image.
    Dilates the ROI mask slightly so neuropil rings around detected cells
    don't register as missed somata.

    prefer='meanImg' (default) uses the raw mean, which preserves dynamic
    range. 'meanImgE' uses the enhanced mean but is often clipped at 1.0.
    """
    from scipy.ndimage import binary_dilation

    if prefer == 'meanImgE' and ops.get('meanImgE') is not None:
        img = ops['meanImgE'].astype(np.float32)
    elif 'meanImg' in ops:
        img = ops['meanImg'].astype(np.float32)
    else:
        img = ops['meanImgE'].astype(np.float32)

    dilated = binary_dilation(roi_mask, iterations=dilate_px)
    residual = img.copy()

    if dilated.any() and (~dilated).any():
        bg = float(np.median(img[~dilated]))
    else:
        bg = float(np.median(img))
    residual[dilated] = bg
    return residual, bg


def detect_residual_blobs(residual: np.ndarray, config: AdaptiveConfig):
    """
    Multi-scale Laplacian-of-Gaussian blob detection tuned to soma size.

    The residual image from 2p recordings has a heavy right tail (a handful of
    very bright pixels) and a diffuse neuropil background. We:
      1. Subtract a local median-filter background at roughly soma scale
      2. Robust-rescale to [0,1] using 99.5th percentile
      3. Require each blob to have 1.5x center/surround contrast
    """
    from skimage.feature import blob_log
    from scipy.ndimage import median_filter

    r = config.soma_diameter_px / 2.0
    tol = config.soma_scale_tolerance
    min_sigma = (r * (1.0 - tol)) / np.sqrt(2.0)
    max_sigma = (r * (1.0 + tol)) / np.sqrt(2.0)

    img = residual.astype(np.float32)

    bg_radius = max(3, int(round(config.soma_diameter_px)))
    bg_local = median_filter(img, size=bg_radius)
    hp = img - bg_local
    hp[hp < 0] = 0

    hi = float(np.quantile(hp, 0.995))
    if hi <= 0:
        return []
    norm = np.clip(hp / hi, 0.0, 1.0)

    blobs = blob_log(
        norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=6,
        threshold=config.blob_min_contrast,
        overlap=0.5,
    )
    if blobs.size == 0:
        return []

    out = []
    for y, x, sigma in blobs:
        radius = sigma * np.sqrt(2.0)
        area = np.pi * radius ** 2
        if area < config.blob_min_area_px or area > config.blob_max_area_px:
            continue
        yi, xi = int(round(y)), int(round(x))
        yi = max(0, min(img.shape[0] - 1, yi))
        xi = max(0, min(img.shape[1] - 1, xi))

        rr = int(round(radius))
        center_val = float(hp[yi, xi])
        y0, y1 = max(0, yi - 2*rr), min(img.shape[0], yi + 2*rr + 1)
        x0, x1 = max(0, xi - 2*rr), min(img.shape[1], xi + 2*rr + 1)
        surround = float(np.median(hp[y0:y1, x0:x1]))
        if center_val < surround * 1.5 + 1e-6:
            continue

        out.append((yi, xi, radius, center_val))
    return out


# ============================================================================
# ROI merging across passes
# ============================================================================

def _stat_to_pixel_set(s):
    ypix = np.asarray(s['ypix']).tolist()
    xpix = np.asarray(s['xpix']).tolist()
    return set(zip(ypix, xpix))


def _iou(pixset_a, pixset_b):
    if not pixset_a or not pixset_b:
        return 0.0
    inter = len(pixset_a & pixset_b)
    if inter == 0:
        return 0.0
    return inter / (len(pixset_a) + len(pixset_b) - inter)


def merge_roi_lists(stat_old, stat_new, iou_threshold=0.3):
    """Combine two ROI lists; dedupe by spatial IoU."""
    old_pixsets = [_stat_to_pixel_set(s) for s in stat_old]
    merged = list(stat_old)
    n_added = 0
    for s_new in stat_new:
        ps_new = _stat_to_pixel_set(s_new)
        if any(_iou(ps_new, ps_old) >= iou_threshold for ps_old in old_pixsets):
            continue
        merged.append(s_new)
        old_pixsets.append(ps_new)
        n_added += 1
    return merged, n_added


# ============================================================================
# Single Suite2p pass wrapper
# ============================================================================

def run_one_pass(tiff_folder: str, save_dir: Path, ops: dict,
                 verbose: bool = True):
    """
    Run Suite2p once with the given ops, saving to save_dir. Returns the
    loaded stat list, ops dict, and path to the plane0 directory.
    """
    ops = dict(ops)
    ops['save_path0'] = str(save_dir)
    ops['save_folder'] = 'suite2p'

    db = {'data_path': [tiff_folder]}

    if verbose:
        print(f"  > running suite2p: threshold_scaling={ops['threshold_scaling']}  "
              f"max_iterations={ops['max_iterations']}")

    output_ops = suite2p.run_s2p(ops=ops, db=db)

    plane0 = Path(save_dir) / 'suite2p' / 'plane0'
    stat = list(np.load(plane0 / 'stat.npy', allow_pickle=True))
    ops_loaded = np.load(plane0 / 'ops.npy', allow_pickle=True).item()
    return stat, ops_loaded, plane0


# ============================================================================
# Regime diagnosis
# ============================================================================

def diagnose_over_detection(stat, plane0: Path, config: AdaptiveConfig):
    """
    Decide whether a pass has locked onto the wrong spatial scale and is
    detecting a flood of tiny non-cell ROIs.

    Triggers when either:
      - Raw ROI count exceeds the configured ceiling, AND the Suite2p
        classifier kept only a small fraction (iscell ratio is low), OR
      - Median ROI pixel area is very small (below configured floor)

    Returns (is_over_detection: bool, reason: str, stats: dict).
    """
    n_total = len(stat)

    areas = np.array([np.asarray(s['xpix']).size for s in stat], dtype=float)
    median_area = float(np.median(areas)) if areas.size else 0.0

    iscell_path = plane0 / 'iscell.npy'
    if iscell_path.exists():
        iscell = np.load(iscell_path, allow_pickle=True)
        if iscell.ndim == 2:
            iscell = iscell[:, 0]
        n_accepted = int((iscell > 0).sum())
        accept_ratio = n_accepted / max(1, n_total)
    else:
        n_accepted = -1
        accept_ratio = -1.0

    diag = {
        'n_total': n_total,
        'n_accepted': n_accepted,
        'accept_ratio': accept_ratio,
        'median_area_px': median_area,
    }

    # Condition 1: huge total count AND classifier rejected most of them
    bad_count = (n_total >= config.over_detection_roi_count and
                 0 <= accept_ratio < config.over_detection_iscell_ratio)
    # Condition 2: median ROI area implausibly small (sub-soma)
    bad_size = (median_area > 0 and median_area < config.over_detection_median_area_px)

    if bad_count and bad_size:
        reason = (f"ROI count {n_total} with only {accept_ratio:.1%} classifier "
                  f"acceptance AND median area {median_area:.0f}px below "
                  f"{config.over_detection_median_area_px}px floor")
        return True, reason, diag
    if bad_count:
        reason = (f"ROI count {n_total} with only {accept_ratio:.1%} classifier "
                  f"acceptance (threshold: >={config.over_detection_roi_count} "
                  f"AND <{config.over_detection_iscell_ratio:.0%})")
        return True, reason, diag
    if bad_size:
        reason = (f"median ROI area {median_area:.0f}px below "
                  f"{config.over_detection_median_area_px}px floor")
        return True, reason, diag

    return False, "within normal ranges", diag


# ============================================================================
# Adaptive loop
# ============================================================================

def _run_under_detection_schedule(config, base_ops, save_root,
                                  initial_stat, initial_ops, initial_plane0,
                                  initial_thr, initial_max_iter):
    """
    Under-detection regime: start from the already-completed pass 0 and keep
    lowering threshold_scaling until residual blobs fall below tolerance.
    """
    passes = [{
        'index': 0,
        'regime': 'under_detection',
        'threshold_scaling': initial_thr,
        'max_iterations': initial_max_iter,
        'n_detected_this_pass': len(initial_stat),
        'n_added_after_merge': len(initial_stat),
        'n_merged_total': len(initial_stat),
        'plane0': str(initial_plane0),
    }]
    merged_stat = list(initial_stat)
    final_ops = initial_ops
    final_plane0 = initial_plane0

    # Initial residual audit on pass 0
    Ly, Lx = initial_ops['Ly'], initial_ops['Lx']
    roi_mask = build_roi_pixel_mask(merged_stat, Ly, Lx)
    residual, _ = compute_residual_image(initial_ops, roi_mask)
    blobs = detect_residual_blobs(residual, config)
    passes[0]['n_residual_blobs'] = len(blobs)

    if config.verbose:
        print(f"  > residual blobs after pass 0: {len(blobs)}")

    if len(blobs) <= config.min_residual_blobs:
        if config.verbose:
            print("  > stopping: pass 0 already captures most cells")
        return merged_stat, passes, final_ops, final_plane0

    # Subsequent passes from the threshold schedule, skipping the first entry
    # since that was already used for pass 0
    sched = list(zip(config.threshold_schedule, config.max_iter_schedule))[1:]
    for j, (thr_scale, max_iter) in enumerate(sched, start=1):
        pass_dir = save_root / f"pass{j:02d}_thr{thr_scale:.2f}"
        pass_dir.mkdir(parents=True, exist_ok=True)

        if config.verbose:
            print(f"\n[pass {j}] under_detection regime  "
                  f"threshold_scaling={thr_scale}  max_iterations={max_iter}")

        ops = dict(base_ops)
        ops['threshold_scaling'] = thr_scale
        ops['max_iterations'] = max_iter

        stat_pass, ops_out, plane0 = run_one_pass(
            config.tiff_folder, pass_dir, ops, verbose=config.verbose
        )
        final_ops = ops_out
        final_plane0 = plane0

        merged_stat, n_added = merge_roi_lists(
            merged_stat, stat_pass, iou_threshold=config.iou_dedup_threshold
        )

        Ly, Lx = ops_out['Ly'], ops_out['Lx']
        roi_mask = build_roi_pixel_mask(merged_stat, Ly, Lx)
        residual, _ = compute_residual_image(ops_out, roi_mask)
        blobs = detect_residual_blobs(residual, config)

        passes.append({
            'index': j,
            'regime': 'under_detection',
            'threshold_scaling': thr_scale,
            'max_iterations': max_iter,
            'n_detected_this_pass': len(stat_pass),
            'n_added_after_merge': n_added,
            'n_merged_total': len(merged_stat),
            'n_residual_blobs': len(blobs),
            'plane0': str(plane0),
        })

        if config.verbose:
            print(f"  > added: {n_added}  total: {len(merged_stat)}  "
                  f"residual_blobs: {len(blobs)}")

        if len(blobs) <= config.min_residual_blobs:
            if config.verbose:
                print("  > stopping: residual blob count below tolerance")
            break
        if n_added < config.min_new_rois_per_pass:
            if config.verbose:
                print("  > stopping: pass added few new ROIs")
            break

    return merged_stat, passes, final_ops, final_plane0


def _run_over_detection_schedule(config, base_ops, save_root):
    """
    Over-detection regime: pass 0 is discarded, and we rerun using explicit
    spatial_scale and tightened max_overlap. Each pass in the schedule is
    kept in isolation; the best pass (fewest ROIs but most accepted by the
    classifier) wins.
    """
    passes = []
    candidates = []  # list of (score, merged_stat, ops, plane0, pass_dict)

    for j, (sp_scale, thr_scale, max_overlap) in enumerate(
        config.spatial_scale_schedule
    ):
        pass_dir = save_root / f"pass_sc{j:02d}_scale{sp_scale}_thr{thr_scale:.2f}"
        pass_dir.mkdir(parents=True, exist_ok=True)

        if config.verbose:
            print(f"\n[spatial pass {j}] over_detection regime  "
                  f"spatial_scale={sp_scale}  threshold_scaling={thr_scale}  "
                  f"max_overlap={max_overlap}")

        ops = dict(base_ops)
        ops['spatial_scale'] = sp_scale
        ops['threshold_scaling'] = thr_scale
        ops['max_overlap'] = max_overlap
        # Keep max_iterations at a moderate value; over-detection doesn't need
        # deep iteration, it needs the right scale
        ops['max_iterations'] = 100

        stat_pass, ops_out, plane0 = run_one_pass(
            config.tiff_folder, pass_dir, ops, verbose=config.verbose
        )

        # Evaluate this pass
        iscell_path = plane0 / 'iscell.npy'
        if iscell_path.exists():
            iscell = np.load(iscell_path, allow_pickle=True)
            if iscell.ndim == 2:
                iscell = iscell[:, 0]
            n_accepted = int((iscell > 0).sum())
        else:
            n_accepted = len(stat_pass)

        areas = np.array([np.asarray(s['xpix']).size for s in stat_pass],
                         dtype=float)
        median_area = float(np.median(areas)) if areas.size else 0.0

        # Score: accepted count weighted by plausibility of soma-size ROIs.
        # We want high n_accepted AND median_area above the floor.
        size_penalty = 1.0 if median_area >= config.over_detection_median_area_px else 0.25
        score = n_accepted * size_penalty

        pass_info = {
            'index': j,
            'regime': 'over_detection',
            'spatial_scale': sp_scale,
            'threshold_scaling': thr_scale,
            'max_overlap': max_overlap,
            'n_detected_this_pass': len(stat_pass),
            'n_accepted': n_accepted,
            'median_area_px': median_area,
            'score': score,
            'plane0': str(plane0),
        }
        passes.append(pass_info)
        candidates.append((score, list(stat_pass), ops_out, plane0, pass_info))

        if config.verbose:
            print(f"  > detected: {len(stat_pass)}  accepted: {n_accepted}  "
                  f"median_area: {median_area:.0f}px  score: {score:.1f}")

    # Pick the best pass by score
    candidates.sort(key=lambda t: t[0], reverse=True)
    best_score, best_stat, best_ops, best_plane0, best_info = candidates[0]

    if config.verbose:
        print(f"\n  > best over-detection pass: spatial_scale="
              f"{best_info['spatial_scale']} thr={best_info['threshold_scaling']} "
              f"(accepted={best_info['n_accepted']}, "
              f"median_area={best_info['median_area_px']:.0f}px)")

    # Mark winner in passes list
    for p in passes:
        p['selected'] = (p is best_info)

    return best_stat, passes, best_ops, best_plane0


def adaptive_detect(config: AdaptiveConfig):
    """
    Main entry point. Runs the adaptive detection pipeline end-to-end.

    Flow:
      1. Run pass 0 with the first entry of threshold_schedule (conservative).
      2. Diagnose: did Suite2p lock onto the wrong spatial scale?
         - If NO: continue under-detection schedule, progressively lowering
           threshold_scaling, merging ROIs across passes, auditing residual.
         - If YES: discard pass 0, restart with spatial_scale_schedule, and
           pick the single best pass by (accepted ROIs, plausible median area).

    Returns a dict with:
        merged_stat      list of deduplicated ROI stat dicts
        passes           per-pass diagnostics (includes 'regime' key)
        regime           'under_detection' or 'over_detection'
        diagnosis        dict from diagnose_over_detection()
        final_ops        ops from the selected Suite2p invocation
        final_plane0     Path to the selected pass's plane0 output
        save_folder      root save folder
    """
    if not config.tiff_folder:
        raise ValueError("config.tiff_folder must be set")
    if not config.save_folder:
        raise ValueError("config.save_folder must be set")

    save_root = Path(config.save_folder)
    save_root.mkdir(parents=True, exist_ok=True)

    base_ops = load_base_ops(config)

    # -------------------------- pass 0 (diagnostic) ------------------------
    initial_thr = config.threshold_schedule[0]
    initial_max_iter = config.max_iter_schedule[0]

    pass0_dir = save_root / f"pass00_thr{initial_thr:.2f}"
    pass0_dir.mkdir(parents=True, exist_ok=True)

    if config.verbose:
        print(f"\n[pass 0 / diagnostic] threshold_scaling={initial_thr}  "
              f"max_iterations={initial_max_iter}")

    ops = dict(base_ops)
    ops['threshold_scaling'] = initial_thr
    ops['max_iterations'] = initial_max_iter

    stat0, ops0, plane0_dir = run_one_pass(
        config.tiff_folder, pass0_dir, ops, verbose=config.verbose
    )

    is_over, reason, diag = diagnose_over_detection(stat0, plane0_dir, config)

    if config.verbose:
        print(f"\n  diagnosis: n_total={diag['n_total']}, "
              f"n_accepted={diag['n_accepted']}, "
              f"accept_ratio={diag['accept_ratio']:.1%}, "
              f"median_area={diag['median_area_px']:.0f}px")
        print(f"  regime: {'OVER_DETECTION' if is_over else 'UNDER_DETECTION'}  "
              f"({reason})")

    # -------------------------- branch on regime ---------------------------
    if is_over:
        regime = 'over_detection'
        merged_stat, passes, final_ops, final_plane0 = _run_over_detection_schedule(
            config, base_ops, save_root
        )
        # Record the discarded pass 0 at the front of the passes list for audit
        passes.insert(0, {
            'index': -1,
            'regime': 'diagnostic_pass0_discarded',
            'threshold_scaling': initial_thr,
            'max_iterations': initial_max_iter,
            'n_detected_this_pass': len(stat0),
            'diagnosis': diag,
            'reason_for_discard': reason,
            'plane0': str(plane0_dir),
        })
    else:
        regime = 'under_detection'
        merged_stat, passes, final_ops, final_plane0 = _run_under_detection_schedule(
            config, base_ops, save_root,
            initial_stat=stat0, initial_ops=ops0, initial_plane0=plane0_dir,
            initial_thr=initial_thr, initial_max_iter=initial_max_iter,
        )

    # -------------------------- persist merged output ----------------------
    merged_out = save_root / 'merged'
    merged_out.mkdir(exist_ok=True)
    np.save(merged_out / 'stat_merged.npy',
            np.array(merged_stat, dtype=object), allow_pickle=True)
    np.save(merged_out / 'ops_final.npy', final_ops, allow_pickle=True)

    import json
    with open(merged_out / 'pass_summary.json', 'w') as f:
        json.dump({
            'regime': regime,
            'diagnosis': diag,
            'passes': passes,
        }, f, indent=2, default=str)

    if config.verbose:
        print(f"\nDone. Regime: {regime}. Merged ROI count: {len(merged_stat)}")
        print(f"Final outputs in: {merged_out}")

    return {
        'merged_stat': merged_stat,
        'passes': passes,
        'regime': regime,
        'diagnosis': diag,
        'final_ops': final_ops,
        'final_plane0': final_plane0,
        'save_folder': str(save_root),
    }


# ============================================================================
# Standalone audit (no Suite2p invocation)
# ============================================================================

def audit_existing_detection(plane0_dir: str, config: Optional[AdaptiveConfig] = None):
    """Residual-blob audit on an already-completed Suite2p plane0 folder."""
    if config is None:
        config = AdaptiveConfig()

    plane0 = Path(plane0_dir)
    stat = list(np.load(plane0 / 'stat.npy', allow_pickle=True))
    ops = np.load(plane0 / 'ops.npy', allow_pickle=True).item()

    Ly, Lx = ops['Ly'], ops['Lx']
    roi_mask = build_roi_pixel_mask(stat, Ly, Lx)
    residual, _ = compute_residual_image(ops, roi_mask)
    blobs = detect_residual_blobs(residual, config)

    return {
        'n_rois': len(stat),
        'n_residual_blobs': len(blobs),
        'residual_blobs': blobs,
        'residual_image': residual,
        'roi_mask': roi_mask,
    }


def visualize_audit(plane0_dir: str, config: Optional[AdaptiveConfig] = None,
                    outpath: Optional[str] = None):
    """Three-panel figure: mean image, residual, and missed-cell candidates."""
    import matplotlib.pyplot as plt

    result = audit_existing_detection(plane0_dir, config)
    ops = np.load(Path(plane0_dir) / 'ops.npy', allow_pickle=True).item()
    img = ops['meanImg']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    vmax = float(np.quantile(img, 0.995))
    axes[0].imshow(img, cmap='gray', vmax=vmax)
    axes[0].set_title(f"Mean image (n_ROIs = {result['n_rois']})")
    axes[0].axis('off')

    rvmax = float(np.quantile(result['residual_image'], 0.995))
    axes[1].imshow(result['residual_image'], cmap='magma', vmax=rvmax)
    axes[1].set_title("Residual (ROI pixels masked out)")
    axes[1].axis('off')

    axes[2].imshow(img, cmap='gray', vmax=vmax)
    for (y, x, r, _) in result['residual_blobs']:
        c = plt.Circle((x, y), r, color='cyan', fill=False, linewidth=1.5)
        axes[2].add_patch(c)
    axes[2].set_title(f"Missed-cell candidates (n = {result['n_residual_blobs']})")
    axes[2].axis('off')

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
    else:
        plt.show()
    return result


# ============================================================================
# Script entry point
# ============================================================================

if __name__ == '__main__':
    config = AdaptiveConfig(
        tiff_folder=r'D:\2024-11-20_00003',
        save_folder=r'D:\adaptive_runs\2024-11-20_00003',
        path_to_ops=r"E:\suite2p_2p_ops_240621.npy",  # set to None to use defaults

        threshold_schedule=(0.88, 0.60, 0.40),
        max_iter_schedule=(100, 200, 300),

        soma_diameter_px=12.0,
        min_residual_blobs=8,
    )

    result = adaptive_detect(config)
    print(f"\nFinal ROI count: {len(result['merged_stat'])}")
    for p in result['passes']:
        print(p)