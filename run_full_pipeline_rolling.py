"""Addon to run_full_pipeline: re-runs every step past suite2p using a
*rolling* percentile baseline for ΔF/F (matching the algorithm in Fig3.py /
analyze_output.py), and writes all outputs under <plane0>/rolling baseline/.

Suite2p (shift + detection) must already be done. This script iterates
existing recording dirs under --detection-root and, for each plane0:

  1. Build a shadow tree at <plane0>/rolling baseline/suite2p/plane0/
     containing hardlinks to F.npy, Fneu.npy, ops.npy, stat.npy, iscell.npy,
     spks.npy. Downstream stages think it's an ordinary recording.
  2. Compute ΔF/F with rolling percentile baseline (win_sec=45, perc=10)
     over ALL ROIs (cellfilter does the cell selection afterwards), plus
     low-pass and SG derivative — same memmap layout as the GPU path,
     prefix `r0p7_`.
  3. Run cellfilter -> r0p7_filtered_* memmaps -> image_all -> hierarchical
     clustering -> spatial heatmaps with propagation vectors. All outputs
     land inside the shadow tree.

Usage
-----
    # all recordings under default detection root
    python run_full_pipeline_rolling.py

    # specific recordings
    python run_full_pipeline_rolling.py --rec 2024-06-04_00009,2024-06-05_00007

    # custom rolling-baseline params
    python run_full_pipeline_rolling.py --win-sec 60 --perc 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

WORKTREE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKTREE))

# Reuse the existing post-suite2p stages verbatim. They take a plane0 / final
# folder and read/write with prefix `r0p7_`, so pointing them at the shadow
# tree gives us rolling-baseline outputs in the right place.
from run_full_pipeline import (
    DEFAULT_DETECTION_ROOT,
    DEFAULT_FPS,
    DEFAULT_PREFIX,
    run_cellfilter,
    make_filtered_memmaps,
    run_image_all,
    run_cluster,
    run_spatial_heatmaps,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SHADOW_DIRNAME = "rolling baseline"
DEFAULT_WIN_SEC = 45.0   # Fig3.rolling_baseline_dff default
DEFAULT_PERC    = 10     # Fig3.rolling_baseline_dff default
DEFAULT_CUTOFF  = 1.0    # match analyze_output_gpu hardcoded value
DEFAULT_R       = 0.7
SUITE2P_INPUTS  = ("F.npy", "Fneu.npy", "ops.npy", "stat.npy",
                   "iscell.npy", "spks.npy")


# ---------------------------------------------------------------------------
# Shadow tree
# ---------------------------------------------------------------------------
def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink src -> dst on the same volume; fall back to copy."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def build_shadow_plane0(plane0: Path) -> tuple[Path, Path]:
    """Create <plane0>/rolling baseline/suite2p/plane0/ with hardlinks to the
    suite2p inputs needed by downstream stages.

    Returns (shadow_final_folder, shadow_plane0). shadow_final_folder is the
    directory whose `suite2p/plane0/` is shadow_plane0 — pass it where the
    original pipeline passes `final_folder`."""
    shadow_final = plane0 / SHADOW_DIRNAME
    shadow_plane0 = shadow_final / "suite2p" / "plane0"
    shadow_plane0.mkdir(parents=True, exist_ok=True)

    for name in SUITE2P_INPUTS:
        src = plane0 / name
        if not src.exists():
            # F.npy / Fneu.npy are required; the rest are optional.
            if name in ("F.npy", "Fneu.npy"):
                raise FileNotFoundError(f"required suite2p input missing: {src}")
            continue
        _link_or_copy(src, shadow_plane0 / name)

    return shadow_final, shadow_plane0


# ---------------------------------------------------------------------------
# Rolling-baseline analyze (CPU; reuses analyze_output.process_suite2p_traces)
# ---------------------------------------------------------------------------
def run_analyze_rolling(shadow_plane0: Path, fps: float, *,
                        prefix: str = DEFAULT_PREFIX,
                        win_sec: float = DEFAULT_WIN_SEC,
                        perc: int = DEFAULT_PERC,
                        cutoff_hz: float = DEFAULT_CUTOFF,
                        r: float = DEFAULT_R,
                        sg_win_ms: float = 333,
                        sg_poly: int = 2) -> None:
    """Compute ΔF/F (rolling percentile baseline) + low-pass + derivative for
    ALL ROIs and write memmaps into shadow_plane0 with the given prefix.

    Skipped if all three memmaps already exist with the expected size."""
    import analyze_output
    import utils

    F = np.load(shadow_plane0 / "F.npy", allow_pickle=True)
    Fneu = np.load(shadow_plane0 / "Fneu.npy", allow_pickle=True)

    # Suite2p convention is (N, T); analyze_output handles orientation.
    N, T = (F.shape if F.shape[0] < F.shape[1] else (F.shape[1], F.shape[0]))
    expected_bytes = T * N * 4
    out_paths = {k: shadow_plane0 / f"{prefix}{k}.memmap.float32"
                 for k in ("dff", "dff_lowpass", "dff_dt")}
    if all(p.exists() and p.stat().st_size == expected_bytes
           for p in out_paths.values()):
        print(f"[rolling/analyze] {shadow_plane0}: memmaps already present, skipping")
        return

    batch_size = max(1, utils.change_batch_according_to_free_ram() * 20)
    print(f"[rolling/analyze] {shadow_plane0}: T={T} N={N}  "
          f"win_sec={win_sec} perc={perc} cutoff_hz={cutoff_hz}")

    dff_path, low_path, dt_path = analyze_output.process_suite2p_traces(
        F, Fneu, fps,
        r=r,
        batch_size=batch_size,
        win_sec=win_sec, perc=perc,
        cutoff_hz=cutoff_hz, sg_win_ms=sg_win_ms, sg_poly=sg_poly,
        out_dir=str(shadow_plane0), prefix=prefix,
        baseline_mode='rolling',
    )
    print(f"[rolling/analyze] wrote:\n  {dff_path}\n  {low_path}\n  {dt_path}")


# ---------------------------------------------------------------------------
# Per-recording driver
# ---------------------------------------------------------------------------
def run_one_rolling(rec_dir: Path, ckpt: Path | None,
                    fps: float, prefix: str,
                    win_sec: float, perc: int) -> None:
    """rec_dir = <detection_root>/<rec> (must contain final/suite2p/plane0)."""
    t0 = time.time()
    label = rec_dir.name
    print(f"\n{'=' * 72}\n[rolling] {label} ({rec_dir})\n{'=' * 72}")

    plane0 = rec_dir / "final" / "suite2p" / "plane0"
    if not plane0.is_dir():
        raise FileNotFoundError(f"no plane0 at {plane0}")

    import utils
    # FPS is recording-specific; resolve from the real plane0 (the shadow tree
    # would also work since _find_recording_root walks upward).
    rec_fps = utils.get_fps_from_notes(str(plane0)) or fps

    shadow_final, shadow_plane0 = build_shadow_plane0(plane0)

    run_analyze_rolling(shadow_plane0, fps=rec_fps, prefix=prefix,
                        win_sec=win_sec, perc=perc)
    run_cellfilter(shadow_plane0, ckpt_path=ckpt)
    make_filtered_memmaps(shadow_plane0, prefix=prefix)
    run_image_all(shadow_final)
    run_cluster(shadow_plane0, fps=rec_fps, prefix=prefix)
    run_spatial_heatmaps(shadow_final, shadow_plane0, fps=rec_fps, prefix=prefix)

    print(f"[rolling] DONE {label} in {time.time() - t0:.1f}s  "
          f"-> {shadow_final}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detection-root", type=str,
                    default=str(DEFAULT_DETECTION_ROOT),
                    help="root containing <rec>/final/suite2p/plane0 dirs")
    ap.add_argument("--rec", type=str, default=None,
                    help="comma-separated recording names; default = all "
                         "subdirs of --detection-root")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="cellfilter checkpoint (defaults to config best.pt)")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS,
                    help="fallback FPS when notes lookup fails")
    ap.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    ap.add_argument("--win-sec", type=float, default=DEFAULT_WIN_SEC,
                    help=f"rolling baseline window in seconds "
                         f"(default {DEFAULT_WIN_SEC})")
    ap.add_argument("--perc", type=int, default=DEFAULT_PERC,
                    help=f"rolling baseline percentile (default {DEFAULT_PERC})")
    args = ap.parse_args()

    detection_root = Path(args.detection_root)
    if not detection_root.is_dir():
        raise SystemExit(f"detection root does not exist: {detection_root}")
    ckpt = Path(args.ckpt) if args.ckpt else None

    if args.rec:
        wanted = {n.strip() for n in args.rec.split(",") if n.strip()}
        rec_dirs = [detection_root / n for n in sorted(wanted)]
        missing = [str(p) for p in rec_dirs if not p.is_dir()]
        if missing:
            raise SystemExit(f"missing recording dirs: {missing}")
    else:
        rec_dirs = sorted(p for p in detection_root.iterdir() if p.is_dir())

    print(f"[rolling] {len(rec_dirs)} recordings under {detection_root}")
    for rec_dir in rec_dirs:
        try:
            run_one_rolling(rec_dir, ckpt, args.fps, args.prefix,
                            args.win_sec, args.perc)
        except Exception as ex:
            print(f"[rolling] FAILED {rec_dir.name}: {ex}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
