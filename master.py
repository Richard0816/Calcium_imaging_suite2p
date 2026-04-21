"""
master.py — single entry point for the calcium imaging analysis pipeline.

After specifying a recording (or a batch directory) you can run any
combination of pipeline steps, or the full default pipeline.

Examples
--------
# Full pipeline on one recording
python master.py --recording E:/data/2p_shifted/Cx/2024-07-01_00018

# Only re-run trace analysis and clustering
python master.py --recording E:/data/2p_shifted/Cx/2024-07-01_00018 --steps analyze cluster

# Run the full pipeline on every recording inside a region folder
python master.py --batch E:/data/2p_shifted/Cx

# List all available steps
python master.py --list-steps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from recording import Recording
import utils                           # run_with_logging lives here
# Pipeline modules are imported lazily inside each step function so that
# --list-steps works even when optional dependencies (cupy, seaborn, …) are absent.


# ── helpers ────────────────────────────────────────────────────────────────

def _log(logfile: str, func: Callable, *args, **kwargs) -> None:
    utils.run_with_logging(logfile, func, *args, **kwargs)


# ── individual step functions ───────────────────────────────────────────────
# Each step receives a Recording as its first argument and any extra kwargs
# passed from the CLI or from run_pipeline().

def step_suite2p(rec: Recording, **kwargs) -> None:
    """Run Suite2p cell detection."""
    import main as suite2p_runner
    _log(
        "suite2p.log",
        suite2p_runner.run_suite2p_on_folder,
        str(rec.path),
        kwargs.get("addon_vals", None),
    )


def step_analyze(rec: Recording, **kwargs) -> None:
    """Neuropil subtraction, dF/F, low-pass, and derivative memmaps."""
    import analyze_output
    _log("fluorescence_analysis.log", analyze_output.run_analysis_on_folder, str(rec.path))


def step_heatmap(rec: Recording, score_threshold: float = 0.15, **kwargs) -> None:
    """Spatial heatmaps and co-activation order plots."""
    import spatial_heatmap
    _log(
        "raster_and_heatmaps_plots.log",
        spatial_heatmap.run_spatial_heatmap,
        str(rec.path),
        score_threshold=score_threshold,
    )
    _log(
        "raster_and_heatmaps_plots.log",
        spatial_heatmap.coactivation_order_heatmaps,
        str(rec.path),
        score_threshold=score_threshold,
    )


def step_image_all(rec: Recording, **kwargs) -> None:
    """Grid heatmaps and event-raster paged figures."""
    import image_all
    _log("image_all.log", image_all.run_full_imaging_on_folder, str(rec.path))


def step_cluster(rec: Recording, method: str = "ward", metric: str = "euclidean", **kwargs) -> None:
    """Hierarchical clustering of ROI traces."""
    import hierarchical_clustering
    _log(
        "hierarchical_clustering.log",
        hierarchical_clustering.main,
        root=rec.plane0,
        fps=rec.fps,
        prefix=rec.prefix,
        method=method,
        metric=metric,
    )


def step_correlate(
    rec: Recording,
    max_lag_seconds: float = 5.0,
    n_surrogates: int = 5000,
    use_gpu: bool = True,
    **kwargs,
) -> None:
    """Cross-correlation between cluster pairs + surrogate significance test."""
    import crosscorrelation
    n = rec.n_clusters()
    if n < 2:
        print(f"[SKIP] correlate — only {n} cluster(s) in {rec.cluster_dir}  (need >= 2)")
        return

    _log(
        "crosscorrelation.log",
        crosscorrelation.run_cluster_cross_correlations_gpu,
        root=rec.plane0,
        prefix=rec.prefix,
        fps=rec.fps,
        cluster_folder="",
        max_lag_seconds=max_lag_seconds,
        cpu_fallback=True,
        zero_lag=True,
        zero_lag_only=False,
    )

    _log(
        "crosscorrelation.log",
        crosscorrelation.run_clusterpair_zero_lag_shift_surrogate_stats,
        root=rec.plane0,
        prefix=rec.prefix,
        fps=rec.fps,
        n_surrogates=n_surrogates,
        min_shift_s=1,
        max_shift_s=500,
        shift_cluster="B",
        two_sided=False,
        seed=0,
        use_gpu=use_gpu,
        fdr_alpha=0.05,
        save_pairwise_csv=True,
    )


def step_fft(rec: Recording, freq_max: float = 15.0, **kwargs) -> None:
    """FFT power-spectrum figures for all ROIs."""
    import fft_all_rois
    _log(
        "fft.log",
        fft_all_rois.main,
        root=rec.plane0,
        fps=rec.fps,
        prefix=rec.prefix,
        freq_max=freq_max,
    )


def step_curate(rec: Recording, csv_path: str | None = None, **kwargs) -> None:
    """Interactive Tkinter curation GUI for a single recording."""
    import roi_curation_app
    csv = Path(csv_path) if csv_path else Path("roi_curation.csv")
    app = roi_curation_app.CurationApp(roots=[rec.plane0], csv_path=csv)
    app.run()


# ── step registry ───────────────────────────────────────────────────────────

STEPS: dict[str, Callable] = {
    "suite2p":  step_suite2p,
    "analyze":  step_analyze,
    "heatmap":  step_heatmap,
    "image_all": step_image_all,
    "cluster":  step_cluster,
    "correlate": step_correlate,
    "fft":      step_fft,
    "curate":   step_curate,
}

# Default pipeline order (all non-optional steps in sequence)
DEFAULT_PIPELINE: list[str] = ["analyze", "heatmap", "image_all", "cluster", "correlate"]


# ── runner ──────────────────────────────────────────────────────────────────

def run_pipeline(rec: Recording, steps: list[str], **kwargs) -> None:
    """Run the requested steps in order for one recording."""
    print(f"\n{'─' * 64}")
    print(f"  Recording : {rec.name}")
    print(f"  Path      : {rec.path}")
    print(f"  FPS       : {rec.fps:.2f}")
    print(f"  Steps     : {', '.join(steps)}")
    print(f"{'─' * 64}\n")

    for step in steps:
        print(f"\n{'=' * 64}")
        print(f"  STEP ▶ {step.upper()}")
        print(f"{'=' * 64}")
        STEPS[step](rec, **kwargs)

    print(f"\n[DONE] {rec.name}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="master.py",
        description="Calcium imaging analysis pipeline — single recording or batch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--recording", "-r", metavar="PATH",
        help="Path to a single recording directory (YYYY-MM-DD_#####).",
    )
    src.add_argument(
        "--batch", "-b", metavar="DIR",
        help="Run on every sub-directory of DIR (one recording per sub-folder).",
    )

    parser.add_argument(
        "--steps", "-s", nargs="+", metavar="STEP",
        choices=list(STEPS),
        help=(
            "Steps to execute (space-separated). "
            f"Default: {' '.join(DEFAULT_PIPELINE)}. "
            f"All available: {', '.join(STEPS)}."
        ),
    )
    parser.add_argument(
        "--list-steps", action="store_true",
        help="Print available steps and exit.",
    )

    # Per-step tunables exposed at the top level for convenience
    parser.add_argument("--score-threshold", type=float, default=0.15,
                        metavar="T", help="Cell-score threshold for heatmap step (default 0.15).")
    parser.add_argument("--max-lag", type=float, default=5.0,
                        metavar="S", help="Max cross-correlation lag in seconds (default 5.0).")
    parser.add_argument("--n-surrogates", type=int, default=5000,
                        metavar="N", help="Surrogate count for significance test (default 5000).")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU for cross-correlation (CPU fallback).")
    parser.add_argument("--prefix", default=Recording.DEFAULT_PREFIX, metavar="PREFIX",
                        help=f"Memmap prefix (default: {Recording.DEFAULT_PREFIX!r}).")
    parser.add_argument("--notes-root", default=Recording.DEFAULT_NOTES_ROOT, metavar="DIR",
                        help="Root directory of Excel metadata sheets.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_steps:
        print("Pipeline steps (default order):")
        for i, s in enumerate(DEFAULT_PIPELINE, 1):
            print(f"  {i}. {s:<12s}  {STEPS[s].__doc__}")
        print("\nAdditional standalone steps:")
        for s in set(STEPS) - set(DEFAULT_PIPELINE):
            print(f"     {s:<12s}  {STEPS[s].__doc__}")
        sys.exit(0)

    steps = args.steps or DEFAULT_PIPELINE

    kwargs = dict(
        score_threshold=args.score_threshold,
        max_lag_seconds=args.max_lag,
        n_surrogates=args.n_surrogates,
        use_gpu=not args.no_gpu,
    )

    rec_kwargs = dict(
        prefix=args.prefix,
        notes_root=args.notes_root,
    )

    if args.recording:
        rec = Recording(args.recording, **rec_kwargs)
        run_pipeline(rec, steps, **kwargs)

    elif args.batch:
        batch_dir = Path(args.batch)
        if not batch_dir.is_dir():
            parser.error(f"--batch path does not exist or is not a directory: {batch_dir}")

        sub_dirs = sorted(d for d in batch_dir.iterdir() if d.is_dir())
        if not sub_dirs:
            print(f"[WARN] No sub-directories found in {batch_dir}")
            sys.exit(0)

        for d in sub_dirs:
            try:
                rec = Recording(d, **rec_kwargs)
                run_pipeline(rec, steps, **kwargs)
            except FileNotFoundError as exc:
                print(f"[SKIP] {d.name}: {exc}")
            except Exception as exc:
                print(f"[ERROR] {d.name}: {exc}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
