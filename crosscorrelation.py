import csv
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import utils

try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
except ImportError:
    cp = None
    cp_signal = None
    print("⚠️ CuPy not available; GPU cross-correlation will fall back to CPU.")


def compute_cross_correlation_gpu(sig1_np, sig2_np, fps, max_lag_seconds=5.0):
    """
    GPU version: return (lags_sec_np, corr_np, best_lag_sec, max_corr).

    sig1_np, sig2_np: 1D numpy arrays (same length).
    """
    if cp is None or cp_signal is None:
        raise RuntimeError("CuPy/cupyx not available; cannot run GPU version.")

    # Move to GPU
    sig1 = cp.asarray(sig1_np, dtype=cp.float32)
    sig2 = cp.asarray(sig2_np, dtype=cp.float32)

    # Demean
    sig1 = sig1 - cp.mean(sig1)
    sig2 = sig2 - cp.mean(sig2)

    n = sig1.shape[0]

    # Full cross-correlation on GPU
    corr = cp_signal.correlate(sig1, sig2, mode="full")  # length 2n-1

    # Lags in samples for mode="full" with equal-length signals
    lags = cp.arange(-(n - 1), n, dtype=cp.int32)  # same length as corr
    lags_sec = lags / float(fps)

    # Restrict to ± max_lag_seconds
    if max_lag_seconds is not None:
        mask = cp.abs(lags_sec) <= max_lag_seconds
        corr = corr[mask]
        lags_sec = lags_sec[mask]

    # Find peak
    idx = int(cp.argmax(corr))
    best_lag_sec = float(lags_sec[idx])
    max_corr = float(corr[idx])

    # Move arrays back to CPU for saving/plotting
    return cp.asnumpy(lags_sec), cp.asnumpy(corr), best_lag_sec, max_corr


def compute_cross_correlation(sig1, sig2, fps, max_lag_seconds=5):
    """Return lags_sec, corr, best_lag_sec, max_corr."""
    sig1 = sig1 - sig1.mean()
    sig2 = sig2 - sig2.mean()

    corr = signal.correlate(sig1, sig2, mode='full')
    lags = signal.correlation_lags(len(sig1), len(sig2), mode='full')
    lags_sec = lags / fps

    # restrict search window
    mask = np.abs(lags_sec) <= max_lag_seconds
    corr = corr[mask]
    lags_sec = lags_sec[mask]

    idx = np.argmax(corr)
    best_lag_sec = lags_sec[idx]
    max_corr = corr[idx]

    return lags_sec, corr, best_lag_sec, max_corr


def run_cluster_cross_correlations_gpu(root: Path,
                                       prefix: str = "r0p7_",
                                       fps: float = 30.0,
                                       cluster_folder: str = None,
                                       max_lag_seconds: float = 5.0,
                                       cpu_fallback: bool = True):
    """
    GPU-accelerated cluster × cluster cross-correlation using CuPy.

    - Loads all *_rois.npy cluster files from:
        root / f"{prefix}cluster_results" / (cluster_folder optional)
    - Computes cross-correlation for every ROI in cluster A vs every ROI in cluster B
      using the GPU.
    - Saves per-pair .npz + .png and a per-cluster-pair summary CSV.

    If CuPy is not available and cpu_fallback=True, silently falls back
    to the CPU version.
    """
    # Check for GPU availability
    use_gpu = (cp is not None and cp_signal is not None)
    if not use_gpu:
        msg = "CuPy not available; "
        if cpu_fallback:
            print(msg + "falling back to CPU cross-correlation.")
        else:
            raise RuntimeError(msg + "set cpu_fallback=True to use CPU instead.")

    # Load ΔF/F (numpy memmap -> numpy array view)
    dff, _, _ = utils.s2p_open_memmaps(root, prefix=prefix)[:3]  # (T, N)
    n_frames, n_rois = dff.shape
    print(f"Loaded ΔF/F: {n_frames} frames × {n_rois} ROIs")

    # Determine cluster folder
    base_dir = root / f"{prefix}cluster_results"
    if cluster_folder is not None:
        base_dir = base_dir / cluster_folder
    base_dir.mkdir(parents=True, exist_ok=True)

    # Find cluster ROI files
    roi_files = sorted(f for f in base_dir.glob("*_rois.npy"))
    if len(roi_files) < 2:
        raise ValueError(f"Need at least two *_rois.npy files in {base_dir} to cross-correlate.")

    # Output root
    xcorr_root = base_dir / "cross_correlation_gpu"
    xcorr_root.mkdir(exist_ok=True)

    # Load clusters
    clusters = {}
    for f in roi_files:
        key = f.stem.replace("_rois", "")  # "C1", "C2", etc.
        rois = np.load(f)
        clusters[key] = rois
        print(f"Loaded {key}: {len(rois)} ROIs from {f}")

    # Loop over cluster pairs
    for cA in clusters:
        for cB in clusters:
            # Only compute each unordered pair once, skip same-cluster
            if cA >= cB:
                continue

            roisA = clusters[cA]
            roisB = clusters[cB]

            pair_dir = xcorr_root / f"{cA}x{cB}"
            pair_dir.mkdir(exist_ok=True)
            print(f"\n▶ [GPU] Computing {cA} × {cB} → {pair_dir}")

            # CSV summary
            summary_path = pair_dir / f"{cA}x{cB}_summary.csv"
            with open(summary_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["roiA", "roiB", "best_lag_sec", "max_corr"])

                # Iterate through all ROI pairs
                for roiA in roisA:
                    sigA = dff[:, roiA]

                    for roiB in roisB:
                        sigB = dff[:, roiB]

                        if use_gpu:
                            lags_sec, corr, best_lag_sec, max_corr = \
                                compute_cross_correlation_gpu(sigA, sigB, fps, max_lag_seconds)
                        else:
                            # CPU fallback (reuse your CPU helper if you prefer)
                            lags_sec, corr, best_lag_sec, max_corr = \
                                compute_cross_correlation(sigA, sigB, fps, max_lag_seconds)

                        # Save numeric output
                        out_npz = {
                            "roiA": int(roiA),
                            "roiB": int(roiB),
                            "lags_sec": lags_sec,
                            "corr": corr,
                            "best_lag_sec": float(best_lag_sec),
                            "max_corr": float(max_corr),
                        }
                        outfile = pair_dir / f"roi{roiA:04d}_roi{roiB:04d}.npz"
                        np.savez(outfile, **out_npz)

                        # Save figure (matplotlib still on CPU, but using small arrays)
                        plt.figure(figsize=(6, 3))
                        plt.plot(lags_sec, corr)
                        plt.axvline(best_lag_sec, color="r", ls="--")
                        plt.title(
                            f"ROI {roiA} vs ROI {roiB}\n"
                            f"Peak lag = {best_lag_sec:.3f}s, Max corr = {max_corr:.3f}"
                        )
                        plt.xlabel("Lag (s)")
                        plt.ylabel("Correlation")
                        plt.tight_layout()
                        plt.savefig(outfile.with_suffix(".png"))
                        plt.close()

                        # CSV row
                        writer.writerow([roiA, roiB, best_lag_sec, max_corr])

            print(f"  ✔ Summary CSV saved to: {summary_path}")

    print("\n✅ GPU cross-correlation computations complete.")
    print(f"Results saved under: {xcorr_root}")

def run_crosscorr_per_coactivation_bin(
    root: Path,
    prefix: str = "r0p7_",
    fps: float = 30.0,
    cluster_folder: str = None,
    bin_sec: float = 0.5,
    frac_required: float = 0.8,
    use_gpu: bool = True,
    max_lag_seconds: float = 5.0
):
    """
    1) Detect coactivation bins using your spatial_heatmap.py logic.
    2) For each coactivation window: crop the ΔF/F traces to that window.
    3) Run cross-correlation (GPU or CPU) on that window.
    4) Save results in per-bin folders: crosscorr_by_coact/binXXXX/
    """

    # ----------------------------------------------------------------------
    # 1. LOAD ΔF/F, STAT, AND DETECT COACT BINS (same logic as spatial_heatmap)
    # ----------------------------------------------------------------------
    # Load Suite2p data
    ops = np.load(root / "ops.npy", allow_pickle=True).item()
    stat = np.load(root / "stat.npy", allow_pickle=True)
    Ly, Lx = ops["Ly"], ops["Lx"]

    # Load memmaps
    low = np.memmap(root / f"{prefix}dff_lowpass.memmap.float32",
                    dtype="float32", mode="r")
    N = len(stat)
    T = low.size // N
    dff = low.reshape(T, N)

    # Compute event onsets per ROI (using same method as your coactivation maps)
    dt = np.memmap(root / f"{prefix}dff_dt.memmap.float32",
                   dtype="float32", mode="r").reshape(T, N)

    # detect events (MAD-z + hysteresis)
    onsets = []
    for i in range(N):
        z, _, _ = utils.mad_z(dt[:, i])
        idxs = utils.hysteresis_onsets(z, z_hi=3.5, z_lo=1.5, fps=fps)
        onsets.append(np.array(idxs) / fps)

    # binning logic
    total_sec = T / fps
    n_bins = int(np.ceil(total_sec / bin_sec))
    edges = np.linspace(0, n_bins * bin_sec, n_bins + 1)

    # activation matrix
    A = np.zeros((N, n_bins), dtype=bool)
    for i, ts in enumerate(onsets):
        if ts.size == 0:
            continue
        bins = np.searchsorted(edges, ts, side='right') - 1
        bins = bins[(bins >= 0) & (bins < n_bins)]
        A[i, np.unique(bins)] = True

    # find bins with sufficient coactivation
    min_count = int(np.ceil(frac_required * N))
    keep_bins = np.where(A.sum(axis=0) >= min_count)[0]

    if keep_bins.size == 0:
        print("⚠️ No coactivation bins found.")
        return

    print(f"Found {len(keep_bins)} coactivation bins: {keep_bins.tolist()}")

    # ----------------------------------------------------------------------
    # 2. LOAD CLUSTER FILES
    # ----------------------------------------------------------------------
    cluster_dir = root / f"{prefix}cluster_results"
    if cluster_folder is not None:
        cluster_dir = cluster_dir / cluster_folder

    # Find cluster ROI files, excluding manual_combined*
    roi_files = [
        f for f in sorted(cluster_dir.glob("*_rois.npy"))
        if "manual_combined" not in f.stem.lower()
    ]

    clusters = {
        f.stem.replace("_rois", ""): np.load(f)
        for f in roi_files
    }

    print(f"Loaded clusters: {list(clusters.keys())}")

    # ----------------------------------------------------------------------
    # 3. OUTPUT FOLDER
    # ----------------------------------------------------------------------
    out_root = cluster_dir / "crosscorr_by_coact"
    out_root.mkdir(exist_ok=True)

    # ----------------------------------------------------------------------
    # 4. MAIN LOOP: CROP + RUN CROSSCORR
    # ----------------------------------------------------------------------
    for b in keep_bins:
        t0 = int(edges[b] * fps)
        t1 = int(edges[b+1] * fps)
        print(f"\n▶ Coactivation bin {b}: frames {t0}–{t1}")

        cropped = dff[t0:t1, :]

        # Create folder for this coactivation event
        bin_dir = out_root / f"bin{b:04d}"
        bin_dir.mkdir(exist_ok=True)

        # For each cluster pair
        for cA in clusters:
            for cB in clusters:
                if cA >= cB:
                    continue

                roisA = clusters[cA]
                roisB = clusters[cB]

                pair_dir = bin_dir / f"{cA}x{cB}"
                pair_dir.mkdir(exist_ok=True)

                # summary CSV
                csv_path = pair_dir / f"{cA}x{cB}_summary.csv"
                with open(csv_path, "w", newline="") as fcsv:
                    w = csv.writer(fcsv)
                    w.writerow(["roiA", "roiB", "best_lag_sec", "max_corr"])

                    # pairwise computation
                    for roiA in roisA:
                        sigA = cropped[:, roiA]
                        for roiB in roisB:
                            sigB = cropped[:, roiB]

                            if use_gpu and cp is not None:
                                lags, corr, lag_best, corr_best = compute_cross_correlation_gpu(
                                    sigA, sigB, fps, max_lag_seconds
                                )
                            else:
                                lags, corr, lag_best, corr_best = compute_cross_correlation(
                                    sigA, sigB, fps, max_lag_seconds
                                )

                            # save .npz
                            npz_path = pair_dir / f"roi{roiA:04d}_roi{roiB:04d}.npz"
                            np.savez(npz_path,
                                     roiA=roiA,
                                     roiB=roiB,
                                     lags_sec=lags,
                                     corr=corr,
                                     best_lag_sec=lag_best,
                                     max_corr=corr_best)

                            # write CSV line
                            w.writerow([roiA, roiB, lag_best, corr_best])

                print(f"  ✔ {cA}×{cB} saved in {pair_dir}")

if __name__ == "__main__":
    root = Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0")
    prefix = "r0p7_filtered_"
    fps = 30.0
    run_crosscorr_per_coactivation_bin(
        root=Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0"),
        prefix="r0p7_",
        fps=30.0,
        cluster_folder="C1C2C4_recluster",
        bin_sec=0.5,
        frac_required=0.8,
        use_gpu=True
    )
    #run_cluster_cross_correlations_gpu(
    #    root=root,
    #    prefix=prefix,
    #    fps=fps,
    #    cluster_folder="C1C3_recluster",  # or None if using base cluster_results
    #    max_lag_seconds=5.0,
    #    cpu_fallback=True,  # set False if you want it to hard-fail when CuPy is missing
    #)