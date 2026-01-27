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


def compute_zero_lag_corr_cpu(sig1, sig2):
    """Zero-lag Pearson correlation (CPU)."""
    sig1 = sig1 - sig1.mean()
    sig2 = sig2 - sig2.mean()
    denom = np.sqrt(np.sum(sig1 * sig1) * np.sum(sig2 * sig2))
    if denom == 0:
        return np.nan
    return float(np.sum(sig1 * sig2) / denom)


def compute_zero_lag_corr_gpu(sig1_np, sig2_np):
    """Zero-lag Pearson correlation (GPU). Requires CuPy."""
    if cp is None:
        raise RuntimeError("CuPy not available for GPU zero-lag correlation.")

    sig1 = cp.asarray(sig1_np, dtype=cp.float32)
    sig2 = cp.asarray(sig2_np, dtype=cp.float32)

    sig1 = sig1 - cp.mean(sig1)
    sig2 = sig2 - cp.mean(sig2)

    denom = cp.sqrt(cp.sum(sig1 * sig1) * cp.sum(sig2 * sig2))
    if float(denom) == 0.0:
        return np.nan

    r0 = cp.sum(sig1 * sig2) / denom
    return float(r0)



def run_cluster_cross_correlations_gpu(root: Path,
                                       prefix: str = "r0p7_",
                                       fps: float = 30.0,
                                       cluster_folder: str = None,
                                       max_lag_seconds: float = 5.0,
                                       cpu_fallback: bool = True,
                                       zero_lag: bool = False,
                                       zero_lag_only: bool = False):
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
    roi_files = [
        f for f in sorted(base_dir.glob("*_rois.npy"))
        if "manual_combined" not in f.stem.lower()
    ]
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
                header = ["roiA", "roiB"]
                if not zero_lag_only:
                    header += ["best_lag_sec", "max_corr"]
                if zero_lag or zero_lag_only:
                    header += ["zero_lag_corr"]
                writer.writerow(header)

                # Iterate through all ROI pairs
                for roiA in roisA:
                    sigA = dff[:, roiA]

                    for roiB in roisB:
                        sigB = dff[:, roiB]

                        # -- zero-lag corr
                        zero_lag_corr = None
                        if zero_lag_only or zero_lag:
                            if use_gpu:
                                zero_lag_corr = compute_zero_lag_corr_gpu(sigA, sigB)
                            else:
                                zero_lag_corr = compute_zero_lag_corr_cpu(sigA, sigB)
                        if not zero_lag_only:
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
                        }

                        if not zero_lag_only:
                            out_npz.update(
                                {"lags_sec": lags_sec,
                                "corr": corr,
                                "best_lag_sec": float(best_lag_sec),
                                "max_corr": float(max_corr)}
                            )
                        if zero_lag_only or zero_lag:
                            out_npz["zero_lag_corr"] = float(zero_lag_corr)

                        outfile = pair_dir / f"roi{roiA:04d}_roi{roiB:04d}.npz"
                        np.savez(outfile, **out_npz)

                        # Save figure (matplotlib still on CPU, but using small arrays)
                        #plt.figure(figsize=(6, 3))
                        #plt.plot(lags_sec, corr)
                        #plt.axvline(best_lag_sec, color="r", ls="--")
                        #plt.title(
                        #    f"ROI {roiA} vs ROI {roiB}\n"
                        #    f"Peak lag = {best_lag_sec:.3f}s, Max corr = {max_corr:.3f}"
                        #)
                        #plt.xlabel("Lag (s)")
                        #plt.ylabel("Correlation")
                        #plt.tight_layout()
                        #plt.savefig(outfile.with_suffix(".png"))
                        #plt.close()
#
                        # CSV row
                        row = [int(roiA), int(roiB)]
                        if not zero_lag_only:
                            row += [float(best_lag_sec), float(max_corr)]
                        if zero_lag_only or zero_lag:
                            row += [float(zero_lag_corr)]
                        writer.writerow(row)

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

def detect_synchronous_bin_onsets(
    dt: np.ndarray,
    fps: float,
    bin_sec: float = 0.5,
    frac_required: float = 0.8,
    z_hi: float = 3.5,
    z_lo: float = 1.5,
    min_sep_s: float = 0.0,
):
    """
    Detect onsets (bin indices) of *new* synchronous/coactivation epochs.

    Definition:
      1) For each ROI, detect event onsets using MAD-z + hysteresis.
      2) Bin event onsets into time bins of length `bin_sec`.
      3) A bin is considered 'synchronous' if >= `frac_required` fraction of ROIs
         have an event in that bin.
      4) A *new* synchronous onset is the first synchronous bin in a contiguous run
         of synchronous bins.

    Parameters
    ----------
    dt : np.ndarray
        Detrended signal, shape (T, N). (Typically your dff_dt memmap.)
    fps : float
        Frames per second.
    bin_sec : float
        Sliding/bin length in seconds (default 0.5).
    frac_required : float
        Fraction of ROIs that must be active in a bin to call it synchronous (default 0.8).
    z_hi, z_lo : float
        Hysteresis thresholds for onset/offset in MAD-z space.
    min_sep_s : float
        Minimum separation (seconds) between events within an ROI.

    Returns
    -------
    onset_bins : np.ndarray
        Bin indices where a new synchronous epoch begins.
    keep_bins : np.ndarray
        All bin indices that meet the synchronous criterion.
    edges : np.ndarray
        Bin edges in seconds, length n_bins+1.
    """
    if dt.ndim != 2:
        raise ValueError(f"dt must be 2D (T, N). Got shape {dt.shape}")

    T, N = dt.shape
    if N < 1 or T < 2:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([0.0, float(T)/fps], dtype=float)

    # --- per-ROI event onsets (seconds) ---
    onsets_sec = []
    for i in range(N):
        z, _, _ = utils.mad_z(dt[:, i])
        idxs = utils.hysteresis_onsets(z, z_hi=z_hi, z_lo=z_lo, fps=fps, min_sep_s=min_sep_s)
        onsets_sec.append(np.asarray(idxs, dtype=np.float32) / float(fps))

    # --- binning ---
    total_sec = T / float(fps)
    n_bins = int(np.ceil(total_sec / float(bin_sec)))
    edges = np.linspace(0.0, n_bins * float(bin_sec), n_bins + 1)

    A = np.zeros((N, n_bins), dtype=bool)
    for i, ts in enumerate(onsets_sec):
        if ts.size == 0:
            continue
        bins = np.searchsorted(edges, ts, side='right') - 1
        bins = bins[(bins >= 0) & (bins < n_bins)]
        if bins.size:
            A[i, np.unique(bins)] = True

    min_count = int(np.ceil(float(frac_required) * N))
    keep_bins = np.where(A.sum(axis=0) >= min_count)[0]

    if keep_bins.size == 0:
        return np.array([], dtype=int), keep_bins.astype(int), edges

    # new epoch onsets = first bin in each contiguous run
    starts = np.insert(np.diff(keep_bins) > 1, 0, True)
    onset_bins = keep_bins[starts]

    return onset_bins.astype(int), keep_bins.astype(int), edges


def run_crosscorr_from_sync_onsets_to_end(
    root: Path,
    prefix: str = "r0p7_",
    fps: float = 30.0,
    cluster_folder: str = None,
    bin_sec: float = 0.5,
    frac_required: float = 0.8,
    use_gpu: bool = True,
    max_lag_seconds: float = 5.0,
    z_hi: float = 3.5,
    z_lo: float = 1.5,
    min_sep_s: float = 0.0,
    max_onsets = None,
):
    """
    Detect *new* synchronous/coactivation epoch onsets (via sliding/bin logic) and for each onset:
      - clip data from onset → end-of-recording
      - run the same cluster×cluster ROI pair cross-correlations on the clipped data

    Output:
      root / f"{prefix}cluster_results" / (cluster_folder optional) /
        crosscorr_from_sync_onsets/
          onset0000_t12p50s/
            C1xC2/
              C1xC2_summary.csv
              roi0126_roi0127.npz
              ...

    Notes
    -----
    - Uses your existing MAD-z + hysteresis event detection (utils.mad_z, utils.hysteresis_onsets).
    - 'Synchronous' means: in a bin of length `bin_sec`, at least `frac_required` fraction of ROIs
      have at least one detected event onset in that bin.
    """

    # ---------------------------
    # 1) LOAD MEMMAPS (T, N)
    # ---------------------------
    ops = np.load(Path(root) / "ops.npy", allow_pickle=True).item()
    stat = np.load(Path(root) / "stat.npy", allow_pickle=True)
    N = len(stat)

    low = np.memmap(Path(root) / f"{prefix}dff_lowpass.memmap.float32", dtype="float32", mode="r")
    T = low.size // N
    dff = low.reshape(T, N)

    dt = np.memmap(Path(root) / f"{prefix}dff_dt.memmap.float32", dtype="float32", mode="r").reshape(T, N)

    # -------------------------------------------
    # 2) DETECT SYNCHRONOUS ONSET BIN INDICES
    # -------------------------------------------
    onset_bins, keep_bins, edges = detect_synchronous_bin_onsets(
        dt=dt,
        fps=fps,
        bin_sec=bin_sec,
        frac_required=frac_required,
        z_hi=z_hi,
        z_lo=z_lo,
        min_sep_s=min_sep_s,
    )

    if onset_bins.size == 0:
        print("⚠️ No synchronous epochs found (no onset bins).")
        return

    if max_onsets is not None:
        onset_bins = onset_bins[: int(max_onsets)]

    onset_times = [float(edges[b]) for b in onset_bins]
    print(f"Found {len(onset_bins)} synchronous epoch onsets (bins): {onset_bins.tolist()}")
    print(f"Onset times (s): {[round(t, 3) for t in onset_times]}")

    # ---------------------------
    # 3) LOAD CLUSTERS
    # ---------------------------
    cluster_dir = Path(root) / f"{prefix}cluster_results"
    if cluster_folder is not None:
        cluster_dir = cluster_dir / cluster_folder

    roi_files = [
        f for f in sorted(cluster_dir.glob("*_rois.npy"))
        if "manual_combined" not in f.stem.lower()
    ]
    if len(roi_files) < 2:
        raise ValueError(f"Need at least two *_rois.npy cluster files in {cluster_dir}.")

    clusters = {f.stem.replace("_rois", ""): np.load(f) for f in roi_files}
    print(f"Loaded clusters: {list(clusters.keys())}")

    # ---------------------------
    # 4) OUTPUT ROOT
    # ---------------------------
    out_root = cluster_dir / "crosscorr_from_sync_onsets"
    out_root.mkdir(exist_ok=True)

    # ---------------------------
    # 5) MAIN LOOP: onset → end
    # ---------------------------
    for k, b0 in enumerate(onset_bins):
        t0_sec = float(edges[b0])
        start_frame = int(round(t0_sec * float(fps)))
        start_frame = max(0, min(start_frame, T - 1))

        clip = dff[start_frame:, :]
        clip_T = clip.shape[0]

        safe_t = f"{t0_sec:.2f}".replace(".", "p")
        onset_dir = out_root / f"onset{k:04d}_t{safe_t}s"
        onset_dir.mkdir(exist_ok=True)

        print(f"\n▶ Sync onset {k}: bin {b0} @ {t0_sec:.3f}s (frame {start_frame}); clip frames {start_frame}–{T} (len={clip_T})")
        if clip_T < 5:
            print("  ⚠️ Clip too short; skipping.")
            continue

        # For each cluster pair
        for cA in clusters:
            for cB in clusters:
                if cA >= cB:
                    continue

                roisA = clusters[cA]
                roisB = clusters[cB]

                pair_dir = onset_dir / f"{cA}x{cB}"
                pair_dir.mkdir(exist_ok=True)

                csv_path = pair_dir / f"{cA}x{cB}_summary.csv"
                with open(csv_path, "w", newline="") as fcsv:
                    w = csv.writer(fcsv)
                    w.writerow(["roiA", "roiB", "best_lag_sec", "max_corr", "clip_start_frame", "clip_start_sec"])

                    for roiA in roisA:
                        sigA = clip[:, int(roiA)]
                        for roiB in roisB:
                            sigB = clip[:, int(roiB)]

                            if use_gpu and cp is not None and cp_signal is not None:
                                lags, corr, lag_best, corr_best = compute_cross_correlation_gpu(
                                    sigA, sigB, fps, max_lag_seconds
                                )
                            else:
                                lags, corr, lag_best, corr_best = compute_cross_correlation(
                                    sigA, sigB, fps, max_lag_seconds
                                )

                            npz_path = pair_dir / f"roi{int(roiA):04d}_roi{int(roiB):04d}.npz"
                            np.savez(
                                npz_path,
                                roiA=int(roiA),
                                roiB=int(roiB),
                                lags_sec=lags,
                                corr=corr,
                                best_lag_sec=float(lag_best),
                                max_corr=float(corr_best),
                                clip_start_frame=int(start_frame),
                                clip_start_sec=float(t0_sec),
                                clip_len_frames=int(clip_T),
                                clip_len_sec=float(clip_T / float(fps)),
                            )

                            w.writerow([int(roiA), int(roiB), float(lag_best), float(corr_best), int(start_frame), float(t0_sec)])

                print(f"  ✔ {cA}×{cB} saved in {pair_dir}")

import numpy as np
import csv
from pathlib import Path

# ----------------------------
# Stats helpers
# ----------------------------

def _bh_fdr(pvals, alpha=0.05):
    """
    Benjamini–Hochberg FDR correction.

    Returns:
        reject (bool array), p_adj (float array)
    """
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(pvals)
    ranked = pvals[order]
    q = alpha * (np.arange(1, m + 1) / m)

    # find largest k where p_k <= q_k
    passed = ranked <= q
    if np.any(passed):
        kmax = np.max(np.where(passed)[0])
        thresh = ranked[kmax]
        reject = pvals <= thresh
    else:
        reject = np.zeros(m, dtype=bool)

    # compute adjusted p-values (monotone)
    p_adj_ranked = ranked * m / np.arange(1, m + 1)
    p_adj_ranked = np.minimum.accumulate(p_adj_ranked[::-1])[::-1]
    p_adj_ranked = np.clip(p_adj_ranked, 0.0, 1.0)

    p_adj = np.empty_like(pvals)
    p_adj[order] = p_adj_ranked
    return reject, p_adj

def _perm_test_with_zero_lag():
    return None


def _perm_test_mean_lag_signflip(lags_sec, n_perm=10_000, seed=0):
    """
    Two-sided sign-flip permutation test for mean(lag) != 0.
    Null: lag signs are random (directionless), centered at 0.
    """
    lags = np.asarray(lags_sec, dtype=float)
    lags = lags[np.isfinite(lags)]
    if lags.size == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    obs = float(np.mean(lags))

    # sign flips: shape (n_perm, n)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, lags.size), replace=True)
    null = np.mean(signs * lags[None, :], axis=1)

    p = float(np.mean(np.abs(null) >= abs(obs)))
    return obs, p


def _xcorr_results_exist(xcorr_root: Path) -> bool:
    """
    Defines what 'existing results' means:
    - at least one cluster-pair folder exists containing either
      a *_summary.csv or at least one .npz file.
    """
    if not xcorr_root.exists():
        return False

    for pair_dir in xcorr_root.iterdir():
        if not pair_dir.is_dir():
            continue
        if any(pair_dir.glob("*_summary.csv")):
            return True
        if any(pair_dir.glob("*.npz")):
            return True
    return False


def _load_lags_for_pair(pair_dir: Path):
    """
    Loads best_lag_sec values for a cluster-pair directory.

    Prefer *_summary.csv because it's fast. Fall back to .npz files.
    Returns:
        lags_sec (np.ndarray)
    """
    # 1) Summary CSV
    summary_files = list(pair_dir.glob("*_summary.csv"))
    if len(summary_files) > 0:
        # If multiple, just take the first
        csv_path = summary_files[0]
        lags = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "best_lag_sec" not in reader.fieldnames:
                # fallback below
                lags = None
            else:
                for row in reader:
                    try:
                        lags.append(float(row["best_lag_sec"]))
                    except Exception:
                        pass
        if lags is not None and len(lags) > 0:
            return np.asarray(lags, dtype=float)

    # 2) Fall back to .npz
    lags = []
    for npz_path in pair_dir.glob("*.npz"):
        try:
            dat = np.load(npz_path, allow_pickle=True)
            # your file naming uses 'best_lag_sec' (based on your code)
            if "best_lag_sec" in dat:
                lags.append(float(dat["best_lag_sec"]))
            elif "lag_best" in dat:
                lags.append(float(dat["lag_best"]))
        except Exception:
            continue

    return np.asarray(lags, dtype=float)


def _compute_clusterpair_lag_stats(
    xcorr_root: Path,
    n_perm=10_000,
    seed=0,
    alpha=0.05,
    min_pairs=10
):
    """
    Runs permutation test per cluster-pair directory under xcorr_root.
    Applies BH-FDR across cluster pairs.
    Writes lag_stats_permtest.csv into xcorr_root.

    Returns:
        stats_rows (list of dict)
    """
    rows = []
    pair_names = []

    pair_dirs = sorted([p for p in xcorr_root.iterdir() if p.is_dir()])
    for pair_dir in pair_dirs:
        pair = pair_dir.name  # e.g., "C1xC2"
        lags = _load_lags_for_pair(pair_dir)
        lags = lags[np.isfinite(lags)]
        n = int(lags.size)

        if n < min_pairs:
            # Skip tiny sample sizes (or keep, but stats will be noisy)
            continue

        mean_lag, p_perm = _perm_test_mean_lag_signflip(lags, n_perm=n_perm, seed=seed)

        rows.append({
            "pair": pair,
            "n_pairs": n,
            "mean_lag_sec": float(mean_lag),
            "median_lag_sec": float(np.median(lags)),
            "iqr_lag_sec": float(np.percentile(lags, 75) - np.percentile(lags, 25)),
            "frac_pos": float(np.mean(lags > 0)),
            "p_perm": float(p_perm),
        })
        pair_names.append(pair)

    # FDR
    pvals = np.array([r["p_perm"] for r in rows], dtype=float)
    reject, p_adj = _bh_fdr(pvals, alpha=alpha)

    for r, rej, padj in zip(rows, reject, p_adj):
        r["p_fdr"] = float(padj)
        r["reject_fdr"] = bool(rej)

    # Save CSV
    out_csv = xcorr_root / "lag_stats_permtest.csv"
    if len(rows) > 0:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    else:
        # still create an empty file with header
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair","n_pairs","mean_lag_sec","median_lag_sec","iqr_lag_sec","frac_pos","p_perm","p_fdr","reject_fdr"])

    print(f"✅ Saved lag stats: {out_csv}")
    return rows

def run_clusterpair_zero_lag_shift_surrogate_stats(
    root: Path,
    prefix: str = "r0p7_",
    fps: float = 30.0,
    cluster_folder: str = None,
    *,
    n_surrogates: int = 1000,
    min_shift_s: float = 1.0,
    max_shift_s: float = 10.0,
    shift_cluster: str = "B",
    two_sided: bool = False,
    seed: int = 0,
    max_pairs: int = 250_000,
    output_dirname: str = "zero_lag_shift_surrogate",
):
    """
    Monte Carlo (time-shift surrogate) test for ZERO-LAG synchrony between cluster pairs.

    For each cluster pair (e.g., C1xC2):
      1) Compute observed zero-lag mean correlation across ROI pairs.
      2) Build a null distribution by circularly shifting EACH ROI trace in one cluster
         (default: cluster B) by an independent random long offset, then recomputing
         the same mean zero-lag correlation.
      3) p-value is computed by counting how often the surrogate statistic is as (or more)
         extreme than the observed statistic (one-sided by default: obs > null).

    This preserves per-ROI autocorrelation and amplitude distribution, while destroying
    fine temporal alignment between clusters.
    """
    root = Path(root)

    # ---- load dF/F memmap ----
    dff, *_ = utils.s2p_open_memmaps(root, prefix=prefix)
    T, n_rois = dff.shape
    if T < 10:
        raise ValueError("Recording too short.")

    # ---- locate clusters ----
    base_dir = root / f"{prefix}cluster_results"
    if cluster_folder is not None:
        base_dir = base_dir / cluster_folder

    roi_files = [
        f for f in sorted(base_dir.glob("*_rois.npy"))
        if "manual_combined" not in f.stem.lower()
    ]
    if len(roi_files) < 2:
        raise ValueError(f"Need at least two *_rois.npy files in {base_dir}")

    clusters = {f.stem.replace("_rois", ""): np.load(f) for f in roi_files}

    # ---- output ----
    out_root = base_dir / output_dirname
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # shift range in frames
    min_shift_f = int(round(min_shift_s * fps))
    max_shift_f = int(round(max_shift_s * fps))
    min_shift_f = max(min_shift_f, 1)
    max_shift_f = max(max_shift_f, min_shift_f)

    if min_shift_f >= T:
        raise ValueError(f"min_shift_s too large for recording length (min_shift_f={min_shift_f}, T={T}).")
    if max_shift_f >= T:
        max_shift_f = T - 1

    def _zscore_matrix(traces_TxN: np.ndarray) -> np.ndarray:
        X = traces_TxN.astype(np.float32, copy=False)
        X = X - np.nanmean(X, axis=0, keepdims=True)
        sd = np.nanstd(X, axis=0, ddof=0, keepdims=True)
        sd[sd == 0] = np.nan
        return X / sd

    def _mean_zero_lag_corr(ZA: np.ndarray, ZB: np.ndarray, pair_idx=None) -> float:
        """
        Mean Pearson correlation at lag=0 across ROI pairs using z-scored matrices.
        If pair_idx provided, uses subsampled pairs (tuple of arrays iA, iB).
        """
        if pair_idx is None:
            C = (ZA.T @ ZB) / ZA.shape[0]   # (nA x nB)
            return float(np.nanmean(C))

        iA, iB = pair_idx
        vals = np.sum(ZA[:, iA] * ZB[:, iB], axis=0) / ZA.shape[0]
        return float(np.nanmean(vals))

    rows = []
    keys = sorted(clusters.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cA, cB = keys[i], keys[j]
            roisA = clusters[cA].astype(int)
            roisB = clusters[cB].astype(int)
            if roisA.size == 0 or roisB.size == 0:
                continue

            XA = np.asarray(dff[:, roisA])
            XB = np.asarray(dff[:, roisB])
            ZA = _zscore_matrix(XA)
            ZB = _zscore_matrix(XB)

            # bound compute by subsampling pairs if needed
            n_pairs_total = roisA.size * roisB.size
            pair_idx = None
            n_pairs_used = n_pairs_total
            if n_pairs_total > max_pairs:
                n_pairs_used = int(max_pairs)
                flat = rng.integers(0, n_pairs_total, size=n_pairs_used, endpoint=False)
                iA = flat // roisB.size
                iB = flat % roisB.size
                pair_idx = (iA, iB)

            obs = _mean_zero_lag_corr(ZA, ZB, pair_idx=pair_idx)

            null = np.empty(n_surrogates, dtype=np.float32)

            if shift_cluster.upper() == "A":
                Z_shift_base = ZA
                Z_fixed = ZB
                n_shift = ZA.shape[1]
                compute = lambda Zs: _mean_zero_lag_corr(Zs, Z_fixed, pair_idx=pair_idx)
            else:
                Z_shift_base = ZB
                Z_fixed = ZA
                n_shift = ZB.shape[1]
                compute = lambda Zs: _mean_zero_lag_corr(Z_fixed, Zs, pair_idx=pair_idx)

            # -------------------------
            # One-surrogate distribution (per-pair r0 values)
            # -------------------------
            save_one_surrogate_dist = True  # or make this a function argument
            one_surrogate_bins = 60
            one_surrogate_max_pairs = 500_000  # subsample pairs if you want

            if save_one_surrogate_dist:
                # 1) Build ONE surrogate-shifted matrix Zs
                shifts = rng.integers(min_shift_f, max_shift_f + 1, size=n_shift, endpoint=False)
                Zs = np.empty_like(Z_shift_base)
                for k, sh in enumerate(shifts):
                    Zs[:, k] = np.roll(Z_shift_base[:, k], int(sh))

                # 2) Compute correlation matrix for that surrogate and flatten to per-pair values
                #    We want per-pair r at lag 0, not the mean.
                if shift_cluster.upper() == "A":
                    # shifted A vs fixed B
                    C = (Zs.T @ Z_fixed) / Zs.shape[0]  # nA x nB
                else:
                    # fixed A vs shifted B
                    C = (Z_fixed.T @ Zs) / Z_fixed.shape[0]  # nA x nB

                vals = C.ravel()
                vals = vals[np.isfinite(vals)]

                # Optional: subsample to keep file sizes and plotting fast
                if vals.size > one_surrogate_max_pairs:
                    idx = rng.choice(vals.size, size=one_surrogate_max_pairs, replace=False)
                    vals_plot = vals[idx]
                else:
                    vals_plot = vals

                # 3) Save histogram + the values (optional)
                hist_dir = out_root / "one_surrogate_pairwise_r0_hists"
                hist_dir.mkdir(exist_ok=True)

                # Save raw values (handy for later comparisons)
                np.save(hist_dir / f"{cA}x{cB}_one_surrogate_pairwise_r0.npy", vals_plot)

                # Plot histogram
                import matplotlib.pyplot as plt

                plt.figure(figsize=(5, 4))
                plt.hist(vals_plot, bins=one_surrogate_bins, density=True)
                plt.xlabel("Pairwise zero-lag correlation (one surrogate)")
                plt.ylabel("Probability density")
                plt.title(f"One surrogate distribution: {cA} × {cB}\n(n={vals_plot.size} pairs)")
                plt.tight_layout()
                plt.savefig(hist_dir / f"{cA}x{cB}_one_surrogate_pairwise_r0_hist.png", dpi=200)
                plt.close()

            for s in range(n_surrogates):
                shifts = rng.integers(min_shift_f, max_shift_f + 1, size=n_shift, endpoint=False)
                Zs = np.empty_like(Z_shift_base)
                for k, sh in enumerate(shifts):
                    Zs[:, k] = np.roll(Z_shift_base[:, k], int(sh))
                null[s] = compute(Zs)

            # -------------------------
            # Save null histogram
            # -------------------------
            hist_dir = out_root / "null_histograms"
            hist_dir.mkdir(exist_ok=True)

            plt.figure(figsize=(5, 4))
            plt.hist(null, bins=40, density=True, alpha=0.7, color="gray", label="Null (time-shift)")
            plt.axvline(obs, color="red", lw=2, label="Observed")

            plt.xlabel("Mean zero-lag correlation")
            plt.ylabel("Probability density")
            plt.title(f"Null distribution: {cA} × {cB}")
            plt.legend(frameon=False)

            hist_path = hist_dir / f"{cA}x{cB}_null_hist.png"
            plt.tight_layout()
            plt.savefig(hist_path, dpi=200)
            plt.close()

            # Monte Carlo p-value by counting (no Gaussian assumption)
            if two_sided:
                p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_surrogates + 1))
                print((np.sum(np.abs(null) >= abs(obs)) + 1))
                print((n_surrogates + 1))
                print('new p val assigned')
            else:
                # one-sided test: obs > null  (equivalently count null >= obs)
                p = float((np.sum(null >= obs) + 1) / (n_surrogates + 1))

            rows.append({
                "pair": f"{cA}x{cB}",
                "clusterA": cA,
                "clusterB": cB,
                "n_rois_A": int(roisA.size),
                "n_rois_B": int(roisB.size),
                "n_pairs_total": int(n_pairs_total),
                "n_pairs_used": int(n_pairs_used),
                "observed_mean_r0": float(obs),
                "null_mean_r0": float(np.nanmean(null)),
                "null_std_r0": float(np.nanstd(null)),
                "p_mc": float(p),
                "n_surrogates": int(n_surrogates),
                "min_shift_s": float(min_shift_s),
                "max_shift_s": float(max_shift_s),
                "shift_cluster": str(shift_cluster),
                "two_sided": bool(two_sided),
                "seed": int(seed),
            })

            print(f"✔ {cA}×{cB}: obs={obs:.4f}, null={float(np.mean(null)):.4f}±{float(np.std(null)):.4f}, p={p:.4g}")

    out_csv = out_root / "zero_lag_shift_surrogate_stats.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair","observed_mean_r0","null_mean_r0","null_std_r0","p_mc"])

    print(f"✅ Saved: {out_csv}")
    return rows

# ----------------------------
# Main wrapper
# ----------------------------

def run_or_load_clusterpair_lag_stats(
    root: Path,
    prefix: str,
    fps: float,
    cluster_folder: str = None,
    max_lag_seconds: float = 5.0,
    use_gpu: bool = True,
    cpu_fallback: bool = True,
    xcorr_dirname: str = "cross_correlation_gpu",
    n_perm: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    min_pairs: int = 10
):
    """
    If cross-correlation outputs exist, load them and compute stats.
    If not, run cross-correlation first, then compute stats.

    Returns:
        rows (list of dict) -- also writes lag_stats_permtest.csv
    """
    root = Path(root)

    # Match your existing directory logic
    base_dir = root / f"{prefix}cluster_results"
    if cluster_folder is not None:
        base_dir = base_dir / cluster_folder
    base_dir.mkdir(parents=True, exist_ok=True)

    xcorr_root = base_dir / xcorr_dirname

    # 1) If no results exist -> run cross-correlation
    if not _xcorr_results_exist(xcorr_root):
        print(f"ℹ️ No existing xcorr results found in: {xcorr_root}")
        print("➡️ Running cross-correlation first...")

        # This calls your existing function in this same file
        run_cluster_cross_correlations_gpu(
            root=root,
            prefix=prefix,
            fps=fps,
            cluster_folder=cluster_folder,
            max_lag_seconds=max_lag_seconds,
            cpu_fallback=cpu_fallback
        )

    else:
        print(f"✅ Found existing xcorr outputs in: {xcorr_root}")
        print("➡️ Skipping recompute; loading saved lags...")

    # 2) Compute stats from whatever is there now
    rows = _compute_clusterpair_lag_stats(
        xcorr_root=xcorr_root,
        n_perm=n_perm,
        seed=seed,
        alpha=alpha,
        min_pairs=min_pairs
    )
    return rows


if __name__ == "__main__":
    root = Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0")
    prefix = "r0p7_filtered_"
    fps = 30.0
    #run_or_load_clusterpair_lag_stats(
    #    root=root,
    #    prefix="r0p7_",
    #    fps=30.0,
    #    cluster_folder="C1C2C4_recluster",
    #    max_lag_seconds=5.0,
    #    xcorr_dirname="cross_correlation_gpu",  # matches your function’s folder name
    #    n_perm=10000,
    #    seed=0,
    #    alpha=0.05,
    #    min_pairs=10
    #)
    #run_cluster_cross_correlations_gpu(
    #    root=root,
    #    prefix="r0p7_",
    #    fps=30.0,
    #    cluster_folder="C1C2C4_recluster",
    #    max_lag_seconds=5.0,
    #    cpu_fallback=True,
    #    zero_lag=True,
    #    zero_lag_only=False,
    #)
    rows = run_clusterpair_zero_lag_shift_surrogate_stats(
        root=root,
        prefix="r0p7_",
        fps=30.0,
        cluster_folder="C1C2C4_recluster",
        n_surrogates=1000,
        min_shift_s=1.0,
        max_shift_s=10.0,
        shift_cluster="B",  # shift C2, leave C1 fixed
        two_sided=True,  # synchrony: usually one-sided
        seed=0,
    )
    #run_crosscorr_from_sync_onsets_to_end(
    #    root=root,
    #    prefix="r0p7_",
    #    fps=30.0,
    #    cluster_folder="C1C2C4_recluster",
    #    bin_sec=0.5,
    #    frac_required=0.8,
    #    use_gpu=True,
    #    max_lag_seconds=5.0,
    #    # optional:
    #    # max_onsets=3,        # only process first 3 synchronous epochs
    #    # min_sep_s=0.3,       # enforce per-ROI event separation
    #)
    #run_crosscorr_per_coactivation_bin(
    #    root=Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0"),
    #    prefix="r0p7_",
    #    fps=30.0,
    #    cluster_folder="C1C2C4_recluster",
    #    bin_sec=0.5,
    #    frac_required=0.8,
    #    use_gpu=True
    #)
