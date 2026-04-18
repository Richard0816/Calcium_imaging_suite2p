from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class StatMethodFigureConfig:
    recording_root: Path
    cluster_a_file: Path
    cluster_b_file: Path
    prefix: str = "r0p7_"
    n_surrogates: int = 10_000
    min_shift_s: float = 5.0
    max_shift_s: float = 30.0
    shift_cluster: str = "A"   # "A" or "B"
    max_pairs: int = 500_000
    seed: int = 123
    bins_pairwise: int = 60
    bins_null_means: int = 50
    save_path: Optional[Path] = None


def _zscore_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _mean_zero_lag_corr(ZA: np.ndarray, ZB: np.ndarray, pair_idx=None) -> float:
    """
    Mean Pearson correlation at lag 0 across ROI pairs using z scored matrices.
    ZA, ZB shape: (T, nRoi)
    """
    if pair_idx is None:
        C = (ZA.T @ ZB) / ZA.shape[0]
        return float(np.nanmean(C))

    iA, iB = pair_idx
    vals = np.sum(ZA[:, iA] * ZB[:, iB], axis=0) / ZA.shape[0]
    return float(np.nanmean(vals))


def _pairwise_zero_lag_corr_values(ZA: np.ndarray, ZB: np.ndarray, pair_idx=None) -> np.ndarray:
    """
    Returns per pair lag 0 Pearson correlations.
    """
    if pair_idx is None:
        C = (ZA.T @ ZB) / ZA.shape[0]
        vals = C.ravel()
    else:
        iA, iB = pair_idx
        vals = np.sum(ZA[:, iA] * ZB[:, iB], axis=0) / ZA.shape[0]

    vals = np.asarray(vals, dtype=np.float32)
    return vals[np.isfinite(vals)]


def make_stat_method_figure(
    dff: np.ndarray,
    fps: float,
    roisA: np.ndarray,
    roisB: np.ndarray,
    cfg: StatMethodFigureConfig,
):
    rng = np.random.default_rng(cfg.seed)

    roisA = np.asarray(roisA, dtype=int).ravel()
    roisB = np.asarray(roisB, dtype=int).ravel()

    if roisA.size == 0 or roisB.size == 0:
        raise ValueError("One or both ROI groups are empty.")

    XA = np.asarray(dff[:, roisA], dtype=np.float32)
    XB = np.asarray(dff[:, roisB], dtype=np.float32)
    ZA = _zscore_matrix(XA)
    ZB = _zscore_matrix(XB)

    # Bound pair count if needed
    n_pairs_total = roisA.size * roisB.size
    pair_idx = None
    n_pairs_used = n_pairs_total

    if n_pairs_total > cfg.max_pairs:
        n_pairs_used = int(cfg.max_pairs)
        flat = rng.integers(0, n_pairs_total, size=n_pairs_used, endpoint=False)
        iA = flat // roisB.size
        iB = flat % roisB.size
        pair_idx = (iA, iB)

    # Observed world
    obs_mean = _mean_zero_lag_corr(ZA, ZB, pair_idx=pair_idx)
    obs_pairwise = _pairwise_zero_lag_corr_values(ZA, ZB, pair_idx=pair_idx)

    # Surrogate setup
    min_shift_f = max(1, int(round(cfg.min_shift_s * fps)))
    max_shift_f = max(min_shift_f, int(round(cfg.max_shift_s * fps)))

    if cfg.shift_cluster.upper() == "A":
        Z_shift_base = ZA
        Z_fixed = ZB
        n_shift = ZA.shape[1]

        def compute_mean(Zs):
            return _mean_zero_lag_corr(Zs, Z_fixed, pair_idx=pair_idx)

        def compute_vals(Zs):
            return _pairwise_zero_lag_corr_values(Zs, Z_fixed, pair_idx=pair_idx)

    else:
        Z_shift_base = ZB
        Z_fixed = ZA
        n_shift = ZB.shape[1]

        def compute_mean(Zs):
            return _mean_zero_lag_corr(Z_fixed, Zs, pair_idx=pair_idx)

        def compute_vals(Zs):
            return _pairwise_zero_lag_corr_values(Z_fixed, Zs, pair_idx=pair_idx)

    # One surrogate world for panel A
    shifts = rng.integers(min_shift_f, max_shift_f + 1, size=n_shift, endpoint=False)
    Zs = np.empty_like(Z_shift_base)
    for k, sh in enumerate(shifts):
        Zs[:, k] = np.roll(Z_shift_base[:, k], int(sh))
    one_surrogate_pairwise = compute_vals(Zs)

    # Null distribution of surrogate means for panel B
    null_means = np.empty(cfg.n_surrogates, dtype=np.float32)
    for s in range(cfg.n_surrogates):
        shifts = rng.integers(min_shift_f, max_shift_f + 1, size=n_shift, endpoint=False)
        Zs = np.empty_like(Z_shift_base)
        for k, sh in enumerate(shifts):
            Zs[:, k] = np.roll(Z_shift_base[:, k], int(sh))
        null_means[s] = compute_mean(Zs)

    null_mean = float(np.nanmean(null_means))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # A. One surrogate pairwise distribution
    axes[0].hist(one_surrogate_pairwise, bins=cfg.bins_pairwise, density=False)
    axes[0].axvline(np.nanmean(one_surrogate_pairwise), color="red", lw=2)
    axes[0].set_title("A  One shuffled surrogate world")
    axes[0].set_xlabel("Pairwise zero lag correlation")
    axes[0].set_ylabel("Count")

    # B. Distribution of surrogate means
    axes[1].hist(null_means, bins=cfg.bins_null_means, density=False)
    axes[1].axvline(null_mean, color="red", lw=2, label="Null mean")
    axes[1].axvline(obs_mean, color="black", lw=2, linestyle="--", label="Observed mean")
    axes[1].set_title(f"B  Mean correlation across {cfg.n_surrogates:,} null samples")
    axes[1].set_xlabel("Mean zero lag correlation")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=False)

    # C. Actual observed pairwise distribution
    axes[2].hist(obs_pairwise, bins=cfg.bins_pairwise, density=False)
    axes[2].axvline(np.nanmean(obs_pairwise), color="black", lw=2)
    axes[2].set_title("C  Actual distribution")
    axes[2].set_xlabel("Pairwise zero lag correlation")
    axes[2].set_ylabel("Count")

    fig.suptitle(
        f"Surrogate based comparison of zero lag correlations\n"
        f"{cfg.cluster_a_file.stem.replace('_rois','')} vs {cfg.cluster_b_file.stem.replace('_rois','')}"
        if cfg.cluster_a_file is not None and cfg.cluster_b_file is not None
        else "Surrogate based comparison of zero lag correlations",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if cfg.save_path is not None:
        cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {cfg.save_path}")

    stats = {
        "observed_mean_r0": float(obs_mean),
        "null_mean_r0": float(null_mean),
        "null_std_r0": float(np.nanstd(null_means)),
        "n_rois_A": int(roisA.size),
        "n_rois_B": int(roisB.size),
        "n_pairs_total": int(n_pairs_total),
        "n_pairs_used": int(n_pairs_used),
        "one_surrogate_pairwise": one_surrogate_pairwise,
        "null_means": null_means,
        "observed_pairwise": obs_pairwise,
    }

    return fig, stats

from pathlib import Path
import numpy as np
import utils

recording_root = Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0")
cluster_a_file = Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0\r0p7_filtered_cluster_results\C1_rois.npy")
cluster_b_file = Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0\r0p7_filtered_cluster_results\C2_rois.npy")

dff, _, _, T, N = utils.s2p_open_memmaps(str(recording_root), prefix="r0p7_")
fps = utils.get_fps_from_notes(str(recording_root))

roisA = np.load(cluster_a_file).astype(int).ravel()
roisB = np.load(cluster_b_file).astype(int).ravel()

cfg = StatMethodFigureConfig(
    recording_root=recording_root,
    cluster_a_file=cluster_a_file,
    cluster_b_file=cluster_b_file,
    prefix="r0p7_filtered_",
    n_surrogates=100_000,
    min_shift_s=20.0,
    max_shift_s=2000.0,
    shift_cluster="A",
    max_pairs=500_000,
    seed=123,
    save_path=None,
)

fig, stats = make_stat_method_figure(
    dff=dff,
    fps=fps,
    roisA=roisA,
    roisB=roisB,
    cfg=cfg,
)

plt.show()

print(stats["observed_mean_r0"])
print(stats["null_mean_r0"])