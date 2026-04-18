from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class LagPermFigureConfig:
    pair_name: str = "C1xC3"
    n_perm: int = 10_000
    seed: int = 0
    bins_pairwise: int = 60
    bins_null_means: int = 50
    save_path: Optional[Path] = None


def _load_best_lags_from_summary(summary_csv: Path) -> np.ndarray:
    df = pd.read_csv(summary_csv)
    if "best_lag_sec" not in df.columns:
        raise ValueError(f"'best_lag_sec' not found in {summary_csv}")
    lags = df["best_lag_sec"].to_numpy(dtype=float)
    lags = lags[np.isfinite(lags)]
    if lags.size == 0:
        raise ValueError(f"No finite lag values found in {summary_csv}")
    return lags


def _signflip_null_means(lags: np.ndarray, n_perm: int, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    obs_mean = float(np.mean(lags))

    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, lags.size), replace=True)
    null_means = np.mean(signs * lags[None, :], axis=1)

    return null_means.astype(np.float32), obs_mean


def _one_signflip_world(lags: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=lags.size, replace=True)
    return (lags * signs).astype(np.float32)


def make_lag_perm_method_figure(
    summary_csv: Path,
    cfg: LagPermFigureConfig,
):
    lags = _load_best_lags_from_summary(summary_csv)

    one_null_world = _one_signflip_world(lags, seed=cfg.seed + 1)
    null_means, obs_mean = _signflip_null_means(lags, n_perm=cfg.n_perm, seed=cfg.seed)
    null_mean = float(np.mean(null_means))

    p_two_sided = float((np.sum(np.abs(null_means) >= abs(obs_mean)) + 1) / (cfg.n_perm + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # A. One sign-flipped surrogate world
    axes[0].hist(one_null_world, bins=cfg.bins_pairwise)
    axes[0].axvline(np.mean(one_null_world), color="red", lw=2)
    axes[0].set_title("A  One sign flipped surrogate world")
    axes[0].set_xlabel("Pairwise best lag (s)")
    axes[0].set_ylabel("Count")

    # B. Null distribution of mean lags
    axes[1].hist(null_means, bins=cfg.bins_null_means)
    axes[1].axvline(null_mean, color="red", lw=2, label="Null mean")
    axes[1].axvline(obs_mean, color="black", lw=2, linestyle="--", label="Observed mean")
    axes[1].set_title(f"B  Mean lag across {cfg.n_perm:,} sign flip null samples")
    axes[1].set_xlabel("Mean lag (s)")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=False)

    # C. Actual observed lag distribution
    axes[2].hist(lags, bins=cfg.bins_pairwise)
    axes[2].axvline(np.mean(lags), color="black", lw=2)
    axes[2].set_title("C  Actual lag distribution")
    axes[2].set_xlabel("Pairwise best lag (s)")
    axes[2].set_ylabel("Count")

    fig.suptitle(
        f"Sign flip permutation test for lag bias\n"
        f"{cfg.pair_name}  |  observed mean = {obs_mean:.4f} s,  p = {p_two_sided:.4g}",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if cfg.save_path is not None:
        cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {cfg.save_path}")

    stats = {
        "pair_name": cfg.pair_name,
        "n_pairs": int(lags.size),
        "observed_mean_lag_sec": float(obs_mean),
        "observed_median_lag_sec": float(np.median(lags)),
        "null_mean_lag_sec": float(null_mean),
        "null_std_lag_sec": float(np.std(null_means)),
        "p_two_sided": float(p_two_sided),
        "one_null_world": one_null_world,
        "null_means": null_means,
        "observed_lags": lags,
    }

    return fig, stats


if __name__ == "__main__":
    summary_csv = Path(
        r"F:\data\2p_shifted\Hip\2024-06-04_00009\suite2p\plane0"
        r"\r0p7_filtered_cluster_results\cross_correlation_gpu\C1xC3\C1xC3_summary.csv"
    )

    cfg = LagPermFigureConfig(
        pair_name="C1xC3",
        n_perm=10_000,
        seed=0,
        save_path=None,
    )

    fig, stats = make_lag_perm_method_figure(summary_csv, cfg)
    plt.show()

    print(stats["observed_mean_lag_sec"])
    print(stats["p_two_sided"])