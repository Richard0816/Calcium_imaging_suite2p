from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations


P35_ROOT = Path(r"F:\data\2p_shifted\Hip\2024-06-04_00009\suite2p\plane0")
P37_ROOT = Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018\suite2p\plane0")

PREFIX = "r0p7_filtered_"

P35_LABEL = "Patient 35 Hippocampus"
P37_LABEL = "Patient 37 Cortex"


def get_xcorr_root(root: Path):
    return root / f"{PREFIX}cluster_results" / "cross_correlation_gpu"


def natural_pair_key(name: str):
    a, b = name.split("x")
    return int(a[1:]), int(b[1:])


def load_pair_data(xcorr_root: Path):

    pair_data = {}

    for d in xcorr_root.iterdir():

        if not d.is_dir():
            continue

        csv = list(d.glob("*_summary.csv"))[0]

        df = pd.read_csv(csv)

        pair_data[d.name] = df

    return dict(sorted(pair_data.items(), key=lambda x: natural_pair_key(x[0])))


def get_metric_arrays(pair_data, metric):

    labels = []
    arrays = []

    for pair, df in pair_data.items():

        arr = df[metric].dropna().values

        if len(arr) == 0:
            continue

        labels.append(pair)
        arrays.append(arr)

    return labels, arrays


def sign_flip_pvalue(values, n_perm=10000):

    observed = np.mean(values)

    signs = np.random.choice([-1, 1], size=(n_perm, len(values)))

    permuted = np.mean(values * signs, axis=1)

    p = np.mean(np.abs(permuted) >= abs(observed))

    return p


def compute_significance(arrays):

    pvals = []

    for arr in arrays:
        pvals.append(sign_flip_pvalue(arr))

    return pvals


def add_significance(ax, positions, arrays):

    pvals = compute_significance(arrays)

    ymax = ax.get_ylim()[1]

    for i, p in enumerate(pvals):

        if p < 0.05:

            ax.text(
                positions[i],
                ymax * 0.93,
                "*",
                ha="center",
                va="bottom",
                fontsize=14,
            )


def plot_violin(ax, labels, arrays, ylabel, title, show_sig=False):

    pos = np.arange(1, len(arrays) + 1)

    parts = ax.violinplot(
        arrays,
        positions=pos,
        showmedians=True,
        showextrema=False,
    )

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_sig:
        add_significance(ax, pos, arrays)


def add_panel(ax, letter):

    ax.text(
        -0.18,
        1.05,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )


def main():

    p35 = load_pair_data(get_xcorr_root(P35_ROOT))
    p37 = load_pair_data(get_xcorr_root(P37_ROOT))

    p35_corr_labels, p35_corr = get_metric_arrays(p35, "zero_lag_corr")
    p35_lag_labels, p35_lag = get_metric_arrays(p35, "best_lag_sec")

    p37_corr_labels, p37_corr = get_metric_arrays(p37, "zero_lag_corr")
    p37_lag_labels, p37_lag = get_metric_arrays(p37, "best_lag_sec")

    fig, ax = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    plot_violin(
        ax[0, 0],
        p35_corr_labels,
        p35_corr,
        "Zero lag correlation",
        P35_LABEL,
        show_sig=False,
    )

    plot_violin(
        ax[1, 0],
        p35_lag_labels,
        p35_lag,
        "Lag at peak correlation (s)",
        P35_LABEL,
        show_sig=True,
    )

    plot_violin(
        ax[0, 1],
        p37_corr_labels,
        p37_corr,
        "Zero lag correlation",
        P37_LABEL,
        show_sig=False,
    )

    plot_violin(
        ax[1, 1],
        p37_lag_labels,
        p37_lag,
        "Lag at peak correlation (s)",
        P37_LABEL,
        show_sig=True,
    )

    add_panel(ax[0, 0], "A")
    add_panel(ax[0, 1], "B")
    add_panel(ax[1, 0], "C")
    add_panel(ax[1, 1], "D")

    fig.savefig("intercluster_final_figure.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()