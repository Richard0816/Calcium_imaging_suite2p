import os
import matplotlib.pyplot as plt
import utils
from typing import Union

class SpatialHeatmapConfig:
    """Configuration parameters for spatial heatmap generation."""

    def __init__(self, folder_name, metric='event_rate', prefix='r0p7_',
                 fps=30.0, z_enter=3.5, z_exit=1.5, min_sep_s=0.3, bin_seconds=None):
        self.folder_name = folder_name
        self.metric = metric
        self.prefix = prefix
        self.fps = fps
        self.z_enter = z_enter
        self.z_exit = z_exit
        self.min_sep_s = min_sep_s
        self.bin_seconds = bin_seconds

        # Derived paths
        self.root = os.path.join(folder_name, "suite2p\\plane0\\")
        self.sample_name = folder_name.split("\\")[-4] if "\\" in folder_name else folder_name.split("/")[-4]

    def get_metric_title(self):
        """Generate title based on metric type."""
        titles = {
            'event_rate': f'Event rate (events/min) — z_enter={self.z_enter}, z_exit={self.z_exit} ({self.sample_name})',
            'mean_dff': f'Mean ΔF/F (low-pass) ({self.sample_name})',
            'peak_dz': f'Peak derivative z (robust) ({self.sample_name})'
        }
        return titles[self.metric]

# ---- Cell masking ----
def _safe_div(x, d):
    d = float(d) if d else 1.0
    return x / d

def soft_cell_mask(scores, score_threshold=0.5, top_k_pct=None):
    """
    Convert probabilities into a boolean mask.
    If top_k_pct is set (e.g., 20 for top 20%), it overrides score_threshold.
    """
    if top_k_pct is not None:
        k = max(1, int(np.ceil(scores.size * (top_k_pct / 100.0))))
        thresh = np.partition(scores, -k)[-k]  # kth largest as cutoff
        return scores >= thresh
    return scores >= score_threshold

import csv
import numpy as np

# ---------- Feature building (same metrics you already compute) ----------

def build_feature_matrix(data, config, t_slice=None):
    """
    Returns X of shape (N, 3): [event_rate, peak_dz, pixel_area].
    """
    signals = {'low': data['low'], 'dt': data['dt']}
    time_slice = t_slice if t_slice is not None else slice(None)

    er = utils.roi_metric(
        signals, which='event_rate', t_slice=time_slice,
        fps=config.fps, z_enter=config.z_enter, z_exit=config.z_exit,
        min_sep_s=config.min_sep_s
    )  # typically events/min

    pz = utils.roi_metric(
        signals, which='peak_dz', t_slice=time_slice,
        fps=config.fps, z_enter=config.z_enter, z_exit=config.z_exit,
        min_sep_s=config.min_sep_s
    )

    area = np.array([s['npix'] for s in data['stat']], dtype=float)
    X = np.column_stack([er, pz, area])
    return X


# ---------- Label loading ----------

def load_labels_from_csv(csv_path, n_rois):
    """
    CSV format: roi,label  (0/1). Extra columns are ignored.
    Missing ROIs are treated as unlabeled and excluded automatically.
    """
    y = np.full(n_rois, fill_value=np.nan, dtype=float)
    with open(csv_path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            roi = int(row['roi'])
            lab = float(row['label'])
            if 0 <= roi < n_rois:
                y[roi] = lab
    keep = ~np.isnan(y)
    return y[keep].astype(int), keep  # y_labeled, mask


def load_labels_from_suite2p(root, n_rois):
    """
    Uses suite2p iscell.npy: first column is iscell (0/1).
    """
    iscell_path = os.path.join(root, 'iscell.npy')
    if not os.path.exists(iscell_path):
        raise FileNotFoundError(f"iscell not found: {iscell_path}")
    arr = np.load(iscell_path, allow_pickle=True)
    # suite2p formats vary; handle [N,2] or [N] gracefully
    if arr.ndim == 2 and arr.shape[1] >= 1:
        y = arr[:, 0]
    else:
        y = arr.reshape(-1)
    y = (y > 0).astype(int)
    if y.size != n_rois:
        raise ValueError(f"iscell size {y.size} != N={n_rois}")
    keep = np.ones(n_rois, dtype=bool)  # all labeled
    return y, keep


# ---------- Standardization & design matrix ----------

def standardize_X(X):
    """
    Z-score standardization per column; returns Xz, (mean, std) for inverse use.
    """
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    return Xz, mu, sd


def add_bias(X):
    """Add bias column of ones."""
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


# ---------- Logistic regression (L2-regularized, Newton’s method) ----------

def fit_logistic_l2(X, y, reg=1e-3, max_iter=100, tol=1e-6):
    """
    Fits w (including bias if X already has bias column).
    X: (M, D), y: (M,)
    Uses Newton-Raphson with L2 ridge (except on bias term).
    """
    M, D = X.shape
    w = np.zeros(D)

    for _ in range(max_iter):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))

        # Gradient and Hessian
        g = X.T @ (p - y) / M
        H = (X.T * (p * (1 - p))).dot(X) / M

        # L2 on non-bias terms
        I = np.eye(D)
        I[0, 0] = 0.0
        g += reg * (I @ w)
        H += reg * I

        # Newton step
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # fall back to gradient step if singular
            step = g

        w_new = w - step
        if np.linalg.norm(w_new - w) < tol * (1.0 + np.linalg.norm(w)):
            w = w_new
            break
        w = w_new
    return w


def predict_proba(X, w):
    z = X @ w
    return 1.0 / (1.0 + np.exp(-z))


# ---------- Threshold selection & metrics ----------

def confusion_at_threshold(y_true, scores, thr):
    y_pred = (scores >= thr).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def metrics_from_confusion(tp, fp, tn, fn):
    eps = 1e-12
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    acc  = (tp + tn) / (tp + tn + fp + fn + eps)
    return dict(precision=prec, recall=rec, f1=f1, accuracy=acc)


def find_best_threshold(y_true, scores, strategy='f1'):
    """
    strategy: 'f1' (max F1) or 'youden' (max sensitivity+specificity-1)
    """
    uniq = np.unique(scores)
    if uniq.size > 512:
        # sample thresholds for speed
        ts = np.quantile(scores, np.linspace(0.0, 1.0, 256))
    else:
        ts = uniq

    best = None
    best_thr = 0.5
    for thr in ts:
        tp, fp, tn, fn = confusion_at_threshold(y_true, scores, thr)
        sens = tp / max(1, (tp + fn))
        spec = tn / max(1, (tn + fp))
        f1   = metrics_from_confusion(tp, fp, tn, fn)['f1']
        score = f1 if strategy == 'f1' else (sens + spec - 1.0)
        if (best is None) or (score > best):
            best = score
            best_thr = float(thr)
    return best_thr

def fit_cell_scoring_from_labels(data, config,
                                 label_source='suite2p',  # 'suite2p' or path to CSV
                                 t_slice=None,
                                 reg=1e-3, strategy='f1'):
    """
    Trains a logistic scorer from labeled ROIs and prints a report.
    Returns: dict with weights, means/stds for features, chosen threshold, and metrics.
    """
    N = data['N']
    X = build_feature_matrix(data, config, t_slice=t_slice)

    # Load labels
    if label_source == 'suite2p':
        y, keep = load_labels_from_suite2p(config.root, N)
    else:
        y, keep = load_labels_from_csv(label_source, N)

    X = X[keep]
    y = y.astype(int)

    # Standardize + add bias
    Xz, mu, sd = standardize_X(X)
    Xd = add_bias(Xz)  # bias in column 0

    # Fit logistic (L2)
    w = fit_logistic_l2(Xd, y, reg=reg)

    # Scores and threshold
    scores = predict_proba(Xd, w)
    thr = find_best_threshold(y, scores, strategy=strategy)

    # Confusion + metrics
    tp, fp, tn, fn = confusion_at_threshold(y, scores, thr)
    m = metrics_from_confusion(tp, fp, tn, fn)

    # Pretty print
    print("\n=== Logistic cell scorer (features: [event_rate, peak_dz, pixel_area]) ===")
    print(f"Used labels: {y.size} ROIs  (positives={y.sum()}, negatives={(y==0).sum()})")
    print(f"Weights (bias, er, pz, area): {w.round(4).tolist()}")
    print(f"Standardization mu: {mu.round(3).tolist()}")
    print(f"Standardization sd: {sd.round(3).tolist()}")
    print(f"Chosen threshold ({strategy}): {thr:.3f}")
    print(f"Confusion @ thr: TP={tp} FP={fp} TN={tn} FN={fn}")
    print("Metrics @ thr:", {k: round(v, 3) for k, v in m.items()})
    print("Note: weights are on standardized features.")

    return dict(
        weights=w, mu=mu.reshape(-1), sd=sd.reshape(-1),
        threshold=thr, metrics=m, keep_mask=keep, scores=scores, y=y
    )
def apply_fitted_scorer(X_raw, fitdict):
    """
    X_raw: (N,3) with columns [event_rate, peak_dz, pixel_area]
    fitdict: output of fit_cell_scoring_from_labels
    Returns: scores in [0,1]
    """
    mu = fitdict['mu']
    sd = fitdict['sd']
    w  = fitdict['weights']
    Xz = (X_raw - mu) / sd
    Xd = add_bias(Xz)
    return predict_proba(Xd, w)

def _load_suite2p_data(config):
    """Load Suite2p metadata and processed signals."""
    ops = np.load(os.path.join(config.root, 'ops.npy'), allow_pickle=True).item()
    stat = np.load(os.path.join(config.root, 'stat.npy'), allow_pickle=True)

    Ly, Lx = ops['Ly'], ops['Lx']
    pix_to_um = ops.get('pix_to_um', None)

    # Load memmaps
    low = np.memmap(os.path.join(config.root, f'{config.prefix}dff_lowpass.memmap.float32'),
                    dtype='float32', mode='r')
    dt = np.memmap(os.path.join(config.root, f'{config.prefix}dff_dt.memmap.float32'),
                   dtype='float32', mode='r')

    # Reshape to (T, N)
    N = len(stat)
    T = low.size // N
    low = low.reshape(T, N)
    dt = dt.reshape(T, N)

    return {
        'stat': stat,
        'Ly': Ly,
        'Lx': Lx,
        'pix_to_um': pix_to_um,
        'low': low,
        'dt': dt,
        'T': T,
        'N': N
    }

def show_spatial(img, title, Lx, Ly, stat, pix_to_um=None, cmap='magma', outpath=None, ):
    """
    Display/save a spatial scalar map with optional µm axes and ROI centroid overlay.
    """
    extent = None
    xlabel, ylabel = 'X (pixels)', 'Y (pixels)'
    if pix_to_um is not None:
        extent = [0, Lx * pix_to_um, 0, Ly * pix_to_um]
        xlabel, ylabel = 'X (µm)', 'Y (µm)'

    plt.figure(figsize=(8, 7))
    im = plt.imshow(img, origin='lower', cmap=cmap, extent=extent, aspect='equal')
    # Light overlay of ROI centroids (helps sanity-check registration)
    xs = [np.median(s['xpix']) for s in stat]
    ys = [np.median(s['ypix']) for s in stat]
    if pix_to_um is not None:
        xs = np.array(xs) * pix_to_um
        ys = np.array(ys) * pix_to_um
    plt.scatter(xs, ys, s=4, c='white', alpha=0.35, linewidths=0)
    plt.colorbar(im, label=title)
    plt.title(title)
    plt.xlabel(xlabel);
    plt.ylabel(ylabel)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close()
        print("Saved", outpath)
    else:
        plt.show()

def _compute_and_save_spatial_map(data, config, t_slice=None, bin_index=None,
                                  scores: Union[np.ndarray, None] = None,
                                  score_threshold: float = 0.5,
                                  top_k_pct: Union[float, None] = None):
    """Compute metric values and generate spatial heatmap."""
    signals = {'low': data['low'], 'dt': data['dt']}
    time_slice = t_slice if t_slice is not None else slice(None)

    vals = utils.roi_metric(signals, which=config.metric, t_slice=time_slice,
                            fps=config.fps, z_enter=config.z_enter,
                            z_exit=config.z_exit, min_sep_s=config.min_sep_s)

    spatial = utils.paint_spatial(vals, data['stat'], data['Ly'], data['Lx'])

    # Generate output path and title
    if bin_index is None:
        out = os.path.join(config.root, f'{config.prefix}spatial_{config.metric}.png')
        title = config.get_metric_title()
    else:
        out = os.path.join(config.root, f'{config.prefix}spatial_{config.metric}_bin{bin_index:03d}.png')
        t0, t1 = t_slice.start, t_slice.stop
        title = f'{config.get_metric_title()}\nWindow {bin_index}: {t0 / config.fps:.1f}–{t1 / config.fps:.1f} s'

    show_spatial(spatial, title, data['Lx'], data['Ly'], data['stat'],
                 pix_to_um=data['pix_to_um'], cmap='magma', outpath=out)

    # 2) Probability-driven maps
    if scores is not None:
        # ROI-wise -> pixel map of probabilities
        spatial_prob = utils.paint_spatial(scores, data['stat'], data['Ly'], data['Lx'])
        show_spatial(spatial_prob, "Cell-likeness probability", data['Lx'], data['Ly'], data['stat'],
                     pix_to_um=data['pix_to_um'], cmap='magma', outpath=out + '_prob.png')

        # Soft mask from scores
        mask = soft_cell_mask(scores, score_threshold=score_threshold, top_k_pct=top_k_pct)

        # Masked metric
        vals_masked = np.where(mask, vals, np.nan)
        spatial_masked = utils.paint_spatial(vals_masked, data['stat'], data['Ly'], data['Lx'])
        show_spatial(spatial_masked, title + " (prob-masked)", data['Lx'], data['Ly'], data['stat'],
                     pix_to_um=data['pix_to_um'], cmap='magma', outpath=out + '_probmask.png')

# 1) Load data as usual
config = SpatialHeatmapConfig(
    folder_name=r'D:\data\2p_shifted\2024-06-03_00007',
    metric='event_rate', fps=30.0, z_enter=3.5, z_exit=1.5, min_sep_s=0.3
)
data = _load_suite2p_data(config)

# 2) Fit on existing labels
fit = fit_cell_scoring_from_labels(
    data, config,
    label_source=r'D:\data\2p_shifted\2024-06-03_00007\suite2p\plane0\criteria.csv',   # or provide CSV path e.g. r'D:\labels.csv'
    t_slice=None,             # or a slice to train on a time window
    reg=1e-3, strategy='f1'   # L2 strength, threshold strategy
)

# 3) Score *all* ROIs on the full recording and export maps
X_all = build_feature_matrix(data, config, t_slice=None)
scores_all = apply_fitted_scorer(X_all, fit)

# Reuse your map exporter with probability overlays
_compute_and_save_spatial_map(
    data, config,
    scores=scores_all,
    score_threshold=fit['threshold'],   # threshold chosen on training labels
    top_k_pct=None
)
