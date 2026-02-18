import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import utils

def load_dff(root: Path, prefix: str):
    plane_dir = root / "suite2p" / "plane0"
    dff, _, _, _, _ = utils.s2p_open_memmaps(plane_dir, prefix)
    return np.asarray(dff)

def build_bin_labels_from_bins_folder(
    coact_dir: Path,
    n_rois: int,
    mode: str = "most",
    roi_file_glob: str = "*.npy",
):
    """
    Expects coact_dir to contain subfolders, one per bin.
    Each bin folder should contain an npy with ROI indices for that bin.
    """
    bin_dirs = [p for p in sorted(coact_dir.iterdir()) if p.is_dir()]
    if not bin_dirs:
        raise FileNotFoundError(f"No bin subfolders found in {coact_dir}")

    # counts[roi, bin]
    counts = np.zeros((n_rois, len(bin_dirs)), dtype=np.int32)

    for b, bdir in enumerate(bin_dirs):
        npy_files = sorted(bdir.glob(roi_file_glob))
        if not npy_files:
            continue

        # Pick the first npy by default. If you have a specific filename, set roi_file_glob.
        roi_idx = np.load(npy_files[0]).astype(int)
        roi_idx = roi_idx[(roi_idx >= 0) & (roi_idx < n_rois)]
        counts[roi_idx, b] += 1

    labels = labels_from_counts(counts, mode=mode)
    bin_names = [d.name for d in bin_dirs]
    return labels, counts, bin_names

def build_bin_labels_from_csv(
    csv_path: Path,
    n_rois: int,
    mode: str = "most",
    bin_col: str = "bin",
    roi_col: str = "roi",
):
    import csv

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if bin_col not in reader.fieldnames or roi_col not in reader.fieldnames:
            raise ValueError(f"CSV must contain columns {bin_col} and {roi_col}")
        for r in reader:
            rows.append((r[bin_col], int(float(r[roi_col]))))

    bin_keys = sorted({bk for bk, _ in rows})
    bin_to_i = {bk: i for i, bk in enumerate(bin_keys)}

    counts = np.zeros((n_rois, len(bin_keys)), dtype=np.int32)
    for bk, roi in rows:
        if 0 <= roi < n_rois:
            counts[roi, bin_to_i[bk]] += 1

    labels = labels_from_counts(counts, mode=mode)
    return labels, counts, [str(bk) for bk in bin_keys]

def labels_from_counts(counts: np.ndarray, mode: str):
    n_rois, n_bins = counts.shape
    labels = np.full(n_rois, -1, dtype=int)

    hits = counts.sum(axis=1)

    if mode == "first":
        for r in range(n_rois):
            if hits[r] == 0:
                continue
            labels[r] = int(np.argmax(counts[r] > 0))
        return labels

    if mode == "most":
        for r in range(n_rois):
            if hits[r] == 0:
                continue
            labels[r] = int(np.argmax(counts[r]))
        return labels

    if mode == "multi":
        for r in range(n_rois):
            if hits[r] == 0:
                continue
            nz = np.flatnonzero(counts[r] > 0)
            labels[r] = int(nz[0]) if nz.size == 1 else -2
        return labels

    raise ValueError("mode must be one of: first, most, multi")

def run_umap(X: np.ndarray, n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0):
    import umap.umap_ as umap

    X_scaled = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    emb = reducer.fit_transform(X_scaled)
    return emb

# --------- user inputs ----------
root = Path(r"F:\data\2p_shifted\Cx\2024-07-01_00018")
prefix_dff = "r0p7_"  # must match ROI index space used for coactivation ROI indices

coact_dir = root / "coactivation"  # change if your folder name differs
mode = "most"  # "first", "most", "multi"
# --------------------------------

X = load_dff(root, prefix_dff)  # (n_rois, n_time)
n_rois = X.shape[0]

# Build labels from coactivation folder
# Option A. Bins are subfolders with an npy inside
labels, counts, bin_names = build_bin_labels_from_bins_folder(
    coact_dir,
    n_rois=n_rois,
    mode=mode,
    roi_file_glob="*rois*.npy",  # adjust if your file names differ
)

print("ROIs:", n_rois)
print("Bins:", len(bin_names))
print("Assigned ROIs:", int(np.sum(labels >= 0)))
print("Unassigned ROIs:", int(np.sum(labels == -1)))

emb = run_umap(X, n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0)

plt.figure()
sc = plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=10, alpha=0.85)
plt.title(f"UMAP labeled by coactivation bin, mode={mode}")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.colorbar(sc, label="bin label, -1 unassigned, -2 multi")
plt.show()
