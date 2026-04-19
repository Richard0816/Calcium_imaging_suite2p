"""
Dataset builder for the cell filter.

One sample = (spatial_patch, trace, label)
  spatial_patch : (3, H, W)  float32 - [mean, max_proj, roi_mask]
  trace         : (1, T)     float32 - per-ROI z-scored dF/F
  label         : 0 or 1

Recordings are resolved by searching {DATA_ROOT}\\Cx\\<rec_id> and
{DATA_ROOT}\\Hip\\<rec_id>. Per-recording tensors (mean, max, normalized
traces, stat) are cached so repeated ROIs from the same recording don't
reload from disk.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# allow `import utils` from the parent directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils  # noqa: E402

from . import config as C


# ---------------- helpers ----------------

def find_recording_root(rec_id: str, data_root: Path = C.DATA_ROOT) -> Path:
    """Return DATA_ROOT\\<region>\\<rec_id>. Raises if not found."""
    for region in ("Cx", "Hip"):
        cand = data_root / region / rec_id
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Recording not found in Cx/ or Hip/: {rec_id}")


def _znorm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = x.mean()
    s = x.std()
    return (x - m) / max(s, eps)


def _pad_to_patch(img: np.ndarray, cy: int, cx: int, size: int) -> tuple[np.ndarray, int, int]:
    """
    Crop a (size, size) patch from `img` centered at (cy, cx), zero-padding
    when the window falls outside the image bounds. Returns (patch, y0, x0)
    where (y0, x0) is the top-left corner in image coordinates.
    """
    h, w = img.shape
    half = size // 2
    y0, x0 = cy - half, cx - half
    y1, x1 = y0 + size, x0 + size

    # compute slices with clipping
    sy0, sy1 = max(0, y0), min(h, y1)
    sx0, sx1 = max(0, x0), min(w, x1)

    patch = np.zeros((size, size), dtype=np.float32)
    py0 = sy0 - y0
    px0 = sx0 - x0
    patch[py0:py0 + (sy1 - sy0), px0:px0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return patch, y0, x0


# ---------------- per-recording cache ----------------

class _RecordingCache:
    """Holds lazily-loaded arrays for a single recording."""
    def __init__(self, rec_id: str):
        self.rec_id = rec_id
        self.root = find_recording_root(rec_id)
        self.plane0 = self.root / "suite2p" / "plane0"

        stat = np.load(self.plane0 / "stat.npy", allow_pickle=True)
        ops = np.load(self.plane0 / "ops.npy", allow_pickle=True).item()

        mean_img = ops.get("meanImgE", None)
        if mean_img is None:
            mean_img = ops.get("meanImg")
        mean_img = np.asarray(mean_img, dtype=np.float32)

        max_img = ops.get("max_proj", None)
        if max_img is None:
            max_img = ops.get("maxImg", None)
        if max_img is None:
            max_img = mean_img
        max_img = np.asarray(max_img, dtype=np.float32)

        # If max_proj is cropped (common with suite2p), pad it back to full FOV
        if max_img.shape != mean_img.shape:
            H, W = mean_img.shape
            y0 = int(ops.get("yrange", [0, H])[0])
            x0 = int(ops.get("xrange", [0, W])[0])
            padded = np.zeros_like(mean_img)
            mh, mw = max_img.shape
            padded[y0:y0 + mh, x0:x0 + mw] = max_img
            max_img = padded

        # dF/F memmap
        dff, _, _, T, N = utils.s2p_open_memmaps(self.plane0, prefix=C.DFF_PREFIX)

        self.stat = stat
        self.mean_img_z = _znorm(mean_img)
        self.max_img_z = _znorm(max_img)
        self.dff = dff
        self.T = T
        self.N = N
        self.H, self.W = mean_img.shape

    def get_patch(self, roi_idx: int, size: int) -> np.ndarray:
        """Return (3, size, size) float32 patch: [mean, max, mask]."""
        s = self.stat[roi_idx]
        xpix = s["xpix"]
        ypix = s["ypix"]
        cy = int(round(float(ypix.mean())))
        cx = int(round(float(xpix.mean())))

        mean_patch, y0, x0 = _pad_to_patch(self.mean_img_z, cy, cx, size)
        max_patch, _, _ = _pad_to_patch(self.max_img_z, cy, cx, size)

        mask_full = np.zeros((self.H, self.W), dtype=np.float32)
        mask_full[ypix, xpix] = 1.0
        mask_patch, _, _ = _pad_to_patch(mask_full, cy, cx, size)

        return np.stack([mean_patch, max_patch, mask_patch], axis=0)

    def get_trace(self, roi_idx: int) -> np.ndarray:
        """Return (T,) float32 per-ROI z-scored trace."""
        trace = np.asarray(self.dff[:, roi_idx], dtype=np.float32)
        return _znorm(trace)


# ---------------- dataset ----------------

class ROIDataset(Dataset):
    """
    Each __getitem__ returns:
      spatial : (3, H, W)
      trace   : (1, T_crop) during training; (1, T_full) during eval
      label   : float tensor, 0.0 or 1.0
    """
    def __init__(
        self,
        labels_df: pd.DataFrame,
        *,
        patch_size: int = C.PATCH_SIZE,
        trace_crop: Optional[int] = C.TRACE_CROP_LEN,
        random_crop: bool = True,
        cache: Optional[dict] = None,
    ):
        self.df = labels_df.reset_index(drop=True)
        self.patch_size = patch_size
        self.trace_crop = trace_crop
        self.random_crop = random_crop
        self._cache = cache if cache is not None else {}

    def _get_rec(self, rec_id: str) -> _RecordingCache:
        if rec_id not in self._cache:
            self._cache[rec_id] = _RecordingCache(rec_id)
        return self._cache[rec_id]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        rec_id = str(row["recording_ID"])
        roi = int(row["ROI_number"])
        label = float(row["user_defined_cell"])

        rec = self._get_rec(rec_id)
        patch = rec.get_patch(roi, self.patch_size)    # (3, H, W)
        trace = rec.get_trace(roi)                     # (T,)

        if self.trace_crop is not None and trace.shape[0] >= self.trace_crop:
            if self.random_crop:
                start = np.random.randint(0, trace.shape[0] - self.trace_crop + 1)
            else:
                start = (trace.shape[0] - self.trace_crop) // 2
            trace = trace[start:start + self.trace_crop]
        elif self.trace_crop is not None and trace.shape[0] < self.trace_crop:
            # pad end with zeros if shorter than crop length
            pad = self.trace_crop - trace.shape[0]
            trace = np.concatenate([trace, np.zeros(pad, dtype=np.float32)])

        return (
            torch.from_numpy(patch),
            torch.from_numpy(trace[None, :]),
            torch.tensor(label, dtype=torch.float32),
        )


# ---------------- splits ----------------

def load_labels(csv_path: Path = C.LABELS_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["recording_ID"] = df["recording_ID"].astype(str)
    df["ROI_number"] = df["ROI_number"].astype(int)
    df["user_defined_cell"] = df["user_defined_cell"].astype(int)
    # drop duplicates, keeping last (so corrections win)
    df = df.drop_duplicates(subset=["recording_ID", "ROI_number"], keep="last")
    return df.reset_index(drop=True)


def split_by_recording(
    df: pd.DataFrame,
    val_frac: float = C.VAL_FRAC,
    seed: int = C.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    recs = np.array(sorted(df["recording_ID"].unique()))
    rng.shuffle(recs)
    n_val = max(1, int(round(len(recs) * val_frac)))
    val_recs = set(recs[:n_val].tolist())
    train_df = df[~df["recording_ID"].isin(val_recs)].reset_index(drop=True)
    val_df = df[df["recording_ID"].isin(val_recs)].reset_index(drop=True)
    return train_df, val_df
