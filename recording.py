"""
recording.py — central descriptor for a single 2-photon recording.

All pipeline modules receive a Recording object so that path
construction and metadata lookup are never duplicated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import utils


class Recording:
    """
    Wraps a single recording directory and exposes all derived paths
    plus metadata (fps, zoom) resolved from the notes spreadsheets.

    Parameters
    ----------
    path : str | Path
        Full path to the recording root, e.g.
        ``E:/data/2p_shifted/Cx/2024-07-01_00018``
    prefix : str
        Memmap / output prefix used throughout the pipeline.
        Defaults to ``"r0p7_filtered_"``.
    notes_root : str | Path
        Root directory of the Excel metadata sheets.
        Defaults to ``F:/notes_recordings``.
    """

    DEFAULT_PREFIX = "r0p7_filtered_"
    DEFAULT_NOTES_ROOT = r"F:\notes_recordings"

    def __init__(
        self,
        path: str | Path,
        prefix: str = DEFAULT_PREFIX,
        notes_root: str = DEFAULT_NOTES_ROOT,
    ) -> None:
        self.path = Path(path).resolve()
        self.prefix = prefix
        self._notes_root = notes_root

        if not self.path.exists():
            raise FileNotFoundError(f"Recording path does not exist: {self.path}")

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Folder name, e.g. ``2024-07-01_00018``."""
        return self.path.name

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"Recording({self.path!r})"

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @property
    def plane0(self) -> Path:
        """``<recording>/suite2p/plane0``"""
        return self.path / "suite2p" / "plane0"

    @property
    def cluster_dir(self) -> Path:
        """``<plane0>/<prefix>cluster_results``"""
        return self.plane0 / f"{self.prefix}cluster_results"

    @property
    def cell_mask_path(self) -> Path:
        return self.plane0 / "r0p7_cell_mask_bool.npy"

    @property
    def predicted_prob_path(self) -> Path:
        return self.plane0 / "predicted_cell_prob.npy"

    def plane0_file(self, name: str) -> Path:
        """Convenience: ``<plane0>/<name>``."""
        return self.plane0 / name

    # ------------------------------------------------------------------
    # Metadata (lazily resolved)
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        """Frames per second from notes spreadsheet (or 15.0 as fallback)."""
        return utils.get_fps_from_notes(str(self.path), notes_root=self._notes_root)

    @property
    def zoom(self) -> float:
        """Zoom factor from notes spreadsheet (or 1.0 as fallback)."""
        return utils.get_zoom_from_notes(str(self.path), notes_root=self._notes_root)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def n_clusters(self) -> int:
        """Number of cluster ROI files (``*_rois.npy``) in cluster_dir."""
        if not self.cluster_dir.exists():
            return 0
        return len(list(self.cluster_dir.glob("*_rois.npy")))

    def has_processed_traces(self) -> bool:
        """True when all three processed memmap files exist."""
        suffixes = [
            f"{self.prefix}dff.memmap.float32",
            f"{self.prefix}dff_lowpass.memmap.float32",
            f"{self.prefix}dff_dt.memmap.float32",
        ]
        return all((self.plane0 / s).exists() for s in suffixes)

    def has_clustering(self) -> bool:
        return self.cluster_dir.exists() and self.n_clusters() > 0
