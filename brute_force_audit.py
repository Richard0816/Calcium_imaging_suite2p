"""Generate audit.png for every pass folder under a brute-force sweep root.

Walks ``SAVE_FOLDER`` for every ``suite2p/plane0/`` directory (sparsery
and cellpose outputs both land here) and writes a three-panel audit PNG
next to it:
  1. Mean image with detected ROI contours (green).
  2. Residual image after masking ROI pixels (magma).
  3. Mean image with missed-cell blob candidates (cyan circles).

Ledger/leaderboard files are not modified. Folders without a
``stat.npy`` + ``ops.npy`` pair (e.g. ``_shared_reg/``) are skipped
automatically by the underlying helper.

Usage:
    python brute_force_audit.py                 # audit the default SAVE_FOLDER
    python brute_force_audit.py <some/other>    # audit a different root
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure we import the worktree copy of adaptive_detection.
WORKTREE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKTREE))

from adaptive_detection import AdaptiveConfig, generate_audit_pngs_for_save_folder
from brute_force_ops import SAVE_FOLDER as DEFAULT_SAVE_FOLDER


def main():
    save_folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SAVE_FOLDER
    root = Path(save_folder)
    if not root.exists():
        print(f'[audit] ERROR: {root} does not exist')
        sys.exit(1)

    # Use the same blob-detector defaults as the adaptive pipeline so the
    # "missed cell candidates" panel is meaningful on 2P data.
    config = AdaptiveConfig(
        tiff_folder='',             # unused by audit
        save_folder=str(root),
        verbose=True,
        soma_diameter_px=8.0,
        soma_scale_tolerance=0.7,
        blob_min_contrast=0.04,
        blob_min_area_px=10,
        blob_center_surround_ratio=1.15,
        blob_num_sigma=10,
    )

    print(f'[audit] root = {root}')
    generate_audit_pngs_for_save_folder(str(root), config=config, verbose=True)
    print('[audit] done.')


if __name__ == '__main__':
    main()
