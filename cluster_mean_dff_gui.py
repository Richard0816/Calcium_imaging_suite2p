"""
cluster_mean_dff_gui.py - Standalone GUI.

Pick a cluster_results folder (e.g. `...\\r0p7_filtered_cluster_results`), choose a
cluster from the dropdown, preview it in the Fig7_1 style, and on Calculate save:
  - <cluster>_mean_dff.png   : mean dF/F trace over time
  - <cluster>_mean_dff.csv   : time_s, mean_dff columns

Assumes the layout produced by the pipeline, i.e. the cluster folder sits next to
the dF/F memmaps inside a suite2p plane directory and is named `<prefix>cluster_results`.
"""

from __future__ import annotations

import csv
import gc
import shutil
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

CLUSTER_RESULTS_SUFFIX = "_cluster_results"
DEFAULT_FPS = 15.07


def _load_cluster_rois(roi_file: Path, max_roi: int) -> np.ndarray:
    rois = np.load(roi_file)
    rois = np.asarray(rois).astype(int).ravel()
    return rois[(rois >= 0) & (rois < max_roi)]


def infer_prefix(cluster_folder: Path) -> str:
    name = cluster_folder.name
    if not name.endswith(CLUSTER_RESULTS_SUFFIX):
        raise ValueError(
            f"Expected folder ending with '{CLUSTER_RESULTS_SUFFIX}', got '{name}'."
        )
    return name[: -len("cluster_results")]


def list_cluster_roi_files(cluster_folder: Path) -> list[Path]:
    return sorted(p for p in cluster_folder.glob("*_rois.npy") if p.is_file())


def cluster_label_from_file(roi_file: Path) -> str:
    return roi_file.stem.removesuffix("_rois")


def build_cluster_heatmap(
    lowpass: np.memmap,
    rois: np.ndarray,
    num_frames: int,
    time_cols_target: int = 1200,
) -> tuple[np.ndarray, int]:
    """Bulk-read the lowpass dF/F for `rois`, downsample in time, and per-row
    normalize to p1-p99. Returns (heatmap_uint8, num_cols)."""
    if rois.size == 0 or num_frames == 0:
        return np.zeros((0, 0), dtype=np.uint8), 0

    dsf = max(1, num_frames // time_cols_target)
    num_cols = max(1, num_frames // dsf)
    used_frames = num_cols * dsf

    data = np.asarray(lowpass[:used_frames, rois], dtype=np.float32).T

    if dsf > 1:
        data = data.reshape(rois.size, num_cols, dsf).mean(axis=2)

    p1 = np.percentile(data, 1, axis=1, keepdims=True)
    p99 = np.percentile(data, 99, axis=1, keepdims=True)
    denom = np.where(p99 > p1, p99 - p1, 1.0)
    norm = np.clip((data - p1) / denom, 0.0, 1.0)
    heatmap = (norm * 255.0 + 0.5).astype(np.uint8)
    return heatmap, num_cols


def open_memmaps_fast(
    recording_root: Path, prefix: str
) -> tuple[np.memmap, np.memmap, np.memmap, int, int]:
    """Lightweight stand-in for utils.s2p_open_memmaps that doesn't fully load F.npy.

    Reads num_frames/num_rois from F.npy's header via mmap, and (for *_filtered_*
    prefixes) trims by r0p7_cell_mask_bool.npy. Returns (dff, lowpass, dt, T, N)
    in (T, N) shape, matching utils.s2p_open_memmaps.
    """
    F_header = np.load(recording_root / "F.npy", mmap_mode="r")
    if F_header.ndim != 2:
        raise ValueError(f"Expected 2D F.npy, got shape {F_header.shape}")
    num_rois_total, num_frames = F_header.shape
    del F_header

    parts = prefix.split("_")
    is_filtered = len(parts) >= 3 and parts[-2] == "filtered"
    if is_filtered:
        mask = np.load(recording_root / "r0p7_cell_mask_bool.npy", allow_pickle=False)
        num_rois = int(np.asarray(mask).astype(bool).sum())
    else:
        num_rois = int(num_rois_total)

    shape = (num_frames, num_rois)
    dff = np.memmap(
        recording_root / f"{prefix}dff.memmap.float32", dtype="float32", mode="r", shape=shape
    )
    low = np.memmap(
        recording_root / f"{prefix}dff_lowpass.memmap.float32",
        dtype="float32",
        mode="r",
        shape=shape,
    )
    dt = np.memmap(
        recording_root / f"{prefix}dff_dt.memmap.float32",
        dtype="float32",
        mode="r",
        shape=shape,
    )
    return dff, low, dt, int(num_frames), int(num_rois)


def resolve_predicted_mask_in_memmap_space(
    recording_root: Path, prefix: str, num_rois: int
) -> tuple[Optional[np.ndarray], str]:
    """Return (mask, message). mask is a bool array of length num_rois selecting
    predicted cells in memmap-ROI space, or None if it can't be resolved."""
    pred_path = recording_root / "predicted_cell_mask.npy"
    if not pred_path.exists():
        return None, f"predicted_cell_mask.npy not found in {recording_root}"

    predicted_full = np.load(pred_path).astype(bool).ravel()

    parts = prefix.split("_")
    is_filtered = len(parts) >= 3 and parts[-2] == "filtered"

    if is_filtered:
        r0p7_path = recording_root / "r0p7_cell_mask_bool.npy"
        if not r0p7_path.exists():
            return None, "r0p7_cell_mask_bool.npy not found; cannot map predicted mask into filtered space"
        r0p7_mask = np.load(r0p7_path).astype(bool).ravel()
        if predicted_full.shape != r0p7_mask.shape:
            return None, (
                f"predicted_cell_mask length {predicted_full.size} != "
                f"r0p7_cell_mask length {r0p7_mask.size}"
            )
        mask = predicted_full[r0p7_mask]
    else:
        mask = predicted_full

    if mask.size != num_rois:
        return None, f"predicted mask length {mask.size} != memmap num_rois {num_rois}"

    return mask, f"predicted_cell_mask applied: {int(mask.sum())}/{mask.size} ROIs kept"


class ClusterMeanDffGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Cluster mean dF/F")
        self.geometry("1100x780")

        self.cluster_folder: Optional[Path] = None
        self.recording_root: Optional[Path] = None
        self.prefix: Optional[str] = None
        self.dff: Optional[np.memmap] = None
        self.lowpass: Optional[np.memmap] = None
        self.derivative: Optional[np.memmap] = None
        self.num_frames: int = 0
        self.num_rois: int = 0
        self.num_rois_orig: int = 0
        self.fps: float = 0.0
        self.roi_files_by_label: dict[str, Path] = {}
        self.cell_mask: Optional[np.ndarray] = None
        self._pred_index_map: Optional[np.ndarray] = None
        self._temp_dir: Optional[Path] = None
        self._busy = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----- UI -------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        folder_row = ttk.Frame(outer)
        folder_row.pack(fill="x", pady=(0, 6))
        ttk.Label(folder_row, text="Cluster folder:").pack(side="left")
        self.folder_var = tk.StringVar()
        ttk.Entry(folder_row, textvariable=self.folder_var).pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(folder_row, text="Browse...", command=self._on_browse).pack(side="left")

        info_row = ttk.Frame(outer)
        info_row.pack(fill="x", pady=(0, 6))
        self.info_var = tk.StringVar(value="No folder loaded.")
        ttk.Label(info_row, textvariable=self.info_var, foreground="#555").pack(side="left")

        select_row = ttk.Frame(outer)
        select_row.pack(fill="x", pady=(0, 6))

        list_frame = ttk.LabelFrame(
            select_row, text="Clusters (Ctrl/Shift-click for multi-select)", padding=4
        )
        list_frame.pack(side="left", fill="y")
        self.cluster_list = tk.Listbox(
            list_frame, selectmode="extended", height=8, width=22, exportselection=False
        )
        self.cluster_list.pack(side="left", fill="y")
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.cluster_list.yview)
        list_scroll.pack(side="left", fill="y")
        self.cluster_list.configure(yscrollcommand=list_scroll.set)

        right_col = ttk.Frame(select_row)
        right_col.pack(side="left", fill="x", expand=True, padx=(12, 0))

        output_row = ttk.Frame(right_col)
        output_row.pack(fill="x")
        ttk.Label(output_row, text="Output folder:").pack(side="left")
        self.output_var = tk.StringVar()
        ttk.Entry(output_row, textvariable=self.output_var, width=50).pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(output_row, text="Browse...", command=self._on_browse_output).pack(side="left")

        btn_row = ttk.Frame(right_col)
        btn_row.pack(fill="x", pady=(8, 0))
        ttk.Button(btn_row, text="Select all", command=self._select_all).pack(side="left")
        ttk.Button(btn_row, text="Clear", command=self._select_clear).pack(side="left", padx=6)
        self.preview_btn = ttk.Button(btn_row, text="Refresh preview", command=self._on_preview)
        self.preview_btn.pack(side="left", padx=(12, 0))
        self.calc_btn = ttk.Button(
            btn_row, text="Calculate (save PNG + CSV)", command=self._on_calculate
        )
        self.calc_btn.pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="")
        ttk.Label(right_col, textvariable=self.status_var, foreground="#0a6").pack(
            anchor="w", pady=(6, 0)
        )

        preview_frame = ttk.LabelFrame(outer, text="Preview (Fig 7_1 style)", padding=6)
        preview_frame.pack(fill="both", expand=True)
        self.preview_fig = Figure(figsize=(10, 5), dpi=100)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=preview_frame)
        self.preview_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._clear_preview("Pick a cluster folder and select a cluster to preview.")

    # ----- Callbacks -------------------------------------------------------
    def _on_browse(self) -> None:
        path = filedialog.askdirectory(title="Select cluster_results folder")
        if not path:
            return
        try:
            self._load_folder(Path(path))
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _on_browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

    def _selected_labels(self) -> list[str]:
        return [self.cluster_list.get(i) for i in self.cluster_list.curselection()]

    def _select_all(self) -> None:
        self.cluster_list.selection_set(0, tk.END)

    def _select_clear(self) -> None:
        self.cluster_list.selection_clear(0, tk.END)

    def _on_preview(self) -> None:
        if self._busy or not self.roi_files_by_label:
            return
        self._run_async(self._render_preview)

    def _on_calculate(self) -> None:
        labels = self._selected_labels()
        if not labels:
            messagebox.showwarning("No cluster", "Select at least one cluster.")
            return
        if self._busy:
            return
        self._run_async(self._do_calculate, labels)

    # ----- Actions ---------------------------------------------------------
    def _load_folder(self, cluster_folder: Path) -> None:
        prefix = infer_prefix(cluster_folder)
        recording_root = cluster_folder.parent

        roi_files = list_cluster_roi_files(cluster_folder)
        if not roi_files:
            raise ValueError(f"No *_rois.npy files found in {cluster_folder}")

        self.info_var.set("Loading memmaps...")
        self.update_idletasks()

        self._cleanup_temp()

        dff, lowpass, _derivative, num_frames, num_rois = open_memmaps_fast(
            recording_root, prefix
        )
        fps = DEFAULT_FPS

        self.cluster_folder = cluster_folder
        self.recording_root = recording_root
        self.prefix = prefix
        self.dff = dff
        self.lowpass = lowpass
        self.derivative = None
        self.num_frames = int(num_frames)
        self.num_rois = int(num_rois)
        self.num_rois_orig = int(num_rois)
        self.fps = float(fps)

        cell_mask, mask_msg = resolve_predicted_mask_in_memmap_space(
            recording_root, prefix, self.num_rois_orig
        )
        self.cell_mask = cell_mask
        print(mask_msg)

        if cell_mask is not None:
            self.info_var.set("Building pred-cell temp memmaps...")
            self.update_idletasks()
            self._build_temp_memmaps()

        self.roi_files_by_label = {cluster_label_from_file(p): p for p in roi_files}
        labels = list(self.roi_files_by_label.keys())
        self.cluster_list.delete(0, tk.END)
        for lbl in labels:
            self.cluster_list.insert(tk.END, lbl)
        if labels:
            self.cluster_list.selection_set(0)

        self.folder_var.set(str(cluster_folder))
        if not self.output_var.get():
            self.output_var.set(str(cluster_folder))
        if self.cell_mask is not None:
            mask_info = f"  pred_cells={int(self.cell_mask.sum())}/{self.cell_mask.size}"
        else:
            mask_info = "  pred_mask=disabled"
        self.info_var.set(
            f"prefix='{prefix}'  recording='{recording_root.name}'  "
            f"frames={self.num_frames}  rois={self.num_rois}  fps={self.fps:g}{mask_info}"
        )

        if labels:
            self._on_preview()

    def _build_temp_memmaps(self) -> None:
        """Materialize dff and lowpass restricted to predicted-cell columns
        and route preview/calculate through them."""
        assert self.cell_mask is not None
        pred_cols = np.flatnonzero(self.cell_mask).astype(np.int64)
        n_pred = int(pred_cols.size)

        self._temp_dir = Path(tempfile.mkdtemp(prefix="cluster_gui_"))
        dff_path = self._temp_dir / "dff_pred.memmap.float32"
        low_path = self._temp_dir / "lowpass_pred.memmap.float32"

        new_dff = np.memmap(
            dff_path, dtype="float32", mode="w+", shape=(self.num_frames, n_pred)
        )
        new_low = np.memmap(
            low_path, dtype="float32", mode="w+", shape=(self.num_frames, n_pred)
        )
        chunk = 4096
        for start in range(0, self.num_frames, chunk):
            end = min(start + chunk, self.num_frames)
            new_dff[start:end] = self.dff[start:end, pred_cols]
            new_low[start:end] = self.lowpass[start:end, pred_cols]
        new_dff.flush()
        new_low.flush()
        del new_dff, new_low

        # Drop original memmap handles, then re-open temp files read-only.
        self.dff = None
        self.lowpass = None
        gc.collect()

        self.dff = np.memmap(
            dff_path, dtype="float32", mode="r", shape=(self.num_frames, n_pred)
        )
        self.lowpass = np.memmap(
            low_path, dtype="float32", mode="r", shape=(self.num_frames, n_pred)
        )

        index_map = np.full(self.num_rois_orig, -1, dtype=np.int64)
        index_map[pred_cols] = np.arange(n_pred, dtype=np.int64)
        self._pred_index_map = index_map
        self.num_rois = n_pred

    def _cleanup_temp(self) -> None:
        self.dff = None
        self.lowpass = None
        self.derivative = None
        self._pred_index_map = None
        gc.collect()
        if self._temp_dir is not None and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except OSError as e:
                print(f"warn: could not remove {self._temp_dir}: {e}")
        self._temp_dir = None

    def _on_close(self) -> None:
        self._cleanup_temp()
        self.destroy()

    def _load_filtered_cluster_rois(self, label: str) -> np.ndarray:
        rois = _load_cluster_rois(self.roi_files_by_label[label], self.num_rois_orig)
        if rois.size and self._pred_index_map is not None:
            mapped = self._pred_index_map[rois]
            rois = mapped[mapped >= 0].astype(np.int64)
        return rois

    def _combined_rois(self, labels: list[str]) -> np.ndarray:
        parts = [self._load_filtered_cluster_rois(lbl) for lbl in labels]
        if not parts:
            return np.empty(0, dtype=int)
        return np.unique(np.concatenate(parts))

    def _combined_label(self, labels: list[str]) -> str:
        return "+".join(labels)

    def _render_preview(self) -> None:
        labels = list(self.roi_files_by_label.keys())
        if not labels:
            self.after(0, lambda: self._clear_preview("No clusters to preview."))
            return

        duration_s = self.num_frames / self.fps if self.fps else 0.0
        rows = []
        for label in labels:
            rois = self._load_filtered_cluster_rois(label)
            if rois.size == 0:
                continue
            heatmap, _ = build_cluster_heatmap(self.lowpass, rois, self.num_frames)
            rows.append({
                "label": label,
                "heatmap": heatmap,
                "duration_s": duration_s,
                "n_rois": int(rois.size),
            })

        if not rows:
            self.after(0, lambda: self._clear_preview("No valid ROIs in any cluster."))
            return
        self.after(0, self._draw_preview, rows)

    def _draw_preview(self, rows: list[dict]) -> None:
        self.preview_fig.clear()
        n = len(rows)
        panel_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for r, row in enumerate(rows):
            ax = self.preview_fig.add_subplot(n, 1, r + 1)
            im = ax.imshow(
                row["heatmap"],
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                extent=[0, row["duration_s"], row["n_rois"], 0],
            )
            ax.set_title(
                f"{panel_letters[r]}  {row['label']} heatmap  (n={row['n_rois']})",
                fontsize=10,
            )
            ax.set_ylabel("ROIs")
            if r == n - 1:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticklabels([])
            self.preview_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
                "Rel. ΔF/F"
            )

        self.preview_fig.set_size_inches(10, max(3.0, 2.2 * n))
        self.preview_fig.tight_layout()
        self.preview_canvas.draw_idle()

    def _do_calculate(self, labels: list[str]) -> None:
        combo_label = self._combined_label(labels)
        rois = self._combined_rois(labels)
        if rois.size == 0:
            self.after(
                0, lambda: messagebox.showwarning("Empty", f"{combo_label} has no valid ROIs.")
            )
            return

        traces = np.asarray(self.dff[:, rois], dtype=np.float32)
        mean_dff = traces.mean(axis=1)
        time_s = np.arange(self.num_frames, dtype=np.float64) / self.fps

        out_dir = Path(self.output_var.get() or self.cluster_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / f"{combo_label}_mean_dff.png"
        csv_path = out_dir / f"{combo_label}_mean_dff.csv"

        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_s", "mean_dff"])
            writer.writerows(zip(time_s.tolist(), mean_dff.tolist()))

        fig = Figure(figsize=(10, 3.5), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time_s, mean_dff, color="#1f77b4", linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean ΔF/F")
        ax.set_title(
            f"{combo_label} mean ΔF/F  (n={rois.size} ROIs, fps={self.fps:g})"
        )
        ax.margins(x=0)
        fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight")

        self.after(
            0,
            lambda: self.status_var.set(f"Saved {png_path.name} and {csv_path.name} to {out_dir}"),
        )

    # ----- Helpers ---------------------------------------------------------
    def _clear_preview(self, message: str) -> None:
        self.preview_fig.clear()
        ax = self.preview_fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.preview_canvas.draw_idle()

    def _run_async(self, fn, *args) -> None:
        if self._busy:
            return
        self._busy = True
        self.calc_btn.state(["disabled"])
        self.preview_btn.state(["disabled"])
        self.status_var.set("Working...")

        def worker() -> None:
            try:
                fn(*args)
            except Exception as e:
                self.after(0, lambda err=e: messagebox.showerror("Error", str(err)))
            finally:
                self.after(0, self._finish_async)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_async(self) -> None:
        self._busy = False
        self.calc_btn.state(["!disabled"])
        self.preview_btn.state(["!disabled"])
        if self.status_var.get() == "Working...":
            self.status_var.set("")


if __name__ == "__main__":
    app = ClusterMeanDffGui()
    app.mainloop()
