"""
clustering_tab.py - GUI for cross-correlation hierarchical clustering.

Workflow
--------
1. Pick a Suite2p plane0 folder.
2. Click "Run analysis":
       - Loads dF/F (default prefix r0p7_filtered_).
       - Computes the pairwise correlation-distance linkage
         (1 - Pearson r between every pair of ROIs).
       - Picks a starting cut (auto target 4-5 clusters) and renders
         the dendrogram + cluster-colored spatial map.
3. Adjust the cut:
       - "Manual threshold" toggle ON -> the vertical slider next to
         the dendrogram drives the cut (the dashed line on the
         dendrogram tracks the slider). Spatial map recolors live.
       - Toggle OFF -> auto-threshold (same logic as
         clustering_cmap.auto_choose_threshold).
4. Recolor:
       - Pick a palette from the dropdown (categorical or continuous).
       - Or click "Per-cluster colors..." to define one hex color per
         cluster manually; this overrides the dropdown until reset.
5. "Export *_rois.npy" writes one ROI list per cluster into the
   r0p7_*cluster_results/gui_recluster/ folder so the rest of the
   pipeline (cross-correlation, summaries) can pick them up.
6. "Run cluster x cluster cross-correlation" runs
   crosscorrelation.run_cluster_cross_correlations_gpu against the
   exported clusters.

The whole thing also runs as a standalone window via `python clustering_tab.py`.
"""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)
from scipy.cluster.hierarchy import (
    dendrogram,
    fcluster,
    linkage,
    set_link_color_palette,
)
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clustering_cmap as cmap_mod  # noqa: E402
import summary_writer  # noqa: E402
import utils  # noqa: E402


DEFAULT_PREFIX = "r0p7_filtered_"
DEFAULT_PALETTE = "tab10"
ABOVE_CUT_COLOR = "gray"
EXPORT_SUBDIR = "gui_recluster"
POLL_MS = 80


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _attach_fig_toolbar(canvas: FigureCanvasTkAgg,
                        parent_frame) -> NavigationToolbar2Tk:
    """Attach a matplotlib navigation toolbar (pan / zoom / Save Figure)
    above ``canvas`` inside ``parent_frame``. Caller must pack/grid the
    canvas widget AFTER this call so the toolbar lands above it."""
    tb_frame = ttk.Frame(parent_frame)
    tb_frame.pack(side="top", fill="x")
    tb = NavigationToolbar2Tk(canvas, tb_frame, pack_toolbar=False)
    tb.update()
    tb.pack(side="left", fill="x")
    return tb


def _correlation_linkage(dff: np.ndarray, method: str = "average") -> np.ndarray:
    """Linkage on the (1 - Pearson r) distance between every pair of ROIs.

    `pdist(metric="correlation")` returns 1 - r for every column pair, so the
    resulting linkage is exactly the cross-correlation hierarchical clustering
    referenced in the GUI prompt. Default to `average` linkage because Ward's
    method assumes a Euclidean distance.
    """
    dff_z = (dff - np.mean(dff, axis=0)) / (np.std(dff, axis=0) + 1e-8)
    dist = pdist(dff_z.T, metric="correlation")
    return linkage(dist, method=method)


def _load_filter_mask(plane0: Path) -> Optional[np.ndarray]:
    """Cell-filter keep mask for a Suite2p plane0 directory.

    Tries, in order:
      1. predicted_cell_mask.npy (cell-filter classifier output, preferred)
      2. iscell.npy (suite2p classifier fallback)
    Returns ``None`` if neither file exists.
    """
    pred_path = plane0 / "predicted_cell_mask.npy"
    if pred_path.exists():
        return np.load(pred_path).astype(bool)
    iscell_path = plane0 / "iscell.npy"
    if iscell_path.exists():
        ic = np.load(iscell_path)
        return ((ic[:, 0] > 0) if ic.ndim == 2 else (ic > 0)).astype(bool)
    return None


def _load_filtered_dff(plane0: Path, prefix: str):
    """Open ``<prefix>dff.memmap.float32`` against the cell-filter mask.

    For a "filtered" prefix (e.g. ``r0p7_filtered_``) the memmap was written
    at the size of the cell-filter keep mask, so we need that exact mask to
    interpret the column count. For an unfiltered prefix, the memmap holds
    every Suite2p ROI.

    Returns (dff_array, T, N_kept, keep_mask). ``keep_mask`` is ``None`` when
    the prefix is unfiltered.
    """
    plane0 = Path(plane0)
    F = np.load(plane0 / "F.npy", mmap_mode="r")
    N_total, T = F.shape
    is_filtered = "filtered" in prefix.split("_")

    if is_filtered:
        mask = _load_filter_mask(plane0)
        if mask is None:
            raise FileNotFoundError(
                f"{plane0}: prefix {prefix!r} requires a cell-filter mask "
                "(predicted_cell_mask.npy or iscell.npy).")
        if mask.size != N_total:
            raise ValueError(
                f"{plane0}: cell-filter mask length {mask.size} does not "
                f"match F.npy ROI count {N_total}.")
        N_kept = int(mask.sum())
    else:
        mask = None
        N_kept = N_total

    dff_path = plane0 / f"{prefix}dff.memmap.float32"
    if not dff_path.exists():
        raise FileNotFoundError(f"Missing dF/F memmap: {dff_path}")
    dff = np.memmap(dff_path, dtype="float32", mode="r",
                    shape=(T, N_kept))
    return dff, T, N_kept, mask


def _stat_for_prefix(plane0: Path, prefix: str):
    """Stat list aligned with the columns of the dF/F memmap.

    For filtered prefixes, restrict ``stat`` to the cell-filter keep mask so
    the i-th stat entry matches the i-th column of dF/F.
    """
    full_stat = list(np.load(plane0 / "stat.npy", allow_pickle=True))
    if "filtered" in prefix.split("_"):
        mask = _load_filter_mask(plane0)
        if mask is not None:
            return ([s for s, keep in zip(full_stat, mask) if keep],
                    np.where(mask)[0])
    return full_stat, None


def _spatial_image(stat, Lx, Ly, roi_rgb: np.ndarray) -> np.ndarray:
    """Paint a per-ROI RGB array onto the FOV (NaN background)."""
    R = utils.paint_spatial(roi_rgb[:, 0], stat, Ly, Lx)
    G = utils.paint_spatial(roi_rgb[:, 1], stat, Ly, Lx)
    B = utils.paint_spatial(roi_rgb[:, 2], stat, Ly, Lx)
    img = np.dstack([R, G, B])
    coverage = utils.paint_spatial(np.ones(len(stat)), stat, Ly, Lx)
    img[coverage == 0] = np.nan
    return img


# ---------------------------------------------------------------------------
# Per-cluster custom-color dialog
# ---------------------------------------------------------------------------


class CustomColorDialog(tk.Toplevel):
    """Modal dialog to override the per-cluster color list.

    `initial_colors` is a list of hex strings (length = current cluster count).
    Clicking a swatch opens askcolor; OK writes the new list to `result`.
    """

    def __init__(self, master, initial_colors: list[str]) -> None:
        super().__init__(master)
        self.title("Per-cluster colors")
        self.transient(master)
        self.resizable(False, False)
        self._colors = list(initial_colors)
        self.result: Optional[list[str]] = None
        self._swatches: list[tk.Button] = []
        self._build_ui()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _build_ui(self) -> None:
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Click a swatch to pick that cluster's color.",
                  foreground="gray").grid(row=0, column=0, columnspan=3,
                                          sticky="w", pady=(0, 6))
        for i, hex_color in enumerate(self._colors):
            ttk.Label(frm, text=f"Cluster {i + 1}").grid(
                row=i + 1, column=0, sticky="w", padx=(0, 8), pady=2)
            btn = tk.Button(frm, width=6, bg=hex_color,
                            relief="ridge",
                            command=lambda idx=i: self._pick(idx))
            btn.grid(row=i + 1, column=1, sticky="w", pady=2)
            self._swatches.append(btn)
            ttk.Label(frm, textvariable=tk.StringVar(value=hex_color)).grid(
                row=i + 1, column=2, sticky="w", padx=(8, 0))

        bar = ttk.Frame(self, padding=(10, 0, 10, 10))
        bar.pack(fill="x")
        ttk.Button(bar, text="Cancel", command=self._cancel).pack(side="right")
        ttk.Button(bar, text="OK",
                   command=self._ok).pack(side="right", padx=(0, 6))

    def _pick(self, idx: int) -> None:
        rgb, hex_color = colorchooser.askcolor(
            color=self._colors[idx], parent=self,
            title=f"Pick color for cluster {idx + 1}")
        if hex_color:
            self._colors[idx] = hex_color
            self._swatches[idx].configure(bg=hex_color)

    def _ok(self) -> None:
        self.result = list(self._colors)
        self.grab_release()
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.grab_release()
        self.destroy()


# ---------------------------------------------------------------------------
# Main tab
# ---------------------------------------------------------------------------


class ClusteringTab(ttk.Frame):
    """Cross-correlation hierarchical clustering tab."""

    def __init__(self, master, state=None) -> None:
        super().__init__(master, padding=10)
        self.state = state

        self._plane0: Optional[Path] = None
        self._prefix = DEFAULT_PREFIX

        # Analysis cache (filled on Run analysis).
        self._dff: Optional[np.ndarray] = None
        self._Z: Optional[np.ndarray] = None
        self._stat = None
        self._Lx: int = 0
        self._Ly: int = 0
        self._zmax: float = 1.0

        # Color state.
        self._palette_name: str = DEFAULT_PALETTE
        self._custom_colors: Optional[list[str]] = None  # overrides dropdown

        # Threshold state.
        self._auto_threshold: float = 0.7  # fraction of max
        self._manual_T: float = 0.7  # absolute distance

        # Worker queue.
        self._q: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None

        self._slider_user_driven = True  # gate slider events during programmatic sets
        self._summary_after_id: Optional[str] = None  # debounce id for auto-write

        self._build_ui()
        self.after(POLL_MS, self._drain_queue)

        # If used inside pipeline_gui, subscribe to plane0 broadcasts.
        if state is not None:
            try:
                state.subscribe_plane0(self._on_plane0_broadcast)
                state.subscribe_lowpass_ready(self._on_plane0_broadcast)
                if getattr(state, "lowpass_plane0", None) is not None:
                    self._on_plane0_broadcast(state.lowpass_plane0)
                elif getattr(state, "plane0", None) is not None:
                    self._on_plane0_broadcast(state.plane0)
            except Exception:
                pass

    # -- UI ----------------------------------------------------------------

    def _build_ui(self) -> None:
        head = ttk.LabelFrame(
            self, text="Cross-correlation clustering "
                       "(1 - Pearson r, hierarchical, average linkage)",
            padding=8)
        head.pack(fill="x", pady=(0, 6))

        row1 = ttk.Frame(head); row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="plane0:").pack(side="left")
        self.path_var = tk.StringVar(value="")
        ttk.Entry(row1, textvariable=self.path_var, width=70).pack(
            side="left", padx=(4, 4), fill="x", expand=True)
        ttk.Button(row1, text="Browse...",
                   command=self._on_browse).pack(side="left")

        row2 = ttk.Frame(head); row2.pack(fill="x", pady=2)
        self.run_btn = ttk.Button(
            row2, text="Run analysis", command=self._on_run, state="disabled")
        self.run_btn.pack(side="left")
        self.progress = ttk.Progressbar(row2, mode="indeterminate", length=140)
        self.progress.pack(side="left", padx=8)
        self.status_var = tk.StringVar(value="Pick a plane0 folder.")
        ttk.Label(row2, textvariable=self.status_var,
                  font=("", 9, "italic")).pack(side="left", fill="x", expand=True)

        # Body: dendrogram + slider + spatial.
        body = ttk.Frame(self); body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=0)
        body.columnconfigure(2, weight=2)
        body.rowconfigure(0, weight=1)

        # Dendrogram canvas.
        dframe = ttk.LabelFrame(body, text="Dendrogram", padding=4)
        dframe.grid(row=0, column=0, sticky="nsew")
        self.d_fig = plt.Figure(figsize=(6.5, 4.5), tight_layout=True)
        self.d_ax = self.d_fig.add_subplot(111)
        self._placeholder(self.d_ax, "Run analysis to populate.")
        self.d_canvas = FigureCanvasTkAgg(self.d_fig, master=dframe)
        _attach_fig_toolbar(self.d_canvas, dframe)
        self.d_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Vertical slider for the cut.
        sframe = ttk.Frame(body)
        sframe.grid(row=0, column=1, sticky="ns", padx=4)
        ttk.Label(sframe, text="cut").pack(side="top")
        self.threshold_scale = tk.Scale(
            sframe, from_=1.0, to=0.0, resolution=0.001,
            orient=tk.VERTICAL, length=320, showvalue=False,
            command=self._on_slider, state="disabled")
        self.threshold_scale.pack(side="top", fill="y", expand=True)
        self.threshold_readout = tk.StringVar(value="-")
        ttk.Label(sframe, textvariable=self.threshold_readout,
                  width=8, anchor="center").pack(side="top")

        # Spatial canvas.
        sp_frame = ttk.LabelFrame(body, text="Spatial (cluster colors)",
                                  padding=4)
        sp_frame.grid(row=0, column=2, sticky="nsew")
        self.s_fig = plt.Figure(figsize=(5.5, 4.5), tight_layout=True)
        self.s_ax = self.s_fig.add_subplot(111)
        self._placeholder(self.s_ax, "Run analysis to populate.")
        self.s_canvas = FigureCanvasTkAgg(self.s_fig, master=sp_frame)
        _attach_fig_toolbar(self.s_canvas, sp_frame)
        self.s_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Bottom controls.
        ctl = ttk.Frame(self, padding=(0, 6, 0, 0))
        ctl.pack(fill="x")

        self.manual_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ctl, text="Manual threshold", variable=self.manual_var,
            command=self._on_manual_toggle).pack(side="left")

        ttk.Label(ctl, text="   palette:").pack(side="left")
        self.palette_var = tk.StringVar(value=DEFAULT_PALETTE)
        palette_box = ttk.Combobox(
            ctl, textvariable=self.palette_var, state="readonly", width=12,
            values=list(cmap_mod.AVAILABLE_PALETTES))
        palette_box.pack(side="left", padx=4)
        palette_box.bind("<<ComboboxSelected>>", self._on_palette_change)

        ttk.Button(ctl, text="Per-cluster colors...",
                   command=self._on_custom_colors).pack(side="left", padx=4)
        ttk.Button(ctl, text="Reset palette",
                   command=self._on_reset_palette).pack(side="left", padx=4)

        ttk.Label(ctl, text="prefix:").pack(side="left", padx=(20, 2))
        self.prefix_var = tk.StringVar(value=DEFAULT_PREFIX)
        ttk.Entry(ctl, textvariable=self.prefix_var, width=18).pack(
            side="left")

        self.export_btn = ttk.Button(
            ctl, text="Export *_rois.npy",
            command=self._on_export, state="disabled")
        self.export_btn.pack(side="right")
        self.xcorr_btn = ttk.Button(
            ctl, text="Run cluster x cluster cross-correlation",
            command=self._on_run_xcorr, state="disabled")
        self.xcorr_btn.pack(side="right", padx=(0, 6))
        self.summary_btn = ttk.Button(
            ctl, text="Save summary",
            command=self._on_save_summary, state="disabled")
        self.summary_btn.pack(side="right", padx=(0, 6))

    # -- Helpers -----------------------------------------------------------

    def _placeholder(self, ax, text: str) -> None:
        ax.clear()
        ax.set_axis_off()
        ax.text(0.5, 0.5, text, ha="center", va="center",
                transform=ax.transAxes)

    def _on_plane0_broadcast(self, plane0) -> None:
        if plane0 is None:
            return
        self.path_var.set(str(plane0))
        self._set_plane0(Path(plane0))

    def _on_browse(self) -> None:
        path = filedialog.askdirectory(
            title="Select Suite2p plane0 folder",
            initialdir=self.path_var.get() or str(Path.home()))
        if not path:
            return
        self.path_var.set(path)
        self._set_plane0(Path(path))

    def _set_plane0(self, plane0: Path) -> None:
        self._plane0 = plane0
        ok = self._inputs_ready(plane0, self.prefix_var.get())
        self.run_btn.config(state="normal" if ok else "disabled")
        if ok:
            self.status_var.set(f"Ready. ({plane0})")
        else:
            self.status_var.set(
                f"Missing dF/F memmaps with prefix {self.prefix_var.get()!r}.")

    def _inputs_ready(self, plane0: Path, prefix: str) -> bool:
        for name in ("F.npy", "stat.npy", "ops.npy",
                     f"{prefix}dff.memmap.float32"):
            if not (plane0 / name).exists():
                return False
        return True

    # -- Run worker --------------------------------------------------------

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "Analysis already running.")
            return
        plane0 = Path(self.path_var.get())
        prefix = self.prefix_var.get().strip() or DEFAULT_PREFIX
        if not self._inputs_ready(plane0, prefix):
            messagebox.showerror("Not ready", f"plane0 missing required files: {plane0}")
            return

        self._prefix = prefix
        self.run_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Computing linkage...")

        def worker():
            try:
                payload = self._compute(plane0, prefix)
                self._q.put(("done", payload))
            except Exception as e:
                self._q.put(("error", f"{e}\n{traceback.format_exc()}"))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _compute(self, plane0: Path, prefix: str) -> dict:
        dff_mm, T, N, _ = _load_filtered_dff(plane0, prefix)
        dff = np.asarray(dff_mm, dtype=np.float32)
        Z = _correlation_linkage(dff)
        stat, _ = _stat_for_prefix(plane0, prefix)
        ops = np.load(plane0 / "ops.npy", allow_pickle=True).item()
        Lx, Ly = int(ops["Lx"]), int(ops["Ly"])
        auto_frac = cmap_mod.auto_choose_threshold(Z, target_counts=(4, 5))
        return {
            "Z": Z, "dff_shape": dff.shape, "stat": stat,
            "Lx": Lx, "Ly": Ly, "auto_frac": auto_frac,
            "T": T, "N": N,
        }

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self.progress.stop()
                    self.run_btn.config(state="normal")
                    self.status_var.set("Analysis failed.")
                    messagebox.showerror(
                        "Analysis failed", payload.split("\n", 1)[0])
        except queue.Empty:
            pass
        self.after(POLL_MS, self._drain_queue)

    def _on_done(self, data: dict) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")

        self._Z = data["Z"]
        self._stat = data["stat"]
        self._Lx = data["Lx"]
        self._Ly = data["Ly"]
        self._zmax = float(np.max(self._Z[:, 2]))

        auto_frac = float(data["auto_frac"])
        self._auto_threshold = auto_frac
        self._manual_T = auto_frac * self._zmax

        # Calibrate slider range to current linkage.
        self._slider_user_driven = False
        self.threshold_scale.config(from_=self._zmax, to=0.0,
                                    resolution=max(self._zmax / 1000.0, 1e-6))
        self.threshold_scale.set(self._manual_T)
        self._slider_user_driven = True

        self._custom_colors = None  # palette resets to dropdown choice
        self.export_btn.config(state="normal")
        self.xcorr_btn.config(state="normal")
        self.summary_btn.config(state="normal")

        self._render_all()
        try:
            self._write_summary()
        except Exception as e:
            print(f"[GUI] cluster summary write failed: {e}")

        n_clusters = int(np.unique(
            fcluster(self._Z, t=self._current_threshold(),
                     criterion="distance")).size)
        self.status_var.set(
            f"OK. N_rois={data['N']}  T={data['T']}  "
            f"auto cut={auto_frac:.2f}xmax  -> {n_clusters} clusters. "
            f"Toggle 'Manual threshold' + slider to tune.")

    # -- Threshold + palette -----------------------------------------------

    def _current_threshold(self) -> float:
        if self.manual_var.get():
            return float(self._manual_T)
        return float(self._auto_threshold) * float(self._zmax)

    def _on_manual_toggle(self) -> None:
        if self.manual_var.get():
            self.threshold_scale.config(state="normal")
            # Snap slider to the auto value as a starting point.
            self._slider_user_driven = False
            self.threshold_scale.set(self._auto_threshold * self._zmax)
            self._slider_user_driven = True
            self._manual_T = self._auto_threshold * self._zmax
        else:
            self.threshold_scale.config(state="disabled")
        self._render_all()
        self._schedule_summary_write()

    def _on_slider(self, raw: str) -> None:
        if not self._slider_user_driven:
            return
        try:
            self._manual_T = float(raw)
        except ValueError:
            return
        if self._Z is not None:
            self._render_all()
            self._schedule_summary_write()

    def _on_palette_change(self, *_):
        self._palette_name = self.palette_var.get() or DEFAULT_PALETTE
        self._custom_colors = None
        if self._Z is not None:
            self._render_all()
            self._schedule_summary_write()

    def _on_custom_colors(self) -> None:
        if self._Z is None:
            messagebox.showinfo("Run first",
                                "Run the analysis before picking colors.")
            return
        n = max(1, int(np.unique(
            fcluster(self._Z, t=self._current_threshold(),
                     criterion="distance")).size))
        base = self._custom_colors or cmap_mod.resolve_palette(
            self._palette_name, n_colors=n)
        # Normalize length to current cluster count.
        if len(base) < n:
            base = base + [base[-1]] * (n - len(base))
        elif len(base) > n:
            base = base[:n]
        dlg = CustomColorDialog(self.winfo_toplevel(), base)
        self.wait_window(dlg)
        if dlg.result is not None:
            self._custom_colors = dlg.result
            self._render_all()
            self._schedule_summary_write()

    def _on_reset_palette(self) -> None:
        self._custom_colors = None
        if self._Z is not None:
            self._render_all()
            self._schedule_summary_write()

    # -- Render ------------------------------------------------------------

    def _resolve_palette_colors(self, n_clusters: int) -> list[str]:
        if self._custom_colors:
            cols = list(self._custom_colors)
            if len(cols) < n_clusters:
                cols = cols + [cols[-1]] * (n_clusters - len(cols))
            return cols[:n_clusters]
        return cmap_mod.resolve_palette(self._palette_name,
                                        n_colors=max(1, n_clusters))

    def _render_all(self) -> None:
        if self._Z is None:
            return
        T = self._current_threshold()
        n_clusters = int(np.unique(
            fcluster(self._Z, t=T, criterion="distance")).size)
        palette_hex = self._resolve_palette_colors(n_clusters)

        # Drive scipy's branch coloring.
        set_link_color_palette(palette_hex)

        # --- Dendrogram ---
        ax = self.d_ax
        ax.clear()
        dendrogram(self._Z, ax=ax,
                   color_threshold=T,
                   above_threshold_color=ABOVE_CUT_COLOR,
                   no_labels=True)
        ax.axhline(T, linestyle="--", linewidth=1.5, color="black")
        ax.set_ylabel("1 - r  (linkage distance)")
        ax.set_xlabel(f"{self._Z.shape[0] + 1} ROIs ({n_clusters} clusters)")
        ax.set_title(f"cut @ {T:.3f}"
                     f"  ({T / self._zmax:.2f} x max)"
                     + ("  [manual]" if self.manual_var.get() else "  [auto]"))
        self.d_canvas.draw_idle()

        self.threshold_readout.set(f"{T:.3f}")

        # --- Spatial ---
        # Use the dendrogram's actual leaf coloring so user-picked colors and
        # the spatial map stay consistent.
        info = dendrogram(self._Z, no_plot=True, color_threshold=T,
                          above_threshold_color=ABOVE_CUT_COLOR)
        leaves = list(info["leaves"])
        leaf_colors = list(info["leaves_color_list"])
        N = len(leaves)
        roi_rgb = np.zeros((N, 3))
        for leaf_idx, color in zip(leaves, leaf_colors):
            roi_rgb[leaf_idx, :] = mpl.colors.to_rgb(color)

        if self._stat is None or len(self._stat) != N:
            self._placeholder(self.s_ax,
                              "Spatial unavailable (stat / dF/F mismatch).")
            self.s_canvas.draw_idle()
            return

        img = _spatial_image(self._stat, self._Lx, self._Ly, roi_rgb)
        ax = self.s_ax
        ax.clear()
        ax.imshow(img, origin="upper", aspect="equal")
        ax.set_title(f"{n_clusters} clusters")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        self.s_canvas.draw_idle()

    # -- Export + cross-correlation ----------------------------------------

    def _export_clusters(self) -> Path:
        """Write *_rois.npy files for the current cut into a fresh subfolder.

        Returns the export directory.
        """
        if self._Z is None or self._plane0 is None:
            raise RuntimeError("Run analysis first.")
        T = self._current_threshold()
        labels = fcluster(self._Z, t=T, criterion="distance")
        out_dir = (self._plane0 / f"{self._prefix}cluster_results"
                   / EXPORT_SUBDIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Clear stale C*_rois.npy first so old runs don't poison crosscorr.
        for stale in out_dir.glob("C*_rois.npy"):
            stale.unlink()
        for k, lbl in enumerate(np.unique(labels), start=1):
            roi_idx = np.where(labels == lbl)[0].astype(int)
            np.save(out_dir / f"C{k}_rois.npy", roi_idx)
        np.save(out_dir / "linkage.npy", self._Z)
        np.save(out_dir / "threshold_used.npy", np.array([T], dtype=float))
        return out_dir

    def _on_export(self) -> None:
        try:
            out_dir = self._export_clusters()
        except Exception as e:
            messagebox.showerror("Export failed", str(e))
            return
        try:
            self._write_summary()
        except Exception as e:
            print(f"[GUI] cluster summary write failed: {e}")
        self.status_var.set(f"Exported clusters to {out_dir}")

    # -- Summary export ----------------------------------------------------

    def _schedule_summary_write(self, delay_ms: int = 600) -> None:
        """Debounced trigger so slider drags don't keep rewriting the file."""
        if self._Z is None or self._plane0 is None:
            return
        if self._summary_after_id is not None:
            try:
                self.after_cancel(self._summary_after_id)
            except Exception:
                pass
        self._summary_after_id = self.after(
            delay_ms, self._summary_after_fire)

    def _summary_after_fire(self) -> None:
        self._summary_after_id = None
        try:
            self._write_summary()
        except Exception as e:
            print(f"[GUI] cluster summary write failed: {e}")

    def _write_summary(self) -> Optional[Path]:
        if self._Z is None or self._plane0 is None:
            return None
        T = self._current_threshold()
        labels = fcluster(self._Z, t=T, criterion="distance")
        n_clusters = int(np.unique(labels).size)
        palette_hex = self._resolve_palette_colors(n_clusters)

        # Reproduce the dendrogram's leaf-to-color choice so the Clusters
        # sheet uses the exact colors the user sees in the spatial panel.
        set_link_color_palette(palette_hex)
        info = dendrogram(self._Z, no_plot=True, color_threshold=T,
                          above_threshold_color=ABOVE_CUT_COLOR)
        leaves = list(info["leaves"])
        leaf_colors = list(info["leaves_color_list"])
        cluster_colors: dict[int, str] = {}
        for leaf_idx, color in zip(leaves, leaf_colors):
            lbl = int(labels[leaf_idx])
            cluster_colors.setdefault(lbl, color)

        try:
            fps = float(utils.get_fps_from_notes(str(self._plane0)))
        except Exception:
            fps = None

        summary_writer.update_recording_meta(
            self._plane0, prefix=self._prefix, fps=fps,
            T=None, N=int(len(labels)),
            extra={"clustering_n_clusters": n_clusters,
                   "clustering_threshold": float(T),
                   "clustering_palette": (
                       "custom" if self._custom_colors else self._palette_name)})
        return summary_writer.write_clusters_sheet(
            self._plane0, labels,
            cluster_colors=cluster_colors,
            threshold=float(T),
            method="average", metric="correlation")

    def _on_save_summary(self) -> None:
        if self._Z is None or self._plane0 is None:
            messagebox.showinfo("Run first",
                                "Run analysis before saving the summary.")
            return
        try:
            path = self._write_summary()
            if path is not None:
                self.status_var.set(f"Summary -> {path}")
        except Exception as e:
            messagebox.showerror("Summary failed", str(e))

    def _on_run_xcorr(self) -> None:
        if self._Z is None or self._plane0 is None:
            return
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "Worker already running.")
            return
        try:
            out_dir = self._export_clusters()
        except Exception as e:
            messagebox.showerror("Export failed", str(e))
            return

        plane0 = self._plane0
        prefix = self._prefix
        cluster_folder = EXPORT_SUBDIR
        try:
            fps = float(utils.get_fps_from_notes(str(plane0), default_fps=30.0))
        except Exception:
            fps = 30.0

        self.xcorr_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self.run_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set(
            f"Cross-correlation running in {out_dir} (fps={fps:.2f})...")

        def worker():
            try:
                import crosscorrelation as xc
                xc.run_cluster_cross_correlations_gpu(
                    plane0, prefix=prefix, fps=fps,
                    cluster_folder=cluster_folder,
                    max_lag_seconds=5.0,
                    cpu_fallback=True,
                    zero_lag=True,
                )
                self._q.put(("xcorr_done", str(out_dir)))
            except Exception as e:
                self._q.put(("xcorr_error",
                             f"{e}\n{traceback.format_exc()}"))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
        # Override drain handlers locally for xcorr-specific outcomes.
        self.after(POLL_MS, self._drain_xcorr_queue)

    def _drain_xcorr_queue(self) -> None:
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "xcorr_done":
                    self.progress.stop()
                    self.xcorr_btn.config(state="normal")
                    self.export_btn.config(state="normal")
                    self.run_btn.config(state="normal")
                    self.status_var.set(
                        f"Cross-correlation complete. Results in {payload}")
                    return
                if kind == "xcorr_error":
                    self.progress.stop()
                    self.xcorr_btn.config(state="normal")
                    self.export_btn.config(state="normal")
                    self.run_btn.config(state="normal")
                    self.status_var.set("Cross-correlation failed.")
                    messagebox.showerror(
                        "Cross-correlation failed",
                        payload.split("\n", 1)[0])
                    return
                # Anything else, push back through the regular drain.
                self._q.put((kind, payload))
                break
        except queue.Empty:
            pass
        self.after(POLL_MS, self._drain_xcorr_queue)


# ---------------------------------------------------------------------------
# Standalone main
# ---------------------------------------------------------------------------


def main() -> None:
    root = tk.Tk()
    root.title("CalLIOPE - Cross-correlation clustering")
    root.geometry("1300x780")
    tab = ClusteringTab(root, state=None)
    tab.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
