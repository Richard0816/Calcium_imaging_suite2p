"""
pipeline_gui.py - top-level GUI for the calcium imaging pipeline.

Current tabs
------------
1. Input & Preprocess   - pick working directory, choose a TIFF, run shift + QC.
2. QC Preview           - animated GIF of the shifted movie + blob detection
                          overlaid on the mean image.
3. Suite2p Detection    - sparse_plus_cellpose detection + dF/F + cell-filter
                          prediction, with live console + before/after ROI maps.
                          Writes r0p7_dff and r0p7_filtered_dff memmaps.
4. Low-pass filter      - FFT, raw mean dF/F, and low-pass mean dF/F preview
                          panels driven by a cutoff slider (0.01 - 10 Hz).
                          Compute button writes r0p7_filtered_dff_lowpass and
                          r0p7_filtered_dff_dt memmaps at the chosen cutoff.
5. Event detection      - Filtered-lowpass heatmap, event raster, and
                          population event-detection diagnostics from
                          utils.detect_event_windows / plot_event_detection.

The GUI is organised so that each tab is a self-contained class; new pipeline
stages (traces, clustering, event detection, etc.) can be dropped in as
additional ``ttk.Frame`` subclasses without touching the rest.
"""

from __future__ import annotations

import contextlib
import io
import queue
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)
from PIL import Image, ImageDraw, ImageFont, ImageTk

import preprocessing
from preprocessing import PreprocessResult
import summary_writer


# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------

class AppState:
    """Mutable state shared across tabs. Tabs subscribe via ``on_result``
    (preprocessing complete) or ``on_plane0`` (suite2p detection complete)."""

    def __init__(self) -> None:
        self.working_dir: Optional[Path] = None
        self.data_root: Optional[Path] = None
        self.selected_tiff: Optional[Path] = None
        self.result: Optional[PreprocessResult] = None
        self.plane0: Optional[Path] = None
        self.lowpass_plane0: Optional[Path] = None
        self._listeners: list = []
        self._plane0_listeners: list = []
        self._lowpass_listeners: list = []

    def subscribe(self, fn) -> None:
        self._listeners.append(fn)

    def set_result(self, result: PreprocessResult) -> None:
        self.result = result
        for fn in self._listeners:
            try:
                fn(result)
            except Exception as e:
                print(f"listener error: {e}")

    def subscribe_plane0(self, fn) -> None:
        self._plane0_listeners.append(fn)

    def set_plane0(self, plane0: Path) -> None:
        self.plane0 = Path(plane0)
        for fn in self._plane0_listeners:
            try:
                fn(self.plane0)
            except Exception as e:
                print(f"plane0 listener error: {e}")

    def subscribe_lowpass_ready(self, fn) -> None:
        self._lowpass_listeners.append(fn)

    def set_lowpass_ready(self, plane0: Path) -> None:
        self.lowpass_plane0 = Path(plane0)
        for fn in self._lowpass_listeners:
            try:
                fn(self.lowpass_plane0)
            except Exception as e:
                print(f"lowpass listener error: {e}")


# ---------------------------------------------------------------------------
# Tab 1: Input & Preprocess
# ---------------------------------------------------------------------------

class PreprocessTab(ttk.Frame):
    POLL_MS = 80

    PARAM_SPEC: list = [
        # Blob detector
        {"name": "soma_diameter_px", "label": "Soma diameter (px)",
         "type": "float", "default": 12.0, "group": "Blob detection",
         "help": "expected soma size, used for LoG sigma range"},
        {"name": "scale_tol", "label": "Scale tolerance",
         "type": "float", "default": 0.5, "group": "Blob detection",
         "help": "sigma range = r * (1 ± tol) / sqrt(2)"},
        {"name": "min_contrast", "label": "Min contrast (LoG)",
         "type": "float", "default": 0.10, "group": "Blob detection",
         "help": "blob_log threshold after robust normalisation"},
        {"name": "min_area_px", "label": "Min area (px²)",
         "type": "int", "default": 25, "group": "Blob detection"},
        {"name": "max_area_px", "label": "Max area (px²)",
         "type": "int", "default": 400, "group": "Blob detection"},
        # QC gif
        {"name": "downsample_t", "label": "Time downsample factor",
         "type": "int", "default": 4, "group": "QC gif",
         "help": "keep every Nth frame for the preview"},
        {"name": "max_size_px", "label": "Max gif size (px)",
         "type": "int", "default": 512, "group": "QC gif"},
        {"name": "playback_fps", "label": "Playback FPS",
         "type": "int", "default": 15, "group": "QC gif"},
        {"name": "clip_low", "label": "Clip low (percentile)",
         "type": "float", "default": 1.0, "group": "QC gif"},
        {"name": "clip_high", "label": "Clip high (percentile)",
         "type": "float", "default": 99.5, "group": "QC gif"},
    ]

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, padding=10)
        self.state = state
        self._log_queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._params: dict = _spec_defaults(self.PARAM_SPEC)

        self._build_ui()
        self.after(self.POLL_MS, self._drain_log_queue)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        # Working directory row
        dir_frame = ttk.LabelFrame(self, text="1. Working directory (raw TIFFs)",
                                   padding=8)
        dir_frame.pack(fill="x", pady=(0, 8))

        self.dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.dir_var).grid(
            row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(dir_frame, text="Browse...", command=self._browse_dir).grid(
            row=0, column=1)
        dir_frame.columnconfigure(0, weight=1)

        # TIFF selection row (multi-select; ctrl/shift to extend)
        tiff_frame = ttk.LabelFrame(
            self,
            text="2. TIFF files  (select 1+; multi-selection -> one grouped "
                 "recording)",
            padding=8,
        )
        tiff_frame.pack(fill="x", pady=(0, 8))

        list_holder = ttk.Frame(tiff_frame)
        list_holder.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        list_holder.columnconfigure(0, weight=1)
        self.tiff_listbox = tk.Listbox(
            list_holder, selectmode="extended", height=6,
            exportselection=False,
        )
        self.tiff_listbox.grid(row=0, column=0, sticky="ew")
        list_sb = ttk.Scrollbar(list_holder, orient="vertical",
                                command=self.tiff_listbox.yview)
        list_sb.grid(row=0, column=1, sticky="ns")
        self.tiff_listbox.config(yscrollcommand=list_sb.set)
        self.tiff_listbox.bind(
            "<<ListboxSelect>>",
            lambda _e: self._update_existing_status(),
        )

        ttk.Button(tiff_frame, text="Refresh",
                   command=self._refresh_tiffs).grid(row=0, column=1, sticky="n")

        ttk.Label(
            tiff_frame,
            text="Identifier (optional, used as folder name; "
                 "default = '+'-joined stems for groups):",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 2))
        self.identifier_var = tk.StringVar()
        self.identifier_var.trace_add(
            "write", lambda *_a: self._update_existing_status()
        )
        ttk.Entry(tiff_frame, textvariable=self.identifier_var).grid(
            row=2, column=0, columnspan=2, sticky="ew")

        tiff_frame.columnconfigure(0, weight=1)

        # Data root (output)
        out_frame = ttk.LabelFrame(
            self, text="3. Output root  (creates data/<recording>/ underneath)",
            padding=8)
        out_frame.pack(fill="x", pady=(0, 8))

        self.data_root_var = tk.StringVar()
        ttk.Entry(out_frame, textvariable=self.data_root_var).grid(
            row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(out_frame, text="Browse...", command=self._browse_data_root).grid(
            row=0, column=1)
        out_frame.columnconfigure(0, weight=1)

        # Action row
        action = ttk.Frame(self)
        action.pack(fill="x", pady=(4, 8))
        self.run_btn = ttk.Button(action, text="Run preprocessing",
                                  command=self._on_run)
        self.run_btn.pack(side="left")
        self.rerun_btn = ttk.Button(action, text="Force rerun",
                                    command=self._on_force_rerun,
                                    state="disabled")
        self.rerun_btn.pack(side="left", padx=(6, 0))
        ttk.Button(action, text="Advanced...",
                   command=self._on_advanced).pack(side="left", padx=(6, 0))
        self.progress = ttk.Progressbar(action, mode="indeterminate", length=200)
        self.progress.pack(side="left", padx=12)
        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(action, textvariable=self.status_var).pack(side="left")

        # Log
        log_frame = ttk.LabelFrame(self, text="Log", padding=4)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        sb.pack(fill="y", side="right")
        self.log.config(yscrollcommand=sb.set)

    # -- Handlers -----------------------------------------------------------

    def _browse_dir(self) -> None:
        path = filedialog.askdirectory(title="Select working directory")
        if not path:
            return
        self.dir_var.set(path)
        self.state.working_dir = Path(path)
        # Default data root to <working_dir>/../data
        if not self.data_root_var.get():
            default = Path(path).parent / "data"
            self.data_root_var.set(str(default))
        self._refresh_tiffs()

    def _browse_data_root(self) -> None:
        path = filedialog.askdirectory(title="Select output (data) root")
        if path:
            self.data_root_var.set(path)

    def _refresh_tiffs(self) -> None:
        wd = self.dir_var.get().strip()
        self.tiff_listbox.delete(0, "end")
        if not wd:
            return
        tiffs = preprocessing.list_tiffs(wd)
        names = [t.name for t in tiffs]
        for n in names:
            self.tiff_listbox.insert("end", n)
        if names:
            self.tiff_listbox.selection_set(0)
        else:
            self._append_log(f"No .tif/.tiff files in {wd}")
        self._update_existing_status()

    @staticmethod
    def _trailing_index(name: str) -> int:
        """Last run of digits in a filename stem; sentinel for files without
        any numeric suffix."""
        import re
        stem = Path(name).stem
        m = re.search(r"(\d+)(?!.*\d)", stem)
        return int(m.group(1)) if m else -1

    def _selected_tiff_names(self) -> list[str]:
        """Selected TIFF filenames, sorted by trailing index in the stem."""
        idxs = self.tiff_listbox.curselection()
        names = [self.tiff_listbox.get(i) for i in idxs]
        names.sort(key=self._trailing_index)
        return names

    def _resolved_identifier(self, names: list[str]) -> str:
        """User-supplied identifier if non-empty, else '+'-joined stems."""
        explicit = self.identifier_var.get().strip()
        if explicit:
            return explicit
        return "+".join(Path(n).stem for n in names)

    def _existing_out_dir(self) -> Optional[Path]:
        """Resolve <data_root>/<identifier>/ if all inputs are populated."""
        wd = self.dir_var.get().strip()
        names = self._selected_tiff_names()
        data_root = self.data_root_var.get().strip()
        if not (wd and names and data_root):
            return None
        return Path(data_root) / self._resolved_identifier(names)

    def _update_existing_status(self) -> None:
        """Inspect candidate output dir; toggle Force-rerun button + status."""
        out_dir = self._existing_out_dir()
        existing = (preprocessing.load_existing_preprocess(out_dir)
                    if out_dir is not None else None)
        if existing is not None:
            self.rerun_btn.config(state="normal")
            self.status_var.set(
                f"Existing outputs at {out_dir} - Run will load them; "
                f"use Force rerun to redo.")
        else:
            self.rerun_btn.config(state="disabled")
            if out_dir is not None:
                n = len(self._selected_tiff_names())
                suffix = f" ({n} TIFFs as one group)" if n > 1 else ""
                self.status_var.set(
                    f"No existing outputs; Run will preprocess{suffix}.")
            else:
                self.status_var.set("Idle.")

    def _on_run(self) -> None:
        self._start_run(force=False)

    def _on_force_rerun(self) -> None:
        out_dir = self._existing_out_dir()
        if out_dir is not None:
            ok = messagebox.askyesno(
                "Force rerun",
                f"This will overwrite existing outputs in:\n{out_dir}\n\n"
                f"Continue?")
            if not ok:
                return
        self._start_run(force=True)

    def _start_run(self, force: bool) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "Preprocessing is already running.")
            return

        wd = self.dir_var.get().strip()
        names = self._selected_tiff_names()
        data_root = self.data_root_var.get().strip()

        if not wd or not names:
            messagebox.showerror(
                "Missing input",
                "Select a working directory and at least one TIFF.")
            return
        if not data_root:
            messagebox.showerror("Missing output",
                                 "Set an output root directory.")
            return

        srcs = [Path(wd) / n for n in names]
        identifier = self._resolved_identifier(names)
        self.state.selected_tiff = srcs[0]
        self.state.data_root = Path(data_root)

        # Reuse existing outputs if present (unless the user forced a rerun).
        if not force:
            out_dir = self._existing_out_dir()
            existing = (preprocessing.load_existing_preprocess(out_dir)
                        if out_dir is not None else None)
            if existing is not None:
                self._append_log(
                    f"--- Loading existing outputs from {out_dir} ---")
                self._on_done(existing)
                return

        self.run_btn.config(state="disabled")
        self.rerun_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Running...")
        if len(srcs) == 1:
            self._append_log(f"--- Preprocessing {srcs[0].name} ---")
        else:
            self._append_log(
                f"--- Preprocessing group '{identifier}' "
                f"({len(srcs)} TIFFs, in trailing-index order):")
            for s in srcs:
                self._append_log(f"      {s.name}")
            self._append_log("---")

        # Snapshot params on the main thread so the worker sees a stable
        # value even if the user opens Advanced again mid-run.
        params = dict(self._params)
        blob_keys = ("scale_tol", "min_contrast", "min_area_px",
                     "max_area_px")
        qc_keys = ("downsample_t", "max_size_px", "playback_fps",
                   "clip_low", "clip_high")
        blob_params = {k: params[k] for k in blob_keys if k in params}
        qc_params = {k: params[k] for k in qc_keys if k in params}
        soma_diameter_px = float(params.get("soma_diameter_px", 12.0))
        explicit_identifier = self.identifier_var.get().strip() or None

        def worker():
            try:
                if len(srcs) == 1:
                    result = preprocessing.preprocess_tiff(
                        src_tiff=srcs[0],
                        data_root=data_root,
                        recording_name=explicit_identifier,
                        soma_diameter_px=soma_diameter_px,
                        progress_cb=lambda m: self._log_queue.put(("log", m)),
                        blob_params=blob_params,
                        qc_params=qc_params,
                    )
                else:
                    result = preprocessing.preprocess_tiff_group(
                        src_tiffs=srcs,
                        data_root=data_root,
                        recording_name=identifier,
                        soma_diameter_px=soma_diameter_px,
                        progress_cb=lambda m: self._log_queue.put(("log", m)),
                        blob_params=blob_params,
                        qc_params=qc_params,
                    )
                self._log_queue.put(("done", result))
            except Exception as e:
                self._log_queue.put(("error", str(e)))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _on_advanced(self) -> None:
        if _open_advanced(self, "Preprocess - Advanced parameters",
                          self.PARAM_SPEC, self._params):
            self._append_log(
                "[GUI] Advanced parameters updated. Re-run preprocessing "
                "to apply.")

    # -- Logging / queue drain ---------------------------------------------

    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self._log_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self._on_error(payload)
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._drain_log_queue)

    def _append_log(self, text: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _on_done(self, result: PreprocessResult) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.rerun_btn.config(state="normal")
        n_frames_str = (f"{result.n_frames} frames"
                        if result.n_frames > 0 else "n_frames=?")
        self.status_var.set(
            f"Done - {result.n_blobs} preview blobs, {n_frames_str}.")
        self._append_log(
            f"Outputs: {result.qc_gif.name}, "
            f"{result.shifted_tiff.name}, mean.npy, blobs.npy")
        self.state.set_result(result)

    def _on_error(self, msg: str) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")
        self._update_existing_status()
        self.status_var.set("Error.")
        self._append_log(f"ERROR: {msg}")
        messagebox.showerror("Preprocessing failed", msg)


# ---------------------------------------------------------------------------
# Tab 2: QC preview (gif + blob detection)
# ---------------------------------------------------------------------------

class QcTab(ttk.Frame):

    FRAME_MS = 66   # ~15 fps gif playback in the viewer

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, padding=10)
        self.state = state
        self._gif_frames: list[ImageTk.PhotoImage] = []
        self._gif_index = 0
        self._gif_job: Optional[str] = None

        self._build_ui()
        state.subscribe(self._on_result)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 6))
        self.header_var = tk.StringVar(
            value="No preprocessing result yet. Run it in the first tab.")
        ttk.Label(header, textvariable=self.header_var,
                  font=("", 10, "italic")).pack(side="left")
        ttk.Button(header, text="Reload from folder...",
                   command=self._reload_from_folder).pack(side="right")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1, uniform="cols")
        body.columnconfigure(1, weight=1, uniform="cols")
        body.rowconfigure(0, weight=1)

        # Left: gif player
        gif_frame = ttk.LabelFrame(body, text="QC movie (GIF)", padding=6)
        gif_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.gif_label = ttk.Label(gif_frame, anchor="center")
        self.gif_label.pack(fill="both", expand=True)
        self.gif_status = tk.StringVar(value="")
        ttk.Label(gif_frame, textvariable=self.gif_status).pack(anchor="w",
                                                                pady=(4, 0))

        # Right: mean image + blob detection
        blob_frame = ttk.LabelFrame(body, text="Mean image + blob detection",
                                    padding=6)
        blob_frame.grid(row=0, column=1, sticky="nsew")

        self.fig = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=self.ax.transAxes)
        self.canvas = FigureCanvasTkAgg(self.fig, master=blob_frame)
        _attach_fig_toolbar(self.canvas, blob_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # -- Reload helper ------------------------------------------------------

    def _reload_from_folder(self) -> None:
        path = filedialog.askdirectory(
            title="Select an already-preprocessed recording folder")
        if not path:
            return
        result = preprocessing.load_existing_preprocess(path)
        if result is None:
            messagebox.showwarning(
                "No outputs",
                f"Folder {path} doesn't contain the expected "
                "shifted tiff / qc.gif / mean.npy.")
            return
        self.state.set_result(result)

    # -- AppState listener --------------------------------------------------

    def _on_result(self, result: PreprocessResult) -> None:
        self.header_var.set(
            f"Recording: {result.out_dir.name}   "
            f"({result.shape_yx[0]} x {result.shape_yx[1]})")
        self._load_gif(result.qc_gif)
        self._draw_blob_preview(result)

    # -- GIF playback -------------------------------------------------------

    def _load_gif(self, gif_path: Path) -> None:
        if self._gif_job is not None:
            self.after_cancel(self._gif_job)
            self._gif_job = None
        self._gif_frames = []
        self._gif_index = 0

        if not gif_path.exists():
            self.gif_status.set("GIF missing.")
            return
        try:
            im = Image.open(str(gif_path))
            while True:
                self._gif_frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(im.tell() + 1)
        except EOFError:
            pass
        except Exception as e:
            self.gif_status.set(f"GIF load error: {e}")
            return

        if not self._gif_frames:
            self.gif_status.set("GIF has no frames.")
            return

        self.gif_status.set(f"{len(self._gif_frames)} frames")
        self._advance_gif()

    def _advance_gif(self) -> None:
        if not self._gif_frames:
            return
        frame = self._gif_frames[self._gif_index]
        self.gif_label.configure(image=frame)
        self.gif_label.image = frame
        self._gif_index = (self._gif_index + 1) % len(self._gif_frames)
        self._gif_job = self.after(self.FRAME_MS, self._advance_gif)

    # -- Blob preview plot --------------------------------------------------

    def _draw_blob_preview(self, result: PreprocessResult) -> None:
        mean = np.load(str(result.mean_image_path))
        blobs = (np.load(str(result.blobs_path))
                 if result.blobs_path.exists() else np.zeros((0, 3)))

        self.ax.clear()
        self.ax.set_axis_off()
        vmax = float(np.quantile(mean, 0.995))
        self.ax.imshow(mean, cmap="gray", vmax=vmax)
        for row in np.atleast_2d(blobs):
            if row.size < 3:
                continue
            y, x, r = float(row[0]), float(row[1]), float(row[2])
            self.ax.add_patch(
                plt.Circle((x, y), r, color="cyan", fill=False, linewidth=1.2))
        self.ax.set_title(f"{len(blobs)} preview blobs")
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Reusable "Advanced..." parameter dialog
# ---------------------------------------------------------------------------

class AdvancedDialog(tk.Toplevel):
    """Modal Toplevel that auto-builds an Entry/Combobox form from a
    list of parameter specs and writes accepted values into ``current``.

    Each spec is a dict with keys:
        name    : str - the dict key in ``current``
        label   : str - field label
        type    : 'float' | 'int' | 'str' | 'bool' | 'choice'
        group   : str - LabelFrame heading
        choices : list[str] - required iff type == 'choice'
        help    : str - optional one-line tooltip text shown next to label
    """

    def __init__(self, master, title: str, specs: list[dict],
                 current: dict) -> None:
        super().__init__(master)
        self.title(title)
        self.transient(master)
        self.resizable(True, True)
        self.geometry("560x600")
        self._specs = specs
        self._current = current
        self._defaults = {s["name"]: s["default"] for s in specs}
        self._vars: dict = {}
        self._accepted = False

        self._build_ui()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _build_ui(self) -> None:
        # Scrollable canvas + inner frame for grouped param sections.
        outer = ttk.Frame(self); outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        inner = ttk.Frame(canvas, padding=8)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _resize(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(win, width=canvas.winfo_width())
        inner.bind("<Configure>", _resize)
        canvas.bind("<Configure>", _resize)

        # Build one LabelFrame per group, in spec order.
        groups: dict = {}
        for spec in self._specs:
            grp = spec.get("group", "Parameters")
            if grp not in groups:
                lf = ttk.LabelFrame(inner, text=grp, padding=8)
                lf.pack(fill="x", pady=(0, 8))
                lf.columnconfigure(1, weight=1)
                groups[grp] = (lf, 0)
            lf, row = groups[grp]
            self._add_field(lf, row, spec)
            groups[grp] = (lf, row + 1)

        # OK / Cancel / Reset buttons.
        btn_row = ttk.Frame(self, padding=(8, 4)); btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Reset to defaults",
                   command=self._reset_defaults).pack(side="left")
        ttk.Button(btn_row, text="Cancel",
                   command=self._on_cancel).pack(side="right")
        ttk.Button(btn_row, text="OK",
                   command=self._on_ok).pack(side="right", padx=(0, 6))

    def _add_field(self, parent, row, spec) -> None:
        name = spec["name"]
        label = spec.get("label", name)
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        cur = self._current.get(name, spec["default"])

        if spec["type"] == "bool":
            var = tk.BooleanVar(value=bool(cur))
            ttk.Checkbutton(parent, variable=var).grid(
                row=row, column=1, sticky="w", pady=2)
        elif spec["type"] == "choice":
            var = tk.StringVar(value=str(cur))
            cb = ttk.Combobox(parent, textvariable=var, state="readonly",
                              values=list(spec.get("choices", [])))
            cb.grid(row=row, column=1, sticky="ew", pady=2)
        else:
            var = tk.StringVar(value=str(cur))
            ttk.Entry(parent, textvariable=var).grid(
                row=row, column=1, sticky="ew", pady=2)

        help_text = spec.get("help")
        if help_text:
            ttk.Label(parent, text=help_text, foreground="gray",
                      font=("", 8, "italic")).grid(
                row=row, column=2, sticky="w", padx=(8, 0), pady=2)

        self._vars[name] = (var, spec)

    def _reset_defaults(self) -> None:
        for name, (var, spec) in self._vars.items():
            d = self._defaults[name]
            if spec["type"] == "bool":
                var.set(bool(d))
            else:
                var.set(str(d))

    def _coerce(self, raw: str, spec: dict):
        t = spec["type"]
        if t == "bool":
            return bool(raw)
        if t == "int":
            return int(float(raw))
        if t == "float":
            return float(raw)
        return raw

    def _on_ok(self) -> None:
        try:
            new_values = {}
            for name, (var, spec) in self._vars.items():
                new_values[name] = self._coerce(var.get(), spec)
        except (ValueError, TypeError) as e:
            messagebox.showerror("Invalid value",
                                 f"Could not parse {name!r}: {e}")
            return
        self._current.update(new_values)
        self._accepted = True
        self.grab_release()
        self.destroy()

    def _on_cancel(self) -> None:
        self.grab_release()
        self.destroy()


def _spec_defaults(specs: list[dict]) -> dict:
    """Build a {name: default} dict from a PARAM_SPEC list."""
    return {s["name"]: s["default"] for s in specs}


def _open_advanced(master, title: str, specs: list[dict],
                   current: dict) -> bool:
    """Open AdvancedDialog modally; returns True if user clicked OK."""
    dlg = AdvancedDialog(master, title, specs, current)
    master.wait_window(dlg)
    return dlg._accepted


def _attach_fig_toolbar(canvas: FigureCanvasTkAgg,
                        parent_frame) -> NavigationToolbar2Tk:
    """Attach a matplotlib navigation toolbar (pan / zoom / Save Figure)
    above ``canvas`` inside ``parent_frame``. Caller must pack/grid the
    canvas widget AFTER this call so the toolbar lands above it.

    "Save Figure" exports to PNG/PDF/SVG/etc via the standard matplotlib
    file dialog -- this is the export path for every plot in the GUI."""
    tb_frame = ttk.Frame(parent_frame)
    tb_frame.pack(side="top", fill="x")
    tb = NavigationToolbar2Tk(canvas, tb_frame, pack_toolbar=False)
    tb.update()
    tb.pack(side="left", fill="x")
    return tb


# ---------------------------------------------------------------------------
# Tab 3: Suite2p Detection (sparse_plus_cellpose + dF/F + cell filter)
# ---------------------------------------------------------------------------

# Stream stdout/stderr from the worker thread into the log queue line-by-line.
# Suite2p and cellpose use plain print() to report
# progress, so redirecting at the IO level captures everything without
# touching the upstream code.
class _QueueWriter(io.TextIOBase):
    def __init__(self, q: "queue.Queue") -> None:
        self.q = q
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.q.put(("log", line))
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self.q.put(("log", self._buf))
            self._buf = ""


class Suite2pTab(ttk.Frame):
    """Runs sparse_plus_cellpose -> dF/F -> cell filter on the preprocessed
    shifted TIFF and renders three panels: live console, raw detected ROIs,
    and ROIs surviving the cell-filter prediction mask.
    """

    POLL_MS = 80

    DEFAULT_OPS_PATH = Path(__file__).parent / "suite2p_2p_ops_240621.npy"
    DEFAULT_CKPT_PATH = Path(r"F:\cellfilter_checkpoints\best.pt")
    DEFAULT_AAV_CSV = Path(__file__).parent / "human_SLE_2p_meta.csv"

    # Background images the user can pick for panels 2 & 3. Only entries
    # whose key is actually present (and 2D) in ops.npy get a radio button.
    # Order here is the radio-button display order.
    KNOWN_BG_IMAGES: list[tuple[str, str]] = [
        ("meanImg",       "Mean"),
        ("meanImgE",      "Mean (enhanced)"),
        ("max_proj",      "Max projection"),
        ("Vcorr",         "Correlation"),
        ("refImg",        "Reg. ref"),
        ("meanImg_chan2", "Mean ch2"),
    ]
    DEFAULT_BG_KEY = "meanImgE"

    PARAM_SPEC: list = [
        # Sparsery (suite2p detection)
        {"name": "threshold_scaling", "label": "threshold_scaling",
         "type": "float", "default": 0.85, "group": "Sparsery"},
        {"name": "high_pass", "label": "high_pass (frames)",
         "type": "int", "default": 100, "group": "Sparsery"},
        {"name": "smooth_sigma", "label": "smooth_sigma",
         "type": "float", "default": 1.0, "group": "Sparsery"},
        {"name": "max_iterations", "label": "max_iterations",
         "type": "int", "default": 1500, "group": "Sparsery"},
        {"name": "spatial_scale", "label": "spatial_scale (0=auto)",
         "type": "int", "default": 0, "group": "Sparsery"},
        {"name": "preclassify", "label": "preclassify",
         "type": "float", "default": 0.0, "group": "Sparsery"},
        {"name": "hard_cap", "label": "ROI hard cap",
         "type": "int", "default": 60000, "group": "Sparsery",
         "help": "abort sparsery if it exceeds this many ROIs"},
        # Cellpose
        {"name": "cellpose_model_type", "label": "model_type",
         "type": "choice", "choices": ["cyto", "cyto2", "nuclei"],
         "default": "cyto2", "group": "Cellpose"},
        {"name": "cellpose_diameter", "label": "diameter (0=auto)",
         "type": "int", "default": 0, "group": "Cellpose"},
        {"name": "cellpose_flow_threshold", "label": "flow_threshold",
         "type": "float", "default": 0.8, "group": "Cellpose"},
        {"name": "cellpose_cellprob_threshold", "label": "cellprob_threshold",
         "type": "float", "default": -1.0, "group": "Cellpose"},
        # Merge
        {"name": "max_overlap", "label": "Cellpose merge max overlap",
         "type": "float", "default": 0.3, "group": "Merge",
         "help": "drop cellpose ROIs > this fraction covered by sparsery"},
        # dF/F
        {"name": "fps_override", "label": "FPS override (0=auto)",
         "type": "float", "default": 0.0, "group": "dF/F",
         "help": "non-zero overrides utils.get_fps_from_notes "
                 "(fallback default is 15.07)"},
        {"name": "neuropil_coef", "label": "Neuropil coefficient (r)",
         "type": "float", "default": 0.7, "group": "dF/F"},
        {"name": "perc", "label": "Baseline percentile",
         "type": "int", "default": 10, "group": "dF/F"},
        {"name": "win_sec", "label": "Rolling window (s)",
         "type": "float", "default": 45.0, "group": "dF/F",
         "help": "only used in 'rolling' baseline mode"},
        # Default lowpass written here (Tab 4 overwrites filtered_*)
        {"name": "default_lowpass_hz", "label": "Default low-pass (Hz)",
         "type": "float", "default": 1.0, "group": "Default low-pass"},
        {"name": "default_sg_win_ms", "label": "SG derivative window (ms)",
         "type": "int", "default": 333, "group": "Default low-pass"},
        {"name": "default_sg_poly", "label": "SG polynomial order",
         "type": "int", "default": 2, "group": "Default low-pass"},
        # GPU dF/F via analyze_output_gpu (CuPy)
        {"name": "use_gpu_dff", "label": "Use GPU for dF/F",
         "type": "bool", "default": True, "group": "GPU",
         "help": "falls back to CPU if CuPy missing or baseline mode "
                 "is 'rolling' (GPU only supports first-N-min mean)"},
        {"name": "gpu_roi_chunk", "label": "GPU ROI chunk (0=all)",
         "type": "int", "default": 0, "group": "GPU",
         "help": "split ROIs into chunks if VRAM is tight"},
    ]

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, padding=10)
        self.state = state
        self._log_queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._final_plane0: Optional[Path] = None
        self._params: dict = _spec_defaults(self.PARAM_SPEC)
        self._bg_var = tk.StringVar(value=self.DEFAULT_BG_KEY)
        self._panel_cache: Optional[dict] = None

        self._build_ui()
        self.after(self.POLL_MS, self._drain_log_queue)
        state.subscribe(self._on_preprocess_result)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        header = ttk.LabelFrame(
            self, text="Suite2p detection (sparsery + cellpose) "
                       "-> dF/F -> cell filter", padding=8)
        header.pack(fill="x", pady=(0, 6))

        # Base ops .npy
        row = ttk.Frame(header); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Base ops:", width=12).pack(side="left")
        self.ops_var = tk.StringVar(value=str(self.DEFAULT_OPS_PATH))
        ttk.Entry(row, textvariable=self.ops_var).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row, text="Browse...", command=self._browse_ops).pack(
            side="left")

        # Cell-filter checkpoint
        row = ttk.Frame(header); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Cell filter:", width=12).pack(side="left")
        self.ckpt_var = tk.StringVar(value=str(self.DEFAULT_CKPT_PATH))
        ttk.Entry(row, textvariable=self.ckpt_var).pack(
            side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(row, text="Browse...", command=self._browse_ckpt).pack(
            side="left")

        # dF/F baseline mode (rolling vs first-N-minutes)
        row = ttk.Frame(header); row.pack(fill="x", pady=2)
        ttk.Label(row, text="dF/F baseline:", width=12).pack(side="left")
        self.baseline_var = tk.StringVar(value="rolling")
        ttk.Radiobutton(
            row, text="Rolling (45 s window, 10th pct)",
            value="rolling", variable=self.baseline_var,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            row, text="First N minutes:",
            value="first_n", variable=self.baseline_var,
        ).pack(side="left")
        self.baseline_min_var = tk.StringVar(value="2")
        ttk.Entry(row, textvariable=self.baseline_min_var,
                  width=6).pack(side="left", padx=(4, 2))
        ttk.Label(row, text="min").pack(side="left")

        # Run + status
        row = ttk.Frame(header); row.pack(fill="x", pady=(4, 0))
        self.run_btn = ttk.Button(
            row, text="Run detection + cell filter",
            command=self._on_run, state="disabled")
        self.run_btn.pack(side="left")
        self.load_btn = ttk.Button(
            row, text="Load existing panels", command=self._on_load_existing,
            state="disabled")
        self.load_btn.pack(side="left", padx=(6, 0))
        self.summary_btn = ttk.Button(
            row, text="Save summary",
            command=self._on_save_summary, state="disabled")
        self.summary_btn.pack(side="left", padx=(6, 0))
        ttk.Button(row, text="Advanced...",
                   command=self._on_advanced).pack(side="left", padx=(6, 0))
        self.progress = ttk.Progressbar(row, mode="indeterminate", length=200)
        self.progress.pack(side="left", padx=12)
        self.status_var = tk.StringVar(value="Run preprocessing first.")
        ttk.Label(row, textvariable=self.status_var).pack(side="left")

        # Body: panel 1 (console), background-image picker, panels 2+3 (image)
        body = ttk.Frame(self); body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=2)
        body.rowconfigure(1, weight=0)  # bg picker (fixed height)
        body.rowconfigure(2, weight=3)
        body.columnconfigure(0, weight=1, uniform="cols")
        body.columnconfigure(1, weight=1, uniform="cols")

        # Panel 1: console
        log_frame = ttk.LabelFrame(body, text="1. Suite2p console", padding=4)
        log_frame.grid(row=0, column=0, columnspan=2, sticky="nsew",
                       pady=(0, 6))
        self.log = tk.Text(log_frame, height=10, wrap="word",
                           state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(log_frame, orient="vertical",
                           command=self.log.yview)
        sb.pack(fill="y", side="right")
        self.log.config(yscrollcommand=sb.set)

        # Background-image picker (shared by panels 2 and 3)
        bg_row = ttk.LabelFrame(
            body, text="Background image (panels 2 & 3)", padding=4)
        bg_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self._bg_radio_holder = ttk.Frame(bg_row)
        self._bg_radio_holder.pack(side="left", fill="x", expand=True)
        ttk.Label(
            self._bg_radio_holder,
            text="Run detection (or load existing) to choose a background.",
        ).pack(side="left")

        # Panel 2: detected ROIs
        det_frame = ttk.LabelFrame(
            body, text="2. Detected ROIs (raw suite2p output)", padding=6)
        det_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 3))
        self.det_fig = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.det_ax = self.det_fig.add_subplot(111)
        self.det_ax.set_axis_off()
        self.det_ax.text(0.5, 0.5, "No detection yet", ha="center",
                         va="center", transform=self.det_ax.transAxes)
        self.det_canvas = FigureCanvasTkAgg(self.det_fig, master=det_frame)
        _attach_fig_toolbar(self.det_canvas, det_frame)
        self.det_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Panel 3: filtered ROIs
        fil_frame = ttk.LabelFrame(
            body, text="3. After cell-filter prediction mask", padding=6)
        fil_frame.grid(row=2, column=1, sticky="nsew", padx=(3, 0))
        self.fil_fig = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.fil_ax = self.fil_fig.add_subplot(111)
        self.fil_ax.set_axis_off()
        self.fil_ax.text(0.5, 0.5, "No filter applied yet", ha="center",
                         va="center", transform=self.fil_ax.transAxes)
        self.fil_canvas = FigureCanvasTkAgg(self.fil_fig, master=fil_frame)
        _attach_fig_toolbar(self.fil_canvas, fil_frame)
        self.fil_canvas.get_tk_widget().pack(fill="both", expand=True)

    # -- Handlers -----------------------------------------------------------

    def _browse_ops(self) -> None:
        path = filedialog.askopenfilename(
            title="Select base Suite2p ops file",
            filetypes=[("NumPy", "*.npy"), ("All files", "*.*")])
        if path:
            self.ops_var.set(path)

    def _browse_ckpt(self) -> None:
        path = filedialog.askopenfilename(
            title="Select cell-filter checkpoint",
            filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")])
        if path:
            self.ckpt_var.set(path)

    def _on_preprocess_result(self, result: PreprocessResult) -> None:
        self.run_btn.config(state="normal")
        existing = self._candidate_plane0(result)
        if existing is not None and self._plane0_has_outputs(existing):
            self.load_btn.config(state="normal")
            self.status_var.set(
                f"Ready: {result.shifted_tiff.name}  "
                f"(existing detection at {existing} - 'Load existing' "
                f"will render without re-running)")
        else:
            self.load_btn.config(state="disabled")
            self.status_var.set(
                f"Ready: {result.shifted_tiff.name}  "
                f"({result.shape_yx[0]}x{result.shape_yx[1]})")

    @staticmethod
    def _candidate_plane0(result: PreprocessResult) -> Path:
        """Resolve the canonical plane0 path the GUI's worker would
        write to for a given preprocess result."""
        return result.out_dir / "detection" / "final" / "suite2p" / "plane0"

    @staticmethod
    def _plane0_has_outputs(plane0: Path) -> bool:
        """Minimal set of files we need to render panels 2 and 3."""
        return (plane0 / "stat.npy").exists() and \
               (plane0 / "ops.npy").exists()

    def _on_load_existing(self) -> None:
        result = self.state.result
        if result is None:
            messagebox.showerror(
                "Missing input",
                "Run preprocessing first so I know which folder to load.")
            return
        plane0 = self._candidate_plane0(result)
        if not self._plane0_has_outputs(plane0):
            messagebox.showerror(
                "No data",
                f"Expected stat.npy / ops.npy at:\n{plane0}\n"
                f"Run detection first.")
            return
        self._append_log(
            f"--- Loading existing panels from {plane0} ---")
        self.status_var.set("Loading existing panels...")
        self._final_plane0 = plane0
        try:
            self._draw_panels(plane0)
            self.status_var.set(f"Loaded -> {plane0}")
        except Exception as e:
            self.status_var.set(f"Render failed: {e}")
            self._append_log(f"[GUI] render error: {e}\n"
                             f"{traceback.format_exc()}")
            return
        # Notify downstream tabs (Low-pass / Event detection) that this
        # plane0 is ready, same as a fresh run would.
        try:
            self.state.set_plane0(plane0)
        except Exception as e:
            self._append_log(f"[GUI] plane0 publish error: {e}")

    def _on_advanced(self) -> None:
        if _open_advanced(self, "Suite2p Detection - Advanced parameters",
                          self.PARAM_SPEC, self._params):
            self._append_log(
                "[GUI] Advanced parameters updated. Re-run detection "
                "to apply.")

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "Detection is already running.")
            return
        result = self.state.result
        if result is None:
            messagebox.showerror("Missing input", "Run preprocessing first.")
            return

        ops_path = self.ops_var.get().strip()
        if not ops_path or not Path(ops_path).exists():
            messagebox.showerror(
                "Missing ops", f"Base ops file not found:\n{ops_path}")
            return

        ckpt_path = self.ckpt_var.get().strip()
        ckpt_present = bool(ckpt_path) and Path(ckpt_path).exists()
        if not ckpt_present:
            self._append_log(
                f"[GUI] No cell-filter checkpoint at {ckpt_path}; "
                "the third panel will fall back to suite2p's iscell.")

        baseline_mode = self.baseline_var.get()
        baseline_min = 2.0
        if baseline_mode == "first_n":
            try:
                baseline_min = float(self.baseline_min_var.get().strip())
                if baseline_min <= 0:
                    raise ValueError("must be > 0")
            except ValueError as e:
                messagebox.showerror(
                    "Invalid baseline",
                    f"First-N-minute baseline must be a positive number "
                    f"(got {self.baseline_min_var.get()!r}): {e}")
                return

        tiff_folder = result.shifted_tiff.parent
        save_folder = result.out_dir / "detection"
        rec_id = result.out_dir.name

        self.run_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Running...")
        self._append_log(
            f"--- Starting detection on {tiff_folder.name} "
            f"(save -> {save_folder}) ---")
        self._final_plane0 = None

        def worker():
            try:
                self._run_pipeline(
                    tiff_folder=tiff_folder,
                    save_folder=save_folder,
                    ops_path=ops_path,
                    ckpt_path=(ckpt_path if ckpt_present else None),
                    rec_id=rec_id,
                    baseline_mode=baseline_mode,
                    baseline_min=baseline_min,
                )
                self._log_queue.put(("done", None))
            except Exception as e:
                self._log_queue.put(
                    ("error", f"{e}\n{traceback.format_exc()}"))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    # -- Worker -------------------------------------------------------------

    def _run_pipeline(
        self,
        tiff_folder: Path,
        save_folder: Path,
        ops_path: str,
        ckpt_path: Optional[str],
        rec_id: str,
        baseline_mode: str,
        baseline_min: float,
    ) -> None:
        writer = _QueueWriter(self._log_queue)
        # Snapshot params on the calling thread.
        params = dict(self._params)
        sparsery_ops = {
            "high_pass":         params.get("high_pass", 100),
            "preclassify":       params.get("preclassify", 0.0),
            "smooth_sigma":      params.get("smooth_sigma", 1.0),
            "sparse_mode":       True,
            "spatial_scale":     params.get("spatial_scale", 0),
            "threshold_scaling": params.get("threshold_scaling", 0.85),
            "max_iterations":    params.get("max_iterations", 1500),
        }
        cellpose_cfg = {
            "cellpose_model_type":
                params.get("cellpose_model_type", "cyto2"),
            "cellpose_diameter":
                params.get("cellpose_diameter", 0),
            "cellpose_flow_threshold":
                params.get("cellpose_flow_threshold", 0.8),
            "cellpose_cellprob_threshold":
                params.get("cellpose_cellprob_threshold", -1.0),
            "cellpose_channel_input": "meanImg",
        }
        hard_cap = int(params.get("hard_cap", 60000))
        max_overlap = float(params.get("max_overlap", 0.3))

        with contextlib.redirect_stdout(writer), \
                contextlib.redirect_stderr(writer):
            # 1. sparse_plus_cellpose detection
            print("[GUI] running sparse_plus_cellpose...")
            import sparse_plus_cellpose as spc
            aav_csv = (str(self.DEFAULT_AAV_CSV)
                       if self.DEFAULT_AAV_CSV.exists() else None)
            final_plane0 = spc.run(
                tiff_folder=str(tiff_folder),
                save_folder=str(save_folder),
                path_to_ops=ops_path,
                sparsery_ops=sparsery_ops,
                cellpose_cfg=cellpose_cfg,
                hard_cap=hard_cap,
                max_overlap=max_overlap,
                aav_info_csv=aav_csv,
                tau_vals=spc.DEFAULT_TAU_VALS,
                verbose=True,
            )
            print(f"[GUI] suite2p plane0 -> {final_plane0}")
            self._log_queue.put(("plane0", final_plane0))

            # 2. dF/F only (no lowpass / no derivative) -> r0p7_dff.memmap.float32
            self._run_dff(final_plane0, baseline_mode, baseline_min)

            # 3. Cell-filter prediction (writes predicted_cell_mask.npy)
            if ckpt_path:
                print("[GUI] running cell-filter prediction...")
                self._run_cellfilter(final_plane0, ckpt_path, rec_id)
                print("[GUI] predicted_cell_mask.npy written")

            # 4. Filtered dF/F memmap, gated by the predicted cell mask
            #    (or the suite2p classifier's iscell when no checkpoint).
            self._run_filtered_dff(final_plane0)

    def _run_dff(
        self, plane0: Path, baseline_mode: str, baseline_min: float,
    ) -> None:
        """Inline dF/F + low-pass + derivative computation. Writes:
            plane0/r0p7_dff.memmap.float32          (T, N)
            plane0/r0p7_dff_lowpass.memmap.float32  (T, N)  @ default 1 Hz
            plane0/r0p7_dff_dt.memmap.float32       (T, N)
        All three are populated so the cellfilter (which goes through
        utils.s2p_open_memmaps) can mmap the full set. Tab 4's compute
        button later overwrites the *filtered* lowpass + dt memmaps at
        the user's chosen cutoff.

        Tries the GPU path (analyze_output_gpu via CuPy) first when
        ``use_gpu_dff`` is enabled and the baseline mode is 'first_n';
        otherwise falls back to the per-cell CPU loop.
        """
        import utils
        F = np.load(plane0 / "F.npy", allow_pickle=False)
        Fneu = np.load(plane0 / "Fneu.npy", allow_pickle=False)
        if F.shape != Fneu.shape or F.ndim != 2:
            raise ValueError(
                f"unexpected F/Fneu shapes: {F.shape}, {Fneu.shape}")
        F = np.asarray(F, dtype=np.float32, order="C")
        Fneu = np.asarray(Fneu, dtype=np.float32, order="C")
        # suite2p F is canonically (N, T)
        N, T = F.shape

        fps_override = float(self._params.get("fps_override", 0.0))
        if fps_override > 0:
            fps = fps_override
            print(f"[GUI] FPS override active: {fps:.4f} Hz")
        else:
            fps = float(utils.get_fps_from_notes(str(plane0)))
        if baseline_mode == "first_n":
            print(f"[GUI] dF/F: first-{baseline_min:g}-min baseline "
                  f"(fps={fps}, n_baseline_frames~"
                  f"{int(baseline_min * 60 * fps)})")
        else:
            print(f"[GUI] dF/F: rolling baseline (fps={fps})")

        # Try GPU first if requested and applicable.
        cutoff_hz_default = float(
            self._params.get("default_lowpass_hz", 1.0))
        sg_win_ms_default = int(self._params.get("default_sg_win_ms", 333))
        sg_poly_default = int(self._params.get("default_sg_poly", 2))
        r_default = float(self._params.get("neuropil_coef", 0.7))

        if self._maybe_run_dff_gpu(
                plane0=plane0, baseline_mode=baseline_mode,
                baseline_min=baseline_min, fps=fps,
                cutoff_hz=cutoff_hz_default,
                sg_win_ms=sg_win_ms_default, sg_poly=sg_poly_default,
                r=r_default):
            return  # GPU path wrote the memmaps already
        cutoff_hz = float(self._params.get("default_lowpass_hz", 1.0))
        sg_win_ms = int(self._params.get("default_sg_win_ms", 333))
        sg_poly = int(self._params.get("default_sg_poly", 2))
        r = float(self._params.get("neuropil_coef", 0.7))
        perc = int(self._params.get("perc", 10))
        win_sec = float(self._params.get("win_sec", 45.0))
        print(f"[GUI] low-pass default cutoff {cutoff_hz:.2f} Hz; "
              f"SG derivative win {sg_win_ms} ms / poly {sg_poly}; "
              f"neuropil r={r:.2f}, baseline pct={perc}, win={win_sec:g}s")

        shape = (T, N)
        dff_path = plane0 / "r0p7_dff.memmap.float32"
        lp_path = plane0 / "r0p7_dff_lowpass.memmap.float32"
        dt_path = plane0 / "r0p7_dff_dt.memmap.float32"
        dff_mm = np.memmap(str(dff_path), mode="w+", dtype="float32",
                           shape=shape)
        lp_mm = np.memmap(str(lp_path), mode="w+", dtype="float32",
                          shape=shape)
        dt_mm = np.memmap(str(dt_path), mode="w+", dtype="float32",
                          shape=shape)

        batch = max(8, utils.change_batch_according_to_free_ram() * 20)
        sos = None

        t0 = time.time()
        for i0 in range(0, N, batch):
            i1 = min(N, i0 + batch)
            for i in range(i0, i1):
                trace = (F[i, :] - r * Fneu[i, :]).astype(np.float32)
                if baseline_mode == "first_n":
                    dff = utils.first_n_min_df_over_f_1d(
                        trace, baseline_min=baseline_min,
                        perc=perc, fps=fps)
                else:
                    dff = utils.robust_df_over_f_1d(
                        trace, win_sec=win_sec, perc=perc, fps=fps)
                lp, _, sos = utils.lowpass_causal_1d(
                    dff, fps=fps, cutoff_hz=cutoff_hz,
                    order=2, zi=None, sos=sos)
                dt = utils.sg_first_derivative_1d(
                    lp, fps=fps, win_ms=sg_win_ms, poly=sg_poly)
                dff_mm[:, i] = dff
                lp_mm[:, i] = lp
                dt_mm[:, i] = dt
            dff_mm.flush(); lp_mm.flush(); dt_mm.flush()
            print(f"[GUI] dF/F batch {i0}-{i1 - 1}/{N - 1} "
                  f"({time.time() - t0:.1f}s)")

        del dff_mm, lp_mm, dt_mm
        print(f"[GUI] r0p7_dff / r0p7_dff_lowpass / r0p7_dff_dt "
              f"memmaps written ({time.time() - t0:.1f}s)")

    def _maybe_run_dff_gpu(
        self, plane0: Path, baseline_mode: str, baseline_min: float,
        fps: float, cutoff_hz: float, sg_win_ms: int, sg_poly: int,
        r: float,
    ) -> bool:
        """If the GPU path is enabled and usable, run it and return True.
        Returns False to indicate the caller should run the CPU fallback.

        The GPU module (analyze_output_gpu) only supports a 'first-N
        seconds mean' baseline, so we skip it for the rolling mode.
        """
        if not bool(self._params.get("use_gpu_dff", True)):
            print("[GUI] GPU dF/F disabled in Advanced parameters; "
                  "using CPU.")
            return False
        if baseline_mode != "first_n":
            print("[GUI] GPU dF/F only supports first-N-min baseline; "
                  "rolling mode falls back to CPU.")
            return False
        try:
            import analyze_output_gpu as gpu
        except Exception as e:
            print(f"[GUI] analyze_output_gpu not importable ({e}); "
                  "using CPU.")
            return False
        if not getattr(gpu, "_CUPY_OK", False):
            print("[GUI] CuPy not available; using CPU dF/F.")
            return False

        baseline_sec = float(baseline_min) * 60.0
        roi_chunk = int(self._params.get("gpu_roi_chunk", 0))
        if roi_chunk <= 0:
            roi_chunk = None
        F = np.load(plane0 / "F.npy", allow_pickle=False)
        Fneu = np.load(plane0 / "Fneu.npy", allow_pickle=False)
        print(f"[GUI] dF/F (GPU): baseline_sec={baseline_sec:g}  "
              f"cutoff={cutoff_hz:g} Hz  sg=({sg_win_ms} ms, p{sg_poly})")
        try:
            t0 = time.time()
            gpu.process_suite2p_traces_gpu(
                F, Fneu, fps=fps, r=r,
                baseline_sec=baseline_sec,
                cutoff_hz=cutoff_hz,
                sg_win_ms=sg_win_ms, sg_poly=sg_poly,
                out_dir=str(plane0), prefix="r0p7_",
                roi_chunk=roi_chunk,
            )
            print(f"[GUI] GPU dF/F + lowpass + dt complete "
                  f"({time.time() - t0:.1f}s)")
            return True
        except Exception as e:
            print(f"[GUI] GPU dF/F failed ({e}); falling back to CPU.")
            return False

    def _run_filtered_dff(self, plane0: Path) -> None:
        """Slice r0p7_dff.memmap.float32 down to ROIs that pass the
        cell-filter mask (or suite2p iscell as a fallback) and write
        r0p7_filtered_dff.memmap.float32.
        """
        # Load shape (T, N) from F.npy header without materializing F.
        F = np.load(plane0 / "F.npy", mmap_mode="r")
        N_total, T = F.shape  # suite2p F is (N, T)

        mask = self._load_keep_mask(plane0, N_total)
        N_kept = int(mask.sum())
        if N_kept == 0:
            print("[GUI] filtered dF/F: 0 ROIs survive the mask; "
                  "skipping filtered memmap")
            return

        dff_path = plane0 / "r0p7_dff.memmap.float32"
        filtered_path = plane0 / "r0p7_filtered_dff.memmap.float32"
        dff = np.memmap(str(dff_path), dtype="float32", mode="r",
                        shape=(T, N_total))
        filt = np.memmap(str(filtered_path), dtype="float32", mode="w+",
                         shape=(T, N_kept))
        # Slice in chunks of frames to avoid materializing the full
        # (T, N_total) array in RAM.
        chunk = max(1, min(T, 4096))
        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            filt[t0:t1, :] = dff[t0:t1, :][:, mask]
        filt.flush()
        del filt, dff
        print(f"[GUI] r0p7_filtered_dff.memmap.float32 written "
              f"(T={T}, N_kept={N_kept}/{N_total})")

    def _run_cellfilter(
        self, plane0: Path, ckpt_path: str, rec_id: str,
    ) -> None:
        import torch
        from cellfilter.model import CellFilter
        from cellfilter.predict import predict_recording

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  device: {device}   ckpt: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = CellFilter().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        predict_recording(rec_id, model, device, plane0=plane0)

    # -- Log queue + completion --------------------------------------------

    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self._log_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "plane0":
                    self._final_plane0 = Path(payload)
                elif kind == "done":
                    self._on_done()
                elif kind == "error":
                    self._on_error(payload)
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._drain_log_queue)

    def _append_log(self, text: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _on_done(self) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")
        plane0 = self._final_plane0
        if plane0 is None:
            self.status_var.set("Done (no plane0 returned).")
            return
        try:
            self._draw_panels(plane0)
            self.status_var.set(f"Done -> {plane0}")
        except Exception as e:
            self.status_var.set(f"Done, render failed: {e}")
            self._append_log(f"[GUI] render error: {e}\n"
                             f"{traceback.format_exc()}")
        # Notify downstream tabs (e.g. Low-pass tab) that detection is done.
        try:
            self.state.set_plane0(plane0)
        except Exception as e:
            self._append_log(f"[GUI] plane0 publish error: {e}")

    def _on_error(self, msg: str) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status_var.set("Error.")
        self._append_log(f"ERROR: {msg}")
        messagebox.showerror("Detection failed", msg.split("\n", 1)[0])

    # -- Panel rendering ----------------------------------------------------

    def _draw_panels(self, plane0: Path) -> None:
        """Load suite2p outputs at ``plane0``, populate the background-image
        radio buttons from whatever 2D images live in ops.npy, then render
        panels 2 and 3 with the currently-selected background."""
        ops = np.load(plane0 / "ops.npy", allow_pickle=True).item()
        stat = np.load(plane0 / "stat.npy", allow_pickle=True)
        n_total = len(stat)

        keep = self._load_keep_mask(plane0, n_total)
        prob_path = plane0 / "predicted_cell_prob.npy"
        probs = (np.load(prob_path).astype(np.float32)
                 if prob_path.exists() else None)

        self._panel_cache = dict(
            plane0=plane0, ops=ops, stat=stat,
            n_total=n_total, keep=keep, probs=probs,
        )

        self._populate_bg_radios(ops)
        self._redraw_with_bg()

        self.summary_btn.config(state="normal")
        try:
            self._write_summary(plane0, stat, keep, probs)
        except Exception as e:
            self._append_log(f"[GUI] summary write failed: {e}")

    def _populate_bg_radios(self, ops: dict) -> None:
        """Rebuild the radio-button row for whatever 2D images the loaded
        ops actually contains. Defaults to DEFAULT_BG_KEY when available."""
        for w in self._bg_radio_holder.winfo_children():
            w.destroy()

        available: list[tuple[str, str]] = []
        for key, label in self.KNOWN_BG_IMAGES:
            img = ops.get(key)
            if not isinstance(img, np.ndarray) or img.ndim != 2:
                continue
            available.append((key, label))

        if not available:
            ttk.Label(self._bg_radio_holder,
                      text="(no 2D images in ops.npy)").pack(side="left")
            return

        keys = {k for k, _ in available}
        if self._bg_var.get() not in keys:
            self._bg_var.set(self.DEFAULT_BG_KEY
                             if self.DEFAULT_BG_KEY in keys
                             else available[0][0])

        for key, label in available:
            ttk.Radiobutton(
                self._bg_radio_holder, text=label, value=key,
                variable=self._bg_var,
                command=self._on_bg_changed,
            ).pack(side="left", padx=(0, 10))

    def _on_bg_changed(self) -> None:
        if self._panel_cache is not None:
            self._redraw_with_bg()

    def _redraw_with_bg(self) -> None:
        """Re-render panels 2 and 3 with the currently-selected background.
        Uses cached data; does NOT reload from disk or rewrite the summary."""
        c = self._panel_cache
        if c is None:
            return
        ops = c["ops"]
        stat = c["stat"]
        n_total = c["n_total"]
        keep = c["keep"]
        probs = c["probs"]

        bg_key = self._bg_var.get()
        bg = ops.get(bg_key)
        bg_label = dict(self.KNOWN_BG_IMAGES).get(bg_key, bg_key)
        if not isinstance(bg, np.ndarray) or bg.ndim != 2:
            bg = ops.get("meanImgE", ops.get("meanImg"))
            bg_label = "Mean (fallback)"
        bg = np.asarray(bg, dtype=np.float32)
        Ly, Lx = bg.shape

        vmax = float(np.quantile(bg, 0.995))
        vmin = float(np.quantile(bg, 0.01))

        label_all = self._build_label_image(stat, Ly, Lx)
        self._render_panel(
            self.det_ax, self.det_canvas, bg, label_all, vmin, vmax,
            f"All detected ROIs (n = {n_total})  [bg: {bg_label}]")

        kept_n = int(keep.sum())
        if probs is not None and probs.shape[0] == n_total:
            score_img = self._build_score_image(stat, keep, probs, Ly, Lx)
            title = (f"After filter (n = {kept_n} / {n_total})  "
                     f"[bg: {bg_label}, coloured by predicted_cell_prob]")
            self._render_score_panel(
                self.fil_ax, self.fil_canvas, bg, score_img,
                vmin, vmax, title)
        else:
            kept_stat = [s for i, s in enumerate(stat) if keep[i]]
            label_kept = self._build_label_image(kept_stat, Ly, Lx)
            keep_src = "iscell.npy (suite2p classifier)"
            self._render_panel(
                self.fil_ax, self.fil_canvas, bg, label_kept,
                vmin, vmax,
                f"After filter (n = {kept_n} / {n_total})  "
                f"[bg: {bg_label}, {keep_src}]")

    @staticmethod
    def _build_label_image(stat, Ly: int, Lx: int) -> np.ndarray:
        label = np.zeros((Ly, Lx), dtype=np.int32)
        for i, s in enumerate(stat, start=1):
            yp = np.asarray(s["ypix"]); xp = np.asarray(s["xpix"])
            ok = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
            label[yp[ok], xp[ok]] = i
        return label

    @staticmethod
    def _build_score_image(stat, keep: np.ndarray, probs: np.ndarray,
                           Ly: int, Lx: int) -> np.ndarray:
        """Return a (Ly, Lx) float32 image where each kept ROI's pixels
        carry that ROI's prediction probability; non-ROI pixels are NaN
        so a masked imshow ignores them.
        """
        img = np.full((Ly, Lx), np.nan, dtype=np.float32)
        for i, s in enumerate(stat):
            if not keep[i]:
                continue
            yp = np.asarray(s["ypix"]); xp = np.asarray(s["xpix"])
            ok = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
            img[yp[ok], xp[ok]] = float(probs[i])
        return img

    @staticmethod
    def _load_keep_mask(plane0: Path, n_total: int) -> np.ndarray:
        mask_path = plane0 / "predicted_cell_mask.npy"
        if mask_path.exists():
            return np.load(mask_path).astype(bool)
        iscell_path = plane0 / "iscell.npy"
        if iscell_path.exists():
            ic = np.load(iscell_path)
            return (ic[:, 0] > 0) if ic.ndim == 2 else (ic > 0).astype(bool)
        return np.ones(n_total, dtype=bool)

    # -- Summary export -----------------------------------------------------

    def _write_summary(self, plane0: Path, stat, keep, probs) -> Path:
        prob_path = plane0 / "predicted_cell_prob.npy"
        p_cell = (np.load(prob_path)
                  if prob_path.exists() else None)
        iscell_path = plane0 / "iscell.npy"
        iscell_arr = None
        if iscell_path.exists():
            ic = np.load(iscell_path)
            iscell_arr = (ic[:, 0] > 0) if ic.ndim == 2 else (ic > 0)
        try:
            import utils
            fps = float(utils.get_fps_from_notes(str(plane0)))
        except Exception:
            fps = None
        summary_writer.update_recording_meta(
            plane0, fps=fps, T=None, N=int(len(stat)))
        return summary_writer.write_rois_sheet(
            plane0, stat, p_cell=p_cell,
            predicted_mask=keep, iscell=iscell_arr)

    def _on_save_summary(self) -> None:
        plane0 = self._final_plane0
        if plane0 is None:
            messagebox.showinfo(
                "No data", "Run detection or 'Load existing panels' first.")
            return
        try:
            stat = np.load(plane0 / "stat.npy", allow_pickle=True)
            n_total = len(stat)
            keep = self._load_keep_mask(plane0, n_total)
            prob_path = plane0 / "predicted_cell_prob.npy"
            probs = (np.load(prob_path).astype(np.float32)
                     if prob_path.exists() else None)
            path = self._write_summary(plane0, stat, keep, probs)
            self.status_var.set(f"Summary -> {path}")
        except Exception as e:
            messagebox.showerror("Summary failed", str(e))

    @staticmethod
    def _render_panel(ax, canvas, mean, label_img, vmin, vmax, title) -> None:
        ax.clear(); ax.set_axis_off()
        ax.imshow(mean, cmap="gray", vmin=vmin, vmax=vmax)
        if label_img.max() > 0:
            overlay = np.ma.masked_where(label_img == 0, label_img)
            ax.imshow(overlay, cmap="nipy_spectral", alpha=0.45,
                      interpolation="nearest")
        ax.set_title(title, fontsize=9)
        canvas.draw_idle()

    @staticmethod
    def _render_score_panel(ax, canvas, mean, score_img, vmin, vmax,
                             title) -> None:
        """Same overlay style as _render_panel but coloured by score.
        Colormap is stretched across [0.5, 1.0] (the cell-filter pass
        range), so the lowest score that survives the threshold maps to
        the bottom of viridis (purple) and 1.0 maps to the top (yellow).
        """
        fig = ax.figure
        # Drop any colorbar created on a previous render, then redraw clean.
        for old_ax in list(fig.axes):
            if old_ax is not ax:
                fig.delaxes(old_ax)
        ax.clear(); ax.set_axis_off()
        ax.imshow(mean, cmap="gray", vmin=vmin, vmax=vmax)
        overlay = np.ma.masked_invalid(score_img)
        if overlay.count() > 0:
            im = ax.imshow(overlay, cmap="viridis", alpha=0.45,
                           vmin=0.5, vmax=1.0, interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label("predicted_cell_prob (>= 0.5 only)",
                         fontsize=8)
            cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9)
        canvas.draw_idle()


# ---------------------------------------------------------------------------
# Tab 4: Low-pass filter (FFT, mean dF/F, mean dF/F low-pass)
# ---------------------------------------------------------------------------

class LowpassTab(ttk.Frame):
    """Three panels:
        1. FFT power spectrum of the population-mean filtered dF/F.
        2. Population-mean dF/F (raw).
        3. Population-mean dF/F passed through a causal low-pass at
           the slider's cutoff (0.01 - 10 Hz, default 1 Hz).
    Panel 3 (and the cutoff line on panel 1) update live as the slider
    moves; panels 1 and 2 are computed once when the data loads.
    """

    CUTOFF_MIN = 0.01
    CUTOFF_MAX = 10.0
    CUTOFF_DEFAULT = 1.0

    POLL_MS = 80

    PARAM_SPEC: list = [
        {"name": "filter_order", "label": "Butterworth order",
         "type": "int", "default": 2, "group": "Low-pass filter"},
        {"name": "sg_win_ms", "label": "SG derivative window (ms)",
         "type": "int", "default": 333, "group": "Derivative"},
        {"name": "sg_poly", "label": "SG polynomial order",
         "type": "int", "default": 2, "group": "Derivative"},
        {"name": "cutoff_min", "label": "Slider min (Hz)",
         "type": "float", "default": 0.01, "group": "Slider bounds"},
        {"name": "cutoff_max", "label": "Slider max (Hz)",
         "type": "float", "default": 10.0, "group": "Slider bounds"},
    ]

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, padding=10)
        self.state = state
        self._plane0: Optional[Path] = None
        self._fps: float = 15.0
        self._mean_dff: Optional[np.ndarray] = None  # (T,) trace shown
        self._trace_label: str = "mean"  # description for plot titles
        self._fft_xf: Optional[np.ndarray] = None
        self._fft_power: Optional[np.ndarray] = None
        self._cutoff_line = None
        self._slider_job: Optional[str] = None
        self._params: dict = _spec_defaults(self.PARAM_SPEC)

        # Compute worker (writes the lowpass + derivative memmaps).
        self._compute_queue: queue.Queue = queue.Queue()
        self._compute_worker: Optional[threading.Thread] = None
        self._n_kept: int = 0
        self._T: int = 0

        self._build_ui()
        self.after(self.POLL_MS, self._drain_compute_queue)
        state.subscribe_plane0(self._on_plane0)
        if state.plane0 is not None:
            # Detection ran before this tab was constructed.
            self._on_plane0(state.plane0)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        header = ttk.LabelFrame(
            self, text="Low-pass filter (causal Butterworth, order 2)",
            padding=8)
        header.pack(fill="x", pady=(0, 6))

        row = ttk.Frame(header); row.pack(fill="x", pady=2)
        ttk.Label(row, text="Cutoff (Hz):", width=12).pack(side="left")
        self.cutoff_var = tk.DoubleVar(value=self.CUTOFF_DEFAULT)
        self.slider = ttk.Scale(
            row, from_=self.CUTOFF_MIN, to=self.CUTOFF_MAX,
            orient="horizontal", variable=self.cutoff_var,
            command=self._on_slider,
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.cutoff_str = tk.StringVar(value=f"{self.CUTOFF_DEFAULT:.2f}")
        entry = ttk.Entry(row, textvariable=self.cutoff_str, width=8)
        entry.pack(side="left")
        entry.bind("<Return>", self._on_entry)
        ttk.Label(row, text="Hz").pack(side="left", padx=(2, 8))
        ttk.Button(row, text="Advanced...",
                   command=self._on_advanced).pack(side="right",
                                                   padx=(0, 6))
        ttk.Button(row, text="Reload from folder...",
                   command=self._reload_from_folder).pack(side="right")

        # Commit row: write the lowpass + derivative memmaps at the
        # currently-selected cutoff once the user is happy with it.
        row = ttk.Frame(header); row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="", width=12).pack(side="left")
        self.compute_btn = ttk.Button(
            row, text="Compute low-pass + derivative (write memmaps)",
            command=self._on_compute, state="disabled")
        self.compute_btn.pack(side="left")
        self.compute_progress = ttk.Progressbar(
            row, mode="indeterminate", length=160)
        self.compute_progress.pack(side="left", padx=12)
        self.compute_status_var = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.compute_status_var,
                  font=("", 9, "italic")).pack(side="left")

        # Trace source toggle: population mean / best ROI / manual ROI #.
        row = ttk.Frame(header); row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="Trace source:", width=12).pack(side="left")
        self.source_var = tk.StringVar(value="mean")
        ttk.Radiobutton(
            row, text="Mean across kept ROIs",
            value="mean", variable=self.source_var,
            command=self._on_source_change,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            row, text="Best-scoring ROI (max predicted_cell_prob)",
            value="best", variable=self.source_var,
            command=self._on_source_change,
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            row, text="Manual ROI #:",
            value="manual", variable=self.source_var,
            command=self._on_source_change,
        ).pack(side="left")
        self.manual_roi_var = tk.StringVar(value="0")
        manual_entry = ttk.Entry(row, textvariable=self.manual_roi_var,
                                 width=7)
        manual_entry.pack(side="left", padx=(2, 0))
        manual_entry.bind("<Return>", self._on_manual_entry)
        manual_entry.bind("<FocusOut>", self._on_manual_entry)

        self.status_var = tk.StringVar(
            value="Run detection first (or reload from a finished folder).")
        ttk.Label(self, textvariable=self.status_var,
                  font=("", 9, "italic")).pack(anchor="w", pady=(0, 6))

        body = ttk.Frame(self); body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1, uniform="rows")
        body.rowconfigure(1, weight=1, uniform="rows")
        body.rowconfigure(2, weight=1, uniform="rows")
        body.columnconfigure(0, weight=1)

        # Panel 1: FFT power spectrum (title set per-trace by _draw_fft)
        f1 = ttk.LabelFrame(body, text="1. FFT power spectrum", padding=4)
        f1.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        self.fft_fig = plt.Figure(figsize=(8, 2.4), tight_layout=True)
        self.fft_ax = self.fft_fig.add_subplot(111)
        self.fft_ax.set_axis_off()
        self.fft_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                         transform=self.fft_ax.transAxes)
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=f1)
        _attach_fig_toolbar(self.fft_canvas, f1)
        self.fft_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Panel 2: raw dF/F
        f2 = ttk.LabelFrame(body, text="2. Raw dF/F", padding=4)
        f2.grid(row=1, column=0, sticky="nsew", pady=(0, 4))
        self.raw_fig = plt.Figure(figsize=(8, 2.4), tight_layout=True)
        self.raw_ax = self.raw_fig.add_subplot(111)
        self.raw_ax.set_axis_off()
        self.raw_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                         transform=self.raw_ax.transAxes)
        self.raw_canvas = FigureCanvasTkAgg(self.raw_fig, master=f2)
        _attach_fig_toolbar(self.raw_canvas, f2)
        self.raw_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Panel 3: low-pass dF/F
        f3 = ttk.LabelFrame(body, text="3. Low-pass dF/F", padding=4)
        f3.grid(row=2, column=0, sticky="nsew")
        self.lp_fig = plt.Figure(figsize=(8, 2.4), tight_layout=True)
        self.lp_ax = self.lp_fig.add_subplot(111)
        self.lp_ax.set_axis_off()
        self.lp_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=self.lp_ax.transAxes)
        self.lp_canvas = FigureCanvasTkAgg(self.lp_fig, master=f3)
        _attach_fig_toolbar(self.lp_canvas, f3)
        self.lp_canvas.get_tk_widget().pack(fill="both", expand=True)

    # -- Plane0 handling ----------------------------------------------------

    def _reload_from_folder(self) -> None:
        path = filedialog.askdirectory(
            title="Select a suite2p plane0 folder containing "
                  "r0p7_filtered_dff.memmap.float32")
        if not path:
            return
        self._on_plane0(Path(path))

    def _on_plane0(self, plane0: Path) -> None:
        self._plane0 = Path(plane0)
        self.status_var.set(f"Loading dF/F from {self._plane0} ...")
        self.update_idletasks()
        try:
            self._load_data(self._plane0)
        except Exception as e:
            self.status_var.set(f"Load error: {e}")
            self.compute_btn.config(state="disabled")
            return
        self._draw_fft()
        self._draw_raw()
        self._draw_lowpass()
        self.status_var.set(
            f"Loaded T={self._mean_dff.size}  N_kept={self._n_kept}  "
            f"fps={self._fps:.2f}  cutoff={self.cutoff_var.get():.3f} Hz")
        self.compute_btn.config(state="normal")

    def _load_data(self, plane0: Path) -> None:
        import utils
        F = np.load(plane0 / "F.npy", mmap_mode="r")
        N_total, T = F.shape
        self._T = int(T)

        mask_path = plane0 / "predicted_cell_mask.npy"
        prob_path = plane0 / "predicted_cell_prob.npy"
        if mask_path.exists():
            mask = np.load(mask_path).astype(bool)
        else:
            iscell_path = plane0 / "iscell.npy"
            if iscell_path.exists():
                ic = np.load(iscell_path)
                mask = ((ic[:, 0] > 0) if ic.ndim == 2
                        else (ic > 0)).astype(bool)
            else:
                mask = np.ones(N_total, dtype=bool)
        N_kept = int(mask.sum())
        self._n_kept = N_kept
        if N_kept == 0:
            raise RuntimeError("No ROIs survive the cell-filter mask.")

        filtered_path = plane0 / "r0p7_filtered_dff.memmap.float32"
        if filtered_path.exists():
            dff = np.memmap(str(filtered_path), dtype="float32", mode="r",
                            shape=(T, N_kept))
            # Map kept-ROI indices back to original Suite2p ROI indices
            # so we can label the "best ROI" with its real number.
            kept_idx = np.flatnonzero(mask)
        else:
            # Fall back to slicing the unfiltered memmap on the fly.
            dff_path = plane0 / "r0p7_dff.memmap.float32"
            if not dff_path.exists():
                raise FileNotFoundError(
                    f"Neither {filtered_path.name} nor {dff_path.name} "
                    f"exists in {plane0}.")
            full = np.memmap(str(dff_path), dtype="float32", mode="r",
                             shape=(T, N_total))
            dff = np.asarray(full[:, mask])
            kept_idx = np.flatnonzero(mask)

        source = self.source_var.get() if hasattr(self, "source_var") \
            else "mean"
        if source == "best":
            if not prob_path.exists():
                # No score file -> warn and fall back to mean.
                print("[GUI] predicted_cell_prob.npy missing; "
                      "falling back to population mean trace.")
                source = "mean"

        if source == "manual":
            # Manual ROI # is interpreted as the *original* suite2p ROI
            # index (0..N_total-1), regardless of whether it survived the
            # cell-filter mask. Load that single column from the unfiltered
            # r0p7_dff memmap so excluded ROIs are still inspectable.
            try:
                roi_idx = int(self.manual_roi_var.get().strip())
            except (ValueError, AttributeError):
                print(f"[GUI] Manual ROI: invalid number "
                      f"{self.manual_roi_var.get()!r}; "
                      f"falling back to mean.")
                source = "mean"
            else:
                if roi_idx < 0 or roi_idx >= N_total:
                    print(f"[GUI] Manual ROI {roi_idx} out of range "
                          f"[0, {N_total}); falling back to mean.")
                    source = "mean"
                else:
                    dff_full_path = plane0 / "r0p7_dff.memmap.float32"
                    if not dff_full_path.exists():
                        print(f"[GUI] {dff_full_path.name} missing; cannot "
                              f"extract manual ROI; falling back to mean.")
                        source = "mean"
                    else:
                        full = np.memmap(str(dff_full_path),
                                         dtype="float32", mode="r",
                                         shape=(T, N_total))
                        trace = np.asarray(full[:, roi_idx],
                                           dtype=np.float32)
                        self._mean_dff = trace
                        excluded = "" if mask[roi_idx] \
                            else " [excluded by mask]"
                        self._trace_label = f"ROI {roi_idx}{excluded}"

        if source == "best":
            probs_full = np.load(prob_path).astype(np.float32)
            # Restrict to kept ROIs (filtered memmap column space).
            probs_kept = probs_full[mask] if probs_full.shape[0] == N_total \
                else probs_full
            best_in_kept = int(np.argmax(probs_kept))
            best_orig = int(kept_idx[best_in_kept])
            best_score = float(probs_kept[best_in_kept])
            trace = np.asarray(dff[:, best_in_kept], dtype=np.float32)
            self._mean_dff = trace
            self._trace_label = (f"ROI {best_orig} (score {best_score:.3f})")
        elif source == "mean":
            # Population-mean trace in chunks (avoids materializing full memmap).
            mean_dff = np.zeros(T, dtype=np.float32)
            chunk = max(1, min(T, 8192))
            for t0 in range(0, T, chunk):
                t1 = min(T, t0 + chunk)
                mean_dff[t0:t1] = np.asarray(dff[t0:t1, :]).mean(axis=1)
            self._mean_dff = mean_dff
            self._trace_label = f"mean across {N_kept} kept ROIs"

        # FPS lookup; falls back to 15 Hz if metadata isn't resolvable.
        self._fps = float(utils.get_fps_from_notes(str(plane0)))

        from fft_all_rois import compute_fft
        self._fft_xf, self._fft_power = compute_fft(self._mean_dff, self._fps)

    # -- Slider / entry binding --------------------------------------------

    def _on_slider(self, _value: str) -> None:
        # Throttle redraws while the slider is being dragged.
        if self._slider_job is not None:
            self.after_cancel(self._slider_job)
        self._slider_job = self.after(80, self._apply_cutoff)

    def _on_entry(self, _event=None) -> None:
        try:
            val = float(self.cutoff_str.get())
        except ValueError:
            self.status_var.set("Cutoff must be a number.")
            return
        val = max(self.CUTOFF_MIN, min(self.CUTOFF_MAX, val))
        self.cutoff_var.set(val)
        self._apply_cutoff()

    def _apply_cutoff(self) -> None:
        cutoff = float(self.cutoff_var.get())
        cutoff = max(self.CUTOFF_MIN, min(self.CUTOFF_MAX, cutoff))
        self.cutoff_str.set(f"{cutoff:.3f}")
        if self._mean_dff is None:
            return
        self._update_cutoff_marker(cutoff)
        self._draw_lowpass()

    # -- Plotting -----------------------------------------------------------

    def _draw_fft(self) -> None:
        if self._fft_xf is None or self._fft_power is None:
            return
        ax = self.fft_ax
        ax.clear(); ax.set_axis_on()
        # Skip the DC bin so log scale doesn't blow up.
        xf = self._fft_xf[1:]; power = self._fft_power[1:]
        ax.semilogy(xf, np.maximum(power, 1e-12), lw=0.8, color="black")
        ax.set_xlim(0, min(self._fps / 2.0, 15.0))
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("Power (log)", fontsize=8)
        ax.set_title(f"FFT - {self._trace_label}", fontsize=9)
        ax.tick_params(labelsize=7)
        self._cutoff_line = ax.axvline(
            self.cutoff_var.get(), color="tab:red", lw=1.2,
            linestyle="--", label=f"cutoff {self.cutoff_var.get():.2f} Hz")
        ax.legend(loc="upper right", fontsize=7)
        self.fft_canvas.draw_idle()

    def _update_cutoff_marker(self, cutoff: float) -> None:
        if self._cutoff_line is None:
            return
        self._cutoff_line.set_xdata([cutoff, cutoff])
        self._cutoff_line.set_label(f"cutoff {cutoff:.2f} Hz")
        leg = self.fft_ax.get_legend()
        if leg is not None:
            leg.get_texts()[0].set_text(f"cutoff {cutoff:.2f} Hz")
        self.fft_canvas.draw_idle()

    def _draw_raw(self) -> None:
        if self._mean_dff is None:
            return
        t = np.arange(self._mean_dff.size, dtype=np.float32) / self._fps
        ax = self.raw_ax
        ax.clear(); ax.set_axis_on()
        ax.plot(t, self._mean_dff, lw=0.6, color="black")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("dF/F", fontsize=8)
        ax.set_title(f"Raw dF/F - {self._trace_label}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, t[-1] if t.size else 1.0)
        self.raw_canvas.draw_idle()

    def _draw_lowpass(self) -> None:
        if self._mean_dff is None:
            return
        import utils
        cutoff = float(self.cutoff_var.get())
        order = int(self._params.get("filter_order", 2))
        lp, _, _ = utils.lowpass_causal_1d(
            self._mean_dff, fps=self._fps, cutoff_hz=cutoff,
            order=order, zi=None, sos=None)
        t = np.arange(lp.size, dtype=np.float32) / self._fps
        ax = self.lp_ax
        ax.clear(); ax.set_axis_on()
        ax.plot(t, lp, lw=0.7, color="tab:blue")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel(f"dF/F low-pass @ {cutoff:.2f} Hz", fontsize=8)
        ax.set_title(f"Low-pass - {self._trace_label}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, t[-1] if t.size else 1.0)
        self.lp_canvas.draw_idle()

    def _on_source_change(self) -> None:
        """Reload the trace from the new source and redraw all panels."""
        if self._plane0 is None:
            return
        try:
            self._load_data(self._plane0)
        except Exception as e:
            self.status_var.set(f"Source switch error: {e}")
            return
        self._draw_fft(); self._draw_raw(); self._draw_lowpass()
        self.status_var.set(
            f"Source: {self._trace_label}  "
            f"T={self._mean_dff.size}  fps={self._fps:.2f}  "
            f"cutoff={self.cutoff_var.get():.3f} Hz")

    def _on_manual_entry(self, _event=None) -> None:
        """User typed in the Manual ROI # entry: snap source to manual and
        reload. Also fired on FocusOut so clicking elsewhere applies the
        value without forcing the user to hit Enter first."""
        if self._plane0 is None:
            return
        self.source_var.set("manual")
        self._on_source_change()

    # -- Advanced parameters ------------------------------------------------

    def _on_advanced(self) -> None:
        if not _open_advanced(
                self, "Low-pass filter - Advanced parameters",
                self.PARAM_SPEC, self._params):
            return
        # Apply slider bounds if they changed.
        new_min = float(self._params.get("cutoff_min", self.CUTOFF_MIN))
        new_max = float(self._params.get("cutoff_max", self.CUTOFF_MAX))
        if new_min < new_max:
            self.CUTOFF_MIN = new_min
            self.CUTOFF_MAX = new_max
            self.slider.configure(from_=new_min, to=new_max)
            cur = float(self.cutoff_var.get())
            self.cutoff_var.set(max(new_min, min(new_max, cur)))
            self._apply_cutoff()
        self.status_var.set(
            "Advanced parameters updated. The next 'Compute' click will "
            "use the new filter / derivative settings.")

    # -- Compute (writes the lowpass + derivative memmaps) ------------------

    def _on_compute(self) -> None:
        if self._compute_worker is not None and self._compute_worker.is_alive():
            messagebox.showinfo(
                "Busy", "Low-pass / derivative compute already running.")
            return
        if self._plane0 is None or self._n_kept == 0 or self._T == 0:
            messagebox.showerror(
                "Not ready",
                "Load a plane0 with r0p7_filtered_dff.memmap.float32 first.")
            return
        cutoff = float(self.cutoff_var.get())
        cutoff = max(self.CUTOFF_MIN, min(self.CUTOFF_MAX, cutoff))

        plane0 = self._plane0
        T = self._T
        N = self._n_kept
        fps = self._fps

        self.compute_btn.config(state="disabled")
        self.compute_progress.start(12)
        self.compute_status_var.set(
            f"Writing lowpass + derivative @ {cutoff:.3f} Hz ...")

        def worker():
            try:
                self._run_compute(plane0, T, N, fps, cutoff)
                self._compute_queue.put(("done", cutoff))
            except Exception as e:
                self._compute_queue.put(
                    ("error", f"{e}\n{traceback.format_exc()}"))

        self._compute_worker = threading.Thread(target=worker, daemon=True)
        self._compute_worker.start()

    def _run_compute(self, plane0: Path, T: int, N: int,
                     fps: float, cutoff: float) -> None:
        """Causal lowpass + SG first derivative, per ROI, written into
        r0p7_filtered_dff_lowpass.memmap.float32 and
        r0p7_filtered_dff_dt.memmap.float32 (both shape (T, N) float32).
        """
        import utils
        src_path = plane0 / "r0p7_filtered_dff.memmap.float32"
        if not src_path.exists():
            raise FileNotFoundError(
                f"Missing {src_path.name}; run tab 3 first.")
        src = np.memmap(str(src_path), dtype="float32", mode="r",
                        shape=(T, N))

        lp_path = plane0 / "r0p7_filtered_dff_lowpass.memmap.float32"
        dt_path = plane0 / "r0p7_filtered_dff_dt.memmap.float32"
        lp_mm = np.memmap(str(lp_path), dtype="float32", mode="w+",
                          shape=(T, N))
        dt_mm = np.memmap(str(dt_path), dtype="float32", mode="w+",
                          shape=(T, N))

        sg_win_ms = int(self._params.get("sg_win_ms", 333))
        sg_poly = int(self._params.get("sg_poly", 2))
        order = int(self._params.get("filter_order", 2))
        sos = None
        report_every = max(1, N // 20)
        t0 = time.time()
        for i in range(N):
            trace = np.asarray(src[:, i], dtype=np.float32)
            lp, _, sos = utils.lowpass_causal_1d(
                trace, fps=fps, cutoff_hz=cutoff, order=order,
                zi=None, sos=sos)
            dt = utils.sg_first_derivative_1d(
                lp, fps=fps, win_ms=sg_win_ms, poly=sg_poly)
            lp_mm[:, i] = lp
            dt_mm[:, i] = dt
            if (i + 1) % report_every == 0:
                self._compute_queue.put(
                    ("status",
                     f"{i + 1}/{N} ROIs ({time.time() - t0:.1f}s)"))

        lp_mm.flush()
        dt_mm.flush()
        del lp_mm, dt_mm, src

    def _drain_compute_queue(self) -> None:
        try:
            while True:
                kind, payload = self._compute_queue.get_nowait()
                if kind == "status":
                    self.compute_status_var.set(payload)
                elif kind == "done":
                    self.compute_progress.stop()
                    self.compute_btn.config(state="normal")
                    self.compute_status_var.set(
                        f"Wrote r0p7_filtered_dff_lowpass.memmap.float32 + "
                        f"r0p7_filtered_dff_dt.memmap.float32 "
                        f"@ {payload:.3f} Hz")
                    # Notify downstream (event-detection tab).
                    if self._plane0 is not None:
                        try:
                            self.state.set_lowpass_ready(self._plane0)
                        except Exception as e:
                            print(f"lowpass_ready publish error: {e}")
                elif kind == "error":
                    self.compute_progress.stop()
                    self.compute_btn.config(state="normal")
                    self.compute_status_var.set("Error.")
                    messagebox.showerror(
                        "Compute failed", payload.split("\n", 1)[0])
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._drain_compute_queue)


# ---------------------------------------------------------------------------
# Tab 5: Event detection
# ---------------------------------------------------------------------------

class EventDetectionTab(ttk.Frame):
    """Three panels:
        1. Sorted lowpass dF/F heatmap (mirrors image_all overview_heatmap).
        2. Sorted event raster (per-ROI hysteresis onsets, 1 px per onset).
        3. plot_event_detection from utils — population event-window
           detection on the per-ROI onset stream.
    Reads plane0/r0p7_filtered_dff_lowpass.memmap.float32 +
    r0p7_filtered_dff_dt.memmap.float32 produced by tab 4.
    """

    POLL_MS = 80
    Z_ENTER = 3.5
    Z_EXIT = 1.5
    MIN_SEP_S = 0.1
    TIME_COLS_TARGET = 1200

    PARAM_SPEC: list = [
        # Per-ROI onset hysteresis (mad_z + hysteresis_onsets)
        {"name": "z_enter", "label": "Hysteresis enter (z)",
         "type": "float", "default": 3.5, "group": "Per-ROI hysteresis",
         "help": "robust z threshold for entering an event"},
        {"name": "z_exit", "label": "Hysteresis exit (z)",
         "type": "float", "default": 1.5, "group": "Per-ROI hysteresis",
         "help": "lower threshold to exit (hysteresis)"},
        {"name": "min_sep_s", "label": "Min separation (s)",
         "type": "float", "default": 0.1, "group": "Per-ROI hysteresis",
         "help": "merge onsets closer than this into one event"},
        # Display
        {"name": "time_cols_target", "label": "Heatmap time columns",
         "type": "int", "default": 1200, "group": "Display"},
        # Population event detection (utils.EventDetectionParams).
        # Density construction
        {"name": "bin_sec", "label": "bin_sec",
         "type": "float", "default": 0.05,
         "group": "Population events - density",
         "help": "histogram bin width for the onset density (s)"},
        {"name": "smooth_sigma_bins", "label": "smooth_sigma_bins",
         "type": "float", "default": 2.0,
         "group": "Population events - density",
         "help": "Gaussian smoothing sigma in bins"},
        {"name": "normalize_by_num_rois",
         "label": "normalize_by_num_rois",
         "type": "bool", "default": True,
         "group": "Population events - density"},
        # Peak detection (scipy.signal.find_peaks)
        {"name": "min_prominence", "label": "min_prominence",
         "type": "float", "default": 0.007,
         "group": "Population events - peaks"},
        {"name": "min_width_bins", "label": "min_width_bins",
         "type": "float", "default": 2.0,
         "group": "Population events - peaks"},
        {"name": "min_distance_bins", "label": "min_distance_bins",
         "type": "float", "default": 3.0,
         "group": "Population events - peaks"},
        # Baseline / noise
        {"name": "baseline_mode", "label": "baseline_mode",
         "type": "choice", "choices": ["rolling", "global"],
         "default": "rolling", "group": "Population events - baseline"},
        {"name": "baseline_percentile", "label": "baseline_percentile",
         "type": "float", "default": 5.0,
         "group": "Population events - baseline"},
        {"name": "baseline_window_s", "label": "baseline_window_s",
         "type": "float", "default": 30.0,
         "group": "Population events - baseline",
         "help": "rolling-mode window length"},
        {"name": "noise_quiet_percentile",
         "label": "noise_quiet_percentile",
         "type": "float", "default": 40.0,
         "group": "Population events - baseline"},
        {"name": "noise_mad_factor", "label": "noise_mad_factor",
         "type": "float", "default": 1.4826,
         "group": "Population events - baseline"},
        # Boundary walking
        {"name": "end_threshold_k", "label": "end_threshold_k",
         "type": "float", "default": 2.0,
         "group": "Population events - boundaries",
         "help": "end = baseline + k * noise"},
        {"name": "max_event_duration_s",
         "label": "max_event_duration_s",
         "type": "float", "default": 10.0,
         "group": "Population events - boundaries"},
        {"name": "merge_gap_s", "label": "merge_gap_s",
         "type": "float", "default": 0.0,
         "group": "Population events - boundaries",
         "help": "merge consecutive events closer than this (s)"},
        # Gaussian-fit refinement
        {"name": "use_gaussian_boundary",
         "label": "use_gaussian_boundary",
         "type": "bool", "default": True,
         "group": "Population events - gaussian fit"},
        {"name": "gaussian_quantile", "label": "gaussian_quantile",
         "type": "float", "default": 0.99,
         "group": "Population events - gaussian fit"},
        {"name": "gaussian_fit_pad_s", "label": "gaussian_fit_pad_s",
         "type": "float", "default": 0.5,
         "group": "Population events - gaussian fit"},
        {"name": "gaussian_min_sigma_s",
         "label": "gaussian_min_sigma_s",
         "type": "float", "default": 0.05,
         "group": "Population events - gaussian fit"},
    ]

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, padding=10)
        self.state = state
        self._plane0: Optional[Path] = None
        self._render_queue: queue.Queue = queue.Queue()
        self._render_worker: Optional[threading.Thread] = None
        self._params: dict = _spec_defaults(self.PARAM_SPEC)
        self._last_data: Optional[dict] = None

        self._build_ui()
        self.after(self.POLL_MS, self._drain_render_queue)
        state.subscribe_plane0(self._on_plane0)
        state.subscribe_lowpass_ready(self._on_plane0)
        if state.lowpass_plane0 is not None:
            self._on_plane0(state.lowpass_plane0)
        elif state.plane0 is not None:
            self._on_plane0(state.plane0)

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        header = ttk.LabelFrame(
            self, text="Event detection (uses filtered lowpass + derivative)",
            padding=8)
        header.pack(fill="x", pady=(0, 6))

        row = ttk.Frame(header); row.pack(fill="x", pady=2)
        self.render_btn = ttk.Button(
            row, text="Render", command=self._on_render,
            state="disabled")
        self.render_btn.pack(side="left")
        self.render_progress = ttk.Progressbar(
            row, mode="indeterminate", length=160)
        self.render_progress.pack(side="left", padx=12)
        self.status_var = tk.StringVar(
            value="Compute lowpass + derivative on tab 4 first.")
        ttk.Label(row, textvariable=self.status_var,
                  font=("", 9, "italic")).pack(side="left")
        ttk.Button(row, text="Advanced...",
                   command=self._on_advanced).pack(side="right",
                                                   padx=(0, 6))
        self.summary_btn = ttk.Button(
            row, text="Save summary",
            command=self._on_save_summary, state="disabled")
        self.summary_btn.pack(side="right", padx=(0, 6))
        ttk.Button(row, text="Reload from folder...",
                   command=self._reload_from_folder).pack(side="right")

        body = ttk.Frame(self); body.pack(fill="both", expand=True)
        # Panels 1+2 are summary heatmaps that don't need much height;
        # panel 3 is the interactive event-detection trace, so it gets
        # most of the vertical space and a matplotlib navigation toolbar
        # for pan / zoom / box-zoom (mirrors plt.show()).
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        body.rowconfigure(2, weight=4)
        body.columnconfigure(0, weight=1)

        # Panel 1: overview heatmap
        f1 = ttk.LabelFrame(
            body, text="1. Filtered lowpass dF/F heatmap "
                       "(sorted by event count)", padding=4)
        f1.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        self.hm_fig = plt.Figure(figsize=(8, 1.6), tight_layout=True)
        self.hm_ax = self.hm_fig.add_subplot(111)
        self.hm_ax.set_axis_off()
        self.hm_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=self.hm_ax.transAxes)
        self.hm_canvas = FigureCanvasTkAgg(self.hm_fig, master=f1)
        _attach_fig_toolbar(self.hm_canvas, f1)
        self.hm_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Panel 2: event raster
        f2 = ttk.LabelFrame(
            body, text="2. Filtered event raster (sorted by event count)",
            padding=4)
        f2.grid(row=1, column=0, sticky="nsew", pady=(0, 4))
        self.er_fig = plt.Figure(figsize=(8, 1.6), tight_layout=True)
        self.er_ax = self.er_fig.add_subplot(111)
        self.er_ax.set_axis_off()
        self.er_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=self.er_ax.transAxes)
        self.er_canvas = FigureCanvasTkAgg(self.er_fig, master=f2)
        _attach_fig_toolbar(self.er_canvas, f2)
        self.er_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Panel 3: detect_event_windows diagnostics, with toolbar.
        f3 = ttk.LabelFrame(
            body, text="3. Population event detection "
                       "(utils.plot_event_detection)", padding=4)
        f3.grid(row=2, column=0, sticky="nsew")
        self.ed_fig = plt.Figure(figsize=(8, 4.8), tight_layout=True)
        self.ed_ax = self.ed_fig.add_subplot(111)
        self.ed_ax.set_axis_off()
        self.ed_ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=self.ed_ax.transAxes)
        self.ed_canvas = FigureCanvasTkAgg(self.ed_fig, master=f3)
        self.ed_toolbar = _attach_fig_toolbar(self.ed_canvas, f3)
        self.ed_canvas.get_tk_widget().pack(side="top", fill="both",
                                            expand=True)

    # -- Plane0 / readiness handling ---------------------------------------

    def _reload_from_folder(self) -> None:
        path = filedialog.askdirectory(
            title="Select a suite2p plane0 folder containing the "
                  "r0p7_filtered_dff_lowpass / _dt memmaps")
        if not path:
            return
        self._on_plane0(Path(path))

    def _on_plane0(self, plane0: Path) -> None:
        self._plane0 = Path(plane0)
        if self._inputs_ready(self._plane0):
            self.render_btn.config(state="normal")
            self.status_var.set(
                f"Ready. Click Render to compute and plot.  "
                f"({self._plane0})")
        else:
            self.render_btn.config(state="disabled")
            self.status_var.set(
                "Filtered lowpass / derivative memmaps not present yet. "
                "Compute them on tab 4 first.")

    def _inputs_ready(self, plane0: Path) -> bool:
        for name in ("F.npy", "r0p7_filtered_dff_lowpass.memmap.float32",
                     "r0p7_filtered_dff_dt.memmap.float32"):
            if not (plane0 / name).exists():
                return False
        return True

    # -- Advanced parameters ------------------------------------------------

    def _on_advanced(self) -> None:
        if _open_advanced(
                self, "Event detection - Advanced parameters",
                self.PARAM_SPEC, self._params):
            self.status_var.set(
                "Advanced parameters updated. Click Render to recompute "
                "with the new settings.")

    # -- Render worker ------------------------------------------------------

    def _on_render(self) -> None:
        if (self._render_worker is not None
                and self._render_worker.is_alive()):
            messagebox.showinfo("Busy", "Render is already running.")
            return
        if self._plane0 is None or not self._inputs_ready(self._plane0):
            messagebox.showerror(
                "Not ready",
                "Compute lowpass + derivative on tab 4 first.")
            return

        plane0 = self._plane0
        self.render_btn.config(state="disabled")
        self.render_progress.start(12)
        self.status_var.set("Rendering ...")

        def worker():
            try:
                payload = self._compute_render_data(plane0)
                self._render_queue.put(("done", payload))
            except Exception as e:
                self._render_queue.put(
                    ("error", f"{e}\n{traceback.format_exc()}"))

        self._render_worker = threading.Thread(target=worker, daemon=True)
        self._render_worker.start()

    def _compute_render_data(self, plane0: Path) -> dict:
        import utils
        # Discover N_kept from F.npy + mask, T from any memmap-via-shape.
        F = np.load(plane0 / "F.npy", mmap_mode="r")
        N_total, T = F.shape

        mask_path = plane0 / "predicted_cell_mask.npy"
        if mask_path.exists():
            mask = np.load(mask_path).astype(bool)
        else:
            iscell_path = plane0 / "iscell.npy"
            if iscell_path.exists():
                ic = np.load(iscell_path)
                mask = ((ic[:, 0] > 0) if ic.ndim == 2
                        else (ic > 0)).astype(bool)
            else:
                mask = np.ones(N_total, dtype=bool)
        N_kept = int(mask.sum())
        if N_kept == 0:
            raise RuntimeError("No ROIs survive the cell-filter mask.")

        lp_path = plane0 / "r0p7_filtered_dff_lowpass.memmap.float32"
        dt_path = plane0 / "r0p7_filtered_dff_dt.memmap.float32"
        lowpass = np.memmap(str(lp_path), dtype="float32", mode="r",
                            shape=(T, N_kept))
        derivative = np.memmap(str(dt_path), dtype="float32", mode="r",
                               shape=(T, N_kept))

        fps = float(utils.get_fps_from_notes(str(plane0)))

        # Snapshot params on the worker thread for stable reads.
        z_enter = float(self._params.get("z_enter", 3.5))
        z_exit = float(self._params.get("z_exit", 1.5))
        min_sep_s = float(self._params.get("min_sep_s", 0.1))
        time_cols_target = int(
            self._params.get("time_cols_target", 1200))

        # Per-ROI onset times (s) + per-ROI binary raster.
        downsample = max(1, T // time_cols_target)
        num_cols = T // downsample
        heatmap = np.zeros((N_kept, num_cols), dtype=np.uint8)
        raster = np.zeros((N_kept, num_cols), dtype=np.uint8)
        event_counts = np.zeros(N_kept, dtype=np.int32)
        onsets_by_roi: list = []

        for i in range(N_kept):
            lp_i = np.asarray(lowpass[:, i], dtype=np.float32)
            dt_i = np.asarray(derivative[:, i], dtype=np.float32)
            z, _, _ = utils.mad_z(dt_i)
            onsets = utils.hysteresis_onsets(
                z, z_enter, z_exit, fps, min_sep_s=min_sep_s)
            event_counts[i] = onsets.size
            onsets_by_roi.append(
                np.asarray(onsets, dtype=np.float64) / fps)

            if downsample > 1:
                trimmed = lp_i[:num_cols * downsample].reshape(
                    num_cols, downsample)
                lp_ds = trimmed.mean(axis=1)
                if onsets.size:
                    bins = (onsets // downsample).clip(0, num_cols - 1)
                    raster[i, np.unique(bins)] = 1
            else:
                lp_ds = lp_i
                if onsets.size:
                    raster[i, onsets.clip(0, num_cols - 1)] = 1

            lo, hi = np.percentile(lp_ds, [1, 99])
            if hi <= lo:
                heatmap[i, :] = 0
            else:
                norm = np.clip((lp_ds - lo) / (hi - lo), 0, 1)
                heatmap[i, :] = (norm * 255.0 + 0.5).astype(np.uint8)

        order = np.argsort(-event_counts)
        heatmap = heatmap[order]
        raster = raster[order]

        # Build a fully-populated EventDetectionParams from the
        # Advanced dialog's values (all 18 knobs map 1:1 to dataclass
        # fields by name).
        ed_kwargs: dict = {}
        try:
            from utils import EventDetectionParams
            params_obj = EventDetectionParams()
            for field_name in (
                "bin_sec", "smooth_sigma_bins", "normalize_by_num_rois",
                "min_prominence", "min_width_bins", "min_distance_bins",
                "baseline_mode", "baseline_percentile",
                "baseline_window_s", "noise_quiet_percentile",
                "noise_mad_factor", "end_threshold_k",
                "max_event_duration_s", "merge_gap_s",
                "use_gaussian_boundary", "gaussian_quantile",
                "gaussian_fit_pad_s", "gaussian_min_sigma_s",
            ):
                if field_name in self._params:
                    setattr(params_obj, field_name,
                            self._params[field_name])
            ed_kwargs["params"] = params_obj
        except Exception as e:
            print(f"[GUI] EventDetectionParams build failed: {e}")

        event_windows, A, first_time, diagnostics = \
            utils.detect_event_windows(
                onsets_by_roi, T=T, fps=fps,
                return_diagnostics=True, **ed_kwargs,
            )

        return {
            "heatmap": heatmap,
            "raster": raster,
            "event_counts": event_counts,
            "diagnostics": diagnostics,
            "event_windows": event_windows,
            "onsets_by_roi": onsets_by_roi,
            "fps": fps,
            "T": T,
            "N_kept": N_kept,
            "downsample": downsample,
        }

    def _drain_render_queue(self) -> None:
        try:
            while True:
                kind, payload = self._render_queue.get_nowait()
                if kind == "done":
                    self._on_render_done(payload)
                elif kind == "error":
                    self.render_progress.stop()
                    self.render_btn.config(state="normal")
                    self.status_var.set("Render error.")
                    messagebox.showerror(
                        "Render failed", payload.split("\n", 1)[0])
        except queue.Empty:
            pass
        self.after(self.POLL_MS, self._drain_render_queue)

    def _on_render_done(self, data: dict) -> None:
        import utils
        self.render_progress.stop()
        self.render_btn.config(state="normal")

        hm = data["heatmap"]
        rs = data["raster"]
        N = data["N_kept"]; T = data["T"]
        ds = data["downsample"]; fps = data["fps"]

        ax = self.hm_ax
        ax.clear(); ax.set_axis_on()
        ax.imshow(hm, aspect="auto", interpolation="nearest")
        ax.set_xlabel(f"Time (downsampled bins ~{ds * 1000.0 / fps:.0f} ms)",
                      fontsize=8)
        ax.set_ylabel("ROIs (most active at top)", fontsize=8)
        ax.tick_params(labelsize=7)
        self.hm_canvas.draw_idle()

        ax = self.er_ax
        ax.clear(); ax.set_axis_on()
        ax.imshow(rs, aspect="auto", interpolation="nearest", cmap="Greys")
        ax.set_xlabel(f"Time (downsampled bins ~{ds * 1000.0 / fps:.0f} ms)",
                      fontsize=8)
        ax.set_ylabel("ROIs (most active at top)", fontsize=8)
        ax.tick_params(labelsize=7)
        self.er_canvas.draw_idle()

        ax = self.ed_ax
        ax.clear(); ax.set_axis_on()
        duration_s = float(T) / float(fps) if fps > 0 else 1.0
        try:
            utils.plot_event_detection(data["diagnostics"], ax=ax)
            # Highlight detected event windows as shaded vertical bands.
            ev = data["event_windows"]
            if ev is not None and len(ev) > 0:
                utils.shade_event_windows(ax, ev, color="C1", alpha=0.20)
            # Force the time axis to span the full recording so the trace
            # doesn't look cut off when the auto-fit picks up only the
            # smoothed-density extent.
            ax.set_xlim(0.0, duration_s)
            ax.relim(); ax.autoscale_view(scalex=False, scaley=True)
        except Exception as e:
            ax.text(0.5, 0.5, f"plot_event_detection error:\n{e}",
                    ha="center", va="center", transform=ax.transAxes)
        self.ed_canvas.draw_idle()
        # Reset the toolbar's home/back/forward stack so 'home' returns
        # to this fresh view (not the empty placeholder it captured at
        # widget construction time).
        try:
            self.ed_toolbar.update()
        except Exception:
            pass

        n_events = (len(data["event_windows"])
                    if data["event_windows"] is not None else 0)
        self.status_var.set(
            f"Rendered  N_kept={N}  T={T}  fps={fps:.2f}  "
            f"duration={duration_s:.1f}s  events={n_events}")

        # Cache for manual "Save summary" + auto-write the events sheets.
        self._last_data = data
        self.summary_btn.config(state="normal")
        if self._plane0 is not None:
            try:
                self._write_summary(self._plane0, data)
            except Exception as e:
                print(f"[GUI] event summary write failed: {e}")

    # -- Summary export -----------------------------------------------------

    def _write_summary(self, plane0: Path, data: dict) -> Path:
        summary_writer.update_recording_meta(
            plane0, fps=data.get("fps"),
            T=data.get("T"), N=data.get("N_kept"))
        return summary_writer.write_events_sheets(
            plane0,
            event_windows=data.get("event_windows"),
            onsets_by_roi=data.get("onsets_by_roi") or [],
            fps=data.get("fps"),
            in_seconds=True,
        )

    def _on_save_summary(self) -> None:
        if self._plane0 is None or self._last_data is None:
            messagebox.showinfo(
                "No data", "Click 'Render' first to compute event windows.")
            return
        try:
            path = self._write_summary(self._plane0, self._last_data)
            self.status_var.set(f"Summary -> {path}")
        except Exception as e:
            messagebox.showerror("Summary failed", str(e))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def _render_emoji_icon(emoji: str, size: int = 128) -> Optional[ImageTk.PhotoImage]:
    """Render ``emoji`` to an RGBA PhotoImage for use with ``Tk.iconphoto``.

    Uses Segoe UI Emoji on Windows (color bitmap glyphs, requires
    Pillow >= 9 for ``embedded_color=True``). Returns None on failure.
    """
    try:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        font = None
        for candidate in ("seguiemj.ttf", "AppleColorEmoji.ttf",
                          "NotoColorEmoji.ttf"):
            try:
                font = ImageFont.truetype(candidate, int(size * 0.8))
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        try:
            draw.text((size // 2, size // 2), emoji, font=font,
                      anchor="mm", embedded_color=True)
        except TypeError:
            draw.text((size // 2, size // 2), emoji, font=font,
                      anchor="mm", fill=(0, 0, 0, 255))

        return ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"icon render failed: {e}")
        return None


class PipelineApp(tk.Tk):
    APP_NAME = "CalLIOPE"
    APP_SUBTITLE = "Calcium Live-imaging Output Pipeline for Epileptiform-recordings"
    APP_EMOJI = "\U0001F3A7"  # headphones

    def __init__(self) -> None:
        super().__init__()
        self.title(f"{self.APP_EMOJI} {self.APP_NAME} - {self.APP_SUBTITLE}")
        self.geometry("1100x720")

        # Headphone window/taskbar icon.
        self._icon_photo = _render_emoji_icon(self.APP_EMOJI, size=128)
        if self._icon_photo is not None:
            try:
                self.iconphoto(True, self._icon_photo)
            except Exception as e:
                print(f"iconphoto failed: {e}")

        self.state_obj = AppState()

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        from clustering_tab import ClusteringTab
        from crosscorrelation_tab import CrossCorrelationTab

        self.preprocess_tab = PreprocessTab(nb, self.state_obj)
        self.qc_tab = QcTab(nb, self.state_obj)
        self.detection_tab = Suite2pTab(nb, self.state_obj)
        self.lowpass_tab = LowpassTab(nb, self.state_obj)
        self.event_tab = EventDetectionTab(nb, self.state_obj)
        self.clustering_tab = ClusteringTab(nb, self.state_obj)
        self.xcorr_tab = CrossCorrelationTab(nb, self.state_obj)

        nb.add(self.preprocess_tab, text="1. Input & Preprocess")
        nb.add(self.qc_tab, text="2. QC Preview")
        nb.add(self.detection_tab, text="3. Suite2p Detection")
        nb.add(self.lowpass_tab, text="4. Low-pass filter")
        nb.add(self.event_tab, text="5. Event detection")
        nb.add(self.clustering_tab, text="6. Clustering")
        nb.add(self.xcorr_tab, text="7. Cross-correlation")


def main() -> None:
    PipelineApp().mainloop()


if __name__ == "__main__":
    main()
