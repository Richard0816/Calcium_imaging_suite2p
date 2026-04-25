"""
pipeline_gui.py - top-level GUI for the calcium imaging pipeline.

Current tabs
------------
1. Input & Preprocess   - pick working directory, choose a TIFF, run shift + QC.
2. QC Preview           - animated GIF of the shifted movie + blob detection
                          overlaid on the mean image.
3. Suite2p Detection    - sparse_plus_cellpose detection + dF/F + cell-filter
                          prediction, with live console + before/after ROI maps.

The GUI is organised so that each tab is a self-contained class; new pipeline
stages (traces, clustering, event detection, etc.) can be dropped in as
additional ``ttk.Frame`` subclasses without touching the rest.
"""

from __future__ import annotations

import contextlib
import io
import queue
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw, ImageFont, ImageTk

import preprocessing
from preprocessing import PreprocessResult


# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------

class AppState:
    """Mutable state shared across tabs. Tabs subscribe via ``on_result``."""

    def __init__(self) -> None:
        self.working_dir: Optional[Path] = None
        self.data_root: Optional[Path] = None
        self.selected_tiff: Optional[Path] = None
        self.result: Optional[PreprocessResult] = None
        self._listeners: list = []

    def subscribe(self, fn) -> None:
        self._listeners.append(fn)

    def set_result(self, result: PreprocessResult) -> None:
        self.result = result
        for fn in self._listeners:
            try:
                fn(result)
            except Exception as e:
                print(f"listener error: {e}")


# ---------------------------------------------------------------------------
# Tab 1: Input & Preprocess
# ---------------------------------------------------------------------------

class PreprocessTab(ttk.Frame):
    POLL_MS = 80

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, padding=10)
        self.state = state
        self._log_queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None

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

        # TIFF selection row
        tiff_frame = ttk.LabelFrame(self, text="2. TIFF file", padding=8)
        tiff_frame.pack(fill="x", pady=(0, 8))

        self.tiff_var = tk.StringVar()
        self.tiff_combo = ttk.Combobox(tiff_frame, textvariable=self.tiff_var,
                                       state="readonly")
        self.tiff_combo.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(tiff_frame, text="Refresh", command=self._refresh_tiffs).grid(
            row=0, column=1)
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
        if not wd:
            return
        tiffs = preprocessing.list_tiffs(wd)
        names = [t.name for t in tiffs]
        self.tiff_combo["values"] = names
        if names:
            self.tiff_combo.current(0)
            self.tiff_var.set(names[0])
        else:
            self.tiff_var.set("")
            self._append_log(f"No .tif/.tiff files in {wd}")

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "Preprocessing is already running.")
            return

        wd = self.dir_var.get().strip()
        name = self.tiff_var.get().strip()
        data_root = self.data_root_var.get().strip()

        if not wd or not name:
            messagebox.showerror("Missing input",
                                 "Select a working directory and TIFF file.")
            return
        if not data_root:
            messagebox.showerror("Missing output", "Set an output root directory.")
            return

        src = Path(wd) / name
        self.state.selected_tiff = src
        self.state.data_root = Path(data_root)

        self.run_btn.config(state="disabled")
        self.progress.start(12)
        self.status_var.set("Running...")
        self._append_log(f"--- Preprocessing {src.name} ---")

        def worker():
            try:
                result = preprocessing.preprocess_tiff(
                    src_tiff=src,
                    data_root=data_root,
                    progress_cb=lambda m: self._log_queue.put(("log", m)),
                )
                self._log_queue.put(("done", result))
            except Exception as e:
                self._log_queue.put(("error", str(e)))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

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
        self.status_var.set(
            f"Done - {result.n_blobs} preview blobs, {result.n_frames} frames.")
        self._append_log(
            f"Outputs: {result.qc_gif.name}, "
            f"{result.shifted_tiff.name}, mean.npy, blobs.npy")
        self.state.set_result(result)

    def _on_error(self, msg: str) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")
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
# Tab 3: Suite2p Detection (sparse_plus_cellpose + dF/F + cell filter)
# ---------------------------------------------------------------------------

# Stream stdout/stderr from the worker thread into the log queue line-by-line.
# Suite2p, cellpose, and analyze_output all use plain print() to report
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

    def __init__(self, master, state: AppState) -> None:
        super().__init__(master, padding=10)
        self.state = state
        self._log_queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._final_plane0: Optional[Path] = None

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

        # Run + status
        row = ttk.Frame(header); row.pack(fill="x", pady=(4, 0))
        self.run_btn = ttk.Button(
            row, text="Run detection + cell filter",
            command=self._on_run, state="disabled")
        self.run_btn.pack(side="left")
        self.progress = ttk.Progressbar(row, mode="indeterminate", length=200)
        self.progress.pack(side="left", padx=12)
        self.status_var = tk.StringVar(value="Run preprocessing first.")
        ttk.Label(row, textvariable=self.status_var).pack(side="left")

        # Body: panel 1 (console) on top, panels 2+3 (image) below
        body = ttk.Frame(self); body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=2)
        body.rowconfigure(1, weight=3)
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

        # Panel 2: detected ROIs
        det_frame = ttk.LabelFrame(
            body, text="2. Detected ROIs (raw suite2p output)", padding=6)
        det_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 3))
        self.det_fig = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.det_ax = self.det_fig.add_subplot(111)
        self.det_ax.set_axis_off()
        self.det_ax.text(0.5, 0.5, "No detection yet", ha="center",
                         va="center", transform=self.det_ax.transAxes)
        self.det_canvas = FigureCanvasTkAgg(self.det_fig, master=det_frame)
        self.det_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Panel 3: filtered ROIs
        fil_frame = ttk.LabelFrame(
            body, text="3. After cell-filter prediction mask", padding=6)
        fil_frame.grid(row=1, column=1, sticky="nsew", padx=(3, 0))
        self.fil_fig = plt.Figure(figsize=(5, 5), tight_layout=True)
        self.fil_ax = self.fil_fig.add_subplot(111)
        self.fil_ax.set_axis_off()
        self.fil_ax.text(0.5, 0.5, "No filter applied yet", ha="center",
                         va="center", transform=self.fil_ax.transAxes)
        self.fil_canvas = FigureCanvasTkAgg(self.fil_fig, master=fil_frame)
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
        self.status_var.set(
            f"Ready: {result.shifted_tiff.name}  "
            f"({result.shape_yx[0]}x{result.shape_yx[1]})")
        self.run_btn.config(state="normal")

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
    ) -> None:
        writer = _QueueWriter(self._log_queue)
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
                aav_info_csv=aav_csv,
                tau_vals=spc.DEFAULT_TAU_VALS,
                verbose=True,
            )
            print(f"[GUI] suite2p plane0 -> {final_plane0}")
            self._log_queue.put(("plane0", final_plane0))

            # 2. dF/F + low-pass + d/dt -> r0p7_*.memmap.float32
            print("[GUI] running dF/F + low-pass + d/dt...")
            from analyze_output import run_analysis_on_folder
            recording_folder = final_plane0.parent.parent  # <save>/final
            run_analysis_on_folder(str(recording_folder))
            print("[GUI] dF/F memmaps written")

            # 3. Cell-filter prediction (writes predicted_cell_mask.npy)
            if ckpt_path:
                print("[GUI] running cell-filter prediction...")
                self._run_cellfilter(final_plane0, ckpt_path, rec_id)
                print("[GUI] predicted_cell_mask.npy written")

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

    def _on_error(self, msg: str) -> None:
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status_var.set("Error.")
        self._append_log(f"ERROR: {msg}")
        messagebox.showerror("Detection failed", msg.split("\n", 1)[0])

    # -- Panel rendering ----------------------------------------------------

    def _draw_panels(self, plane0: Path) -> None:
        """Render panels 2 and 3 from suite2p outputs at ``plane0``."""
        ops = np.load(plane0 / "ops.npy", allow_pickle=True).item()
        stat = np.load(plane0 / "stat.npy", allow_pickle=True)
        mean = np.asarray(
            ops.get("meanImgE", ops.get("meanImg")), dtype=np.float32)
        Ly, Lx = mean.shape
        n_total = len(stat)

        # Panel 2: every detected ROI
        label_all = self._build_label_image(stat, Ly, Lx)

        # Panel 3: ROIs surviving the cell-filter mask (or fall back to iscell)
        keep = self._load_keep_mask(plane0, n_total)
        kept_stat = [s for i, s in enumerate(stat) if keep[i]]
        label_kept = self._build_label_image(kept_stat, Ly, Lx)

        vmax = float(np.quantile(mean, 0.995))
        vmin = float(np.quantile(mean, 0.01))
        self._render_panel(
            self.det_ax, self.det_canvas, mean, label_all, vmin, vmax,
            f"All detected ROIs (n = {n_total})")
        kept_n = int(keep.sum())
        keep_src = ("predicted_cell_mask.npy"
                    if (plane0 / "predicted_cell_mask.npy").exists()
                    else "iscell.npy (suite2p classifier)")
        self._render_panel(
            self.fil_ax, self.fil_canvas, mean, label_kept, vmin, vmax,
            f"After filter (n = {kept_n} / {n_total})  [{keep_src}]")

    @staticmethod
    def _build_label_image(stat, Ly: int, Lx: int) -> np.ndarray:
        label = np.zeros((Ly, Lx), dtype=np.int32)
        for i, s in enumerate(stat, start=1):
            yp = np.asarray(s["ypix"]); xp = np.asarray(s["xpix"])
            ok = (yp >= 0) & (yp < Ly) & (xp >= 0) & (xp < Lx)
            label[yp[ok], xp[ok]] = i
        return label

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

        self.preprocess_tab = PreprocessTab(nb, self.state_obj)
        self.qc_tab = QcTab(nb, self.state_obj)
        self.detection_tab = Suite2pTab(nb, self.state_obj)

        nb.add(self.preprocess_tab, text="1. Input & Preprocess")
        nb.add(self.qc_tab, text="2. QC Preview")
        nb.add(self.detection_tab, text="3. Suite2p Detection")


def main() -> None:
    PipelineApp().mainloop()


if __name__ == "__main__":
    main()
