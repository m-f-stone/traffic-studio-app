#!/usr/bin/env python3
"""
roi_tool_gui.py  —  ROI Studio with Zoom, Resizable UI, and Open Image/Video

• Open a VIDEO (mp4/mov/mkv/avi) OR a still IMAGE (jpg/png/bmp/tif) from the GUI
• Draw multiple color-coded polygon ROIs over the current frame/image
• Semi-transparent fill + vertex handles while drawing
• Load/Save ROIs (new schema, legacy compatible)
• Mask preview toggle (darken outside included ROIs)
• Export a union mask PNG; batch-apply it to a folder of images
• Zoom slider, Ctrl+MouseWheel zoom, Fit-to-window, scrollbars, resizable window
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog

# -------------------------
# Utility helpers
# -------------------------
def color_for_name(name: str) -> tuple[int,int,int]:
    """Deterministic bright-ish BGR color from a name."""
    h = abs(hash(name))
    b = 60 + (h % 196); g = 60 + ((h // 101) % 196); r = 60 + ((h // 10007) % 196)
    return int(b), int(g), int(r)

def ensure_polygon(points):
    """Return cleaned list of (x,y) int tuples; drop duplicate terminal point."""
    out = []
    for (x, y) in points:
        x = int(round(x)); y = int(round(y))
        if not out or (out[-1][0] != x or out[-1][1] != y):
            out.append((x, y))
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out

def legacy_to_new(json_data):
    """Convert legacy {name: [[x,y],...]} to new schema with include=true."""
    if "rois" in json_data:
        return json_data
    rois = {}
    for name, pts in json_data.items():
        if name == "calibration": continue
        rois[name] = {"points": ensure_polygon(pts), "include": True}
    return {"meta": {}, "rois": rois}

def draw_filled_polygons(base_bgr, rois, alpha=0.25):
    """Overlay semi-transparent filled polygons for all visible ROIs."""
    overlay = base_bgr.copy()
    drew = False
    for name, spec in rois.items():
        pts = spec.get("points") or []
        if len(pts) >= 3 and spec.get("visible", True):
            color = spec.get("color", color_for_name(name))
            arr = np.array(pts, np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [arr], color)
            drew = True
    if drew:
        return cv2.addWeighted(overlay, alpha, base_bgr, 1 - alpha, 0)
    return base_bgr

def union_mask(shape_hw, rois):
    """Return a binary uint8 mask (255 inside union of included ROIs)."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for name, spec in rois.items():
        if not spec.get("include", True): continue
        pts = spec.get("points") or []
        if len(pts) >= 3:
            arr = np.array(pts, np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [arr], 255)
    return mask

def apply_mask_to_image(img_bgr, mask):
    """Black out pixels outside mask (keeps original inside)."""
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

# -------------------------
# Main GUI
# -------------------------
class ROIStudio(tk.Tk):
    def __init__(self, media_path: Path | None, load_json: Path | None, save_json: Path | None):
        super().__init__()
        self.title("ROI Studio — Draw, Preview, Export Mask")
        self.configure(background="#202225")
        self.resizable(True, True)

        # media state
        self.media_path: Path | None = None
        self.is_video = False
        self.cap = None                # cv2.VideoCapture (if video)
        self.total_frames = 1
        self.idx = 0
        self.frame = None
        self.h = self.w = 0

        # ROI state
        self.rois: dict[str, dict] = {}
        self.save_json_path = Path(save_json) if save_json else None
        if load_json:
            # ROIs will be loaded AFTER media (so colors scale properly)
            self._pending_roi_load = Path(load_json)
        else:
            self._pending_roi_load = None

        # drawing state
        self.drawing = False
        self.current_points: list[tuple[int,int]] = []

        # zoom state
        self.zoom = 1.0
        self.fit_mode = False

        # UI
        self._build_ui()
        # Open initial media (if provided), else ask user to open
        if media_path:
            self.load_media(Path(media_path))
        else:
            # show empty canvas until user opens something
            self._blank_canvas()
        # Load ROIs if requested
        if self._pending_roi_load:
            self.load_from_json(self._pending_roi_load)
            self._pending_roi_load = None

    # ---------- UI ----------
    def _build_ui(self):
        # Menu bar
        menubar = tk.Menu(self)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Open Image…", command=self._open_image_dialog)
        filem.add_command(label="Open Video…", command=self._open_video_dialog)
        filem.add_separator()
        filem.add_command(label="Load ROIs…", command=self.load_json_dialog)
        filem.add_command(label="Save ROIs", command=self.save_to_json)
        filem.add_separator()
        filem.add_command(label="Export Union Mask…", command=self.export_mask_png)
        filem.add_separator()
        filem.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filem)
        self.config(menu=menubar)

        # Shortcuts
        self.bind_all("<Control-s>", lambda e: self.save_to_json())
        self.bind_all("<Control-o>", lambda e: self.load_json_dialog())
        # Frame navigation shortcuts
        self.bind_all("<Left>", lambda e: self.prev_frame())
        self.bind_all("<Right>", lambda e: self.next_frame())
        self.bind_all("<Control-Left>", lambda e: self.jump_backward())
        self.bind_all("<Control-Right>", lambda e: self.jump_forward())

        # Layout: Canvas (left) & Controls (right)
        self.columnconfigure(0, weight=1); self.rowconfigure(0, weight=1)
        main = ttk.Frame(self); main.grid(sticky="nsew", padx=8, pady=8)
        main.columnconfigure(0, weight=1); main.rowconfigure(0, weight=1)

        # Canvas + scrollbars
        left = ttk.Frame(main); left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1); left.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(left, bg="#000000", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar = ttk.Scrollbar(left, orient="vertical", command=self.canvas.yview)
        xbar = ttk.Scrollbar(left, orient="horizontal", command=self.canvas.xview)
        ybar.grid(row=0, column=1, sticky="ns"); xbar.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)

        # Mouse & zoom
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Double-Button-1>", self.on_double_left)
        self.canvas.bind("<Control-MouseWheel>", self.on_ctrl_wheel)
        self.bind("<plus>", lambda e: self.set_zoom(self.zoom * 1.1))
        self.bind("<minus>", lambda e: self.set_zoom(self.zoom / 1.1))
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # Right panel
        side = ttk.Frame(main); side.grid(row=0, column=1, sticky="nsw", padx=(10,0))
        # Nav + slider
        nav = ttk.LabelFrame(side, text="Frame Navigation")
        nav.pack(fill="x", pady=(0,8))

        # Frame slider
        self.slider = ttk.Scale(nav, from_=0, to=0, command=self._on_slider, length=240)
        self.slider.grid(row=0, column=0, columnspan=6, sticky="ew", padx=4, pady=4)

        # Frame navigation buttons (row 1: single step)
        self.btn_prev = ttk.Button(nav, text="◀ Prev", command=self.prev_frame, width=8)
        self.btn_prev.grid(row=1, column=0, columnspan=3, sticky="ew", padx=(4,2), pady=(2,0))
        self.btn_next = ttk.Button(nav, text="Next ▶", command=self.next_frame, width=8)
        self.btn_next.grid(row=1, column=3, columnspan=3, sticky="ew", padx=(2,4), pady=(2,0))

        # Quick scroll controls (row 2: jump increments)
        quick_frame = ttk.Frame(nav)
        quick_frame.grid(row=2, column=0, columnspan=6, sticky="ew", padx=4, pady=(4,4))

        ttk.Label(quick_frame, text="Quick Jump:", font=("", 8)).pack(side="left", padx=(0,4))

        # Jump size selector
        self.jump_size = tk.IntVar(value=10)
        jump_sizes = [("10", 10), ("25", 25), ("50", 50), ("100", 100)]
        for i, (label, val) in enumerate(jump_sizes):
            rb = ttk.Radiobutton(quick_frame, text=label, variable=self.jump_size, value=val, width=4)
            rb.pack(side="left", padx=1)

        # Quick scroll buttons (row 3)
        self.btn_jump_back = ttk.Button(nav, text="◀◀ Back", command=self.jump_backward, width=8)
        self.btn_jump_back.grid(row=3, column=0, columnspan=3, sticky="ew", padx=(4,2), pady=(2,4))
        self.btn_jump_forward = ttk.Button(nav, text="Forward ▶▶", command=self.jump_forward, width=8)
        self.btn_jump_forward.grid(row=3, column=3, columnspan=3, sticky="ew", padx=(2,4), pady=(2,4))

        # Zoom
        zoom_box = ttk.LabelFrame(side, text="Zoom")
        zoom_box.pack(fill="x", pady=(0,8))
        self.zoom_var = tk.DoubleVar(value=100.0)
        zslider = ttk.Scale(zoom_box, from_=10, to=400, variable=self.zoom_var, command=self._on_zoom_slider, length=220)
        zslider.pack(side="left", padx=(6,4), pady=6)
        ttk.Button(zoom_box, text="100%", command=lambda: self.set_zoom(1.0)).pack(side="left", padx=2)
        ttk.Button(zoom_box, text="Fit", command=self.fit_to_window).pack(side="left", padx=2)

        # ROI actions
        roi_actions = ttk.LabelFrame(side, text="ROI")
        roi_actions.pack(fill="x", pady=(0,8))
        ttk.Button(roi_actions, text="➕ New ROI (draw)", command=self.start_draw).pack(fill="x", pady=2)
        ttk.Button(roi_actions, text="✔ Finish ROI", command=self.finish_roi).pack(fill="x", pady=2)
        ttk.Button(roi_actions, text="↶ Undo Point", command=self.undo_point).pack(fill="x", pady=2)

        # ROI list
        list_box = ttk.LabelFrame(side, text="ROIs")
        list_box.pack(fill="both", expand=True)
        self.roi_list = tk.Listbox(list_box, height=10, activestyle="dotbox", exportselection=False)
        self.roi_list.pack(fill="both", expand=True, padx=4, pady=4)
        self.roi_list.bind("<<ListboxSelect>>", lambda e: self._refresh_image())

        toggles = ttk.Frame(list_box); toggles.pack(fill="x", padx=4, pady=(0,6))
        self.include_var = tk.BooleanVar(value=True)
        self.visible_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggles, variable=self.include_var, text="Include in Mask", command=self._toggle_include).pack(anchor="w")
        ttk.Checkbutton(toggles, variable=self.visible_var, text="Visible", command=self._toggle_visible).pack(anchor="w")

        list_btns = ttk.Frame(list_box); list_btns.pack(fill="x", padx=4, pady=(0,6))
        ttk.Button(list_btns, text="Rename", command=self.rename_roi).pack(side="left", expand=True, fill="x", padx=(0,4))
        ttk.Button(list_btns, text="Delete", command=self.delete_roi).pack(side="left", expand=True, fill="x")

        # Mask & Export
        preview = ttk.LabelFrame(side, text="Mask & Export")
        preview.pack(fill="x")
        self.preview_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(preview, variable=self.preview_var, text="Mask Preview (darken outside)", command=self._refresh_image)\
            .pack(anchor="w", padx=4, pady=(4,2))
        ttk.Button(preview, text="Export Union Mask (PNG)...", command=self.export_mask_png).pack(fill="x", padx=4, pady=4)
        ttk.Button(preview, text="Batch Apply Mask to Images...", command=self.batch_apply_mask_dialog).pack(fill="x", padx=4, pady=(0,8))

        # I/O (Load / Save)
        io = ttk.LabelFrame(side, text="I/O")
        io.pack(fill="x", pady=(8, 0))

        ttk.Button(io, text="Load ROIs…", command=self.load_json_dialog).pack(side="left", expand=True, fill="x", padx=(6, 3), pady=6)
        ttk.Button(io, text="Save ROIs",  command=self.save_to_json).pack(side="left", expand=True, fill="x", padx=(3, 6), pady=6)

        # small path hint for current save target
        self._save_hint = tk.StringVar(value="Save → (auto)")
        ttk.Label(io, textvariable=self._save_hint).pack(anchor="w", padx=8, pady=(0, 8))

        # Style
        style = ttk.Style(self)
        try: style.theme_use("clam")
        except tk.TclError: pass
        try:
            style.configure("TFrame", background="#202225")
            style.configure("TLabelframe", background="#202225", foreground="#bfbfbf")
            style.configure("TLabelframe.Label", background="#202225", foreground="#bfbfbf")
            style.configure("TCheckbutton", background="#202225", foreground="#dcdcdc")
            style.configure("TLabel", background="#202225", foreground="#f0f0f0")
            style.configure("TButton", padding=6)
        except Exception:
            pass

    # ---------- Media I/O ----------
    def _blank_canvas(self):
        self.h, self.w = 720, 1280
        blank = np.zeros((self.h, self.w, 3), np.uint8)
        cv2.putText(blank, "File > Open Image... or Open Video...", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
        self.frame = blank
        self._update_nav(enabled=False)
        self._refresh_image()

    def _update_nav(self, enabled: bool):
        state = ("normal" if enabled and self.is_video else "disabled")
        self.btn_prev.configure(state=state)
        self.btn_next.configure(state=state)
        self.btn_jump_back.configure(state=state)
        self.btn_jump_forward.configure(state=state)
        self.slider.configure(state=state)

    def _open_image_dialog(self):
        p = filedialog.askopenfilename(title="Open Image", filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),("All","*.*")])
        if p: self.load_media(Path(p))

    def _open_video_dialog(self):
        p = filedialog.askopenfilename(title="Open Video", filetypes=[("Videos","*.mp4 *.mov *.mkv *.avi"),("All","*.*")])
        if p: self.load_media(Path(p))

    def load_media(self, path: Path):
        # cleanup previous media
        if self.cap is not None:
            try: self.cap.release()
            except: pass
            self.cap = None
        self.media_path = path
        # Reset drawing & view
        self.drawing = False; self.current_points.clear()
        # NEW: update window title with media name
        self.title(f"ROI Studio — {self.media_path.name}  |  ROIs: {self.save_json_path.name if self.save_json_path else '(unsaved)'}")

        ext = path.suffix.lower()
        self.is_video = ext in {".mp4",".mov",".mkv",".avi",".webm"}

        if self.is_video:
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                messagebox.showerror("Open Video", f"Cannot open video: {path}")
                self._blank_canvas(); return
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            self.idx = 0
            self._read_frame()
            self.h, self.w = self.frame.shape[:2]
            self.slider.configure(from_=0, to=max(0, self.total_frames-1))
            self._update_nav(enabled=True)
        else:
            im = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if im is None:
                messagebox.showerror("Open Image", f"Cannot open image: {path}")
                self._blank_canvas(); return
            self.frame = im; self.h, self.w = im.shape[:2]
            self.total_frames = 1; self.idx = 0
            self._update_nav(enabled=False)

        # Reset drawing & view
        self.drawing = False; self.current_points.clear()
        # Auto-load pending ROIs (if file dropped in earlier)
        if self._pending_roi_load:
            self.load_from_json(self._pending_roi_load); self._pending_roi_load = None
        self._refresh_image()

    # ---------- ROI JSON ----------
    def load_from_json(self, path: Path):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            data = legacy_to_new(data)
            self.rois.clear()
            for name, spec in data.get("rois", {}).items():
                pts = ensure_polygon(spec.get("points", []))
                self.rois[name] = {"points": pts, "include": bool(spec.get("include", True)),
                                   "visible": True, "color": color_for_name(name)}
            self.save_json_path = path  # NEW: use loaded file as current save target
            if hasattr(self, "_save_hint"):
                self._save_hint.set(f"Save → {self.save_json_path}")
            self.title(f"ROI Studio — {self.media_path.name if self.media_path else ''}  |  ROIs: {self.save_json_path.name}")
            self._sync_roi_list(); self._refresh_image()
        except Exception as e:
            messagebox.showerror("Load ROIs", f"Failed to load JSON:\n{e}")

    def save_to_json(self):
        if self.frame is None:
            messagebox.showwarning("Save ROIs", "Open an image or video first."); return

        out = {"meta": {}, "rois": {}}
        if self.media_path:
            if self.is_video:
                out["meta"] = {"video_path": str(self.media_path), "width": self.w, "height": self.h, "frame_index": self.idx}
            else:
                out["meta"] = {"image_path": str(self.media_path), "width": self.w, "height": self.h}

        for name, spec in self.rois.items():
            out["rois"][name] = {"points": spec.get("points", []), "include": bool(spec.get("include", True))}

        # pick default save path if we don't have one yet
        target = self.save_json_path or Path(f"{(self.media_path.stem if self.media_path else 'rois')}_rois.json")
        try:
            target.write_text(json.dumps(out, indent=2), encoding="utf-8")
            self.save_json_path = target
            # NEW: update hint and window title
            if hasattr(self, "_save_hint"):
                self._save_hint.set(f"Save → {self.save_json_path}")
            self.title(f"ROI Studio — {self.media_path.name if self.media_path else ''}  |  ROIs: {self.save_json_path.name}")
            messagebox.showinfo("Saved", f"ROIs saved to:\n{target}")
        except Exception as e:
            messagebox.showerror("Save ROIs", f"Failed to save JSON:\n{e}")


    def load_json_dialog(self):
        fn = filedialog.askopenfilename(title="Load ROI JSON", filetypes=[("JSON","*.json")])
        if fn:
            self.load_from_json(Path(fn))

    # ---------- Frame handling ----------
    def _read_frame(self):
        if not self.is_video or self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(self.idx, self.total_frames - 1)))
        ok, frame = self.cap.read()
        self.frame = frame if ok else self.frame

    def prev_frame(self):
        if not self.is_video: return
        if self.idx > 0:
            self.idx -= 1
            self.drawing = False; self.current_points.clear()
            self._read_frame(); self._refresh_image()

    def next_frame(self):
        if not self.is_video: return
        if self.idx < self.total_frames - 1:
            self.idx += 1
            self.drawing = False; self.current_points.clear()
            self._read_frame(); self._refresh_image()

    def jump_backward(self):
        """Jump backward by the selected jump size."""
        if not self.is_video: return
        jump = self.jump_size.get()
        new_idx = max(0, self.idx - jump)
        if new_idx != self.idx:
            self.idx = new_idx
            self.drawing = False; self.current_points.clear()
            self._read_frame()
            self.slider.set(self.idx)
            self._refresh_image()

    def jump_forward(self):
        """Jump forward by the selected jump size."""
        if not self.is_video: return
        jump = self.jump_size.get()
        new_idx = min(self.total_frames - 1, self.idx + jump)
        if new_idx != self.idx:
            self.idx = new_idx
            self.drawing = False; self.current_points.clear()
            self._read_frame()
            self.slider.set(self.idx)
            self._refresh_image()

    def _on_slider(self, _val):
        if not self.is_video: return
        try: self.idx = int(float(_val))
        except Exception: return
        self.drawing = False; self.current_points.clear()
        self._read_frame(); self._refresh_image()

    # ---------- Zoom ----------
    def set_zoom(self, new_zoom: float, anchor_canvas_xy: tuple[float,float] | None = None):
        new_zoom = float(max(0.1, min(4.0, new_zoom)))
        old_zoom = self.zoom
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        if anchor_canvas_xy is not None:
            ax, ay = anchor_canvas_xy
            img_x = self.canvas.canvasx(ax) / old_zoom
            img_y = self.canvas.canvasy(ay) / old_zoom
            self.zoom = new_zoom; self.zoom_var.set(self.zoom * 100.0)
            self._refresh_image()
            sw, sh = int(self.w * self.zoom), int(self.h * self.zoom)
            if sw > 1 and sh > 1:
                self.canvas.xview_moveto(max(0.0, min((img_x * self.zoom - ax) / max(1, sw), 1.0)))
                self.canvas.yview_moveto(max(0.0, min((img_y * self.zoom - ay) / max(1, sh), 1.0)))
        else:
            self.zoom = new_zoom; self.zoom_var.set(self.zoom * 100.0)
            self._refresh_image()
        self.fit_mode = False

    def _on_zoom_slider(self, _val):
        try: pct = float(_val)
        except Exception: return
        self.set_zoom(pct / 100.0)

    def on_ctrl_wheel(self, event):
        factor = 1.1 if event.delta > 0 else 1/1.1
        self.set_zoom(self.zoom * factor, anchor_canvas_xy=(event.x, event.y))

    def fit_to_window(self):
        self.update_idletasks()
        view_w = max(1, self.canvas.winfo_width()); view_h = max(1, self.canvas.winfo_height())
        zx = view_w / (self.w if self.w else 1); zy = view_h / (self.h if self.h else 1)
        self.fit_mode = True; self.set_zoom(max(0.1, min(4.0, min(zx, zy))))

    def on_canvas_configure(self, _event):
        if self.fit_mode: self.fit_to_window()

    # ---------- Drawing ----------
    def start_draw(self):
        if self.frame is None:
            messagebox.showwarning("Draw ROI", "Open an image or video first.")
            return
        self.drawing = True; self.current_points.clear(); self._refresh_image()

    def finish_roi(self):
        if not self.drawing or len(self.current_points) < 3:
            self.drawing = False; self.current_points.clear(); self._refresh_image(); return
        pts = ensure_polygon(self.current_points)
        name = simpledialog.askstring("ROI Name", "Enter ROI name:")
        if not name:
            self.drawing = False; self.current_points.clear(); self._refresh_image(); return
        self.rois[name] = {"points": pts, "include": True, "visible": True, "color": color_for_name(name)}
        self._sync_roi_list(); self.drawing = False; self.current_points.clear(); self._refresh_image()

    def undo_point(self):
        if self.drawing and self.current_points:
            self.current_points.pop(); self._refresh_image()

    def on_left_click(self, event):
        if not self.drawing: return
        cx = self.canvas.canvasx(event.x); cy = self.canvas.canvasy(event.y)
        x = int(max(0, min((self.w - 1 if self.w else 0), round(cx / self.zoom))))
        y = int(max(0, min((self.h - 1 if self.h else 0), round(cy / self.zoom))))
        self.current_points.append((x, y)); self._refresh_image()

    def on_right_click(self, _event):
        self.undo_point()

    def on_double_left(self, _event):
        self.finish_roi()

    # ---------- ROI list ----------
    def _sync_roi_list(self):
        self.roi_list.delete(0, tk.END)
        for name in self.rois.keys():
            self.roi_list.insert(tk.END, name)
        self._update_toggle_vars()

    def _selected_name(self):
        sel = self.roi_list.curselection()
        return (self.roi_list.get(sel[0]) if sel else None)

    def _update_toggle_vars(self):
        name = self._selected_name()
        if not name:
            self.include_var.set(True); self.visible_var.set(True); return
        self.include_var.set(bool(self.rois[name].get("include", True)))
        self.visible_var.set(bool(self.rois[name].get("visible", True)))

    def _toggle_include(self):
        name = self._selected_name()
        if not name: return
        self.rois[name]["include"] = bool(self.include_var.get()); self._refresh_image()

    def _toggle_visible(self):
        name = self._selected_name()
        if not name: return
        self.rois[name]["visible"] = bool(self.visible_var.get()); self._refresh_image()

    def rename_roi(self):
        name = self._selected_name()
        if not name: return
        new_name = simpledialog.askstring("Rename ROI", f"Rename '{name}' to:")
        if not new_name or new_name == name: return
        self.rois[new_name] = self.rois.pop(name)
        self.rois[new_name]["color"] = color_for_name(new_name)
        self._sync_roi_list(); self._refresh_image()

    def delete_roi(self):
        name = self._selected_name()
        if not name: return
        if messagebox.askyesno("Delete ROI", f"Delete ROI '{name}'?"):
            self.rois.pop(name, None); self._sync_roi_list(); self._refresh_image()

    # ---------- Export / Apply ----------
    def export_mask_png(self):
        if self.frame is None:
            messagebox.showwarning("Export Mask", "Open an image or video first.")
            return
        mask = union_mask((self.h, self.w), self.rois)
        default = (self.media_path.with_suffix(".mask.png") if self.media_path else Path("mask.png"))
        fn = filedialog.asksaveasfilename(title="Save mask PNG", defaultextension=".png",
                                          initialfile=default.name, filetypes=[("PNG","*.png")])
        if not fn: return
        try:
            cv2.imwrite(fn, mask); messagebox.showinfo("Saved", f"Mask saved to:\n{fn}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save mask:\n{e}")

    def batch_apply_mask_dialog(self):
        if self.frame is None:
            messagebox.showwarning("Mask Images", "Open an image or video first.")
            return
        src = filedialog.askdirectory(title="Images folder to mask")
        if not src: return
        dst = filedialog.askdirectory(title="Output folder for masked images")
        if not dst: return
        self.batch_apply_mask(Path(src), Path(dst))

    def batch_apply_mask(self, src_dir: Path, out_dir: Path):
        mask = union_mask((self.h, self.w), self.rois)
        out_dir.mkdir(parents=True, exist_ok=True)
        exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
        n = 0
        for p in sorted(Path(src_dir).glob("*")):
            if p.suffix.lower() not in exts: continue
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None: continue
            if (img.shape[1], img.shape[0]) != (self.w, self.h):
                img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
            out = apply_mask_to_image(img, mask)
            cv2.imwrite(str(Path(out_dir)/p.name), out); n += 1
        messagebox.showinfo("Done", f"Masked {n} image(s) to:\n{out_dir}")

    # ---------- Rendering ----------
    def _refresh_image(self):
        if self.frame is None:
            return
        disp = self.frame.copy()

        if self.preview_var.get():
            mask = union_mask((self.h, self.w), self.rois)
            disp = apply_mask_to_image(disp, mask)

        disp = draw_filled_polygons(disp, self.rois, alpha=0.25)

        for name, spec in self.rois.items():
            if not spec.get("visible", True): continue
            pts = spec.get("points") or []
            if len(pts) >= 2:
                arr = np.array(pts, np.int32).reshape(-1, 1, 2)
                cv2.polylines(disp, [arr], True, spec.get("color", color_for_name(name)), 2)
            if pts:
                cv2.putText(disp, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            spec.get("color", color_for_name(name)), 2, lineType=cv2.LINE_AA)

        if self.drawing and self.current_points:
            arr = np.array(self.current_points, np.int32).reshape(-1, 1, 2)
            cv2.polylines(disp, [arr], False, (0, 255, 0), 2)
            for (x, y) in self.current_points:
                cv2.circle(disp, (x, y), 3, (255, 255, 255), -1)
            cv2.putText(disp, "Left-click: add  |  Right-click: undo  |  Double-click or 'Finish ROI' to close",
                        (10, max(24, self.h - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240,240,240), 1, cv2.LINE_AA)

        # frame counter
        if self.is_video:
            cv2.putText(disp, f"Frame {self.idx + 1}/{self.total_frames}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,255), 2, cv2.LINE_AA)

        sw, sh = int(self.w * self.zoom), int(self.h * self.zoom)
        disp = cv2.resize(disp, (sw, sh), interpolation=cv2.INTER_NEAREST)
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

        self._tk_img = ImageTk.PhotoImage(Image.fromarray(disp_rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        self.canvas.config(scrollregion=(0, 0, sw, sh))

        # update toggles panel defaults
        self._update_toggle_vars()

# -------------------------
# CLI entrypoint
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="ROI Studio — draw/edit ROIs, preview mask, export & batch-apply")
    ap.add_argument("media", nargs="?", help="(Optional) Path to a video OR image.")
    ap.add_argument("--load", help="Load an existing ROI JSON (optional).")
    ap.add_argument("--save", help="Default JSON path for 'Save ROIs' (optional).")
    args = ap.parse_args()

    media = Path(args.media) if args.media else None
    app = ROIStudio(media, Path(args.load) if args.load else None, Path(args.save) if args.save else None)
    app.mainloop()

if __name__ == "__main__":
    main()
