# studio/roi_utils.py  (append to the bottom)
from __future__ import annotations  # prevents early evaluation of type hints (safe for 3.11/3.13)
from pathlib import Path

import json
import re
import cv2
import numpy as np

def crop_margin_and_relabel(images_dir: Path, out_images_dir: Path,
                            labels_dir: Path | None = None, out_labels_dir: Path | None = None,
                            margin_frac: float = 0.10, min_area_frac: float = 1e-3):
    """
    Crop all images to the inner rectangle (1-2*margin) and update YOLO labels if present.
    images_dir -> out_images_dir; labels_dir (optional) -> out_labels_dir.
    Based on your crop_margin_dataset approach (single split variant).  :contentReference[oaicite:4]{index=4}
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    if labels_dir:
        out_labels_dir = out_labels_dir or out_images_dir.parent / "labels" / out_images_dir.name
        out_labels_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    n_img, n_lbl = 0, 0
    for imp in sorted(images_dir.glob("*")):
        if imp.suffix.lower() not in exts: continue
        im = cv2.imread(str(imp))
        if im is None: continue
        h, w = im.shape[:2]
        L = int(round(margin_frac * w)); R = w - L
        T = int(round(margin_frac * h)); B = h - T
        cw, ch = (R - L), (B - T)
        if cw <= 1 or ch <= 1:  # margin too large
            continue
        crop = im[T:B, L:R]
        cv2.imwrite(str(out_images_dir / imp.name), crop); n_img += 1

        # labels?
        if labels_dir:
            lblp = labels_dir / (imp.stem + ".txt")
            out_lines = []
            if lblp.exists():
                for line in lblp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.split()
                    if len(parts) != 5:  # keep malformed as-is? We skip.
                        continue
                    cls = int(float(parts[0]))
                    x, y, bw, bh = (float(parts[i]) for i in range(1,5))
                    # denormalize to original image
                    X = x * w; Y = y * h; WW = bw * w; HH = bh * h
                    # center must fall inside crop
                    if not (L <= X < R and T <= Y < B):
                        continue
                    # shift to crop coordinates
                    X -= L; Y -= T
                    x1 = max(0.0, X - WW/2.0); y1 = max(0.0, Y - HH/2.0)
                    x2 = min(cw,  X + WW/2.0); y2 = min(ch,  Y + HH/2.0)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    # re-normalize
                    xn = ((x1 + x2) / 2.0) / cw
                    yn = ((y1 + y2) / 2.0) / ch
                    wn = (x2 - x1) / cw
                    hn = (y2 - y1) / ch
                    if wn * hn < min_area_frac:
                        continue
                    out_lines.append(f"{cls} {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}")
            if out_lines:
                (out_labels_dir / (imp.stem + ".txt")).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
                n_lbl += 1
    return n_img, n_lbl

def normalize_for_tracking(template_path: Path,
                           out_path: Path,
                           include_lanes: bool = False,
                           lane_pattern: str = r"(lane|_ln|\b[ltr]\b|through|left|right)") -> Path:
    """
    Converts a template ROI JSON into the flat mapping expected by the analyzer:
      { "roiA": [[x,y], ...], "roiB": [[x,y], ...] }

    If include_lanes=True, force "include": true for any ROI whose name matches
    `lane_pattern` (regex, case-insensitive), then keep only include==true.

    Example default pattern matches names like:
      'north_lane', 'south_ln2', 'east_left', 'west_through_lane3', etc.
    """
    rois = load_roi_any(template_path)
    if include_lanes:
        rx = re.compile(lane_pattern, re.IGNORECASE)
        for name, spec in rois.items():
            if rx.search(name):
                spec["include"] = True

    flat = {name: spec["points"] for name, spec in rois.items() if spec.get("include", True)}
    json.dump(flat, out_path.open("w"), indent=2)
    return out_path