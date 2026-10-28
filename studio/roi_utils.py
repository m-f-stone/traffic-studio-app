# studio/roi_utils.py
"""ROI helpers: load ROI JSON in either schema, build union masks, apply them to
image folders in parallel, crop image borders while relabelling YOLO boxes, and
flatten ROI templates for the analyzer."""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

from .logutil import setup_logger

log = setup_logger()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ProgressFn = Optional[Callable[[float, str], None]]
CancelFn = Optional[Callable[[], bool]]


def list_images(images_dir: Path) -> list[Path]:
    d = Path(images_dir)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def default_workers() -> int:
    return max(1, min(16, os.cpu_count() or 2))


def run_parallel(items: Iterable, fn: Callable, *, workers: int | None = None,
                 progress_callback: ProgressFn = None, cancel_check: CancelFn = None,
                 what: str = "images") -> tuple[list, bool]:
    """Apply fn to every item on a thread pool and return (results, cancelled).

    OpenCV releases the GIL while decoding, encoding and masking, so threads give
    a near-linear speedup for per-image work without the pickling and spawn cost
    of multiprocessing (which also misbehaves under PyInstaller). Results come
    back in completion order.
    """
    items = list(items)
    n = len(items)
    if n == 0:
        return [], False
    results: list = []
    cancelled = False
    done = 0
    step = max(1, n // 100)
    with ThreadPoolExecutor(max_workers=workers or default_workers()) as ex:
        futures = [ex.submit(fn, it) for it in items]
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if progress_callback and (done % step == 0 or done == n):
                progress_callback(done / n, f"{done}/{n} {what}")
            if cancel_check and cancel_check():
                cancelled = True
                ex.shutdown(wait=False, cancel_futures=True)
                break
    return results, cancelled


# ---------------------------------------------------------------------------
# Border crop + YOLO relabel
# ---------------------------------------------------------------------------
def relabel_for_crop(label_text: str, w: int, h: int, L: int, T: int, cw: int, ch: int,
                     min_area_frac: float = 1e-3) -> list[str]:
    """Shift and clip normalized YOLO boxes from a (w, h) image into the crop
    whose top-left corner is (L, T) and size is (cw, ch). Boxes whose centre
    falls outside the crop, collapse to nothing, or end up smaller than
    min_area_frac of the crop are dropped."""
    out: list[str] = []
    for line in label_text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        x, y, bw, bh = (float(parts[i]) for i in range(1, 5))
        X, Y, WW, HH = x * w, y * h, bw * w, bh * h
        if not (L <= X < L + cw and T <= Y < T + ch):
            continue
        X -= L
        Y -= T
        x1 = max(0.0, X - WW / 2.0); y1 = max(0.0, Y - HH / 2.0)
        x2 = min(cw, X + WW / 2.0);  y2 = min(ch, Y + HH / 2.0)
        if x2 <= x1 or y2 <= y1:
            continue
        xn = ((x1 + x2) / 2.0) / cw
        yn = ((y1 + y2) / 2.0) / ch
        wn = (x2 - x1) / cw
        hn = (y2 - y1) / ch
        if wn * hn < min_area_frac:
            continue
        out.append(f"{cls} {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}")
    return out


def _crop_one(imp: Path, out_images_dir: Path, labels_dir: Path | None, out_labels_dir: Path | None,
              margin_frac: float, min_area_frac: float) -> tuple[bool, bool]:
    """Crop one image; returns (image_written, label_written)."""
    im = cv2.imread(str(imp))
    if im is None:
        return False, False
    h, w = im.shape[:2]
    L = int(round(margin_frac * w)); R = w - L
    T = int(round(margin_frac * h)); B = h - T
    cw, ch = R - L, B - T
    if cw <= 1 or ch <= 1:  # margin too large
        return False, False
    cv2.imwrite(str(out_images_dir / imp.name), im[T:B, L:R])
    if not labels_dir:
        return True, False
    lblp = labels_dir / (imp.stem + ".txt")
    if not lblp.exists():
        return True, False
    out_lines = relabel_for_crop(lblp.read_text(encoding="utf-8", errors="ignore"),
                                 w, h, L, T, cw, ch, min_area_frac)
    if not out_lines:
        return True, False
    (out_labels_dir / (imp.stem + ".txt")).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return True, True


def crop_margin_and_relabel(images_dir: Path, out_images_dir: Path,
                            labels_dir: Path | None = None, out_labels_dir: Path | None = None,
                            margin_frac: float = 0.10, min_area_frac: float = 1e-3, *,
                            workers: int | None = None,
                            progress_callback: ProgressFn = None,
                            cancel_check: CancelFn = None) -> tuple[int, int]:
    """Crop every image to the inner rectangle (1 - 2*margin) and rewrite YOLO
    labels if a labels dir is given. Returns (images_written, label_files_written)."""
    images_dir = Path(images_dir)
    out_images_dir = Path(out_images_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = Path(labels_dir) if labels_dir else None
    if labels_dir:
        out_labels_dir = Path(out_labels_dir) if out_labels_dir else out_images_dir.parent / "labels" / out_images_dir.name
        out_labels_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_labels_dir = None

    results, cancelled = run_parallel(
        list_images(images_dir),
        lambda p: _crop_one(p, out_images_dir, labels_dir, out_labels_dir, margin_frac, min_area_frac),
        workers=workers, progress_callback=progress_callback, cancel_check=cancel_check,
        what="images cropped",
    )
    n_img = sum(1 for img, _ in results if img)
    n_lbl = sum(1 for _, lbl in results if lbl)
    if cancelled:
        log.warning("Crop cancelled after %d images.", n_img)
    return n_img, n_lbl


# ---------------------------------------------------------------------------
# ROI JSON + masks
# ---------------------------------------------------------------------------
def load_roi_any(path: Path) -> dict:
    """Load ROI JSON in any supported format and return a normalised dict.

    Handles:
      - New schema: {"meta": {...}, "rois": {"name": {"points": [...], "include": bool}}}
      - Legacy flat: {"name": [[x,y], ...], ...}  (optional "calibration" key stripped)

    Returns:
        dict of {name: {"points": [[x,y], ...], "include": bool}}
    """
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if "rois" in raw and isinstance(raw["rois"], dict):
        return {name: {"points": spec.get("points", []),
                       "include": spec.get("include", True)}
                for name, spec in raw["rois"].items()}

    rois = {}
    for name, pts in raw.items():
        if name == "calibration":
            continue
        if isinstance(pts, list):
            rois[name] = {"points": pts, "include": True}
    return rois


def included_polygons(rois: dict) -> dict:
    """{name: points} for every ROI whose include flag is set."""
    return {name: spec["points"] for name, spec in rois.items() if spec.get("include", True)}


def mask_from_rois(flat_rois: dict, w: int, h: int) -> np.ndarray:
    """Binary uint8 mask (255 inside the union of the polygons, 0 outside)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in flat_rois.values():
        if isinstance(pts, list) and len(pts) >= 3:
            arr = np.array(pts, np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [arr], 255)
    return mask


def mask_for_images(images_dir: Path, roi_json: Path) -> np.ndarray:
    """Union mask of the included ROIs, sized to the first image in images_dir."""
    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    im = cv2.imread(str(images[0]))
    if im is None:
        raise ValueError(f"Could not read image {images[0]}")
    h, w = im.shape[:2]
    flat = included_polygons(load_roi_any(Path(roi_json)))
    if not flat:
        raise ValueError(f"No included ROIs in {roi_json}")
    return mask_from_rois(flat, w, h)


def apply_mask_to_image(src_path: Path, dst_path: Path, mask: np.ndarray) -> bool:
    """Zero out pixels outside the mask and write to dst_path. A mask built for a
    different frame size is scaled to the image. Returns False if the image
    could not be read."""
    img = cv2.imread(str(src_path))
    if img is None:
        return False
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    masked = cv2.bitwise_and(img, img, mask=mask)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst_path), masked))


def mask_images(images_dir: Path, out_dir: Path, mask: np.ndarray, *,
                workers: int | None = None,
                progress_callback: ProgressFn = None,
                cancel_check: CancelFn = None) -> dict:
    """Apply one mask to every image in images_dir, writing same-named files to out_dir.

    Returns {"images": written, "skipped": unreadable, "cancelled": bool}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results, cancelled = run_parallel(
        list_images(images_dir),
        lambda p: apply_mask_to_image(p, out_dir / p.name, mask),
        workers=workers, progress_callback=progress_callback, cancel_check=cancel_check,
        what="images masked",
    )
    written = sum(1 for ok in results if ok)
    if cancelled:
        log.warning("Masking cancelled after %d images.", written)
    return {"images": written, "skipped": len(results) - written, "cancelled": cancelled}


def normalize_for_tracking(template_path: Path,
                           out_path: Path,
                           include_lanes: bool = False,
                           lane_pattern: str = r"(lane|_ln|\b[ltr]\b|through|left|right)") -> Path:
    """
    Converts a template ROI JSON into the flat mapping expected by the analyzer:
      { "roiA": [[x,y], ...], "roiB": [[x,y], ...] }

    If include_lanes=True, force "include": true for any ROI whose name matches
    `lane_pattern` (regex, case-insensitive), then keep only include==true.
    """
    rois = load_roi_any(template_path)
    if include_lanes:
        rx = re.compile(lane_pattern, re.IGNORECASE)
        for name, spec in rois.items():
            if rx.search(name):
                spec["include"] = True

    flat = included_polygons(rois)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(flat, f, indent=2)
    return out_path
