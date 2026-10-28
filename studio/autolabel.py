# studio/autolabel.py
"""Auto-label an image folder with a YOLO model, with progress and cancellation."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .devices import resolve_device
from .logutil import setup_logger
from .roi_utils import list_images

log = setup_logger()

DEFAULT_WEIGHT_PATTERNS = (
    "runs/detect/*/weights/best.pt",
    "runs/detect/*/weights/last.pt",
    "runs/*/weights/best.pt",
    "runs/*/weights/last.pt",
)


def find_latest_best(patterns=None, fallback=None, root: Path | None = None) -> str:
    """Newest trained weights under root (default: current directory)."""
    root = Path(root) if root else Path.cwd()
    hits: list[Path] = []
    for pat in (patterns or DEFAULT_WEIGHT_PATTERNS):
        hits.extend(root.glob(pat))
    if hits:
        return str(max(hits, key=lambda p: p.stat().st_mtime))
    if fallback:
        return fallback
    raise FileNotFoundError(f"No trained weights found under {root}.")


def _resolve_device(requested=None):
    """Kept for callers that imported the old private helper."""
    return resolve_device(requested)


def autolabel(images_dir: Path, *,
              model_path: str | None = None,
              imgsz=960, conf=0.70, iou=0.60,
              device: str | int | None = None,     # None/'auto'/'cpu'/0/'0,1'
              half: bool | None = None,            # None -> follows device
              max_det=800, augment=False,
              classes=None, project="runs/autolabel", name="autolabel_run",
              weights_pattern=None, search_root: Path | None = None,
              progress_callback: Optional[Callable[[float, str], None]] = None,
              cancel_check: Optional[Callable[[], bool]] = None) -> dict:
    """Autolabel images using YOLO and save 5-column YOLO txt labels.

    If model_path is not given, the newest best.pt/last.pt under search_root is
    used. Inference streams one image at a time so progress is real and Cancel
    stops after the current image.

    Returns {"images", "detections", "labels_dir", "weights", "cancelled"}.
    """
    from ultralytics import YOLO  # lazy: torch import is slow and not needed for the GUI to start

    images_dir = Path(images_dir)
    weights = model_path or find_latest_best(weights_pattern, root=search_root)
    dev, can_half = resolve_device(device)
    if half is None:
        half = can_half

    n_images = len(list_images(images_dir))
    log.info("Autolabel using weights: %s", weights)
    log.info("Device: %s  |  half: %s  |  %d images", dev, half, n_images)
    if progress_callback:
        progress_callback(0.0, f"Loading {Path(str(weights)).name}")

    model = YOLO(str(weights))
    stream = model.predict(
        source=str(images_dir), stream=True, verbose=False,
        imgsz=imgsz, conf=conf, iou=iou, device=dev, half=half,
        max_det=max_det, augment=augment, classes=classes,
        save_txt=True, save_conf=False,   # 5-column files for LabelImg
        project=str(project), name=name,
    )

    labels_dir: Path | None = None
    processed = detections = 0
    cancelled = False
    for res in stream:
        processed += 1
        detections += len(res.boxes) if res.boxes is not None else 0
        if labels_dir is None:
            save_dir = getattr(res, "save_dir", None) or getattr(getattr(model, "predictor", None), "save_dir", None)
            if save_dir:
                labels_dir = Path(save_dir) / "labels"
        if progress_callback and (processed % 5 == 0 or processed == n_images):
            progress_callback(processed / max(n_images, 1), f"Labeled {processed}/{n_images} images")
        if cancel_check and cancel_check():
            cancelled = True
            log.warning("Autolabel cancelled after %d images.", processed)
            break

    if labels_dir is None:
        labels_dir = Path(project) / name / "labels"
    log.info("Autolabel: %d images, %d detections -> %s", processed, detections, labels_dir)
    return {"images": processed, "detections": detections, "labels_dir": labels_dir,
            "weights": str(weights), "cancelled": cancelled}
