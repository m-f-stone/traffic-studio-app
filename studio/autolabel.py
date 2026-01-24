# studio/autolabel.py
from ultralytics import YOLO
from glob import glob
from pathlib import Path
from .logutil import setup_logger
log = setup_logger()

def find_latest_best(patterns=None, fallback=None):
    patterns = patterns or [
        r"runs/detect/*/weights/best.pt",
        r"runs/detect/*/weights/last.pt",
        r"runs/*/weights/best.pt",
        r"runs/*/weights/last.pt",
    ]
    hits = []
    for pat in patterns:
        hits.extend(glob(pat))
    hits = sorted(hits, key=lambda p: Path(p).stat().st_mtime) if hits else []
    if hits:
        return hits[-1]
    if fallback:
        return fallback
    raise FileNotFoundError("No trained weights found.")

def _resolve_device(requested=None):
    """
    Returns (device_for_ultralytics, can_half).
    - If CUDA is available and requested is None/'auto', returns (0, True).
    - If CUDA not available, returns ('cpu', False) and logs a note.
    - If requested is 'cpu', always returns ('cpu', False).
    - If requested is an int/str like '0' but CUDA is unavailable, falls back to CPU with a warning.
    """
    import torch
    if requested is None or str(requested).lower() in ("", "auto"):
        if torch.cuda.is_available():
            return 0, True
        log.info("CUDA not available; running on CPU.")
        return "cpu", False

    # explicit request
    req = requested
    if isinstance(req, str) and req.lower() == "cpu":
        return "cpu", False
    if torch.cuda.is_available():
        # '0', '0,1', or int 0 are acceptable
        return req, True
    log.warning("Requested CUDA device=%s but CUDA is not available; falling back to CPU.", requested)
    return "cpu", False

def autolabel(images_dir: Path, *,
              model_path: str | None = None,
              imgsz=960, conf=0.70, iou=0.60,
              device: str | int | None = None,     # <-- can be None/'auto'/'cpu'/0
              half: bool | None = None,            # <-- if None it follows device
              max_det=800, augment=False,
              classes=None, project="runs/autolabel", name="autolabel_run",
              weights_pattern=None):
    """
    Autolabel images using YOLOv8.
    If model_path is not given, search the newest best.pt/last.pt under runs/*/weights.
    Device is auto-detected unless explicitly set. FP16 is used only on CUDA.
    """
    mp = model_path or find_latest_best(weights_pattern)
    dev, can_half = _resolve_device(device)
    if half is None:
        half = can_half

    log.info("Autolabel using weights: %s", mp)
    log.info("Device: %s  |  half: %s", dev, half)

    model = YOLO(mp)
    return model.predict(
        source=str(images_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=dev,
        half=half,
        max_det=max_det,
        augment=augment,
        classes=classes,
        save_txt=True,
        save_conf=False,   # keep 5-column files for LabelImg
        project=project,
        name=name,
    )
