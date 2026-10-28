# studio/trainer.py
"""YOLO training with per-epoch progress and cooperative cancellation."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .devices import resolve_device
from .logutil import setup_logger

log = setup_logger()


def train(weights="yolov8s.pt", data_yaml="data.yaml", *,
          imgsz=640, epochs=60, batch=-1, device=None, workers=2, cache="disk",
          name="traffic_v8s", project: str | Path | None = None,
          mosaic=1.0, mixup=0.0, copy_paste=0.0, degrees=0.0,
          hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
          fl_gamma=None,
          close_mosaic=10, lr0=None, lrf=None, cos_lr=True, warmup_epochs=3.0,
          freeze=None, optimizer=None,
          progress_callback: Optional[Callable[[float, str], None]] = None,
          cancel_check: Optional[Callable[[], bool]] = None) -> dict:
    """Train, then validate unless cancelled.

    Cancel takes effect at the end of the current epoch (ultralytics checks its
    stop flag once per epoch); the checkpoint for that epoch is still saved.

    Returns {"save_dir": Path, "best": Path | None, "cancelled": bool}.
    """
    from ultralytics import YOLO  # lazy: keeps GUI start-up fast

    dev, _ = resolve_device(device)
    model = YOLO(weights)
    overrides = dict(
        data=data_yaml, imgsz=imgsz, epochs=epochs, batch=batch,
        device=dev, workers=workers, cache=cache, name=name,
        mosaic=mosaic, mixup=mixup, copy_paste=copy_paste, degrees=degrees,
        hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v,
        close_mosaic=close_mosaic, cos_lr=cos_lr, warmup_epochs=warmup_epochs,
    )
    if project is not None:
        overrides["project"] = str(project)
    if fl_gamma is not None:
        overrides["fl_gamma"] = fl_gamma
    if lr0 is not None:
        overrides["lr0"] = lr0
    if lrf is not None:
        overrides["lrf"] = lrf
    if freeze is not None:
        overrides["freeze"] = freeze
    if optimizer is not None:
        overrides["optimizer"] = optimizer

    state = {"cancelled": False}

    def _on_train_epoch_end(trainer):
        done = trainer.epoch + 1
        if progress_callback:
            progress_callback(done / max(trainer.epochs, 1), f"Epoch {done}/{trainer.epochs}")
        if cancel_check and cancel_check():
            state["cancelled"] = True
            trainer.stop = True
            log.warning("Cancel requested; stopping after epoch %d.", done)

    model.add_callback("on_train_epoch_end", _on_train_epoch_end)

    log.info("Training with overrides: %s",
             {k: overrides[k] for k in sorted(overrides) if k not in ("data", "name", "project")})
    model.train(**overrides)

    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir",
                            Path(project or "runs/detect") / name))
    best = save_dir / "weights" / "best.pt"

    if not state["cancelled"]:
        try:
            metrics = model.val(data=data_yaml, imgsz=imgsz, device=dev, plots=True)
            log.info("Validation mAP50-95: %.4f",
                     float(metrics.box.map) if hasattr(metrics, "box") else float(metrics.map))
        except Exception as e:
            log.warning("Validation after training failed: %s", e)

    return {"save_dir": save_dir, "best": best if best.exists() else None, "cancelled": state["cancelled"]}
