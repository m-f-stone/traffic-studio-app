from ultralytics import YOLO
import multiprocessing as mp
from .logutil import setup_logger
log = setup_logger()

def train(weights="yolov8s.pt", data_yaml="data.yaml", *,
          imgsz=640, epochs=60, batch=-1, device=0, workers=2, cache="disk",
          name="traffic_v8s",
          mosaic=1.0, mixup=0.0, copy_paste=0.0, degrees=0.0,
          hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
          fl_gamma=None,
          close_mosaic=10, lr0=None, lrf=None, cos_lr=True, warmup_epochs=3.0,
          freeze=None, optimizer=None):
    def _run():
        model = YOLO(weights)
        overrides = dict(
            data=data_yaml, imgsz=imgsz, epochs=epochs, batch=batch,
            device=device, workers=workers, cache=cache, name=name,
            mosaic=mosaic, mixup=mixup, copy_paste=copy_paste, degrees=degrees,
            hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v,
            close_mosaic=close_mosaic, cos_lr=cos_lr, warmup_epochs=warmup_epochs
        )
        if fl_gamma is not None:
            overrides['fl_gamma'] = fl_gamma
        if lr0 is not None: overrides['lr0'] = lr0
        if lrf is not None: overrides['lrf'] = lrf
        if freeze is not None: overrides['freeze'] = freeze
        if optimizer is not None: overrides['optimizer'] = optimizer
        log.info("Training with overrides: %s", {k: overrides[k] for k in sorted(overrides) if k not in ('data','name')})
        model.train(**overrides)
        try:
            metrics = model.val(data=data_yaml, imgsz=imgsz, device=device, plots=True)
            log.info("Validation mAP50-95: %.4f", float(metrics.box.map) if hasattr(metrics, 'box') else float(metrics.map))
        except Exception as e:
            log.warning("Validation after training failed: %s", e)

    if __name__ == "__main__":
        mp.freeze_support()
    _run()
