"""Device resolution shared by the autolabel, trainer, and analyzer entry points."""
from __future__ import annotations

import logging

log = logging.getLogger("traffic_studio")


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(requested=None) -> tuple[str | int, bool]:
    """Map a user device request onto something ultralytics accepts.

    Returns (device, can_use_half):
      None / '' / 'auto'  -> CUDA device 0 when available, otherwise 'cpu'
      'cpu'               -> 'cpu'
      '0', 0, '0,1', ...  -> as requested when CUDA is available, otherwise 'cpu' with a warning
    Half precision is only offered on CUDA.
    """
    req = "" if requested is None else str(requested).strip().lower()
    if req in ("", "auto"):
        if cuda_available():
            return 0, True
        log.info("CUDA not available; running on CPU.")
        return "cpu", False
    if req == "cpu":
        return "cpu", False
    if cuda_available():
        return (int(req) if req.isdigit() else requested), True
    log.warning("Requested CUDA device=%s but CUDA is not available; falling back to CPU.", requested)
    return "cpu", False
