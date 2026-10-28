import logging, sys, time, queue


class GuiLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        try:
            msg = self.format(record)
            self.q.put((time.strftime("%H:%M:%S"), record.levelname, msg))
        except Exception:
            pass


def _replace_gui_handler(logger: logging.Logger, q: queue.Queue, fmt: logging.Formatter):
    logger.handlers[:] = [h for h in logger.handlers if not isinstance(h, GuiLogHandler)]
    gh = GuiLogHandler(q)
    gh.setFormatter(fmt)
    logger.addHandler(gh)


def setup_logger(q=None, level=logging.INFO):
    """Return the shared 'traffic_studio' logger.

    Safe to call from every module at import time: the console handler is only
    added once. Passing a queue attaches the GUI log pane handler to this logger
    and to ultralytics' logger, so training epochs and inference summaries show
    up in the app instead of only on stdout.
    """
    logger = logging.getLogger("traffic_studio")
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, GuiLogHandler)
                      for h in logger.handlers)
    if not has_console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if q is not None:
        _replace_gui_handler(logger, q, fmt)
        _replace_gui_handler(logging.getLogger("ultralytics"), q, logging.Formatter("%(message)s"))
    return logger
