import logging, sys, time, queue, threading

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

def setup_logger(q=None, level=logging.INFO):
    logger = logging.getLogger("traffic_studio")
    logger.setLevel(level)
    logger.handlers[:] = []
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(sh)
    if q:
        gh = GuiLogHandler(q); gh.setFormatter(fmt); logger.addHandler(gh)
    return logger
