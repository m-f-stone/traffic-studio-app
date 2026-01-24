import threading, queue, traceback
from typing import Callable, Any, Optional

class Worker(threading.Thread):
    """
    Enhanced worker thread with progress reporting and cancellation support.

    Usage:
        def my_task(progress_callback=None, cancel_check=None):
            for i in range(100):
                if cancel_check and cancel_check():
                    return None  # Task was cancelled
                if progress_callback:
                    progress_callback(i / 100, f"Processing item {i}")
                # ... do work ...

        worker = Worker(my_task, on_done, log, on_progress=update_progress_bar)
        worker.start()
        # Later: worker.cancel()
    """
    def __init__(self, fn, on_done, log, on_progress: Optional[Callable[[float, str], None]] = None, *args, **kwargs):
        super().__init__(daemon=True)
        self.fn, self.on_done, self.log = fn, on_done, log
        self.on_progress = on_progress
        self.args, self.kwargs = args, kwargs
        self.exc = None
        self._cancel_requested = False

    def cancel(self):
        """Request cancellation of the task."""
        self._cancel_requested = True

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancel_requested

    def _progress_callback(self, fraction: float, message: str = ""):
        """Internal progress callback wrapper."""
        if self.on_progress and not self._cancel_requested:
            self.on_progress(fraction, message)

    def run(self):
        try:
            # Inject progress_callback and cancel_check into kwargs if function accepts them
            import inspect
            sig = inspect.signature(self.fn)
            if 'progress_callback' in sig.parameters:
                self.kwargs['progress_callback'] = self._progress_callback
            if 'cancel_check' in sig.parameters:
                self.kwargs['cancel_check'] = self.is_cancelled

            res = self.fn(*self.args, **self.kwargs)

            if self._cancel_requested:
                self.on_done(success=False, result=None, cancelled=True)
            else:
                self.on_done(success=True, result=res, cancelled=False)
        except Exception as e:
            self.exc = e
            self.log.exception("Task failed: %s", e)
            self.on_done(success=False, result=None, cancelled=False)
