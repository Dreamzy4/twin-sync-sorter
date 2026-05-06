"""Non-blocking print() replacement for Isaac Sim heavy-logging environments.

Isaac's stdout flush can stall the simulation step when many ``print()`` calls
hit it from CV / motion / telemetry threads. This module installs a queue-backed
async writer that keeps ``print()`` non-blocking and drains in a daemon thread.

Usage::

    from async_logger import install_async_print
    install_async_print()  # call once at startup

Disable for debugging via ``DTCV_SYNC_PRINT=1``. Drops messages silently if the
queue overflows (rare; logs the drop count on shutdown).
"""

import atexit
import builtins
import os
import queue
import sys
import threading
import time

_ORIGINAL_PRINT = builtins.print
_QUEUE = queue.Queue(maxsize=4096)
_STOP = object()
_THREAD = None
_INSTALLED = False
_DROPPED = 0
_LOCK = threading.Lock()


def _writer():
    while True:
        item = _QUEUE.get()
        if item is _STOP:
            _QUEUE.task_done()
            break
        text, stream, flush = item
        try:
            stream.write(text)
            if flush:
                stream.flush()
        except Exception:
            pass
        finally:
            _QUEUE.task_done()


def _async_print(*args, sep=" ", end="\n", file=None, flush=False):
    global _DROPPED
    stream = sys.stdout if file is None else file

    if stream not in (sys.stdout, sys.stderr):
        return _ORIGINAL_PRINT(*args, sep=sep, end=end, file=file, flush=flush)

    try:
        text = sep.join(str(arg) for arg in args) + end
    except Exception:
        text = "<print-format-error>" + end

    try:
        _QUEUE.put_nowait((text, stream, flush))
    except queue.Full:
        _DROPPED += 1


def install_async_print():
    global _THREAD, _INSTALLED
    if os.environ.get("DTCV_SYNC_PRINT") == "1":
        return

    with _LOCK:
        if _INSTALLED:
            return
        _THREAD = threading.Thread(target=_writer, name="dtcv-async-print", daemon=True)
        _THREAD.start()
        builtins.print = _async_print
        _INSTALLED = True


def flush_async_print(timeout=1.0):
    if not _INSTALLED:
        return
    deadline = time.monotonic() + max(0.0, float(timeout))
    while time.monotonic() < deadline and not _QUEUE.empty():
        time.sleep(0.01)


def shutdown_async_print():
    global _INSTALLED
    if not _INSTALLED:
        return
    flush_async_print(timeout=1.0)
    try:
        _QUEUE.put(_STOP, block=True, timeout=0.5)
    except queue.Full:
        pass
    if _THREAD is not None:
        _THREAD.join(timeout=0.5)
    builtins.print = _ORIGINAL_PRINT
    if _DROPPED:
        _ORIGINAL_PRINT(f"[AsyncLog] dropped {_DROPPED} log lines")
    _INSTALLED = False


atexit.register(shutdown_async_print)
