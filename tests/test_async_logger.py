"""Tests for async_logger - the queue-backed print() replacement.

Covers:
* ``install_async_print`` is a no-op under ``DTCV_SYNC_PRINT=1``.
* Idempotency: a second ``install_async_print`` call doesn't double-wrap.
* End-to-end happy path: a printed line lands on the captured stdout.
* Overflow behaviour: with the queue at capacity and the writer thread
  starved, additional puts are dropped (counter increments) rather than
  blocking the caller.
* ``shutdown_async_print`` restores the original ``print`` and prints a
  drop-count summary when any drops occurred.

Each test isolates async_logger by reloading the module and running
``shutdown_async_print`` on teardown so global state from earlier tests
does not leak into later ones.
"""

import builtins
import importlib
import io
import queue
import sys
import threading

import pytest

from tests import REAL_BUILTIN_PRINT


@pytest.fixture
def fresh_logger(monkeypatch):
    """Reload async_logger so each test starts from clean module state.

    The ``builtins.print`` reset before the reload is critical: ``robot_motion``
    auto-installs the async writer at import time, so by the time these tests
    run ``builtins.print`` is already a queue-pushing wrapper. Without the
    reset, the reload would capture that wrapper as ``_ORIGINAL_PRINT`` and
    a later ``_ORIGINAL_PRINT(file=sink)`` call would recurse.
    """
    monkeypatch.delenv("DTCV_SYNC_PRINT", raising=False)
    builtins.print = REAL_BUILTIN_PRINT

    import async_logger

    importlib.reload(async_logger)
    yield async_logger

    try:
        async_logger.shutdown_async_print()
    except Exception:
        pass
    builtins.print = REAL_BUILTIN_PRINT


def test_install_is_noop_when_sync_env_set(monkeypatch, fresh_logger):
    monkeypatch.setenv("DTCV_SYNC_PRINT", "1")
    original = builtins.print
    fresh_logger.install_async_print()
    assert builtins.print is original
    assert fresh_logger._INSTALLED is False


def test_install_is_idempotent(fresh_logger):
    fresh_logger.install_async_print()
    swapped_once = builtins.print
    fresh_logger.install_async_print()
    swapped_twice = builtins.print
    assert swapped_once is swapped_twice
    assert fresh_logger._INSTALLED is True


def test_print_lands_on_stdout(fresh_logger, capsys):
    """Happy path: install, print, flush, capture."""
    fresh_logger.install_async_print()
    print("hello-async", flush=True)
    fresh_logger.flush_async_print(timeout=1.0)
    # Drain by stopping the writer cleanly.
    fresh_logger.shutdown_async_print()
    captured = capsys.readouterr()
    assert "hello-async" in captured.out


def test_overflow_increments_drop_counter(fresh_logger):
    """If the queue fills up faster than the writer drains, drops are counted."""
    # Block the writer thread by stuffing a sentinel that the test fixture
    # never consumes; instead, just bypass install and write directly.
    fresh_logger.install_async_print()
    # Choke the queue by replacing the writer's get with something blocking.
    # Easier: fill the queue past capacity by direct put_nowait via the
    # exposed _async_print, while the writer is blocked on an artificial lock.
    block = threading.Event()

    def _hold(*_args, **_kwargs):
        block.wait()
        return None

    # Replace writer's stream so each get() effectively no-ops fast - but we
    # actually want to stall it. Inject a fake stream whose .write() blocks
    # until we release `block`.
    class _BlockingStream:
        def write(self, _):
            block.wait(timeout=2.0)

        def flush(self):
            pass

    # Redirect both stdout and stderr-routes through the blocking stream by
    # using a custom file= argument NOT in (sys.stdout, sys.stderr) would
    # short-circuit to the original print, so we must instead stall the
    # actual writer. Replace sys.stdout temporarily.
    real_stdout = sys.stdout
    sys.stdout = _BlockingStream()
    try:
        # Fill the queue past capacity (4096). Push enough to ensure overflow
        # given the writer is blocked on the very first item.
        for i in range(4200):
            print(f"line-{i}", flush=False)
        # Some drops must have occurred.
        assert fresh_logger._DROPPED > 0
    finally:
        block.set()
        sys.stdout = real_stdout
        # Drain whatever queue can still flow before shutdown.
        try:
            while True:
                fresh_logger._QUEUE.get_nowait()
                fresh_logger._QUEUE.task_done()
        except queue.Empty:
            pass


def test_print_passes_non_std_streams_through_to_original(fresh_logger):
    """Writing to a custom file= must bypass the queue (no truncation, no drops)."""
    fresh_logger.install_async_print()
    sink = io.StringIO()
    print("custom-stream", file=sink)
    assert "custom-stream" in sink.getvalue()


def test_shutdown_restores_original_print(fresh_logger):
    fresh_logger.install_async_print()
    swapped = builtins.print
    assert swapped is not fresh_logger._ORIGINAL_PRINT
    fresh_logger.shutdown_async_print()
    assert builtins.print is fresh_logger._ORIGINAL_PRINT
    assert fresh_logger._INSTALLED is False


def test_flush_is_safe_before_install(fresh_logger):
    """flush_async_print() must not crash if logger was never installed."""
    fresh_logger.flush_async_print(timeout=0.05)


def test_shutdown_is_safe_before_install(fresh_logger):
    fresh_logger.shutdown_async_print()
    assert builtins.print is fresh_logger._ORIGINAL_PRINT
