"""Run a task item by item in a separate worker process that is replaced every so often.

Used for parsing: Docling's PDF pipeline starts new threads for every document and the numeric libraries
under it keep pools of helper threads alive afterwards, so a long batch in one process piles up thousands of
threads (5,014 after ~950 documents) until it stalls. A worker that is replaced every `per_worker` items takes
its leaked threads with it. The same isolation turns a crash in native code or a hang on one document into a
failure of that item only: a fresh worker starts and the batch goes on.
"""

import multiprocessing
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from typing import Any


class WorkerCrashed(RuntimeError):
    """The worker process died while running an item (e.g. a crash in native code)."""


class WorkerTimeout(RuntimeError):
    """An item ran longer than the time limit; its worker was stopped."""


def _new_pool(initializer: Callable[[], None] | None, per_worker: int) -> ProcessPoolExecutor:
    """One worker process (spawned, so it starts clean), replaced after `per_worker` items."""
    return ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                               initializer=initializer, max_tasks_per_child=per_worker)


def _kill(pool: ProcessPoolExecutor) -> None:
    """Stop a pool whose worker is stuck: terminate its process, then drop the pool without waiting."""
    for process in list(getattr(pool, "_processes", {}).values()):
        process.terminate()
    pool.shutdown(wait=False, cancel_futures=True)


def run_isolated(items: Iterable[Any], task: Callable[[Any], Any], *,
                 initializer: Callable[[], None] | None = None, per_worker: int = 50,
                 timeout: float | None = None) -> Iterator[tuple[Any, Any, BaseException | None]]:
    """Yield (item, result, error) for each item, running task(item) in a worker process.

    One item at a time, in order. The worker runs `initializer` once when it starts (e.g. loading models)
    and is replaced after `per_worker` items. An exception raised by the task comes back as the error; a
    worker that dies gives WorkerCrashed, an item that runs longer than `timeout` seconds gives WorkerTimeout
    (the worker is stopped); in both cases a new worker takes the next item. `task` and `initializer` must be
    importable module-level functions (the worker is a separate Python process)."""
    pool = _new_pool(initializer, per_worker)
    try:
        for item in items:
            start = time.monotonic()
            future = pool.submit(task, item)
            try:
                yield item, future.result(timeout=timeout), None
            except FutureTimeout:
                _kill(pool)
                pool = _new_pool(initializer, per_worker)
                yield item, None, WorkerTimeout(f"no result after {time.monotonic() - start:.0f} s; worker stopped")
            except BrokenProcessPool as e:
                pool.shutdown(wait=False, cancel_futures=True)
                pool = _new_pool(initializer, per_worker)
                yield item, None, WorkerCrashed(f"the worker process died ({e})")
            except Exception as e:  # noqa: BLE001 - the task's own error, passed back to the caller
                yield item, None, e
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# ---------------------------------------------------------------------------- tasks used by the tests

def _test_task(item: tuple[str, float]) -> int:
    """For tests: ('ok', _) -> this worker's pid; ('fail', _) raises; ('crash', _) kills the worker;
    ('sleep', seconds) sleeps."""
    import os

    kind, value = item
    if kind == "fail":
        raise ValueError("bad item")
    if kind == "crash":
        os._exit(3)
    if kind == "sleep":
        time.sleep(value)
    return os.getpid()
