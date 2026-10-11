from kb.core.workers import WorkerCrashed, WorkerTimeout, _test_task, run_isolated


def test_items_run_in_order_in_a_worker_that_is_replaced_every_n_items():
    """Results come back in order; after `per_worker` items a new worker process takes over."""
    out = list(run_isolated([("ok", 0)] * 4, _test_task, per_worker=2, timeout=60))
    pids = [result for _, result, error in out]
    assert all(error is None for _, _, error in out)
    assert pids[0] == pids[1] != pids[2] == pids[3]


def test_a_failing_crashing_or_hanging_item_fails_alone_and_the_batch_goes_on():
    """A task error comes back as the error; a dead worker and a timeout start a fresh worker for the next item."""
    items = [("fail", 0), ("ok", 0), ("crash", 0), ("ok", 0), ("sleep", 30), ("ok", 0)]
    out = list(run_isolated(items, _test_task, per_worker=50, timeout=5))
    errors = [type(error).__name__ if error else None for _, _, error in out]
    assert errors == ["ValueError", None, "WorkerCrashed", None, "WorkerTimeout", None]
    assert isinstance(out[2][2], WorkerCrashed) and isinstance(out[4][2], WorkerTimeout)
    assert out[1][1] != out[3][1] != out[5][1]          # a new worker after the crash and after the timeout
