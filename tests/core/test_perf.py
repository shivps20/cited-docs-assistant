import sys

import pytest

from kb.core.perf import disable_power_throttling, keep_awake
from kb.ingest.parse import ParseStats


@pytest.mark.skipif(sys.platform != "win32", reason="Windows power throttling API")
def test_power_throttling_opt_out_applies_on_windows():
    assert disable_power_throttling() is True


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows behaviour")
def test_power_throttling_is_a_no_op_elsewhere():
    assert disable_power_throttling() is False


def test_keep_awake_is_accepted_on_windows_and_released_after_errors():
    with pytest.raises(ValueError), keep_awake() as accepted:
        assert accepted is (sys.platform == "win32")
        raise ValueError("the block's error is not swallowed")


def stats(seconds, cpu, cached=False):
    return ParseStats("d", 24, seconds, cached, 300, 20, 5, 44, 0, 0, cpu_seconds=cpu)


def test_a_parse_that_used_almost_no_cpu_counts_as_stalled():
    assert stats(20963, 29).stalled                          # asleep for most of it
    assert not stats(15.4, 29).stalled                       # a normal parse uses more CPU than wall time
    assert not stats(90, 1).stalled                          # short: not worth a warning
    assert not stats(0, 0, cached=True).stalled
