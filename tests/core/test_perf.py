import sys

import pytest

from kb.core.perf import disable_power_throttling


@pytest.mark.skipif(sys.platform != "win32", reason="Windows power throttling API")
def test_power_throttling_opt_out_applies_on_windows():
    assert disable_power_throttling() is True


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows behaviour")
def test_power_throttling_is_a_no_op_elsewhere():
    assert disable_power_throttling() is False
