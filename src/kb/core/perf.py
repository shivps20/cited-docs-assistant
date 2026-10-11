"""Opt out of Windows power throttling (EcoQoS) so CPU model work keeps full speed.

On hybrid Intel CPUs (performance + efficiency cores) with Windows 11, a process that Windows
treats as background work (e.g. its console window is not in front, or an eco power plan is
active) can be moved to efficiency cores at low clock speed. The CPU reranker then took 25-50 s
instead of ~6 s for 20 candidates (TD-2). A process can opt out with SetProcessInformation(
ProcessPowerThrottling); this module does that once at startup. No effect on other systems.

Long batches (kb parse, chunk, index, the evaluations) also keep Windows awake while they run
(`keep_awake`): with the screen off, a laptop goes into Modern Standby, which first slows every
process down heavily and then sleeps. A 221-document `kb parse` took 7.1 h this way; the same
documents parse in about 1.3 h awake (the slowest one: 20,963 s in the batch, 15 s on its own).
"""

import ctypes
import sys
from collections.abc import Iterator
from contextlib import contextmanager

_ES_CONTINUOUS = 0x80000000                    # SetThreadExecutionState flags
_ES_SYSTEM_REQUIRED = 0x00000001               # "this thread needs the system awake" (the display may sleep)
_ES_DISPLAY_REQUIRED = 0x00000002              # keep the display on: with it off, Modern Standby pauses the process

_PROCESS_POWER_THROTTLING = 4                  # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_THROTTLING_CURRENT_VERSION = 1
_THROTTLING_EXECUTION_SPEED = 0x1              # the "slow down to save power" behaviour


class _ThrottlingState(ctypes.Structure):
    """PROCESS_POWER_THROTTLING_STATE from the Windows API."""

    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def disable_power_throttling() -> bool:
    """Ask Windows never to throttle this process's execution speed; True if it was applied."""
    if sys.platform != "win32":
        return False
    try:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Explicit types: without them ctypes truncates the process handle and the call fails.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        # ControlMask = take control of execution-speed throttling; StateMask = 0 means "off".
        state = _ThrottlingState(_THROTTLING_CURRENT_VERSION, _THROTTLING_EXECUTION_SPEED, 0)
        ok = kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING,
                                            ctypes.byref(state), ctypes.sizeof(state))
    except (AttributeError, OSError):
        return False                            # older Windows without the API
    return bool(ok)


@contextmanager
def keep_awake() -> Iterator[bool]:
    """Keep Windows from sleeping, and the display on, while the block runs.

    The display matters: on Modern Standby laptops a screen that turns off (by timeout or a lock policy) lets
    Windows pause desktop programs even when plugged in with the lid open (a parse used 14 s of CPU in 1,240 s).

    Yields True when the request was accepted. Closing the laptop lid can still put it to sleep,
    depending on the power settings. No effect on other systems.
    """
    kernel32 = None
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
            kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
        except (AttributeError, OSError):
            kernel32 = None
    accepted = bool(kernel32 and kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED))
    try:
        yield accepted
    finally:
        if accepted:
            kernel32.SetThreadExecutionState(_ES_CONTINUOUS)       # back to normal sleep behaviour
