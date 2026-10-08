"""Opt out of Windows power throttling (EcoQoS) so CPU model work keeps full speed.

On hybrid Intel CPUs (performance + efficiency cores) with Windows 11, a process that Windows
treats as background work (e.g. its console window is not in front, or an eco power plan is
active) can be moved to efficiency cores at low clock speed. The CPU reranker then took 25-50 s
instead of ~6 s for 20 candidates (TD-2). A process can opt out with SetProcessInformation(
ProcessPowerThrottling); this module does that once at startup. No effect on other systems.
"""

import ctypes
import sys

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
