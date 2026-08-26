"""CPU-priority and suspend/resume controls for capture and encode processes.

The live pipeline records a stream while simultaneously encoding and scoring the
previous round. Encoding is CPU-hungry enough to starve the capture loop, so
background work is de-prioritised and long encodes can be suspended outright
while the HUD is being read.

Every function is best-effort: priority control is an optimisation, never a
correctness requirement, so failures are reported and swallowed rather than
raised. Windows uses ctypes against kernel32/ntdll; POSIX uses SIGSTOP/SIGCONT.
"""

from __future__ import annotations

import os
import signal
import subprocess

# Windows process access rights. PROCESS_SUSPEND_RESUME alone often fails
# OpenProcess against a child ffmpeg, so query rights are requested too.
_PROCESS_SUSPEND_RESUME = 0x0800
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_THREAD_PRIORITY_BELOW_NORMAL = -1

# SIGSTOP/SIGCONT do not exist in Python's signal module on Windows, so they are
# looked up defensively rather than referenced directly.
_SIGSTOP = getattr(signal, "SIGSTOP", None)
_SIGCONT = getattr(signal, "SIGCONT", None)


def deprioritize_background_thread() -> None:
    """Lower calling thread CPU priority so screenshot/ffmpeg/HUD paths stay responsive (Windows)."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), _THREAD_PRIORITY_BELOW_NORMAL)
    except Exception:
        pass


def subprocess_creationflags_low_priority() -> int:
    """Windows: start child processes (ffmpeg / caption burn) below normal CPU priority.

    Keeps the main HUD capture thread and interactive loop more responsive under
    heavy encode load. On non-Windows, returns 0 (no extra flags).
    """
    if os.name != "nt":
        return 0
    # The constant only exists in the Windows build of the subprocess module.
    try:
        return int(getattr(subprocess, "CREATE_BELOW_NORMAL_PRIORITY_CLASS", 0))
    except (AttributeError, ValueError, TypeError):
        return 0


def _windows_process_control(pid: int, action: str, tag: str) -> bool:
    """Call ``NtSuspendProcess``/``NtResumeProcess`` on ``pid``. Returns success.

    Shared by suspend and resume, which differ only in the ntdll entry point and
    the wording of their diagnostics.
    """
    suffix = f" {tag}" if tag else ""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        ntdll = ctypes.windll.ntdll
        access = _PROCESS_SUSPEND_RESUME | _PROCESS_QUERY_LIMITED_INFORMATION
        handle = kernel32.OpenProcess(access, False, ctypes.c_uint(pid))
        if not handle:
            err = int(kernel32.GetLastError())
            print(
                f"[live] WARN: OpenProcess({action}) failed pid={pid} winerr={err}{suffix}",
                flush=True,
            )
            return False
        try:
            entry = "NtSuspendProcess" if action == "suspend" else "NtResumeProcess"
            status = int(getattr(ntdll, entry)(handle))
            if status != 0:
                print(
                    f"[live] WARN: {entry} failed pid={pid} status={status:#x}{suffix}",
                    flush=True,
                )
            return status == 0
        finally:
            kernel32.CloseHandle(handle)
    except Exception as exc:
        print(f"[live] WARN: {action} pid={pid} raised {exc!r}{suffix}", flush=True)
        return False


def _posix_signal(pid: int, sig: int | None) -> bool:
    if sig is None:
        return False
    try:
        os.kill(pid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def os_suspend_pid(pid: int, *, tag: str = "") -> bool:
    """Suspend every thread in ``pid`` (Windows ``NtSuspendProcess``; POSIX ``SIGSTOP``). Best-effort."""
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_process_control(pid, "suspend", tag)
    return _posix_signal(pid, _SIGSTOP)


def os_resume_pid(pid: int, *, tag: str = "") -> bool:
    """Resume ``pid`` after :func:`os_suspend_pid` (Windows ``NtResumeProcess``; POSIX ``SIGCONT``)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_process_control(pid, "resume", tag)
    return _posix_signal(pid, _SIGCONT)
