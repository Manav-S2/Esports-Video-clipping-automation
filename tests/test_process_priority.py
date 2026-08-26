"""Unit tests for pipeline.process_priority.

The Windows paths use ctypes against kernel32/ntdll, so os.name is patched and
the POSIX branch is exercised via a mocked os.kill. SIGSTOP/SIGCONT are patched
by name because they do not exist on Windows, letting these tests run anywhere.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import process_priority  # noqa: E402
from pipeline.process_priority import (  # noqa: E402
    deprioritize_background_thread,
    os_resume_pid,
    os_suspend_pid,
    subprocess_creationflags_low_priority,
)

FAKE_SIGSTOP = 19
FAKE_SIGCONT = 18


class DeprioritizeTests(unittest.TestCase):
    def test_is_a_noop_off_windows(self):
        with mock.patch.object(process_priority.os, "name", "posix"):
            self.assertIsNone(deprioritize_background_thread())

    def test_never_raises_when_ctypes_fails(self):
        # Priority control is an optimisation; a failure must not break the run.
        with mock.patch.object(process_priority.os, "name", "nt"), \
             mock.patch.dict("sys.modules", {"ctypes": None}):
            self.assertIsNone(deprioritize_background_thread())


class CreationFlagsTests(unittest.TestCase):
    def test_zero_off_windows(self):
        with mock.patch.object(process_priority.os, "name", "posix"):
            self.assertEqual(subprocess_creationflags_low_priority(), 0)

    def test_below_normal_flag_on_windows(self):
        fake = mock.Mock(CREATE_BELOW_NORMAL_PRIORITY_CLASS=0x4000)
        with mock.patch.object(process_priority.os, "name", "nt"), \
             mock.patch.object(process_priority, "subprocess", fake):
            self.assertEqual(subprocess_creationflags_low_priority(), 0x4000)

    def test_missing_flag_degrades_to_zero(self):
        fake = mock.Mock(spec=[])  # attribute absent, as on POSIX builds
        with mock.patch.object(process_priority.os, "name", "nt"), \
             mock.patch.object(process_priority, "subprocess", fake):
            self.assertEqual(subprocess_creationflags_low_priority(), 0)


class InvalidPidTests(unittest.TestCase):
    def test_non_positive_pids_are_rejected(self):
        for pid in (0, -1, -999):
            self.assertFalse(os_suspend_pid(pid))
            self.assertFalse(os_resume_pid(pid))

    def test_invalid_pid_never_signals(self):
        with mock.patch.object(process_priority.os, "kill") as kill:
            os_suspend_pid(0)
            os_resume_pid(-5)
        kill.assert_not_called()


class PosixSuspendResumeTests(unittest.TestCase):
    def test_suspend_sends_sigstop(self):
        with mock.patch.object(process_priority.os, "name", "posix"), \
             mock.patch.object(process_priority, "_SIGSTOP", FAKE_SIGSTOP), \
             mock.patch.object(process_priority.os, "kill") as kill:
            self.assertTrue(os_suspend_pid(4321))
        kill.assert_called_once_with(4321, FAKE_SIGSTOP)

    def test_resume_sends_sigcont(self):
        with mock.patch.object(process_priority.os, "name", "posix"), \
             mock.patch.object(process_priority, "_SIGCONT", FAKE_SIGCONT), \
             mock.patch.object(process_priority.os, "kill") as kill:
            self.assertTrue(os_resume_pid(4321))
        kill.assert_called_once_with(4321, FAKE_SIGCONT)

    def test_unavailable_signal_returns_false(self):
        # signal.SIGSTOP is absent on Windows; the lookup must degrade, not raise.
        with mock.patch.object(process_priority.os, "name", "posix"), \
             mock.patch.object(process_priority, "_SIGSTOP", None), \
             mock.patch.object(process_priority.os, "kill") as kill:
            self.assertFalse(os_suspend_pid(4321))
        kill.assert_not_called()

    def test_dead_process_returns_false(self):
        with mock.patch.object(process_priority.os, "name", "posix"), \
             mock.patch.object(process_priority, "_SIGSTOP", FAKE_SIGSTOP), \
             mock.patch.object(process_priority.os, "kill", side_effect=ProcessLookupError):
            self.assertFalse(os_suspend_pid(4321))

    def test_permission_error_returns_false(self):
        with mock.patch.object(process_priority.os, "name", "posix"), \
             mock.patch.object(process_priority, "_SIGCONT", FAKE_SIGCONT), \
             mock.patch.object(process_priority.os, "kill", side_effect=PermissionError):
            self.assertFalse(os_resume_pid(4321))

    def test_generic_oserror_returns_false(self):
        with mock.patch.object(process_priority.os, "name", "posix"), \
             mock.patch.object(process_priority, "_SIGSTOP", FAKE_SIGSTOP), \
             mock.patch.object(process_priority.os, "kill", side_effect=OSError("nope")):
            self.assertFalse(os_suspend_pid(4321))


class WindowsDispatchTests(unittest.TestCase):
    def test_suspend_dispatches_to_windows_helper(self):
        with mock.patch.object(process_priority.os, "name", "nt"), \
             mock.patch.object(
                 process_priority, "_windows_process_control", return_value=True
             ) as helper:
            self.assertTrue(os_suspend_pid(99, tag="ffmpeg"))
        helper.assert_called_once_with(99, "suspend", "ffmpeg")

    def test_resume_dispatches_to_windows_helper(self):
        with mock.patch.object(process_priority.os, "name", "nt"), \
             mock.patch.object(
                 process_priority, "_windows_process_control", return_value=True
             ) as helper:
            self.assertTrue(os_resume_pid(99))
        helper.assert_called_once_with(99, "resume", "")

    def test_windows_failure_is_reported_not_raised(self):
        with mock.patch.object(process_priority.os, "name", "nt"), \
             mock.patch.dict("sys.modules", {"ctypes": None}):
            self.assertFalse(os_suspend_pid(99, tag="encode"))


if __name__ == "__main__":
    unittest.main()
