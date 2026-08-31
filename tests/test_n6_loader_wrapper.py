from __future__ import annotations

import subprocess
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from odbench_outer.n6_loader_wrapper import (
    _is_stlink_gdbserver,
    _stop_process,
    _with_stlink_frequency,
    patch_gcc_compiler,
)


class N6LoaderWrapperTests(unittest.TestCase):
    def test_identifies_only_stlink_gdbserver_commands(self) -> None:
        self.assertTrue(_is_stlink_gdbserver(["/tools/ST-LINK_gdbserver", "-d"]))
        self.assertTrue(_is_stlink_gdbserver(["ST-LINK_gdbserver.exe", "-d"]))
        self.assertFalse(_is_stlink_gdbserver(["arm-none-eabi-gdb", "-batch"]))
        self.assertFalse(_is_stlink_gdbserver("ST-LINK_gdbserver -d"))

    def test_rewrites_stlink_frequency_only(self) -> None:
        command = ["/tools/ST-LINK_gdbserver", "-d", "--frequency", "2000"]
        self.assertEqual(
            _with_stlink_frequency(command, 200),
            ["/tools/ST-LINK_gdbserver", "-d", "--frequency", "200"],
        )
        gdb = ["arm-none-eabi-gdb", "-batch"]
        self.assertIs(_with_stlink_frequency(gdb, 200), gdb)

    def test_stop_process_terminates_running_process(self) -> None:
        process = Mock()
        process.poll.return_value = None
        _stop_process(process, timeout=2)
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=2)
        process.kill.assert_not_called()

    def test_stop_process_kills_process_that_does_not_terminate(self) -> None:
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("gdbserver", 2), 0]
        _stop_process(process, timeout=2)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_count, 2)

    def test_patch_reaps_gdbserver_before_returning(self) -> None:
        process = Mock()
        process.poll.return_value = None

        class VendorSubprocess:
            Popen = Mock(return_value=process)

        class GCCCompiler:
            logger = Mock()

            def load_and_run(self) -> int:
                VendorSubprocess.Popen(
                    ["/tools/ST-LINK_gdbserver", "-d", "--frequency", "2000"]
                )
                return 5

        original_popen = VendorSubprocess.Popen
        patch_gcc_compiler(
            GCCCompiler,
            VendorSubprocess,
            frequency_khz=200,
            failure_settle_seconds=0,
        )
        result = GCCCompiler().load_and_run()

        self.assertEqual(result, 5)
        self.assertIs(VendorSubprocess.Popen, original_popen)
        original_popen.assert_called_once_with(
            ["/tools/ST-LINK_gdbserver", "-d", "--frequency", "200"]
        )
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5.0)

    def test_patch_waits_after_failed_load(self) -> None:
        process = Mock()
        process.poll.return_value = None

        class VendorSubprocess:
            Popen = Mock(return_value=process)

        class GCCCompiler:
            logger = Mock()

            def load_and_run(self) -> int:
                VendorSubprocess.Popen(
                    ["/tools/ST-LINK_gdbserver", "--frequency", "2000"]
                )
                return 5

        patch_gcc_compiler(
            GCCCompiler,
            VendorSubprocess,
            failure_settle_seconds=3,
        )
        with patch("odbench_outer.n6_loader_wrapper.time.sleep") as sleep:
            GCCCompiler().load_and_run()
        sleep.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
