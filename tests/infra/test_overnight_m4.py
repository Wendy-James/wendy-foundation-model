"""Offline regression tests for safe Kaggle status handling in unattended runner."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/infra/overnight_m4.py"
spec = importlib.util.spec_from_file_location("overnight_m4_for_test", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

DENIED = (
    "Cannot access kernel 'wendyzhan040513/wendyfm-fp32-pretrain' "
    "(Permission 'kernels.get' was denied)."
)
MY_LIST = (
    "ref title author lastRunTime totalVotes\n"
    "wendyzhan040513/wendyfm-model-smoke wendyfm-model-smoke Wendy\n"
    "wendyzhan040513/wendyfm-tokenizer-smoke wendyfm-tokenizer-smoke Wendy\n"
)


def proc(command, code=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, code, stdout, stderr)


class KernelStatusTests(unittest.TestCase):
    def test_explicit_complete_status(self):
        with patch.object(runner.subprocess, "run", return_value=proc([], stdout="KernelWorkerStatus.COMPLETE")) as run:
            self.assertEqual(runner.kernel_status(), "KernelWorkerStatus.COMPLETE")
            self.assertEqual(run.call_count, 1)

    def test_not_found_uses_no_listing(self):
        with patch.object(runner.subprocess, "run", return_value=proc([], 1, stderr="404 not found")) as run:
            self.assertIsNone(runner.kernel_status())
            self.assertEqual(run.call_count, 1)

    def test_denied_and_complete_owned_listing_proves_absent(self):
        with patch.object(runner.subprocess, "run", side_effect=[
            proc([], 1, stderr=DENIED), proc([], stdout=MY_LIST)
        ]) as run:
            self.assertIsNone(runner.kernel_status())
            self.assertEqual(run.call_count, 2)

    def test_denied_but_owned_target_stops(self):
        listing = MY_LIST + runner.KERNEL + " target Wendy\n"
        with patch.object(runner.subprocess, "run", side_effect=[
            proc([], 1, stderr=DENIED), proc([], stdout=listing)
        ]):
            with self.assertRaisesRegex(RuntimeError, "listed as owned"):
                runner.kernel_status()

    def test_denied_and_list_failure_stops(self):
        with patch.object(runner.subprocess, "run", side_effect=[
            proc([], 1, stderr=DENIED), proc([], 1, stderr="Unauthorized")
        ]):
            with self.assertRaisesRegex(RuntimeError, "list --mine failed"):
                runner.kernel_status()

    def test_denied_and_empty_list_stops(self):
        with patch.object(runner.subprocess, "run", side_effect=[
            proc([], 1, stderr=DENIED), proc([], stdout="ref title\n")
        ]):
            with self.assertRaisesRegex(RuntimeError, "empty/possibly truncated"):
                runner.kernel_status()

    def test_denied_and_full_page_stops(self):
        listing = "\n".join(
            f"wendyzhan040513/job-{i} title Wendy" for i in range(100)
        )
        with patch.object(runner.subprocess, "run", side_effect=[
            proc([], 1, stderr=DENIED), proc([], stdout=listing)
        ]):
            with self.assertRaisesRegex(RuntimeError, "empty/possibly truncated"):
                runner.kernel_status()

    def test_unknown_permissions_error_does_not_create_job(self):
        with patch.object(runner.subprocess, "run", return_value=proc([], 1, stderr="401 Unauthorized")):
            with self.assertRaisesRegex(RuntimeError, "status uncertain"):
                runner.kernel_status()


if __name__ == "__main__":
    unittest.main()
