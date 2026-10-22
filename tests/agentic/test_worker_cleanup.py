"""B2-H1 regression adapted locally; independent QA source stays unchanged."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal cleanup regression")
def test_sigterm_resistant_policy_still_has_a_bounded_deadline(tmp_path):
    script = tmp_path / "b2_resistant_worker.py"
    script.write_text(
        "import signal, time, sys, multiprocessing\n"
        "from pathlib import Path\n"
        "from wendyfm.agentic.runtime import decide\n"
        "from wendyfm.agentic.environment import LookupEnvironment\n"
        "def resistant(observation):\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    Path(sys.argv[1]).touch()\n"
        "    time.sleep(30)\n"
        "if __name__ == '__main__':\n"
        "    try:\n"
        "        decide(resistant, LookupEnvironment().reset(17), 1.0)\n"
        "    except TimeoutError:\n"
        "        assert Path(sys.argv[1]).is_file(), 'worker did not reach SIGTERM ignore'\n"
        "        assert not multiprocessing.active_children(), 'worker leaked'\n"
        "    else:\n"
        "        raise AssertionError('expected policy timeout')\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path / "ready")], start_new_session=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    timed_out = False
    try:
        process.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        # The independent harness kills the entire group, including the spawned worker.
        if timed_out:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=3)
    assert not timed_out, "policy cleanup exceeded the independent 3-second deadline"
    assert process.returncode == 0


def test_cleanup_failure_preserves_original_exception():
    from wendyfm.agentic.processes import cleanup_on_exit

    def failed_cleanup():
        raise RuntimeError("forced cleanup failed")

    with pytest.raises(TimeoutError, match="decision deadline") as caught:
        with cleanup_on_exit(failed_cleanup):
            raise TimeoutError("decision deadline")
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert caught.value.__notes__ == ["policy worker cleanup also failed"]


def test_cleanup_failure_on_success_is_reported():
    from wendyfm.agentic.processes import cleanup_on_exit

    def failed_cleanup():
        raise RuntimeError("forced cleanup failed")

    with pytest.raises(RuntimeError, match="forced cleanup failed"):
        with cleanup_on_exit(failed_cleanup):
            pass
