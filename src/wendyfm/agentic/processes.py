"""Bounded shutdown shared by the scripted and persistent model workers."""

import time
from contextlib import contextmanager

TERMINATE_SECONDS = 0.5
KILL_SECONDS = 0.5


def stop_worker(process, connection):
    """Allow SIGTERM cleanup, then force exit; never join without a deadline."""
    try:
        if process.is_alive():
            deadline = time.monotonic() + TERMINATE_SECONDS
            process.terminate()
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            deadline = time.monotonic() + KILL_SECONDS
            process.kill()
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            # Keep the process handle available for diagnosis/recovery.
            raise RuntimeError("policy worker survived forced cleanup deadline")
        process.join(timeout=0)
        process.close()
    finally:
        connection.close()


@contextmanager
def cleanup_on_exit(close):
    """Report cleanup failures without replacing an in-flight policy exception."""
    try:
        yield
    except BaseException as original:
        try:
            close()
        except Exception as cleanup_error:
            original.add_note("policy worker cleanup also failed")
            raise original from cleanup_error
        raise
    else:
        close()
