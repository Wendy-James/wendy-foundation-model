#!/usr/bin/env python3
"""Unattended, bounded WendyFM M4 runner. No paid GPU, no automatic merges."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

BRANCH = "feat/m4-single-gpu"
KERNEL = "wendyzhan040513/wendyfm-fp32-pretrain"
TASKS = (
    ("scaling-plan", ("docs/m5-scaling-protocol.md",),
     "Write only docs/m5-scaling-protocol.md: a controlled, falsifiable WendyFM "
     "single-GPU profiling and scaling plan (fixed SHA, model/context/batch "
     "sweeps, precise throughput and CUDA memory definitions, warmup, OOM, "
     "stability, FP32 before BF16, cost ceiling, stop conditions). "
     "Do not fabricate benchmark numbers."),
    ("cpu-profiling", ("scripts/profile_cpu.py", "tests/test_profile_cpu.py"),
     "Create only scripts/profile_cpu.py and tests/test_profile_cpu.py: "
     "a deterministic, bounded CPU-only measurement harness reusing existing "
     "WendyFM model/training APIs. Include real wall-time and target token "
     "definitions, warmup, finite loss, and small local JSON results. "
     "Test failures, no fabricated CUDA metrics, no new dependencies."),
)
stop = threading.Event()
guard = threading.Lock()
state = {}
root = None
repo = None
pin = None
venv = None
finish_at = 0.0

def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

def record(key, status, note=""):
    with guard:
        state[key] = {"status": status, "note": note[:400], "utc": stamp()}
        temp = root / "status.tmp"
        temp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        temp.replace(root / "status.json")
        with (root / "runner.log").open("a") as f:
            f.write(stamp() + " " + key + " " + status + " " + note[:150] + "\n")

def cmd(args, cwd=None, timeout=120, log_path=None, input_text=None):
    if stop.is_set():
        raise RuntimeError("stop requested")
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    if cwd and str(cwd) != str(repo):
        env["PYTHONPATH"] = str(Path(cwd) / "src")
    proc = subprocess.run(args, cwd=cwd or repo, input=input_text, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout, env=env)
    if log_path:
        with log_path.open("a") as f:
            f.write(proc.stdout[-100000:])
    if proc.returncode:
        raise RuntimeError(f"{args[0]} failed (exit {proc.returncode}); see logs")
    return proc.stdout.strip()

def git(*args, cwd=None):
    return cmd(["git", *args], cwd=cwd).strip()

def setup():
    global pin, venv
    if not (repo / ".git").exists():
        raise RuntimeError("M4 worktree not found")
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("incorrect branch; preserve existing worktree")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("dirty M4 worktree; don't overwrite")
    git("fetch", "origin", BRANCH)
    pin = git("rev-parse", "HEAD")
    if pin != git("rev-parse", "origin/" + BRANCH) or not re.fullmatch("[0-9a-f]{40}", pin):
        raise RuntimeError("local/remote M4 SHA mismatch")
    for rel in ("scripts/infra/prepare_kaggle_gpu_pretrain.py",
                "scripts/infra/remote_gpu_pretrain.py", "configs/gpu_fp32.json"):
        if not (repo / rel).is_file():
            raise RuntimeError("missing " + rel)
    choices = (repo / ".venv",
               Path.home() / "workspaces/projects/wendyfm-m3-trainer/.venv",
               Path.home() / "workspaces/projects/wendy-foundation-model/.venv")
    venv = next((p for p in choices
                 if (p / "bin/python").is_file() and (p / "bin/ruff").is_file()), None)
    if venv is None:
        raise RuntimeError("No existing project .venv; will not auto-install")
    for tool in ("git", "python3", "kaggle", "codex", "gh"):
        if not shutil.which(tool):
            raise RuntimeError("missing CLI: " + tool)
    help_output = cmd(["kaggle", "kernels", "push", "--help"])
    if "--accelerator" not in help_output or "--timeout" not in help_output:
        raise RuntimeError("Kaggle CLI cannot cap accelerator/runtime")
    record("preflight", "PASS", "pinned M4 SHA=" + pin)

def make_worktree(name):
    target = root / "worktrees" / name.replace("/", "_")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or git("ls-remote", "--heads", "origin", name):
        raise RuntimeError("branch/worktree already exists; no overwrite")
    git("worktree", "add", "-b", name, str(target), pin)
    return target

def commit_scoped(tree, branch, files, title):
    if git("rev-parse", "HEAD", cwd=tree) != pin:
        raise RuntimeError("unexpected agent commit")
    actual = [line[3:] for line in git("status", "--porcelain", "--untracked-files=all", cwd=tree).splitlines()]
    if not actual or sorted(actual) != sorted(files):
        raise RuntimeError("unexpected or missing file edits: " + repr(actual))
    for rel in files:
        item = tree / rel
        if not item.is_file() or item.is_symlink() or item.stat().st_size > 100000:
            raise RuntimeError("large/nonregular file: " + rel)
        data = item.read_bytes()
        for token in (b"-----BEGIN PRIVATE KEY", b"OPENAI_API_KEY=", b"WANDB_API_KEY=",
                      b"HF_TOKEN=", b"KAGGLE_KEY=", b"/Users/admin/.netrc"):
            if token in data:
                raise RuntimeError("possible credential found")
    cmd([str(venv / "bin/python"), "-m", "pytest", "-q"], cwd=tree,
        timeout=300, log_path=root / (branch.replace("/", "_") + "-pytest.log"))
    cmd([str(venv / "bin/ruff"), "check", "."], cwd=tree,
        timeout=180, log_path=root / (branch.replace("/", "_") + "-ruff.log"))
    git("diff", "--check", cwd=tree)
    git("add", "--", *files, cwd=tree)
    if sorted(git("diff", "--cached", "--name-only", cwd=tree).splitlines()) != sorted(files):
        raise RuntimeError("unexpected staged changes")
    git("diff", "--cached", "--check", cwd=tree)
    git("commit", "-m", title, cwd=tree)
    git("push", "-u", "origin", branch, cwd=tree)
    if not git("ls-remote", "origin", "refs/heads/" + branch, cwd=tree).startswith(git("rev-parse", "HEAD", cwd=tree)):
        raise RuntimeError("push verification failed")
    try:
        cmd(["gh", "pr", "create", "--draft", "--base", BRANCH, "--head", branch,
             "--title", title, "--body",
             "WendyFM bounded overnight result. Review required; no automatic merge."],
            cwd=tree, log_path=root / (branch.replace("/", "_") + "-pr.log"))
    except Exception:
        record(branch, "PUSHED_PR_PENDING", "GitHub branch exists; draft PR creation failed")
        return
    record(branch, "PUSHED_PR_DRAFT")

def github_gpu_evidence(payload):
    man = payload["manifest"]
    if man["source_commit_sha"] != pin or payload.get("checkpoint_resume_exact") is not True:
        raise RuntimeError("invalid measured GPU evidence")
    branch = "auto/m4-gpu-evidence-" + root.name
    tree = make_worktree(branch)
    json_path = "benchmarks/infra/m4_gpu_fp32_" + root.name + ".json"
    doc_path = "docs/m4_gpu_fp32_" + root.name + ".md"
    (tree / json_path).parent.mkdir(parents=True, exist_ok=True)
    (tree / json_path).write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    report = (
        "# Measured WendyFM M4 single-GPU experiment\n\n"
        "Pinned SHA: " + pin + "\n\n"
        "GPU device: " + str(man["cuda_device_name"]) + "\n\n"
        "Optimizer steps: " + str(man["optimizer_steps"]) + "\n\n"
        "Exact CUDA resume: " + str(payload["checkpoint_resume_exact"]) + "\n\n"
        "Validation loss before: " + str(man["initial_validation_loss"]) + "\n\n"
        "Validation loss after: " + str(man["validation_loss"]) + "\n\n"
        "Measured bounded-run target tokens/s: " + str(man["tokens_per_second"]) + "\n\n"
        "Training wall time seconds: " + str(man["training_wall_time_seconds"]) + "\n\n"
        "Peak GPU allocated bytes: " + str(man["peak_cuda_memory_bytes"]) + "\n\n"
        "Tiny synthetic fixture; 8 steps only. Not a real model quality or steady-state "
        "performance/scaling benchmark. No weights, private logs or secrets committed.\n"
    )
    (tree / doc_path).write_text(report)
    commit_scoped(tree, branch, [json_path, doc_path],
                  "exp(m4): record real bounded Kaggle FP32 GPU evidence")

def kernel_status():
    """Return current status, or None only if an owned slug is verifiably absent.

    Kaggle sometimes reports "Permission 'kernels.get' was denied" for a
    not-yet-created kernel. That message is NOT by itself proof of absence:
    confirm using a successful, non-truncated authenticated --mine listing.
    """
    p = subprocess.run(["kaggle", "kernels", "status", KERNEL],
                       capture_output=True, text=True, timeout=60)
    s = (p.stdout or "") + (p.stderr or "")
    if p.returncode == 0:
        return s.strip()
    lower = s.lower()
    if any(t in lower for t in ("404", "not found", "does not exist", "no kernel")):
        return None
    if "cannot access kernel" in lower and "permission 'kernels.get' was denied" in lower:
        listing = subprocess.run(["kaggle", "kernels", "list", "--mine",
                                  "--page-size", "100"],
                                 capture_output=True, text=True, timeout=60)
        if listing.returncode != 0:
            raise RuntimeError("Kaggle list --mine failed; will not submit")
        owner = KERNEL.split("/", 1)[0]
        refs = re.findall(
            r"(?m)^\s*(" + re.escape(owner) + r"/[A-Za-z0-9_-]+)\s+",
            listing.stdout or "",
        )
        if not refs or len(refs) >= 100:
            raise RuntimeError("owned kernel list empty/possibly truncated; will not submit")
        if KERNEL in refs:
            raise RuntimeError("kernel listed as owned but status inaccessible; will not submit")
        return None
    raise RuntimeError("remote kernel status uncertain: will not submit")

def gpu_worker():
    try:
        record("gpu", "PREFLIGHT")
        exists = kernel_status()
        marker = root / "submission-attempted.json"
        if exists is None:
            if marker.exists():
                raise RuntimeError("submission already attempted; no retry")
            job = root / "kaggle-job"
            job.mkdir()
            cmd(["python3", "scripts/infra/prepare_kaggle_gpu_pretrain.py", "prepare",
                 "--job-dir", str(job), "--user", "wendyzhan040513", "--sha", pin])
            md = json.loads((job / "kernel-metadata.json").read_text())
            source = (job / "remote_gpu_pretrain.py").read_text()
            if not (md["id"] == KERNEL and md["is_private"] is True
                    and md["enable_gpu"] is True and md["enable_tpu"] is False
                    and md["enable_internet"] is True
                    and all(not md[k] for k in ("dataset_sources", "competition_sources",
                                                "kernel_sources", "model_sources"))
                    and ('EXPECTED_COMMIT_SHA = "' + pin + '"') in source):
                raise RuntimeError("unsafe/unpinned GPU metadata")
            marker.write_text(json.dumps({"sha": pin, "kernel": KERNEL, "utc": stamp()}))
            record("gpu", "SUBMITTING_ONCE")
            cmd(["kaggle", "kernels", "push", "-p", str(job),
                 "--accelerator", "NvidiaTeslaT4", "--timeout", "300"],
                timeout=180, log_path=root / "kaggle_push.log")
            record("gpu", "SUBMITTED")
        elif "KernelWorkerStatus.COMPLETE" not in exists:
            # A pre-existing kernel is not ours: never overwrite it.
            raise RuntimeError("kernel already exists and is not COMPLETE; no resubmission")
        else:
            record("gpu", "EXISTING_COMPLETE", "will verify SHA from downloaded JSON")
        expiry = min(time.monotonic() + 90*60, finish_at)
        while time.monotonic() < expiry and not stop.is_set():
            s = kernel_status()
            if s and "KernelWorkerStatus.COMPLETE" in s:
                break
            if s and any(x in s for x in ("KernelWorkerStatus.ERROR",
                                          "KernelWorkerStatus.FAILED",
                                          "KernelWorkerStatus.CANCELLED")):
                raise RuntimeError("remote job failed; do not resubmit")
            record("gpu", "WAITING", (s or "unknown")[:150])
            stop.wait(60)
        else:
            record("gpu", "PENDING", "No retry; inspect Kaggle after return")
            return
        out = root / "kaggle-output"
        out.mkdir(exist_ok=True)
        cmd(["kaggle", "kernels", "output", KERNEL, "-p", str(out)],
            timeout=180, log_path=root / "kaggle_download.log")
        results = list(out.rglob("gpu-pretrain-result.json"))
        if len(results) != 1:
            raise RuntimeError("GPU results missing or ambiguous")
        result = results[0]
        cmd(["python3", "scripts/infra/prepare_kaggle_gpu_pretrain.py",
             "verify", "--result", str(result), "--sha", pin])
        github_gpu_evidence(json.loads(result.read_text()))
        record("gpu", "PASS", "verified M4 results committed to isolated draft PR")
    except Exception as e:
        record("gpu", "BLOCKED", str(e))

def codex_worker(calls, limit_minutes):
    for name, paths, prompt in TASKS[:calls]:
        if stop.is_set():
            break
        branch = "auto/m5-" + name + "-" + root.name
        try:
            tree = make_worktree(branch)
            record(name, "CODING")
            instruction = (prompt + "\nRead AGENTS.md and current project files. "
                           "Do not edit anything outside your allowed files. "
                           "No network, no GPU, no paid services, no dependency "
                           "installation. Do not commit/push. Use only preinstalled "
                           "project .venv at " + str(venv) + ". "
                           "Stop after scoped tests finish.")
            cmd(["codex", "exec", "--sandbox", "workspace-write",
                 "-C", str(tree), "-"], cwd=tree, timeout=limit_minutes * 60,
                input_text=instruction, log_path=root / ("codex_" + name + ".log"))
            record(name, "VALIDATING")
            commit_scoped(tree, branch, list(paths),
                          "feat(m5): " + name + " bounded overnight work")
        except Exception as e:
            record(name, "BLOCKED", str(e))
    record("codex", "DONE", "at most " + str(calls) + " API-key calls")

def on_signal(signum, frame):
    stop.set()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path,
                    default=Path.home() / "workspaces/projects/wendyfm-m4-gpu")
    ap.add_argument("--hours", type=float, default=14.5)
    ap.add_argument("--codex-calls", type=int, choices=(0, 1, 2), default=2)
    ap.add_argument("--codex-minutes", type=int, default=20)
    args = ap.parse_args()
    if not 14 <= args.hours <= 16 or not 1 <= args.codex_minutes <= 30:
        ap.error("hours must be 14-16; each Codex call 1-30 minutes")
    global root, repo, finish_at, state
    repo = args.repo.expanduser().resolve()
    root = Path.home() / "wendyfm-overnight" / (
        "m4-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    root.mkdir(parents=True, exist_ok=False)
    (root.parent / "latest-m4.txt").write_text(str(root) + "\n")
    lock = (root.parent / "m4.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another overnight M4 controller is running")
    finish_at = time.monotonic() + 3600 * args.hours
    state = {"meta": {"utc": stamp(), "hours": args.hours,
                      "max_codex_calls": args.codex_calls,
                      "max_codex_minutes_each": args.codex_minutes,
                      "max_gpu_submissions": 1, "paid_gpu_allowed": False}}
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        setup()
    except Exception as e:
        record("controller", "BLOCKED", str(e))
        raise SystemExit(2)
    record("controller", "RUNNING", "GPU x1 and bounded Codex work in parallel")
    with ThreadPoolExecutor(max_workers=2) as pool:
        g = pool.submit(gpu_worker)
        c = pool.submit(codex_worker, args.codex_calls, args.codex_minutes)
        next_beat = 0.0
        while time.monotonic() < finish_at and not stop.is_set():
            if time.monotonic() >= next_beat:
                record("heartbeat", "ALIVE",
                       "gpu_done=" + str(g.done()) + " codex_done=" + str(c.done()))
                next_beat = time.monotonic() + 3600
            stop.wait(60)
    record("controller", "EXITED",
           "14+ hours elapsed or stop requested; inspect status.json and Draft PRs")

if __name__ == "__main__":
    main()
