# WendyFM M4 unattended overnight runbook (2026-10-09)

This is a proposed, reviewable, **opt-in** Mac runner. It has NOT been run on the user's Mac by ChatGPT. Only execute after the local safety preflight and free Kaggle GPU quota are confirmed.

## Limits and responsibilities

- B1: a Mac process coordinates one already-prepared **private**, SHA-pinned Kaggle M4 FP32 job; it verifies downloaded JSON, creates a limited experiment-evidence branch/draft PR, and writes hourly status for **14.5 hours**.
- B2: the controller can launch **up to two** bounded \`codex exec --sandbox workspace-write\` tasks in isolated worktrees (a scaling **protocol** and a CPU profiling **harness**). Each gets at most 20 minutes. The controller owns the tests, commits, pushes, and draft PRs; agents must not push.
- No automatic PR merges, no paid GPU, no Kaggle retries, no model weight uploads, no company assets, no dependency installation.
- **Important:** time-bounding Codex does NOT enforce a strict token/currency budget. The current Mac Codex CLI is API-key authenticated. To avoid any Codex API spend, launch with \`--codex-calls 0\`.
- This does not run 14.5 hours of GPU training. The GPU run is limited to 300 seconds; the remainder is lightweight monitoring/heartbeats. Extra GPU spending is not authorized.
- \`NvidiaTeslaT4\` selects Kaggle's standard T4 ×2 allocation; WendyFM uses \`cuda:0\` only. No multi-GPU speedup is claimed.
- Existing Kaggle kernel slug is **never** overwritten or resubmitted. If it already exists in a nonterminal state, stop and inspect; a previously COMPLETE job may only be reused after downloaded SHA-verified result validation.

## Requirements

- Mac plugged into AC, lid open and network available. \`caffeinate\` helps prevent **idle** sleep; it cannot guarantee operation with lid closed, system shutdown, power outage or network loss.
- Worktree \`~/workspaces/projects/wendyfm-m4-gpu\` exists, is clean, on \`feat/m4-single-gpu\`, and HEAD matches \`origin/feat/m4-single-gpu\`.
- Git/GitHub CLI, Kaggle CLI, Codex CLI and a preexisting WendyFM Python/Ruff virtualenv are available. No tools installed automatically.
- Kaggle free GPU quota checked by operator. One private run with \`NvidiaTeslaT4 --timeout 300\`.
- The published controller is kept in the separate \`ops/m4-overnight-20261009\` branch/PR and must not be merged without review.

## B1 launch (once; no repeated launches)

Run these commands **in B1 Mac terminal**. They retrieve a reviewed version via authenticated Git and run it outside the M4 worktree:

~~~sh
set -e
cd ~/workspaces/projects/wendyfm-m4-gpu
git fetch origin ops/m4-overnight-20261009
mkdir -p ~/wendyfm-overnight
git show origin/ops/m4-overnight-20261009:scripts/infra/overnight_m4.py > ~/wendyfm-overnight/overnight_m4.py
/usr/bin/python3 -m py_compile ~/wendyfm-overnight/overnight_m4.py

if [ -f ~/wendyfm-overnight/m4.pid ] && kill -0 "$(cat ~/wendyfm-overnight/m4.pid)" 2>/dev/null; then
  echo "BLOCKED: controller already running"
else
  nohup /usr/bin/python3 ~/wendyfm-overnight/overnight_m4.py \
    --repo ~/workspaces/projects/wendyfm-m4-gpu \
    --hours 14.5 --codex-calls 2 --codex-minutes 20 \
    > ~/wendyfm-overnight/m4-launch.log 2>&1 < /dev/null &
  echo $! > ~/wendyfm-overnight/m4.pid
  nohup caffeinate -i -s -w "$(cat ~/wendyfm-overnight/m4.pid)" \
    > ~/wendyfm-overnight/caffeinate.log 2>&1 < /dev/null &
  sleep 15
  cat ~/wendyfm-overnight/m4-launch.log
  DIR="$(cat ~/wendyfm-overnight/latest-m4.txt)"
  cat "$DIR/status.json"
fi
~~~

Only leave after the controller status has passed preflight and reads \`controller: RUNNING\`. If \`BLOCKED\`, fix/inspect in daylight; **never automatically resubmit a GPU job**. The \`codex-calls 2\` option incurs bounded-but-not-monetarily-hard-capped API usage; change to \`0\` if cost control is more important than unattended code generation.

## Status and stop

~~~sh
DIR="$(cat ~/wendyfm-overnight/latest-m4.txt)"
cat "$DIR/status.json"
tail -60 "$DIR/runner.log"
ps -p "$(cat ~/wendyfm-overnight/m4.pid)" -o pid,etime,command
~~~

Gracefully request stop with:

~~~sh
kill -TERM "$(cat ~/wendyfm-overnight/m4.pid)"
~~~

If stopped, do not rerun the controller without inspecting its Kaggle submission marker and remote status. Logs and raw remote output remain local; only vetted small results are sent to GitHub.

## Review the next morning

1. Inspect \`status.json\` and \`runner.log\`; distinguish PASS, PENDING, BLOCKED and PUSHED_PR_DRAFT.
2. Examine the Kaggle private job and verified JSON. Do not claim a GPU baseline unless \`status=PASS\` and the exact SHA and checkpoint comparisons pass.
3. Inspect any new \`auto/m4-gpu-evidence-*\` and \`auto/m5-*\` draft PRs. Do not automatically merge.
4. Keep M4 and existing PRs untouched unless you explicitly approve a merge.
5. If dependencies, Kaggle quota, credentials, test checks or sandbox block work, report the failure rather than removing safeguards.

Experiments are bounded and synthetic until a separately reviewed real-text pretraining/scaling study is authorized.


## GPU-only recovery after Kaggle \`kernels.get\` access-denied (2026-10-10)

The initial B1 run reached \`gpu=BLOCKED\` because the nonexistent
\`wendyzhan040513/wendyfm-fp32-pretrain\` slug returned
\`Permission 'kernels.get' was denied\` rather than a 404. The same
authenticated Kaggle account could list its own notebooks, and the
target slug was absent from a complete short \`--mine\` list.

The revised \`kernel_status()\` treats this very specific denial as
*possibly absent* and verifies the owned kernel listing before any
submission. If listing fails, is empty, contains the target, or appears
truncated, the recovery remains blocked. Do not treat arbitrary 401/403
messages as permission to submit.

**Do not restart the running 14.5h controller**: it has already finished
the two Codex tasks and owns \`m4.lock\`. The new \`--gpu-only\` option uses
its own local lock/status folder, makes **zero Codex calls**, performs
**at most one** private Kaggle submission with a 300-second timeout,
verifies the small result, and publishes a separate draft evidence PR.

B1 Mac terminal after confirming available *free* Kaggle GPU quota:

~~~sh
(
set -e
cd ~/workspaces/projects/wendyfm-m4-gpu
git fetch origin ops/m4-overnight-20261009
mkdir -p ~/wendyfm-overnight
git show origin/ops/m4-overnight-20261009:scripts/infra/overnight_m4.py \
  > ~/wendyfm-overnight/overnight_m4.py
/usr/bin/python3 -m py_compile ~/wendyfm-overnight/overnight_m4.py
PIDFILE="$HOME/wendyfm-overnight/m4-gpu-only.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "BLOCKED: GPU-only recovery is already running"
  exit 1
fi
nohup /usr/bin/python3 ~/wendyfm-overnight/overnight_m4.py \
  --gpu-only --repo ~/workspaces/projects/wendyfm-m4-gpu \
  > ~/wendyfm-overnight/m4-gpu-only-launch.log 2>&1 < /dev/null &
echo $! > "$PIDFILE"
nohup caffeinate -i -s -w "$(cat "$PIDFILE")" \
  > ~/wendyfm-overnight/m4-gpu-only-caffeinate.log 2>&1 < /dev/null &
sleep 10
DIR="$(cat ~/wendyfm-overnight/latest-m4-gpu-only.txt)"
cat "$DIR/status.json"
tail -20 "$DIR/runner.log"
)
~~~

A successful first status reads \`preflight=PASS\`,
\`controller=GPU_ONLY_RUNNING\`, \`gpu=SUBMITTED/WAITING\` (possibly
\`PREFLIGHT\` or \`SUBMITTING_ONCE\` at the instant checked).
A later \`gpu=PASS\` plus \`controller=GPU_ONLY_DONE\` and an actual
remote evidence draft PR constitute success. A \`gpu=BLOCKED\` means
inspect the log and Kaggle site; **do not re-run blindly**. The old
controller's \`gpu=BLOCKED\` state will correctly remain unchanged:
the recovery writes to its own folder.

This path is an operator-initiated fix, not a hot patch of the old
Python process and not an automatic retry. It does not touch A1/A2,
or change the already completed M5 draft PRs.
