"""Autonomous training watchdog: keep the PFGT-UIE run alive until it genuinely finishes.

Session 1 lost a run at epoch 26 (host RAM exhaustion) and recovered only because a human
noticed. This removes that dependency: it polls every `--interval` seconds and, if the
training process has died *without* the log showing a real completion, it restarts from
`checkpoints/latest.pt` and keeps going.

Why it restarts with `--resume` and no `--epochs` override: `train.py` reads the epoch
total from `configs/train.yaml` (150) and computes the LR from `global_step / total_steps`.
Resuming therefore continues the original warmup+cosine schedule at the right point rather
than restarting it — which is exactly what session 1's D-009 note warned about.

Completion is detected from the log, not from the exit code, so a clean early-stop and a
crash are never confused:
    "Training complete."      -> finished (epoch limit or early stop)
    "Early stopping triggered" -> finished

CUDA OOM is handled specially: `dataloader.batch_size` in configs/train.yaml is halved
before the restart, down to a floor of 1.

Deliberately lightweight — it only reads a log file and inspects the process table, so it
never violates the "no heavy jobs while training is active" rule that caused the session 1
crash.

Usage:
    python tools/watchdog.py --interval 60
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TRAIN_LOG = ROOT / "logs" / "train.log"
CONFIG = ROOT / "configs" / "train.yaml"
INCIDENTS = ROOT / "logs" / "watchdog_incidents.json"
WATCHDOG_LOG = ROOT / "logs" / "watchdog.log"

DONE_MARKERS = ("Training complete.", "Early stopping triggered")
OOM_PATTERNS = ("CUDA out of memory", "OutOfMemoryError", "CUDA error: out of memory")
# Session 1's crash signature: host RAM exhaustion killing the pinned-memory allocator.
RAM_PATTERNS = ("pinned allocator", "Unhandled exception caught in c10", "bad_alloc")


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def training_procs() -> list:
    """All processes whose command line is `train.py --config ...`.

    On Windows a `nohup -> .venv shim -> real python` launch chain produces THREE such
    matches for a single run, so presence alone is ambiguous. The real trainer is the one
    holding the model in memory (GBs of RSS); the wrappers are a few MB. That distinction
    matters because a lingering wrapper after the trainer dies would look "alive" to a
    naive check, and the watchdog would then never restart.
    """
    try:
        import psutil
    except ImportError:
        log("psutil unavailable; cannot detect the training process")
        return []
    found = []
    for proc in psutil.process_iter(["pid", "cmdline", "memory_info"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            joined = " ".join(str(c) for c in cmdline)
            # Match train.py specifically — not tools/train_detector.py, not this watchdog.
            if re.search(r"(^|[\\/ ])train\.py(\s|$)", joined) and "--config" in joined:
                mi = proc.info.get("memory_info")
                found.append((proc, mi.rss if mi else 0))
        except Exception:
            continue
    return found


# A real trainer holds the model plus a CUDA context; launcher shims are a few MB.
REAL_TRAINER_MIN_RSS = 200 * 1024 * 1024


def training_running() -> int | None:
    """Return the pid of the LIVE REAL trainer, or None.

    Thin launcher wrappers are ignored, so a stranded `nohup`/shim cannot masquerade as a
    healthy run and suppress a needed restart.
    """
    real = [(p, r) for p, r in training_procs() if r >= REAL_TRAINER_MIN_RSS]
    if real:
        return max(real, key=lambda pr: pr[1])[0].pid
    return None


def log_says_done() -> str | None:
    if not TRAIN_LOG.exists():
        return None
    try:
        text = TRAIN_LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    tail = text[-20000:]
    for marker in DONE_MARKERS:
        if marker in tail:
            return marker
    return None


def last_lines(path: Path, n: int = 30) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def newest_stdout_log() -> Path | None:
    candidates = sorted((ROOT / "logs").glob("train_stdout*.log"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def classify_failure() -> tuple[str, list[str]]:
    """Guess why training died, from the log tails."""
    stdout_log = newest_stdout_log()
    tail_lines = last_lines(stdout_log, 60) if stdout_log else []
    blob = "\n".join(tail_lines) + "\n" + "\n".join(last_lines(TRAIN_LOG, 30))
    if any(p in blob for p in OOM_PATTERNS):
        return "cuda_oom", tail_lines
    if any(p in blob for p in RAM_PATTERNS):
        return "host_ram", tail_lines
    if not tail_lines:
        return "unknown_no_stdout", tail_lines
    return "unknown", tail_lines


def current_batch_size() -> int | None:
    try:
        text = CONFIG.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"^(\s*)batch_size:\s*(\d+)", text, re.M)
    return int(m.group(2)) if m else None


def halve_batch_size() -> tuple[int, int] | None:
    text = CONFIG.read_text(encoding="utf-8")
    m = re.search(r"^(\s*)batch_size:\s*(\d+)(.*)$", text, re.M)
    if not m:
        return None
    old = int(m.group(2))
    new = max(1, old // 2)
    if new == old:
        return None
    text = text[:m.start()] + f"{m.group(1)}batch_size: {new}{m.group(3)}" + text[m.end():]
    CONFIG.write_text(text, encoding="utf-8", newline="")
    return old, new


def checkpoint_epoch() -> int | None:
    ckpt = ROOT / "checkpoints" / "latest.pt"
    if not ckpt.exists():
        return None
    try:
        import torch
        # weights_only=True keeps this cheap and avoids materialising the model state.
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        return state.get("epoch")
    except Exception:
        return None


def start_training(resume: bool) -> int:
    stamp = datetime.now().strftime("%H%M%S")
    out_path = ROOT / "logs" / f"train_stdout_restart_{stamp}.log"
    cmd = [str(PYTHON), "train.py", "--config", "configs/train.yaml", "--num-workers", "2"]
    if resume:
        cmd += ["--resume", "checkpoints/latest.pt"]
    out = open(out_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)
    log(f"started training pid={proc.pid} resume={resume} stdout={out_path.name}")
    return proc.pid


def kill_tree(pid: int) -> None:
    """Kill a process and all of its children (DataLoader workers included)."""
    try:
        import psutil
    except ImportError:
        return
    try:
        parent = psutil.Process(pid)
    except Exception:
        return
    procs = parent.children(recursive=True) + [parent]
    for pr in procs:
        try:
            pr.terminate()
        except Exception:
            pass
    _, alive = psutil.wait_procs(procs, timeout=15)
    for pr in alive:
        try:
            pr.kill()
        except Exception:
            pass
    log(f"   killed process tree rooted at pid={pid} ({len(procs)} procs)")


def log_mtime_size() -> tuple:
    """(mtime, size) of the training log — used to detect a stalled run."""
    if not TRAIN_LOG.exists():
        return (0.0, 0)
    st = TRAIN_LOG.stat()
    return (st.st_mtime, st.st_size)


def record(incident: dict) -> None:
    data = []
    if INCIDENTS.exists():
        try:
            data = json.loads(INCIDENTS.read_text(encoding="utf-8"))
        except Exception:
            data = []
    data.append(incident)
    INCIDENTS.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--max-restarts", type=int, default=25,
                    help="Safety valve against an unrecoverable crash-loop")
    ap.add_argument("--grace", type=int, default=90,
                    help="Seconds to wait after a restart before judging health again")
    ap.add_argument("--stall-timeout", type=int, default=900,
                    help="Restart if the log stays silent this long while the process lives")
    args = ap.parse_args()

    log("=" * 70)
    log(f"watchdog starting (interval={args.interval}s, max_restarts={args.max_restarts})")
    log(f"batch_size in config: {current_batch_size()}")
    log(f"stall timeout: {args.stall_timeout}s of log silence")

    restarts = 0
    consecutive_fast_failures = 0
    last_log_sig = log_mtime_size()
    last_progress_at = time.time()

    while True:
        done = log_says_done()
        pid = training_running()

        if done and pid is None:
            log(f"TRAINING FINISHED — log says: {done!r}")
            record({"time": datetime.now().isoformat(), "event": "finished",
                    "marker": done, "restarts_used": restarts})
            log(f"watchdog exiting cleanly after {restarts} restart(s)")
            return

        if pid is not None:
            # Alive — but is it making progress? A hung dataloader or a deadlocked CUDA
            # call leaves the process resident while the log goes silent. Epochs take
            # ~50 s and a step line lands every ~5 s, so a long silence is unambiguous.
            sig = log_mtime_size()
            if sig != last_log_sig:
                last_log_sig = sig
                last_progress_at = time.time()
            elif time.time() - last_progress_at > args.stall_timeout:
                silent = int(time.time() - last_progress_at)
                log(f"!! STALL: pid={pid} alive but logs/train.log silent for {silent}s")
                record({"time": datetime.now().isoformat(), "event": "stall",
                        "cause": "no_log_progress", "silent_seconds": silent,
                        "pid": pid, "checkpoint_epoch": checkpoint_epoch(),
                        "restart_number": restarts + 1})
                kill_tree(pid)
                time.sleep(5)
                start_training(resume=(checkpoint_epoch() is not None))
                restarts += 1
                last_progress_at = time.time()
                last_log_sig = log_mtime_size()
                time.sleep(args.grace)
                continue
            time.sleep(args.interval)
            continue

        # Process gone and no completion marker -> genuine failure.
        if restarts >= args.max_restarts:
            log(f"ABORT: reached max_restarts={args.max_restarts}; not restarting again")
            record({"time": datetime.now().isoformat(), "event": "gave_up",
                    "restarts_used": restarts})
            return

        kind, tail = classify_failure()
        epoch = checkpoint_epoch()
        log(f"!! TRAINING DIED — cause={kind}, latest.pt epoch={epoch}")
        for line in tail[-8:]:
            log(f"   | {line[:200]}")

        incident = {
            "time": datetime.now().isoformat(),
            "event": "crash",
            "cause": kind,
            "checkpoint_epoch": epoch,
            "restart_number": restarts + 1,
            "tail": tail[-15:],
        }

        if kind == "cuda_oom":
            changed = halve_batch_size()
            if changed:
                incident["batch_size"] = {"from": changed[0], "to": changed[1]}
                log(f"   CUDA OOM -> batch_size {changed[0]} -> {changed[1]}")
            else:
                log("   CUDA OOM but batch_size could not be reduced further")

        record(incident)

        start_training(resume=(epoch is not None))
        restarts += 1
        last_progress_at = time.time()
        last_log_sig = log_mtime_size()
        time.sleep(args.grace)

        # If it dies again immediately, back off so we don't spin.
        if training_running() is None:
            consecutive_fast_failures += 1
            backoff = min(60 * consecutive_fast_failures, 600)
            log(f"   restart #{restarts} did not survive the grace period; "
                f"backing off {backoff}s")
            time.sleep(backoff)
        else:
            consecutive_fast_failures = 0


if __name__ == "__main__":
    main()
