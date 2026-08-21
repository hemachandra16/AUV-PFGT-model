"""Block until the training run genuinely finishes.

Exits ONLY when the training process is gone AND the log carries a real completion marker,
so a crashed run cannot be mistaken for a finished one.
"""
import re, sys, time
import psutil

DONE = ("Training complete.", "Early stopping triggered")
LOG = "logs/train.log"

def alive(pattern, min_rss):
    for p in psutil.process_iter(['pid', 'cmdline', 'memory_info']):
        try:
            cl = ' '.join(map(str, p.info['cmdline'] or []))
            mi = p.info['memory_info']
            if re.search(pattern, cl) and mi and mi.rss > min_rss:
                return p.info['pid']
        except Exception:
            continue
    return None

def log_done():
    try:
        return any(m in open(LOG, encoding='utf-8', errors='replace').read()[-20000:] for m in DONE)
    except OSError:
        return False

while True:
    trainer = alive(r'[\/ ]train\.py(\s|$)', 200e6)
    wd = alive(r'[\/ ]watchdog\.py(\s|$)', 15e6)
    if trainer is None and log_done():
        print("TRAINING FINISHED (log marker present, process gone)")
        sys.exit(0)
    if trainer is None and wd is None:
        print("WARNING: trainer AND watchdog both gone with no completion marker")
        sys.exit(1)
    time.sleep(60)
