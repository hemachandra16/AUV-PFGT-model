"""Read-only process census: how many real trainers and watchdogs are alive."""
import os, re, sys
import psutil
me = os.getpid()
trainers, watchdogs = [], []
for p in psutil.process_iter(['pid','cmdline','memory_info']):
    if p.info['pid'] == me: continue
    cl = ' '.join(map(str, p.info['cmdline'] or []))
    rss = p.info['memory_info'].rss if p.info['memory_info'] else 0
    if re.search(r'[\/ ]watchdog\.py(\s|$)', cl) and rss > 15e6:
        watchdogs.append((p.info['pid'], rss/1e6))
    if re.search(r'[\/ ]train\.py(\s|$)', cl) and '--config' in cl and rss > 200e6:
        trainers.append((p.info['pid'], rss/1e6))
print("real trainers :", [(pid, f"{m:.0f}MB") for pid,m in trainers])
print("real watchdogs:", [(pid, f"{m:.0f}MB") for pid,m in watchdogs])
if len(trainers) != 1: print("WARNING: expected exactly 1 trainer")
if len(watchdogs) != 1: print("WARNING: expected exactly 1 watchdog")
sys.exit(0 if (len(trainers)==1 and len(watchdogs)==1) else 1)
