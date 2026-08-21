"""Block until the watchdog exits (it exits only when training genuinely completes)."""
import re, sys, time
import psutil

def watchdog_alive():
    for p in psutil.process_iter(['pid','cmdline','memory_info']):
        try:
            cl = ' '.join(map(str, p.info['cmdline'] or []))
            mi = p.info['memory_info']
            if re.search(r'[\/ ]watchdog\.py(\s|$)', cl) and mi and mi.rss > 15e6:
                return p.info['pid']
        except Exception:
            continue
    return None

while True:
    pid = watchdog_alive()
    if pid is None:
        print("WATCHDOG EXITED")
        sys.exit(0)
    time.sleep(60)
