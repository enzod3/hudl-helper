#!/usr/bin/env python
"""Run run_pair.py over a dots file, N plays at a time (sam3env python, from ~/sam3_work):
  batch.py DOTS.json [--jobs 6] [--pre 2.5]
Skips plays whose run_pair.json shows every step (through second_pass) ok."""
import argparse, json, os, subprocess, time
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("dots"); ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--pre", default="2.5")
a = ap.parse_args()
PY = os.environ.get("MULTIVIEW_SAM_PY", os.path.expanduser("~/sam3env/bin/python"))


def done(c):
    f = f"pairs/{c}/run_pair.json"
    if not os.path.exists(f):
        return False
    st = json.load(open(f))["steps"]
    return "recon" in st and all(v["ok"] for v in st.values())


todo = [p for p in json.load(open(a.dots)) if not done(p["clip"])]
print("todo", len(todo), flush=True)
running = []
while todo or running:
    running = [(c, pr) for c, pr in running if pr.poll() is None]
    while todo and len(running) < a.jobs:
        p = todo.pop(0)
        log = open(f"logs/run_pair_{p['clip']}.log", "a")
        pr = subprocess.Popen([PY, os.path.join(_REPO, "scripts/run_pair.py"), p["clip"], p["wide"], p["ez"], "--pre", a.pre,
                               "--wide-snap-fallback", str(p["snap_t"])],
                              stdout=log, stderr=subprocess.STDOUT)
        running.append((p["clip"], pr))
        print("start", p["clip"], time.strftime("%H:%M:%S"), flush=True)
    time.sleep(10)
print("BATCH DONE", time.strftime("%H:%M:%S"), flush=True)
