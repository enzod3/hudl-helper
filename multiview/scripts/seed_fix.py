"""SAM re-run with field-filtered seeds:  seed_fix.py PAIR_DIR VIEW [--suffix _seed] [--max-objects 32] [--reprompt K]
Frames / window as track_<view>_crop. Seeds: detections of the first window frame whose foot point (box bottom-centre
through H) lies on the field (x -0.5..120.5, y -0.5..53.8 yd), highest score first, up to --max-objects.
Writes track_<view><suffix>/ (segment.py layout). Runs segment.py under the GPU lock."""
import argparse, json, os, subprocess
import numpy as np
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PY = os.environ.get("MULTIVIEW_SAM_PY", os.path.expanduser("~/sam3env/bin/python"))
ap = argparse.ArgumentParser(); ap.add_argument("pair"); ap.add_argument("view")
ap.add_argument("--suffix", default="_seed"); ap.add_argument("--max-objects", type=int, default=32)
a = ap.parse_args()
P, v = a.pair, a.view
src = os.path.join(P, f"track_{v}_crop"); dst = os.path.join(P, f"track_{v}{a.suffix}")
meta = json.load(open(os.path.join(src, "frames.json")))
os.makedirs(os.path.join(dst, "frames"), exist_ok=True)
for f in sorted(os.listdir(os.path.join(src, "frames"))):
    d = os.path.join(dst, "frames", f)
    if not os.path.exists(d):
        os.symlink(os.path.realpath(os.path.join(src, "frames", f)), d)
json.dump(meta, open(os.path.join(dst, "frames.json"), "w"), indent=1)
cal = json.load(open(os.path.join(P, v, "calib.json")))
dets = json.load(open(os.path.join(P, v, "dets_all.json")))
st = meta["source_frame_start"]
Hi = np.linalg.inv(np.array(cal["H"][st]))
keep, off = [], 0
for d in dets["frames"][st]["detections"]:
    x1, y1, x2, y2 = d["box_xyxy"]; q = Hi @ [(x1 + x2) / 2, y2, 1.0]; x, y = q[:2] / q[2]
    if -0.5 <= x <= 120.5 and -0.5 <= y <= 53.8:
        keep.append(d)
    else:
        off += 1
d0 = dict(dets); f0 = dict(dets["frames"][st]); f0["detections"] = keep; d0["frames"] = [f0]
json.dump(d0, open(os.path.join(dst, "dets0.json"), "w"))
r = subprocess.run(["flock", os.path.expanduser("~/sam3_work/logs/gpu.lock"), os.path.expanduser("~/sam3env/bin/python"),
                    "pipeline/segment.py", "track", "--out-dir", dst, "--init-dets", os.path.join(dst, "dets0.json"),
                    "--min-score", "0.3", "--max-objects", str(a.max_objects), "--dtype", "fp16"], capture_output=True, text=True)
print(json.dumps({"pair": P, "view": v, "seeds_on_field": len(keep), "seeds_off_field_removed": off, "ok": r.returncode == 0,
                  "err": r.stderr[-300:] if r.returncode else None}))
