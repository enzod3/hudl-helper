#!/usr/bin/env python
"""Real ground-contact samples from Enzo's dots (sam3env python, from ~/sam3_work).

  foot_real.py DOTS.json OUT.npz

Per dot (foot point on the Wide frame at frame_t): the RF-DETR detection on that Wide frame (dets_all detections +
dropped) whose bottom-centre is nearest the dot, accepted if the dot lies within the box widened by 30% and within
35% of the box height of its bottom edge. Stores the crop (foot_crops convention), the dot in crop coordinates,
the box, the frame's H (field->image), the dot in yards, and the SAM mask foot point at that frame when a tracked
mask overlaps the box (IoU >= 0.4; bottom-centre of the mask's lowest 10%).
"""
import argparse, json, os, sys

import cv2
import numpy as np
from pycocotools import mask as mask_utils

sys.path[:0] = __import__("glob").glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "*", ""))  # all src/multiview/* module dirs (flat sibling imports)
from foot_crops import crop, to_crop_norm  # noqa: E402


def to_field(H, u, v):
    p = np.linalg.inv(H) @ np.array([u, v, 1.0])
    return p[:2] / p[2]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dots"); ap.add_argument("out")
    a = ap.parse_args()
    X, Y, META = [], [], []
    stats = {"dots": 0, "matched": 0, "with_mask": 0}
    for p in json.load(open(a.dots)):
        c = p["clip"]; P = f"pairs/{c}"
        try:
            cal = json.load(open(f"{P}/wide/calib.json")); dets = json.load(open(f"{P}/wide/dets_all.json"))["frames"]
        except FileNotFoundError:
            continue
        pts = np.array(cal["pts_s"]); sf = int(np.argmin(np.abs(pts - p["frame_t"])))
        H = np.array(cal["H"][sf])
        frames = sorted(os.listdir(f"{P}/wide/frames"))
        img = cv2.imread(f"{P}/wide/frames/{frames[sf]}")
        D = [d["box_xyxy"] for d in dets[sf]["detections"] + dets[sf].get("dropped", [])]
        # SAM masks at this frame, if the tracking window covers it
        masks = []
        tdir = f"{P}/track_wide_crop"
        if os.path.exists(f"{tdir}/masks.json"):
            a0 = json.load(open(f"{tdir}/frames.json"))["source_frame_start"]
            mf = json.load(open(f"{tdir}/masks.json"))["frames"]
            k = sf - a0
            if 0 <= k < len(mf):
                masks = mf[k]["objects"]
        for m in p["marks"]:
            stats["dots"] += 1
            dx, dy = m["x"] * 1920, m["y"] * 1080
            best, bd = None, 1e9
            for b in D:
                w, h = b[2] - b[0], b[3] - b[1]
                if not (b[0] - 0.3 * w <= dx <= b[2] + 0.3 * w and abs(dy - b[3]) <= 0.35 * h):
                    continue
                d = np.hypot((b[0] + b[2]) / 2 - dx, b[3] - dy)
                if d < bd:
                    best, bd = b, d
            if best is None:
                continue
            stats["matched"] += 1
            mfoot = None
            for o in masks:
                if iou(o["box_xyxy"], best) >= 0.4:
                    mm = mask_utils.decode(o["rle"]); ys, xs = np.nonzero(mm)
                    if len(ys) >= 15:
                        bot, top = ys.max(), ys.min(); sel = ys >= bot - 0.1 * (bot - top + 1)
                        mfoot = [float(xs[sel].mean()), float(bot)]
                        stats["with_mask"] += 1
                    break
            u, v = to_crop_norm(best, (dx, dy))
            X.append(crop(img, best)); Y.append((u, v))
            META.append({"clip": c, "role": m["role"], "box": [float(x) for x in best], "dot_px": [dx, dy],
                         "dot_yd": to_field(H, dx, dy).tolist(), "H": H.tolist(), "mask_foot_px": mfoot})
    np.savez_compressed(a.out, X=np.array(X, np.uint8), Y=np.array(Y, np.float32))
    json.dump(META, open(a.out.replace(".npz", "_meta.json"), "w"))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
