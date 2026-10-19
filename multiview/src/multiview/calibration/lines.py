#!/usr/bin/env python
"""Long paint-line candidates on a keyframe, numbered for labelling (sam3env python).

  lines.py FRAME.jpg DETS_ALL.json FRAME_INDEX OUT.jpg OUT.json [--view wide|ez]

OUT.json: [{"id", "rho", "theta", "p1", "p2"}] lines in (rho, theta) image form (x cos t + y sin t = rho),
p1/p2 = endpoints clipped to the image. OUT.jpg draws each with its id.
"""
import argparse, json, os, sys

import cv2
import numpy as np

sys.path[:0] = __import__("glob").glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "*", ""))  # all src/multiview/* module dirs (flat sibling imports)
from fieldmask import field_mask  # noqa: E402
from fieldmodel import paint_mask  # noqa: E402


def clip_line(rho, th, w, h):
    c, s = np.cos(th), np.sin(th)
    pts = []
    for x in (0, w - 1):
        if abs(s) > 1e-6:
            y = (rho - x * c) / s
            if 0 <= y <= h - 1:
                pts.append((x, y))
    for y in (0, h - 1):
        if abs(c) > 1e-6:
            x = (rho - y * s) / c
            if 0 <= x <= w - 1:
                pts.append((x, y))
    if len(pts) < 2:
        return None
    pts = sorted(set((round(a, 1), round(b, 1)) for a, b in pts))
    return pts[0], pts[-1]


def find_lines(paint, max_lines=40, thresh=180):
    h, w = paint.shape
    raw = cv2.HoughLines(paint, 1, np.pi / 720, thresh)
    if raw is None:
        return []
    raw = raw[:, 0, :]
    keep = []
    for rho, th in raw:  # sorted by votes; suppress near-duplicates
        if rho < 0:
            rho, th = -rho, th - np.pi
        dup = False
        for r2, t2 in keep:
            dth = abs(((th - t2) + np.pi / 2) % np.pi - np.pi / 2)
            if dth < np.radians(2.0) and abs(rho - r2) < 18:
                dup = True
                break
        if not dup:
            keep.append((float(rho), float(th)))
        if len(keep) >= max_lines:
            break
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame"); ap.add_argument("dets"); ap.add_argument("index", type=int)
    ap.add_argument("out_jpg"); ap.add_argument("out_json")
    ap.add_argument("--view", default="wide")
    a = ap.parse_args()
    img = cv2.imread(a.frame)
    d = json.load(open(a.dets))["frames"][a.index]
    fm, _ = field_mask(img, view=a.view, erode_px=0)
    boxes = [x["box_xyxy"] for x in d["detections"] + d.get("dropped", [])]
    paint = paint_mask(img, cv2.dilate(fm, np.ones((41, 41), np.uint8)), boxes)
    h, w = paint.shape
    lines = find_lines(paint)
    out = img.copy()
    out[paint > 0] = (0, 0, 255)
    recs = []
    for i, (rho, th) in enumerate(lines):
        e = clip_line(rho, th, w, h)
        if e is None:
            continue
        p1, p2 = e
        col = tuple(int(c) for c in np.random.default_rng(i).integers(80, 255, 3))
        cv2.line(out, tuple(map(int, p1)), tuple(map(int, p2)), col, 2)
        for t in (0.15, 0.5, 0.85):
            q = (int(p1[0] + t * (p2[0] - p1[0])), int(p1[1] + t * (p2[1] - p1[1])))
            cv2.putText(out, str(i), q, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(out, str(i), q, cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        recs.append({"id": i, "rho": rho, "theta": th, "p1": p1, "p2": p2})
    cv2.imwrite(a.out_jpg, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(a.out_jpg.replace(".jpg", "_paint.png"), paint)
    json.dump(recs, open(a.out_json, "w"), indent=1)
    print(len(recs), "lines")


if __name__ == "__main__":
    main()
