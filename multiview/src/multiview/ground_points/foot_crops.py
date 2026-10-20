#!/usr/bin/env python
"""Crops for the ground-contact head (OpenCV + NumPy).

Crop convention (shared by training and inference): box (x1,y1,x2,y2) expanded by 10% of its width left/right,
15% of its height on top and 30% below, resized to 64x128 (w x h). The foot target is (u, v) in crop-normalised
coordinates (0..1 across the expanded box).

  foot_crops.py synth PLAY_ROOT IMAGES_DIR OUT.npz [--suffix ''|_ez] [--every 3] [--holdout 1190-1199]
"""
import argparse, glob, json, os

import cv2
import numpy as np

CW, CH = 64, 128


def crop_box(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return x1 - 0.10 * w, y1 - 0.15 * h, x2 + 0.10 * w, y2 + 0.30 * h


def crop(img, box):
    X1, Y1, X2, Y2 = crop_box(box)
    src = np.float32([[X1, Y1], [X2, Y1], [X1, Y2]])
    dst = np.float32([[0, 0], [CW, 0], [0, CH]])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, (CW, CH), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def to_crop_norm(box, pt):
    X1, Y1, X2, Y2 = crop_box(box)
    return (pt[0] - X1) / (X2 - X1), (pt[1] - Y1) / (Y2 - Y1)


def from_crop_norm(box, uv):
    X1, Y1, X2, Y2 = crop_box(box)
    return X1 + uv[0] * (X2 - X1), Y1 + uv[1] * (Y2 - Y1)


def synth(a):
    lo, hi = map(int, a.holdout.split("-"))
    X, Y, M = [], [], []
    for sd in sorted(glob.glob(os.path.join(a.root, "s*" + a.suffix))):
        name = os.path.basename(sd)
        core = name[1:len(name) - len(a.suffix)] if a.suffix else name[1:]
        if not core.isdigit():
            continue
        seed = int(core)
        hold = lo <= seed <= hi
        if hold != a.holdout_only:
            continue
        for k, fr in enumerate(sorted(os.listdir(sd))):
            if k % a.every:
                continue
            pj = os.path.join(sd, fr, "players.json")
            ij = os.path.join(a.images, f"play_{name}_{fr}.jpg")
            if not (os.path.exists(pj) and os.path.exists(ij)):
                continue
            img = cv2.imread(ij)
            sx, sy = img.shape[1] / 1920.0, img.shape[0] / 1080.0
            for p in json.load(open(pj))["players"]:
                bb, ft = p.get("bbox"), p.get("footPointImage")
                if not bb or not ft or not p.get("onField") or p.get("visibleRatio", 0) < 0.3 or bb[3] < 16:
                    continue
                box = [bb[0] * sx, bb[1] * sy, (bb[0] + bb[2]) * sx, (bb[1] + bb[3]) * sy]
                foot = (ft[0] * sx, ft[1] * sy)
                u, v = to_crop_norm(box, foot)
                if not (-0.2 < u < 1.2 and -0.2 < v < 1.4):
                    continue
                X.append(crop(img, box)); Y.append((u, v))
                M.append((seed, int(fr), p["trackId"], bb[3] * sy, float(p.get("visibleRatio", 1)), int(p.get("truncated", False))))
    np.savez_compressed(a.out, X=np.array(X, np.uint8), Y=np.array(Y, np.float32), M=np.array(M, np.float32))
    print(a.out, len(X))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("root"); s.add_argument("images"); s.add_argument("out")
    s.add_argument("--suffix", default=""); s.add_argument("--every", type=int, default=3)
    s.add_argument("--holdout", default="1190-1199"); s.add_argument("--holdout-only", action="store_true")
    a = ap.parse_args()
    {"synth": synth}[a.cmd](a)
