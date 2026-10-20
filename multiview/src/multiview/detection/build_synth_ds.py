#!/usr/bin/env python
"""Synthetic part of the RF-DETR fine-tune set (runs on the Mac; OpenCV + NumPy).

  build_synth_ds.py PLAY_ROOT IMAGES_DIR OUT_DIR [--every 12] [--shrink 0.77] [--holdout 1190-1199]

PLAY_ROOT/s<seed>/<frame>/players.json (finalized: mask 'bbox' in px), IMAGES_DIR/play_s<seed>_<frame>.jpg.
Labels: class 0 = on-field player (role != REF, onField, visibleRatio >= 0.3, h >= 8 px after shrink),
class 1 = official (REF); sideline people are left unlabelled (as in the field-masked real data).
Each image is shrunk by --shrink about its centre and reflect-padded back to 1920x1080 so players match real size.
Spec augmentations, one per image at random: motion blur, JPEG re-compression, low-res (down/up-sample), none.
Writes OUT_DIR/images/*.jpg, OUT_DIR/labels/*.txt (YOLO), OUT_DIR/manifest.json.
"""
import argparse, glob, json, os, random

import cv2
import numpy as np


def augment(img, rng):
    r = rng.random()
    if r < 0.25:
        k = rng.choice([7, 9, 11, 15]); ang = rng.uniform(0, 180)
        ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), ang, 1.0)
        ker = cv2.warpAffine(ker, M, (k, k)); ker /= max(ker.sum(), 1e-6)
        return cv2.filter2D(img, -1, ker), "motion_blur"
    if r < 0.5:
        q = rng.randint(25, 50)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR), f"jpeg{q}"
    if r < 0.7:
        f = rng.uniform(0.4, 0.6); h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), "lowres"
    return img, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("play_root"); ap.add_argument("images"); ap.add_argument("out")
    ap.add_argument("--every", type=int, default=12); ap.add_argument("--shrink", type=float, default=0.77)
    ap.add_argument("--holdout", default="1190-1199")
    a = ap.parse_args()
    lo, hi = map(int, a.holdout.split("-"))
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True); os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    rng = random.Random(0)
    man = {"shrink": a.shrink, "holdout_seeds": [lo, hi], "images": 0, "players": 0, "officials": 0, "aug": {}}
    for sd in sorted(glob.glob(os.path.join(a.play_root, "s*"))):
        name = os.path.basename(sd)
        if not name[1:].isdigit():
            continue
        seed = int(name[1:])
        if lo <= seed <= hi:
            continue
        frames = sorted(os.listdir(sd))
        for k, fr in enumerate(frames):
            if (k + seed) % a.every:
                continue
            pj = os.path.join(sd, fr, "players.json"); ij = os.path.join(a.images, f"play_{name}_{fr}.jpg")
            if not (os.path.exists(pj) and os.path.exists(ij)):
                continue
            img = cv2.imread(ij); H, W = img.shape[:2]
            s = a.shrink
            small = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
            px, py = (W - small.shape[1]) // 2, (H - small.shape[0]) // 2
            canvas = cv2.copyMakeBorder(small, py, H - small.shape[0] - py, px, W - small.shape[1] - px, cv2.BORDER_REFLECT)
            lines = []
            for p in json.load(open(pj))["players"]:
                bb = p.get("bbox")
                if not bb:
                    continue
                if p["role"] == "REF":
                    cls = 1
                elif p.get("onField") and p.get("visibleRatio", 0) >= 0.3:
                    cls = 0
                else:
                    continue
                x, y, w, h = bb
                x, y, w, h = x * s + px, y * s + py, w * s, h * s
                if h < 8:
                    continue
                lines.append(f"{cls} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
                man["players" if cls == 0 else "officials"] += 1
            out_img, aug = augment(canvas, rng)
            man["aug"][aug] = man["aug"].get(aug, 0) + 1
            stem = f"synth_{name}_{fr}"
            cv2.imwrite(os.path.join(a.out, "images", stem + ".jpg"), out_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            open(os.path.join(a.out, "labels", stem + ".txt"), "w").write("\n".join(lines) + ("\n" if lines else ""))
            man["images"] += 1
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(json.dumps(man))


if __name__ == "__main__":
    main()
