#!/usr/bin/env python
"""Assemble the RF-DETR fine-tune datasets in Roboflow YOLO layout (sam3env python, from ~/sam3_work).

  build_ft_ds.py --out ft/ds_real [--synth ft/src/ds_synth] [--every 5]

Sources:
  pseudo  : SAM 3.1 masks of the reconstructed plays (pairs/<clip>/track_<view>_crop/masks.json), box = mask bbox,
            class from recon/teams.json (official -> 1, A/B/unknown -> 0); every --every-th clip frame; clip frames
            with spec augmentation (blur / JPEG / low-res) at random.
  sam3    : the 554 SAM 3-labelled frames (ft/src/yolo, class 0 only).
  synth   : optional synthetic frames (already augmented).
Split by game (clip id prefix): NYIT (54201*) -> test, Susquehanna yolo frames (54746*) -> valid, all else -> train.
"""
import argparse, glob, json, os, random, shutil, sys

import cv2
import numpy as np
from pycocotools import mask as mask_utils

sys.path[:0] = __import__("glob").glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "*", ""))  # all src/multiview/* module dirs (flat sibling imports)
from build_synth_ds import augment  # noqa: E402


def game(clip):
    return "nyit" if clip.startswith("54201") else ("susq" if clip.startswith("54746") else clip[:5])


def put(out, split, stem, img, lines):
    os.makedirs(os.path.join(out, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, split, "labels"), exist_ok=True)
    if isinstance(img, str):
        shutil.copy(img, os.path.join(out, split, "images", stem + ".jpg"))
    else:
        cv2.imwrite(os.path.join(out, split, "images", stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    open(os.path.join(out, split, "labels", stem + ".txt"), "w").write("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--synth"); ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--clips", default="scratch/all55.txt")
    ap.add_argument("--det-boxes", action="store_true", help="pseudo boxes from matched RF-DETR detections (+ unmatched on-field detections)")
    a = ap.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    rng = random.Random(1)
    stats = {}

    def count(split, src, lines):
        k = f"{split}/{src}"
        s = stats.setdefault(k, {"images": 0, "player": 0, "official": 0})
        s["images"] += 1
        for l in lines:
            s["player" if l.startswith("0 ") else "official"] += 1

    # pseudo-labels (train games only: NYIT plays are excluded from training, never used as test GT)
    for c in open(a.clips).read().split():
        if game(c) == "nyit":
            continue
        labels = json.load(open(f"pairs/{c}/recon/teams.json"))["labels"]
        for v in ("wide", "ez"):
            tdir = f"pairs/{c}/track_{v}_crop"
            masks = json.load(open(f"{tdir}/masks.json"))["frames"]
            frames = sorted(os.listdir(f"{tdir}/frames"))
            a0 = json.load(open(f"{tdir}/frames.json"))["source_frame_start"]
            dets = json.load(open(f"pairs/{c}/{v}/dets_all.json"))["frames"] if a.det_boxes else None
            for fr in masks[::a.every]:
                k = fr["frame_idx"]
                img = cv2.imread(os.path.join(tdir, "frames", frames[k]))
                H, W = img.shape[:2]
                boxes = []
                for o in fr["objects"]:
                    lab = labels.get(f"{v}:{o['id']}", "unknown").rstrip("?")
                    m = mask_utils.decode(o["rle"])
                    ys, xs = np.nonzero(m)
                    if len(xs) < 30:
                        continue
                    boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 1 if lab == "official" else 0])
                if dets is not None:
                    # box geometry from the detector (SAM boxes shrink to heads / merge with neighbours); class from
                    # the SAM track's team label; on-field detections no SAM track covers are added as players
                    D = [d["box_xyxy"] for d in dets[a0 + k]["detections"]]
                    used = set()
                    for b in boxes:
                        best, bj = 0.3, -1
                        for j, d in enumerate(D):
                            if j in used:
                                continue
                            ix = max(0, min(b[2], d[2]) - max(b[0], d[0])); iy = max(0, min(b[3], d[3]) - max(b[1], d[1]))
                            inter = ix * iy; u = (b[2] - b[0]) * (b[3] - b[1]) + (d[2] - d[0]) * (d[3] - d[1]) - inter
                            v_ = inter / u if u > 0 else 0
                            if v_ > best:
                                best, bj = v_, j
                        if bj >= 0:
                            used.add(bj); b[:4] = D[bj]
                    boxes += [list(D[j]) + [0] for j in range(len(D)) if j not in used and D[j][3] - D[j][1] >= 10]
                lines = []
                for x1, y1, x2, y2, cls in boxes:
                    if y2 - y1 < 10:
                        continue
                    lines.append(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                img, _ = augment(img, rng)
                put(a.out, "train", f"pseudo_{c}_{v}_{k:04d}", img, lines)
                count("train", "pseudo", lines)
    # SAM 3 frames
    for f in sorted(glob.glob("ft/src/yolo/images/*/*.jpg")):
        stem = os.path.splitext(os.path.basename(f))[0]
        clip = stem.split("_")[0]
        split = {"nyit": "test", "susq": "valid"}.get(game(clip), "train")
        lab = f.replace("/images/", "/labels/").replace(".jpg", ".txt")
        lines = [l.strip() for l in open(lab) if l.strip()] if os.path.exists(lab) else []
        put(a.out, split, f"sam3_{stem}", f, lines)
        count(split, "sam3", lines)
    # synthetic
    if a.synth:
        for f in sorted(glob.glob(os.path.join(a.synth, "images", "*.jpg"))):
            stem = os.path.splitext(os.path.basename(f))[0]
            lab = os.path.join(a.synth, "labels", stem + ".txt")
            lines = [l.strip() for l in open(lab) if l.strip()]
            put(a.out, "train", stem, f, lines)
            count("train", "synth", lines)
    open(os.path.join(a.out, "data.yaml"), "w").write(
        "train: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames: ['player', 'official']\n")
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
