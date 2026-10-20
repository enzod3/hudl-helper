#!/usr/bin/env python
"""Contact sheets of candidate identity switches (sam3env python):
  switch_sheets.py PAIR_DIR [--suffix _crop] [--summary recon/recon_summary_pass1.json]
One row per candidate (view, sam_id, tau): crops around that SAM id's mask at clip frames k-6, k-2, k+2, k+6 with the
mask outlined, labelled. Writes PAIR_DIR/recon/switches_<view>.jpg (several rows each).
"""
import argparse, json, os

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair"); ap.add_argument("--suffix", default="_crop")
    ap.add_argument("--summary", default="recon/recon_summary_pass1.json")
    a = ap.parse_args()
    summ = json.load(open(os.path.join(a.pair, a.summary)))
    sync = json.load(open(os.path.join(a.pair, "sync.json")))
    cands = [w for w in summ["candidate_id_switches"] if "sam_id" in w]
    for v in ("wide", "ez"):
        cv_ = [w for w in cands if w["view"] == v and w["sam_id"] < 1000]
        if not cv_:
            continue
        tdir = os.path.join(a.pair, f"track_{v}{a.suffix}")
        meta = json.load(open(os.path.join(tdir, "frames.json")))
        masks = json.load(open(os.path.join(tdir, "masks.json")))["frames"]
        frames = sorted(os.listdir(os.path.join(tdir, "frames")))
        pts = json.load(open(os.path.join(a.pair, v, "frames_pts.json")))["pts_s"]
        rel = np.array(pts) - sync[v]["snap_pts"]
        rows = []
        last_c = None
        for w in cv_:
            sf = int(np.argmin(np.abs(rel - w["tau"])))
            k = sf - meta["source_frame_start"]
            tiles = []
            for dk in (-6, -2, 2, 6):
                kk = k + dk
                tile = np.zeros((220, 220, 3), np.uint8)
                if 0 <= kk < len(frames):
                    obj = next((o for o in masks[kk]["objects"] if o["id"] == w["sam_id"]), None)
                    img = cv2.imread(os.path.join(tdir, "frames", frames[kk]))
                    if obj is not None:
                        x1, y1, x2, y2 = obj["box_xyxy"]
                        m = mask_utils.decode(obj["rle"]).astype(np.uint8)
                        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    else:
                        cx, cy = None, None
                    if cx is None and last_c is not None:
                        cx, cy = last_c
                    if cx is not None:
                        last_c = (cx, cy)
                        s = 150
                        xa, ya = int(max(0, cx - s)), int(max(0, cy - s))
                        crop = img[ya:ya + 2 * s, xa:xa + 2 * s]
                        tile = cv2.resize(crop, (220, 220))
                    lab = f"{v} id{w['sam_id']} f{kk} {'(absent)' if obj is None else ''}"
                    cv2.putText(tile, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                tiles.append(tile)
            row = np.hstack(tiles)
            cv2.putText(row, f"tau {w['tau']:+.2f} jump {w['jump_yd']} yd", (4, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            rows.append(row)
        cv2.imwrite(os.path.join(a.pair, "recon", f"switches_{v}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(v, len(rows))


if __name__ == "__main__":
    main()
