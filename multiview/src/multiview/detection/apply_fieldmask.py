#!/usr/bin/env python
"""Apply the field mask to raw detections on the CPU (sam3env or rfdetr_env python), outside the GPU lock.

  apply_fieldmask.py RAW_DETS.json OUT_DETS.json --view wide|ez [--workers 8]

RAW_DETS.json is detect.py output without --field-mask; OUT has the same layout as detect.py --field-mask
(detections kept, 'dropped' with reasons, 'field_mask' info, 'num_before_field_mask').
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import cv2

sys.path[:0] = __import__("glob").glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "*", ""))  # all src/multiview/* module dirs (flat sibling imports)
from fieldmask import field_mask, filter_detections  # noqa: E402

VIEW = "wide"


def one(fr):
    bgr = cv2.imread(fr["path"])
    mask, info = field_mask(bgr, view=VIEW)
    green = info.pop("green")
    kept, dropped = filter_detections(fr["detections"], mask, green=green)
    return {**fr, "num_before_field_mask": len(fr["detections"]), "detections": kept, "dropped": dropped, "field_mask": info}


def init(view):
    global VIEW
    VIEW = view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("out")
    ap.add_argument("--view", default="wide")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    d = json.load(open(a.raw))
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.view,)) as ex:
        d["frames"] = list(ex.map(one, d["frames"], chunksize=8))
    d["field_mask"] = True
    json.dump(d, open(a.out, "w"))
    print(json.dumps({"frames": len(d["frames"]), "kept_median": sorted(len(f["detections"]) for f in d["frames"])[len(d["frames"]) // 2]}))


if __name__ == "__main__":
    main()
