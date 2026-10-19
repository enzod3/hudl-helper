#!/usr/bin/env python
"""RF-DETR detection worker (run with ~/rfdetr_env/bin/python).

Usage:
  detect.py --model large|medium --out dets.json [--tiles 2] [--threshold 0.3]
            [--classes person] [--viz-dir DIR] FRAME.jpg [FRAME.jpg ...]
  detect.py ... --list frames.txt        (one frame path per line)

Output JSON:
  {"model", "rfdetr_version", "tiles", "threshold", "classes",
   "frames": [{"path", "width", "height", "seconds",
               "detections": [{"box_xyxy": [x1,y1,x2,y2], "score", "class_id", "class_name"}]}],
   "stats": {"mean_seconds_per_frame", "peak_gpu_mem_allocated_mb", "peak_gpu_mem_reserved_mb"}}
Boxes are in source-image pixel coordinates. With --tiles 2 the frame is split into
2x2 tiles overlapping by --overlap (fraction of tile size); tile detections are shifted
back to frame coordinates and merged with class-wise NMS.
"""
import argparse, json, os, sys, time

# keep RF-DETR weight cache inside ~/sam3_work (rfdetr defaults to ~/.roboflow/models)
os.environ.setdefault("RF_HOME", os.path.expanduser("~/sam3_work/rfdetr_weights"))

import cv2
import numpy as np
import rfdetr
import torch
import torchvision
from rfdetr import RFDETRLarge, RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

sys.path[:0] = __import__("glob").glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "*", ""))  # all src/multiview/* module dirs (flat sibling imports)
from fieldmask import field_mask, filter_detections  # noqa: E402

MODELS = {"large": RFDETRLarge, "medium": RFDETRMedium}


def tile_windows(w, h, n, overlap):
    """n x n windows covering (w, h); each window overlaps its neighbour by `overlap` of its size."""
    tw = int(np.ceil(w / (n - (n - 1) * overlap)))
    th = int(np.ceil(h / (n - (n - 1) * overlap)))
    xs = np.linspace(0, w - tw, n).astype(int)
    ys = np.linspace(0, h - th, n).astype(int)
    return [(x, y, tw, th) for y in ys for x in xs]


def run_model(model, rgb, threshold):
    det = model.predict(rgb, threshold=threshold, include_source_image=False)
    return det.xyxy.astype(np.float32), det.confidence.astype(np.float32), det.class_id.astype(np.int64)


def detect_frame(model, rgb, threshold, tiles, overlap, nms_iou):
    h, w = rgb.shape[:2]
    if tiles <= 1:
        return run_model(model, rgb, threshold)
    boxes, scores, cls = [], [], []
    for x, y, tw, th in tile_windows(w, h, tiles, overlap):
        b, s, c = run_model(model, np.ascontiguousarray(rgb[y:y + th, x:x + tw]), threshold)
        b[:, [0, 2]] += x
        b[:, [1, 3]] += y
        boxes.append(b); scores.append(s); cls.append(c)
    b, s, c = np.concatenate(boxes), np.concatenate(scores), np.concatenate(cls)
    if len(b) == 0:
        return b, s, c
    keep = torchvision.ops.batched_nms(torch.from_numpy(b), torch.from_numpy(s), torch.from_numpy(c), nms_iou).numpy()
    return b[keep], s[keep], c[keep]


def draw(bgr, dets, path, dropped=(), mask=None):
    out = bgr.copy()
    if mask is not None:
        out[mask == 0] = (out[mask == 0] * 0.45).astype(np.uint8)
    for d in dropped:
        x1, y1, x2, y2 = map(int, d["box_xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box_xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{d['score']:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="*")
    ap.add_argument("--list")
    ap.add_argument("--model", choices=MODELS, default="large")
    ap.add_argument("--weights", default=None, help="fine-tuned checkpoint (player/official classes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--classes", default="person", help="comma list of COCO names, or 'all'")
    ap.add_argument("--tiles", type=int, default=1)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--viz-dir")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "eager"], default="fp16",
                    help="model.inference() dtype; 'eager' skips the inference optimisation")
    ap.add_argument("--compile", action="store_true", help="torchscript-trace in model.inference()")
    ap.add_argument("--view", choices=["wide", "ez"], default="wide", help="camera angle, for the field mask")
    ap.add_argument("--field-mask", action="store_true",
                    help="drop persons whose foot point is off the playing field (fieldmask.py); kept in 'dropped'")
    args = ap.parse_args()

    frames = list(args.frames)
    if args.list:
        frames += [l.strip() for l in open(args.list) if l.strip()]
    keep_names = None if args.classes == "all" else set(args.classes.split(","))

    if args.weights:
        # fine-tuned player/official model: class 0 = player (reported as 'person' so downstream filters are
        # unchanged), class 1 = official
        model = MODELS[args.model](pretrain_weights=args.weights)
        names = {0: "person", 1: "official"}
    else:
        model = MODELS[args.model]()
        names = COCO_CLASSES
    if args.dtype != "eager":
        model.inference(compile=args.compile, dtype=torch.float16 if args.dtype == "fp16" else torch.float32)

    images = [cv2.imread(p) for p in frames]
    for _ in range(args.warmup):
        detect_frame(model, images[0][:, :, ::-1].copy(), args.threshold, args.tiles, args.overlap, args.nms_iou)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

    out_frames, times = [], []
    for p, bgr in zip(frames, images):
        rgb = bgr[:, :, ::-1].copy()
        torch.cuda.synchronize(); t = time.perf_counter()
        b, s, c = detect_frame(model, rgb, args.threshold, args.tiles, args.overlap, args.nms_iou)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        times.append(dt)
        dets = []
        for bb, ss, cc in zip(b, s, c):
            name = names.get(int(cc), str(int(cc)))
            if keep_names is None or name in keep_names:
                dets.append({"box_xyxy": [round(float(v), 1) for v in bb], "score": round(float(ss), 4),
                             "class_id": int(cc), "class_name": name})
        rec = {"path": os.path.abspath(p), "width": bgr.shape[1], "height": bgr.shape[0],
               "seconds": round(dt, 4), "detections": dets}
        mask = None
        dropped = []
        if args.field_mask:
            mask, minfo = field_mask(bgr, view=args.view)
            rec["num_before_field_mask"] = len(dets)
            green = minfo.pop("green")
            dets, dropped = filter_detections(dets, mask, green=green)
            rec.update({"detections": dets, "dropped": dropped, "field_mask": minfo})
        out_frames.append(rec)
        if args.viz_dir:
            os.makedirs(args.viz_dir, exist_ok=True)
            draw(bgr, dets, os.path.join(args.viz_dir, f"{os.path.splitext(os.path.basename(p))[0]}_{args.model}_t{args.tiles}"
                                         f"{'_fm' if args.field_mask else ''}.jpg"), dropped, mask)

    stats = {"mean_seconds_per_frame": round(float(np.mean(times)), 4),
             "peak_gpu_mem_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
             "peak_gpu_mem_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1)}
    res = {"model": args.model, "weights": args.weights, "dtype": args.dtype, "compile": args.compile, "rfdetr_version": rfdetr.__version__ if hasattr(rfdetr, "__version__") else None,
           "tiles": args.tiles, "threshold": args.threshold, "classes": args.classes, "field_mask": args.field_mask,
           "frames": out_frames, "stats": stats}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({"model": args.model, "tiles": args.tiles, "dtype": args.dtype, "compile": args.compile, **stats,
                      "counts": [len(f["detections"]) for f in out_frames],
                      **({"counts_before_mask": [f["num_before_field_mask"] for f in out_frames]} if args.field_mask else {})}))


if __name__ == "__main__":
    main()
