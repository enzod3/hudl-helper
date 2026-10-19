#!/usr/bin/env python
"""Detector comparison on the held-out test split (rfdetr_env python, from ~/sam3_work).

  eval_det.py DATASET_DIR --model coco|PATH_TO_CHECKPOINT [--name NAME] [--tiles 2] [--threshold 0.3]

GT: DATASET_DIR/test labels, class 0 (player). Both GT and predictions are restricted to the playing field with the
same field mask as the pipeline (foot point inside the mask), so sideline people do not count either way.
Predictions: COCO model -> class 'person'; fine-tuned -> class 'player' (officials are reported separately).
Matching: greedy by score, IoU >= 0.5. Reported: recall (all, clearly visible = h >= 40 px, not cut by the image
border, IoU < 0.1 with every other GT), false positives per frame, precision, recall by GT height bucket, and
crowded recall (GT overlapping another GT with IoU >= 0.15 or centres closer than 0.6 x box width).
Writes DATASET_DIR/../eval_<name>.json.
"""
import argparse, glob, json, os, sys, time

os.environ.setdefault("RF_HOME", os.path.expanduser("~/sam3_work/rfdetr_weights"))
import cv2
import numpy as np
import torch

sys.path[:0] = __import__("glob").glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "*", ""))  # all src/multiview/* module dirs (flat sibling imports)
from fieldmask import field_mask  # noqa: E402

BUCKETS = [(0, 40), (40, 60), (60, 90), (90, 130), (130, 10000)]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def load_model(spec):
    from rfdetr import RFDETRLarge
    if spec == "coco":
        m = RFDETRLarge()
    else:
        m = RFDETRLarge(pretrain_weights=spec)
    try:
        m.optimize_for_inference(dtype=torch.float16)
    except Exception:
        try:
            m.inference(dtype=torch.float16)
        except Exception:
            pass
    return m


def predict(m, rgb, thr, tiles, overlap=0.2):
    import torchvision
    H, W = rgb.shape[:2]
    if tiles <= 1:
        wins = [(0, 0, W, H)]
    else:
        tw = int(np.ceil(W / (tiles - (tiles - 1) * overlap))); th = int(np.ceil(H / (tiles - (tiles - 1) * overlap)))
        wins = [(x, y, tw, th) for y in np.linspace(0, H - th, tiles).astype(int) for x in np.linspace(0, W - tw, tiles).astype(int)]
    B, S, C = [], [], []
    for x, y, w, h in wins:
        d = m.predict(np.ascontiguousarray(rgb[y:y + h, x:x + w]), threshold=thr, include_source_image=False)
        b = d.xyxy.astype(np.float32).copy(); b[:, [0, 2]] += x; b[:, [1, 3]] += y
        B.append(b); S.append(d.confidence.astype(np.float32)); C.append(d.class_id.astype(np.int64))
    b, s, c = np.concatenate(B), np.concatenate(S), np.concatenate(C)
    if len(b) and tiles > 1:
        k = torchvision.ops.batched_nms(torch.from_numpy(b), torch.from_numpy(s), torch.from_numpy(c), 0.5).numpy()
        b, s, c = b[k], s[k], c[k]
    return b, s, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--model", required=True); ap.add_argument("--name")
    ap.add_argument("--tiles", type=int, default=2); ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    name = a.name or ("coco" if a.model == "coco" else os.path.basename(os.path.dirname(a.model)))
    m = load_model(a.model)
    names = getattr(m, "class_names", None)
    if a.model == "coco":
        from rfdetr.assets.coco_classes import COCO_CLASSES
        player_ids, official_ids = {k for k, v in COCO_CLASSES.items() if v == "person"}, set()
    else:
        # fine-tuned: class indices follow data.yaml names (player, official); probe both 0/1-based conventions
        player_ids, official_ids = None, None
    imgs = sorted(glob.glob(os.path.join(a.dataset, a.split, "images", "*.jpg")))
    tot = {"gt": 0, "tp": 0, "fp": 0, "clear_gt": 0, "clear_tp": 0, "crowd_gt": 0, "crowd_tp": 0, "official_pred": 0}
    bucket = {f"{lo}-{hi}": [0, 0] for lo, hi in BUCKETS}
    class_hist = {}
    t_inf = 0.0
    for f in imgs:
        bgr = cv2.imread(f); H, W = bgr.shape[:2]
        fm, info = field_mask(bgr, view="ez" if "_end_" in f or "_ez_" in f else "wide")
        lab = f.replace("/images/", "/labels/").replace(".jpg", ".txt")
        gt = []
        for l in open(lab):
            p = l.split()
            if not p or p[0] != "0":
                continue
            cx, cy, w, h = [float(v) for v in p[1:5]]
            x1, y1, x2, y2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
            fx, fy = int(min(max((x1 + x2) / 2, 0), W - 1)), int(min(max(y2 - 3, 0), H - 1))
            if fm[fy, fx] > 0:
                gt.append([x1, y1, x2, y2])
        t0 = time.time()
        b, s, c = predict(m, bgr[:, :, ::-1].copy(), a.threshold, a.tiles)
        t_inf += time.time() - t0
        for k in c:
            class_hist[int(k)] = class_hist.get(int(k), 0) + 1
        if player_ids is None:
            # infer fine-tuned class ids: the most frequent class is 'player'
            ks = sorted(class_hist, key=lambda k: -class_hist[k])
            player_ids = {ks[0]} if ks else {0}
        keep = []
        for bb, ss, cc in zip(b, s, c):
            fx, fy = int(min(max((bb[0] + bb[2]) / 2, 0), W - 1)), int(min(max(bb[3] - 3, 0), H - 1))
            if fm[fy, fx] == 0:
                continue
            if int(cc) in player_ids:
                keep.append((float(ss), bb))
            else:
                tot["official_pred"] += 1
        keep.sort(key=lambda x: -x[0])
        matched = [False] * len(gt)
        for ss, bb in keep:
            best, bi = 0.5, -1
            for j, g in enumerate(gt):
                if not matched[j]:
                    v = iou(bb, g)
                    if v >= best:
                        best, bi = v, j
            if bi >= 0:
                matched[bi] = True; tot["tp"] += 1
            else:
                tot["fp"] += 1
        tot["gt"] += len(gt)
        for j, g in enumerate(gt):
            h = g[3] - g[1]; w = g[2] - g[0]
            others = [gt[k] for k in range(len(gt)) if k != j]
            mx = max([iou(g, o) for o in others], default=0.0)
            close = any(abs((g[0] + g[2]) / 2 - (o[0] + o[2]) / 2) < 0.6 * w and abs(g[3] - o[3]) < 0.5 * h for o in others)
            for lo, hi in BUCKETS:
                if lo <= h < hi:
                    bucket[f"{lo}-{hi}"][0] += 1; bucket[f"{lo}-{hi}"][1] += int(matched[j])
            clear = h >= 40 and g[1] > 2 and g[3] < H - 2 and mx < 0.1
            if clear:
                tot["clear_gt"] += 1; tot["clear_tp"] += int(matched[j])
            if mx >= 0.15 or close:
                tot["crowd_gt"] += 1; tot["crowd_tp"] += int(matched[j])
    n = len(imgs)
    res = {"name": name, "model": a.model, "tiles": a.tiles, "threshold": a.threshold, "images": n,
           "recall": round(tot["tp"] / max(1, tot["gt"]), 4), "precision": round(tot["tp"] / max(1, tot["tp"] + tot["fp"]), 4),
           "fp_per_frame": round(tot["fp"] / max(1, n), 2),
           "recall_clearly_visible": round(tot["clear_tp"] / max(1, tot["clear_gt"]), 4), "n_clear": tot["clear_gt"],
           "recall_crowded": round(tot["crowd_tp"] / max(1, tot["crowd_gt"]), 4), "n_crowded": tot["crowd_gt"],
           "recall_by_height_px": {k: {"n": v[0], "recall": round(v[1] / max(1, v[0]), 4)} for k, v in bucket.items()},
           "official_preds_on_field": tot["official_pred"], "class_hist": class_hist, "player_class_ids": sorted(player_ids),
           "seconds_per_image": round(t_inf / max(1, n), 3), "gt_players": tot["gt"]}
    out = os.path.join(os.path.dirname(os.path.abspath(a.dataset.rstrip("/"))), f"eval_{name}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
