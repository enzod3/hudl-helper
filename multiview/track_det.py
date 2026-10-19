#!/usr/bin/env python
"""Detector-only tracking (no SAM): ByteTrack-style linking of RF-DETR boxes, frame to frame (CPU).

  track_det.py PAIR_DIR VIEW [--suffix _bt] [--high 0.5] [--max-lost 30] [--min-hits 3] [--window crop]

Detections: PAIR_DIR/VIEW/dets_all.json (field-masked 'detections'). Window: the frames of track_<view>_crop
(frames.json source_frame_start / num_frames), so the output lines up with the SAM tracks frame for frame.
Each track carries a constant-velocity Kalman filter on the box (cx, cy, w, h and their velocities).
  stage 1: high-confidence detections (score >= --high) vs all tracks (active + lost), Hungarian on 1 - IoU of the
           predicted box, accepted at IoU >= 0.2
  stage 2: remaining low-confidence detections vs remaining active tracks, accepted at IoU >= 0.5
  unmatched high detections start new tracks; a track becomes confirmed after --min-hits matches; a track unmatched
  for more than --max-lost frames is removed.
Only frames where a confirmed track matched a real detection are written (no predicted boxes), the same rule as the
detector-gated SAM observations. Output: PAIR_DIR/track_<view><suffix>/tracks.json
  {"source_frame_start", "num_frames", "frames": [{"frame_idx", "objects": [{"id", "box_xyxy", "score"}]}], "stats"}
"""
import argparse, json, os, time

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ub = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(ua[:, None] + ub[None] - inter, 1e-9)


class Track:
    F = np.eye(8)
    for i in range(4):
        F[i, i + 4] = 1.0
    Hm = np.eye(4, 8)

    def __init__(self, tid, box, score):
        cx, cy, w, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], float)
        self.P = np.diag([h, h, h, h, 10 * h, 10 * h, 10 * h, 10 * h]) ** 2 * 0.01
        self.id, self.hits, self.lost, self.score = tid, 1, 0, score

    def box(self):
        cx, cy, w, h = self.x[:4]
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def predict(self):
        h = max(self.x[3], 1.0)
        Q = np.diag([0.05 * h, 0.05 * h, 0.05 * h, 0.05 * h, 0.01 * h, 0.01 * h, 0.01 * h, 0.01 * h]) ** 2
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + Q
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)

    def update(self, box, score):
        z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])
        h = max(z[3], 1.0)
        R = np.diag([0.05 * h, 0.05 * h, 0.05 * h, 0.05 * h]) ** 2
        S = self.Hm @ self.P @ self.Hm.T + R
        K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.Hm @ self.x); self.P = (np.eye(8) - K @ self.Hm) @ self.P
        self.hits += 1; self.lost = 0; self.score = score


def match(tracks, dets, thr):
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    M = iou_matrix([t.box() for t in tracks], [d["box_xyxy"] for d in dets])
    r, c = linear_sum_assignment(1 - M)
    pairs = [(i, j) for i, j in zip(r, c) if M[i, j] >= thr]
    mt = {i for i, _ in pairs}; md = {j for _, j in pairs}
    return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(dets)) if j not in md]


def track(frames_dets, high=0.5, max_lost=30, min_hits=3):
    tracks, nxt, out = [], 1, []
    for dets in frames_dets:
        for t in tracks:
            t.predict()
        hi = [d for d in dets if d["score"] >= high]; lo = [d for d in dets if d["score"] < high]
        p1, ut, ud = match(tracks, hi, 0.2)
        objs = []
        for i, j in p1:
            tracks[i].update(hi[j]["box_xyxy"], hi[j]["score"])
        active = [tracks[i] for i in ut if tracks[i].lost == 0]
        p2, ut2, _ = match(active, lo, 0.5)
        for i, j in p2:
            active[i].update(lo[j]["box_xyxy"], lo[j]["score"])
        matched = {id(tracks[i]) for i, _ in p1} | {id(active[i]) for i, _ in p2}
        for t in tracks:
            if id(t) not in matched:
                t.lost += 1
        for j in ud:
            tracks.append(Track(nxt, hi[j]["box_xyxy"], hi[j]["score"])); nxt += 1
            matched.add(id(tracks[-1]))
        tracks = [t for t in tracks if t.lost <= max_lost]
        for t in tracks:
            if id(t) in matched and t.hits >= min_hits:
                objs.append({"id": int(t.id), "box_xyxy": [round(float(v), 1) for v in t.box()], "score": round(float(t.score), 4)})
        out.append(objs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair"); ap.add_argument("view")
    ap.add_argument("--suffix", default="_bt"); ap.add_argument("--high", type=float, default=0.5)
    ap.add_argument("--max-lost", type=int, default=30); ap.add_argument("--min-hits", type=int, default=3)
    a = ap.parse_args()
    t0 = time.perf_counter()
    meta = json.load(open(os.path.join(a.pair, f"track_{a.view}_crop", "frames.json")))
    dets = json.load(open(os.path.join(a.pair, a.view, "dets_all.json")))["frames"]
    st, n = meta["source_frame_start"], meta["num_frames"]
    res = track([dets[st + k]["detections"] for k in range(n)], a.high, a.max_lost, a.min_hits)
    od = os.path.join(a.pair, f"track_{a.view}{a.suffix}"); os.makedirs(od, exist_ok=True)
    stats = {"seconds": round(time.perf_counter() - t0, 3), "frames": n, "tracks": len({o["id"] for f in res for o in f}),
             "objects_per_frame_median": float(np.median([len(f) for f in res])) if res else 0.0,
             "params": {"high": a.high, "max_lost": a.max_lost, "min_hits": a.min_hits}}
    json.dump({"source_frame_start": st, "num_frames": n, "frames": [{"frame_idx": k, "objects": f} for k, f in enumerate(res)],
               "stats": stats}, open(os.path.join(od, "tracks.json"), "w"))
    print(json.dumps({"pair": a.pair, "view": a.view, **stats}))


if __name__ == "__main__":
    main()
