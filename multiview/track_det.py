#!/usr/bin/env python
"""Detector-only tracking (no SAM): ByteTrack-style linking of RF-DETR boxes, frame to frame (CPU).

  track_det.py PAIR_DIR VIEW [--suffix _bt] [--high 0.5] [--max-lost 30] [--min-hits 3] [--reid 1] [--relink-gap 60]

Detections: PAIR_DIR/VIEW/dets_all.json (field-masked 'detections'). Window: the frames of track_<view>_crop
(frames.json source_frame_start / num_frames), so the output lines up with the SAM tracks frame for frame.
Each track carries a constant-velocity Kalman filter on the box (cx, cy, w, h and their velocities).
  stage 1: high-confidence detections (score >= --high) vs all tracks (active + lost), Hungarian on 1 - IoU of the
           predicted box, accepted at IoU >= 0.2
  stage 2: remaining low-confidence detections vs remaining active tracks, accepted at IoU >= 0.5
  unmatched high detections start new tracks; a track becomes confirmed after --min-hits matches; a track unmatched
  for more than --max-lost frames is removed.
Only frames where a confirmed track matched a real detection are written (no predicted boxes), the same rule as the
detector-gated SAM observations.
Appearance (--reid 1): per detection an HSV colour histogram of the jersey region (15-50% of the box height) and of the
shorts / socks region (55-90%), central 60% of the width, L2-normalised (256-d). Each track keeps an exponential moving
average. (a) Matching: cost = 1 - IoU + --app-w * (1 - cosine(track, detection)), same IoU gates. (b) Re-link pass after
tracking: a track that ends and another that starts within --relink-gap frames are joined when the constant-velocity
prediction of the first lands within (1 + 0.05 * gap) box heights of the second's start and their appearance cosine is
>= --relink-sim (Hungarian, one successor / predecessor each). (c) Officials: a track whose jersey region shows striped
dark / white alternation on most of its frames is dropped (not a player). Output: PAIR_DIR/track_<view><suffix>/tracks.json
  {"source_frame_start", "num_frames", "frames": [{"frame_idx", "objects": [{"id", "box_xyxy", "score"}]}], "stats"}
"""
import argparse, json, os, time

import cv2
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
        self.feat = None

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


def appearance(img, box):
    """256-d L2-normalised HSV histogram: jersey region + shorts/socks region; and a stripe score of the jersey."""
    h_, w_ = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    xa, xb = int(max(0, x1 + 0.2 * bw)), int(min(w_, x2 - 0.2 * bw))
    feats, stripe = [], 0.0
    for lo, hi in ((0.15, 0.50), (0.55, 0.90)):
        ya, yb = int(max(0, y1 + lo * bh)), int(min(h_, y1 + hi * bh))
        patch = img[ya:yb, xa:xb]
        if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
            feats.append(np.zeros(128, np.float32)); continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        feats.append(hist / max(hist.sum(), 1e-9))
        if lo == 0.15:
            L = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(float)
            cls = np.where(L < 80, 0, np.where(L > 175, 2, 1))
            trans = []
            for row in cls:
                r_ = row[row != 1]
                trans.append(int(np.sum(r_[1:] != r_[:-1])) if len(r_) > 1 else 0)
            stripe = float(np.mean(trans)) if trans else 0.0
            dark, white = float(np.mean(L < 80)), float(np.mean(L > 175))
            stripe = stripe if (dark > 0.15 and white > 0.12) else 0.0
    f = np.concatenate(feats)
    return f / max(np.linalg.norm(f), 1e-9), stripe


def match(tracks, dets, thr, app_w=0.0):
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    M = iou_matrix([t.box() for t in tracks], [d["box_xyxy"] for d in dets])
    cost = 1 - M
    if app_w > 0:
        A = np.array([t.feat for t in tracks]); B = np.array([d["feat"] for d in dets])
        cost = cost + app_w * (1 - A @ B.T)
    r, c = linear_sum_assignment(cost)
    pairs = [(i, j) for i, j in zip(r, c) if M[i, j] >= thr]
    mt = {i for i, _ in pairs}; md = {j for _, j in pairs}
    return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(dets)) if j not in md]


def track(frames_dets, high=0.5, max_lost=30, min_hits=3, app_w=0.0):
    tracks, nxt, out = [], 1, []
    for dets in frames_dets:
        for t in tracks:
            t.predict()
        hi = [d for d in dets if d["score"] >= high]; lo = [d for d in dets if d["score"] < high]
        p1, ut, ud = match(tracks, hi, 0.2, app_w)
        objs = []
        det_of = {}
        for i, j in p1:
            tracks[i].update(hi[j]["box_xyxy"], hi[j]["score"]); det_of[id(tracks[i])] = hi[j]
        active = [tracks[i] for i in ut if tracks[i].lost == 0]
        p2, ut2, _ = match(active, lo, 0.5, app_w)
        for i, j in p2:
            active[i].update(lo[j]["box_xyxy"], lo[j]["score"]); det_of[id(active[i])] = lo[j]
        for t in tracks:
            d = det_of.get(id(t))
            if d is not None and d.get("feat") is not None:
                t.feat = d["feat"] if t.feat is None else (0.9 * t.feat + 0.1 * d["feat"]) / max(np.linalg.norm(0.9 * t.feat + 0.1 * d["feat"]), 1e-9)
        matched = {id(tracks[i]) for i, _ in p1} | {id(active[i]) for i, _ in p2}
        for t in tracks:
            if id(t) not in matched:
                t.lost += 1
        for j in ud:
            tracks.append(Track(nxt, hi[j]["box_xyxy"], hi[j]["score"])); nxt += 1
            tracks[-1].feat = hi[j].get("feat"); det_of[id(tracks[-1])] = hi[j]
            matched.add(id(tracks[-1]))
        tracks = [t for t in tracks if t.lost <= max_lost]
        for t in tracks:
            if id(t) in matched and t.hits >= min_hits:
                d = det_of.get(id(t), {})
                box = d.get("box_xyxy", t.box())   # the matched detection itself, not the Kalman-smoothed box
                objs.append({"id": int(t.id), "box_xyxy": [round(float(v), 1) for v in box], "score": round(float(t.score), 4),
                             "_feat": d.get("feat"), "_stripe": d.get("stripe", 0.0)})
        out.append(objs)
    return out


def relink(res, max_gap=60, min_sim=0.8):
    """Join a track that ends with one that starts up to max_gap frames later (same player re-acquired)."""
    tr = {}
    for k, objs in enumerate(res):
        for o in objs:
            tr.setdefault(o["id"], []).append((k, o))
    info = {}
    for tid, lst in tr.items():
        fe = [o["_feat"] for _, o in lst if o.get("_feat") is not None]
        f = np.mean(fe, 0) if fe else None
        (k0, o0), (k1, o1) = lst[0], lst[-1]
        tail = lst[-6:]
        c = lambda b: np.array([(b[0] + b[2]) / 2, b[3]])
        vel = (c(tail[-1][1]["box_xyxy"]) - c(tail[0][1]["box_xyxy"])) / max(1, tail[-1][0] - tail[0][0]) if len(tail) > 1 else np.zeros(2)
        info[tid] = {"k0": k0, "k1": k1, "b0": o0["box_xyxy"], "b1": o1["box_xyxy"], "vel": vel,
                     "feat": None if f is None else f / max(np.linalg.norm(f), 1e-9)}
    ids = sorted(info)
    cand = []
    for a in ids:
        A = info[a]
        for b in ids:
            B = info[b]
            gap = B["k0"] - A["k1"]
            if a == b or gap < 1 or gap > max_gap or A["feat"] is None or B["feat"] is None:
                continue
            h = max(A["b1"][3] - A["b1"][1], 1.0)
            pred = np.array([(A["b1"][0] + A["b1"][2]) / 2, A["b1"][3]]) + A["vel"] * gap
            d = np.linalg.norm(np.array([(B["b0"][0] + B["b0"][2]) / 2, B["b0"][3]]) - pred) / h
            sim = float(A["feat"] @ B["feat"])
            if d <= 1.0 + 0.05 * gap and sim >= min_sim:
                cand.append((d / (1.0 + 0.05 * gap) + (1 - sim) * 4, a, b))
    cand.sort()
    nxt, prv = {}, {}
    for _, a, b in cand:
        if a in nxt or b in prv:
            continue
        nxt[a] = b; prv[b] = a
    root = {}
    for tid in ids:
        r = tid
        while r in prv:
            r = prv[r]
        root[tid] = r
    for objs in res:
        for o in objs:
            o["id"] = root[o["id"]]
    return len(nxt)


def drop_officials(res, min_frac=0.5, stripe_thr=2.0):
    """Remove tracks whose jersey shows dark/white stripes on >= min_frac of their frames."""
    st = {}
    for objs in res:
        for o in objs:
            s_ = st.setdefault(o["id"], [0, 0]); s_[0] += 1; s_[1] += int(o.get("_stripe", 0.0) >= stripe_thr)
    off = {t for t, (n, k) in st.items() if n >= 5 and k >= min_frac * n}
    for i, objs in enumerate(res):
        res[i] = [o for o in objs if o["id"] not in off]
    return len(off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair"); ap.add_argument("view")
    ap.add_argument("--suffix", default="_bt"); ap.add_argument("--high", type=float, default=0.5)
    ap.add_argument("--max-lost", type=int, default=30); ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--reid", type=int, default=1, help="1: colour-histogram appearance in matching, re-link pass, official filter")
    ap.add_argument("--app-w", type=float, default=0.5); ap.add_argument("--relink-gap", type=int, default=60)
    ap.add_argument("--relink-sim", type=float, default=0.8)
    a = ap.parse_args()
    t0 = time.perf_counter()
    meta = json.load(open(os.path.join(a.pair, f"track_{a.view}_crop", "frames.json")))
    dets = json.load(open(os.path.join(a.pair, a.view, "dets_all.json")))["frames"]
    st, n = meta["source_frame_start"], meta["num_frames"]
    fd = [[dict(d) for d in dets[st + k]["detections"]] for k in range(n)]
    n_relinked = n_off = 0
    if a.reid:
        fdir = os.path.join(a.pair, a.view, "frames")
        frames = sorted(f for f in os.listdir(fdir) if f.endswith(".jpg"))
        for k in range(n):
            if not fd[k]:
                continue
            img = cv2.imread(os.path.join(fdir, frames[st + k]))
            for d in fd[k]:
                d["feat"], d["stripe"] = appearance(img, d["box_xyxy"])
    res = track(fd, a.high, a.max_lost, a.min_hits, a.app_w if a.reid else 0.0)
    if a.reid:
        n_relinked = relink(res, a.relink_gap, a.relink_sim)
        n_off = drop_officials(res)
    for objs in res:
        for o in objs:
            o.pop("_feat", None); o.pop("_stripe", None)
    od = os.path.join(a.pair, f"track_{a.view}{a.suffix}"); os.makedirs(od, exist_ok=True)
    stats = {"seconds": round(time.perf_counter() - t0, 3), "frames": n, "tracks": len({o["id"] for f in res for o in f}),
             "objects_per_frame_median": float(np.median([len(f) for f in res])) if res else 0.0,
             "relinked": n_relinked, "official_tracks_dropped": n_off,
             "params": {"high": a.high, "max_lost": a.max_lost, "min_hits": a.min_hits, "reid": a.reid, "app_w": a.app_w,
                        "relink_gap": a.relink_gap, "relink_sim": a.relink_sim}}
    json.dump({"source_frame_start": st, "num_frames": n, "frames": [{"frame_idx": k, "objects": f} for k, f in enumerate(res)],
               "stats": stats}, open(os.path.join(od, "tracks.json"), "w"))
    print(json.dumps({"pair": a.pair, "view": a.view, **stats}))


if __name__ == "__main__":
    main()
