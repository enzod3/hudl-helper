#!/usr/bin/env python
"""Single-view field positions from SAM tracks + cross-view check + bird's-eye video (sam3env python).

  birdseye.py PAIR_DIR [--views wide ez] [--out-video birdseye.mp4]

Needs PAIR_DIR/sync.json, PAIR_DIR/<view>/calib.json (H per source frame, field yards -> image px) and
PAIR_DIR/track_<view>/{masks.json,frames.json} (segment.py on frames source_frame_start..).
Foot point per mask: bottom-centre of the mask's lowest 10% of rows (x = mean column of those pixels, y = lowest
row); fallback to the box bottom-centre when the mask is tiny. Mapped through H(t)^-1 to yards.
Shared timeline: tau = pts - snap_pts(view) (the sync offset is snap_wide - snap_ez, so this is the same as
wide_time = ez_time + offset).
Writes PAIR_DIR/field_tracks.json and PAIR_DIR/crossview.json: at the snap (and over all common frames),
the views' players are matched by nearest neighbour (Hungarian, gated at 5 yd) and distances reported.
"""
import argparse, json, os, subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

FIELD_LEN, FIELD_W = 120.0, 160.0 / 3.0


def foot_point(m, box):
    ys, xs = np.nonzero(m)
    if len(ys) < 15:
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, float(y2), "box"
    top, bot = ys.min(), ys.max()
    cut = bot - 0.10 * (bot - top + 1)
    sel = ys >= cut
    return float(xs[sel].mean()), float(bot), "mask"


def view_tracks(pair, v, suffix=""):
    sync = json.load(open(os.path.join(pair, "sync.json")))
    cal = json.load(open(os.path.join(pair, v, "calib.json")))
    tdir = os.path.join(pair, f"track_{v}{suffix}")
    meta = json.load(open(os.path.join(tdir, "frames.json")))
    masks = json.load(open(os.path.join(tdir, "masks.json")))
    a = meta["source_frame_start"]
    snap = sync[v]["snap_pts"]
    pts = cal["pts_s"]
    out = {}   # id -> list of {tau, x, y, src}
    for fr in masks["frames"]:
        sf = a + fr["frame_idx"]
        Hinv = np.linalg.inv(np.array(cal["H"][sf]))
        tau = pts[sf] - snap
        for o in fr["objects"]:
            m = mask_utils.decode(o["rle"]).astype(bool)
            fx, fy, src = foot_point(m, o["box_xyxy"])
            p = cv2.perspectiveTransform(np.array([[[fx, fy]]], float), Hinv)[0, 0]
            out.setdefault(o["id"], []).append({"tau": round(tau, 4), "frame": sf, "x": round(float(p[0]), 3),
                                                "y": round(float(p[1]), 3), "img": [round(fx, 1), round(fy, 1)], "src": src,
                                                "calib_residual_px": cal["info"][sf]["residual_px"]})
    return out


def positions_at(tracks, tau, tol=0.02):
    ids, P = [], []
    for i, rows in tracks.items():
        best = min(rows, key=lambda r: abs(r["tau"] - tau))
        if abs(best["tau"] - tau) <= tol:
            ids.append(i); P.append((best["x"], best["y"]))
    return ids, np.array(P).reshape(-1, 2)


def match(Pa, Pb, gate=5.0):
    if len(Pa) == 0 or len(Pb) == 0:
        return np.zeros(0), []
    D = np.linalg.norm(Pa[:, None] - Pb[None], axis=2)
    r, c = linear_sum_assignment(D)
    keep = D[r, c] <= gate
    return D[r, c][keep], list(zip(r[keep], c[keep]))


def stats(d):
    if len(d) == 0:
        return {"n": 0}
    return {"n": int(len(d)), "median_yd": round(float(np.median(d)), 3), "p95_yd": round(float(np.percentile(d, 95)), 3),
            "mean_yd": round(float(np.mean(d)), 3), "max_yd": round(float(np.max(d)), 3)}


def draw_field(ppy, margin, x_range):
    x0, x1 = x_range
    W = int((x1 - x0) * ppy + 2 * margin); H = int(FIELD_W * ppy + 2 * margin)
    img = np.full((H, W, 3), (40, 110, 40), np.uint8)
    def P(x, y):
        return int(margin + (x - x0) * ppy), int(margin + (FIELD_W - y) * ppy)   # y=0 (Wide side) at the bottom
    for x in range(int(np.ceil(x0)), int(x1) + 1):
        if 0 <= x <= FIELD_LEN:
            col, th = ((255, 255, 255), 2) if x % 5 == 0 else ((70, 140, 70), 1)
            cv2.line(img, P(x, 0), P(x, FIELD_W), col, th)
            if x % 10 == 0 and 10 <= x <= 110:
                yd = int(x - 10 if x <= 60 else 110 - x)
                cv2.putText(img, str(yd), (P(x, 12)[0] - 12, P(x, 12)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for y in (0, FIELD_W):
        cv2.line(img, P(max(0, x0), y), P(min(FIELD_LEN, x1), y), (255, 255, 255), 2)
    for y in (20.0, FIELD_W - 20.0):
        for x in range(int(np.ceil(x0)), int(x1) + 1):
            cv2.line(img, P(x, y - 0.67), P(x, y), (255, 255, 255), 1)
    return img, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair")
    ap.add_argument("--views", nargs=2, default=["wide", "ez"])
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--track-suffix", default="", help="read track_<view><suffix>/ (e.g. _crop)")
    a = ap.parse_args()
    va, vb = a.views
    T = {v: view_tracks(a.pair, v, a.track_suffix) for v in a.views}
    json.dump(T, open(os.path.join(a.pair, f"field_tracks{a.track_suffix}.json"), "w"))
    taus = sorted({r["tau"] for rows in T[va].values() for r in rows})
    # cross-view at the snap and over all common frames
    ia, Pa = positions_at(T[va], 0.0); ib, Pb = positions_at(T[vb], 0.0)
    d0, pairs0 = match(Pa, Pb)
    all_d, per_t = [], []
    for tau in taus:
        _, A = positions_at(T[va], tau); _, B = positions_at(T[vb], tau)
        d, _ = match(A, B)
        all_d += list(d)
        per_t.append({"tau": tau, **stats(d)})
    res = {"pair": os.path.basename(os.path.abspath(a.pair)),
           "snap": {**stats(d0), "players_view_a": len(ia), "players_view_b": len(ib),
                    "pairs": [{"a_id": int(ia[i]), "b_id": int(ib[j]), "dist_yd": round(float(np.linalg.norm(Pa[i] - Pb[j])), 3),
                               "a_xy": Pa[i].round(2).tolist(), "b_xy": Pb[j].round(2).tolist()} for i, j in pairs0]},
           "all_frames": stats(np.array(all_d)), "per_frame": per_t,
           "views": a.views, "gate_yd": 5.0, "note": "nearest-neighbour (Hungarian) matching, no identity info"}
    json.dump(res, open(os.path.join(a.pair, f"crossview{a.track_suffix}.json"), "w"), indent=1)
    print(json.dumps({"snap": {k: v for k, v in res["snap"].items() if k != "pairs"}, "all_frames": res["all_frames"]}))

    # bird's-eye video
    allx = [r["x"] for v in a.views for rows in T[v].values() for r in rows if -5 < r["x"] < 125]
    x_range = (max(-2, np.percentile(allx, 1) - 8), min(122, np.percentile(allx, 99) + 8))
    base, P = draw_field(14, 30, x_range)
    H, W = base.shape[:2]
    W2, H2 = W + (W % 2), H + 60 + ((H + 60) % 2)
    out = os.path.join(a.pair, f"birdseye{a.track_suffix}.mp4")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W2}x{H2}", "-r", str(a.fps),
                             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], stdin=subprocess.PIPE)
    cols = {va: (255, 160, 40), vb: (40, 140, 255)}
    trails = {v: {} for v in a.views}
    for tau in taus:
        img = np.zeros((H2, W2, 3), np.uint8); img[:H, :W] = base
        for v in a.views:
            for i, rows in T[v].items():
                best = min(rows, key=lambda r: abs(r["tau"] - tau))
                if abs(best["tau"] - tau) > 0.02:
                    continue
                trails[v].setdefault(i, []).append(P(best["x"], best["y"]))
                tr = trails[v][i][-20:]
                for p, q in zip(tr[:-1], tr[1:]):
                    cv2.line(img, p, q, cols[v], 1)
                c = P(best["x"], best["y"])
                if v == va:
                    cv2.circle(img, c, 6, cols[v], -1)
                else:
                    cv2.rectangle(img, (c[0] - 5, c[1] - 5), (c[0] + 5, c[1] + 5), cols[v], 2)
                cv2.putText(img, str(i), (c[0] + 6, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cols[v], 1)
        cv2.putText(img, f"t = {tau:+.2f} s from snap   {va} (filled circles)   {vb} (squares)", (12, H + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        proc.stdin.write(img.tobytes())
    proc.stdin.close(); proc.wait()


if __name__ == "__main__":
    main()
