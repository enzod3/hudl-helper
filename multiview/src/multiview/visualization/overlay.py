#!/usr/bin/env python
"""Reconstruction overlaid on the real footage (sam3env python, CPU only).

  overlay.py PAIR_DIR [--recon-name recon] [--views wide ez] [--width 1280] [--stack]

For each view: the source frames over the reconstructed window (recon time t -> that view's frame through its
detected snap, as in recon.py), every slot's fused position from recon.jsonl projected back into the camera through
H(t) (field yards -> image px) and drawn at the player's feet: a ring of 0.6 yd radius on the ground (so it is
foreshortened like the field), coloured by team (A dark, B light), the slot label, a trail of the last ~1 s of
projected positions. Ring style by source: solid thick = fused (both views), dashed = one view, thin grey = inferred.
Also the line of scrimmage at the snap (recon_summary.json) projected onto the field, and t from the snap.
Slot positions are in the Wide-aligned field frame (recon.py removes the End Zone's cross-view bias and noise-model
bias before fusing); in the End Zone view they are drawn in that camera's own frame, i.e. with both biases added back,
or, when recon.py refined the End Zone mapping per 1 s window (ez_map.json), through that frame's inverse map.
Writes PAIR_DIR/<recon-name>/overlay_<view>.mp4 and, with --stack, overlay_stack.mp4 (wide | ez on top, the
bird's-eye replay.mp4 below, 1920x1080).
"""
import argparse, json, os, subprocess

import cv2
import imageio_ffmpeg
import numpy as np

FPS = 30.0
FIELD_W = 160.0 / 3.0
TEAM_COL = {"A": (30, 30, 30), "B": (245, 245, 245)}
TEXT_COL = {"A": (255, 255, 255), "B": (20, 20, 20)}
INFERRED = (150, 150, 150)
RING = np.stack([np.cos(np.linspace(0, 2 * np.pi, 25)), np.sin(np.linspace(0, 2 * np.pi, 25))], 1) * 0.6


def project(H, pts, ref_sign):
    """field (N,2) -> image (N,2); rows behind the camera are NaN."""
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    ok = (np.sign(p[:, 2]) == ref_sign) & (np.abs(p[:, 2]) > 1e-12)
    out = np.full((len(p), 2), np.nan)
    out[ok] = p[ok, :2] / p[ok, 2:3]
    return out


def front_sign(H):
    """Homogeneous sign of points in front of the camera (the field point under the image centre)."""
    c = np.linalg.inv(H) @ np.array([960.0, 540.0, 1.0])
    return np.sign((H @ np.array([c[0] / c[2], c[1] / c[2], 1.0]))[2])


def writer(path, w, h):
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             "-crf", "22", path], stdin=subprocess.PIPE)


def polyline_clipped(img, pts, col, th):
    """Draw consecutive segments whose ends both lie near the image (far-off / near-horizon points blow up and
    would draw spikes across the frame)."""
    h, w = img.shape[:2]
    ok = np.isfinite(pts).all(1)
    ok[ok] &= (pts[ok, 0] > -0.5 * w) & (pts[ok, 0] < 1.5 * w) & (pts[ok, 1] > -0.5 * h) & (pts[ok, 1] < 1.5 * h)
    for i in range(len(pts) - 1):
        if ok[i] and ok[i + 1]:
            cv2.line(img, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)), col, th, cv2.LINE_AA)


def dashed(img, pts, col, th):
    for i in range(0, len(pts) - 1, 2):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), col, th, cv2.LINE_AA)


def view_shift(rd, v):
    """Offset from the Wide-aligned slot frame to camera v's own field frame (zero for Wide)."""
    if v != "ez":
        return np.zeros(2)
    s = json.load(open(os.path.join(rd, "recon_summary.json")))
    sh = np.array(s["bias"]["ez_minus_wide_median_yd"]) if s["bias"].get("applied", True) else np.zeros(2)
    if s.get("noise_model") and os.path.exists(s["noise_model"]):
        sh = sh + np.array(json.load(open(s["noise_model"]))["noise_model"]["e"]["bias_xy"])
    return sh


def view_map(rd, v):
    """Per-frame affine Wide-aligned field -> camera v's own field frame (ez_map.json from recon.py), or None."""
    p = os.path.join(rd, "ez_map.json")
    if v != "ez" or not os.path.exists(p):
        return None
    return {int(k): np.array(m) for k, m in json.load(open(p)).items()}


def render_view(pair, v, rows_by_f, NF, grid_t, los, width, shift=np.zeros(2), fmap=None):
    """Returns the list of overlay frames (BGR, width x width*9/16) for one view, one per recon frame."""
    cal = json.load(open(os.path.join(pair, v, "calib.json")))
    sync = json.load(open(os.path.join(pair, "sync.json")))
    frames = sorted(f for f in os.listdir(os.path.join(pair, v, "frames")) if f.endswith(".jpg"))
    rel = np.array(cal["pts_s"]) - sync[v]["snap_pts"]
    s = width / 1920.0
    hgt = int(round(1080 * s / 2) * 2)
    trail = {}   # player_id -> list of (frame, field xy)
    out = []
    def to_view(f, x, y):
        if fmap is not None and f in fmap:
            return tuple(fmap[f] @ np.array([x, y, 1.0]))
        return (x + shift[0], y + shift[1])

    for f in range(NF):
        sf = int(np.argmin(np.abs(rel - grid_t[f])))
        for r in rows_by_f.get(f, []):
            trail.setdefault(r["player_id"], []).append((f, to_view(f, r["x_yards"], r["y_yards"])))
        if abs(rel[sf] - grid_t[f]) > 0.5 / FPS:
            img = np.zeros((hgt, width, 3), np.uint8)
            cv2.putText(img, f"{v}: no frame at t = {grid_t[f]:+.2f} s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            out.append(img); continue
        H = np.array(cal["H"][sf]); sg = front_sign(H)
        img = cv2.resize(cv2.imread(os.path.join(pair, v, "frames", frames[sf])), (width, hgt), interpolation=cv2.INTER_AREA)
        Hs = np.diag([s, s, 1.0]) @ H
        if los is not None and np.isfinite(los):
            lp = project(Hs, np.array([to_view(f, los, yy) for yy in np.linspace(0, FIELD_W, 80)]), sg)
            polyline_clipped(img, lp, (0, 220, 255), 2)
        for r in rows_by_f.get(f, []):
            team, src = r["team"], r["source"]
            col = TEAM_COL[team] if src != "inferred" else INFERRED
            tr = [xy for g, xy in trail[r["player_id"]] if g > f - FPS]
            polyline_clipped(img, project(Hs, tr, sg), col, 1)
            ring = project(Hs, RING + list(to_view(f, r["x_yards"], r["y_yards"])), sg)
            if not np.isfinite(ring).all():
                continue
            ring = ring.astype(np.int32)
            if not ((ring[:, 0] >= 0) & (ring[:, 0] < width) & (ring[:, 1] >= 0) & (ring[:, 1] < hgt)).any():
                continue
            if src in ("fused", "fused_split"):
                cv2.polylines(img, [ring], True, (0, 0, 0) if team == "B" else (255, 255, 255), 4, cv2.LINE_AA)
                cv2.polylines(img, [ring], True, col, 2, cv2.LINE_AA)
            elif src == "inferred":
                cv2.polylines(img, [ring], True, col, 1, cv2.LINE_AA)
            else:
                dashed(img, ring, col, 2)
            x, y = int(ring[:, 0].mean()), int(ring[:, 1].max())
            lab = r["player_id"]
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (x - tw // 2 - 2, y + 2), (x + tw // 2 + 2, y + th + 6), col, -1)
            cv2.putText(img, lab, (x - tw // 2, y + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        TEXT_COL[team] if src != "inferred" else (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(img, f"{v}  t = {grid_t[f]:+.2f} s from snap   solid = both cameras, dashed = one camera, grey = inferred,"
                    f" yellow = LOS", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(img)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair"); ap.add_argument("--recon-name", default="recon")
    ap.add_argument("--views", nargs="+", default=["wide", "ez"]); ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--stack", action="store_true")
    ap.add_argument("--suffix", default="", help="output name suffix, e.g. _v2 -> overlay_wide_v2.mp4")
    a = ap.parse_args()
    rd = os.path.join(a.pair, a.recon_name)
    rows = [json.loads(l) for l in open(os.path.join(rd, "recon.jsonl"))]
    los = json.load(open(os.path.join(rd, "recon_summary.json"))).get("los_x_at_snap")
    NF = max(r["frame"] for r in rows) + 1
    grid_t = {}
    rows_by_f = {}
    for r in rows:
        grid_t[r["frame"]] = r["t"]; rows_by_f.setdefault(r["frame"], []).append(r)
    grid_t = [grid_t[f] for f in range(NF)]
    views = [v for v in a.views if os.path.exists(os.path.join(a.pair, v, "calib.json"))]
    vids = {}
    for v in views:
        fr = render_view(a.pair, v, rows_by_f, NF, grid_t, los, a.width, view_shift(rd, v), view_map(rd, v))
        p = writer(os.path.join(rd, f"overlay_{v}{a.suffix}.mp4"), fr[0].shape[1], fr[0].shape[0])
        for im in fr:
            p.stdin.write(im.tobytes())
        p.stdin.close(); p.wait()
        vids[v] = fr
        print(v, len(fr), "frames ->", os.path.join(rd, f"overlay_{v}{a.suffix}.mp4"))
    if a.stack:
        cap = cv2.VideoCapture(os.path.join(rd, "replay.mp4"))
        p = writer(os.path.join(rd, f"overlay_stack{a.suffix}.mp4"), 1920, 1080)
        for f in range(NF):
            canvas = np.zeros((1080, 1920, 3), np.uint8)
            for k, v in enumerate(views[:2]):
                canvas[:540, k * 960:(k + 1) * 960] = cv2.resize(vids[v][f], (960, 540), interpolation=cv2.INTER_AREA)
            ok, rep = cap.read()
            if ok:
                sc = min(540 / rep.shape[0], 1920 / rep.shape[1])
                rep = cv2.resize(rep, (int(rep.shape[1] * sc), int(rep.shape[0] * sc)), interpolation=cv2.INTER_AREA)
                x0 = (1920 - rep.shape[1]) // 2
                canvas[540:540 + rep.shape[0], x0:x0 + rep.shape[1]] = rep
            p.stdin.write(canvas.tobytes())
        p.stdin.close(); p.wait()
        print("stack ->", os.path.join(rd, f"overlay_stack{a.suffix}.mp4"))


if __name__ == "__main__":
    main()
