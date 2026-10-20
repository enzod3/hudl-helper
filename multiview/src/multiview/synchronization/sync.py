#!/usr/bin/env python
"""Stage B visual sync for one play cut in two angles (run with ~/sam3env/bin/python).

  sync.py PAIR_DIR [--views wide ez] [--preview] [--sheets]

Each PAIR_DIR/<view>/ needs frames/*.jpg, frames_pts.json (segment.py extract --fps 0) and dets_all.json
(detect.py --field-mask over all frames). Per view:
  1. camera motion: frame-to-frame homography from background corners (LK flow, person boxes masked out, RANSAC)
  2. player motion: field-masked detections at t-K are carried to t through the camera homographies and matched
     to detections at t (Hungarian); displacement is divided by box height (perspective-normalised)
  3. snap = first sustained rise of the fraction of players moving (>= HI for SUSTAIN s) out of a quiet
     stretch, backed up to the last frame where the fraction was <= LO
Then offset = snap_wide_pts - snap_ez_pts (wide_time = ez_time + offset), checked by normalised
cross-correlation of the two curves' rising edges on a common 1/120 s grid within +-1 s of the snap-based offset
(raw-curve correlation is reported too; it is nearly flat because the curves are step-like).

Writes PAIR_DIR/sync.json, PAIR_DIR/<view>/motion.json, optional PAIR_DIR/<view>/sheet_*.jpg (contact sheets
around the detected snap) and PAIR_DIR/side_by_side.mp4 (shared timeline, time from wide snap burned in).
"""
import argparse, json, os, subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from scipy.optimize import linear_sum_assignment

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SCALE = 0.5          # analysis resolution (960x540 for 1080p)
K = 3                # displacement baseline in frames (0.1 s at 30 fps)
MOVE_THR = 0.10      # player heights per K frames that counts as moving (~1 height/s)
HI, LO, SUSTAIN = 0.15, 0.06, 0.5   # onset: frac >= HI for SUSTAIN s (window mean) after a quiet stretch at <= LO


def load_view(vdir):
    pts = np.array(json.load(open(os.path.join(vdir, "frames_pts.json")))["pts_s"])
    dets = json.load(open(os.path.join(vdir, "dets_all.json")))["frames"]
    frames = sorted(f for f in os.listdir(os.path.join(vdir, "frames")) if f.endswith(".jpg"))
    assert len(frames) == len(pts) == len(dets), (len(frames), len(pts), len(dets))
    return frames, pts, dets


def camera_homographies(vdir, frames, dets):
    """H[t] maps analysis-scale pixels of frame t-1 to frame t (H[0] = I)."""
    Hs, quality = [np.eye(3)], [1.0]
    prev = None
    for t, f in enumerate(frames):
        g = cv2.imread(os.path.join(vdir, "frames", f), cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        mask = np.full(g.shape, 255, np.uint8)
        for d in dets[t]["detections"] + dets[t].get("dropped", []):
            x1, y1, x2, y2 = (np.array(d["box_xyxy"]) * SCALE).astype(int)
            cv2.rectangle(mask, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), 0, -1)
        if prev is not None:
            pg, pmask = prev
            p0 = cv2.goodFeaturesToTrack(pg, maxCorners=1000, qualityLevel=0.01, minDistance=7, mask=pmask)
            H, q = np.eye(3), 0.0
            if p0 is not None and len(p0) >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, p0, None, winSize=(21, 21), maxLevel=3)
                ok = st.ravel() == 1
                if ok.sum() >= 12:
                    Hc, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 1.5)
                    if Hc is not None:
                        H, q = Hc, float(inl.mean())
            Hs.append(H); quality.append(q)
        prev = (g, mask)
    return Hs, quality


def foot_points(frame_dets):
    b = np.array([d["box_xyxy"] for d in frame_dets["detections"]], float).reshape(-1, 4) * SCALE
    return np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1), b[:, 3] - b[:, 1]


def motion_curve(Hs, dets):
    T = len(dets)
    frac = np.full(T, np.nan); med = np.full(T, np.nan); nmatch = np.zeros(T, int)
    for t in range(K, T):
        H = np.eye(3)
        for s in range(t - K + 1, t + 1):
            H = Hs[s] @ H
        p0, h0 = foot_points(dets[t - K])
        p1, h1 = foot_points(dets[t])
        if len(p0) == 0 or len(p1) == 0:
            continue
        p0w = cv2.perspectiveTransform(p0[None].astype(np.float64), H)[0]
        cost = np.linalg.norm(p0w[:, None] - p1[None], axis=2) / h1[None]
        r, c = linear_sum_assignment(cost)
        ok = cost[r, c] < 0.6
        if ok.sum() < 3:
            continue
        disp = cost[r, c][ok]
        nmatch[t] = int(ok.sum())
        frac[t] = float((disp > MOVE_THR).mean())
        med[t] = float(np.median(disp))
    return frac, med, nmatch


def smooth(x, k=5):
    y = x.copy()
    v = np.where(np.isnan(x), 0, x)
    w = np.where(np.isnan(x), 0, 1.0)
    ker = np.ones(k)
    num = np.convolve(v, ker, "same"); den = np.convolve(w, ker, "same")
    y = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    return y


def detect_snap(pts, frac, quiet_s=0.7, follow_s=1.5):
    """Rise of the moving fraction out of a quiet pre-snap stretch.
    Trigger: window of SUSTAIN s whose mean smoothed fraction >= HI with every frame >= HI*2/3, preceded by >= quiet_s
    at <= LO. Onset: back up from the trigger to the last frame at <= LO; the motion at a frame is measured over
    [t-K, t], so the onset is the start of that window. When several onsets qualify (pre-snap shifts / motion),
    the one with the most sustained motion over the next follow_s s is taken - the snap moves everybody.
    End Zone motion is mostly along the camera axis and moves fewer players per box-height, hence the low thresholds."""
    f = np.nan_to_num(smooth(frac, 5))
    dt = float(np.median(np.diff(pts)))
    need, quiet = max(1, int(round(SUSTAIN / dt))), max(1, int(round(quiet_s / dt)))
    follow = max(1, int(round(follow_s / dt)))
    cands, t = [], quiet
    while t < len(f) - need:
        w = f[t:t + need]
        if w.mean() >= HI and w.min() >= HI * 2 / 3:
            s = t
            while s > 0 and f[s - 1] > LO:
                s -= 1
            if s >= quiet and f[max(0, s - quiet):s].max() <= HI:
                onset = max(0, s - K + 1)
                cands.append((float(f[t:t + follow].mean()), onset, t))
            # skip past this motion episode
            while t < len(f) and f[t] > LO:
                t += 1
        t += 1
    if not cands:
        return {"snap_frame": None, "snap_pts": None, "trigger_frame": None, "n_candidates": 0}
    score, onset, trig = max(cands)
    return {"snap_frame": int(onset), "snap_pts": float(pts[onset]), "trigger_frame": int(trig), "n_candidates": len(cands),
            "candidates_s": [round(float(pts[o]), 3) for _, o, _ in cands]}


def onset_signal(sig):
    """Rising edges only: positive part of the derivative of the smoothed curve. Raw motion curves are step-like
    (still -> moving) and correlate almost equally well over a wide range of lags; their rising edges do not."""
    f = np.nan_to_num(smooth(sig, 5))
    return np.clip(np.gradient(f), 0, None)


def xcorr(pts_a, sig_a, pts_b, sig_b, center, search=1.0, win=(-1.0, 1.5), snap_a=0.0, step=1 / 120, edges=True):
    """Find lag L (b_time + L = a_time) maximising NCC of the two curves inside win around snap_a."""
    if edges:
        sig_a, sig_b = onset_signal(sig_a), onset_signal(sig_b)
    else:
        sig_a, sig_b = np.nan_to_num(smooth(sig_a, 3)), np.nan_to_num(smooth(sig_b, 3))
    grid = np.arange(snap_a + win[0], snap_a + win[1], step)
    fa = np.interp(grid, pts_a, sig_a, left=np.nan, right=np.nan)
    lags = np.arange(center - search, center + search + 1e-9, step)
    scores = []
    for L in lags:
        fb = np.interp(grid - L, pts_b, sig_b, left=np.nan, right=np.nan)
        ok = ~np.isnan(fa) & ~np.isnan(fb)
        if ok.sum() < len(grid) * 0.6:
            scores.append(np.nan); continue
        a, b = fa[ok] - fa[ok].mean(), fb[ok] - fb[ok].mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        scores.append(float((a * b).sum() / den) if den > 0 else np.nan)
    scores = np.array(scores)
    if np.all(np.isnan(scores)):
        return None
    i = int(np.nanargmax(scores))
    peak = float(scores[i])
    far = np.abs(lags - lags[i]) > 0.15
    second = float(np.nanmax(scores[far])) if np.any(far & ~np.isnan(scores)) else float("nan")
    half = peak / 2
    j0 = i
    while j0 > 0 and not np.isnan(scores[j0 - 1]) and scores[j0 - 1] >= half:
        j0 -= 1
    j1 = i
    while j1 < len(scores) - 1 and not np.isnan(scores[j1 + 1]) and scores[j1 + 1] >= half:
        j1 += 1
    return {"lag_s": round(float(lags[i]), 4), "peak_ncc": round(peak, 3), "second_peak_ncc_outside_0.15s": round(second, 3),
            "fwhm_s": round(float(lags[j1] - lags[j0]), 3),
            "curve": {"lags_s": [round(float(x), 4) for x in lags[::4]], "ncc": [None if np.isnan(x) else round(float(x), 3) for x in scores[::4]]}}


def scrimmage_crop(dets, center, cw=720, ch=405, W=1920, H=1080):
    """Crop window around the densest cluster of field-masked players at the snap (the line of scrimmage)."""
    b = np.array([d["box_xyxy"] for d in dets[center]["detections"]], float).reshape(-1, 4)
    if len(b) == 0:
        return None
    c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    nb = (np.linalg.norm(c[:, None] - c[None], axis=2) < 150).sum(1)
    cx, cy = c[int(np.argmax(nb))]
    x0 = int(np.clip(cx - cw / 2, 0, W - cw)); y0 = int(np.clip(cy - ch / 2, 0, H - ch))
    return x0, y0, cw, ch


def contact_sheet(vdir, frames, pts, center, out, step=2, n=16, cols=4, w=480, crop=None):
    idx = [center + (i - n // 2) * step for i in range(n)]
    tiles = []
    for i in idx:
        if 0 <= i < len(frames):
            im = cv2.imread(os.path.join(vdir, "frames", frames[i]))
            if crop is not None:
                x0, y0, cw, ch = crop
                im = im[y0:y0 + ch, x0:x0 + cw]
            im = cv2.resize(im, (w, int(w * im.shape[0] / im.shape[1])))
            cv2.putText(im, f"f{i} {pts[i]:.3f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(im, f"f{i} {pts[i]:.3f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            im = np.zeros((int(w * 9 / 16), w, 3), np.uint8)
        tiles.append(im)
    rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def side_by_side(pair, views, data, offset, snap_wide, out, fps=30.0):
    """Shared timeline tau = wide_time - snap_wide = ez_time + offset - snap_wide."""
    va, vb = views
    pa, pb = data[va]["pts"], data[vb]["pts"]
    t0 = min(pa[0], pb[0] + offset) - snap_wide
    t1 = max(pa[-1], pb[-1] + offset) - snap_wide
    taus = np.arange(t0, t1, 1 / fps)
    W, H = 960, 540
    proc = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2 * W}x{H + 60}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out],
                            stdin=subprocess.PIPE)
    for tau in taus:
        canvas = np.zeros((H + 60, 2 * W, 3), np.uint8)
        labels = []
        for k, (v, pts, shift) in enumerate([(va, pa, 0.0), (vb, pb, offset)]):
            t_src = tau + snap_wide - shift
            i = int(np.argmin(np.abs(pts - t_src)))
            if abs(pts[i] - t_src) <= 0.5 / fps + 1e-6:
                im = cv2.imread(os.path.join(pair, v, "frames", data[v]["frames"][i]))
                canvas[:H, k * W:(k + 1) * W] = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
                labels.append(f"{v}: f{i} pts {pts[i]:.3f}s")
            else:
                labels.append(f"{v}: (no frame)")
        cv2.putText(canvas, f"t = {tau:+.3f} s from snap", (12, H + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, labels[0], (520, H + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        cv2.putText(canvas, labels[1], (1300, H + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2)
        if abs(tau) < 0.5 / fps:
            cv2.rectangle(canvas, (0, 0), (2 * W - 1, H - 1), (0, 0, 255), 6)
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close(); proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair")
    ap.add_argument("--views", nargs="+", default=["wide", "ez"], help="one view = single-view mode (snap only)")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--sheets", action="store_true")
    args = ap.parse_args()
    data, res = {}, {"pair": os.path.basename(os.path.abspath(args.pair)), "params": {
        "scale": SCALE, "K_frames": K, "move_thr_heights": MOVE_THR, "hi": HI, "lo": LO, "sustain_s": SUSTAIN}}
    for v in args.views:
        vdir = os.path.join(args.pair, v)
        frames, pts, dets = load_view(vdir)
        Hs, q = camera_homographies(vdir, frames, dets)
        frac, med, nmatch = motion_curve(Hs, dets)
        snap = detect_snap(pts, frac)
        cam_shift = [float(np.hypot(H[0, 2], H[1, 2]) / SCALE) for H in Hs]
        data[v] = {"frames": frames, "pts": pts, "frac": frac, "med": med, "snap": snap}
        json.dump({"pts_s": pts.tolist(), "frac_moving": [None if np.isnan(x) else round(float(x), 4) for x in frac],
                   "median_disp_heights": [None if np.isnan(x) else round(float(x), 4) for x in med],
                   "matched": nmatch.tolist(), "cam_inlier_ratio": [round(x, 3) for x in q],
                   "cam_translation_px": [round(x, 2) for x in cam_shift], **snap},
                  open(os.path.join(vdir, "motion.json"), "w"))
        res[v] = {**snap, "num_frames": len(frames), "median_players_matched": int(np.median(nmatch[K:])),
                  "cam_inlier_ratio_median": round(float(np.median(q[1:])), 3),
                  "cam_translation_px_max": round(float(np.max(cam_shift)), 1),
                  "persons_per_frame_median": int(np.median([len(d["detections"]) for d in dets]))}
        if args.sheets and snap["snap_frame"] is not None:
            contact_sheet(vdir, frames, pts, snap["snap_frame"], os.path.join(vdir, "sheet_snap.jpg"))
            contact_sheet(vdir, frames, pts, snap["snap_frame"], os.path.join(vdir, "sheet_snap_crop.jpg"), step=1,
                          n=24, cols=6, w=400, crop=scrimmage_crop(dets, snap["snap_frame"]))
            contact_sheet(vdir, frames, pts, snap["snap_frame"], os.path.join(vdir, "sheet_snap_zoom.jpg"), step=1,
                          n=24, cols=6, w=400, crop=scrimmage_crop(dets, snap["snap_frame"], cw=400, ch=225))
            contact_sheet(vdir, frames, pts, len(frames) // 2, os.path.join(vdir, "sheet_overview.jpg"),
                          step=max(1, len(frames) // 16))
    va, vb = (args.views + [None])[:2]
    sa = data[va]["snap"]["snap_pts"]
    sb = data[vb]["snap"]["snap_pts"] if vb else None
    if vb is None:
        res["single_view"] = va
        res["ez"] = {"snap_frame": None, "snap_pts": None}
    if sa is not None and sb is not None:
        off = sa - sb
        res["offset_snap_s"] = round(off, 4)
        xc = xcorr(data[va]["pts"], data[va]["frac"], data[vb]["pts"], data[vb]["frac"], off, snap_a=sa)
        xm = xcorr(data[va]["pts"], data[va]["med"], data[vb]["pts"], data[vb]["med"], off, snap_a=sa)
        res["xcorr_frac_moving"] = xc
        res["xcorr_median_disp"] = xm
        res["xcorr_frac_moving_raw_curve"] = xcorr(data[va]["pts"], data[va]["frac"], data[vb]["pts"], data[vb]["frac"],
                                                   off, snap_a=sa, win=(-1.5, 2.5), edges=False)
        res["offset_definition"] = f"{va}_time = {vb}_time + offset"
        if args.preview:
            side_by_side(args.pair, args.views, data, off, sa, os.path.join(args.pair, "side_by_side.mp4"))
    json.dump(res, open(os.path.join(args.pair, "sync.json"), "w"), indent=1)
    xk = ("xcorr_frac_moving", "xcorr_median_disp", "xcorr_frac_moving_raw_curve")
    brief = {k: v for k, v in res.items() if k not in xk + ("params",)}
    for k in xk:
        if res.get(k):
            brief[k] = {kk: vv for kk, vv in res[k].items() if kk != "curve"}
    print(json.dumps(brief))


if __name__ == "__main__":
    main()
