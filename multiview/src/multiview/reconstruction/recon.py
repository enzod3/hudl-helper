#!/usr/bin/env python
"""End-to-end single-play reconstruction: 22 global player slots in field yards (sam3env python).

  recon.py PAIR_DIR [--suffix _crop] [--play-id ID] [--no-bias-fix] [--teams-only]

Inputs per view V in (wide, ez): PAIR_DIR/V/calib.json (H(t), field->image), PAIR_DIR/track_V<suffix>/
(masks.json, frames.json, frames/), PAIR_DIR/sync.json.

Stages
 1 observations: per SAM mask, foot point (bottom-centre of the lowest 10% of the mask) -> yards through H(t)^-1;
   covariance = J diag(su^2, sv^2) J^T with J the image->field Jacobian at the foot (su = 2 px + 0.10 box width,
   sv = 3 px + 0.06 box height), so each camera's depth axis gets the larger error; later scaled by one global
   factor estimated from cross-view residuals (chi-square calibration).
 2 cross-view bias: robust translation of the End Zone positions onto the Wide frame, from isolated players
   (no other player within 3 yd in either view, Hungarian pairs < 3 yd).
 3 teams: torso pixels (20-55% of the mask height) -> fraction dark / white / mid + stripe transitions; officials =
   striped; remaining tracks split by dark-minus-white into A (dark jerseys) and B (light). A/B are defined by
   jersey brightness, so they agree across views and plays of the same game.
 4 per-view tracklets: SAM ids split at gaps > 8 frames or speed > 18 yd/s (candidate identity switches); linked
   forward in time when a constant-velocity prediction lands near the next tracklet start (same team);
   constant-velocity Kalman + RTS smoother in yards; frames without a mask are kept as 'occluded' predictions.
 5 cross-view association: tracklet-to-tracklet costs (median Mahalanobis distance over shared observed frames,
   team must agree), greedy merge into entities with no in-view time overlap.
 6 slots: officials dropped; per team the 11 entities with most on-field observations -> A_01..A_11 / B_01..B_11
   (numbered by lateral position at the snap). Per frame: inverse-covariance fusion of the views (single-view
   covariance x2), Kalman + RTS over the clip, frames without any observation -> 'inferred'.
Outputs: PAIR_DIR/recon/recon.jsonl, recon_summary.json, teams.json, replay.mp4.
"""
import argparse, json, os, subprocess, sys

import cv2
import imageio_ffmpeg
import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

FIELD_LEN, FIELD_W = 120.0, 160.0 / 3.0
FPS = 30.0
VIEW_NAME = {"wide": "sideline", "ez": "endzone"}


# ---------------------------------------------------------------- observations
def foot_point(m, box):
    ys, xs = np.nonzero(m)
    if len(ys) < 15:
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, float(y2), "box"
    top, bot = ys.min(), ys.max()
    sel = ys >= bot - 0.10 * (bot - top + 1)
    return float(xs[sel].mean()), float(bot), "mask"


def img_to_field(Hinv, u, v):
    p = Hinv @ np.array([u, v, 1.0])
    return p[:2] / p[2]


def jacobian(Hinv, u, v, e=1.0):
    f0 = img_to_field(Hinv, u, v)
    return np.stack([(img_to_field(Hinv, u + e, v) - f0) / e, (img_to_field(Hinv, u, v + e) - f0) / e], 1)


def torso_features(img, m):
    ys, xs = np.nonzero(m)
    if len(ys) < 40:
        return None
    top, bot = ys.min(), ys.max()
    h = bot - top + 1
    sel = (ys >= top + 0.20 * h) & (ys <= top + 0.55 * h)
    if sel.sum() < 20:
        return None
    lab = cv2.cvtColor(img[ys[sel], xs[sel]].reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    L = lab[:, 0]
    dark, white = float((L < 80).mean()), float((L > 175).mean())
    # stripe score: dark<->white alternations along torso rows (officials' stripes)
    rows = {}
    for y, x, l in zip(ys[sel], xs[sel], L):
        rows.setdefault(y, []).append((x, l))
    trans, n = 0, 0
    for r in rows.values():
        if len(r) < 6:
            continue
        r.sort()
        cls = [0 if l < 80 else (2 if l > 175 else 1) for _, l in r]
        c2 = [c for c in cls if c != 1]
        trans += sum(1 for a, b in zip(c2[:-1], c2[1:]) if a != b); n += 1
    stripes = trans / max(1, n)
    return {"dark": dark, "white": white, "L": float(np.median(L)), "a": float(np.median(lab[:, 1])),
            "b": float(np.median(lab[:, 2])), "stripes": float(stripes)}


FOOT_HEAD = None   # (model, device) when --foot-head is given
DET_GATE = 0       # --det-gate N: drop SAM observations in runs of >= N frames that no detector box confirms


def load_foot_head(path):
    import torch
    sys.path[:0] = __import__("glob").glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "*", ""))  # all src/multiview/* module dirs (flat sibling imports)
    from train_foot import Head
    # CPU by default: the head is tiny, and many recon processes on the GPU next to SAM runs exhaust its memory
    dev = "cuda" if (os.environ.get("FOOT_HEAD_GPU") == "1" and torch.cuda.is_available()) else "cpu"
    torch.set_num_threads(4)
    m = Head().to(dev); m.load_state_dict(torch.load(path, map_location=dev)); m.eval()
    return m, dev


def head_feet(img, boxes):
    """Foot point (px) and sigma (px, u/v) per box from the ground-contact head (foot_crops convention)."""
    import torch
    from foot_crops import crop, from_crop_norm, crop_box
    m, dev = FOOT_HEAD
    X = np.stack([crop(img, b) for b in boxes])
    with torch.no_grad():
        x = torch.from_numpy(X).to(dev).permute(0, 3, 1, 2).float() / 255.0
        mu, ls = m(x)
    mu, sg = mu.cpu().numpy(), ls.exp().cpu().numpy()
    out = []
    for b, uv, s in zip(boxes, mu, sg):
        X1, Y1, X2, Y2 = crop_box(b)
        fx, fy = from_crop_norm(b, uv)
        out.append((float(fx), float(fy), float(s[0] * (X2 - X1)), float(s[1] * (Y2 - Y1))))
    return out


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def iou_track(frames_boxes, thr=0.3, max_miss=5):
    """Greedy IoU tracker over per-frame box lists -> list (per frame) of track ids."""
    tracks, nxt, out = {}, 0, []
    for boxes in frames_boxes:
        ids = [-1] * len(boxes)
        cand = sorted(((box_iou(t["box"], b), tid, j) for tid, t in tracks.items() for j, b in enumerate(boxes)), reverse=True)
        used_t, used_b = set(), set()
        for v, tid, j in cand:
            if v < thr:
                break
            if tid in used_t or j in used_b:
                continue
            ids[j] = tid; used_t.add(tid); used_b.add(j); tracks[tid].update(box=boxes[j], miss=0)
        for j, b in enumerate(boxes):
            if ids[j] < 0:
                ids[j] = nxt; tracks[nxt] = {"box": b, "miss": 0}; nxt += 1
        for tid in list(tracks):
            if tid not in used_t and tid not in ids:
                tracks[tid]["miss"] += 1
                if tracks[tid]["miss"] > max_miss:
                    del tracks[tid]
        out.append(ids)
    return out


def load_view_det(pair, v, suffix, sam_obs=None, bt_suffix=None):
    """Observations from per-frame RF-DETR detections (field-masked dets_all) over the SAM tracking window, linked
    by an IoU tracker (ids offset by 2000). Feet from the ground-contact head when loaded, else box bottom-centre;
    team colour from a box-interior torso rectangle. With sam_obs (adaptive mode): a detection overlapping another
    detection (IoU >= 0.1) is replaced by the SAM observation of that frame whose box matches it (IoU >= 0.4).
    With bt_suffix: boxes and ids come from track_det.py (track_<view><bt_suffix>/tracks.json, ByteTrack-style)."""
    sync = json.load(open(os.path.join(pair, "sync.json")))
    cal = json.load(open(os.path.join(pair, v, "calib.json")))
    tdir = os.path.join(pair, f"track_{v}{suffix}")
    meta = json.load(open(os.path.join(tdir, "frames.json")))
    dets = json.load(open(os.path.join(pair, v, "dets_all.json")))["frames"]
    frames = sorted(os.listdir(os.path.join(tdir, "frames")))
    a0, n = meta["source_frame_start"], meta["num_frames"]
    snap, pts = sync[v]["snap_pts"], cal["pts_s"]
    if bt_suffix:
        bt = json.load(open(os.path.join(pair, f"track_{v}{bt_suffix}", "tracks.json")))["frames"]
        boxes_pf = [[o["box_xyxy"] for o in bt[k]["objects"]] if k < len(bt) else [] for k in range(n)]
        ids_pf = [[o["id"] for o in bt[k]["objects"]] if k < len(bt) else [] for k in range(n)]
    else:
        boxes_pf = [[d["box_xyxy"] for d in dets[a0 + k]["detections"]] for k in range(n)]
        ids_pf = iou_track(boxes_pf)
    sam_by_k = {}
    for o in (sam_obs or []):
        sam_by_k.setdefault(o["k"], []).append(o)
    obs, n_sam_used, n_crowded = [], 0, 0
    for k in range(n):
        sf = a0 + k
        H = np.array(cal["H"][sf]); Hinv = np.linalg.inv(H)
        boxes = boxes_pf[k]
        if not boxes:
            continue
        img = cv2.imread(os.path.join(tdir, "frames", frames[k]))
        hf = head_feet(img, boxes) if FOOT_HEAD else None
        for j, b in enumerate(boxes):
            crowded = any(box_iou(b, b2) >= 0.1 for jj, b2 in enumerate(boxes) if jj != j)
            n_crowded += int(crowded)
            if crowded and sam_obs is not None:
                best = max(sam_by_k.get(k, []), key=lambda o: box_iou(o["box"], b), default=None)
                if best is not None and box_iou(best["box"], b) >= 0.4:
                    obs.append({**best, "sam_id": int(best["sam_id"])}); n_sam_used += 1
                    continue
            x1, y1, x2, y2 = b
            if hf is not None:
                fu, fv, su, sv = hf[j]; su, sv = max(su, 1.5), max(sv, 1.5); src = "head"
            else:
                fu, fv, su, sv, src = (x1 + x2) / 2, y2, 2.0 + 0.10 * (x2 - x1), 3.0 + 0.06 * (y2 - y1), "box"
            m = np.zeros(img.shape[:2], bool)
            m[int(y1 + 0.15 * (y2 - y1)):int(y2), int(x1 + 0.2 * (x2 - x1)):int(x2 - 0.2 * (x2 - x1))] = True
            p_ = img_to_field(Hinv, fu, fv); J = jacobian(Hinv, fu, fv)
            obs.append({"view": v, "sam_id": 2000 + ids_pf[k][j], "k": k, "sf": sf, "tau": round(pts[sf] - snap, 4),
                        "xy": p_, "cov": J @ np.diag([su * su, sv * sv]) @ J.T, "box": [x1, y1, x2, y2],
                        "foot": (fu, fv), "foot_src": src, "col": torso_features(img, m), "calib_res": cal["info"][sf]["residual_px"]})
    return obs, cal, meta, {"crowded_detections": n_crowded, "sam_observations_used": n_sam_used,
                            "detections": sum(len(b) for b in boxes_pf)}


def load_view(pair, v, suffix):
    sync = json.load(open(os.path.join(pair, "sync.json")))
    cal = json.load(open(os.path.join(pair, v, "calib.json")))
    tdir = os.path.join(pair, f"track_{v}{suffix}")
    meta = json.load(open(os.path.join(tdir, "frames.json")))
    masks = json.load(open(os.path.join(tdir, "masks.json")))
    frames = sorted(os.listdir(os.path.join(tdir, "frames")))
    a0 = meta["source_frame_start"]
    snap = sync[v]["snap_pts"]
    pts = cal["pts_s"]
    obs = []
    for fr in masks["frames"]:
        k = fr["frame_idx"]
        sf = a0 + k
        H = np.array(cal["H"][sf]); Hinv = np.linalg.inv(H)
        img = cv2.imread(os.path.join(tdir, "frames", frames[k]))
        hf = head_feet(img, [o["box_xyxy"] for o in fr["objects"]]) if (FOOT_HEAD and fr["objects"]) else None
        for oi, o in enumerate(fr["objects"]):
            m = mask_utils.decode(o["rle"]).astype(bool)
            x1, y1, x2, y2 = o["box_xyxy"]
            if hf is not None:
                fu, fv, su, sv = hf[oi]
                su, sv, src = max(su, 1.5), max(sv, 1.5), "head"
            else:
                fu, fv, src = foot_point(m, o["box_xyxy"])
                su, sv = 2.0 + 0.10 * (x2 - x1), 3.0 + 0.06 * (y2 - y1)
            p = img_to_field(Hinv, fu, fv)
            J = jacobian(Hinv, fu, fv)
            C = J @ np.diag([su * su, sv * sv]) @ J.T
            obs.append({"view": v, "sam_id": int(o["id"]), "k": k, "sf": sf, "tau": round(pts[sf] - snap, 4),
                        "xy": p, "cov": C, "box": [x1, y1, x2, y2], "foot": (fu, fv), "foot_src": src,
                        "col": torso_features(img, m), "calib_res": cal["info"][sf]["residual_px"]})
    if DET_GATE > 0:
        obs = det_gate(obs, json.load(open(os.path.join(pair, v, "dets_all.json")))["frames"], DET_GATE)
    return obs, cal, meta


def det_gate(obs, dets, n_min, iou_min=0.3):
    """Detector-gated SAM: an observation is confirmed when some RF-DETR box (kept or field-mask dropped) of that frame
    overlaps its mask box (IoU >= iou_min). Runs of >= n_min consecutive unconfirmed observations of one SAM id are
    removed (mask drifted onto turf / another object / merged); shorter gaps (occlusion) are kept."""
    by = {}
    for o in obs:
        by.setdefault(o["sam_id"], []).append(o)
    out = []
    for sid, rows in by.items():
        rows.sort(key=lambda o: o["k"])
        ok = []
        for o in rows:
            D = [d["box_xyxy"] for d in dets[o["sf"]]["detections"] + dets[o["sf"]].get("dropped", [])]
            ok.append(max([box_iou(o["box"], b) for b in D], default=0.0) >= iou_min)
        i = 0
        while i < len(rows):
            if ok[i]:
                out.append(rows[i]); i += 1; continue
            j = i
            while j < len(rows) and not ok[j] and rows[j]["k"] - rows[i]["k"] == j - i:
                j += 1
            if j - i < n_min:
                out += rows[i:j]
            i = j
    return out



# ---------------------------------------------------------------- teams
def classify_teams(obs_by_view):
    tracks = {}
    for v, obs in obs_by_view.items():
        for o in obs:
            if o["col"] is not None:
                tracks.setdefault((v, o["sam_id"]), []).append(o["col"])
    feats = {}
    for key, cols in tracks.items():
        feats[key] = {f: float(np.median([c[f] for c in cols])) for f in cols[0]}
        feats[key]["n"] = len(cols)
    labels = {}
    keys = list(feats)
    # officials: striped (many dark<->white alternations) with substantial dark and white fractions
    for k in keys:
        f = feats[k]
        if f["stripes"] >= 2.0 and f["dark"] > 0.15 and f["white"] > 0.12:
            labels[k] = "official"
    rest = [k for k in keys if k not in labels]
    s = np.array([feats[k]["dark"] - feats[k]["white"] for k in rest])
    if len(rest) >= 2:
        # 2-means on dark-minus-white
        c = np.array([s.min(), s.max()])
        for _ in range(20):
            asg = np.abs(s[:, None] - c[None]).argmin(1)
            c = np.array([s[asg == j].mean() if (asg == j).any() else c[j] for j in range(2)])
        hi = int(np.argmax(c))
        for k, a_ in zip(rest, asg):
            margin = abs(s[rest.index(k)] - c[a_]) < abs(s[rest.index(k)] - c[1 - a_]) - 0.05
            labels[k] = ("A" if a_ == hi else "B") if margin else ("A?" if a_ == hi else "B?")
    return labels, feats


# ---------------------------------------------------------------- Kalman (constant velocity, yards)
def kalman_smooth(frames, meas, covs, q_acc=6.0, dt=1 / FPS):
    """frames: sorted int frame indices covering the span (all of them); meas/covs: dict frame -> z (2,), R (2,2).
    Returns dict frame -> (x(2,), P(2,2), observed bool)."""
    F = np.eye(4); F[0, 2] = F[1, 3] = dt
    G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
    Q = G @ G.T * q_acc ** 2
    Hm = np.zeros((2, 4)); Hm[0, 0] = Hm[1, 1] = 1
    f0 = frames[0]
    z0 = meas[f0] if f0 in meas else next(meas[f] for f in frames if f in meas)
    x = np.array([z0[0], z0[1], 0, 0.0]); P = np.diag([4.0, 4.0, 25.0, 25.0])
    xs, Ps, xps, Pps = [], [], [], []
    for i, f in enumerate(frames):
        if i > 0:
            x = F @ x; P = F @ P @ F.T + Q
        xps.append(x.copy()); Pps.append(P.copy())
        if f in meas:
            R = covs[f]
            S = Hm @ P @ Hm.T + R
            K = P @ Hm.T @ np.linalg.inv(S)
            x = x + K @ (meas[f] - Hm @ x); P = (np.eye(4) - K @ Hm) @ P
        xs.append(x.copy()); Ps.append(P.copy())
    for i in range(len(frames) - 2, -1, -1):   # RTS
        A = Ps[i] @ F.T @ np.linalg.inv(Pps[i + 1])
        xs[i] = xs[i] + A @ (xs[i + 1] - xps[i + 1])
        Ps[i] = Ps[i] + A @ (Ps[i + 1] - Pps[i + 1]) @ A.T
    return {f: (xs[i][:2], Ps[i][:2, :2], f in meas, xs[i][2:]) for i, f in enumerate(frames)}


# ---------------------------------------------------------------- per-view tracklets
def build_tracklets(obs, frame_of, labels, view, max_gap=8, max_speed=0.35, win=3):
    """Split a SAM id at gaps > max_gap frames, or where the median position of the next `win` observations is
    farther from the median of the previous `win` than a running player could go (max_speed yd/frame, ~10 yd/s,
    plus 1.5 yd for foot-point noise) - a persistent jump, i.e. a candidate identity switch."""
    by_id = {}
    for o in obs:
        by_id.setdefault(o["sam_id"], []).append(o)
    tracklets, switches = [], []
    outliers = 0
    for sid, rows in list(by_id.items()):
        rows.sort(key=lambda o: o["tau"])
        # short mask excursions (a mask briefly leaking onto another player / merging two players): drop
        # observations > 2.5 yd from the running median of +-3 observations
        P = np.array([r["xy"] for r in rows])
        keep = []
        for i in range(len(rows)):
            med = np.median(P[max(0, i - 3):i + 4], 0)
            keep.append(np.linalg.norm(P[i] - med) <= 2.5)
        outliers += int(len(rows) - sum(keep))
        rows = [r for r, kk in zip(rows, keep) if kk]
        by_id[sid] = rows
    for sid, rows in by_id.items():
        if not rows:
            continue
        cur = [rows[0]]
        for n in range(1, len(rows)):
            o = rows[n]
            gap = frame_of(o["tau"]) - frame_of(cur[-1]["tau"])
            if gap > max_gap:
                tracklets.append(cur); cur = [o]; continue
            if len(cur) >= win and n + win <= len(rows):
                prev = np.median([r["xy"] for r in cur[-win:]], 0)
                nxt = np.median([r["xy"] for r in rows[n:n + win]], 0)
                span = frame_of(rows[n + win - 1]["tau"]) - frame_of(cur[-win]["tau"])
                d = float(np.linalg.norm(nxt - prev))
                sd = float(np.sqrt(np.linalg.eigvalsh(np.median([r["cov"] for r in cur[-win:]], 0)).max()))
                if d > 1.5 + 3.0 * sd + max_speed * span:
                    switches.append({"view": view, "sam_id": sid, "tau": o["tau"], "jump_yd": round(d, 2), "span_frames": span})
                    tracklets.append(cur); cur = [o]; continue
            cur.append(o)
        tracklets.append(cur)
    switches.append({"outlier_observations_dropped": outliers})
    out = []
    for i, rows in enumerate(tracklets):
        team = labels.get((view, rows[0]["sam_id"]), "unknown")
        out.append({"view": view, "sam_ids": [rows[0]["sam_id"]], "obs": {frame_of(o["tau"]): o for o in rows}, "team": team})
    return out, switches


def team_compatible(a, b):
    a, b = a.rstrip("?"), b.rstrip("?")
    if "official" in (a, b):
        return a == b
    return a == b or "unknown" in (a, b)


def link_tracklets(tracklets, max_gap_frames=30):
    """Greedy forward linking within a view: end of a -> start of b."""
    def ends(t):
        fs = sorted(t["obs"]); return fs[0], fs[-1]
    cands = []
    for i, a in enumerate(tracklets):
        a0, a1 = ends(a)
        fa = sorted(a["obs"])[-6:]
        pa = np.array([a["obs"][f]["xy"] for f in fa])
        vel = (pa[-1] - pa[0]) / max(1, fa[-1] - fa[0]) if len(fa) > 1 else np.zeros(2)
        for j, b in enumerate(tracklets):
            if i == j or not team_compatible(a["team"], b["team"]):
                continue
            b0, b1 = ends(b)
            gap = b0 - a1
            if gap < 1 or gap > max_gap_frames:
                continue
            pred = pa[-1] + vel * gap
            d = float(np.linalg.norm(b["obs"][b0]["xy"] - pred))
            if d < 1.5 + 0.08 * gap:
                cands.append((d / (1.5 + 0.08 * gap), i, j))
    cands.sort()
    nxt, prv = {}, {}
    for c, i, j in cands:
        if i in nxt or j in prv:
            continue
        nxt[i] = j; prv[j] = i
    chains, seen = [], set()
    for i in range(len(tracklets)):
        if i in prv:
            continue
        ch, k = [], i
        while k is not None and k not in seen:
            ch.append(k); seen.add(k); k = nxt.get(k)
        chains.append(ch)
    merged = []
    for ch in chains:
        obs, sids, teams = {}, [], []
        for k in ch:
            obs.update(tracklets[k]["obs"]); sids += tracklets[k]["sam_ids"]; teams.append(tracklets[k]["team"])
        team = max(set(teams), key=lambda t: sum(len(tracklets[k]["obs"]) for k in ch if tracklets[k]["team"] == t))
        merged.append({"view": tracklets[ch[0]]["view"], "sam_ids": sids, "obs": obs, "team": team, "n_links": len(ch) - 1})
    return merged, len(chains) and sum(len(c) - 1 for c in chains)


# ---------------------------------------------------------------- association
def associate(tw, te, cov_scale, gate_yd=2.0, min_common=8):
    cands = []
    for i, a in enumerate(tw):
        for j, b in enumerate(te):
            if not team_compatible(a["team"], b["team"]):
                continue
            common = sorted(set(a["obs"]) & set(b["obs"]))
            if len(common) < min_common:
                continue
            d = np.array([np.linalg.norm(a["obs"][f]["xy"] - b["obs"][f]["xy"]) for f in common])
            mah = []
            for f in common:
                r = a["obs"][f]["xy"] - b["obs"][f]["xy"]
                S = (a["obs"][f]["cov"] + b["obs"][f]["cov"]) * cov_scale
                mah.append(float(np.sqrt(r @ np.linalg.solve(S, r))))
            if np.median(d) < gate_yd:
                cands.append((float(np.median(mah)), float(np.median(d)), len(common), i, j))
    cands.sort()
    ent_of = {("w", i): i for i in range(len(tw))}
    ent_of.update({("e", j): len(tw) + j for j in range(len(te))})
    members = {e: [k] for k, e in ent_of.items()}

    def frames_of(key):
        return set((tw if key[0] == "w" else te)[key[1]]["obs"])

    accepted = []
    for m, d, n, i, j in cands:
        ea, eb = ent_of[("w", i)], ent_of[("e", j)]
        if ea == eb:
            continue
        mem = members[ea] + members[eb]
        ok = True
        for side in ("w", "e"):
            ks = [k for k in mem if k[0] == side]
            for x in range(len(ks)):
                for y in range(x + 1, len(ks)):
                    if frames_of(ks[x]) & frames_of(ks[y]):
                        ok = False
        if not ok:
            continue
        for k in members[eb]:
            ent_of[k] = ea
        members[ea] = mem; del members[eb]
        accepted.append({"wide_track": i, "ez_track": j, "median_mahal": round(m, 2), "median_yd": round(d, 2), "common": n})
    return members, accepted



# ---------------------------------------------------------------- segment-level association (window graph)
def merge_second_pass(tracks, max_med_yd=1.0, min_overlap=5):
    """Second-pass tracks (sam_id >= 1000) that overlap a first-pass track of the same view (>= min_overlap
    frames, median distance <= max_med_yd) are the same player: their observations fill the first-pass track's
    gaps (first-pass observations win where both exist). Others stay as new tracks."""
    p1 = [t for t in tracks if min(t["sam_ids"]) < 1000]
    p2 = [t for t in tracks if min(t["sam_ids"]) >= 1000]
    out, n = list(p1), 0
    for t in p2:
        best, bd = None, 1e9
        for u in p1:
            common = set(t["obs"]) & set(u["obs"])
            if len(common) < min_overlap:
                continue
            d = float(np.median([np.linalg.norm(t["obs"][f]["xy"] - u["obs"][f]["xy"]) for f in common]))
            if d < bd:
                best, bd = u, d
        if best is not None and bd <= max_med_yd:
            for f, o in t["obs"].items():
                best["obs"].setdefault(f, o)
            best["sam_ids"] = best["sam_ids"] + t["sam_ids"]
            n += 1
        else:
            out.append(t)
    return out, n


def window_associate(TR, NF, gate_yd=2.5, win=15, min_obs=3, cont_cost=0.3, vel_w=0.05, box_support=None, box_min=0.6):
    """Nodes = (view, track, window) segments with >= min_obs observations in a 0.5 s window. Links: within-view
    continuity (same track, consecutive windows, cost cont_cost) and cross-view matches found per window by Hungarian
    on (median distance over shared observed frames / gate + vel_w * |velocity difference| yd/s), team-compatible,
    gated at gate_yd. Links are added cheapest first (Kruskal-style) unless the merged component would hold two
    different tracks of the same view in the same window. Components = entities.
    box_support(wide_track, ez_track, frames) -> fraction of frames where the Wide position, projected into the End
    Zone image, falls on the End Zone track's own box: pairs beyond the yard gate are still linked when that fraction is
    >= box_min (End Zone foot points in a crowd can be yards off along the field while the box is the right player)."""
    nodes = {}
    for v in ("wide", "ez"):
        for ti, t in enumerate(TR[v]):
            for w in range(0, (NF + win - 1) // win):
                fs = [f for f in range(w * win, min(NF, (w + 1) * win)) if f in t["obs"]]
                if len(fs) >= min_obs:
                    nodes[(v, ti, w)] = fs
    links = []
    for (v, ti, w), fs in nodes.items():
        if (v, ti, w + 1) in nodes:
            links.append((cont_cost, (v, ti, w), (v, ti, w + 1), "cont"))

    def vel(t, fs):
        if len(fs) < 2:
            return np.zeros(2)
        p = np.array([t["obs"][f]["xy"] for f in fs]); tt = np.array(fs) / FPS
        A = np.c_[tt, np.ones(len(tt))]
        return np.linalg.lstsq(A, p, rcond=None)[0][0]

    n_win = (NF + win - 1) // win
    cross_found = box_links = 0
    for w in range(n_win):
        W = [(ti, fs) for (v, ti, ww), fs in nodes.items() if v == "wide" and ww == w]
        E = [(ti, fs) for (v, ti, ww), fs in nodes.items() if v == "ez" and ww == w]
        if not W or not E:
            continue
        C = np.full((len(W), len(E)), 1e6)
        for i, (wi, fw) in enumerate(W):
            tw = TR["wide"][wi]
            for j, (ej, fe) in enumerate(E):
                te = TR["ez"][ej]
                if not team_compatible(tw["team"], te["team"]):
                    continue
                common = sorted(set(fw) & set(fe))
                if len(common) < min_obs:
                    continue
                d = float(np.median([np.linalg.norm(tw["obs"][f]["xy"] - te["obs"][f]["xy"]) for f in common]))
                dv = float(np.linalg.norm(vel(tw, common) - vel(te, common)))
                if d <= gate_yd:
                    C[i, j] = d / gate_yd + vel_w * dv
                elif box_support is not None:
                    s_ = box_support(tw, te, common)
                    if s_ >= box_min:
                        C[i, j] = 1.0 + (1.0 - s_) + vel_w * dv
                        box_links += 1
        r, c = linear_sum_assignment(C)
        for i, j in zip(r, c):
            if C[i, j] < 1e5:
                links.append((float(C[i, j]), ("wide", W[i][0], w), ("ez", E[j][0], w), "cross"))
                cross_found += 1
    links.sort(key=lambda x: x[0])
    comp = {n: n for n in nodes}
    slots = {n: {(n[0], n[2]): n[1]} for n in nodes}   # root -> {(view, window): track}
    members = {n: [n] for n in nodes}

    def find(n):
        while comp[n] != n:
            comp[n] = comp[comp[n]]; n = comp[n]
        return n
    accepted = {"cont": 0, "cross": 0, "rejected_conflict": 0}
    for cost, a, b, kind in links:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        sa, sb = slots[ra], slots[rb]
        if any(k in sa and sa[k] != tr for k, tr in sb.items()):
            accepted["rejected_conflict"] += 1
            continue
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra; sa, sb = sb, sa
        comp[rb] = ra; sa.update(sb); members[ra] += members[rb]
        del slots[rb], members[rb]
        accepted[kind] += 1
    ents = []
    for root, mem in members.items():
        obs = {}
        for (v, ti, w) in mem:
            for f in nodes[(v, ti, w)]:
                obs.setdefault(f, []).append((v, TR[v][ti]["obs"][f], ti))
        teams = {}
        for (v, ti, w) in mem:
            tm = TR[v][ti]["team"].rstrip("?")
            teams[tm] = teams.get(tm, 0) + len(nodes[(v, ti, w)])
        ents.append({"obs": obs, "team": max(teams, key=teams.get), "segments": len(mem)})
    # observations of tracks that fell in windows with < min_obs: attach to the entity holding that track's
    # neighbouring window, if any, so nothing observed is lost
    owner = {}
    for ei, e in enumerate(ents):
        for f, lst in e["obs"].items():
            for v, o, ti in lst:
                owner[(v, ti, f // win)] = ei
    for v in ("wide", "ez"):
        for ti, t in enumerate(TR[v]):
            for f, o in t["obs"].items():
                w = f // win
                if (v, ti, w) in nodes:
                    continue
                ei = owner.get((v, ti, w - 1), owner.get((v, ti, w + 1)))
                if ei is not None:
                    ents[ei]["obs"].setdefault(f, []).append((v, o, ti))
    return ents, {"nodes": len(nodes), "cross_candidates": cross_found, "cross_by_box": box_links, **accepted}


def under_box(box, u, v):
    """Image point (u, v) under the player in box: inside the box widened by 25% of its width, from 40% of its height
    down to 25% of its height below its bottom, or within 12 px of its bottom-centre."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return bool((x1 - 0.25 * w <= u <= x2 + 0.25 * w and y1 + 0.4 * h <= v <= y2 + 0.25 * h)
                or np.hypot(u - (x1 + x2) / 2, v - y2) <= 12)


def entity_views(e):
    return {v for lst in e["obs"].values() for v, _, _ in lst}


def rematch_single_view(ents, on_box, min_frac=0.6, min_common=10):
    """A single-camera entity is the same player as a same-team entity carrying the other camera when, on their
    common frames, that entity's other-camera position projected into this camera falls on this entity's own box
    (on_box(view, xy_from_other_view, own_obs)). Its observations are merged into the best such entity instead of
    becoming a slot of their own."""
    def first(e, f, v):
        return next((o for vv, o, _ in e["obs"].get(f, []) if vv == v), None)
    drop, merges = set(), []
    for i, e in enumerate(ents):
        vs = entity_views(e)
        if len(vs) != 1 or e["team"] not in ("A", "B"):
            continue
        v = next(iter(vs)); other = "wide" if v == "ez" else "ez"
        best = None
        for k, q in enumerate(ents):
            if k == i or k in drop or q["team"] != e["team"] or other not in entity_views(q):
                continue
            fr = [f for f in e["obs"] if first(q, f, other) is not None]
            if len(fr) < min_common:
                continue
            frac = float(np.mean([on_box(v, first(q, f, other)["xy"], first(e, f, v)) for f in fr]))
            if frac >= min_frac and (best is None or frac > best[0]):
                best = (frac, k, len(fr))
        if best:
            frac, k, n = best
            for f, lst in e["obs"].items():
                ents[k]["obs"].setdefault(f, []).extend(lst)
            ents[k]["segments"] += e["segments"]
            drop.add(i); merges.append({"view": v, "frames": n, "box_frac": round(frac, 2)})
    return [e for k, e in enumerate(ents) if k not in drop], merges


def fit_ez_maps(ents, NF, win=30, half=15, min_pairs=20):
    """Pass-1 cross-camera pairs (frames where an entity holds both a Wide and an End Zone observation) -> per window
    of `win` frames a robust affine map End Zone field position -> Wide field position, fitted on the pairs within
    +-half frames around the window (iteratively reweighted: residuals > max(1 yd, 2.5 x median) dropped)."""
    pairs = []
    for e in ents:
        for f, lst in e["obs"].items():
            w = [o for v, o, _ in lst if v == "wide"]; z = [o for v, o, _ in lst if v == "ez"]
            if w and z:
                ow = min(w, key=lambda o: np.trace(o["cov"])); oz = min(z, key=lambda o: np.trace(o["cov"]))
                pairs.append((f, oz["xy"], ow["xy"]))
    maps = {}
    for w0 in range(0, NF, win):
        s_ = [(e_, w_) for f, e_, w_ in pairs if w0 - half <= f < w0 + win + half]
        if len(s_) < min_pairs:
            continue
        E = np.array([a for a, _ in s_]); W = np.array([b for _, b in s_])
        A = np.c_[E, np.ones(len(E))]; keep = np.ones(len(E), bool)
        for _ in range(3):
            M = np.linalg.lstsq(A[keep], W[keep], rcond=None)[0]
            r = np.linalg.norm(A @ M - W, axis=1); keep = r < max(1.0, 2.5 * float(np.median(r)))
        maps[w0] = M
    return maps, len(pairs)


def map_at(maps, f, win=30):
    if not maps:
        return None
    return maps[min(maps, key=lambda w0: abs(w0 + win / 2 - f))]


def entity_track(e):
    """Per-frame mean position of an entity's observations (for merging / hygiene)."""
    return {f: np.mean([o["xy"] for _, o, _ in lst], 0) for f, lst in e["obs"].items()}


def merge_time_disjoint(ents, max_gap=30):
    """Merge same-team entities that do not overlap in time when the first one's constant-velocity prediction lands
    near the second one's start (one successor / predecessor each, cheapest first)."""
    tr = [entity_track(e) for e in ents]
    cands = []
    for i, a in enumerate(ents):
        fa = sorted(tr[i])
        if not fa:
            continue
        tail = fa[-6:]
        pa = np.array([tr[i][f] for f in tail])
        va = (pa[-1] - pa[0]) / max(1, tail[-1] - tail[0]) if len(tail) > 1 else np.zeros(2)
        for j, b in enumerate(ents):
            if i == j or a["team"] != b["team"] or a["team"] == "official":
                continue
            fb = sorted(tr[j])
            if not fb or fb[0] <= fa[-1]:
                continue
            gap = fb[0] - fa[-1]
            if gap > max_gap:
                continue
            d = float(np.linalg.norm(tr[j][fb[0]] - (pa[-1] + va * gap)))
            lim = 1.5 + 0.1 * gap
            if d < lim:
                cands.append((d / lim, i, j))
    cands.sort()
    nxt, prv = {}, {}
    for c, i, j in cands:
        if i in nxt or j in prv:
            continue
        nxt[i] = j; prv[j] = i
    out, seen = [], set()
    for i in range(len(ents)):
        if i in prv or i in seen:
            continue
        e = {"obs": dict(ents[i]["obs"]), "team": ents[i]["team"], "segments": ents[i]["segments"], "merged_from": [i]}
        k = nxt.get(i)
        seen.add(i)
        while k is not None and k not in seen:
            for f, lst in ents[k]["obs"].items():
                e["obs"].setdefault(f, []).extend(lst)
            e["segments"] += ents[k]["segments"]; e["merged_from"].append(k); seen.add(k)
            k = nxt.get(k)
        out.append(e)
    return out, sum(len(e["merged_from"]) - 1 for e in out)

def pair_single_view(ents, gate_yd=4.0, min_common=20):
    """Entity-level second pass: pair entities seen only by Wide with entities seen only by End Zone (same team),
    Hungarian on the median distance over shared frames (>= min_common), gated at gate_yd. Catches players whose
    per-window distances in the crowded line exceed the window gate (foot points in a scrum differ by view)."""
    def views(e):
        return {v for lst in e["obs"].values() for v, _, _ in lst}
    W = [i for i, e in enumerate(ents) if views(e) == {"wide"} and e["team"] in ("A", "B")]
    E = [i for i, e in enumerate(ents) if views(e) == {"ez"} and e["team"] in ("A", "B")]
    if not W or not E:
        return ents, 0
    tr = {i: entity_track(ents[i]) for i in W + E}
    C = np.full((len(W), len(E)), 1e6)
    for a_, i in enumerate(W):
        for b_, j in enumerate(E):
            if ents[i]["team"] != ents[j]["team"]:
                continue
            common = sorted(set(tr[i]) & set(tr[j]))
            if len(common) < min_common:
                continue
            d = float(np.median([np.linalg.norm(tr[i][f] - tr[j][f]) for f in common]))
            if d <= gate_yd:
                C[a_, b_] = d
    r, c = linear_sum_assignment(C)
    drop, n = set(), 0
    for a_, b_ in zip(r, c):
        if C[a_, b_] >= 1e5:
            continue
        i, j = W[a_], E[b_]
        for f, lst in ents[j]["obs"].items():
            ents[i]["obs"].setdefault(f, []).extend(lst)
        ents[i]["segments"] += ents[j]["segments"]
        ents[i].setdefault("paired_single_view", []).append(round(float(C[a_, b_]), 2))
        drop.add(j); n += 1
    return [e for k, e in enumerate(ents) if k not in drop], n


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair")
    ap.add_argument("--suffix", default="_crop")
    ap.add_argument("--extra-suffix", default="", help="also load second-pass tracks track_<view><extra>/ (e.g. _p2)")
    ap.add_argument("--play-id", default=None)
    ap.add_argument("--no-bias-fix", action="store_true")
    ap.add_argument("--single-view-inflation", type=float, default=2.0)
    ap.add_argument("--assoc-gate", type=float, default=2.0, help="median cross-view distance gate (yd) for tracklet pairs")
    ap.add_argument("--assoc-min-common", type=int, default=8)
    ap.add_argument("--obs", choices=["sam", "det", "adaptive", "bt"], default="sam",
                    help="observations: SAM masks (full), RF-DETR detection tracks only, or detections + SAM where boxes overlap")
    ap.add_argument("--views", nargs="+", default=["wide", "ez"], help="cameras to use (single-view mode: --views wide)")
    ap.add_argument("--out-name", default="recon", help="output folder under the play folder")
    ap.add_argument("--foot-head", default=None, help="ground-contact head weights (train_foot.py head.pt)")
    ap.add_argument("--noise-model", default=None, help="fit_fusion.json: use its dot-fitted per-view noise model")
    ap.add_argument("--split-yd", type=float, default=1.5, help="line splitting radius (yd); 0 = off")
    ap.add_argument("--pair-gate", type=float, default=4.0, help="entity-level Wide-only/EZ-only pairing gate (yd)")
    ap.add_argument("--max-from-los", type=float, default=18.0, help="drop entities deeper than this behind the LOS on their own side at the snap (yd)")
    ap.add_argument("--box-assoc", type=int, default=1,
                    help="1: also match cameras by image-space box co-occupancy, re-match single-camera entities, prefer corroborated players for slots")
    ap.add_argument("--teams-per-view", type=int, default=1, help="1: classify dark / light teams within each camera separately")
    ap.add_argument("--bt-suffix", default="_bt", help="--obs bt: read track_<view><bt-suffix>/tracks.json (track_det.py)")
    ap.add_argument("--det-gate", type=int, default=5, help="drop SAM observations in runs of >= N frames with no confirming detector box (0 = off)")
    ap.add_argument("--ez-refine", type=int, default=1,
                    help="1: two-pass End Zone refinement - fit a per-1 s field mapping End Zone -> Wide on pass-1 cross-camera pairs, correct all End Zone observations, associate again")
    ap.add_argument("--max-extrapolate", type=int, default=5,
                    help="write no rows more than this many frames before a slot's first / after its last observation (-1 = write all)")
    a = ap.parse_args()
    global FOOT_HEAD, DET_GATE
    DET_GATE = a.det_gate
    if a.foot_head:
        FOOT_HEAD = load_foot_head(a.foot_head)
    pair = a.pair
    play = a.play_id or os.path.basename(os.path.abspath(pair))
    out_dir = os.path.join(pair, a.out_name); os.makedirs(out_dir, exist_ok=True)
    views = ["wide", "ez"]
    O, CAL, META = {}, {}, {}
    p2_info, obs_mode_info = {}, {}
    for v in views:
        if v not in a.views and not os.path.exists(os.path.join(pair, f"track_{v}{a.suffix}", "masks.json")):
            O[v], CAL[v], META[v] = [], None, None   # single-view play: this camera does not exist
            continue
        if a.obs == "bt":
            # detector-only tracks (track_det.py): no SAM masks are read
            O[v], CAL[v], META[v], obs_mode_info[v] = load_view_det(pair, v, a.suffix, None, a.bt_suffix)
            if v not in a.views:
                O[v] = []
            continue
        O[v], CAL[v], META[v] = load_view(pair, v, a.suffix)
        if a.extra_suffix and a.obs == "sam" and os.path.exists(os.path.join(pair, f"track_{v}{a.extra_suffix}", "masks.json")):
            o2, _, _ = load_view(pair, v, a.extra_suffix)
            for o in o2:
                o["sam_id"] += 1000
            O[v] += o2
            p2_info[v] = len(o2)
        if a.obs in ("det", "adaptive"):
            O[v], _, _, obs_mode_info[v] = load_view_det(pair, v, a.suffix, O[v] if a.obs == "adaptive" else None)
        if v not in a.views:
            O[v] = []   # single-view mode: the other camera contributes nothing
    # common timeline: wide clip frames
    taus = sorted({o["tau"] for o in O["wide"]} | {o["tau"] for o in O["ez"]})
    t0 = min(taus)
    frame_of = lambda tau: int(round((tau - t0) * FPS))
    NF = frame_of(max(taus)) + 1
    grid_tau = [round(t0 + f / FPS, 4) for f in range(NF)]

    # ---- 2. cross-view bias on isolated players (raw observations)
    def per_frame(obs):
        d = {}
        for o in obs:
            d.setdefault(frame_of(o["tau"]), []).append(o)
        return d
    PW, PE = per_frame(O["wide"]), per_frame(O["ez"])

    def isolated_pairs():
        out = []
        for f in set(PW) & set(PE):
            A = np.array([o["xy"] for o in PW[f]]); B = np.array([o["xy"] for o in PE[f]])
            if len(A) < 2 or len(B) < 2:
                continue
            C = np.linalg.norm(A[:, None] - B[None], axis=2)
            r, c = linear_sum_assignment(C)
            DA = np.linalg.norm(A[:, None] - A[None], axis=2); np.fill_diagonal(DA, 99)
            DB = np.linalg.norm(B[:, None] - B[None], axis=2); np.fill_diagonal(DB, 99)
            for i, j in zip(r, c):
                if C[i, j] < 3 and DA[i].min() > 3 and DB[j].min() > 3:
                    out.append((PW[f][i], PE[f][j]))
        return out
    iso = isolated_pairs()
    d_iso = np.array([e["xy"] - w["xy"] for w, e in iso]) if iso else np.zeros((0, 2))
    bias = np.median(d_iso, 0) if len(d_iso) >= 10 else np.zeros(2)
    bias_report = {"n_isolated_pairs": int(len(d_iso)), "ez_minus_wide_median_yd": bias.round(3).tolist(),
                   "before": {"median_yd": round(float(np.median(np.linalg.norm(d_iso, axis=1))), 3) if len(d_iso) else None},
                   "applied": not a.no_bias_fix}
    bias_applied = bias if not a.no_bias_fix else np.zeros(2)
    if not a.no_bias_fix:
        for o in O["ez"]:
            o["xy"] = o["xy"] - bias
        d2 = d_iso - bias
        bias_report["after"] = {"median_yd": round(float(np.median(np.linalg.norm(d2, axis=1))), 3) if len(d2) else None,
                                "residual_median_dxdy": np.median(d2, 0).round(3).tolist() if len(d2) else None}

    # ---- global covariance scale from isolated pairs (chi-square, 2 dof: median of d^2 = 1.386)
    if len(iso) >= 10:
        m2 = []
        for w, e in iso:
            r = (e["xy"] - w["xy"]) - (0 if a.no_bias_fix else 0)
            S = w["cov"] + e["cov"]
            m2.append(float(r @ np.linalg.solve(S, r)))
        cov_scale = float(np.median(m2) / 1.386)
    else:
        cov_scale = 1.0
    noise = json.load(open(a.noise_model))["noise_model"] if a.noise_model else None
    nb = {v: np.zeros(2) for v in views}   # noise-model bias removed from observations (added back in obs_* fields)
    if noise:
        # dot-fitted per-view noise (fit_fusion.py): sigma along / across the camera's depth axis grows with the
        # distance from the camera's nearest visible ground point (field point under the image bottom-centre);
        # per-view bias removed. Replaces the Jacobian covariance and the chi-square scale.
        cov_scale = 1.0
        key = {"wide": "w", "ez": "e"}
        for v in views:
            m = noise[key[v]]
            dep, crs = (1, 0) if v == "wide" else (0, 1)
            near_cache = {}
            for o in O[v]:
                sf = o["sf"]
                if sf not in near_cache:
                    Hinv = np.linalg.inv(np.array(CAL[v]["H"][sf]))
                    near_cache[sf] = img_to_field(Hinv, 960.0, 1079.0)
                d = float(np.linalg.norm(o["xy"] - near_cache[sf]))
                sd = np.zeros(2)
                sd[dep] = m["depth"]["a"] + m["depth"]["b"] * d
                sd[crs] = m["cross"]["a"] + m["cross"]["b"] * d
                o["cov"] = np.diag(sd ** 2)
                o["xy"] = o["xy"] - np.array(m["bias_xy"])
            nb[v] = np.array(m["bias_xy"])
    else:
        for v in views:
            for o in O[v]:
                o["cov"] = o["cov"] * cov_scale
    sig = {v: {"sigma_x_yd_median": round(float(np.median([np.sqrt(o["cov"][0, 0]) for o in O[v]])), 3),
               "sigma_y_yd_median": round(float(np.median([np.sqrt(o["cov"][1, 1]) for o in O[v]])), 3)} for v in views}
    if len(d_iso):
        emp = d_iso - (0 if a.no_bias_fix else bias)
        sig["empirical_crossview_residual_std_xy"] = np.std(emp, 0).round(3).tolist()

    # ---- 3. teams
    if a.teams_per_view:
        # jersey brightness differs between the cameras (lighting, front/back vs side view): split dark / light within
        # each camera, so 'A = darker team' holds in both views instead of one camera's light team landing in A
        labels, feats = {}, {}
        for v in views:
            if O[v]:
                lv, fv = classify_teams({v: O[v]})
                labels.update(lv); feats.update(fv)
    else:
        labels, feats = classify_teams(O)
    json.dump({"labels": {f"{k[0]}:{k[1]}": v for k, v in labels.items()},
               "features": {f"{k[0]}:{k[1]}": {kk: round(vv, 3) for kk, vv in f.items()} for k, f in feats.items()}},
              open(os.path.join(out_dir, "teams.json"), "w"), indent=1)

    # ---- 4. per-view tracklets + linking
    TR, SW, LINKS = {}, [], {}
    for v in views:
        tl, sw = build_tracklets(O[v], frame_of, labels, v)
        merged, nl = link_tracklets(tl)
        merged, n_dup = merge_second_pass(merged)
        TR[v], LINKS[v] = merged, {"tracklets": len(tl), "after_linking": len(merged), "links": int(nl),
                                   "second_pass_tracks_merged_into_first": n_dup}
        SW += sw

    # ---- 5. segment-level association -> entities, then merge time-disjoint same-team entities
    # image-space co-occupancy: a position from one camera, projected into the other camera's image at that
    # observation's frame, falls on that observation's box. Positions are in the Wide-aligned frame; the End Zone
    # image needs the cross-view and noise-model biases added back.
    ez_back = bias_applied + nb["ez"]

    ez_maps = {}

    def to_ez_frame(xy, f):
        """Wide-aligned field position -> End Zone (bias-corrected) frame: inverse of the pass-1 map when refined."""
        M = map_at(ez_maps, f) if ez_maps else None
        if M is None:
            return np.asarray(xy, float)
        L = np.array([[M[0, 0], M[1, 0]], [M[0, 1], M[1, 1]]])
        return np.linalg.solve(L, np.asarray(xy, float) - M[2])

    def on_box(view, xy, o):
        if CAL[view] is None:
            return False
        q = (to_ez_frame(xy, frame_of(o["tau"])) + ez_back) if view == "ez" else np.asarray(xy, float)
        p = np.array(CAL[view]["H"][o["sf"]]) @ np.array([q[0], q[1], 1.0])
        if abs(p[2]) < 1e-12:
            return False
        return under_box(o["box"], p[0] / p[2], p[1] / p[2])

    box_support = None
    if a.box_assoc and CAL["ez"] is not None and "ez" in a.views and "wide" in a.views:
        box_support = lambda tw, te, common: float(np.mean([on_box("ez", tw["obs"][f]["xy"], te["obs"][f]) for f in common]))
    def associate_all():
        e0, st = window_associate(TR, NF, gate_yd=a.assoc_gate, box_support=box_support)
        e1, nm = merge_time_disjoint(e0)
        e1, npair = pair_single_view(e1, gate_yd=a.pair_gate)
        rm = []
        if box_support is not None:
            e1, rm = rematch_single_view(e1, on_box)
        return e0, st, e1, nm, npair, rm

    ents0, assoc_stats, ents, n_merged, n_paired, rematched = associate_all()
    ez_refine = {"applied": False}
    ez_maps = {}
    if a.ez_refine and CAL["ez"] is not None and "ez" in a.views and "wide" in a.views:
        # two-pass End Zone refinement: the field mapping of the End Zone camera disagrees with the Wide one by ~1 yd
        # in a way that varies over the field and the clip; fit it per 1 s window on pass-1 cross-camera pairs, correct
        # every End Zone observation (in place: tracklets share the observation dicts), then associate again
        ez_maps, n_pairs = fit_ez_maps(ents, NF)
        if ez_maps:
            singles_before = sum(1 for e in ents if len(entity_views(e)) == 1)
            for o in O["ez"]:
                M = map_at(ez_maps, frame_of(o["tau"]))
                o["xy_ez"] = o["xy"]
                o["xy"] = np.r_[o["xy"], 1.0] @ M
            ents0, assoc_stats, ents, n_merged, n_paired, rematched = associate_all()
            ez_refine = {"applied": True, "pairs": n_pairs, "windows": len(ez_maps),
                         "single_camera_entities_pass1": singles_before,
                         "single_camera_entities_pass2": sum(1 for e in ents if len(entity_views(e)) == 1)}
    snap_f = min(max(frame_of(0.0), 0), NF - 1)   # clips whose first observation comes after the snap
    for e in ents:
        tr_ = entity_track(e)
        e["track"] = tr_
        e["n_obs"] = len(tr_)
        e["onfield"] = sum(1 for p in tr_.values() if -1 <= p[0] <= FIELD_LEN + 1 and 0.5 <= p[1] <= FIELD_W - 0.5)
        e["both_views"] = sum(1 for lst in e["obs"].values() if len({v for v, _, _ in lst}) == 2)
        near = sorted(tr_, key=lambda f: abs(f - snap_f))
        e["snap_xy"] = tr_[near[0]] if near and abs(near[0] - snap_f) <= 10 else None

    json.dump([{"i": k, "team": e["team"], "n_obs": e["n_obs"], "onfield": e["onfield"], "both_view_frames": e["both_views"],
                "views": sorted({v for lst in e["obs"].values() for v, _, _ in lst}),
                "snap_xy": None if e["snap_xy"] is None else [round(float(c), 2) for c in e["snap_xy"]],
                "frames": [min(e["track"]), max(e["track"])] if e["track"] else None,
                "tracks": sorted({f"{v}:{ti}" for lst in e["obs"].values() for v, _, ti in lst})}
               for k, e in enumerate(ents)], open(os.path.join(out_dir, "entities_debug.json"), "w"), indent=1)

    # ---- 6. slot hygiene + selection (never invent: fewer than 11 real candidates -> missing slots)
    # LOS at the snap: the x that leaves the fewest players on the wrong side of it for their team (each team's
    # side from its median x); ties -> middle of the best interval. Robust to spread formations / deep players.
    on_field = lambda e: e["snap_xy"] is not None and e["team"] in ("A", "B") and 1.0 < e["snap_xy"][1] < FIELD_W - 1.0
    XA = np.array([e["snap_xy"][0] for e in ents if on_field(e) and e["team"] == "A"])
    XB = np.array([e["snap_xy"][0] for e in ents if on_field(e) and e["team"] == "B"])
    if len(XA) and len(XB):
        sA = 1.0 if np.median(XA) > np.median(XB) else -1.0
        cands = np.sort(np.concatenate([XA, XB]))
        mids = (cands[:-1] + cands[1:]) / 2 if len(cands) > 1 else cands
        cost = [int(np.sum((XA - L) * sA < 0) + np.sum((XB - L) * -sA < 0)) for L in mids]
        best = np.flatnonzero(np.array(cost) == min(cost))
        los = float(np.median(mids[best]))
        side = {"A": sA, "B": -sA}
        los_misplaced = int(min(cost))
    else:
        los, side, los_misplaced = float("nan"), {"A": 0.0, "B": 0.0}, None
    dropped = {"official": 0, "off_field": 0, "sideline": 0, "wrong_side_of_los_at_snap": 0,
               "too_deep_at_snap": 0, "too_short": 0}
    keep = []
    for e in ents:
        P_ = np.array(list(e["track"].values()))
        if e["team"] == "official":
            dropped["official"] += 1; continue
        if e["n_obs"] < 8:
            dropped["too_short"] += 1; continue
        if e["onfield"] < 0.5 * e["n_obs"]:
            dropped["off_field"] += 1; continue
        my = float(np.median(P_[:, 1]))
        if my < 1.5 or my > FIELD_W - 1.5:
            dropped["sideline"] += 1; continue
        if e["snap_xy"] is not None and side.get(e["team"]):
            depth = (e["snap_xy"][0] - los) * side[e["team"]]   # + = on its own side of the LOS
            if depth < -3.0:
                dropped["wrong_side_of_los_at_snap"] += 1; continue
            if depth > a.max_from_los:
                dropped["too_deep_at_snap"] += 1; continue
        keep.append(e)
    slots = []
    for team in ("A", "B"):
        cand = [e for e in keep if e["team"] == team]
        # players confirmed by both cameras first (>= 10 fused frames), then by observation count: a single-camera
        # entity only takes a slot when fewer than 11 corroborated players exist
        cand.sort(key=lambda e: (e["both_views"] < 10 if a.box_assoc else 0, -e["onfield"], -e["n_obs"]))
        slots += [(team, e) for e in cand[:11]]
        if len(cand) > 11:
            dropped[f"excess_{team}"] = len(cand) - 11

    # line splitting: for a slot seen only by End Zone at a frame, a Wide observation within split_yd of it that is
    # also within split_yd of another player's End Zone position (either team) is a merged blob covering both; it
    # then contributes to this slot along x only (the Wide camera's well-measured axis), End Zone supplies y
    wide_by_f = {}
    for t_ in TR["wide"]:
        tm = t_["team"].rstrip("?")
        for f, o in t_["obs"].items():
            wide_by_f.setdefault(f, []).append((o, tm))
    ez_pos = {}   # f -> list of (slot index, End Zone xy) for every slot with an End Zone observation
    for si, (team, e) in enumerate(slots):
        for f, lst in e["obs"].items():
            vs = {v_ for v_, _, _ in lst}
            if "ez" in vs:
                ez_pos.setdefault(f, []).append((si, min((o_ for v_, o_, _ in lst if v_ == "ez"), key=lambda o_: np.trace(o_["cov"]))["xy"]))
    split_frames = 0
    split_dbg = {"ez_only_frames": 0, "wide_within": 0, "wide_within_any_team": 0, "blob_ge2": 0}

    fused_tracks = []
    disagreement = []
    conflicts, conflict_yd = 0, 3.0
    counts = {"both": 0, "one": 0, "none": 0}
    for si, (team, e) in enumerate(slots):
        meas, covs, srcs, vis = {}, {}, {}, {}
        conf_frames = set()
        for f in range(NF):
            zs, Rs, vv = [], [], []
            for v_, o_, _ in e["obs"].get(f, []):
                zs.append(o_["xy"]); Rs.append(o_["cov"]); vv.append(v_)
            if not zs:
                continue
            split = False
            if a.split_yd > 0 and set(vv) == {"ez"}:
                ze = zs[int(np.argmin([np.trace(R) for R in Rs]))]
                others = [(sj, xy) for sj, xy in ez_pos.get(f, []) if sj != si]   # other players (either team)
                best = None
                split_dbg["ez_only_frames"] += 1
                if any(np.linalg.norm(o_w["xy"] - ze) <= a.split_yd for o_w, _ in wide_by_f.get(f, [])):
                    split_dbg["wide_within_any_team"] += 1
                for o_w, tm in wide_by_f.get(f, []):
                    if tm == "official":
                        continue
                    dw = float(np.linalg.norm(o_w["xy"] - ze))
                    if dw > a.split_yd:
                        continue
                    split_dbg["wide_within"] += 1
                    near = sum(1 for sj, xy in others if np.linalg.norm(xy - o_w["xy"]) <= a.split_yd)
                    if near >= 1:
                        split_dbg["blob_ge2"] += 1
                    if near >= 1 and (best is None or dw < best[0]):
                        best = (dw, o_w)
                if best is not None:
                    o_w = best[1]
                    Rw = np.array([[o_w["cov"][0, 0], 0.0], [0.0, 1e4]])
                    zs.append(np.array([o_w["xy"][0], ze[1]])); Rs.append(Rw); vv.append("wide")
                    split = True
                    split_frames += 1
            if len(vv) > len(set(vv)):   # two segments of the same view (merged entities): keep the lower-variance one per view
                best = {}
                for zz, RR, v_ in zip(zs, Rs, vv):
                    if v_ not in best or np.trace(RR) < np.trace(best[v_][1]):
                        best[v_] = (zz, RR)
                vv = list(best); zs = [best[v_][0] for v_ in vv]; Rs = [best[v_][1] for v_ in vv]
            if len(set(vv)) == 2 and not split and np.linalg.norm(zs[vv.index("wide")] - zs[vv.index("ez")]) > conflict_yd:
                # views disagree: do not average; keep the one closest to this slot's last fused position
                conflicts += 1
                conf_frames.add(f)
                last = meas[max(meas)] if meas else None
                if last is not None:
                    i = int(np.argmin([np.linalg.norm(zz - last) for zz in zs]))
                else:
                    i = int(np.argmin([np.trace(R) for R in Rs]))
                z, C = zs[i], Rs[i] * a.single_view_inflation
                srcs[f] = VIEW_NAME[vv[i]]; vv = [vv[i]]
                meas[f], covs[f], vis[f] = z, C, set(vv)
                continue
            if len(set(vv)) == 2:
                W = sum(np.linalg.inv(R) for R in Rs)
                C = np.linalg.inv(W)
                z = C @ sum(np.linalg.inv(R) @ zz for zz, R in zip(zs, Rs))
                iw = vv.index("wide"); ie = vv.index("ez")
                if split:
                    srcs[f] = "fused_split"
                else:
                    disagreement.append(float(np.linalg.norm(zs[iw] - zs[ie])))
                    srcs[f] = "fused"
            else:
                z, C = zs[0], Rs[0] * a.single_view_inflation
                srcs[f] = VIEW_NAME[vv[0]]
            meas[f], covs[f], vis[f] = z, C, set(vv)
        if not meas:
            continue
        sm = kalman_smooth(list(range(NF)), meas, covs)
        e["sm"], e["srcs"], e["vis"], e["conflict"] = sm, srcs, vis, conf_frames
        fused_tracks.append((team, e))
        for f in range(NF):
            n = len(vis.get(f, ()))
            counts["both" if n == 2 else ("one" if n == 1 else "none")] += 1
    # numbering by lateral position at the snap
    final = []
    for team in ("A", "B"):
        tt = [e for t, e in fused_tracks if t == team]
        tt.sort(key=lambda e: (e["sm"][snap_f][0][1], e["sm"][snap_f][0][0]))
        for n, e in enumerate(tt, 1):
            final.append((f"{team}_{n:02d}", team, e))

    # ---- visibility per camera (projected slot position inside that camera's image at that frame)
    sync = json.load(open(os.path.join(pair, "sync.json")))
    rel = {v: np.array(CAL[v]["pts_s"]) - sync[v]["snap_pts"] for v in views if CAL[v] is not None}

    def in_view(v, f, xy):
        if v not in rel:
            return None
        sf = int(np.argmin(np.abs(rel[v] - grid_tau[f])))
        if abs(rel[v][sf] - grid_tau[f]) > 0.5 / FPS:
            return None
        Hm = np.array(CAL[v]["H"][sf])
        p = Hm @ np.array([xy[0], xy[1], 1.0])
        # in front of the camera = same homogeneous sign as the field point under the image centre
        c = np.linalg.inv(Hm) @ np.array([960.0, 540.0, 1.0])
        ref_sign = np.sign((Hm @ np.array([c[0] / c[2], c[1] / c[2], 1.0]))[2])
        if np.sign(p[2]) != ref_sign or abs(p[2]) < 1e-12:
            return False
        u, w_ = p[0] / p[2], p[1] / p[2]
        return bool(0 <= u < 1920 and 0 <= w_ < 1080)

    lines = []
    extrapolated_rows_dropped = 0
    for pid, team, e in final:
        obs_f = sorted(e["srcs"])
        lo_f, hi_f = obs_f[0] - a.max_extrapolate, obs_f[-1] + a.max_extrapolate
        for f in range(NF):
            if a.max_extrapolate >= 0 and not (lo_f <= f <= hi_f):
                extrapolated_rows_dropped += 1   # never seen this early / late: no position is written
                continue
            xy, P, observed, vel = e["sm"][f]
            vis = {}
            for v in views:
                if v in e["vis"].get(f, ()):
                    vis[VIEW_NAME[v]] = "visible"
                else:
                    iv = in_view(v, f, xy)
                    vis[VIEW_NAME[v]] = "occluded" if iv else ("out_of_view" if iv is False else "no_frame")
            src = e["srcs"].get(f, "inferred")
            sd = float(np.sqrt(np.trace(P)))
            per_view = {}
            for v_, o_, _ in e["obs"].get(f, []):
                if v_ not in per_view or np.trace(o_["cov"]) < np.trace(per_view[v_]["cov"]):
                    per_view[v_] = o_
            lines.append({"play_id": play, "t": grid_tau[f], "frame": f, "player_id": pid, "team": team,
                          "x_yards": round(float(xy[0]), 3), "y_yards": round(float(xy[1]), 3),
                          "covariance": [[round(float(P[0, 0]), 4), round(float(P[0, 1]), 4)], [round(float(P[1, 0]), 4), round(float(P[1, 1]), 4)]],
                          "position_confidence": round(float(np.exp(-sd / 1.0)), 3),
                          "speed_yps": round(float(np.linalg.norm(vel)), 2),
                          "visibility": vis, "source": src, "estimated": src == "inferred", "conflict": f in e["conflict"],
                          "verified": src in ("fused", "fused_split"),
                          "obs_sideline": [round(float(c), 3) for c in per_view["wide"]["xy"] + nb["wide"]] if "wide" in per_view else None,
                          "obs_endzone": [round(float(c), 3) for c in per_view["ez"]["xy"] + nb["ez"]] if "ez" in per_view else None,
                          "obs_endzone_raw": [round(float(c), 3) for c in (per_view["ez"].get("xy_ez", per_view["ez"]["xy"]) + nb["ez"] + bias_applied)] if "ez" in per_view else None})
    with open(os.path.join(out_dir, "recon.jsonl"), "w") as fh:
        for l in lines:
            fh.write(json.dumps(l) + "\n")

    tot = max(1, sum(counts.values()))
    n_slots = len(final)
    summary = {
        "play_id": play, "frames": NF, "t_range": [grid_tau[0], grid_tau[-1]],
        "slots_filled": {"A": sum(1 for p, t, e in final if t == "A"), "B": sum(1 for p, t, e in final if t == "B")},
        "slot_frames_observed": {k: round(v / tot, 3) for k, v in counts.items()},
        "slot_frames_observed_of_22": {k: round(v / (22 * NF), 3) for k, v in counts.items()},
        "crossview_disagreement_fused_frames": {"n": len(disagreement),
                                                 "median_yd": round(float(np.median(disagreement)), 3) if disagreement else None,
                                                 "p95_yd": round(float(np.percentile(disagreement, 95)), 3) if disagreement else None},
        "crossview_conflict_frames": conflicts, "conflict_threshold_yd": conflict_yd,
        "bias": bias_report, "cov_scale": round(cov_scale, 3), "sigma": sig, "noise_model": a.noise_model,
        "teams": {t: sum(1 for k, l in labels.items() if l.rstrip("?") == t) for t in ("A", "B", "official")},
        "teams_uncertain": sum(1 for l in labels.values() if l.endswith("?")),
        "tracking": LINKS, "candidate_id_switches": SW, "association": assoc_stats,
        "entities_before_merge": len(ents0), "entities": len(ents), "time_disjoint_merges": n_merged,
        "single_view_pairs": n_paired, "second_pass_observations": p2_info,
        "obs_mode": a.obs, "views_used": a.views, "obs_mode_info": obs_mode_info,
        "los_x_at_snap": round(los, 2), "los_misplaced_players": los_misplaced, "team_side_of_los": side, "slot_hygiene_dropped": dropped,
        "players_associated": sum(1 for p, t, e in final if sum(1 for s_ in e["srcs"].values() if s_ in ("fused", "fused_split")) >= 10),
        "players_associated_without_split": sum(1 for p, t, e in final if sum(1 for s_ in e["srcs"].values() if s_ == "fused") >= 10),
        "line_split_frames": split_frames, "line_split_debug": split_dbg,
        "players_associated_def": "slots with >= 10 fused (both-view) frames",
        "single_view_inflation": a.single_view_inflation,
        "box_assoc": a.box_assoc, "single_view_rematched": rematched, "extrapolated_rows_dropped": extrapolated_rows_dropped,
        "ez_refine": ez_refine,
    }
    json.dump(summary, open(os.path.join(out_dir, "recon_summary.json"), "w"), indent=1)
    if ez_maps:
        # affine Wide-aligned field -> End Zone camera's own field frame, per frame (overlay.py draws with it)
        em = {}
        for f in range(NF):
            M = map_at(ez_maps, f)
            L = np.array([[M[0, 0], M[1, 0]], [M[0, 1], M[1, 1]]]); Li = np.linalg.inv(L)
            em[f] = np.c_[Li, -Li @ M[2] + ez_back].round(6).tolist()
        json.dump(em, open(os.path.join(out_dir, "ez_map.json"), "w"))
    json.dump({"association": assoc_stats,
               "slots": [{"player_id": p, "team": t, "frames_fused": sum(1 for s_ in e["srcs"].values() if s_ == "fused"),
                          "frames_observed": len(e["srcs"]), "segments": e["segments"], "merged_from": e.get("merged_from")}
                         for p, t, e in final]}, open(os.path.join(out_dir, "associations.json"), "w"), indent=1)
    brief = {k: v for k, v in summary.items() if k not in ("candidate_id_switches", "sigma")}
    brief["n_candidate_id_switches"] = sum(1 for w in SW if "sam_id" in w)
    brief["outlier_observations_dropped"] = {w2: sum(w["outlier_observations_dropped"] for w in SW if "outlier_observations_dropped" in w) for w2 in ["total"]}
    brief["sigma"] = sig
    print(json.dumps(brief))
    render_replay(out_dir, final, NF, grid_tau, snap_f)


def render_replay(out_dir, final, NF, grid_tau, snap_f):
    ppy, margin = 14, 30
    xs = [e["sm"][f][0][0] for _, _, e in final for f in range(NF)]
    x0, x1 = max(-2.0, np.percentile(xs, 1) - 8), min(122.0, np.percentile(xs, 99) + 8)
    W = int((x1 - x0) * ppy + 2 * margin); Hh = int(FIELD_W * ppy + 2 * margin)
    W += W % 2
    base = np.full((Hh, W, 3), (40, 110, 40), np.uint8)
    P = lambda x, y: (int(margin + (x - x0) * ppy), int(margin + (FIELD_W - y) * ppy))
    for x in range(int(np.ceil(x0)), int(x1) + 1):
        if 0 <= x <= FIELD_LEN:
            cv2.line(base, P(x, 0), P(x, FIELD_W), (255, 255, 255) if x % 5 == 0 else (60, 130, 60), 2 if x % 5 == 0 else 1)
            if x % 10 == 0 and 10 <= x <= 110:
                yd = int(x - 10 if x <= 60 else 110 - x)
                cv2.putText(base, str(yd), (P(x, 12)[0] - 12, P(x, 12)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for y in (20.0, FIELD_W - 20.0):
                cv2.line(base, P(x, y - 0.67), P(x, y), (255, 255, 255), 1)
    for y in (0, FIELD_W):
        cv2.line(base, P(max(0, x0), y), P(min(FIELD_LEN, x1), y), (255, 255, 255), 2)
    H2 = Hh + 70
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H2}", "-r", str(FPS),
                             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", os.path.join(out_dir, "replay.mp4")],
                            stdin=subprocess.PIPE)
    col = {"A": ((40, 40, 40), (255, 255, 255)), "B": ((245, 245, 245), (0, 0, 200))}
    src_col = {"fused": (0, 255, 0), "fused_split": (0, 200, 120), "sideline": (255, 160, 0), "endzone": (0, 160, 255), "inferred": (128, 128, 128)}
    for f in range(NF):
        img = np.zeros((H2, W, 3), np.uint8); img[:Hh] = base
        for pid, team, e in final:
            pts = [P(*e["sm"][g][0]) for g in range(max(0, f - 30), f + 1)]
            for p, q in zip(pts[:-1], pts[1:]):
                cv2.line(img, p, q, col[team][1], 1)
        over = img.copy()
        for pid, team, e in final:
            xy, Pc, obs, _ = e["sm"][f]
            c = P(*xy)
            conf = float(np.exp(-np.sqrt(np.trace(Pc))))
            layer = img.copy()
            cv2.circle(layer, c, 8, col[team][0], -1)
            cv2.circle(layer, c, 8, src_col[e["srcs"].get(f, "inferred")], 2)
            cv2.putText(layer, pid[2:], (c[0] + 9, c[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col[team][1], 1)
            alpha = 0.25 + 0.75 * conf
            img = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)
        # timeline
        cv2.rectangle(img, (20, Hh + 15), (W - 20, Hh + 25), (90, 90, 90), -1)
        xs_ = 20 + (W - 40) * f / max(1, NF - 1)
        xsn = 20 + (W - 40) * snap_f / max(1, NF - 1)
        cv2.line(img, (int(xsn), Hh + 10), (int(xsn), Hh + 30), (0, 0, 255), 2)
        cv2.circle(img, (int(xs_), Hh + 20), 7, (255, 255, 255), -1)
        cv2.putText(img, f"t = {grid_tau[f]:+.2f} s   ring: green=fused orange=sideline blue=endzone grey=inferred   alpha=confidence",
                    (20, Hh + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        proc.stdin.write(img.tobytes())
    proc.stdin.close(); proc.wait()


if __name__ == "__main__":
    main()
