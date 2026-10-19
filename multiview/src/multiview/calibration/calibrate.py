#!/usr/bin/env python
"""Time-varying field homography H(t) for one view (sam3env python).

  calibrate.py VIEW_DIR LABELS.json [--view wide|ez] [--hashes ncaa] [--overlay] [--every 1]

VIEW_DIR has frames/*.jpg, frames_pts.json and dets_all.json (players are masked out of the paint).
LABELS.json (keyframe initialisation). Either {"frame": k, "ref": {"view_dir": D, "frame": k_ref}} to register
the keyframe to an already calibrated keyframe of the same physical camera (SIFT on background, PTZ homography),
or labels written by hand from the painted numbers:
  {"frame": 125, "lines_json": "...lines.json",           # lines.py output for that frame
   "x_lines": {"5": 40, "2": 45},                          # detected line id -> canonical x (yards)
   "y_lines": {"1": 0.0},                                  # detected line id -> canonical y
   "x_pts": [[[px,py],[px,py], x]], "y_pts": [[[px,py],[px,py], y]]}   # extra lines given by two image points
H maps canonical field yards -> image pixels. Initial H from the labelled lines (line DLT), then a robust
chamfer fit of the NCAA paint model (yard lines, hash marks, sideline ticks, sidelines) to the white-paint
mask. Other frames: predicted by the background-feature camera motion (prev image -> cur image), then re-fit
with gating (accept the re-fit only if it does not raise the residual and stays near the prediction); a frame
whose prediction is poor (camera motion inlier ratio low / large shift) gets a wide-basin re-fit. Every
--anchor-every frames (and after fast pans) the keyframe is registered straight to the current frame (SIFT, PTZ
homography) and re-fitted; that H replaces the chained one if it matches the paint better ("reanchor").
reg_drift_yd: the keyframe's fixed field point registered straight to frame t vs H(t) (independent of the chain).

Writes VIEW_DIR/calib.json: per frame H, residual_px (median distance of visible model paint points to the
nearest detected paint, inliers < 6 px at full res), inlier_frac, n_visible, refit flags, stability drift (yd):
a fixed field point is carried through the camera-motion chain and mapped back through H(t)^-1.
"""
import argparse, json, os, sys

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path[:0] = __import__("glob").glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "*", ""))  # all src/multiview/* module dirs (flat sibling imports)
import fieldmodel as fmod  # noqa: E402
from fieldmask import field_mask  # noqa: E402

S = 0.5            # working scale for paint / distance transform
CAP = 12.0         # px (working scale) residual cap
INLIER = 3.0       # px (working scale) = 6 px full res


def safe_inv(H):
    """Inverse of a homography, or None if it is degenerate."""
    try:
        if not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
            return None
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None


def field_dist(H, q, P0):
    Hi = safe_inv(H)
    if Hi is None:
        return float("nan")
    return float(np.linalg.norm(fmod.project(Hi, [q])[0] - P0))


def line_from_pts(p, q):
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def rho_theta_line(rho, th):
    return np.array([np.cos(th), np.sin(th), -rho])


def dlt_lines(img_lines, fld_lines):
    """H (field->image) from line correspondences: field line L ~ H^T l."""
    Ti = np.diag([1 / 1000, 1 / 1000, 1.0])   # image normalisation (points p' = Ti p -> lines l' = Ti^-T l)
    Tf = np.diag([1 / 100, 1 / 100, 1.0])
    A = []
    for l, L in zip(img_lines, fld_lines):
        l = np.linalg.inv(Ti).T @ l; l /= np.linalg.norm(l)
        L = np.linalg.inv(Tf).T @ L; L /= np.linalg.norm(L)
        M = np.zeros((3, 9))
        for i in range(3):
            for j in range(3):
                M[i, j * 3 + i] = l[j]
        Lx = np.array([[0, -L[2], L[1]], [L[2], 0, -L[0]], [-L[1], L[0], 0]])
        A.append(Lx @ M)
    A = np.vstack(A)
    h = np.linalg.svd(A)[2][-1]
    Hn = h.reshape(3, 3)
    H = np.linalg.inv(Ti) @ Hn @ Tf
    return H / H[2, 2]


class Fitter:
    def __init__(self, step=0.25):
        self.P, self.kind = fmod.model_points(step)

    def prepare(self, bgr, dets, view):
        fm, _ = field_mask(bgr, view=view, erode_px=0)
        boxes = [x["box_xyxy"] for x in dets["detections"] + dets.get("dropped", [])]
        paint = fmod.paint_mask(bgr, cv2.dilate(fm, np.ones((41, 41), np.uint8)), boxes)
        small = cv2.resize(paint, None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
        small = (small > 40).astype(np.uint8) * 255
        dt = cv2.distanceTransform(255 - small, cv2.DIST_L2, 5).astype(np.float32)
        valid = np.full(small.shape, 255, np.uint8)  # where paint could be seen: field, not under a player
        valid &= cv2.resize(cv2.dilate(fm, np.ones((41, 41), np.uint8)), (small.shape[1], small.shape[0]))
        for b in boxes:
            x1, y1, x2, y2 = [int(round(c * S)) for c in b]
            cv2.rectangle(valid, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), 0, -1)
        self.dt, self.valid, self.paint = dt, valid, paint
        self.h, self.w = dt.shape

    def _proj(self, Hs):
        p = cv2.perspectiveTransform(self.P.reshape(-1, 1, 2), Hs).reshape(-1, 2)
        ok = np.isfinite(p).all(1) & (p[:, 0] >= 0) & (p[:, 0] < self.w - 1) & (p[:, 1] >= 0) & (p[:, 1] < self.h - 1)
        # points behind the camera project with negative w: check denominator sign
        den = self.P @ Hs[2, :2] + Hs[2, 2]
        # H is only defined up to scale; normalising by H[2,2] flips the sign when the field origin lies behind the
        # camera. "In front of the camera" = the sign shared by the majority of in-image points.
        sgn = 1.0 if np.sum(ok & (den > 0)) >= np.sum(ok & (den < 0)) else -1.0
        ok &= den * sgn > 0
        return p, ok

    def residuals(self, Hs, mask_pts=None):
        p, ok = self._proj(Hs)
        if mask_pts is not None:
            ok &= mask_pts
        q = p[ok].astype(np.float32)
        if len(q) == 0:
            return np.zeros(0, np.float32), 0
        vis = self.valid[q[:, 1].astype(int), q[:, 0].astype(int)] > 0
        if not vis.any():
            return np.zeros(0, np.float32), 0
        d = cv2.remap(self.dt, q[:, 0].reshape(-1, 1), q[:, 1].reshape(-1, 1), cv2.INTER_LINEAR).ravel()
        return np.minimum(d[vis], CAP), int(vis.sum())

    def fit(self, H, f_scale=1.5, iters=60, basin=1.0):
        Ss = np.diag([S, S, 1.0])
        Hs0 = Ss @ H
        Hs0 = Hs0 / Hs0[2, 2]
        # fix the set of model points that are visible at the start so the cost does not jump
        p, ok = self._proj(Hs0)
        if ok.sum() < 30:
            return H, None
        q = p[ok]
        vis = np.zeros(len(ok), bool)
        vis[np.where(ok)[0][self.valid[q[:, 1].astype(int), q[:, 0].astype(int)] > 0]] = True
        if vis.sum() < 30:
            return H, None
        Pv = self.P[vis]
        dt = self.dt if basin <= 1 else cv2.GaussianBlur(self.dt, (0, 0), basin)

        def fun(h):
            Hs = np.append(h, 1.0).reshape(3, 3)
            pp = cv2.perspectiveTransform(Pv.reshape(-1, 1, 2), Hs).reshape(-1, 2)
            out = np.full(len(pp), CAP, np.float32)
            inb = np.isfinite(pp).all(1) & (pp[:, 0] >= 0) & (pp[:, 0] < self.w - 1) & (pp[:, 1] >= 0) & (pp[:, 1] < self.h - 1)
            if not inb.any():
                return out
            qq = pp[inb].astype(np.float32)
            out[inb] = np.minimum(cv2.remap(dt, qq[:, 0].reshape(-1, 1), qq[:, 1].reshape(-1, 1), cv2.INTER_LINEAR).ravel(), CAP)
            return out

        r = least_squares(fun, Hs0.ravel()[:8], loss="soft_l1", f_scale=f_scale, max_nfev=iters * 9, x_scale="jac")
        Hs = np.append(r.x, 1.0).reshape(3, 3)
        Hf = np.linalg.inv(Ss) @ Hs
        return Hf / Hf[2, 2], r

    def score(self, H):
        Ss = np.diag([S, S, 1.0])
        d, n = self.residuals(Ss @ H)
        if n == 0:
            return {"residual_px": None, "inlier_frac": 0.0, "n_visible": 0}
        inl = d < INLIER
        return {"residual_px": round(float(np.median(d[inl]) / S), 2) if inl.any() else None,
                "inlier_frac": round(float(inl.mean()), 3), "n_visible": n}


def camera_motion(frames_dir, frames, dets):
    """C[t]: full-res image homography frame t-1 -> t (C[0] = I), with inlier ratio."""
    Cs, q = [np.eye(3)], [1.0]
    prev = None
    Ss = np.diag([S, S, 1.0])
    for t, f in enumerate(frames):
        g = cv2.imread(os.path.join(frames_dir, f), cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
        m = np.full(g.shape, 255, np.uint8)
        for d in dets[t]["detections"] + dets[t].get("dropped", []):
            x1, y1, x2, y2 = (np.array(d["box_xyxy"]) * S).astype(int)
            cv2.rectangle(m, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), 0, -1)
        if prev is not None:
            pg, pm = prev
            C, qq = np.eye(3), 0.0
            p0 = cv2.goodFeaturesToTrack(pg, 1000, 0.01, 7, mask=pm)
            if p0 is not None and len(p0) >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, p0, None, winSize=(21, 21), maxLevel=3)
                ok = st.ravel() == 1
                if ok.sum() >= 12:
                    Hc, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 1.5)
                    if Hc is not None:
                        C, qq = np.linalg.inv(Ss) @ Hc @ Ss, float(inl.mean())
            Cs.append(C); q.append(qq)
        prev = (g, m)
    return Cs, q


def register_to_reference(ref_img, ref_dets, img, dets):
    """Image homography ref -> img from SIFT matches on everything except players (PTZ camera: valid for all depths)."""
    def prep(im, d):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        m = np.full(g.shape, 255, np.uint8)
        for x in d["detections"] + d.get("dropped", []):
            x1, y1, x2, y2 = [int(c) for c in x["box_xyxy"]]
            cv2.rectangle(m, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), 0, -1)
        return g, m
    sift = cv2.SIFT_create(nfeatures=6000)
    g1, m1 = prep(ref_img, ref_dets); g2, m2 = prep(img, dets)
    k1, d1 = sift.detectAndCompute(g1, m1); k2, d2 = sift.detectAndCompute(g2, m2)
    mt = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in mt if a.distance < 0.75 * b.distance]
    p1 = np.float32([k1[a.queryIdx].pt for a in good]); p2 = np.float32([k2[a.trainIdx].pt for a in good])
    M, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    return M, {"matches": len(good), "inliers": int(inl.sum()) if inl is not None else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("view_dir"); ap.add_argument("labels")
    ap.add_argument("--view", default="wide")
    ap.add_argument("--hashes", default="ncaa")
    ap.add_argument("--hash-dir", default="outward", choices=["outward", "inward", "centered"])
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--init-only", action="store_true")
    ap.add_argument("--reanchor-max-yd", type=float, default=3.0,
                    help="reject a keyframe re-registration that disagrees with the chained H by more than this")
    ap.add_argument("--anchor-every", type=int, default=30,
                    help="re-register the keyframe straight to the current frame every N frames (and after fast pans); 0=off")
    a = ap.parse_args()
    fmod.set_hashes(a.hashes)
    fmod.set_hash_dir(a.hash_dir)
    vd = a.view_dir
    lab = json.load(open(a.labels))
    frames = sorted(f for f in os.listdir(os.path.join(vd, "frames")) if f.endswith(".jpg"))
    pts = json.load(open(os.path.join(vd, "frames_pts.json")))["pts_s"]
    dets = json.load(open(os.path.join(vd, "dets_all.json")))["frames"]
    k = lab["frame"]

    # ---- keyframe init
    fit = Fitter()
    img = cv2.imread(os.path.join(vd, "frames", frames[k]))
    reg = None
    refs = lab.get("refs") or ([lab["ref"]] if "ref" in lab else [])
    if refs:   # register to already calibrated keyframes of the same physical camera; keep the best paint fit
        fit.prepare(img, dets[k], a.view)
        best = None
        tried = []
        for ref in refs:
            rvd, rk = ref["view_dir"], ref["frame"]
            try:
                rc = json.load(open(os.path.join(rvd, "calib.json")))
                rframes = sorted(f for f in os.listdir(os.path.join(rvd, "frames")) if f.endswith(".jpg"))
                rimg = cv2.imread(os.path.join(rvd, "frames", rframes[rk]))
                rdets = json.load(open(os.path.join(rvd, "dets_all.json")))["frames"][rk]
                M, rg = register_to_reference(rimg, rdets, img, dets[k])
            except Exception as ex:
                tried.append({"ref": rvd, "error": str(ex)[:80]}); continue
            if M is None or rg["inliers"] < 12:
                tried.append({"ref": rvd, **rg}); continue
            Hc = M @ np.array(rc["H"][rk]); Hc /= Hc[2, 2]
            Hr_, _ = fit.fit(Hc, basin=4.0, iters=80)
            Hr_, _ = fit.fit(Hr_, basin=1.0, iters=80)
            sc_ = fit.score(Hr_)
            tried.append({"ref": rvd, **rg, "inlier_frac": sc_["inlier_frac"], "residual_px": sc_["residual_px"]})
            if sc_["residual_px"] is not None and (best is None or sc_["inlier_frac"] > best[0]):
                best = (sc_["inlier_frac"], Hc, rg, rvd)
            if best is not None and best[0] >= 0.8:
                break
        if best is None:
            raise SystemExit(f"calibration failed: no reference keyframe registered to frame {k} ({len(tried)} tried): {tried}")
        H0, reg = best[1], {**best[2], "ref": best[3], "tried": tried}
        il = []
    else:
        L = {r["id"]: rho_theta_line(r["rho"], r["theta"]) for r in json.load(open(lab["lines_json"]))}
        il, fl = [], []
    if reg is None:
        for i, x in lab.get("x_lines", {}).items():
            il.append(L[int(i)]); fl.append(np.array([1.0, 0.0, -x]))
        for i, y in lab.get("y_lines", {}).items():
            il.append(L[int(i)]); fl.append(np.array([0.0, 1.0, -y]))
        for p, q, x in lab.get("x_pts", []):
            il.append(line_from_pts(p, q)); fl.append(np.array([1.0, 0.0, -x]))
        for p, q, y in lab.get("y_pts", []):
            il.append(line_from_pts(p, q)); fl.append(np.array([0.0, 1.0, -y]))
        H0 = dlt_lines(il, fl)
    fit.prepare(img, dets[k], a.view)
    s0 = fit.score(H0)
    H1, _ = fit.fit(H0, basin=4.0, iters=80)   # wide basin first
    H1, _ = fit.fit(H1, basin=1.0, iters=80)
    s1 = fit.score(H1)
    print(json.dumps({"keyframe": k, "init_mode": "registered" if reg else "labelled_lines", "registration": reg,
                      "init": s0, "refined": s1}))
    os.makedirs(os.path.join(vd, "calib"), exist_ok=True)
    cv2.imwrite(os.path.join(vd, "calib", f"key_{k:05d}_init.jpg"), fmod.draw_grid(img, H0, one_yard=False))
    cv2.imwrite(os.path.join(vd, "calib", f"key_{k:05d}_fit.jpg"), fmod.draw_grid(img, H1))
    if a.init_only:
        return

    # ---- propagate through all frames
    Cs, cq = camera_motion(os.path.join(vd, "frames"), frames, dets)
    T = len(frames)
    Hs = [None] * T
    info = [None] * T
    Hs[k] = H1
    info[k] = {**s1, "mode": "keyframe"}
    key_img, key_dets = img, dets[k]
    centre_img = np.array([960.0, 700.0])
    P0 = fmod.project(np.linalg.inv(H1), [centre_img])[0]
    reg_rejected = {}   # frame -> yd: registrations rejected as inconsistent with the chain
    reg_drift = {}   # frame -> yd: keyframe image point registered straight to frame t (SIFT), vs H(t)
    for direction in (1, -1):
        rng = range(k + 1, T) if direction == 1 else range(k - 1, -1, -1)
        since = 0
        for t in rng:
            since += 1
            prev = t - direction
            C = Cs[t] if direction == 1 else np.linalg.inv(Cs[prev])   # prev image -> t image
            Hp = C @ Hs[prev]; Hp /= Hp[2, 2]
            im = cv2.imread(os.path.join(vd, "frames", frames[t]))
            fit.prepare(im, dets[t], a.view)
            sp = fit.score(Hp)
            big = cq[max(t, prev)] < 0.6
            Hr, _ = fit.fit(Hp, basin=4.0 if big else 1.0, iters=40)
            if big:
                Hr, _ = fit.fit(Hr, basin=1.0, iters=40)
            sr = fit.score(Hr)
            # gating: accept re-fit if it does not lose inliers and the image moves < 25 px vs prediction
            probe = fmod.project(Hp, fit.P[::50]); probe_r = fmod.project(Hr, fit.P[::50])
            okp = np.isfinite(probe).all(1) & np.isfinite(probe_r).all(1)
            inside = okp & (probe[:, 0] > 0) & (probe[:, 0] < 1920) & (probe[:, 1] > 0) & (probe[:, 1] < 1080)
            shift = float(np.median(np.linalg.norm(probe[inside] - probe_r[inside], axis=1))) if inside.any() else 1e9
            accept = sr["inlier_frac"] >= sp["inlier_frac"] - 0.02 and shift < (60 if big else 25) and safe_inv(Hr) is not None
            Hs[t] = Hr if accept else Hp
            info[t] = {**(sr if accept else sp), "mode": "refit" if accept else "predicted", "shift_px": round(shift, 1),
                       "cam_inlier": round(cq[max(t, prev)], 3), "wide_basin": bool(big)}
            # re-anchor: every `anchor_every` frames (and after a fast pan) register the keyframe straight to frame t
            if a.anchor_every and (since >= a.anchor_every or big):
                since = 0
                try:
                    M, rinfo = register_to_reference(key_img, key_dets, im, dets[t])
                except Exception:
                    M, rinfo = None, {"inliers": 0}
                if M is not None and rinfo["inliers"] >= 40:
                    q = fmod.project(M, [centre_img])[0]
                    Ha, _ = fit.fit(M @ H1 / (M @ H1)[2, 2], basin=4.0, iters=40)
                    Ha, _ = fit.fit(Ha, basin=1.0, iters=40)
                    sa = fit.score(Ha)
                    cur = info[t]
                    inside_q = 0 <= q[0] < 1920 and 0 <= q[1] < 1080
                    dis = field_dist(Hs[t], q, P0) if inside_q else 0.0
                    dis = 0.0 if not np.isfinite(dis) else dis
                    # the re-anchored H wins if it matches the paint better, or if the chained H has drifted away from
                    # the direct keyframe registration (> 0.75 yd) and the re-anchored H matches the paint about as well
                    # (paint alone is ambiguous along the depth axis when the near sideline is hidden)
                    # but never jump more than reanchor_max_yd: yard lines repeat every 5 yd, so a wrong registration
                    # (repeating stands / scenery) can fit the paint as well as the right one
                    if dis <= a.reanchor_max_yd and (sa["inlier_frac"] > cur["inlier_frac"] + 0.01 or
                                                     (dis > 0.75 and sa["inlier_frac"] >= cur["inlier_frac"] - 0.05)):
                        Hs[t] = Ha
                        info[t] = {**sa, "mode": "reanchor", "cam_inlier": cur["cam_inlier"], "reg_inliers": rinfo["inliers"],
                                   "chain_vs_registration_yd": round(dis, 3)}
                    if inside_q:
                        rd = field_dist(Hs[t], q, P0)
                        if np.isfinite(rd) and dis <= a.reanchor_max_yd:   # inconsistent registrations are not drift
                            reg_drift[t] = rd
                        elif np.isfinite(rd):
                            reg_rejected[t] = round(dis, 2)

    # ---- stability: carry the keyframe image point of a fixed field point through the camera chain
    drift = [None] * T
    q = centre_img.copy(); drift[k] = 0.0
    for t in range(k + 1, T):
        q = fmod.project(Cs[t], [q])[0]
        drift[t] = field_dist(Hs[t], q, P0)
    q = centre_img.copy()
    for t in range(k - 1, -1, -1):
        Ci = safe_inv(Cs[t + 1])
        q = fmod.project(Ci, [q])[0] if Ci is not None else q
        drift[t] = field_dist(Hs[t], q, P0)

    res = [i["residual_px"] for i in info if i["residual_px"] is not None]
    if not res:
        raise SystemExit("calibration failed: the field model never matched the paint (no frame with inliers)")
    dfin = [d for d in drift if d is not None and np.isfinite(d)]
    drift = [d if (d is not None and np.isfinite(d)) else -1.0 for d in drift]
    summary = {"keyframe": k, "init_mode": "registered" if reg else "labelled_lines", "registration": reg,
               "keyframe_init": s0, "keyframe_refined": s1, "frames": T,
               "residual_px_median": round(float(np.median(res)), 2), "residual_px_p95": round(float(np.percentile(res, 95)), 2),
               "inlier_frac_median": round(float(np.median([i["inlier_frac"] for i in info])), 3),
               "refit_frac": round(float(np.mean([i["mode"] != "predicted" for i in info])), 3),
               "drift_yd_median": round(float(np.median(dfin)), 3) if dfin else None,
               "drift_yd_max": round(float(np.max(dfin)), 3) if dfin else None,
               "reanchor_frac": round(float(np.mean([i["mode"] == "reanchor" for i in info])), 3),
               "reg_drift_yd_median": round(float(np.median(list(reg_drift.values()))), 3) if reg_drift else None,
               "reg_drift_yd_max": round(float(np.max(list(reg_drift.values()))), 3) if reg_drift else None,
               "reg_drift_n": len(reg_drift), "registrations_rejected": len(reg_rejected),
               "stability_point_yd": [round(float(v), 2) for v in P0], "hashes": a.hashes, "hash_dir": a.hash_dir}
    json.dump({"summary": summary, "pts_s": pts, "H": [h.tolist() for h in Hs], "info": info,
               "drift_yd": [round(d, 4) for d in drift], "reg_drift_yd": {str(t): round(v, 4) for t, v in reg_drift.items()}}, open(os.path.join(vd, "calib.json"), "w"))
    print(json.dumps(summary))
    if a.overlay:
        import imageio_ffmpeg, subprocess
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        W, Hh = 960, 540
        proc = subprocess.Popen([ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{Hh}", "-r", "30",
                                 "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                                 os.path.join(vd, "grid_overlay.mp4")], stdin=subprocess.PIPE)
        for t in range(T):
            im = cv2.imread(os.path.join(vd, "frames", frames[t]))
            o = fmod.draw_grid(im, Hs[t])
            i = info[t]
            cv2.putText(o, f"f{t} {pts[t]:.2f}s {i['mode']} res {i['residual_px']} px inl {i['inlier_frac']:.2f} drift {drift[t]:.2f} yd",
                        (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            proc.stdin.write(cv2.resize(o, (W, Hh), interpolation=cv2.INTER_AREA).tobytes())
        proc.stdin.close(); proc.wait()


if __name__ == "__main__":
    main()
