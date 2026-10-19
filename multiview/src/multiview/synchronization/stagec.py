#!/usr/bin/env python
"""Stage C sync: time offset between the views from field-plane player positions (sam3env python).

  stagec.py PAIR_DIR [--apply] [--step 1]

Uses each view's field-masked detections (dets_all.json) and H(t) (calib.json): foot points -> yards. For every
candidate offset (wide_time = ez_time + offset, 1-frame steps over the whole feasible range), cost = mean over
overlapping End Zone frames of the median (over that frame's End Zone players) of the distance to the nearest Wide
player at the Wide frame closest in time (capped at 3 yd), in position + velocity space (velocity from a
nearest-neighbour link 0.2 s later, scaled by --vel-scale), frames weighted by End Zone motion. Needs >= 30 overlapping frames.
Confidence: 1 - best/median(cost) (how far the minimum stands out) and the gap to the best cost more than 0.3 s
away. With --apply and the End Zone snap missing (clip starts after the snap), writes the End Zone snap time
implied by the Wide snap into sync.json with method 'stage_c'. Always writes PAIR_DIR/stagec.json.
"""
import argparse, json, os

import numpy as np
from scipy.spatial import cKDTree


def field_points(pair, v):
    cal = json.load(open(os.path.join(pair, v, "calib.json")))
    dets = json.load(open(os.path.join(pair, v, "dets_all.json")))["frames"]
    out = []
    for t, fr in enumerate(dets):
        Hinv = np.linalg.inv(np.array(cal["H"][t]))
        pts = []
        for d in fr["detections"]:
            x1, y1, x2, y2 = d["box_xyxy"]
            p = Hinv @ np.array([(x1 + x2) / 2, y2, 1.0])
            if abs(p[2]) > 1e-12:   # sign of p[2] is arbitrary (H defined up to scale)
                pts.append(p[:2] / p[2])
        out.append(np.array(pts).reshape(-1, 2))
    return np.array(cal["pts_s"]), out


def with_velocity(P, ts, k=6, link=2.0):
    """Append a velocity estimate (yd/s) to each point: nearest neighbour k frames later (within `link` yd)."""
    out = []
    for t in range(len(P)):
        a = P[t]
        if len(a) == 0:
            out.append(np.zeros((0, 4))); continue
        u = min(len(P) - 1, t + k)
        b = P[u]
        v = np.zeros_like(a)
        if u != t and len(b):
            d, j = cKDTree(b).query(a)
            ok = d <= link
            v[ok] = (b[j[ok]] - a[ok]) / (ts[u] - ts[t])
        out.append(np.hstack([a, v]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair"); ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-overlap", type=int, default=30)
    ap.add_argument("--vel-scale", type=float, default=0.0, help="seconds: 1 yd/s velocity difference counts as this many yd (0 = positions only)")
    ap.add_argument("--all-frames", action="store_true", help="do not restrict to post-snap frames")
    a = ap.parse_args()
    tw, PW = field_points(a.pair, "wide")
    te, PE = field_points(a.pair, "ez")
    # position + velocity (yd/s scaled by a.vel_scale s) so static pre-snap formations do not fit every offset;
    # frames weighted by how much the End Zone players move
    PW = with_velocity(PW, tw); PE = with_velocity(PE, te)
    # post-snap only (static pre-snap formations match many offsets): Wide from its snap; End Zone from its snap
    # if detected, else every frame (a clip that starts after the snap is all post-snap)
    sync0 = json.load(open(os.path.join(a.pair, "sync.json")))
    w0 = sync0["wide"]["snap_pts"] if (sync0["wide"]["snap_pts"] is not None and not a.all_frames) else -1e9
    e0 = sync0["ez"]["snap_pts"] if (sync0["ez"]["snap_pts"] is not None and not a.all_frames) else -1e9
    PW = [p if t >= w0 else np.zeros((0, 4)) for p, t in zip(PW, tw)]
    PE = [p if t >= e0 else np.zeros((0, 4)) for p, t in zip(PE, te)]
    sc = np.array([1, 1, a.vel_scale, a.vel_scale])
    trees = [cKDTree(p * sc) if len(p) else None for p in PW]
    wE = [float(np.median(np.linalg.norm(p[:, 2:], axis=1))) + 0.3 if len(p) else 0.0 for p in PE]
    dt = float(np.median(np.diff(tw)))
    offs = np.arange(tw[0] - te[-1], tw[-1] - te[0] + 1e-9, dt)
    costs, n_ov = [], []
    for off in offs:
        per, ws = [], []
        for k, t in enumerate(te):
            tt = t + off
            if tt < tw[0] or tt > tw[-1] or len(PE[k]) == 0:
                continue
            j = int(round((tt - tw[0]) / dt))
            if trees[j] is None or len(PE[k]) == 0:
                continue
            d, _ = trees[j].query(PE[k] * sc)
            per.append(float(np.median(np.minimum(d, 3.0)))); ws.append(wE[k])
        n_ov.append(len(per))
        costs.append(float(np.average(per, weights=ws)) if len(per) >= a.min_overlap else np.nan)
    costs = np.array(costs)
    if np.all(np.isnan(costs)):
        res = {"ok": False, "reason": "no offset with enough overlapping frames"}
    else:
        i = int(np.nanargmin(costs))
        far = np.abs(offs - offs[i]) > 0.3
        second = float(np.nanmin(costs[far])) if np.any(far & ~np.isnan(costs)) else float("nan")
        med = float(np.nanmedian(costs))
        res = {"ok": True, "offset_s": round(float(offs[i]), 4), "best_cost_yd": round(float(costs[i]), 3),
               "median_cost_yd": round(med, 3), "second_best_cost_yd_outside_0.3s": round(second, 3),
               "confidence": round(1 - float(costs[i]) / med, 3) if med > 0 else None,
               "margin_yd": round(second - float(costs[i]), 3), "overlap_frames": n_ov[i],
               "definition": "wide_time = ez_time + offset"}
    sync_f = os.path.join(a.pair, "sync.json")
    sync = json.load(open(sync_f))
    if res.get("ok") and sync.get("offset_snap_s") is not None:
        res["snap_offset_s"] = sync["offset_snap_s"]
        res["stage_c_minus_snap_s"] = round(res["offset_s"] - sync["offset_snap_s"], 3)
    if a.apply and res.get("ok") and sync["ez"]["snap_pts"] is None and sync["wide"]["snap_pts"] is not None:
        ez_snap = sync["wide"]["snap_pts"] - res["offset_s"]
        sync["ez"]["snap_pts"] = round(ez_snap, 4)
        sync["ez"]["snap_frame"] = int(round((ez_snap - te[0]) / dt))
        sync["ez"]["method"] = "stage_c"
        sync["offset_snap_s"] = res["offset_s"]
        sync["sync_method"] = "stage_c"
        sync["stage_c_confidence"] = res["confidence"]
        json.dump(sync, open(sync_f, "w"), indent=1)
        res["applied"] = True
    json.dump({**res, "curve": {"offsets_s": [round(float(o), 3) for o in offs[::3]],
                                "cost_yd": [None if np.isnan(c) else round(float(c), 3) for c in costs[::3]]}},
              open(os.path.join(a.pair, "stagec.json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
