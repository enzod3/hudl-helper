#!/usr/bin/env python
"""Accuracy against hand-tapped ground points (sam3env python):  eval_dots.py DOTS.json [--pairs-root pairs]

DOTS.json: [{clip, snap_t, frame_t, ball {x,y}, marks [{role, x, y}]}] with x,y normalised on the Wide frame at
frame_t seconds into the Wide clip. Reference = dots mapped through the Wide H at that frame (so it shares the Wide
calibration); positions compared at the same time from recon.jsonl:
  fused   : recon slot position (x_yards, y_yards)
  wide    : the slot's raw Wide observation at that frame (obs_sideline)
  ez      : raw End Zone observation, bias-corrected onto Wide (obs_endzone)
  ez_raw  : raw End Zone observation without the cross-view bias correction (obs_endzone_raw) - the check that is
            independent of the Wide calibration except for the shared field model
Matching: Hungarian with a 3 yd gate against the slots of the team that matches the dots best (offense).
Also: ball dot vs the recon LOS estimate and the play's yard-line grid.
Writes eval_dots.json next to DOTS.json and prints a summary.
"""
import argparse, json, os

import numpy as np
from scipy.optimize import linear_sum_assignment


def to_field(H, u, v):
    p = np.linalg.inv(H) @ np.array([u, v, 1.0])
    return p[:2] / p[2]


def match(ref, pos, gate=3.0):
    if len(ref) == 0 or len(pos) == 0:
        return np.zeros(0), []
    D = np.linalg.norm(ref[:, None] - pos[None], axis=2)
    r, c = linear_sum_assignment(D)
    ok = D[r, c] <= gate
    return D[r, c][ok], list(zip(r[ok], c[ok]))


def stats(d):
    d = np.asarray(d)
    if len(d) == 0:
        return {"n": 0}
    return {"n": int(len(d)), "median_yd": round(float(np.median(d)), 3), "p95_yd": round(float(np.percentile(d, 95)), 3),
            "mean_yd": round(float(np.mean(d)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dots"); ap.add_argument("--pairs-root", default="pairs")
    ap.add_argument("--gate", type=float, default=3.0)
    ap.add_argument("--recon-name", default="recon", help="recon output folder name inside each play folder")
    ap.add_argument("--out", default=None, help="output json (default: eval_dots.json next to DOTS.json)")
    a = ap.parse_args()
    plays = json.load(open(a.dots))
    res, allerr = [], {m: [] for m in ("fused", "wide", "ez", "ez_raw")}
    tot_dots, tot_matched = 0, {m: 0 for m in allerr}
    for p in plays:
        P = os.path.join(a.pairs_root, p["clip"])
        r = {"clip": p["clip"]}
        try:
            cal = json.load(open(os.path.join(P, "wide", "calib.json")))
            sync = json.load(open(os.path.join(P, "sync.json")))
            rows = [json.loads(l) for l in open(os.path.join(P, a.recon_name, "recon.jsonl"))]
            summ = json.load(open(os.path.join(P, a.recon_name, "recon_summary.json")))
        except FileNotFoundError as e:
            r["error"] = f"missing {e.filename}"; res.append(r); continue
        pts = np.array(cal["pts_s"])
        sf = int(np.argmin(np.abs(pts - p["frame_t"])))
        H = np.array(cal["H"][sf])
        ref = np.array([to_field(H, m["x"] * 1920, m["y"] * 1080) for m in p["marks"]])
        ball = to_field(H, p["ball"]["x"] * 1920, p["ball"]["y"] * 1080)
        tau = float(pts[sf] - sync["wide"]["snap_pts"])
        frames = sorted({rw["t"] for rw in rows})
        if not frames:
            r["error"] = "empty reconstruction"; res.append(r); tot_dots += len(ref); continue
        t_near = min(frames, key=lambda t: abs(t - tau))
        r.update({"wide_frame": sf, "tau_from_detected_snap": round(tau, 3), "recon_t_used": t_near,
                  "detected_snap_wide_s": sync["wide"]["snap_pts"], "hudl_snap_t": p["snap_t"],
                  "snap_diff_s": round(sync["wide"]["snap_pts"] - p["snap_t"], 3),
                  "ball_yd": ball.round(2).tolist(), "los_x_recon": summ.get("los_x_at_snap"),
                  "ball_minus_los_x_yd": round(float(ball[0] - summ["los_x_at_snap"]), 2) if summ.get("los_x_at_snap") else None,
                  "wide_calib_residual_px": cal["info"][sf]["residual_px"], "n_dots": len(ref)})
        if abs(t_near - tau) > 0.05:
            r["error"] = f"dot time {tau:+.2f}s outside the reconstructed window"
            res.append(r); tot_dots += len(ref); continue
        at = [rw for rw in rows if rw["t"] == t_near]
        best = None
        for team in ("A", "B"):
            pos = np.array([[rw["x_yards"], rw["y_yards"]] for rw in at if rw["team"] == team]).reshape(-1, 2)
            d, _ = match(ref, pos, a.gate)
            score = (len(d), -float(np.sum(d)))
            if best is None or score > best[0]:
                best = (score, team)
        team = best[1]
        r["offense_team_matched"] = team
        mine = [rw for rw in at if rw["team"] == team]
        per = {}
        for mode, key in (("fused", None), ("wide", "obs_sideline"), ("ez", "obs_endzone"), ("ez_raw", "obs_endzone_raw")):
            if key is None:
                pos = np.array([[rw["x_yards"], rw["y_yards"]] for rw in mine]).reshape(-1, 2)
            else:
                pos = np.array([rw[key] for rw in mine if rw.get(key)]).reshape(-1, 2)
            d, pr = match(ref, pos, a.gate)
            per[mode] = {**stats(d), "matched": len(d), "of_dots": len(ref)}
            allerr[mode] += list(d); tot_matched[mode] += len(d)
            if mode == "fused":
                r["matches"] = [{"role": p["marks"][i]["role"], "ref": ref[i].round(2).tolist(),
                                 "slot": mine[j]["player_id"], "slot_xy": pos[j].round(2).tolist(), "err_yd": round(float(np.linalg.norm(ref[i] - pos[j])), 2)}
                                for i, j in pr]
        r["per_mode"] = per
        tot_dots += len(ref)
        res.append(r)
    overall = {m: {**stats(allerr[m]), "matched": tot_matched[m], "of_dots": tot_dots} for m in allerr}
    out = {"gate_yd": a.gate, "overall": overall, "plays": res}
    json.dump(out, open(a.out or os.path.join(os.path.dirname(os.path.abspath(a.dots)), "eval_dots.json"), "w"), indent=1)
    print(json.dumps({"overall": overall}, indent=1))
    for r in res:
        print(r["clip"], r.get("error", ""), "snapdiff", r.get("snap_diff_s"), "ball-LOS", r.get("ball_minus_los_x_yd"),
              {m: (v.get("matched"), v.get("median_yd")) for m, v in r.get("per_mode", {}).items()})


if __name__ == "__main__":
    main()
