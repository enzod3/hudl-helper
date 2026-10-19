#!/usr/bin/env python
"""Fit a per-view noise model on half the plays and evaluate fusion on the other half (sam3env python).

  fit_fusion.py DOTS.json [--pairs-root pairs] [--gate 3.0]

Split by clip id parity: even -> train, odd -> held-out test.
For each play: dots -> yards through the Wide H at frame_t (reference); the offense slots at that time from
recon.jsonl are matched to the dots (Hungarian, gate, on the recon position); for each match the slot's raw
per-view observations (obs_sideline, obs_endzone) give residuals vs the dot.
Noise model per view: residual along the camera's depth axis and across it (Wide: depth = y, cross = x;
End Zone: depth = x, cross = y), sigma = a + b * distance from the camera's nearest visible ground point
(the field point under the bottom-centre of the image at that frame). Fitted by robust least squares on |residual|
(scaled so a Gaussian's sigma comes out: sigma ~ 1.2533 * mean|r|) on the training half.
Fusion on the test half: inverse-covariance weighting of the two raw observations with the fitted sigmas (plus a
per-view mean bias fitted on train), single view when only one exists. Reported for test: fused (fitted), recon's
own fused output, Wide-only, End Zone-only, End Zone raw (no cross-view bias correction).
"""
import argparse, json, os

import numpy as np
from scipy.optimize import linear_sum_assignment


def to_field(H, u, v):
    p = np.linalg.inv(H) @ np.array([u, v, 1.0])
    return p[:2] / p[2]


def gather(plays, root, gate):
    rows, info = [], []
    for p in plays:
        P = os.path.join(root, p["clip"])
        try:
            cw = json.load(open(os.path.join(P, "wide", "calib.json")))
            ce = json.load(open(os.path.join(P, "ez", "calib.json")))
            sync = json.load(open(os.path.join(P, "sync.json")))
            rec = [json.loads(l) for l in open(os.path.join(P, "recon", "recon.jsonl"))]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            info.append({"clip": p["clip"], "status": f"missing {getattr(e, 'filename', e)}"}); continue
        tw = np.array(cw["pts_s"]); sf = int(np.argmin(np.abs(tw - p["frame_t"])))
        H = np.array(cw["H"][sf])
        ref = np.array([to_field(H, m["x"] * 1920, m["y"] * 1080) for m in p["marks"]])
        tau = float(tw[sf] - sync["wide"]["snap_pts"])
        ts = sorted({r["t"] for r in rec})
        tn = min(ts, key=lambda t: abs(t - tau))
        if abs(tn - tau) > 0.05:
            info.append({"clip": p["clip"], "status": f"dot time {tau:+.2f}s outside window"}); continue
        at = [r for r in rec if r["t"] == tn]
        # nearest visible ground of each camera at this time (bottom-centre of the image)
        te = np.array(ce["pts_s"]) - sync["ez"]["snap_pts"]
        se = int(np.argmin(np.abs(te - tau)))
        near_w = to_field(H, 960, 1079)
        near_e = to_field(np.array(ce["H"][se]), 960, 1079)
        best = None
        for team in ("A", "B"):
            mine = [r for r in at if r["team"] == team]
            pos = np.array([[r["x_yards"], r["y_yards"]] for r in mine]).reshape(-1, 2)
            if len(pos) == 0:
                continue
            D = np.linalg.norm(ref[:, None] - pos[None], axis=2)
            ri, ci = linear_sum_assignment(D)
            ok = D[ri, ci] <= gate
            sc = (int(ok.sum()), -float(D[ri, ci][ok].sum()))
            if best is None or sc > best[0]:
                best = (sc, [(i, mine[j]) for i, j, k in zip(ri, ci, ok) if k])
        matched = best[1] if best else []
        for i, r in matched:
            rows.append({"clip": p["clip"], "ref": ref[i], "recon": np.array([r["x_yards"], r["y_yards"]]),
                         "w": np.array(r["obs_sideline"]) if r.get("obs_sideline") else None,
                         "e": np.array(r["obs_endzone"]) if r.get("obs_endzone") else None,
                         "e_raw": np.array(r["obs_endzone_raw"]) if r.get("obs_endzone_raw") else None,
                         "near_w": near_w, "near_e": near_e})
        info.append({"clip": p["clip"], "status": "ok", "dots": len(ref), "matched": len(matched)})
    return rows, info


def axes(view):
    # (depth axis index, cross axis index)
    return (1, 0) if view == "w" else (0, 1)


def fit_view(rows, view):
    R = [r for r in rows if r[view] is not None]
    if len(R) < 8:
        return None
    dep, crs = axes(view)
    res = np.array([r[view] - r["ref"] for r in R])
    dist = np.array([np.linalg.norm(r["ref"] - r["near_" + view]) for r in R])
    bias = np.median(res, 0)
    out = {"n": len(R), "bias_xy": bias.round(3).tolist()}
    for name, ax in (("depth", dep), ("cross", crs)):
        a_ = np.abs(res[:, ax] - bias[ax]) * 1.2533
        A = np.c_[np.ones(len(dist)), dist]
        coef = np.linalg.lstsq(A, a_, rcond=None)[0]
        for _ in range(3):  # Huber-ish reweighting
            r_ = a_ - A @ coef
            w = 1.0 / np.maximum(1.0, np.abs(r_) / (1.4826 * np.median(np.abs(r_)) + 1e-6))
            coef = np.linalg.lstsq(A * w[:, None], a_ * w, rcond=None)[0]
        coef = np.maximum(coef, [0.05, 0.0])
        out[name] = {"a": round(float(coef[0]), 4), "b": round(float(coef[1]), 5)}
    out["dist_range"] = [round(float(dist.min()), 1), round(float(dist.max()), 1)]
    return out


def cov(model, view, r):
    dep, crs = axes(view)
    d = np.linalg.norm(r[view] - r["near_" + view])
    s = np.zeros(2)
    s[dep] = model["depth"]["a"] + model["depth"]["b"] * d
    s[crs] = model["cross"]["a"] + model["cross"]["b"] * d
    return np.diag(s ** 2)


def stats(d):
    d = np.asarray(d)
    if len(d) == 0:
        return {"n": 0}
    return {"n": int(len(d)), "median_yd": round(float(np.median(d)), 3), "p95_yd": round(float(np.percentile(d, 95)), 3),
            "mean_yd": round(float(np.mean(d)), 3)}


def evaluate(rows, models, n_dots):
    out = {}
    errs = {"fused_fitted": [], "recon_fused": [], "wide_only": [], "ez_only": [], "ez_raw": []}
    for r in rows:
        errs["recon_fused"].append(np.linalg.norm(r["recon"] - r["ref"]))
        zs, Cs = [], []
        for v in ("w", "e"):
            if r[v] is not None and models.get(v):
                zs.append(r[v] - np.array(models[v]["bias_xy"])); Cs.append(cov(models[v], v, r))
        if r["w"] is not None:
            errs["wide_only"].append(np.linalg.norm(r["w"] - r["ref"]))
        if r["e"] is not None:
            errs["ez_only"].append(np.linalg.norm(r["e"] - r["ref"]))
        if r["e_raw"] is not None:
            errs["ez_raw"].append(np.linalg.norm(r["e_raw"] - r["ref"]))
        if zs:
            W = sum(np.linalg.inv(C) for C in Cs)
            z = np.linalg.solve(W, sum(np.linalg.inv(C) @ z for z, C in zip(zs, Cs)))
            errs["fused_fitted"].append(np.linalg.norm(z - r["ref"]))
    for k, v in errs.items():
        out[k] = {**stats(v), "of_dots": n_dots}
    # same-subset comparison: matches observed by BOTH views
    both = [r for r in rows if r["w"] is not None and r["e"] is not None]
    sub = {"n": len(both)}
    if both and models.get("w") and models.get("e"):
        fz = []
        for r in both:
            zs = [r["w"] - np.array(models["w"]["bias_xy"]), r["e"] - np.array(models["e"]["bias_xy"])]
            Cs = [cov(models["w"], "w", r), cov(models["e"], "e", r)]
            W = sum(np.linalg.inv(C) for C in Cs)
            z = np.linalg.solve(W, sum(np.linalg.inv(C) @ zz for zz, C in zip(zs, Cs)))
            fz.append(np.linalg.norm(z - r["ref"]))
        sub["fused_fitted"] = stats(fz)
        sub["wide_only"] = stats([np.linalg.norm(r["w"] - r["ref"]) for r in both])
        sub["ez_only"] = stats([np.linalg.norm(r["e"] - r["ref"]) for r in both])
        sub["ez_raw"] = stats([np.linalg.norm(r["e_raw"] - r["ref"]) for r in both])
    out["both_views_subset"] = sub
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dots"); ap.add_argument("--pairs-root", default="pairs"); ap.add_argument("--gate", type=float, default=3.0)
    a = ap.parse_args()
    plays = json.load(open(a.dots))
    train = [p for p in plays if int(p["clip"]) % 2 == 0]
    test = [p for p in plays if int(p["clip"]) % 2 == 1]
    rtr, itr = gather(train, a.pairs_root, a.gate)
    rte, ite = gather(test, a.pairs_root, a.gate)
    models = {"w": fit_view(rtr, "w"), "e": fit_view(rtr, "e")}
    n_dots_test = sum(i.get("dots", 0) for i in ite if i["status"] == "ok")
    res = {"split": "clip id parity: even=train, odd=test",
           "train": {"plays_ok": sum(i["status"] == "ok" for i in itr), "plays": len(train), "matches": len(rtr)},
           "test": {"plays_ok": sum(i["status"] == "ok" for i in ite), "plays": len(test), "matches": len(rte), "dots": n_dots_test},
           "noise_model": models, "test_results": evaluate(rte, models, n_dots_test),
           "train_results": evaluate(rtr, models, sum(i.get("dots", 0) for i in itr if i["status"] == "ok")),
           "per_play": itr + ite}
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(a.dots)), "fit_fusion.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_play"}, indent=1))


if __name__ == "__main__":
    main()
