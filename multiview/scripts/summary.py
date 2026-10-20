#!/usr/bin/env python
"""Per-play summary CSV across a dots file (sam3env python):
  summary.py DOTS.json [--eval eval_dots.json] [--out plays_summary.csv]
Columns: clip, status, failed_step, sync_method, wide_snap, ez_snap, offset_s, stagec_offset_s, stagec_minus_snap_s,
stagec_conf, stagec_margin_yd, calib residual / paint inliers / registration drift (median, max) per view,
slots A/B, frames both/one/none, players_associated, crossview median/p95, dots, matched + median error per mode.
Also writes worst_plays.json: plays ranked by problems, each with the reason(s).
"""
import argparse, csv, json, os


def load(p, default=None):
    try:
        return json.load(open(p))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dots"); ap.add_argument("--eval", default=None); ap.add_argument("--out", default="plays_summary.csv")
    ap.add_argument("--pairs-root", default="pairs")
    a = ap.parse_args()
    plays = json.load(open(a.dots))
    ev = load(a.eval, {}) if a.eval else {}
    evp = {p["clip"]: p for p in ev.get("plays", [])}
    rows, worst = [], []
    for p in plays:
        c = p["clip"]; P = os.path.join(a.pairs_root, c)
        rp = load(os.path.join(P, "run_pair.json"), {"steps": {}})
        st = rp["steps"]
        failed = [k for k, v in st.items() if not v["ok"]]
        sync = load(os.path.join(P, "sync.json"), {})
        sc = load(os.path.join(P, "stagec.json"), {})
        summ = load(os.path.join(P, "recon", "recon_summary.json"), {})
        cal = {v: (load(os.path.join(P, v, "calib.json"), {}) or {}).get("summary", {}) for v in ("wide", "ez")}
        e = evp.get(c, {})
        pm = e.get("per_mode", {})
        r = {"clip": c, "play_type": p.get("pt"),
             "status": "ok" if st and not failed and "recon" in st else ("failed" if failed else "incomplete"),
             "failed_step": failed[0] if failed else "", "error": (st[failed[0]].get("error", "") if failed else "")[:120],
             "sync_method": sync.get("sync_method", "snap" if sync else ""),
             "wide_snap_s": (sync.get("wide") or {}).get("snap_pts"), "ez_snap_s": (sync.get("ez") or {}).get("snap_pts"),
             "hudl_snap_s": p.get("snap_t"), "offset_s": sync.get("offset_snap_s"),
             "stagec_offset_s": sc.get("offset_s"), "stagec_minus_snap_s": sc.get("stage_c_minus_snap_s"),
             "stagec_conf": sc.get("confidence"), "stagec_margin_yd": sc.get("margin_yd")}
        for v in ("wide", "ez"):
            s = cal[v]
            r[f"{v}_res_px"] = s.get("residual_px_median"); r[f"{v}_inlier"] = s.get("inlier_frac_median")
            r[f"{v}_regdrift_med_yd"] = s.get("reg_drift_yd_median"); r[f"{v}_regdrift_max_yd"] = s.get("reg_drift_yd_max")
        sf = summ.get("slots_filled", {}); ob = summ.get("slot_frames_observed_of_22", {})
        cv = summ.get("crossview_disagreement_fused_frames", {})
        r.update({"slots_A": sf.get("A"), "slots_B": sf.get("B"), "both": ob.get("both"), "one": ob.get("one"), "none": ob.get("none"),
                  "players_associated": summ.get("players_associated"), "crossview_med_yd": cv.get("median_yd"),
                  "crossview_p95_yd": cv.get("p95_yd"), "los_x": summ.get("los_x_at_snap"),
                  "ball_minus_los_yd": e.get("ball_minus_los_x_yd"), "dots": e.get("n_dots", len(p["marks"])),
                  "dot_eval_error": e.get("error", "")})
        for m in ("fused", "wide", "ez", "ez_raw"):
            r[f"{m}_matched"] = (pm.get(m) or {}).get("matched"); r[f"{m}_med_yd"] = (pm.get(m) or {}).get("median_yd")
        rows.append(r)
        # problems
        why = []
        if r["status"] != "ok":
            why.append(f"pipeline {r['status']} at {r['failed_step'] or '?'}: {r['error']}")
        if r["sync_method"] in ("stage_c", "stage_c_override") and (r["stagec_conf"] or 0) < 0.25:
            why.append(f"{r['sync_method']} sync with low confidence {r['stagec_conf']}")
        if r["stagec_minus_snap_s"] is not None and abs(r["stagec_minus_snap_s"]) > 0.5 and r["sync_method"] == "snap":
            why.append(f"snap and Stage C offsets disagree by {r['stagec_minus_snap_s']} s")
        for v in ("wide", "ez"):
            if (r[f"{v}_res_px"] or 0) > 2.5 or (r[f"{v}_inlier"] is not None and r[f"{v}_inlier"] < 0.4):
                why.append(f"{v} calibration weak (res {r[f'{v}_res_px']} px, inliers {r[f'{v}_inlier']})")
            if (r[f"{v}_regdrift_max_yd"] or 0) > 2.0:
                why.append(f"{v} calibration drift up to {r[f'{v}_regdrift_max_yd']} yd")
        if r["ball_minus_los_yd"] is not None and abs(r["ball_minus_los_yd"]) > 2.0:
            why.append(f"LOS estimate {r['ball_minus_los_yd']} yd from the ball dot")
        if r["dot_eval_error"]:
            why.append(r["dot_eval_error"])
        elif r["fused_matched"] is not None and r["dots"] and r["fused_matched"] < 0.5 * r["dots"]:
            why.append(f"only {r['fused_matched']}/{r['dots']} dots matched")
        if r["fused_med_yd"] is not None and r["fused_med_yd"] > 2.0:
            why.append(f"fused median error {r['fused_med_yd']} yd")
        if r["none"] is not None and r["none"] > 0.15:
            why.append(f"{round(100 * r['none'])}% of slot-frames unobserved")
        if why:
            worst.append({"clip": c, "n_problems": len(why), "reasons": why})
    keys = list(rows[0].keys())
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    worst.sort(key=lambda x: -x["n_problems"])
    json.dump(worst, open(os.path.join(os.path.dirname(os.path.abspath(a.out)), "worst_plays.json"), "w"), indent=1)
    print(json.dumps({"plays": len(rows), "ok": sum(r["status"] == "ok" for r in rows),
                      "sync_methods": {m: sum(r["sync_method"] == m for r in rows) for m in {r["sync_method"] for r in rows}},
                      "plays_with_problems": len(worst)}))


if __name__ == "__main__":
    main()
