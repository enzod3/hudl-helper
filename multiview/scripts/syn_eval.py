#!/usr/bin/env python
"""Phase 10: tracking metrics on held-out synthetic plays with exact ground truth (sam3env python, ~/sam3_work).

  syn_eval.py [--root pairs_syn] [--seeds 1190-1199] [--weights ft/run_real2/checkpoint_best_total.pth]
              [--modes sam det] [--thr 1.5]

Per play (Wide only, ground-truth homography so calibration error is excluded): RF-DETR (fine-tuned) + field mask on
every frame -> SAM 3.1 tracking from frame 0 -> recon.py --views wide --foot-head (modes: full SAM masks, detector
tracks). Metrics in field coordinates against ground-truth on-field players (officials excluded), per frame Hungarian
matching with a --thr yd gate:
  DetA = TP/(TP+FN+FP); AssA = mean over TP of the TPA/(TPA+FNA+FPA) of its (gt id, pred id) pair; HOTA = sqrt(DetA*AssA)
  (single threshold); IDF1 (global one-to-one id assignment maximising identity-true positives); ID switches (a gt
  id's matched pred id changes); fragmentations (a gt id's matched run is interrupted and resumes); identity
  completeness (fraction of gt players whose best pred id covers >= 90% of their frames); position error of matches.
Writes ROOT/syn_metrics.json.
"""
import argparse, json, os, subprocess, sys

import numpy as np
from scipy.optimize import linear_sum_assignment
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PY = os.environ.get("MULTIVIEW_SAM_PY", os.path.expanduser("~/sam3env/bin/python"))
PYR = os.environ.get("MULTIVIEW_RFDETR_PY", os.path.expanduser("~/rfdetr_env/bin/python"))


LOCK = os.path.expanduser("~/sam3_work/logs/gpu.lock")


def sh(cmd, log, gpu=False):
    if gpu:
        cmd = ["flock", LOCK] + cmd   # same lock file run_pair.py uses (flock(2) on both sides)
    with open(log, "a") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{[c for c in cmd if c.endswith('.py')][:1]} failed, see {log}")


def prepare(P, weights, log):
    vd = os.path.join(P, "wide")
    frames = sorted(os.listdir(os.path.join(vd, "frames")))
    if not os.path.exists(os.path.join(vd, "dets_all.json")):
        open(os.path.join(vd, "frames.txt"), "w").write("".join(os.path.join(vd, "frames", f) + "\n" for f in frames))
        sh([PYR, os.path.join(_REPO, "src/multiview/detection/detect.py"), "--model", "large", "--tiles", "2", "--view", "wide", "--warmup", "1",
            "--list", os.path.join(vd, "frames.txt"), "--out", os.path.join(vd, "dets_raw.json")] + (["--weights", weights] if weights else []), log, gpu=True)
        sh([PY, os.path.join(_REPO, "src/multiview/detection/apply_fieldmask.py"), os.path.join(vd, "dets_raw.json"), os.path.join(vd, "dets_all.json"), "--view", "wide"], log)
    td = os.path.join(P, "track_wide_crop")
    if not os.path.exists(os.path.join(td, "stats.json")):
        os.makedirs(os.path.join(td, "frames"), exist_ok=True)
        for k, f in enumerate(frames):
            dst = os.path.join(td, "frames", f"{k:05d}.jpg")
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(os.path.join(vd, "frames", f)), dst)
        json.dump({"video": None, "start": 0.0, "end": len(frames) / 30.0, "fps": 30, "width": 1920, "height": 1080,
                   "num_frames": len(frames), "source_frame_start": 0, "snap_frame_in_clip": 0}, open(os.path.join(td, "frames.json"), "w"))
        d = json.load(open(os.path.join(vd, "dets_all.json"))); d["frames"] = d["frames"][:1]
        json.dump(d, open(os.path.join(td, "dets0.json"), "w"))
        sh([PY, os.path.join(_REPO, "src/multiview/segmentation/segment.py"), "track", "--out-dir", td, "--init-dets", os.path.join(td, "dets0.json"),
            "--min-score", "0.3", "--max-objects", "30", "--dtype", "fp16"], log, gpu=True)


def metrics(P, recon_name, thr):
    gt = json.load(open(os.path.join(P, "gt.json")))
    rows = [json.loads(l) for l in open(os.path.join(P, recon_name, "recon.jsonl"))]
    by_f = {}
    for r in rows:
        if r["source"] != "inferred":   # score only observed / fused positions (inferred gaps reported separately)
            by_f.setdefault(r["frame"], []).append(r)
    inferred = sum(1 for r in rows if r["source"] == "inferred")
    TP = FN = FP = 0
    pairs, matches_by_gt, errs = {}, {}, []
    gt_frames, pred_frames = {}, {}
    T = len(gt)
    for f in range(T):
        G = [(p["trackId"], p["fieldX"], p["fieldY"]) for p in gt[f] if p["onField"] and p["role"] != "REF"]
        Pp = [(r["player_id"], r["x_yards"], r["y_yards"]) for r in by_f.get(f, [])]
        for g in G:
            gt_frames[g[0]] = gt_frames.get(g[0], 0) + 1
        for p in Pp:
            pred_frames[p[0]] = pred_frames.get(p[0], 0) + 1
        if G and Pp:
            D = np.linalg.norm(np.array([g[1:] for g in G])[:, None] - np.array([p[1:] for p in Pp])[None], axis=2)
            ri, ci = linear_sum_assignment(D)
            ok = D[ri, ci] <= thr
            ri, ci = ri[ok], ci[ok]
        else:
            ri, ci = np.array([], int), np.array([], int)
        TP += len(ri); FN += len(G) - len(ri); FP += len(Pp) - len(ci)
        for i, j in zip(ri, ci):
            key = (G[i][0], Pp[j][0]); pairs[key] = pairs.get(key, 0) + 1
            matches_by_gt.setdefault(G[i][0], []).append((f, Pp[j][0]))
            errs.append(float(D[i, j]))
    det_a = TP / max(1, TP + FN + FP)
    ass = []
    for (g, p), n in pairs.items():
        tpa = n; fna = gt_frames[g] - n; fpa = pred_frames[p] - n
        ass += [tpa / (tpa + fna + fpa)] * n
    ass_a = float(np.mean(ass)) if ass else 0.0
    # IDF1: one-to-one assignment of gt ids to pred ids maximising shared matched frames
    gids, pids = sorted(gt_frames), sorted(pred_frames)
    W = np.zeros((len(gids), len(pids)))
    for (g, p), n in pairs.items():
        W[gids.index(g), pids.index(p)] = n
    ri, ci = linear_sum_assignment(-W) if W.size else (np.array([]), np.array([]))
    idtp = float(W[ri, ci].sum()) if W.size else 0.0
    idf1 = 2 * idtp / max(1, sum(gt_frames.values()) + sum(pred_frames.values()))
    idsw = frag = 0; complete = 0
    for g, ms in matches_by_gt.items():
        ms.sort()
        for (f0, p0), (f1, p1) in zip(ms[:-1], ms[1:]):
            idsw += int(p1 != p0)
            frag += int(f1 > f0 + 1)
        best = max((n for (gg, p), n in pairs.items() if gg == g), default=0)
        complete += int(best >= 0.9 * gt_frames[g])
    return {"HOTA": round(float(np.sqrt(det_a * ass_a)), 4), "DetA": round(det_a, 4), "AssA": round(ass_a, 4),
            "IDF1": round(idf1, 4), "ID_switches": idsw, "fragmentations": frag,
            "identity_complete_frac": round(complete / max(1, len(gt_frames)), 4), "gt_players": len(gt_frames),
            "recall": round(TP / max(1, TP + FN), 4), "precision": round(TP / max(1, TP + FP), 4),
            "pos_err_median_yd": round(float(np.median(errs)), 3) if errs else None,
            "pos_err_p95_yd": round(float(np.percentile(errs, 95)), 3) if errs else None,
            "inferred_slot_frames": inferred, "frames": T}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="pairs_syn"); ap.add_argument("--seeds", default="1190-1199")
    ap.add_argument("--weights", default="ft/run_real2/checkpoint_best_total.pth")
    ap.add_argument("--modes", nargs="+", default=["sam", "det"]); ap.add_argument("--thr", type=float, default=1.5)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    res = {"thr_yd": a.thr, "per_play": {}, "modes": a.modes}
    for seed in range(lo, hi + 1):
        P = os.path.join(a.root, str(seed))
        if not os.path.isdir(P):
            continue
        log = os.path.join(P, "syn_eval.log")
        try:
            prepare(P, a.weights, log)
            out = {}
            for m in a.modes:
                name = f"recon_{m}"
                sh([PY, os.path.join(_REPO, "src/multiview/reconstruction/recon.py"), P, "--views", "wide", "--assoc-gate", "2.5", "--foot-head", "foot/run1/head.pt",
                    "--obs", m, "--out-name", name], log)
                out[m] = metrics(P, name, a.thr)
            res["per_play"][str(seed)] = out
            print(seed, json.dumps(out), flush=True)
        except Exception as e:
            res["per_play"][str(seed)] = {"error": str(e)}
            print(seed, "ERROR", e, flush=True)
    for m in a.modes:
        vals = [v[m] for v in res["per_play"].values() if m in v]
        if vals:
            res[f"mean_{m}"] = {k: round(float(np.mean([x[k] for x in vals if x[k] is not None])), 4)
                                for k in vals[0] if isinstance(vals[0][k], (int, float))}
    json.dump(res, open(os.path.join(a.root, "syn_metrics.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k.startswith("mean_")}, indent=1))


if __name__ == "__main__":
    main()
