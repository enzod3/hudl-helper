#!/usr/bin/env python
"""Second SAM pass for one play (sam3env python):  second_pass.py PAIR_DIR [--suffix _crop] [--min-run 6]

Reads PAIR_DIR/recon/recon.jsonl (first pass). For each view, a slot 'needs help' at frame f when it lies inside
that camera's image but is not observed there (visibility 'occluded'), or is inferred / conflicting, for a run of
>= min-run frames. The frame with the most such slots is chosen per view; SAM 3.1 is re-prompted there with all
field-masked RF-DETR detections of that frame (dets_all.json: tiled + masked) and propagated to the end of the clip.
Output: PAIR_DIR/track_<view>_p2/ (segment.py layout; frames.json has source_frame_start) and
PAIR_DIR/recon/second_pass.json. recon.py --extra-suffix _p2 then merges these tracks.
"""
import argparse, fcntl, json, os, subprocess

import numpy as np
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

PY = os.environ.get("MULTIVIEW_SAM_PY", os.path.expanduser("~/sam3env/bin/python"))
VIEW_NAME = {"wide": "sideline", "ez": "endzone"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair")
    ap.add_argument("--suffix", default="_crop")
    ap.add_argument("--min-run", type=int, default=6)
    a = ap.parse_args()
    P = a.pair
    rows = [json.loads(l) for l in open(os.path.join(P, "recon", "recon.jsonl"))]
    NF = max(r["frame"] for r in rows) + 1
    out = {"views": {}}
    for v in ("wide", "ez"):
        need = np.zeros(NF, int)
        by_slot = {}
        for r in rows:
            bad = r["visibility"][VIEW_NAME[v]] == "occluded" or r["source"] == "inferred" or r.get("conflict", False)
            by_slot.setdefault(r["player_id"], np.zeros(NF, bool))[r["frame"]] = bad
        for pid, b in by_slot.items():
            f = 0
            while f < NF:
                if b[f]:
                    g = f
                    while g < NF and b[g]:
                        g += 1
                    if g - f >= a.min_run:
                        need[f:g] += 1
                    f = g
                else:
                    f += 1
        lo, hi = 3, NF - 20
        if hi <= lo or need[lo:hi].max() == 0:
            out["views"][v] = {"skipped": True, "reason": "no slot needs help"}
            continue
        f_star = int(lo + np.argmax(need[lo:hi]))
        tdir = os.path.join(P, f"track_{v}{a.suffix}")
        meta = json.load(open(os.path.join(tdir, "frames.json")))
        src0 = meta["source_frame_start"] + f_star
        frames = sorted(os.listdir(os.path.join(tdir, "frames")))
        odir = os.path.join(P, f"track_{v}_p2")
        os.makedirs(os.path.join(odir, "frames"), exist_ok=True)
        for k, fr in enumerate(frames[f_star:]):
            dst = os.path.join(odir, "frames", f"{k:05d}.jpg")
            if not os.path.exists(dst):
                os.symlink(os.path.realpath(os.path.join(tdir, "frames", fr)), dst)
        m2 = dict(meta); m2["source_frame_start"] = src0; m2["num_frames"] = len(frames) - f_star
        m2["second_pass_from_clip_frame"] = f_star
        json.dump(m2, open(os.path.join(odir, "frames.json"), "w"), indent=1)
        dets = json.load(open(os.path.join(P, v, "dets_all.json")))
        d0 = dict(dets); d0["frames"] = [dets["frames"][src0]]
        json.dump(d0, open(os.path.join(odir, "dets0.json"), "w"))
        lock = open(os.path.expanduser("~/sam3_work/logs/gpu.lock"), "w"); fcntl.flock(lock, fcntl.LOCK_EX)
        r = subprocess.run([PY, os.path.join(_REPO, "src/multiview/segmentation/segment.py"), "track", "--out-dir", odir, "--init-dets", os.path.join(odir, "dets0.json"),
                            "--min-score", "0.3", "--max-objects", "30", "--dtype", "fp16"], capture_output=True, text=True)
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
        st = json.load(open(os.path.join(odir, "stats.json"))) if r.returncode == 0 else None
        out["views"][v] = {"frame_in_clip": f_star, "source_frame": src0, "slots_needing_help_at_frame": int(need[f_star]),
                           "init_boxes": len(d0["frames"][0]["detections"]), "ok": r.returncode == 0,
                           "alive_last": st and st["objects_alive_last_frame"], "error": None if r.returncode == 0 else r.stderr[-500:]}
    json.dump(out, open(os.path.join(P, "recon", "second_pass.json"), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
