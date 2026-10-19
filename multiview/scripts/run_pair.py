#!/usr/bin/env python
"""One command per paired play (run with ~/sam3env/bin/python from ~/sam3_work):

  run_pair.py CLIP WIDE_URL EZ_URL [--pre 1.5] [--post 3.5] [--ref-pair 5474607970] [--steps all]

Steps (each skipped when its output exists): download -> probe (ingest.py) -> frames (segment.py extract --fps 0)
-> detect (RF-DETR L fp16, 2x2 tiles, every frame, rfdetr_env, under the GPU lock; then the field mask on the CPU
   with apply_fieldmask.py outside the lock) -> sync (sync.py; snap per view) -> calibrate (both
views, keyframe = detected snap, seeded by SIFT registration to the reference pair's labelled keyframes, re-anchored
every 30 frames) -> stagec (field-plane offset: fills a missing End Zone snap / overrides a disagreeing one)
-> track (SAM 3.1 multiplex with crop prompting, snap-PRE .. snap+POST) -> recon (recon.py)
-> second_pass (second_pass.py: re-prompt SAM where slots are occluded / inferred / conflicting) -> recon again.
GPU steps hold ~/sam3_work/logs/gpu.lock so several plays can run at once.
Writes pairs/CLIP/run_pair.json with step timings and statuses.
"""
import argparse, fcntl, json, os, subprocess, sys, time
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROOT = os.environ.get("MULTIVIEW_WORK", os.path.expanduser("~/sam3_work"))
PY = os.environ.get("MULTIVIEW_SAM_PY", os.path.expanduser("~/sam3env/bin/python"))
PYR = os.environ.get("MULTIVIEW_RFDETR_PY", os.path.expanduser("~/rfdetr_env/bin/python"))
REF_FRAMES = {"wide": 125, "ez": 111}   # labelled keyframes of the reference pair


def sh(cmd, log, cwd=ROOT):
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n"); fh.flush()
        r = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[1] if len(cmd) > 1 else cmd[0]} failed (exit {r.returncode}); see {log}")


class GPULock:
    def __enter__(self):
        self.fh = open(os.path.join(ROOT, "logs", "gpu.lock"), "w")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        fcntl.flock(self.fh, fcntl.LOCK_UN); self.fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("wide_url")
    ap.add_argument("ez_url", help="End Zone URL, or '-' for single-view (Wide only) mode")
    ap.add_argument("--pre", type=float, default=1.5)
    ap.add_argument("--post", type=float, default=3.5)
    ap.add_argument("--ref-pair", default="5474607970")
    ap.add_argument("--recon-args", default="--assoc-gate 2.5")
    ap.add_argument("--wide-snap-fallback", type=float, default=None,
                    help="Wide snap time (s) to use when the snap detector finds none (e.g. the Hudl tag)")
    ap.add_argument("--pairs-root", default="pairs", help="folder under ~/sam3_work holding the play folders")
    ap.add_argument("--det-weights", default=None, help="fine-tuned RF-DETR checkpoint for detect.py --weights")
    ap.add_argument("--second-pass", action="store_true", help="also run the second SAM pass (about +40% GPU time per play)")
    a = ap.parse_args()
    P = os.path.join(a.pairs_root, a.clip)
    os.makedirs(os.path.join(ROOT, P), exist_ok=True)
    log = os.path.join(ROOT, P, "run_pair.log")
    status = {"clip": a.clip, "steps": {}}
    single = a.ez_url in ("-", "none", "")
    views = ["wide"] if single else ["wide", "ez"]

    def step(name, fn):
        t = time.time()
        try:
            fn()
            status["steps"][name] = {"ok": True, "s": round(time.time() - t, 1)}
        except Exception as e:
            status["steps"][name] = {"ok": False, "s": round(time.time() - t, 1), "error": str(e)}
            json.dump(status, open(os.path.join(ROOT, P, "run_pair.json"), "w"), indent=1)
            raise
        json.dump(status, open(os.path.join(ROOT, P, "run_pair.json"), "w"), indent=1)

    def download():
        for v, url in (("wide", a.wide_url), ("ez", a.ez_url))[:len(views)]:
            f = os.path.join(ROOT, P, f"{v}.mp4")
            if not os.path.exists(f) or os.path.getsize(f) == 0:
                sh(["curl", "-sfL", "-o", f, url], log)

    def probe():
        for v in views:
            if not os.path.exists(os.path.join(ROOT, P, f"{v}.mp4.probe.json")):
                sh([PY, os.path.join(_REPO, "src/multiview/ingestion/ingest.py"), f"{P}/{v}.mp4"], log)

    def frames():
        for v in views:
            if not os.path.exists(os.path.join(ROOT, P, v, "frames_pts.json")):
                sh([PY, os.path.join(_REPO, "src/multiview/segmentation/segment.py"), "extract", "--video", f"{P}/{v}.mp4", "--fps", "0", "--out-dir", f"{P}/{v}"], log)
            lst = os.path.join(ROOT, P, v, "frames.txt")
            fr = sorted(os.listdir(os.path.join(ROOT, P, v, "frames")))
            open(lst, "w").write("".join(os.path.join(P, v, "frames", f) + "\n" for f in fr))

    def detect():
        # GPU part (RF-DETR) holds the lock; the field mask runs afterwards on the CPU, outside the lock
        for v in views:
            if os.path.exists(os.path.join(ROOT, P, v, "dets_all.json")):
                continue
            raw = os.path.join(ROOT, P, v, "dets_raw.json")
            if not os.path.exists(raw):
                with GPULock():
                    sh([PYR, os.path.join(_REPO, "src/multiview/detection/detect.py"), "--model", "large", "--tiles", "2", "--view", v,
                        "--warmup", "1", "--list", f"{P}/{v}/frames.txt", "--out", f"{P}/{v}/dets_raw.json"]
                       + (["--weights", a.det_weights] if a.det_weights else []), log)
            sh([PY, os.path.join(_REPO, "src/multiview/detection/apply_fieldmask.py"), f"{P}/{v}/dets_raw.json", f"{P}/{v}/dets_all.json", "--view", v], log)

    def sync():
        if not os.path.exists(os.path.join(ROOT, P, "sync.json")):
            sh([PY, os.path.join(_REPO, "src/multiview/synchronization/sync.py"), P, "--sheets"] + (["--views", "wide"] if single else ["--preview"]), log)
        s = json.load(open(os.path.join(ROOT, P, "sync.json")))
        if s["wide"]["snap_frame"] is None:
            if a.wide_snap_fallback is None:
                raise RuntimeError("snap not detected in the Wide view")
            # fall back to the play's tagged snap time (Hudl metadata) for the Wide view
            pts = json.load(open(os.path.join(ROOT, P, "wide", "frames_pts.json")))["pts_s"]
            sf = min(range(len(pts)), key=lambda i: abs(pts[i] - a.wide_snap_fallback))
            s["wide"].update({"snap_frame": sf, "snap_pts": pts[sf], "method": "tag_fallback"})
            json.dump(s, open(os.path.join(ROOT, P, "sync.json"), "w"), indent=1)
        # End Zone snap missing (e.g. the clip starts after the snap): Stage C fills it after calibration

    def calibrate():
        s = json.load(open(os.path.join(ROOT, P, "sync.json")))
        procs = []
        for v in views:
            if os.path.exists(os.path.join(ROOT, P, v, "calib.json")):
                continue
            lab = os.path.join(ROOT, "calib", "labels", f"{a.clip}_{v}.json")
            nfr = len(os.listdir(os.path.join(ROOT, P, v, "frames")))
            kf = s[v]["snap_frame"] if s[v]["snap_frame"] is not None else nfr // 3
            kf = int(min(max(kf, 0), nfr - 1))
            refs = [{"view_dir": f"pairs/{a.ref_pair}/{v}", "frame": REF_FRAMES[v]}]
            # more references: other plays' calibrated keyframes of the same camera with good fits, same game first
            cands = []
            for c2 in sorted(os.listdir(os.path.join(ROOT, "pairs"))):
                if c2 in (a.clip, a.ref_pair):
                    continue
                f2 = os.path.join(ROOT, "pairs", c2, v, "calib.json")
                if not os.path.exists(f2):
                    continue
                try:
                    with open(f2) as fh:
                        head = json.load(fh)["summary"]
                except Exception:
                    continue
                if (head.get("inlier_frac_median") or 0) >= 0.6 and head.get("residual_px_median") is not None:
                    cands.append((c2[:5] != a.clip[:5], -head["inlier_frac_median"], c2, head["keyframe"]))
            refs += [{"view_dir": f"pairs/{c2}/{v}", "frame": kfr} for _, _, c2, kfr in sorted(cands)[:8]]
            json.dump({"frame": kf, "refs": refs}, open(lab, "w"))
            fh = open(os.path.join(ROOT, P, f"calib_{v}.log"), "w")
            procs.append(subprocess.Popen([PY, os.path.join(_REPO, "src/multiview/calibration/calibrate.py"), f"{P}/{v}", lab, "--view", v, "--overlay",
                                           "--anchor-every", "30"], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT))
        for p in procs:
            if p.wait() != 0:
                raise RuntimeError("calibrate failed; see calib_<view>.log")

    def stagec():
        if single:
            return
        # field-plane sync: fills a missing End Zone snap, and overrides the snap-based offset when the two disagree
        # by > 0.5 s and the Stage C minimum is clear (margin >= 0.2 yd, confidence >= 0.2)
        sh([PY, os.path.join(_REPO, "src/multiview/synchronization/stagec.py"), P, "--apply"], log)
        sc = json.load(open(os.path.join(ROOT, P, "stagec.json")))
        s = json.load(open(os.path.join(ROOT, P, "sync.json")))
        if s["ez"]["snap_pts"] is None:
            raise RuntimeError("no End Zone snap and Stage C could not sync")
        s.setdefault("sync_method", "snap")
        if (s["sync_method"] == "snap" and sc.get("ok") and abs(sc.get("stage_c_minus_snap_s", 0)) > 0.5
                and sc.get("margin_yd", 0) >= 0.2 and (sc.get("confidence") or 0) >= 0.2):
            pts = json.load(open(os.path.join(ROOT, P, "ez", "frames_pts.json")))["pts_s"]
            ez_snap = s["wide"]["snap_pts"] - sc["offset_s"]
            s["ez"]["snap_pts_snap_detector"] = s["ez"]["snap_pts"]
            s["ez"]["snap_pts"] = round(ez_snap, 4)
            s["ez"]["snap_frame"] = int(round((ez_snap - pts[0]) / (pts[1] - pts[0])))
            s["offset_snap_s"] = sc["offset_s"]
            s["sync_method"] = "stage_c_override"
        s["stage_c"] = {k: sc.get(k) for k in ("ok", "offset_s", "confidence", "margin_yd", "stage_c_minus_snap_s")}
        json.dump(s, open(os.path.join(ROOT, P, "sync.json"), "w"), indent=1)

    def track():
        s = json.load(open(os.path.join(ROOT, P, "sync.json")))
        for v in views:
            out = os.path.join(ROOT, P, f"track_{v}_crop")
            if os.path.exists(os.path.join(out, "stats.json")):
                continue
            pts = json.load(open(os.path.join(ROOT, P, v, "frames_pts.json")))["pts_s"]
            dets = json.load(open(os.path.join(ROOT, P, v, "dets_all.json")))
            snap = s[v]["snap_frame"]
            fps = round(1 / (pts[1] - pts[0]))
            st, en = max(0, snap - int(round(a.pre * fps))), min(len(pts), snap + int(round(a.post * fps)) + 1)
            os.makedirs(os.path.join(out, "frames"), exist_ok=True)
            fr = sorted(os.listdir(os.path.join(ROOT, P, v, "frames")))
            for k, i in enumerate(range(st, en)):
                dst = os.path.join(out, "frames", f"{k:05d}.jpg")
                if not os.path.exists(dst):
                    os.symlink(os.path.join(ROOT, P, v, "frames", fr[i]), dst)
            json.dump({"video": os.path.join(ROOT, P, f"{v}.mp4"), "start": pts[st], "end": pts[en - 1], "fps": fps,
                       "width": 1920, "height": 1080, "num_frames": en - st, "source_frame_start": st,
                       "snap_frame_in_clip": snap - st}, open(os.path.join(out, "frames.json"), "w"), indent=1)
            d0 = dict(dets); d0["frames"] = [dets["frames"][st]]
            json.dump(d0, open(os.path.join(out, "dets0.json"), "w"))
            with GPULock():
                sh([PY, os.path.join(_REPO, "src/multiview/segmentation/segment.py"), "track", "--out-dir", f"{P}/track_{v}_crop", "--init-dets",
                    f"{P}/track_{v}_crop/dets0.json", "--min-score", "0.3", "--max-objects", "30", "--dtype", "fp16",
                    "--preview"], log)

    def recon():
        sh([PY, os.path.join(_REPO, "src/multiview/reconstruction/recon.py"), P] + a.recon_args.split() + (["--views", "wide"] if single else []), log)

    def second_pass():
        import shutil
        shutil.copy(os.path.join(ROOT, P, "recon", "recon_summary.json"), os.path.join(ROOT, P, "recon", "recon_summary_pass1.json"))
        sh([PY, os.path.join(_REPO, "src/multiview/segmentation/second_pass.py"), P], log)
        sh([PY, os.path.join(_REPO, "src/multiview/reconstruction/recon.py"), P, "--extra-suffix", "_p2"] + a.recon_args.split(), log)

    for name, fn in [("download", download), ("probe", probe), ("frames", frames), ("detect", detect),
                     ("sync", sync), ("calibrate", calibrate), ("stagec", stagec), ("track", track), ("recon", recon)] + ([("second_pass", second_pass)] if a.second_pass else []):
        step(name, fn)
    print(json.dumps(status))


if __name__ == "__main__":
    main()
