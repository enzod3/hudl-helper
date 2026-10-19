#!/usr/bin/env python
"""Probe videos with PyAV (run with ~/sam3env/bin/python): ingest.py VIDEO [VIDEO ...]

Writes VIDEO.probe.json next to each file:
  {path, sha256, width, height, codec, time_base, avg_fps, r_fps, nominal_frames, decoded_frames,
   duration_s, start_pts_s, pts_s: [per decoded frame, presentation order], frame_dt: {min, median, max},
   vfr: bool, audio: [{codec, sample_rate, channels, duration_s}]}
pts are taken from the decoded frames (not assumed from fps), so variable-frame-rate files keep true timing.
"""
import hashlib, json, sys

import av
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path):
    with av.open(path) as c:
        v = c.streams.video[0]
        tb = v.time_base
        pts = [float(fr.pts * tb) for fr in c.decode(v)]
        audio = [{"codec": a.codec_context.name, "sample_rate": a.rate, "channels": a.channels,
                  "duration_s": float(a.duration * a.time_base) if a.duration else None} for a in c.streams.audio]
        info = {"path": path, "width": v.codec_context.width, "height": v.codec_context.height,
                "codec": v.codec_context.name, "time_base": str(tb),
                "avg_fps": float(v.average_rate) if v.average_rate else None,
                "r_fps": float(v.base_rate) if v.base_rate else None,
                "nominal_frames": v.frames, "container_duration_s": c.duration / 1e6 if c.duration else None}
    pts = sorted(pts)
    dt = np.diff(pts) if len(pts) > 1 else np.array([0.0])
    info.update({"sha256": sha256(path), "decoded_frames": len(pts), "start_pts_s": pts[0],
                 "duration_s": pts[-1] - pts[0] + float(np.median(dt)),
                 "frame_dt": {"min": float(dt.min()), "median": float(np.median(dt)), "max": float(dt.max())},
                 "vfr": bool(dt.max() - dt.min() > 1e-3), "audio": audio, "pts_s": [round(p, 6) for p in pts]})
    return info


if __name__ == "__main__":
    for p in sys.argv[1:]:
        info = probe(p)
        json.dump(info, open(p + ".probe.json", "w"))
        print(json.dumps({k: v for k, v in info.items() if k != "pts_s"}))
