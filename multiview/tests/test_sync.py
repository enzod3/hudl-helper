import numpy as np

import sync


def curve(t_snap, fps=30.0, n=300, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    frac = np.where(t < t_snap, 0.02, 0.45) + rng.normal(0, 0.01, n)
    return t, np.clip(frac, 0, 1)


def test_detect_snap_finds_onset():
    t, f = curve(4.0)
    s = sync.detect_snap(t, f)
    assert s["snap_pts"] is not None and abs(s["snap_pts"] - 4.0) < 0.2


def test_detect_snap_prefers_sustained_motion_over_shift():
    t, f = curve(6.0)
    f[60:75] = 0.4            # a short pre-snap shift at 2.0-2.5 s
    s = sync.detect_snap(t, f)
    assert abs(s["snap_pts"] - 6.0) < 0.2


def test_no_snap_in_flat_curve():
    t = np.arange(200) / 30.0
    assert sync.detect_snap(t, np.full(200, 0.02))["snap_pts"] is None
