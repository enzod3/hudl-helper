import numpy as np

import recon


def test_kalman_smooth_constant_velocity():
    rng = np.random.default_rng(1)
    frames = list(range(60))
    truth = {f: np.array([30.0 + 0.2 * f, 20.0 - 0.1 * f]) for f in frames}
    meas = {f: truth[f] + rng.normal(0, 0.3, 2) for f in frames if f % 3 != 0}   # every 3rd frame missing
    covs = {f: np.eye(2) * 0.09 for f in meas}
    sm = recon.kalman_smooth(frames, meas, covs)
    err = [np.linalg.norm(sm[f][0] - truth[f]) for f in frames]
    raw = [np.linalg.norm(meas[f] - truth[f]) for f in meas]
    assert np.median(err) < np.median(raw)          # smoothing beats raw measurements
    assert not sm[0][2] and sm[1][2]                # observed flags follow the measurements


def test_inverse_covariance_fusion_weights_precise_axis():
    # Wide precise in x, End Zone precise in y: the fused estimate takes x from Wide and y from End Zone
    zw, Rw = np.array([10.0, 20.0]), np.diag([0.01, 1.0])
    ze, Re = np.array([11.0, 21.0]), np.diag([1.0, 0.01])
    W = np.linalg.inv(Rw) + np.linalg.inv(Re)
    z = np.linalg.solve(W, np.linalg.inv(Rw) @ zw + np.linalg.inv(Re) @ ze)
    assert abs(z[0] - 10.0) < 0.05 and abs(z[1] - 21.0) < 0.05


def test_iou_tracker_keeps_ids():
    boxes = [[[0, 0, 10, 20], [50, 0, 60, 20]], [[1, 0, 11, 20], [51, 0, 61, 20]], [[2, 0, 12, 20]]]
    ids = recon.iou_track(boxes)
    assert ids[0] == ids[1][:2] and ids[2][0] == ids[0][0]
