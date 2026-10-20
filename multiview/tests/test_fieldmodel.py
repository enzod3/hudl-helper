import numpy as np

import fieldmodel as fm


def test_ncaa_hashes_default():
    fm.set_hashes("ncaa")
    assert fm.HASH_Y == [20.0, fm.FIELD_W - 20.0]


def test_nfl_and_hs_hashes():
    fm.set_hashes("nfl")
    assert abs(fm.HASH_Y[0] - 70.75 / 3.0) < 1e-9
    fm.set_hashes("hs")
    assert abs(fm.HASH_Y[1] - 2 * fm.FIELD_W / 3.0) < 1e-9
    fm.set_hashes("ncaa")


def test_model_segments_counts():
    segs = fm.model_segments()
    kinds = [k for _, _, k in segs]
    assert kinds.count("yard") == 21           # x = 10, 15, ..., 110
    assert kinds.count("sideline") == 2 and kinds.count("endline") == 2
    assert kinds.count("hash") == 2 * 80       # every non-5 yard between the goal lines, both rows
    xs = sorted({p[0] for p, q, k in segs if k == "yard"})
    assert xs[0] == 10 and xs[-1] == 110


def test_hash_direction_outward():
    fm.set_hash_dir("outward")
    near = [(p, q) for p, q, k in fm.model_segments() if k == "hash" and p[1] < fm.FIELD_W / 2]
    (x1, y1), (x2, y2) = near[0]
    assert max(y1, y2) == fm.HASH_Y[0] and min(y1, y2) < fm.HASH_Y[0]


def test_project_identity():
    pts = np.array([[10.0, 0.0], [60.0, 26.67]])
    assert np.allclose(fm.project(np.eye(3), pts), pts)
