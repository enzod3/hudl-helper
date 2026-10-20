import numpy as np

import calibrate as cal
import fieldmodel as fm


def camera_H():
    """A plausible sideline camera: field yards -> image px."""
    H = np.array([[30.0, -12.0, 200.0], [1.5, -9.0, 900.0], [0.002, -0.012, 1.0]])
    return H / H[2, 2]


def img_line(H, p, q):
    a, b = fm.project(H, [p, q])
    return cal.line_from_pts(a, b)


def test_dlt_lines_recovers_homography():
    H = camera_H()
    il, fl = [], []
    for x in (40.0, 45.0, 50.0, 55.0):
        il.append(img_line(H, (x, 0.0), (x, 53.3))); fl.append(np.array([1.0, 0.0, -x]))
    for y in (19.667, 33.667, 0.0):
        il.append(img_line(H, (20.0, y), (80.0, y))); fl.append(np.array([0.0, 1.0, -y]))
    H2 = cal.dlt_lines(il, fl)
    pts = np.array([[45.0, 10.0], [52.0, 30.0], [60.0, 26.0]])
    assert np.max(np.abs(fm.project(H, pts) - fm.project(H2, pts))) < 0.5   # px


def test_safe_inv_rejects_degenerate():
    assert cal.safe_inv(np.zeros((3, 3))) is None
    assert cal.safe_inv(camera_H()) is not None
