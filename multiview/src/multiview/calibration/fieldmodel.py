"""Canonical football field model + paint-mask utilities (OpenCV/NumPy; importable from either venv).

Canonical frame (yards): x = 0..120 along the length (x=10 and x=110 are the goal lines, x=60 the 50-yd line),
y = 0..53.33 across, y=0 is the sideline nearest the Wide (sideline) camera. Hash marks: inner edges at
HASH_Y from the near sideline (NCAA 20 / 33.33; NFL 23.58 / 29.75), configurable via set_hashes().
"""
import cv2
import numpy as np

FIELD_LEN, FIELD_W = 120.0, 160.0 / 3.0
HASH_Y = [20.0, FIELD_W - 20.0]          # NCAA default
HASH_LEN = 2.0 / 3.0                     # 24-inch marks
HASH_DIR = "outward"                     # marks extend from HASH_Y toward the nearer sideline ("outward") or the centre
TICK_INSET, TICK_LEN = 4.0 / 36.0, 2.0 / 3.0   # sideline yard ticks: 4 in inside the sideline, 24 in long


def set_hash_dir(d):
    global HASH_DIR
    assert d in ("outward", "inward", "centered")
    HASH_DIR = d


def set_hashes(level="ncaa"):
    global HASH_Y
    HASH_Y = {"ncaa": [20.0, FIELD_W - 20.0], "nfl": [70.75 / 3.0, FIELD_W - 70.75 / 3.0],
              "hs": [FIELD_W / 3.0, 2 * FIELD_W / 3.0]}[level]


def model_segments():
    """List of ((x1,y1),(x2,y2), kind) paint segments in yards."""
    segs = []
    for x in np.arange(10, 111, 5):
        segs.append(((x, 0.0), (x, FIELD_W), "yard"))
    segs += [((0, 0), (0, FIELD_W), "endline"), ((FIELD_LEN, 0), (FIELD_LEN, FIELD_W), "endline"),
             ((0, 0), (FIELD_LEN, 0), "sideline"), ((0, FIELD_W), (FIELD_LEN, FIELD_W), "sideline")]
    for x in np.arange(11, 110, 1):
        if x % 5 == 0:
            continue
        # hash marks: near hash extends from its inner edge toward the near sideline, far hash toward the far one
        if HASH_DIR == "outward":
            n0, n1, f0, f1 = HASH_Y[0] - HASH_LEN, HASH_Y[0], HASH_Y[1], HASH_Y[1] + HASH_LEN
        elif HASH_DIR == "inward":
            n0, n1, f0, f1 = HASH_Y[0], HASH_Y[0] + HASH_LEN, HASH_Y[1] - HASH_LEN, HASH_Y[1]
        else:
            n0, n1, f0, f1 = HASH_Y[0] - HASH_LEN / 2, HASH_Y[0] + HASH_LEN / 2, HASH_Y[1] - HASH_LEN / 2, HASH_Y[1] + HASH_LEN / 2
        segs.append(((x, n0), (x, n1), "hash"))
        segs.append(((x, f0), (x, f1), "hash"))
        segs.append(((x, TICK_INSET), (x, TICK_INSET + TICK_LEN), "tick"))
        segs.append(((x, FIELD_W - TICK_INSET - TICK_LEN), (x, FIELD_W - TICK_INSET), "tick"))
    return segs


def model_points(step=0.25):
    """Dense samples along every paint segment: (N,2) yards and kind labels."""
    pts, kinds = [], []
    for (x1, y1), (x2, y2), k in model_segments():
        n = max(2, int(np.ceil(np.hypot(x2 - x1, y2 - y1) / step)) + 1)
        t = np.linspace(0, 1, n)
        pts.append(np.stack([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], 1))
        kinds += [k] * n
    return np.concatenate(pts), np.array(kinds)


def paint_mask(bgr, field=None, boxes=(), tophat=15, pad=8):
    """Thin bright paint lines: white top-hat of the V channel, low saturation; minus players; inside `field`."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat, tophat))
    th = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, k)
    m = ((th > 35) & (hsv[:, :, 1] < 90)).astype(np.uint8) * 255
    if field is not None:
        m &= field
    for b in boxes:
        x1, y1, x2, y2 = [int(round(c)) for c in b]
        cv2.rectangle(m, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), 0, -1)
    return m


def project(H, pts):
    p = np.asarray(pts, float).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def draw_grid(img, H, color_yard=(0, 255, 255), color_grid=(0, 160, 255), one_yard=True, label=True):
    """Overlay the field model (image = H @ field)."""
    out = img.copy()
    h, w = img.shape[:2]

    def seg(p, q, col, th):
        a = project(H, np.linspace(p, q, 40))
        # drop samples behind the camera / wildly outside
        ok = np.all(np.isfinite(a), 1) & (np.abs(a[:, 0]) < 4 * w) & (np.abs(a[:, 1]) < 4 * h)
        a = a[ok].astype(np.int32)
        if len(a) >= 2:
            cv2.polylines(out, [a], False, col, th, cv2.LINE_AA)

    if one_yard:
        for x in np.arange(0, FIELD_LEN + 0.1, 1):
            if x % 5:
                seg((x, 0), (x, FIELD_W), color_grid, 1)
        for y in np.arange(0, FIELD_W, 1):
            seg((0, y), (FIELD_LEN, y), color_grid, 1)
    for (p, q, k) in model_segments():
        col = {"yard": color_yard, "endline": (0, 0, 255), "sideline": (0, 0, 255), "hash": (255, 0, 255),
               "tick": (255, 0, 255)}[k]
        seg(p, q, col, 2 if k in ("yard", "sideline", "endline") else 2)
    if label:
        for x in np.arange(10, 111, 10):
            for y in (12.0, FIELD_W - 12.0):
                c = project(H, [(x, y)])[0]
                if np.all(np.isfinite(c)) and 0 <= c[0] < w and 0 <= c[1] < h:
                    yd = int(x - 10 if x <= 60 else 110 - x)
                    cv2.putText(out, str(yd), (int(c[0]) - 14, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out
