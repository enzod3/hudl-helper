"""Playing-field mask for filtering person detections (OpenCV + NumPy only; importable from either venv).

field_mask(bgr) -> (mask uint8 0/255, info dict)
  1. green turf (HSV) closed morphologically; largest component; convex hull filled
     (players, logos and dark night turf inside the field are filled back in).
  2. sideline cut: long white line segments (HoughLinesP on white pixels inside the turf); the long lines
     of the view-specific sideline family are sideline candidates ('wide': near-horizontal, 'ez': steep).
     The turf beyond the innermost candidate on each side (outer long lines are track / coaching-box lines;
     a candidate must be >= 45% white along its whole extension, which rejects dashed hash rows)
     is removed if that is only a margin strip,
     so benches standing on green margins are cut.
  3. eroded by `erode_px` so people standing at the turf edge (notches in the green) fall outside.
filter_detections(dets, mask, ...) keeps a person if its foot point (bottom-centre) is inside the mask and
its box is not cut off by the top/bottom image border (near-camera sideline people have their feet out of frame).
"""
import cv2
import numpy as np


def _green(hsv, strict=False):
    # strict: turf colour for the field hull (relaxed thresholds also catch trees behind the end zone);
    # relaxed: includes dark / low-saturation night turf, used for the standing-on-turf check
    return cv2.inRange(hsv, (28, 40, 25) if strict else (28, 25, 15), (100, 255, 255))


def _turf(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = _green(hsv, strict=True)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k, iterations=2)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
    if n <= 1:
        return np.zeros_like(green), hsv
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    comp = np.where(lab == big, 255, 0).astype(np.uint8)
    # convex hull: the visible field (+ margins) is convex in the image; this fills players, logos and night
    # shadows that are not green enough. Margins beyond the sidelines are removed by the sideline cut.
    hull = cv2.convexHull(cv2.findNonZero(comp))
    filled = np.zeros_like(comp)
    cv2.fillConvexPoly(filled, hull, 255)
    return filled, hsv


def _line_angle(seg):
    x1, y1, x2, y2 = seg
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0


def _line_coverage(white, turf, seg, step=4):
    """Fraction of the segment's full-image extension (inside the turf) that is painted white: a solid sideline
    stays high even when people stand on parts of it; a hash row (4-inch dashes every yard) is ~0.1."""
    h, w = white.shape
    x1, y1, x2, y2 = seg
    d = np.array([x2 - x1, y2 - y1], float); d /= np.linalg.norm(d)
    t = np.arange(-2 * max(h, w), 2 * max(h, w), step)
    xs = np.round(x1 + t * d[0]).astype(int); ys = np.round(y1 + t * d[1]).astype(int)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[ok], ys[ok]
    inside = turf[ys, xs] > 0
    if inside.sum() < 20:
        return 0.0
    return float((white[ys, xs] > 0)[inside].mean())


def _sideline_cut(turf, hsv, view, min_len_frac=0.3, max_cut_frac=0.2, min_coverage=0.45):
    """Cut turf beyond the innermost long white line of the sideline family on each side of the field.
    Sideline family by view: 'wide' = within 15 deg of horizontal; 'ez' = 20-160 deg (yard lines are horizontal).
    A cut is applied only if it removes <= max_cut_frac of the turf (a margin strip, not half the field)."""
    h, w = turf.shape
    white = cv2.inRange(hsv, (0, 0, 165), (179, 70, 255))
    white = cv2.bitwise_and(white, cv2.dilate(turf, np.ones((25, 25), np.uint8)))
    segs = cv2.HoughLinesP(white, 1, np.pi / 360, threshold=120, minLineLength=int(min_len_frac * w), maxLineGap=25)
    white_d = cv2.dilate(white, np.ones((5, 5), np.uint8))
    info = {"segments": 0, "sidelines": [], "rejected_cuts": 0}
    if segs is None:
        return turf, info
    segs = segs.reshape(-1, 4).astype(np.float64)
    info["segments"] = int(len(segs))
    ang = np.array([_line_angle(s) for s in segs])
    if view == "wide":
        fam = np.minimum(ang, 180 - ang) < 15
    elif view == "ez":
        fam = (ang > 20) & (ang < 160)
    else:
        raise ValueError(f"view must be wide or ez, got {view}")
    cand = segs[fam]
    cov = np.array([_line_coverage(white_d, turf, sg) for sg in cand]) if len(cand) else np.zeros(0)
    cand = cand[cov >= min_coverage]
    info["candidates"] = int(len(cand))
    if len(cand) == 0:
        return turf, info
    ys, xs = np.nonzero(turf)
    centre = np.array([xs.mean(), ys.mean()])
    turf_px = len(xs)
    best = {}  # innermost candidate on each side of the turf centre (outer ones are track / coaching-box lines)
    for x1, y1, x2, y2 in cand:
        d = np.array([x2 - x1, y2 - y1]); nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        off = float(np.dot(np.array([x1, y1]) - centre, nrm))
        # orient the normal to point away from the centre so both sides compare on |off|
        if off < 0:
            nrm, off = -nrm, -off
        side = "a" if (nrm[1] > 0 or (nrm[1] == 0 and nrm[0] > 0)) else "b"
        if side not in best or off < best[side][0]:
            best[side] = (off, (x1, y1, x2, y2), nrm)
    out = turf.copy()
    yy, xx = np.mgrid[0:h, 0:w]
    for off, (x1, y1, x2, y2), nrm in best.values():
        beyond = ((xx - x1) * nrm[0] + (yy - y1) * nrm[1]) > 6
        cut = np.count_nonzero(beyond & (out > 0))
        if cut > max_cut_frac * turf_px:
            info["rejected_cuts"] += 1
            continue
        out[beyond] = 0
        info["sidelines"].append([round(float(v), 1) for v in (x1, y1, x2, y2)])
    return out, info


def field_mask(bgr, view="wide", erode_px=12, use_sidelines=True):
    """Returns (mask, info); info["green"] is the raw turf-colour mask used by filter_detections' foot check."""
    turf, hsv = _turf(bgr)
    info = {"turf_frac": round(float(np.count_nonzero(turf)) / turf.size, 3)}
    mask = turf
    if use_sidelines and info["turf_frac"] > 0.05:
        mask, li = _sideline_cut(turf, hsv, view)
        info.update(li)
    if erode_px:
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)))
    info["mask_frac"] = round(float(np.count_nonzero(mask)) / mask.size, 3)
    info["green"] = _green(hsv)
    return mask, info


def filter_detections(dets, mask, border_px=4, foot_lift_px=3, green=None, patch_h=8, min_green=0.3,
                      edge_zone_px=60):
    """dets: list of {box_xyxy,...}; returns (kept, dropped) with a 'drop_reason' on dropped ones.
    With `green`, a person is also dropped if the strip just under the box (patch_h px, box width) is less than
    `min_green` turf-coloured: people on the track / bench area beyond the turf. That check only applies within
    `edge_zone_px` of the mask edge, so players standing on painted lines or logos in the field are kept."""
    h, w = mask.shape
    edge_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5) if green is not None else None
    kept, dropped = [], []
    for d in dets:
        x1, y1, x2, y2 = d["box_xyxy"]
        fx = int(min(max((x1 + x2) / 2, 0), w - 1))
        fy = int(min(max(y2 - foot_lift_px, 0), h - 1))
        if y2 >= h - border_px or y1 <= border_px:
            dropped.append({**d, "drop_reason": "cut_by_frame_edge"})
        elif mask[fy, fx] == 0:
            dropped.append({**d, "drop_reason": "foot_outside_field"})
        elif green is not None and edge_dist[fy, fx] < edge_zone_px and _foot_green(green, x1, x2, y2, patch_h) < min_green:
            dropped.append({**d, "drop_reason": "not_standing_on_turf"})
        else:
            kept.append(d)
    return kept, dropped


def _foot_green(green, x1, x2, y2, patch_h):
    h, w = green.shape
    xa, xb = int(max(0, x1)), int(min(w, x2))
    ya, yb = int(min(h - 1, max(0, y2 - 2))), int(min(h, y2 + patch_h))
    patch = green[ya:yb, xa:xb]
    return float(np.count_nonzero(patch)) / patch.size if patch.size else 0.0
