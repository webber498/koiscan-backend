"""
Shape-corroboration check for the 5 Roboflow-supported species (Trichodina,
Chilodonella, White Spot, Gill Flukes, Skin Fluke) - Costia is out of scope,
it has its own separate motion-only path in motion_costia.py.

Built from two real misclassifications in one day (2026-10-13): a Trichodina
video briefly misread as Chilodonella (the two "vibrate in place" vs. "glide"
at measurably different speeds, but nothing in the pipeline looks at
movement), and a Gill Fluke misread as Chilodonella because the fluke's body
happened to be compressed (round, Chilodonella-like) in the frames Roboflow
sampled - the fluke's real tell, that it visibly stretches and compresses
over time, can't be seen in any single still frame.

Ported from the validated prototype at
C:\\Users\\webbe\\Downloads\\koiscan_motion_detection\\flow_shape.py after it
was chosen over two other approaches that were tried and failed on real
footage first: a tracker's own box-size output (TrackerVit's box smoothly
collapses/drifts in a way its own confidence score doesn't predict and a
jump filter can't catch, since the collapse is gradual not sudden), and
independent per-frame re-segmentation (no temporal continuity between
frames, which made the shape signal noisier than the original problem, not
better). Sparse optical-flow point tracking has real temporal continuity
(each point is explicitly carried from the previous frame, not re-detected
from scratch) while still capturing genuine shape change: if the organism
stretches, tracked points on its head/tail visibly move apart; if it
compresses, they converge. Validated across 9 real videos spanning all 5
species - a clean two-group split: static/round ciliates (Trichodina,
Chilodonella, White Spot) show a tight aspect-ratio range, mobile/flexing
flukes (Gill Flukes, Skin Fluke) show a much wider one.

Phase 2 (this module, 2026-10-13): computed and LOGGED ONLY per the approved
plan - not allowed to touch what gets shown to the user yet. Only 9 data
points exist so far, nowhere near enough to trust a hard threshold (this
project has been burned by exactly that trap before) - group membership
should be judged by eye against logged production numbers for a while
before Phase 3/4 ever lets this influence a real result.

One real lesson already learned validating this: n_reseeds == 0 (points
tracked continuously, no re-detection needed) is necessary but NOT
sufficient to trust a reading - two "reliable" Skin Fluke readings 500x
apart initially looked like a bug, and turned out to both be genuine
biology (confirmed by watching the actual frames). A surprising number is
a reason to go watch the video, never to blindly trust or discard it.
"""
import cv2
import numpy as np

# Confirmed against main.py's own class-name strings (its NO_DETECTION_COMMENTARY_PROMPT
# and comments) - "White Spot" not "Ich", "Gill Flukes" plural not "Gill Fluke".
STATIC_GROUP = {"White Spot", "Trichodina", "Chilodonella"}
MOBILE_GROUP = {"Gill Flukes", "Skin Fluke"}

TRACK_SECONDS = 10
MAX_CORNERS = 30
MIN_POINTS_TO_RESEED = 8
LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def _pca_spread(points):
    """Returns (major_axis_length, minor_axis_length) of a 2D point cloud."""
    if len(points) < 3:
        return None
    pts = np.array(points, dtype=np.float64)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)  # ascending
    minor, major = np.sqrt(max(eigvals[0], 0)), np.sqrt(max(eigvals[1], 0))
    return major, minor


def _seed_points(gray, cx, cy, half_w, half_h, max_corners=MAX_CORNERS):
    mask = np.zeros_like(gray)
    mx1, my1 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
    mx2, my2 = min(gray.shape[1], int(cx + half_w)), min(gray.shape[0], int(cy + half_h))
    mask[my1:my2, mx1:mx2] = 255
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01,
                                   minDistance=5, mask=mask)
    return pts.reshape(-1, 2) if pts is not None else np.zeros((0, 2))


def check_shape_group(video_path: str, fps: float, target_size: int, detection: dict):
    """Tracks the given detection's organism via sparse optical flow for
    ~10s and returns the raw aspect-ratio stats, or None if tracking
    couldn't get started (e.g. the seed frame can't be read). detection is
    one of main.py's own detection dicts - x/y/width/height are expected in
    the 1024x1024-resized-frame space Roboflow actually saw, same as
    everywhere else in main.py, and are converted back to the video's real
    pixel resolution here (mirrors the scaling motion_costia.py already
    does the same way, just inverted)."""
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not orig_w or not orig_h:
        cap.release()
        return None
    scale_x, scale_y = orig_w / target_size, orig_h / target_size

    seed_frame_number = int(round(detection["timestamp"] * fps)) if fps else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, seed_frame_number)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None

    cx, cy = detection["x"] * scale_x, detection["y"] * scale_y
    half_w, half_h = (detection["width"] * scale_x) / 2, (detection["height"] * scale_y) / 2

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = _seed_points(prev_gray, cx, cy, half_w, half_h)

    max_frames = int(fps * TRACK_SECONDS) if fps else 0
    spreads = []
    n_reseeds = 0

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(points) >= 2:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points.astype(np.float32), None, **LK_PARAMS)
            points = new_points[status.reshape(-1) == 1]
        else:
            points = np.zeros((0, 2))

        if len(points) < MIN_POINTS_TO_RESEED:
            if len(points) >= 1:
                cx, cy = points[:, 0].mean(), points[:, 1].mean()
            fresh = _seed_points(gray, cx, cy, half_w, half_h)
            if len(fresh) > len(points):
                points = fresh
                n_reseeds += 1

        spread = _pca_spread(points)
        if spread:
            spreads.append(spread)

        prev_gray = gray

    cap.release()

    if not spreads:
        return {"num_points": 0, "aspect_ratio_mean": 1.0, "aspect_ratio_range": 0.0, "aspect_ratio_std": 0.0, "n_reseeds": n_reseeds}

    ratios = [major / minor if minor > 1e-6 else 1.0 for major, minor in spreads]
    return {
        "num_points": len(spreads),
        "aspect_ratio_mean": round(float(np.mean(ratios)), 3),
        "aspect_ratio_range": round(float(max(ratios) - min(ratios)), 3),
        "aspect_ratio_std": round(float(np.std(ratios)), 3),
        "n_reseeds": n_reseeds,
    }
