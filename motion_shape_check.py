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

# Phase 3 (2026-10-13): threshold for letting this signal actually promote a
# candidate, picked from the real numbers gathered validating this method,
# not guessed. Every CLEAN (n_reseeds==0) static-group reading measured so
# far: Ich 0.046, Trichodina 0.106, Chilodonella 0.106 - all comfortably
# under 0.2. Every mobile-group reading except one resting-moment outlier:
# Gill Fluke 0.389/2.561, Skin Fluke 1.567/1.836/7.697 - all comfortably
# over 0.35. Picked 0.35 to sit in that gap with margin on both sides.
# Deliberately NOT centered/symmetric - see the one-directional note at this
# constant's only use site for why a LOW reading is never used to confirm a
# static primary (a resting fluke measured as low as 0.014, lower than any
# static reading seen).
SHAPE_PROMOTE_MOBILE_THRESHOLD = 0.35

# Phase 3 (2026-10-13): fixed, deliberately capped confidence shown for a
# shape-promoted result, used IN PLACE OF Roboflow's own raw score for that
# detection. Found testing this today: Roboflow's real score for the
# promoted candidate can be low (it's whatever that single sampled frame
# scored, often low precisely because the organism looked ambiguous there -
# the same reason it needed promoting in the first place), which looked
# wrong and undermined trust next to the demoted candidate's own higher
# score, even once the duplicate was correctly hidden. Rather than compute
# some blended number from Roboflow's score and the shape signal - neither
# one is calibrated as a real probability, so combining them would just be
# false precision - this mirrors the same principle already used for Costia
# (COSTIA_FIXED_CONFIDENCE in motion_costia.py): a flat, honest number in
# the "Medium" tier (ResultView.tsx: >=0.8 High, >=0.5 Medium, else Low).
# Set a little above Costia's 0.55: unlike Costia, Roboflow DID genuinely
# detect this organism on real pixels, just with a low score on that one
# frame - the shape signal adds real corroboration on top of a real
# detection, not instead of one.
SHAPE_PROMOTED_FIXED_CONFIDENCE = 0.6

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


def shape_signal_supports_mobile(stats):
    """True only when the reading is BOTH high enough and trustworthy enough
    to say "this looks like a flexing/mobile organism, not a static one" -
    requires n_reseeds==0 (the one real false-positive case found during
    validation, a messy Chilodonella video reading 4.113, needed 4 reseeds
    and should never have been trusted) as well as clearing
    SHAPE_PROMOTE_MOBILE_THRESHOLD."""
    return (
        stats is not None
        and stats["n_reseeds"] == 0
        and stats["aspect_ratio_range"] >= SHAPE_PROMOTE_MOBILE_THRESHOLD
    )


# Phase 3 extension (2026-10-13): the reverse direction - demoting a
# MOBILE_GROUP primary in favor of a STATIC_GROUP secondary - needs an extra
# safeguard the "promote to mobile" direction didn't. The one known
# resting-fluke reading (0.014) scores LOWER than every genuinely static
# reading seen so far (0.037, 0.046, 0.106, 0.106), so a low reading alone
# can't safely confirm a mobile-group primary is actually wrong - there's no
# threshold that cleanly separates "really static" from "a fluke resting for
# this particular 10s window." Real resting flukes don't change their
# internal anatomy just because they're holding still, though, so requiring
# the existing label-plausibility check to have ALSO independently rejected
# this primary (see this threshold's only use site in main.py) closes that
# gap - a real resting fluke should still pass that check even on a quiet
# video, so requiring both signals to agree filters out the risk a shape
# reading alone can't.
SHAPE_DEMOTE_STATIC_THRESHOLD = 0.15


def shape_signal_supports_static(stats):
    """Mirrors shape_signal_supports_mobile for the opposite direction. NOT
    sufficient on its own to demote a primary - see SHAPE_DEMOTE_STATIC_THRESHOLD's
    docstring for the required second signal (the primary's own
    label-plausibility verdict) that makes using this safe."""
    return (
        stats is not None
        and stats["n_reseeds"] == 0
        and stats["aspect_ratio_range"] <= SHAPE_DEMOTE_STATIC_THRESHOLD
    )
