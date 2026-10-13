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


# Texture gate (2026-10-13) - see the full investigation in check_shape_group's
# docstring. Real numbers measured directly: known-static bubbles (artifacts)
# 3.8-5.8, a confirmed-real Gill Fluke in a SOFTER/lower-res video 256-345,
# plain background texture in a crisp video 173, a confirmed-real
# Chilodonella in that same crisp video 1202. Absolute sharpness varies a lot
# between videos (the Gill Fluke reading is lower than plain background in a
# different, crisper video) - so this is set low, just high enough to
# reliably exclude smooth artifacts (which read single digits, a >30x gap
# below every real-content reading seen so far) rather than trying to also
# distinguish organism from background, which isn't what this gate is for.
# Not yet validated across many examples - grounded in an actual measured
# gap, not a guess, but still a first cut. Two earlier fix attempts (camera-
# motion compensation, then periodic template-match drift-detection) were
# tried and ruled out first - see commit history / check_shape_group's
# docstring for why each didn't work.
MIN_TEXTURE_VARIANCE = 50


def check_shape_group(video_path: str, fps: float, target_size: int, detection: dict):
    """Tracks the given detection's organism via sparse optical flow for
    ~10s and returns the raw aspect-ratio AND translational-movement stats,
    or None if tracking couldn't get started (e.g. the seed frame can't be
    read). detection is one of main.py's own detection dicts - x/y/width/
    height are expected in the 1024x1024-resized-frame space Roboflow
    actually saw, same as everywhere else in main.py, and are converted
    back to the video's real pixel resolution here (mirrors the scaling
    motion_costia.py already does the same way, just inverted).

    Added 2026-10-13 (same day, after Adam pointed out a real miss the
    shape-only signal couldn't have caught): path_length_bl/speed_bl_per_sec
    measure how far the tracked points' CENTROID travels across the screen,
    reusing the exact same tracked points already being followed for the
    shape reading - genuinely different information from aspect-ratio range.
    An organism that stretches and compresses in place (a fluke) can score
    high on shape variability while barely translating; an organism that
    glides across the slide (Chilodonella, per Adam) or spins in place
    (Trichodina) moves very differently again - and a static air bubble or
    debris shouldn't translate at all. Normalized to body-lengths (using the
    detection's own box size) so it's comparable across different zoom
    levels, the same convention the original (pre-shape-signal) tracker
    prototype used for its own speed_bl_per_sec stat.

    RELIABILITY CAVEAT, found the same day testing against real footage: a
    visually-confirmed STATIC air bubble read path_length_bl of 5-19 across
    three separate real tests (two different crops of the same source video,
    with and without unrelated screen-recording UI in frame) - clearly wrong
    if taken at face value. Camera-motion compensation was tried and ruled
    out as the fix (see the loop below) - whatever's really inflating this
    number on real footage isn't understood yet. Because of this, main.py
    currently only LOGS path_length_bl/speed_bl_per_sec for visibility and
    does NOT let them promote or demote a result - only the (separately and
    more thoroughly validated) aspect-ratio/shape stats are trusted to act."""
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
    body_size_px = half_w + half_h  # average of real-world box width/height

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Texture gate (2026-10-13), checked BEFORE tracking even starts. Found
    # the real cause of path_length_bl's bad readings by direct investigation:
    # a smooth object (a bubble) has almost no internal texture for optical
    # flow to anchor to, only its thin ring edge - and tracking a point sitting
    # exactly on an edge is a classic weak spot (the aperture problem), so
    # points slowly slide along the edge over many frames, with each reseed
    # re-centering on wherever the track CURRENTLY (already wrongly) believes
    # the object is, compounding the drift rather than correcting it.
    # Confirmed directly: the tracked centroid ended up 200px from the real,
    # still-unmoved bubble after a 10s track.
    #
    # Two things were tried and ruled out before this: camera-motion
    # compensation (didn't fix the bubble's reading - real cause wasn't
    # camera panning) and periodic template-matching drift-detection (the
    # match score stayed high, 0.7-0.99, even 200px off-target, because this
    # kind of blurry mucus/tissue background is repetitive enough that
    # "generic blur" matches "more generic blur" almost as well as the real
    # original spot - not discriminative enough on this content).
    #
    # What actually separates a trustworthy object from one: Laplacian
    # variance (a standard texture-richness measure) of the seed crop,
    # BEFORE any tracking. Measured directly on real footage: a known-static
    # bubble read 5.8, a confirmed-real organism in a clean video read 1202,
    # and even plain BACKGROUND texture in that same clean video read 173 -
    # a two-orders-of-magnitude gap between "real, trackable organism" and
    # "artifact/too smooth to trust", with a clear margin above ordinary
    # background texture too. MIN_TEXTURE_VARIANCE sits in that gap. If the
    # seed region fails this, there's no point spending a ~10s tracking pass
    # on it at all - return immediately with insufficient_texture=True.
    seed_y1, seed_y2 = max(0, int(cy - half_h)), int(cy + half_h)
    seed_x1, seed_x2 = max(0, int(cx - half_w)), int(cx + half_w)
    seed_crop = prev_gray[seed_y1:seed_y2, seed_x1:seed_x2]
    texture_variance = float(cv2.Laplacian(seed_crop, cv2.CV_64F).var()) if seed_crop.size else 0.0
    if texture_variance < MIN_TEXTURE_VARIANCE:
        cap.release()
        return {"num_points": 0, "aspect_ratio_mean": 1.0, "aspect_ratio_range": 0.0, "aspect_ratio_std": 0.0,
                "n_reseeds": 0, "path_length_bl": 0.0, "speed_bl_per_sec": 0.0,
                "texture_variance": round(texture_variance, 1), "insufficient_texture": True}

    points = _seed_points(prev_gray, cx, cy, half_w, half_h)

    max_frames = int(fps * TRACK_SECONDS) if fps else 0
    spreads = []
    centroids = []
    n_reseeds = 0
    frames_processed = 0

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames_processed += 1
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

        if len(points) >= 1:
            centroids.append((float(points[:, 0].mean()), float(points[:, 1].mean())))

        spread = _pca_spread(points)
        if spread:
            spreads.append(spread)

        prev_gray = gray

    cap.release()

    path_length_px = sum(
        ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        for (x1, y1), (x2, y2) in zip(centroids, centroids[1:])
    )
    path_length_bl = round(path_length_px / body_size_px, 3) if body_size_px > 1e-6 else 0.0
    tracked_seconds = frames_processed / fps if fps else 0
    speed_bl_per_sec = round(path_length_bl / tracked_seconds, 3) if tracked_seconds > 1e-6 else 0.0

    if not spreads:
        return {"num_points": 0, "aspect_ratio_mean": 1.0, "aspect_ratio_range": 0.0, "aspect_ratio_std": 0.0,
                "n_reseeds": n_reseeds, "path_length_bl": path_length_bl, "speed_bl_per_sec": speed_bl_per_sec,
                "texture_variance": round(texture_variance, 1), "insufficient_texture": False}

    ratios = [major / minor if minor > 1e-6 else 1.0 for major, minor in spreads]
    return {
        "num_points": len(spreads),
        "aspect_ratio_mean": round(float(np.mean(ratios)), 3),
        "aspect_ratio_range": round(float(max(ratios) - min(ratios)), 3),
        "aspect_ratio_std": round(float(np.std(ratios)), 3),
        "n_reseeds": n_reseeds,
        "path_length_bl": path_length_bl,
        "speed_bl_per_sec": speed_bl_per_sec,
        "texture_variance": round(texture_variance, 1),
        "insufficient_texture": False,
    }


def shape_signal_supports_mobile(stats):
    """True only when the reading is BOTH high enough and trustworthy enough
    to say "this looks like a flexing/mobile organism, not a static one" -
    requires n_reseeds==0 (the one real false-positive case found during
    validation, a messy Chilodonella video reading 4.113, needed 4 reseeds
    and should never have been trusted), the seed region having enough real
    texture to track reliably at all (added after proving a smooth artifact
    like an air bubble can read a high, misleadingly "mobile-looking"
    aspect-ratio value too - see MIN_TEXTURE_VARIANCE's docstring), as well
    as clearing SHAPE_PROMOTE_MOBILE_THRESHOLD."""
    return (
        stats is not None
        and stats["n_reseeds"] == 0
        and not stats.get("insufficient_texture", False)
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
        and not stats.get("insufficient_texture", False)
        and stats["aspect_ratio_range"] <= SHAPE_DEMOTE_STATIC_THRESHOLD
    )


# Movement-corroboration thresholds (2026-10-13), for path_length_bl -
# UNLIKE the aspect-ratio thresholds above, these are NOT yet backed by a
# set of real measured examples the way those were (9 videos before any
# threshold was trusted). This is a first, reasoned starting point: a
# genuinely gliding/crawling organism should cover at least a full body
# length or so over a 10s window, while a static air bubble or debris
# shouldn't meaningfully travel at all. Log every raw path_length_bl value
# and revisit these two numbers once real production examples come in -
# same calibration discipline used to derive SHAPE_PROMOTE_MOBILE_THRESHOLD,
# just not done yet for this particular metric.
MOVEMENT_STATIC_THRESHOLD_BL = 0.3
MOVEMENT_REAL_THRESHOLD_BL = 1.0


def shows_no_real_movement(stats):
    """True when the tracked centroid barely moved and the track is
    trustworthy (n_reseeds==0, and the seed region had enough real texture
    to track reliably in the first place - see MIN_TEXTURE_VARIANCE's
    docstring for why that check matters: a smooth artifact like an air
    bubble can "track" just fine by point-count alone while actually
    sliding off its own edge) - consistent with a static artifact (an air
    bubble, debris) rather than a living, moving organism."""
    return (
        stats is not None
        and stats["n_reseeds"] == 0
        and not stats.get("insufficient_texture", False)
        and stats["path_length_bl"] < MOVEMENT_STATIC_THRESHOLD_BL
    )


def shows_clear_real_movement(stats):
    """True when the tracked centroid travelled a real distance and the
    track is trustworthy (n_reseeds==0, seed region had enough real texture)
    - consistent with a genuinely moving, living organism (gliding, spinning
    in place with some drift, etc.) rather than a static artifact."""
    return (
        stats is not None
        and stats["n_reseeds"] == 0
        and not stats.get("insufficient_texture", False)
        and stats["path_length_bl"] >= MOVEMENT_REAL_THRESHOLD_BL
    )
