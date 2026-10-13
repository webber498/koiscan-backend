"""
Costia motion-detection fallback — the first of the "three-source fused
decisioning" signals to reach production (Roboflow model detection, motion
detection, Claude label-plausibility check; see project memory 2026-08-22
"Confirmed roadmap direction"). Ported from the validated rebuild at
C:\\Users\\webbe\\Downloads\\koiscan_motion_detection\\ (not a repo, kept
out of prod until now) after a real multi-video sweep and one serious bug
(screen-recorded non-microscopy content getting mislabelled) was found and
fixed there — see that folder's module docstrings for the full history.

Costia is structurally invisible to a single still frame — it's identified
by its distinctive erratic darting movement, not static appearance, so
Roboflow has near-zero recall on it no matter how well trained. This module
gives the app a real path to detecting Costia for the first time: find
where in the video something is moving with Costia's diffuse, thin-spread
activity signature (find_local_peak_candidates), then track it
(run_hybrid, a reseed-aware tracker built specifically because Costia is
too small/fast for plain TrackerMIL to reliably follow).

Deliberately conservative given how often even a visually-confirmed track
still needed a human's judgement call during validation (see memory): the
confidence value returned is a fixed, capped constant, never a computed
number presented as if it were precise - same "no raw-percentage-implying-
precision-the-model-doesn't-have" principle already used for every other
detection shown in the app.
"""
import base64
import cv2
import numpy as np

MAX_STEP_SHIFT = 30
MIN_CORR_RESPONSE = 0.01
VIGNETTE_ERODE_FRAC = 0.25  # widened 2026-09-10: 0.10 was confirmed too narrow, still let the vignette/fold boundary dominate every top-8 slot on real footage
MAX_COMPONENT_DIM_FRAC = 0.35
UI_MIN_LINE_COUNT = 4
ANALYSIS_WIDTH = 800
# Blob-size suppression (see the comment at its use site in
# find_local_peak_candidates for the full reasoning and the real numbers
# this was calibrated against): threshold the persistence map at this
# fraction of a candidate's own peak, then reject if the connected blob at
# that threshold is bigger than MAX_BLOB_SIZE_PX (analysis-resolution
# pixels, ANALYSIS_WIDTH=800 wide). Calibrated 2026-09-10 against one
# known-good video (blobs 38-66px) and several hand-confirmed-bad ones
# (two clear artifacts at 274-313px, others just as tight as the good
# case - this filter catches only the "big blob" class of false positive).
BLOB_SIZE_PEAK_FRAC = 0.7
MAX_BLOB_SIZE_PX = 150

BOX_SIZE = 70
TRACK_SECONDS = 10
CANDIDATE_WINDOW_SECONDS = 12
# Raised 2026-09-10 from the old default of 8: on a video with confirmed
# real, visible Costia (multiple, per a human watching the full clip), NONE
# of the top 8 candidates by persistence contained it - direct frame-by-frame
# review of all 8 showed static debris/tracker jitter, no genuine movement.
# The real organism (confirmed via a visible tracker relocation event - the
# box jumped to a new position and the same distinctly-shaped object was
# still inside it) turned up at rank 29 of 40. This is a dense, busy-texture
# video where diffuse Costia activity gets heavily outnumbered by ordinary
# debris/tissue motion at similar persistence values - the ranking isn't
# wrong so much as genuine signal sits much deeper in it than 8 covers on
# footage like this. Deliberately not tuned to exactly 29 - kept generous
# headroom above the one confirmed case found so far.
CANDIDATE_TOP_K = 40

# Fixed, deliberately capped confidence for any Costia motion-detection
# result - never computed from persistence/speed, which aren't calibrated
# as probabilities and shouldn't be presented as if they were. Sits in the
# "Medium" tier (ResultView.tsx: >=0.8 High, >=0.5 Medium, else Low) -
# multiple real detections were confirmed correct during validation, but
# the underlying signal is still genuinely harder to be certain about than
# a direct Roboflow detection, so it's capped well below High.
COSTIA_FIXED_CONFIDENCE = 0.55


def _resize_for_analysis(frame):
    h, w = frame.shape[:2]
    scale = ANALYSIS_WIDTH / w
    small = cv2.resize(frame, (ANALYSIS_WIDTH, int(h * scale)))
    return small, scale


def _stabilize_and_accumulate(frames_gray):
    stabilized = [frames_gray[0]]
    cum_dx, cum_dy = 0.0, 0.0
    max_drift = 0.0
    prev = frames_gray[0].astype(np.float32)
    for gray in frames_gray[1:]:
        cur = gray.astype(np.float32)
        (dx, dy), response = cv2.phaseCorrelate(prev, cur)
        if response >= MIN_CORR_RESPONSE and (dx ** 2 + dy ** 2) ** 0.5 <= MAX_STEP_SHIFT:
            cum_dx += dx
            cum_dy += dy
        max_drift = max(max_drift, (cum_dx ** 2 + cum_dy ** 2) ** 0.5)
        M = np.float32([[1, 0, cum_dx], [0, 1, cum_dy]])
        aligned = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        stabilized.append(aligned)
        prev = cur
    return stabilized, max_drift


def _bright_field_mask(sample_gray):
    _, mask = cv2.threshold(sample_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.ones_like(sample_gray, dtype=np.uint8) * 255
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255


def _is_ui_chrome(patch_gray):
    edges = cv2.Canny(patch_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=25, maxLineGap=4)
    count = 0 if lines is None else len(lines)
    return count >= UI_MIN_LINE_COUNT


def _find_local_peak_candidates(video_path, window_seconds=CANDIDATE_WINDOW_SECONDS, top_k=8, min_area=3, neighborhood=15):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(fps * window_seconds)

    frames_gray = []
    scale = 1.0
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        small, scale = _resize_for_analysis(frame)
        frames_gray.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(frames_gray) < 10:
        return []

    stabilized, max_drift = _stabilize_and_accumulate(frames_gray)
    border = int(max_drift) + 5

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=len(stabilized), varThreshold=16, detectShadows=False)
    kernel = np.ones((3, 3), np.uint8)
    persistence = np.zeros(stabilized[0].shape, dtype=np.float32)
    for gray in stabilized:
        fg = bg_sub.apply(gray)
        closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        persistence += (closed > 0).astype(np.float32)
    persistence = cv2.GaussianBlur(persistence, (9, 9), 0)

    if border > 0:
        persistence[:border, :] = 0
        persistence[-border:, :] = 0
        persistence[:, :border] = 0
        persistence[:, -border:] = 0

    bright = _bright_field_mask(stabilized[len(stabilized) // 2])
    bright_area = float(np.count_nonzero(bright))
    erode_px = max(1, int(VIGNETTE_ERODE_FRAC * (bright_area ** 0.5)))
    erode_kernel = np.ones((erode_px, erode_px), np.uint8)
    bright_eroded = cv2.erode(bright, erode_kernel, iterations=1)
    persistence[bright_eroded == 0] = 0

    dilated = cv2.dilate(persistence, np.ones((neighborhood, neighborhood), np.uint8))
    peak_mask = (persistence == dilated) & (persistence >= min_area)
    ys, xs = np.where(peak_mask)
    scores = persistence[ys, xs]

    h, w = persistence.shape
    order = np.argsort(-scores)
    candidates = []
    seen = []
    for idx in order:
        if len(candidates) >= top_k:
            break
        x, y, score = int(xs[idx]), int(ys[idx]), float(scores[idx])
        if any(((x - sx) ** 2 + (y - sy) ** 2) ** 0.5 < neighborhood for sx, sy in seen):
            continue
        if neighborhood > MAX_COMPONENT_DIM_FRAC * min(h, w):
            continue

        # Blob-size suppression: threshold the persistence map at a fraction
        # of THIS candidate's own peak value and measure the connected
        # component it sits in. Confirmed directly (2026-09-10) against
        # real hand-labeled examples: a genuine tiny organism produces a
        # small, tight component (known-good case: 42px at 70%-of-peak);
        # a boundary-shimmer artifact or a much larger competing organism
        # produces a large, spread-out one (274-313px on two confirmed-bad
        # cases). This does NOT catch every false positive on its own
        # (several confirmed-bad cases were just as tight as the genuine
        # one - a small tight blob is necessary but not sufficient
        # evidence) but removes a real, meaningful class of them.
        blob_thresh = score * BLOB_SIZE_PEAK_FRAC
        blob_mask = (persistence >= blob_thresh).astype(np.uint8)
        _, blob_labels = cv2.connectedComponents(blob_mask)
        blob_size = int(np.count_nonzero(blob_labels == blob_labels[y, x]))
        if blob_size > MAX_BLOB_SIZE_PX:
            continue

        pad = 40
        px1, py1 = max(0, x - pad), max(0, y - pad)
        px2, py2 = min(w, x + pad), min(h, y + pad)
        patch = stabilized[len(stabilized) // 2][py1:py2, px1:px2]
        if patch.size > 0 and _is_ui_chrome(patch):
            continue
        seen.append((x, y))
        candidates.append({"center_x_orig": x / scale, "center_y_orig": y / scale, "persistence": score})

    return candidates


def _find_best_seed_frame(video_path, center_x_orig, center_y_orig, window_seconds=CANDIDATE_WINDOW_SECONDS,
                           box_size=BOX_SIZE, warmup_frames=15):
    """Scans the window for the frame with peak local motion overlap at the
    candidate's location. The first `warmup_frames` are fed to the
    background subtractor to build its model but are NOT eligible to win -
    MOG2's first few `.apply()` calls flag large swathes of the frame as
    "foreground" with no history yet to compare against, and a video with a
    fade-in from black (a real, observed case) compounds this by feeding it
    genuinely large frame-to-frame brightness change for its first several
    frames too. Without this, frame 0 of a fading-in intro can spuriously
    "win" and get returned as the seed frame - confirmed directly: this
    exact failure put the tracking box on a still-solid-black frame."""
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(fps * window_seconds)
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=max_frames, varThreshold=16, detectShadows=False)
    best_frame_num, best_overlap = warmup_frames, -1
    half = box_size // 2
    x1, y1 = int(center_x_orig - half), int(center_y_orig - half)
    x2, y2 = int(center_x_orig + half), int(center_y_orig + half)
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg = bg_sub.apply(gray)
        if i < warmup_frames:
            continue
        h, w = fg.shape
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        overlap = int(np.count_nonzero(fg[cy1:cy2, cx1:cx2]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_frame_num = i
    cap.release()
    return best_frame_num


def _run_hybrid_tracker(video_path, seed_frame_number, seed_box, duration_seconds=TRACK_SECONDS,
                         recheck_interval=6, min_blob_area=6, min_confirm_distance=8):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, seed_frame_number)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None

    # Clamp to valid frame bounds before handing to TrackerMIL - a candidate
    # near the frame edge can otherwise produce a negative-origin or
    # out-of-frame box, which crashes cv2 with a low-level "bad allocation"
    # error rather than a catchable Python exception (confirmed directly).
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = seed_box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        cap.release()
        return None
    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    tracker = cv2.TrackerMIL_create()
    tracker.init(frame, bbox)
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=16, detectShadows=False)
    bg_sub.apply(frame)

    centers, box_sizes = [], []
    max_frames = int(fps * duration_seconds)
    pending_reseed = None
    seed_frame_img = frame.copy()

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        ok, bbox = tracker.update(frame)
        if not ok:
            break
        bx, by, bw, bh = bbox
        fg_mask = bg_sub.apply(frame)

        if i % recheck_interval == 0 and i > 0:
            search_pad = 1.5
            sx1 = max(0, int(bx - bw * search_pad))
            sy1 = max(0, int(by - bh * search_pad))
            sx2 = min(frame.shape[1], int(bx + bw * (1 + search_pad)))
            sy2 = min(frame.shape[0], int(by + bh * (1 + search_pad)))
            local_mask = fg_mask[sy1:sy2, sx1:sx2]
            contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best = None
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_blob_area:
                    continue
                mx, my, mw, mh = cv2.boundingRect(c)
                blob_cx, blob_cy = sx1 + mx + mw / 2, sy1 + my + mh / 2
                cur_cx, cur_cy = bx + bw / 2, by + bh / 2
                dist = ((blob_cx - cur_cx) ** 2 + (blob_cy - cur_cy) ** 2) ** 0.5
                if dist < min_confirm_distance:
                    continue
                if best is None or area > best[2]:
                    best = (blob_cx, blob_cy, area)
            if best is not None:
                bcx, bcy, _ = best
                if pending_reseed is not None:
                    px, py = pending_reseed
                    if ((bcx - px) ** 2 + (bcy - py) ** 2) ** 0.5 < min_confirm_distance:
                        # Clamp to keep the WHOLE box in-frame, not just its
                        # top-left corner - a reseed near the right/bottom
                        # edge could otherwise still push x2/y2 past the
                        # frame, crashing TrackerMIL.init() with "bad
                        # allocation" the same way the seed box could before
                        # being clamped above (confirmed directly on a real
                        # video's reseed event, 2026-09-12).
                        new_x1 = max(0, min(bcx - bw / 2, frame_w - bw))
                        new_y1 = max(0, min(bcy - bh / 2, frame_h - bh))
                        tracker = cv2.TrackerMIL_create()
                        new_bbox = (int(new_x1), int(new_y1), int(bw), int(bh))
                        tracker.init(frame, new_bbox)
                        bbox = new_bbox
                        bx, by, bw, bh = bbox
                        pending_reseed = None
                    else:
                        pending_reseed = (bcx, bcy)
                else:
                    pending_reseed = (bcx, bcy)
            else:
                pending_reseed = None

        cx, cy = bx + bw / 2, by + bh / 2
        centers.append((cx, cy))
        box_sizes.append((bw, bh))

    cap.release()
    if len(centers) < fps * 2:  # require at least ~2s of genuine tracked motion before trusting this candidate
        return None
    return {"seed_frame_img": seed_frame_img, "num_points": len(centers)}


def find_costia_detection(video_path: str, fps: float, target_size: int, plausibility_checker=None):
    """Tries each motion candidate in persistence order until one produces a
    genuine sustained track AND (if plausibility_checker is given) passes
    that check. Returns a detection dict shaped exactly like a Roboflow
    prediction (so it can drop straight into best_per_class / detections
    downstream), or None if nothing usable was found - the expected,
    common outcome for any video that isn't actually Costia.

    plausibility_checker, if given: a callable(detection_dict) -> bool,
    called on each successfully-tracked candidate before accepting it -
    see the caller in main.py, which wires up the existing
    check_label_plausibility Claude check for this."""
    candidates = _find_local_peak_candidates(video_path, top_k=CANDIDATE_TOP_K)

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not orig_w or not orig_h:
        return None

    sx, sy = target_size / orig_w, target_size / orig_h

    for cand in candidates:
        seed_frame_num = _find_best_seed_frame(video_path, cand["center_x_orig"], cand["center_y_orig"])
        # Deliberately NOT gated by a frozen-runway liveness check here (unlike
        # the Roboflow-seeded path) - that check operates on whole-frame pixel
        # difference, but Costia's whole signature is that individual
        # organisms are too tiny to move a frame-wide average (that's the
        # entire reason find_local_peak_candidates uses local maxima instead
        # of a frame-wide signal). Adding it here rejected every real
        # candidate on the exact video previously confirmed correct.

        x, y = cand["center_x_orig"], cand["center_y_orig"]
        box = (x - BOX_SIZE / 2, y - BOX_SIZE / 2, x + BOX_SIZE / 2, y + BOX_SIZE / 2)

        # Check plausibility on the (cheap) seed frame BEFORE the expensive
        # ~10s tracking pass - the candidate-finder always returns its best-
        # ranked motion within THIS video even when nothing Costia-like is
        # really there (confirmed directly: it tracked on-screen caption
        # text on an unrelated video), so most candidates on most videos
        # will fail this and there's no reason to pay the tracking cost
        # first only to throw it away.
        if plausibility_checker is not None:
            seed_frame_img = _read_frame(video_path, seed_frame_num)
            if seed_frame_img is None:
                continue
            precheck_detection = _build_detection(seed_frame_img, x, y, sx, sy, seed_frame_num, fps, target_size)
            if not plausibility_checker(precheck_detection):
                continue

        track = _run_hybrid_tracker(video_path, seed_frame_num, box)
        if track is None:
            continue

        return _build_detection(track["seed_frame_img"], x, y, sx, sy, seed_frame_num, fps, target_size)

    return None


def _read_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _build_detection(frame_img, x, y, sx, sy, seed_frame_num, fps, target_size=None):
    if target_size is not None:
        frame_img = cv2.resize(frame_img, (target_size, target_size))
    _, buffer = cv2.imencode('.jpg', frame_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    return {
        "class": "Costia",
        "confidence": COSTIA_FIXED_CONFIDENCE,
        "timestamp": seed_frame_num / fps if fps else 0,
        "frame_base64": frame_b64,
        "x": x * sx,
        "y": y * sy,
        "width": BOX_SIZE * sx,
        "height": BOX_SIZE * sy,
    }
