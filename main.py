from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import requests
import base64
import tempfile
import os
import json
import re
import anthropic

from motion_costia import find_costia_detection
from motion_shape_check import (
    check_shape_group, shape_signal_supports_mobile, shape_signal_supports_static,
    STATIC_GROUP, MOBILE_GROUP, SHAPE_PROMOTED_FIXED_CONFIDENCE,
)

# Per Adam's own domain knowledge (2026-10-13): within STATIC_GROUP, Trichodina
# spins/vibrates largely in place while Chilodonella glides a real distance
# across the slide - genuinely different movement characters once an organism
# is already confirmed to be alive and moving (see check_movement_plausibility's
# "movement_pattern" field). Nothing established yet for White Spot, so it's
# deliberately left out rather than guessed.
EXPECTED_MOVEMENT_PATTERN = {
    "Trichodina": "spinning_in_place",
    "Chilodonella": "translating_across_frame",
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL_ID = "adams-workspace-vwxcv/koi-parasites-3-gwtia-11-rfdetr-large-t2"
FRAME_INTERVAL = 1
TARGET_SIZE = 1024
# 2026-10-12: lowered from 39 to 30 per Roboflow's own sweep after the v11
# retrain - their data showed 30 is this model's actual best-F1 point,
# trading a bit of precision (55.1% -> 49.8%) for meaningfully more recall
# (40.4% -> 44.4%). Was left at the old value pending this decision -
# Adam approved the move once the two detection-accuracy fixes above landed.
CONFIDENCE_THRESHOLD = 30
# Roboflow calls are the dominant cost of a scan (one HTTP round-trip per
# sampled frame, ~1/sec of video) - running a bounded batch concurrently
# instead of one-at-a-time cuts wall-clock wait time roughly in proportion
# to this number, without changing which frames get analysed or how.
MAX_CONCURRENT_ROBOFLOW_CALLS = 6

QUALITY_PROMPT = """You are checking the quality of a microscope video frame from a koi fish mucus scrape. A hobbyist has filmed this through a microscope eyepiece, usually with a phone.

A genuine microscope wet-mount frame usually shows: a circular or vignetted field of view from looking through the eyepiece, pale or translucent mucus/tissue with visible texture, debris or organisms at high magnification, and fairly flat, even lighting from the microscope's own light source. This is NOT the same as a photo of open water, a pond, or a fish above the surface — do not assume blue/green or murky colouring alone means it isn't a microscope view; mucus samples often look watery or indistinct even when correctly captured.

Do NOT try to identify any parasites. Only assess whether this image is usable.

Answer these three questions:
1. Is this actually a microscope view of a wet mount sample? (Not a photo of a fish, a pond, a screen showing something else, or an unrelated image.)
2. Is it in focus enough that small organisms would be distinguishable?
3. Is this a direct capture, or is it a photograph of a computer/phone screen? Look for moire patterns, scan lines, screen glare, visible cursors or UI elements.

Respond with ONLY a JSON object, no other text and no markdown fences:
{"is_sample": true/false, "in_focus": true/false, "is_screen_photo": true/false, "note": "one short sentence for the user, or empty string if all is well"}

The note should be plain English addressed to the user, and should never say the video is unusable or ask them to refilm — the analysis has already run and a result is being shown. It is advisory only. For example: "This looks like a recording of a screen rather than a direct capture, which can reduce accuracy." Keep it under 20 words. Leave it as an empty string if quality is fine."""


SAME_ORGANISM_INSTRUCTIONS = """These two images are cropped from the same koi mucus-scrape microscope video, showing two separate detections an object-detection model made at different moments in the video.

The object-detection model is known to sometimes mislabel the SAME organism as two different parasite classes when it's seen again later — pose, angle, lighting, focus, and how much of the body is in frame can all look quite different between two sightings of one real organism, especially if there's a time gap between them. Weigh persistent structural cues (overall body shape and proportions, relative size, distinctive features like a fin/sucker/segment pattern) much more heavily than incidental differences in pose, lighting, or exact framing — those alone are NOT evidence of a different organism.

Do these two crops plausibly show the SAME physical organism, seen twice — or do they look like genuinely different organisms? If you're genuinely unsure after weighing the above, prefer "same_organism": true — image 2 is hidden from the user whenever you answer true, so a false "same" only costs a rare genuine second finding going unshown, while a false "different" would wrongly tell the user two separate parasites are present and undermine trust in the whole result.

Respond with ONLY a JSON object, no other text and no markdown fences:
{"same_organism": true/false, "reasoning": "one short sentence explaining your judgement"}"""


LABEL_PLAUSIBILITY_INSTRUCTIONS = """These are cropped image(s) from a koi mucus-scrape microscope video, all of the same candidate organism at different moments in the video if more than one is shown. An object-detection model has labelled what's in them as a specific koi parasite (named in the message above).

Look closely at the actual visible detail inside the organism, not just its outline shape. Does the internal structure genuinely look consistent with that parasite, or does it look more like something else — for example a free-living ciliate/protozoan (like a Paramecium), debris, an air bubble, or another artefact? An elongated or fluke-like outline alone is not enough to confirm a fluke-type parasite; internal detail matters more than silhouette.

Air bubbles are a common false positive in wet-mount microscopy and can visually resemble round parasites like Trichodina at a glance. Look for the tell-tale signs of a bubble specifically: a smooth, sharply-defined circular or oval edge, often with a bright specular highlight or a uniform halo/ring, and little or no genuine internal granular texture or visible organelles. If multiple images are shown and the object looks essentially static and unchanged between them - no real internal detail, no structural variation - treat that absence of any organism-like detail as evidence against it being a living parasite, not as a neutral or inconclusive result.

If multiple images are shown, also weigh how the organism's shape changes between them - visible stretching and compressing over time is consistent with a real fluke-type parasite's natural movement, not evidence against it, even if any single frame alone looks rounder or more ambiguous.

Respond with ONLY a JSON object, no other text and no markdown fences:
{"label_plausible": true/false, "reasoning": "one short sentence explaining your judgement, focused on what you actually see"}"""


COSTIA_CANDIDATE_INSTRUCTIONS = """This is a cropped region from a koi mucus-scrape microscope video. A motion-tracking algorithm flagged real movement matching Costia's signature SPECIFICALLY inside the small red-outlined box drawn on this image — not anywhere else in the crop, which is shown only for surrounding context (a comparison point for "is the boxed area actually different from the texture around it"). Costia is tiny (~7-10 microns) and identified almost entirely by erratic darting movement, not distinctive static appearance — at this magnification it typically looks like an indistinct pale blob, not a parasite with visible internal organs.

Look ONLY at the content INSIDE the red box. Ignore anything elsewhere in the image, even if it looks like a real organism — that is not what triggered this candidate, and judging it instead is a real failure mode seen before (wider context accidentally pulled in an unrelated organism visible elsewhere in frame).

Because Costia cannot be confirmed from a still image, do not try to identify the species. Just judge: is there a real, single bounded shape inside the red box, distinguishable from the surrounding granular texture — or is the area inside the red box just background texture, debris, or (another real failure mode seen before) text/caption content?

Respond with ONLY a JSON object, no other text and no markdown fences:
{"plausible_organism": true/false, "reasoning": "one short sentence, referencing specifically what is inside the red box"}"""


def crop_detection(img_base64: str, det: dict, padding_factor: float = 0.6):
    """Crop the region around a detection (x/y are the box centre) with some
    padding for context, so a comparison model has a focused, uncluttered view."""
    img_bytes = base64.b64decode(img_base64)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    cx, cy = det["x"], det["y"]
    pad_w = det["width"] * (1 + padding_factor)
    pad_h = det["height"] * (1 + padding_factor)
    x1 = max(0, int(cx - pad_w / 2))
    y1 = max(0, int(cy - pad_h / 2))
    x2 = min(w, int(cx + pad_w / 2))
    y2 = min(h, int(cy + pad_h / 2))

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


def crop_and_mark_detection(img_base64: str, det: dict, padding_factor: float = 4.0):
    """Like crop_detection, but with much wider padding AND the exact
    detection box drawn on the crop in red — built for the Costia motion-
    candidate plausibility check, where a plain tight crop turned out to be
    too small/low-context to reliably judge (confirmed directly: Claude
    couldn't tell "generic background texture" from "a real tiny organism"
    in an unmarked ~70px crop), but a wider unmarked crop caused a
    different failure — it sometimes judged an unrelated organism visible
    elsewhere in the wider context instead of the actual candidate spot.
    Marking the exact box lets Claude use the wider context as a comparison
    point while still judging the right specific location."""
    img_bytes = base64.b64decode(img_base64)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    cx, cy = det["x"], det["y"]
    pad_w = det["width"] * (1 + padding_factor)
    pad_h = det["height"] * (1 + padding_factor)
    x1 = max(0, int(cx - pad_w / 2))
    y1 = max(0, int(cy - pad_h / 2))
    x2 = min(w, int(cx + pad_w / 2))
    y2 = min(h, int(cy + pad_h / 2))

    crop = img[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

    bw, bh = det["width"], det["height"]
    bx1, by1 = int(cx - bw / 2) - x1, int(cy - bh / 2) - y1
    bx2, by2 = int(cx + bw / 2) - x1, int(cy + bh / 2) - y1
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


def select_diverse_frames(frames, max_frames=3):
    """Picks up to max_frames entries spread across the CHRONOLOGICAL order
    of 'frames' (sorted by timestamp), rather than just the highest-
    confidence ones - several high-confidence frames of the same candidate
    often cluster around the same pose (e.g. an organism sitting still for
    a couple of seconds), which wouldn't actually give the label-
    plausibility check any new information. Spreading across time instead
    gives it a genuinely different look at the candidate each time - built
    for a real case (2026-10-12) where a gill fluke's visible stretching/
    compressing was the actual tell, but a single frame could look
    ambiguously round."""
    frames_sorted = sorted(frames, key=lambda f: f["timestamp"])
    if len(frames_sorted) <= max_frames:
        return frames_sorted
    if max_frames <= 1:
        return frames_sorted[:1]
    idxs = sorted({round(i * (len(frames_sorted) - 1) / (max_frames - 1)) for i in range(max_frames)})
    return [frames_sorted[i] for i in idxs]


def check_same_organism(primary_crop_b64: str, primary_class: str, primary_confidence: float,
                         other_crop_b64: str, other_class: str, other_confidence: float,
                         primary_timestamp: float = None, other_timestamp: float = None):
    """PROTOTYPE: ask Claude whether a secondary detection is plausibly the
    same organism as the primary one, misread as a different class. Flagged
    detections are hidden entirely on the frontend (a mislabelled repeat
    undermines trust in the whole result more than a rare hidden second
    finding costs) — this never affects the primary detection itself."""
    if not ANTHROPIC_API_KEY:
        return None

    gap_note = ""
    if primary_timestamp is not None and other_timestamp is not None:
        gap = abs(other_timestamp - primary_timestamp)
        gap_note = f' They were detected {gap:.1f} seconds apart in the video.'

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        # Note: unlike the Haiku calls elsewhere in this file, claude-sonnet-5
        # rejects an assistant-turn prefill ("This model does not support
        # assistant message prefill") — the conversation must end on a user
        # message, so the JSON has to be parsed out of a normal free-text
        # reply instead of forced to start with "{".
        message = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f'Image 1: detected as "{primary_class}" at {round(primary_confidence * 100)}% confidence.'},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": primary_crop_b64}},
                        {"type": "text", "text": f'Image 2: detected as "{other_class}" at {round(other_confidence * 100)}% confidence.{gap_note}'},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": other_crop_b64}},
                        {"type": "text", "text": SAME_ORGANISM_INSTRUCTIONS},
                    ],
                },
            ],
        )

        text_block = next((block.text for block in message.content if block.type == "text"), "")
        print(f"Same-organism check raw response: {text_block[:300]}")
        match = re.search(r'\{.*\}', text_block, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in response: {text_block[:200]}")
        result = json.loads(match.group(0))

        return {
            "same_organism": bool(result.get("same_organism", False)),
            "reasoning": str(result.get("reasoning", ""))[:200],
        }

    except Exception as e:
        print(f"Same-organism check failed: {str(e)}")
        return None


def check_label_plausibility(crop_b64_list, claimed_class: str, confidence: float, timestamps=None):
    """PROTOTYPE, upgraded 2026-10-12 to accept MULTIPLE crops of the same
    candidate (up to 3, picked by select_diverse_frames to spread across
    the video's timeline, not just the highest-confidence frames which
    tend to cluster around the same pose). Built after a real case: a gill
    fluke that visibly stretches and compresses was misjudged against a
    single compressed-looking frame, and the same video gave a different
    verdict on a re-run because a different near-tied-confidence frame won
    the "representative" slot - showing the model several moments at once
    (closer to what a human watching the whole video sees) is a more
    direct fix than hoping one lucky frame gets picked. Sanity-checks the
    claimed class against what's actually visible (internal detail, not
    just outline shape) — e.g. a Paramecium being misread as Gill Flukes
    off silhouette alone. Only ever flags the detection for the frontend
    to caveat, never removes or downgrades it on its own (the caller
    decides what to do with the result)."""
    if not ANTHROPIC_API_KEY:
        return None
    if not crop_b64_list:
        return None

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        n = len(crop_b64_list)
        intro = (
            f'This candidate was labelled "{claimed_class}" at {round(confidence * 100)}% confidence '
            f'by the parasite-detection model. Below {"is" if n == 1 else "are"} {n} crop{"s" if n != 1 else ""} '
            f'of the same candidate{" from different moments in the video" if n > 1 else ""}.'
        )
        content = [{"type": "text", "text": intro}]
        for i, crop_b64 in enumerate(crop_b64_list):
            label = f"Image {i + 1}"
            if timestamps and i < len(timestamps):
                label += f" (at {timestamps[i]:.1f}s)"
            content.append({"type": "text", "text": f"{label}:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": crop_b64}})
        content.append({"type": "text", "text": LABEL_PLAUSIBILITY_INSTRUCTIONS})

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[
                {"role": "user", "content": content},
                {"role": "assistant", "content": "{"},
            ],
        )

        raw = "{" + message.content[0].text
        print(f"Label plausibility raw response: {raw}")
        result = json.loads(raw)

        return {
            "label_plausible": bool(result.get("label_plausible", True)),
            "reasoning": str(result.get("reasoning", ""))[:400],
        }

    except Exception as e:
        print(f"Label plausibility check failed: {str(e)}")
        return None


def check_costia_candidate_plausible(crop_b64: str):
    """Gate for the Costia motion-detection path — deliberately NOT the same
    check as check_label_plausibility above. That check asks "does the
    visible internal detail match this specific claimed species" which is
    the wrong question for Costia: it has no distinctive static appearance
    to confirm, that's the whole reason it needs movement tracking instead
    of a still-frame classifier. Tested directly: reusing
    check_label_plausibility here rejected 100% of candidates, including
    the exact video previously confirmed as genuine Costia by a human. This
    check only rules out the clearly-not-an-organism cases (on-screen text,
    a ruler, empty background, an artifact) that the motion candidate-finder
    has no way to distinguish from real movement on its own — confirmed
    directly to fire on on-screen caption text on an unrelated video."""
    if not ANTHROPIC_API_KEY:
        return None

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": crop_b64}},
                        {"type": "text", "text": COSTIA_CANDIDATE_INSTRUCTIONS},
                    ],
                },
                {"role": "assistant", "content": "{"},
            ],
        )

        raw = "{" + message.content[0].text
        print(f"Costia candidate plausibility raw response: {raw}")
        result = json.loads(raw)

        return {
            "plausible_organism": bool(result.get("plausible_organism", True)),
            "reasoning": str(result.get("reasoning", ""))[:400],
        }

    except Exception as e:
        print(f"Costia candidate plausibility check failed: {str(e)}")
        return None


MOTION_PLAUSIBILITY_INSTRUCTIONS = """These {n} images are closely-spaced consecutive frames from a koi mucus-scrape microscope video, spanning about {duration:.1f} real seconds. All frames show the EXACT SAME fixed crop of the slide - the crop position never changes between frames, only time passes - so any change you see between frames is something actually happening in the scene, not camera movement.

Look at how the scene changes across this sequence, frame by frame. Does the organism near the center show genuine biological movement - gliding or crawling across the slide, independently changing position relative to the background clutter around it, or visibly changing shape/pose over time - consistent with a living, motile organism? Or does it stay essentially fixed in place relative to its surroundings throughout, like a static air bubble, piece of debris, or optical artifact that isn't moving under its own power?

Weigh relative position (compared to fixed background features, not the frame edges) much more heavily than subtle flickering or focus changes, which can happen even to something that isn't really moving.

If it does show real movement, also judge the CHARACTER of that movement: does it mostly spin, rotate, or vibrate around a roughly fixed spot without actually travelling far (like a top spinning in place)? Or does it clearly glide/crawl/travel a real distance across the visible field over time, ending up near different background features than it started near? If it shows no real movement, or the pattern genuinely isn't clear either way, say so honestly rather than guessing.

Separately, judge the organism's BODY behaviour. Does the body itself genuinely bend, stretch, curl or compress in a flexible, worm-like way (the shape deforms)? Or does it act like a rigid body - keeping the same basic shape and only APPEARING to change outline because it rotates or tumbles to show a different side? Be careful: a flat, leaf-shaped cell turning edge-on versus face-on looks elongated in one frame and round in the next, but that is rotation of a rigid body, not flexing. Only say it is flexing if the body's own shape visibly deforms (bends, curls, stretches along its length) between frames.

Respond with ONLY a JSON object, no other text and no markdown fences:
{{"shows_organism_movement": true/false, "movement_pattern": "spinning_in_place"/"translating_across_frame"/"unclear", "body_deformation": "flexing_body"/"rigid_body"/"unclear", "reasoning": "one short sentence describing what you actually observe changing (or not) across the sequence, relative to the background"}}"""


def _extract_movement_burst(video_path: str, fps: float, target_size: int, detection: dict,
                             duration_seconds: float = 3.0, num_frames: int = 10, padding_factor: float = 2.5,
                             crop_size: int = 400):
    """Extracts num_frames crops from a FIXED pixel region of the real video
    (centered on the detection's position, with generous padding for
    background context) spread across duration_seconds of real time
    starting at the detection's own timestamp. Deliberately does NOT
    recenter the crop window per frame (unlike crop_detection) - the whole
    point is to let a human-like viewer (here, Claude's vision) see the
    organism's position change RELATIVE TO a fixed background, the same
    cue a person watching the video naturally uses, instead of us trying to
    re-derive "did it move" mathematically via point-tracking."""
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not orig_w or not orig_h:
        cap.release()
        return [], []
    scale_x, scale_y = orig_w / target_size, orig_h / target_size
    cx, cy = detection["x"] * scale_x, detection["y"] * scale_y
    pad_w = detection["width"] * scale_x * padding_factor
    pad_h = detection["height"] * scale_y * padding_factor
    x1, y1 = max(0, int(cx - pad_w / 2)), max(0, int(cy - pad_h / 2))
    x2, y2 = min(orig_w, int(cx + pad_w / 2)), min(orig_h, int(cy + pad_h / 2))

    seed_frame = int(round(detection["timestamp"] * fps)) if fps else 0
    total_span_frames = max(1, int(fps * duration_seconds)) if fps else num_frames
    # Found in real production logs 2026-10-14: most test clips are only 6-19s
    # long, so a 25s window starting at the candidate's own timestamp ran off
    # the end of the video - some scans got only 4 frames, others fewer than the
    # 3-frame minimum and silently skipped the check entirely. Clamp the window
    # to fit inside the video: shrink it to the video's real length if needed,
    # and shift the start earlier (not just truncate) so the full available
    # span is still used.
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frames > 0:
        total_span_frames = min(total_span_frames, video_frames - 1)
        seed_frame = max(0, min(seed_frame, video_frames - 1 - total_span_frames))
    step = max(1, total_span_frames // max(1, num_frames - 1))

    crops_b64, timestamps = [], []
    for i in range(num_frames):
        frame_num = seed_frame + i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        h, w = crop.shape[:2]
        if max(h, w) > crop_size:
            scale = crop_size / max(h, w)
            crop = cv2.resize(crop, (int(w * scale), int(h * scale)))
        _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        crops_b64.append(base64.b64encode(buffer).decode('utf-8'))
        timestamps.append(frame_num / fps if fps else 0)
    cap.release()
    return crops_b64, timestamps


def check_movement_plausibility(video_path: str, fps: float, target_size: int, detection: dict,
                                 duration_seconds: float = 3.0, num_frames: int = 10, padding_factor: float = 2.5):
    """PROTOTYPE (2026-10-13): built after Adam pointed out he could easily
    tell a real organism from an air bubble just by watching the video, even
    on footage blurry enough to defeat both the still-frame label-
    plausibility check and our own point-tracking movement signal (which
    reduces 10s of motion to a couple of scalar numbers, fragile exactly
    where a human's holistic perception of motion isn't). Shows Claude's
    vision a dense burst of closely-spaced frames from a FIXED crop window
    instead, closer to what a person watching the clip actually sees, rather
    than trying to re-derive "did it move" mathematically.

    Found testing directly: a 3s window (the original default) was too short
    to catch a real organism's own slow, loosely-circular drift - confirmed
    by eye across a longer ~20s span before widening this. duration_seconds/
    num_frames/padding_factor are exposed so the caller can tune the window
    to the kind of movement actually being looked for."""
    if not ANTHROPIC_API_KEY:
        return None

    crops_b64, timestamps = _extract_movement_burst(
        video_path, fps, target_size, detection,
        duration_seconds=duration_seconds, num_frames=num_frames, padding_factor=padding_factor,
    )
    if len(crops_b64) < 3:
        return None

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        content = [{"type": "text", "text": f"Candidate class from the detection model: {detection.get('class', 'unknown')}."}]
        for i, crop in enumerate(crops_b64):
            content.append({"type": "text", "text": f"Frame {i + 1} (t={timestamps[i]:.2f}s):"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": crop}})
        content.append({
            "type": "text",
            "text": MOTION_PLAUSIBILITY_INSTRUCTIONS.format(
                n=len(crops_b64), duration=timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
            ),
        })

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[
                {"role": "user", "content": content},
                {"role": "assistant", "content": "{"},
            ],
        )

        raw = "{" + message.content[0].text
        print(f"Movement-visual-plausibility raw response: {raw}")
        result = json.loads(raw)

        movement_pattern = result.get("movement_pattern")
        return {
            "shows_organism_movement": bool(result.get("shows_organism_movement", False)),
            "movement_pattern": movement_pattern if movement_pattern in (
                "spinning_in_place", "translating_across_frame", "unclear"
            ) else "unclear",
            "body_deformation": result.get("body_deformation") if result.get("body_deformation") in (
                "flexing_body", "rigid_body", "unclear"
            ) else "unclear",
            "reasoning": str(result.get("reasoning", ""))[:400],
        }

    except Exception as e:
        print(f"Movement-visual-plausibility check failed: {str(e)}")
        return None


NO_DETECTION_COMMENTARY_PROMPT = """You are looking at frames from a koi mucus-scrape microscope video. A specialist model already screened this video for a specific list of koi parasites (Costia, Chilodonella, Gill Flukes, Skin Fluke, White Spot, and others) and found none of them.

Look across these frames for anything that looks like a distinct living organism, even if you don't recognise the exact species. Common harmless things people mistake for parasites include rotifers, ciliates, and other free-living microorganisms, or bits of plant/algae/debris — none of these are koi parasites.

If you can see something worth mentioning, describe it in general, appropriately hedged terms — never claim a confident species identification, and never claim certainty that it's harmless just because it isn't on the parasite list. If nothing distinct and identifiable is visible, or you're not confident enough to say anything useful, leave the commentary empty.

This is shown to a koi hobbyist purely as educational context, not a diagnosis or reassurance — the parasite screen already ran and found nothing; this is only about what else, if anything, is visible.

Respond with ONLY a JSON object, no other text and no markdown fences:
{"organism_visible": true/false, "commentary": "one or two short plain-English sentences addressed to the user, or empty string"}"""

MAX_COMMENTARY_FRAMES = 12


def select_representative_frames(frames: list, max_count: int = MAX_COMMENTARY_FRAMES) -> list:
    """Evenly subsample so long videos don't send dozens of frames to Claude in one call."""
    if len(frames) <= max_count:
        return frames
    step = len(frames) / max_count
    return [frames[int(i * step)] for i in range(max_count)]


def check_no_detection_commentary(frames: list):
    """When the parasite model found nothing, ask Claude to describe anything
    else visible in general terms. Never a diagnosis, never used to identify
    a parasite — only shown alongside a genuine no-detection result."""
    if not ANTHROPIC_API_KEY or not frames:
        return None

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        content = []
        for f in frames:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": f}})
        content.append({"type": "text", "text": NO_DETECTION_COMMENTARY_PROMPT})

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[
                {"role": "user", "content": content},
                {"role": "assistant", "content": "{"},
            ],
        )

        raw = "{" + message.content[0].text
        print(f"No-detection commentary raw response: {raw}")
        result = json.loads(raw)

        # max_tokens=300 is a *token* budget (~4 chars/token), so this is a
        # generous backstop, not the routine limit — it should rarely bind.
        return {
            "organism_visible": bool(result.get("organism_visible", False)),
            "commentary": str(result.get("commentary", ""))[:600],
        }

    except Exception as e:
        print(f"No-detection commentary failed: {str(e)}")
        return None


def check_frame_quality(img_base64: str):
    if not ANTHROPIC_API_KEY:
        print("Quality check skipped: no ANTHROPIC_API_KEY set")
        return None

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_base64
                            }
                        },
                        {"type": "text", "text": QUALITY_PROMPT}
                    ]
                },
                {"role": "assistant", "content": "{"}
            ]
        )

        raw = "{" + message.content[0].text
        print(f"Quality check raw response: {raw[:300]}")

        result = json.loads(raw)

        return {
            "is_sample": bool(result.get("is_sample", True)),
            "in_focus": bool(result.get("in_focus", True)),
            "is_screen_photo": bool(result.get("is_screen_photo", False)),
            "note": str(result.get("note", ""))[:200],
        }

    except Exception as e:
        print(f"Quality check failed: {str(e)}")
        return None


def _call_roboflow(frame_count: int, img_base64: str):
    """Runs in a worker thread - one Roboflow inference call for one sampled
    frame. Returns (frame_count, img_base64, response_or_None, error_or_None)
    so the caller can process results as they complete, same as the old
    inline per-frame handling did."""
    try:
        response = requests.post(
            f"https://detect.roboflow.com/{MODEL_ID}",
            params={
                "api_key": ROBOFLOW_API_KEY,
                "confidence": CONFIDENCE_THRESHOLD
            },
            data=img_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30
        )
        return frame_count, img_base64, response, None
    except requests.exceptions.Timeout:
        return frame_count, img_base64, None, "timed out"
    except requests.exceptions.RequestException as e:
        return frame_count, img_base64, None, str(e)


def _vision_body_deformation(video_path, fps, target_size, detection):
    """Asks the vision movement check one narrow question about the candidate's
    own body: does it flex (bend/curl/stretch, a fluke-like behaviour) or is it a
    rigid body that only looks like it changes shape by rotating? Returns
    "flexing_body", "rigid_body", "unclear", or None if the check couldn't run.
    Built after a real production miss (2026-10-14): a flat, leaf-shaped
    Chilodonella tumbling edge-on/face-on read aspect_ratio_range=3.69 on a
    clean track and was promoted to Skin Fluke - the shape signal measures
    outline change but can't tell non-rigid flexing from rigid rotation."""
    try:
        m = check_movement_plausibility(
            video_path, fps, target_size, detection,
            duration_seconds=25.0, num_frames=12, padding_factor=4.0,
        )
    except Exception as e:
        print(f"Body-deformation check failed (non-fatal): {e}")
        return None
    return m.get("body_deformation") if m else None


def _apply_shape_promotion(detections, old_primary, new_primary, reasoning, shape_stats):
    """Shared mechanics for letting the shape-corroboration signal swap which
    detection is primary, used by both directions (promoting a mobile
    secondary, or demoting a mobile primary back down to a static one). Does
    three things that are easy to get wrong individually (confirmed directly
    - the first version of this only did the reorder, which looked like a
    completely different, confusingly-contradictory result to the end user):
    shows a fixed, honest confidence instead of old_primary's/new_primary's
    own possibly-low-or-misleading raw Roboflow score, carries the
    same-organism-duplicate flag over to whichever one ends up demoted (that
    check only ever marks the ORIGINAL secondary, so without this the
    demoted detection wrongly reappears as a separate result), and reorders
    detections in place."""
    new_primary["shape_promoted"] = True
    new_primary["roboflow_confidence"] = new_primary["confidence"]
    new_primary["confidence"] = SHAPE_PROMOTED_FIXED_CONFIDENCE
    new_primary["shape_promotion_reasoning"] = reasoning
    # The same-organism check ran earlier with old_primary as primary, so any
    # "same organism" verdict for this pair is sitting on new_primary (it was
    # the "other" detection at check time) - move it onto old_primary, which
    # is the one now demoted to secondary. Without this the frontend's
    # duplicate-secondary filter (which only ever checks a SECONDARY's own
    # flag) has no idea the demoted detection is a duplicate of the new
    # primary, and shows it as a separate, confusingly different result -
    # confirmed directly as a real bug testing this today.
    old_primary["same_organism_as_primary"] = new_primary.pop("same_organism_as_primary", None)
    old_primary["same_organism_reasoning"] = new_primary.pop("same_organism_reasoning", None)
    print(f"Primary changed from {old_primary['class']} to {new_primary['class']} based on the shape-corroboration signal ({shape_stats})")
    detections.remove(new_primary)
    detections.insert(0, new_primary)
    return new_primary


@app.get("/")
def health_check():
    return {"status": "KoiScan backend running"}


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
        interval = max(1, int(fps * FRAME_INTERVAL))

        print(f"Video opened: fps={fps}, interval={interval}, model={MODEL_ID}, size={TARGET_SIZE}, conf={CONFIDENCE_THRESHOLD}")

        # Pass 1: read and encode every sampled frame. Cheap, local, entirely
        # sequential (video decoding doesn't benefit from parallelizing and
        # frame order matters for interval sampling) - the expensive part is
        # the network round-trip to Roboflow, handled in pass 2 below.
        sampled = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                frame_resized = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE))
                _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                sampled.append((frame_count, img_base64))
            frame_count += 1
        cap.release()

        sampled_frames = [b64 for _, b64 in sampled]  # kept for the no-detection-commentary path below

        # Pass 2: the Roboflow calls themselves, run concurrently (bounded)
        # instead of one-at-a-time - this is the actual latency win. Note:
        # unlike the old sequential version, an auth error no longer stops
        # in-flight/queued calls early (all sampled frames' requests are
        # already dispatched by the time the first response comes back) -
        # a real but small cost (a handful of fast-failing calls) traded for
        # the parallelism win on the common, non-error path.
        best_per_class = {}
        # How many distinct sampled frames each class showed up in at all,
        # not just its single best-confidence frame. A one-off high-
        # confidence misread on a single frame (e.g. a blurry Chilodonella
        # read as Trichodina) shouldn't out-rank a class the model saw
        # consistently across many independent frames - frequency across
        # frames is a more robust signal than any one frame's confidence,
        # since each frame is an independent look at (presumably) the same
        # organism. Real bug this fixes, confirmed via Railway logs 2026-10-12:
        # a video the owner confirmed was "obviously Chilodonella" showed
        # BOTH "Chilodonella" and "Trichodina" across its sampled frames,
        # and the app picked Trichodina purely because one single frame
        # happened to score a higher confidence than Chilodonella's best
        # frame - with no regard for which class actually showed up more.
        class_frame_counts = {}
        # Every detection (not just the best one) per class, so the
        # label-plausibility check can be given several genuinely different
        # moments of the same candidate later, not just whichever single
        # frame happened to score highest - see select_diverse_frames.
        class_all_frames = {}
        roboflow_attempts = len(sampled)
        roboflow_failures = 0
        auth_error = False

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_ROBOFLOW_CALLS) as executor:
            futures = [executor.submit(_call_roboflow, fc, b64) for fc, b64 in sampled]
            for future in as_completed(futures):
                frame_count_i, img_base64_i, response, error = future.result()

                if error is not None:
                    roboflow_failures += 1
                    print(f"Frame {frame_count_i}: Roboflow request failed - {error}")
                    continue

                print(f"Frame {frame_count_i}: status={response.status_code} body={response.text[:300]}")

                if response.ok:
                    result = response.json()
                    predictions = result.get("predictions", [])
                    for pred in predictions:
                        parasite_class = pred["class"]
                        confidence = pred["confidence"]
                        class_frame_counts[parasite_class] = class_frame_counts.get(parasite_class, 0) + 1
                        entry = {
                            "class": parasite_class,
                            "confidence": confidence,
                            "timestamp": frame_count_i / fps,
                            "frame_base64": img_base64_i,
                            "x": pred["x"],
                            "y": pred["y"],
                            "width": pred["width"],
                            "height": pred["height"]
                        }
                        class_all_frames.setdefault(parasite_class, []).append(entry)
                        if parasite_class not in best_per_class or confidence > best_per_class[parasite_class]["confidence"]:
                            best_per_class[parasite_class] = entry
                else:
                    roboflow_failures += 1
                    print(f"Frame {frame_count_i}: Roboflow error - status={response.status_code} body={response.text}")
                    if response.status_code in [401, 403]:
                        auth_error = True

        for parasite_class, det in best_per_class.items():
            det["frame_count"] = class_frame_counts.get(parasite_class, 0)

        print(f"Processing complete: {frame_count} frames total, {roboflow_attempts} roboflow calls, {roboflow_failures} failures, auth_error={auth_error}, classes_found={list(best_per_class.keys())}, frame_counts={class_frame_counts}")

        if auth_error:
            return {
                "error": "ROBOFLOW_UNAVAILABLE",
                "message": "Unable to reach the AI model. Please try again later."
            }

        if roboflow_attempts > 0 and roboflow_failures == roboflow_attempts:
            return {
                "error": "ROBOFLOW_UNAVAILABLE",
                "message": "Unable to reach the AI model. Please try again later."
            }

        if roboflow_attempts > 0 and roboflow_failures / roboflow_attempts > 0.8 and not best_per_class:
            return {
                "error": "ROBOFLOW_DEGRADED",
                "message": "The AI model is experiencing issues. Results may be unreliable."
            }

        # Motion-detection fallback for Costia: a single still frame can't
        # capture the erratic-darting-movement signal that identifies it, so
        # Roboflow has near-zero recall here no matter how well trained -
        # see motion_costia.py's docstring for the full history. Only runs
        # when Roboflow found NOTHING at all (not merely "didn't find
        # Costia specifically") - this path is genuinely expensive (motion
        # analysis + tracking + a Claude call per candidate tried, tested at
        # up to ~100s on a video with no real Costia in it), so a scan that
        # already has a clear, confident detection of another species
        # shouldn't also pay this cost just to check for Costia too.
        if not best_per_class:
            def _costia_plausibility_check(det):
                # The motion candidate-finder always returns its best-ranked
                # guess within THIS video, even when nothing organism-like is
                # really there (confirmed directly: it tracked on-screen
                # caption text on an unrelated video) - gate on a check
                # purpose-built for Costia (see check_costia_candidate_plausible's
                # docstring for why check_label_plausibility isn't the right
                # tool here - tested directly, it rejected 100% of candidates
                # including previously-confirmed-genuine ones).
                crop = crop_and_mark_detection(det["frame_base64"], det)
                if not crop:
                    return True  # couldn't crop - fail open, don't block on a local bug
                result = check_costia_candidate_plausible(crop)
                if result is None:
                    return True  # check itself failed - fail open, same as the other Claude checks
                return result["plausible_organism"]

            try:
                costia_detection = find_costia_detection(
                    tmp_path, fps, TARGET_SIZE, plausibility_checker=_costia_plausibility_check
                )
                if costia_detection:
                    print(f"Motion detection found a Costia candidate at {costia_detection['timestamp']:.1f}s")
                    costia_detection["frame_count"] = 1
                    best_per_class["Costia"] = costia_detection
            except Exception as e:
                print(f"Costia motion detection failed (non-fatal): {e}")

        # Sort by frame_count first (how many independent sampled frames
        # agreed on this class), confidence only as a tiebreaker - see the
        # note above class_frame_counts for why frequency beats a single
        # frame's confidence.
        detections = sorted(
            best_per_class.values(),
            key=lambda x: (x.get("frame_count", 0), x["confidence"]),
            reverse=True,
        ) if best_per_class else []

        # PROTOTYPE, upgraded 2026-10-12: label-plausibility used to only
        # ever add a flag for the frontend to caveat the primary with - it
        # never changed what was actually shown. Per Adam's explicit product
        # call the same session ("focus on making sure the primary result is
        # correct", and separately: don't show users a hedge, just be right)
        # it now gets a real say: try candidates in the backend's own ranked
        # order (frame-frequency first), and promote the first one that
        # actually passes this visual check instead of blindly trusting
        # whichever ranked first. Falls back to the original top choice if
        # NONE of them pass (better to show something than nothing) - still
        # flagged with label_plausible=False so that's visible in the data
        # even though the UI doesn't currently surface it.
        if detections:
            chosen = None
            for cand in detections:
                # Upgraded 2026-10-12: give the check several genuinely
                # different moments of this candidate (spread across time,
                # not just highest-confidence frames which tend to cluster
                # around the same pose) instead of just one frame - a gill
                # fluke that visibly stretches/compresses was misjudged
                # against a single compressed-looking frame before this.
                # class_all_frames.get(...) falls back to [cand] itself for
                # the Costia motion-detection path, which was never added to
                # class_all_frames (it doesn't come from the per-frame
                # Roboflow loop at all).
                diverse_frames = select_diverse_frames(class_all_frames.get(cand["class"], [cand]))
                crops, crop_timestamps = [], []
                for f in diverse_frames:
                    c = crop_detection(f["frame_base64"], f)
                    if c:
                        crops.append(c)
                        crop_timestamps.append(f["timestamp"])
                if not crops:
                    chosen = cand  # can't check this one - fail open, stop here
                    break
                # Diagnostic logging added 2026-10-12: a gill fluke video gave
                # two different label-plausibility verdicts on back-to-back
                # identical-looking runs, most likely because a different
                # near-tied-confidence frame won the "representative frame"
                # slot between runs (network-completion-order dependent, not
                # random model flakiness) - if the organism's shape/pose
                # genuinely differs between those frames, two honestly
                # different verdicts on two different images is expected,
                # not a bug. This line makes that provable instead of
                # inferred, and now also shows every timestamp actually used.
                print(f"Checking label plausibility for {cand['class']} using {len(crops)} frame(s) at timestamps {[round(t, 2) for t in crop_timestamps]}s")
                plausibility = check_label_plausibility(crops, cand["class"], cand["confidence"], timestamps=crop_timestamps)
                if plausibility is None:
                    chosen = cand  # check itself failed (e.g. no API key) - fail open
                    break
                cand["label_plausible"] = plausibility["label_plausible"]
                cand["label_plausibility_reasoning"] = plausibility["reasoning"]
                if plausibility["label_plausible"]:
                    chosen = cand
                    break
            # True only when every candidate was actually, successfully checked
            # and every one of them came back implausible - not when the loop
            # broke early because a check couldn't run at all. A clean signal
            # that something is off (see the movement-corroboration block below).
            all_candidates_rejected = chosen is None
            if chosen is None:
                chosen = detections[0]  # none of them passed - fall back to the original top

            if chosen is not detections[0]:
                print(f"Primary promoted from {detections[0]['class']} to {chosen['class']} after it failed the label-plausibility check")
                detections.remove(chosen)
                detections.insert(0, chosen)

            primary = detections[0]
            primary_crop = crop_detection(primary["frame_base64"], primary)

            if primary_crop:
                # For any secondary detections, is this plausibly the same
                # organism as the primary, misread as a different class?
                for det in detections[1:]:
                    other_crop = crop_detection(det["frame_base64"], det)
                    if not other_crop:
                        continue
                    comparison = check_same_organism(
                        primary_crop, primary["class"], primary["confidence"],
                        other_crop, det["class"], det["confidence"],
                        primary_timestamp=primary.get("timestamp"),
                        other_timestamp=det.get("timestamp"),
                    )
                    if comparison:
                        det["same_organism_as_primary"] = comparison["same_organism"]
                        det["same_organism_reasoning"] = comparison["reasoning"]

            # PROTOTYPE, Phase 3 (2026-10-13): the shape-corroboration signal now
            # gets a real say in the result, not just a log line - per Adam's
            # explicit call once the very first real production example (this
            # same day) showed exactly why: a Gill Fluke compressed in its
            # sampled frames read as a free-living ciliate to the still-frame
            # label-plausibility check (label_plausible=False on BOTH
            # candidates), while this shape signal correctly read strongly
            # "mobile" for the same video (aspect_ratio_range=4.861,
            # n_reseeds=0) - see motion_shape_check.py's docstring for the
            # full validation history behind the thresholds used here.
            #
            # Two directions, deliberately NOT symmetric. Promoting a mobile
            # secondary over a static primary just needs a strongly-mobile
            # reading. The reverse - demoting a mobile primary down to a
            # static secondary - ALSO requires the mobile primary to have
            # already failed its own label-plausibility check: a resting
            # mobile fluke was measured reading as low as 0.014 (lower than
            # any static-group reading seen), so a low reading alone can't
            # safely confirm a mobile primary is wrong - but a real fluke's
            # internal anatomy doesn't change just because it's holding
            # still, so requiring the independent visual check to ALSO have
            # rejected it closes that gap (see motion_shape_check.py's
            # SHAPE_DEMOTE_STATIC_THRESHOLD docstring for the full reasoning).
            #
            # Trigger stays narrow on purpose: only runs this expensive,
            # still-fairly-new check when the primary and some OTHER detected
            # candidate actually disagree on movement group - a direct match
            # for the real bugs found today, keeping the cost off ordinary scans.
            other_classes = {d["class"] for d in detections[1:]}
            cross_group_ambiguity = (
                (primary["class"] in STATIC_GROUP and other_classes & MOBILE_GROUP) or
                (primary["class"] in MOBILE_GROUP and other_classes & STATIC_GROUP)
            )
            shape_check_applied = False
            if cross_group_ambiguity:
                try:
                    shape_stats = check_shape_group(tmp_path, fps, TARGET_SIZE, primary)
                except Exception as e:
                    shape_stats = None
                    print(f"Shape-corroboration check failed (non-fatal): {e}")
                if shape_stats:
                    print(f"Shape-check for primary={primary['class']} vs. secondaries={other_classes}: {shape_stats}")
                    # The shape signal only discriminates "static vs. mobile" at the
                    # group level - it can't tell Trichodina from Chilodonella, or Gill
                    # Flukes from Skin Fluke, apart. Found testing today: when BOTH
                    # static species showed up as secondaries on one video, this used to
                    # just grab whichever one Roboflow happened to rank higher
                    # (Trichodina over Chilodonella, 2 frames vs. 1) with no actual
                    # evidence favoring it - a specific-looking guess with nothing behind
                    # it. Only act when exactly ONE class from the target group is
                    # present among the secondaries - real evidence narrows it to one
                    # candidate; two or more means the group split alone can't resolve
                    # which species it is, and inventing an answer anyway would be worse
                    # than leaving the (known-uncertain) original primary in place.
                    mobile_classes_present = other_classes & MOBILE_GROUP
                    static_classes_present = other_classes & STATIC_GROUP
                    if (primary["class"] in STATIC_GROUP and len(mobile_classes_present) == 1
                            and shape_signal_supports_mobile(shape_stats)):
                        mobile_cand = next(d for d in detections[1:] if d["class"] in MOBILE_GROUP)
                        deformation = _vision_body_deformation(tmp_path, fps, TARGET_SIZE, primary)
                        print(f"Body-deformation check for primary={primary['class']}: {deformation}")
                        if deformation == "rigid_body":
                            # Veto only: a rigid body that tumbles/rotates shows a changing
                            # outline too, so a high aspect-ratio range alone isn't proof of
                            # flexing. Leave the primary alone.
                            print(f"Shape promotion to {mobile_cand['class']} vetoed - vision judges the body rigid (rotating), not flexing")
                        else:
                            reasoning = (
                                f"Movement tracking found aspect_ratio_range={shape_stats['aspect_ratio_range']} "
                                f"(n_reseeds={shape_stats['n_reseeds']}) - strongly consistent with a flexing/mobile "
                                f"organism, not the static reading expected for {primary['class']}."
                            )
                            primary = _apply_shape_promotion(detections, primary, mobile_cand, reasoning, shape_stats)
                            shape_check_applied = True
                    elif (primary["class"] in MOBILE_GROUP and primary.get("label_plausible") is False
                          and len(static_classes_present) == 1
                          and shape_signal_supports_static(shape_stats)):
                        static_cand = next(d for d in detections[1:] if d["class"] in STATIC_GROUP)
                        deformation = _vision_body_deformation(tmp_path, fps, TARGET_SIZE, primary)
                        print(f"Body-deformation check for primary={primary['class']}: {deformation}")
                        if deformation == "flexing_body":
                            print(f"Shape demotion to {static_cand['class']} vetoed - vision judges the body visibly flexing")
                        else:
                            reasoning = (
                                f"Movement tracking found aspect_ratio_range={shape_stats['aspect_ratio_range']} "
                                f"(n_reseeds={shape_stats['n_reseeds']}) - strongly consistent with a static/round "
                                f"organism, and {primary['class']} already failed its own label-plausibility check."
                            )
                            primary = _apply_shape_promotion(detections, primary, static_cand, reasoning, shape_stats)
                            shape_check_applied = True

            # PROTOTYPE (2026-10-13): movement corroboration via Claude's own
            # vision, not point-tracking math. Built same day after a real,
            # multi-round investigation: the shape-only signal (does the
            # outline flex?) couldn't catch an air bubble out-ranking a
            # genuinely moving Chilodonella, since both were in the same
            # movement group. A hand-built point-tracking replacement
            # (path_length_bl) was tried next, but proved fragile - a smooth,
            # low-texture object's tracked points can silently drift off it
            # (the aperture problem), and even once that was fixed with a
            # texture gate, Adam pointed out the deeper issue directly: he
            # could easily tell the real organism from the bubble just by
            # watching the video, even on footage blurry enough to defeat
            # every one of these still-frame or scalar-statistic checks.
            #
            # This asks Claude's vision the question directly instead of
            # re-deriving it mathematically: shown a dense burst of frames
            # from a FIXED crop window over a real ~25s span (closer to
            # what a person watching the clip actually perceives), does this
            # candidate visibly move relative to the background, or stay put?
            # Verified directly against the real bubble-vs-Chilodonella pair
            # from today's testing: correctly read the bubble as not moving
            # and the real organism as "gliding... relative to fixed
            # background debris" - succeeding exactly where the point-
            # tracking approach kept failing. See check_movement_plausibility's
            # docstring for why the window needed to be this long (a shorter
            # 3s test window missed the organism's own slow drift entirely).
            if all_candidates_rejected and not shape_check_applied:
                # Diagnostic logging added 2026-10-13: the SAME labeled candidate
                # gave contradictory movement_pattern verdicts (translating vs.
                # spinning) across two back-to-back real test runs on the same
                # video. Printing the exact seed coordinates/timestamp used each
                # time makes it provable whether Roboflow picked a different
                # underlying position this run (already confirmed non-
                # deterministic elsewhere in this project) vs. the same position
                # getting a genuinely different vision-model judgement.
                print(f"Movement-check seed for primary={primary['class']}: "
                      f"timestamp={primary.get('timestamp'):.2f}s x={primary.get('x'):.1f} y={primary.get('y'):.1f} "
                      f"w={primary.get('width'):.1f} h={primary.get('height'):.1f}")
                try:
                    primary_motion = check_movement_plausibility(
                        tmp_path, fps, TARGET_SIZE, primary,
                        duration_seconds=25.0, num_frames=12, padding_factor=4.0,
                    )
                except Exception as e:
                    primary_motion = None
                    print(f"Movement-visual-plausibility check failed (non-fatal): {e}")
                if primary_motion:
                    print(f"Movement-visual-check for primary={primary['class']}: {primary_motion}")
                    if not primary_motion["shows_organism_movement"]:
                        moving_cands = []
                        for cand in detections[1:]:
                            try:
                                cand_motion = check_movement_plausibility(
                                    tmp_path, fps, TARGET_SIZE, cand,
                                    duration_seconds=25.0, num_frames=12, padding_factor=4.0,
                                )
                            except Exception as e:
                                cand_motion = None
                                print(f"Movement-visual-plausibility check failed for {cand['class']} (non-fatal): {e}")
                            if cand_motion:
                                print(f"Movement-visual-check for {cand['class']}: {cand_motion}")
                                if cand_motion["shows_organism_movement"]:
                                    moving_cands.append((cand, cand_motion))
                        if len(moving_cands) == 1:
                            alt_cand, alt_motion = moving_cands[0]
                            reasoning = (
                                f"Visual movement check found {primary['class']} showing no real movement "
                                f"({primary_motion['reasoning']}) while {alt_cand['class']} showed clear "
                                f"movement ({alt_motion['reasoning']}) - more consistent with {alt_cand['class']} "
                                f"being the actual living organism, especially since every candidate here "
                                f"failed its own visual plausibility check."
                            )
                            primary = _apply_shape_promotion(detections, primary, alt_cand, reasoning, alt_motion)
                    else:
                        # PROTOTYPE (2026-10-13): found testing the check above - a real,
                        # confirmed-moving organism can still be given the WRONG SPECIES
                        # NAME by Roboflow (the detection was real, just mislabeled), which
                        # the "is primary static" check above can't catch since the primary
                        # really is moving. Per Adam's own domain knowledge: Trichodina
                        # spins/vibrates largely in place, Chilodonella glides a real
                        # distance across the slide - two movement GROUP members
                        # (STATIC_GROUP) with genuinely different movement characters once
                        # you know both are alive. If primary's claimed class has a known
                        # expected pattern and the OBSERVED pattern contradicts it, check
                        # whether exactly one other candidate's claimed class matches the
                        # pattern actually observed - and independently confirm THAT
                        # candidate also moves with its own expected pattern - before
                        # relabeling. Only Trichodina/Chilodonella have an established
                        # expected pattern so far (nothing conclusive yet for White Spot).
                        expected_pattern = EXPECTED_MOVEMENT_PATTERN.get(primary["class"])
                        observed_pattern = primary_motion.get("movement_pattern")
                        if expected_pattern and observed_pattern not in (None, "unclear") and observed_pattern != expected_pattern:
                            pattern_matched_cands = []
                            for cand in detections[1:]:
                                if EXPECTED_MOVEMENT_PATTERN.get(cand["class"]) != observed_pattern:
                                    continue
                                try:
                                    cand_motion = check_movement_plausibility(
                                        tmp_path, fps, TARGET_SIZE, cand,
                                        duration_seconds=25.0, num_frames=12, padding_factor=4.0,
                                    )
                                except Exception as e:
                                    cand_motion = None
                                    print(f"Movement-visual-plausibility check failed for {cand['class']} (non-fatal): {e}")
                                if cand_motion:
                                    print(f"Movement-visual-check for {cand['class']}: {cand_motion}")
                                    if (cand_motion["shows_organism_movement"]
                                            and cand_motion.get("movement_pattern") == observed_pattern):
                                        pattern_matched_cands.append((cand, cand_motion))
                            if len(pattern_matched_cands) == 1:
                                alt_cand, alt_motion = pattern_matched_cands[0]
                                reasoning = (
                                    f"Visual movement check found {primary['class']} moving with a "
                                    f"'{observed_pattern}' pattern ({primary_motion['reasoning']}) - inconsistent "
                                    f"with {primary['class']}'s own expected movement, but matching "
                                    f"{alt_cand['class']}'s expected pattern instead, independently confirmed "
                                    f"({alt_motion['reasoning']})."
                                )
                                primary = _apply_shape_promotion(detections, primary, alt_cand, reasoning, alt_motion)

        # Quality check: use the best detection frame, or a middle frame if nothing was found
        quality = None
        commentary = None
        if detections:
            quality = check_frame_quality(detections[0]["frame_base64"])
        elif sampled_frames:
            middle = sampled_frames[len(sampled_frames) // 2]
            quality = check_frame_quality(middle)
            commentary = check_no_detection_commentary(select_representative_frames(sampled_frames))

        if quality:
            print(f"Quality: is_sample={quality['is_sample']} in_focus={quality['in_focus']} screen_photo={quality['is_screen_photo']} note={quality['note']}")

        if commentary:
            print(f"Commentary: organism_visible={commentary['organism_visible']} text={commentary['commentary']}")

        if detections:
            return {
                "detected": True,
                "detections": detections,
                "detection": detections[0],
                "quality": quality
            }
        else:
            return {"detected": False, "quality": quality, "commentary": commentary}

    finally:
        os.unlink(tmp_path)


@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {"predictions": []}

    frame_resized = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE))
    _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    response = requests.post(
        f"https://detect.roboflow.com/{MODEL_ID}",
        params={
            "api_key": ROBOFLOW_API_KEY,
            "confidence": CONFIDENCE_THRESHOLD
        },
        data=img_base64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30
    )

    if response.ok:
        return response.json()
    return {"predictions": []}
