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
from motion_shape_check import check_shape_group, STATIC_GROUP, MOBILE_GROUP

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

Look closely at the actual visible detail inside the organism, not just its outline shape. Does the internal structure genuinely look consistent with that parasite, or does it look more like something else — for example a free-living ciliate/protozoan (like a Paramecium), debris, or an artefact? An elongated or fluke-like outline alone is not enough to confirm a fluke-type parasite; internal detail matters more than silhouette.

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

            # PROTOTYPE, Phase 2 (2026-10-13): shape-corroboration check, computed
            # and LOGGED ONLY - must not affect detections or what's returned to
            # the user (see motion_shape_check.py's docstring for the full
            # reasoning and the two real misclassifications this is built
            # against). Deliberately narrow trigger: only runs when the primary
            # and some OTHER detected candidate actually disagree on which of
            # the two movement groups they're in - this is a direct match for
            # both real bugs found today, and keeps the cost of this expensive,
            # still-unproven check off every ordinary scan.
            other_classes = {d["class"] for d in detections[1:]}
            cross_group_ambiguity = (
                (primary["class"] in STATIC_GROUP and other_classes & MOBILE_GROUP) or
                (primary["class"] in MOBILE_GROUP and other_classes & STATIC_GROUP)
            )
            if cross_group_ambiguity:
                try:
                    shape_stats = check_shape_group(tmp_path, fps, TARGET_SIZE, primary)
                    if shape_stats:
                        print(f"Shape-check (log-only) for primary={primary['class']} vs. secondaries={other_classes}: {shape_stats}")
                except Exception as e:
                    print(f"Shape-corroboration check failed (non-fatal): {e}")

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
