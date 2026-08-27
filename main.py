from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import requests
import base64
import tempfile
import os
import json
import anthropic

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
CONFIDENCE_THRESHOLD = 39

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

Do these two crops plausibly show the SAME physical organism, seen twice — or do they look like genuinely different organisms? If you're genuinely unsure after weighing the above, prefer "same_organism": true — this only adds an advisory note for the user to double-check, it never removes or changes either detection, so a false "same" costs far less than a false "different" (which would wrongly tell the user two separate parasites are present).

Respond with ONLY a JSON object, no other text and no markdown fences:
{"same_organism": true/false, "reasoning": "one short sentence explaining your judgement"}"""


LABEL_PLAUSIBILITY_INSTRUCTIONS = """This is a cropped image from a koi mucus-scrape microscope video. An object-detection model has labelled what's in this crop as a specific koi parasite (named in the message above).

Look closely at the actual visible detail inside the organism, not just its outline shape. Does the internal structure genuinely look consistent with that parasite, or does it look more like something else — for example a free-living ciliate/protozoan (like a Paramecium), debris, or an artefact? An elongated or fluke-like outline alone is not enough to confirm a fluke-type parasite; internal detail matters more than silhouette.

Respond with ONLY a JSON object, no other text and no markdown fences:
{"label_plausible": true/false, "reasoning": "one short sentence explaining your judgement, focused on what you actually see"}"""


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


def check_same_organism(primary_crop_b64: str, primary_class: str, primary_confidence: float,
                         other_crop_b64: str, other_class: str, other_confidence: float,
                         primary_timestamp: float = None, other_timestamp: float = None):
    """PROTOTYPE: ask Claude whether a secondary detection is plausibly the
    same organism as the primary one, misread as a different class. Never
    used to remove a detection — only to flag it for the user."""
    if not ANTHROPIC_API_KEY:
        return None

    gap_note = ""
    if primary_timestamp is not None and other_timestamp is not None:
        gap = abs(other_timestamp - primary_timestamp)
        gap_note = f' They were detected {gap:.1f} seconds apart in the video.'

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
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
                {"role": "assistant", "content": "{"},
            ],
        )

        raw = "{" + message.content[0].text
        print(f"Same-organism check raw response: {raw[:300]}")
        result = json.loads(raw)

        return {
            "same_organism": bool(result.get("same_organism", False)),
            "reasoning": str(result.get("reasoning", ""))[:200],
        }

    except Exception as e:
        print(f"Same-organism check failed: {str(e)}")
        return None


def check_label_plausibility(crop_b64: str, claimed_class: str, confidence: float):
    """PROTOTYPE: sanity-check the primary detection's claimed class against
    what's actually visible (internal detail, not just outline shape) — e.g.
    a Paramecium being misread as Gill Flukes off silhouette alone. Only
    ever flags the detection for the frontend to caveat, never removes or
    downgrades it."""
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
                        {"type": "text", "text": f'This crop was labelled "{claimed_class}" at {round(confidence * 100)}% confidence by the parasite-detection model.'},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": crop_b64}},
                        {"type": "text", "text": LABEL_PLAUSIBILITY_INSTRUCTIONS},
                    ],
                },
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

        best_per_class = {}
        sampled_frames = []
        frame_count = 0
        roboflow_attempts = 0
        roboflow_failures = 0
        auth_error = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                frame_resized = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE))
                _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                sampled_frames.append(img_base64)

                try:
                    roboflow_attempts += 1
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

                    print(f"Frame {frame_count}: status={response.status_code} body={response.text[:300]}")

                    if response.ok:
                        result = response.json()
                        predictions = result.get("predictions", [])
                        for pred in predictions:
                            parasite_class = pred["class"]
                            confidence = pred["confidence"]
                            if parasite_class not in best_per_class or confidence > best_per_class[parasite_class]["confidence"]:
                                best_per_class[parasite_class] = {
                                    "class": parasite_class,
                                    "confidence": confidence,
                                    "timestamp": frame_count / fps,
                                    "frame_base64": img_base64,
                                    "x": pred["x"],
                                    "y": pred["y"],
                                    "width": pred["width"],
                                    "height": pred["height"]
                                }
                    else:
                        roboflow_failures += 1
                        print(f"Frame {frame_count}: Roboflow error - status={response.status_code} body={response.text}")
                        if response.status_code in [401, 403]:
                            print("Auth error detected — stopping processing immediately")
                            auth_error = True
                            break

                except requests.exceptions.Timeout:
                    roboflow_failures += 1
                    print(f"Frame {frame_count}: Roboflow request timed out")
                except requests.exceptions.RequestException as e:
                    roboflow_failures += 1
                    print(f"Frame {frame_count}: Roboflow request failed - {str(e)}")

            frame_count += 1

        cap.release()

        print(f"Processing complete: {frame_count} frames total, {roboflow_attempts} roboflow calls, {roboflow_failures} failures, auth_error={auth_error}, classes_found={list(best_per_class.keys())}")

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

        detections = sorted(best_per_class.values(), key=lambda x: x["confidence"], reverse=True) if best_per_class else []

        # PROTOTYPE: two Claude-vision sanity checks on the raw Roboflow
        # detections. Both only ever add a flag for the frontend to show as
        # a caveat — neither removes or downgrades a detection.
        if detections:
            primary = detections[0]
            primary_crop = crop_detection(primary["frame_base64"], primary)

            if primary_crop:
                # 1. Does the claimed class match the visible internal detail,
                # or did the model likely key on outline shape alone? (e.g.
                # the Gill Flukes/Paramecium mixup this was built for.)
                plausibility = check_label_plausibility(primary_crop, primary["class"], primary["confidence"])
                if plausibility:
                    primary["label_plausible"] = plausibility["label_plausible"]
                    primary["label_plausibility_reasoning"] = plausibility["reasoning"]

                # 2. For any secondary detections, is this plausibly the same
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
