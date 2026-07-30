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
MODEL_ID = "koi-parasites-3-gwtia/9"
FRAME_INTERVAL = 1
TARGET_SIZE = 576
CONFIDENCE_THRESHOLD = 39

QUALITY_PROMPT = """You are checking the quality of a microscope video frame from a koi fish mucus scrape. A hobbyist has filmed this through a microscope eyepiece, usually with a phone.

Do NOT try to identify any parasites. Only assess whether this image is usable.

Answer these three questions:
1. Is this actually a microscope view of a wet mount sample? (Not a photo of a fish, a pond, a screen showing something else, or an unrelated image.)
2. Is it in focus enough that small organisms would be distinguishable?
3. Is this a direct capture, or is it a photograph of a computer/phone screen? Look for moire patterns, scan lines, screen glare, visible cursors or UI elements.

Respond with ONLY a JSON object, no other text and no markdown fences:
{"is_sample": true/false, "in_focus": true/false, "is_screen_photo": true/false, "note": "one short sentence for the user, or empty string if all is well"}

The note should be plain English addressed to the user, and should never say the video is unusable or ask them to refilm — the analysis has already run and a result is being shown. It is advisory only. For example: "This looks like a recording of a screen rather than a direct capture, which can reduce accuracy." Keep it under 20 words. Leave it as an empty string if quality is fine."""


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

        # Quality check: use the best detection frame, or a middle frame if nothing was found
        quality = None
        if detections:
            quality = check_frame_quality(detections[0]["frame_base64"])
        elif sampled_frames:
            middle = sampled_frames[len(sampled_frames) // 2]
            quality = check_frame_quality(middle)

        if quality:
            print(f"Quality: is_sample={quality['is_sample']} in_focus={quality['in_focus']} screen_photo={quality['is_screen_photo']} note={quality['note']}")

        if detections:
            return {
                "detected": True,
                "detections": detections,
                "detection": detections[0],
                "quality": quality
            }
        else:
            return {"detected": False, "quality": quality}

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
