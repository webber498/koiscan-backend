from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import requests
import base64
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
MODEL_ID = "koi-parasites-3-gwtia/7"
FRAME_INTERVAL = 3
TARGET_SIZE = 640

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

        best_detection = None
        best_confidence = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                frame_resized = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE))
                _, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_base64 = base64.b64encode(buffer).decode('utf-8')

                response = requests.post(
                    f"https://detect.roboflow.com/{MODEL_ID}",
                    params={"api_key": ROBOFLOW_API_KEY},
                    data=img_base64,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )

                if response.ok:
                    result = response.json()
                    print(f"Frame {frame_count}: {len(result.get('predictions', []))} predictions")
                    predictions = result.get("predictions", [])
                    for pred in predictions:
                        if pred["confidence"] > best_confidence:
                            best_confidence = pred["confidence"]
                            best_detection = {
                                "class": pred["class"],
                                "confidence": pred["confidence"],
                                "timestamp": frame_count / fps,
                                "frame_base64": img_base64,
                                "x": pred["x"],
                                "y": pred["y"],
                                "width": pred["width"],
                                "height": pred["height"]
                            }

            frame_count += 1

        cap.release()

        if best_detection:
            return {"detected": True, "detection": best_detection}
        else:
            return {"detected": False}

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
        params={"api_key": ROBOFLOW_API_KEY},
        data=img_base64,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    if response.ok:
        return response.json()
    return {"predictions": []}
