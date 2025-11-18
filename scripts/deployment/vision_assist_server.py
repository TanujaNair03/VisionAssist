"""
Vision Assistance System - Optimized Online Implementation
Non-blocking gTTS + Frame Skipping for Smooth Video
"""

import os
import io
import time
import base64
import asyncio
import threading
from typing import Dict, List, Optional, Deque
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# Core Libraries
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Audio
from gtts import gTTS
from pydub import AudioSegment

# Server
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse
import uvicorn

# ============================================================================
# CONFIGURATION & OPTIMIZATION
# ============================================================================

YOLO_MODEL_PATH = './yolov8n.pt' # Ensure this path is correct
CONFIDENCE_THRESHOLD = 0.45
PROCESS_WIDTH = 640 # Downscale images to this width for speed
SKIP_FRAMES = 2 # Process 1 frame, skip 2 (Effective 10-15 FPS processing)

# Audio Config
TTS_LANGUAGE = 'en'
MAX_AUDIO_WORKERS = 3 # Allow 3 simultaneous network calls to Google

# Filtering
NOISY_CLASSES = [24, 25, 26, 27, 31, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 73, 74, 75, 77, 78, 79]
KNOWN_HEIGHTS = {0: 1.7, 1: 1.0, 2: 1.5, 3: 1.2, 5: 3.0, 7: 3.5}
DEFAULT_HEIGHT = 2.0
FOCAL_LENGTH = 762.99

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Thread pool for gTTS so it doesn't block the video loop
audio_executor = ThreadPoolExecutor(max_workers=MAX_AUDIO_WORKERS)

pipeline_state = {
    'model': None,
    'track_history': {},
    'cooldowns': {},
    'frame_counter': 0
}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_model():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Loading YOLO on {device}...")
        model = YOLO(YOLO_MODEL_PATH)
        model.to(device)
        return model
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return None

def generate_gtts_background(text: str) -> Optional[bytes]:
    """
    Generates MP3 in a background thread to avoid freezing video.
    """
    try:
        # Optimization: Very short cache check could go here
        tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        
        # Convert to MP3 bytes immediately
        return buffer.getvalue()
    except Exception as e:
        print(f"⚠️ gTTS Error: {e}")
        return None

def estimate_distance(box, cls_id):
    h_px = box[3] - box[1]
    if h_px <= 0: return 0.0
    real_h = KNOWN_HEIGHTS.get(cls_id, DEFAULT_HEIGHT)
    return (real_h * FOCAL_LENGTH) / h_px

def get_direction(center_x, width):
    ratio = center_x / width
    if ratio < 0.33: return "Left"
    if ratio > 0.66: return "Right"
    return "Ahead"

def resize_frame(frame, target_width):
    """Resize frame keeping aspect ratio"""
    h, w = frame.shape[:2]
    if w <= target_width: return frame, 1.0
    scale = target_width / w
    new_dim = (target_width, int(h * scale))
    return cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA), scale

def process_yolo(frame):
    """Main detection logic"""
    model = pipeline_state['model']
    if not model: return frame, [], []

    # OPTIMIZATION: Work on smaller image
    small_frame, scale_factor = resize_frame(frame, PROCESS_WIDTH)
    
    results = model.track(small_frame, conf=CONFIDENCE_THRESHOLD, persist=True, verbose=False)
    
    detections = []
    alerts_to_generate = []
    current_time = time.time()
    h, w = small_frame.shape[:2]

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, cls_id, track_id in zip(boxes, clss, ids):
            cls_id = int(cls_id)
            track_id = int(track_id)

            if cls_id in NOISY_CLASSES or cls_id == 0: continue # Skip humans & clutter

            # Scale box back up to original frame size for drawing
            orig_box = box / scale_factor
            
            # Distance & Logic
            dist = estimate_distance(box, cls_id) # Use small box for calc (proportional)
            cx = (box[0] + box[2]) / 2
            direction = get_direction(cx, w)
            
            cls_name = model.names[cls_id]
            label = f"{cls_name} {dist:.1f}m"

            detections.append({
                'box': orig_box,
                'label': label,
                'color': (0, 255, 0)
            })

            # Alert Logic (Throttled)
            # Only alert if < 10m and cooldown passed
            if dist < 10.0:
                last_alert = pipeline_state['cooldowns'].get(track_id, 0)
                # 8 second cooldown per object
                if current_time - last_alert > 8.0:
                    alert_text = f"{cls_name}, {int(dist)} meters, {direction}"
                    alerts_to_generate.append(alert_text)
                    pipeline_state['cooldowns'][track_id] = current_time

    return detections, alerts_to_generate

# ============================================================================
# SERVER SETUP
# ============================================================================

app = FastAPI()

@app.on_event("startup")
def startup():
    pipeline_state['model'] = load_model()

@app.get("/")
def index():
    # Ensure this points to your actual HTML file location
    if os.path.exists("websocket_camera_client.html"):
        return HTMLResponse(open("websocket_camera_client.html", "r").read())
    return "Client HTML not found"

@app.websocket("/ws/camera/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("command") == "process_frame":
                pipeline_state['frame_counter'] += 1
                
                # 1. Decode Image
                try:
                    f_bytes = base64.b64decode(data['frame_data'])
                    np_arr = np.frombuffer(f_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                except:
                    continue

                annotated_frame = frame.copy()
                audio_payloads = []

                # 2. OPTIMIZATION: Frame Skipping
                # Only run YOLO every N frames to keep UI responsive
                if pipeline_state['frame_counter'] % (SKIP_FRAMES + 1) == 0:
                    detections, alerts = process_yolo(frame)
                    
                    # Draw detections
                    for d in detections:
                        x1, y1, x2, y2 = map(int, d['box'])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), d['color'], 2)
                        cv2.putText(annotated_frame, d['label'], (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, d['color'], 2)
                    
                    # 3. OPTIMIZATION: Non-blocking Audio
                    if alerts:
                        loop = asyncio.get_event_loop()
                        combined_text = ". ".join(alerts)
                        # Run gTTS in separate thread, await the result asynchronously
                        audio_bytes = await loop.run_in_executor(
                            audio_executor, generate_gtts_background, combined_text
                        )
                        if audio_bytes:
                            b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                            audio_payloads.append(b64_audio)
                else:
                    # On skipped frames, just return original image (or detection placeholder)
                    # This makes the video feel smoother even if detections update slower
                    pass

                # 4. Encode and Send Response
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpg_b64 = base64.b64encode(buffer).decode('utf-8')

                response = {
                    "type": "processing_result",
                    "annotated_frame": jpg_b64,
                    "audio_data": audio_payloads,
                    "frame_count": pipeline_state['frame_counter']
                }
                
                await websocket.send_json(response)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")

if __name__ == "__main__":
    # Use 0.0.0.0 to be accessible externally
    uvicorn.run(app, host="0.0.0.0", port=8000)