"""
Vision Assistance System - Single File Modular Implementation
Real-time object detection with audio alerts via HTTP API
With Performance Logging

All modules consolidated into functions for easy deployment
"""

# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================

import os
import io
import time
import json
import base64
import asyncio
import threading
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

# Core ML/CV Libraries
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Audio Processing
from gtts import gTTS
from pydub import AudioSegment

# HTTP Server
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
import uvicorn

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Configuration
YOLO_MODEL_PATH = './yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.4
FOCAL_LENGTH_PIXELS = 762.99

# Object Detection & Filtering
NOISY_CLASSES_TO_IGNORE = [
    24, 25, 26, 27, 31, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 73, 74, 75, 77, 78, 79
]

# Known object heights (meters) for distance estimation
KNOWN_OBJECT_HEIGHTS = {
    0: 1.7,  # person
    1: 1.0,  # bicycle
    2: 1.5,  # car
    3: 1.2,  # motorcycle
    5: 3.0,  # bus
    7: 3.5   # truck
}

DEFAULT_OBJECT_HEIGHT = 2.0  # meters

# Alert Distance Thresholds
ALERT_DISTANCE_PERSON = 2.0   # meters
ALERT_DISTANCE_OBJECT = 12.0  # meters

# Motion Detection
HISTORY_FRAMES = 15
MOVEMENT_THRESHOLD_PIXELS = 5
DIRECTION_THRESHOLD_RATIO = 0.3

# Alert Management
ALERT_CLASS_COOLDOWN_SEC = 8.0
ALERT_REPEAT_DELAY_SEC = 15.0
GLOBAL_ALERT_COOLDOWN = 3.0

# Audio Configuration
TTS_LANGUAGE = 'en'
TTS_SLOW_SPEECH = False
AUDIO_CACHE_SIZE = 50
PARALLEL_TTS_WORKERS = 4
ENABLE_LOCAL_AUDIO = False
CONTINUOUS_AUDIO_MODE = True
CONTINUOUS_AUDIO_INTERVAL = 2.0
NARRATION_MAX_OBJECTS = 3

# Server Configuration
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8000

# Logging Configuration
ENABLE_PERFORMANCE_LOGGING = True
PERFORMANCE_LOG_FILE = 'performance_log.csv'
LOG_BATCH_SIZE = 10

# WebSocket Streaming Configuration
STREAMING_FPS = 30
STREAMING_FRAME_DELAY = 1.0 / STREAMING_FPS
MAX_WEBSOCKET_CLIENTS = 5
WEBSOCKET_SEND_TIMEOUT = 1.0

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AlertEvent:
    """Represents an alert event"""
    timestamp: float
    track_id: int
    class_id: int
    class_name: str
    distance: float
    direction: str
    motion: str
    alert_text: str

@dataclass
class PerformanceMetrics:
    """Represents performance metrics for a frame"""
    timestamp: float
    frame_count: int
    yolo_time: float
    distance_calc_time: float
    direction_motion_time: float
    alert_processing_time: float
    audio_generation_time: float
    annotation_time: float
    total_frame_time: float
    detections_count: int
    alerts_count: int
    
    def to_dict(self):
        """Convert to dictionary for CSV logging"""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'frame_count': self.frame_count,
            'yolo_time_ms': round(self.yolo_time * 1000, 2),
            'distance_calc_time_ms': round(self.distance_calc_time * 1000, 2),
            'direction_motion_time_ms': round(self.direction_motion_time * 1000, 2),
            'alert_processing_time_ms': round(self.alert_processing_time * 1000, 2),
            'audio_generation_time_ms': round(self.audio_generation_time * 1000, 2),
            'annotation_time_ms': round(self.annotation_time * 1000, 2),
            'total_frame_time_ms': round(self.total_frame_time * 1000, 2),
            'fps': round(1.0 / self.total_frame_time if self.total_frame_time > 0 else 0, 2),
            'detections_count': self.detections_count,
            'alerts_count': self.alerts_count
        }

# ============================================================================
# PERFORMANCE LOGGING FUNCTIONS
# ============================================================================

class PerformanceLogger:
    """Manages performance logging to CSV file"""
    
    def __init__(self, log_file: str = PERFORMANCE_LOG_FILE):
        self.log_file = log_file
        self.metrics_buffer = []
        self.log_lock = threading.Lock()
        self.initialize_log_file()
    
    def initialize_log_file(self):
        """Initialize CSV log file with headers"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'datetime', 'frame_count',
                    'yolo_time_ms', 'distance_calc_time_ms', 'direction_motion_time_ms',
                    'alert_processing_time_ms', 'audio_generation_time_ms',
                    'annotation_time_ms', 'total_frame_time_ms', 'fps',
                    'detections_count', 'alerts_count'
                ])
                writer.writeheader()
            print(f"📊 Performance log initialized: {self.log_file}")
    
    def log_metrics(self, metrics: PerformanceMetrics):
        """Add metrics to buffer and write if batch size reached"""
        with self.log_lock:
            self.metrics_buffer.append(metrics.to_dict())
            
            if len(self.metrics_buffer) >= LOG_BATCH_SIZE:
                self.flush_buffer()
    
    def flush_buffer(self):
        """Write buffered metrics to file"""
        if not self.metrics_buffer:
            return
        
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.metrics_buffer[0].keys())
                writer.writerows(self.metrics_buffer)
            
            print(f"📝 Logged {len(self.metrics_buffer)} performance records")
            self.metrics_buffer.clear()
        except Exception as e:
            print(f"⚠️ Failed to write performance log: {e}")
    
    def get_summary_stats(self, last_n: int = 100):
        """Get summary statistics from recent logs"""
        try:
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)[-last_n:]
            
            if not rows:
                return None
            
            stats = {
                'total_frames': len(rows),
                'avg_yolo_ms': round(np.mean([float(r['yolo_time_ms']) for r in rows]), 2),
                'avg_distance_ms': round(np.mean([float(r['distance_calc_time_ms']) for r in rows]), 2),
                'avg_direction_ms': round(np.mean([float(r['direction_motion_time_ms']) for r in rows]), 2),
                'avg_alert_ms': round(np.mean([float(r['alert_processing_time_ms']) for r in rows]), 2),
                'avg_audio_ms': round(np.mean([float(r['audio_generation_time_ms']) for r in rows]), 2),
                'avg_annotation_ms': round(np.mean([float(r['annotation_time_ms']) for r in rows]), 2),
                'avg_total_ms': round(np.mean([float(r['total_frame_time_ms']) for r in rows]), 2),
                'avg_fps': round(np.mean([float(r['fps']) for r in rows]), 2),
                'avg_detections': round(np.mean([int(r['detections_count']) for r in rows]), 2),
                'avg_alerts': round(np.mean([int(r['alerts_count']) for r in rows]), 2)
            }
            
            return stats
        except Exception as e:
            print(f"⚠️ Failed to get summary stats: {e}")
            return None

# Global performance logger
performance_logger = None

# ============================================================================
# DETECTION ENGINE FUNCTIONS
# ============================================================================

def get_device() -> str:
    """Detect and return the best available device"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    return device

def load_yolo_model(model_path: str = YOLO_MODEL_PATH):
    """Load YOLO model and move to device"""
    try:
        print(f"Loading YOLO model from: {model_path}")
        device = get_device()
        model = YOLO(model_path)
        model.to(device)
        print("✅ YOLO model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        raise

def detect_and_track(model, frame: np.ndarray, conf_threshold: float = CONFIDENCE_THRESHOLD) -> List[Dict]:
    """Run YOLO detection and tracking on a frame"""
    if model is None:
        raise RuntimeError("Model not loaded")

    results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)
    detections = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            track_id = int(track_id)
            class_id = int(class_id)

            if class_id in NOISY_CLASSES_TO_IGNORE:
                continue

            class_name = model.names.get(class_id, f'Class{class_id}')
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            detections.append({
                'box': box.tolist(),
                'class_id': class_id,
                'track_id': track_id,
                'class_name': class_name,
                'center': (cx, cy)
            })

    return detections

def estimate_distance(box: List[float], class_id: int) -> float:
    """Estimate distance to object based on bounding box height"""
    x1, y1, x2, y2 = box
    box_height_pixels = y2 - y1

    if box_height_pixels <= 0:
        return float('inf')

    object_real_height = KNOWN_OBJECT_HEIGHTS.get(class_id, DEFAULT_OBJECT_HEIGHT)
    estimated_distance = (object_real_height * FOCAL_LENGTH_PIXELS) / box_height_pixels

    return estimated_distance

def get_direction_motion(track_id: int, frame_width: int, track_histories: Dict) -> Tuple[str, str]:
    """Analyze object direction and motion based on tracking history"""
    direction_str = "Ahead"
    motion_str = "Static"

    if (track_id in track_histories and len(track_histories[track_id]) == HISTORY_FRAMES):
        history = track_histories[track_id]
        oldest_pos, newest_pos = history[0], history[-1]

        dx = newest_pos[0] - oldest_pos[0]
        dy = newest_pos[1] - oldest_pos[1]

        if (abs(dx) > MOVEMENT_THRESHOLD_PIXELS or abs(dy) > MOVEMENT_THRESHOLD_PIXELS):
            motion_str = "Moving"

        frame_center_x = frame_width / 2
        relative_pos = (newest_pos[0] - frame_center_x) / frame_center_x

        if relative_pos > DIRECTION_THRESHOLD_RATIO:
            direction_str = "Right"
        elif relative_pos < -DIRECTION_THRESHOLD_RATIO:
            direction_str = "Left"

    return direction_str, motion_str

def annotate_frame(frame: np.ndarray, detections: List[Dict], frame_width: int, track_histories: Dict) -> np.ndarray:
    """Annotate frame with detection information"""
    annotated_frame = frame.copy()

    for detection in detections:
        box = detection['box']
        track_id = detection['track_id']
        class_name = detection['class_name']
        class_id = detection['class_id']

        x1, y1, x2, y2 = map(int, box)
        distance = estimate_distance(box, class_id)
        direction, motion = get_direction_motion(track_id, frame_width, track_histories)

        info_text = f"ID:{track_id} {class_name} {distance:.1f}m {direction}"
        if motion == "Moving":
            info_text += f" {motion}"

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, info_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return annotated_frame

# ============================================================================
# AUDIO ENGINE FUNCTIONS
# ============================================================================

def generate_tts_audio(text: str, audio_cache: Dict) -> Optional[AudioSegment]:
    """Generate TTS audio for given text with caching"""
    if text in audio_cache:
        return audio_cache[text]

    try:
        tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=TTS_SLOW_SPEECH)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_segment = AudioSegment.from_file(audio_buffer, format="mp3")

        if len(audio_cache) >= AUDIO_CACHE_SIZE:
            oldest_key = next(iter(audio_cache))
            del audio_cache[oldest_key]

        audio_cache[text] = audio_segment
        return audio_segment

    except Exception as e:
        print(f"⚠️ TTS generation failed for '{text}': {e}")
        return None

def export_audio_bytes(audio_segment: AudioSegment, format: str = "mp3") -> bytes:
    """Export AudioSegment to bytes"""
    buffer = io.BytesIO()
    audio_segment.export(buffer, format=format)
    return buffer.getvalue()

def play_audio_locally(audio_segment: AudioSegment):
    """Play audio locally on the server (requires audio output)"""
    try:
        from pydub.playback import play
        play(audio_segment)
    except Exception as e:
        print(f"⚠️ Local audio playback failed: {e}")

# ============================================================================
# ALERT MANAGER FUNCTIONS
# ============================================================================

def should_alert(detection: Dict, current_time: float, frame_width: int, track_histories: Dict, cooldown_state: Dict) -> Optional[AlertEvent]:
    """Determine if an alert should be generated for a detection"""
    class_id = int(detection['class_id'])
    
    # CRITICAL: Skip all humans completely
    if class_id == 0:
        return None
    
    track_id = detection['track_id']
    class_name = detection['class_name']
    box = detection['box']

    distance = estimate_distance(box, class_id)
    direction, motion = get_direction_motion(track_id, frame_width, track_histories)

    is_close_object = (distance < ALERT_DISTANCE_OBJECT)

    if not is_close_object:
        return None

    if not is_cooldown_ready(track_id, class_id, current_time, cooldown_state):
        return None

    alert_text = generate_alert_text(class_name, distance, direction, motion)

    alert_event = AlertEvent(
        timestamp=current_time,
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        distance=distance,
        direction=direction,
        motion=motion,
        alert_text=alert_text
    )

    update_cooldowns(track_id, class_id, current_time, cooldown_state)
    return alert_event

def is_cooldown_ready(track_id: int, class_id: int, current_time: float, cooldown_state: Dict) -> bool:
    """Check if all cooldown conditions are met"""
    last_track_alert = cooldown_state['alerted_tracks'].get(track_id, -float('inf'))
    if (current_time - last_track_alert) < ALERT_REPEAT_DELAY_SEC:
        return False

    last_class_alert = cooldown_state['last_alert_time_by_class'].get(class_id, -float('inf'))
    if (current_time - last_class_alert) < ALERT_CLASS_COOLDOWN_SEC:
        return False

    if (current_time - cooldown_state['last_global_alert_time']) < GLOBAL_ALERT_COOLDOWN:
        return False

    return True

def update_cooldowns(track_id: int, class_id: int, current_time: float, cooldown_state: Dict):
    """Update cooldown tracking"""
    cooldown_state['alerted_tracks'][track_id] = current_time
    cooldown_state['last_alert_time_by_class'][class_id] = current_time

def generate_alert_text(class_name: str, distance: float, direction: str, motion: str) -> str:
    """Generate contextual alert text"""
    dist_str = f"{distance:.0f}"

    if motion == "Moving":
        alert_base = f"{class_name} moving, about {dist_str} meters, {direction}."
    else:
        alert_base = f"{class_name}, about {dist_str} meters, {direction}."

    alert_text = f"Caution: {alert_base}"
    return alert_text

def process_detections(detections: List[Dict], current_time: float, frame_width: int, track_histories: Dict, cooldown_state: Dict) -> List[AlertEvent]:
    """Process all detections and generate alerts"""
    potential_alerts = []

    for detection in detections:
        alert_event = should_alert(detection, current_time, frame_width, track_histories, cooldown_state)
        if alert_event:
            potential_alerts.append(alert_event)

    if not potential_alerts:
        return []

    potential_alerts = [p for p in potential_alerts if p.class_id != 0]

    if not potential_alerts:
        return []

    potential_alerts.sort(key=lambda x: x.distance)
    highest_priority = potential_alerts[0]

    if (current_time - cooldown_state['last_global_alert_time']) >= GLOBAL_ALERT_COOLDOWN:
        cooldown_state['last_global_alert_time'] = current_time
        return [highest_priority]

    return []

def generate_continuous_narration_from_detections(detections: List[Dict], frame_width: int, track_histories: Dict) -> str:
    """Generate continuous narration from detection data format (excluding humans)"""
    if not detections:
        return "Area clear."

    detection_info = []
    for detection in detections:
        if int(detection['class_id']) == 0:
            continue
            
        bbox = detection.get('bbox', detection.get('box', [0, 0, 100, 100]))
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            continue
            
        distance = estimate_distance([x1, y1, x2, y2], detection['class_id'])
        track_id = detection['track_id']
        direction = get_direction_from_position(detection['center'][0], frame_width)
        motion = get_motion_from_history(track_id, track_histories)

        detection_info.append({
            'name': detection['class_name'],
            'distance': distance,
            'direction': direction,
            'motion': motion
        })

    if not detection_info:
        return "Area clear."

    detection_info.sort(key=lambda x: x['distance'])
    top_objects = detection_info[:NARRATION_MAX_OBJECTS]

    if len(top_objects) == 1:
        obj = top_objects[0]
        if obj['motion'] == "Moving":
            return f"{obj['name']} moving {obj['direction']}, {obj['distance']:.0f} meters"
        else:
            return f"{obj['name']} {obj['direction']}, {obj['distance']:.0f} meters"

    elif len(top_objects) == 2:
        obj1, obj2 = top_objects
        part1 = f"{obj1['name']} {obj1['direction']}, {obj1['distance']:.0f} meters"
        part2 = f"{obj2['name']} {obj2['direction']}, {obj2['distance']:.0f} meters"
        return f"{part1}. {part2}"

    else:
        closest = top_objects[0]
        if closest['motion'] == "Moving":
            main_part = f"{closest['name']} moving {closest['direction']}, {closest['distance']:.0f} meters"
        else:
            main_part = f"{closest['name']} {closest['direction']}, {closest['distance']:.0f} meters"

        others = top_objects[1:]
        directions = {}
        for obj in others:
            dir_key = obj['direction']
            if dir_key not in directions:
                directions[dir_key] = 0
            directions[dir_key] += 1

        if directions:
            other_parts = []
            for direction, count in directions.items():
                if count == 1:
                    other_parts.append(f"one {direction.lower()}")
                else:
                    other_parts.append(f"{count} {direction.lower()}")

            other_summary = ", ".join(other_parts)
            return f"{main_part}. Others: {other_summary}"
        else:
            return main_part

def get_direction_from_position(center_x: float, frame_width: int) -> str:
    """Get direction based on position in frame"""
    left_zone = frame_width * 0.33
    right_zone = frame_width * 0.67

    if center_x < left_zone:
        return "Left"
    elif center_x > right_zone:
        return "Right"
    else:
        return "Center"

def get_motion_from_history(track_id: int, track_histories: Dict) -> str:
    """Determine if object is moving based on tracking history"""
    if track_id not in track_histories:
        return "Stationary"

    history = list(track_histories[track_id])
    if len(history) < 5:
        return "Stationary"

    recent_positions = history[-5:]
    x_positions = [pos[0] for pos in recent_positions]
    x_range = max(x_positions) - min(x_positions)
    movement_threshold = 30

    return "Moving" if x_range > movement_threshold else "Stationary"

def should_provide_continuous_audio(current_time: float, pipeline_state: Dict) -> bool:
    """Determine if continuous audio should be provided"""
    if not CONTINUOUS_AUDIO_MODE:
        return False

    last_audio_time = pipeline_state.get('last_continuous_audio_time', -float('inf'))
    return (current_time - last_audio_time) >= CONTINUOUS_AUDIO_INTERVAL

def cleanup_inactive_tracks(active_track_ids: Set[int], track_histories: Dict, cooldown_state: Dict):
    """Remove tracking data for inactive tracks"""
    inactive_tracks = set(track_histories.keys()) - active_track_ids
    for track_id in inactive_tracks:
        del track_histories[track_id]
        if track_id in cooldown_state['alerted_tracks']:
            del cooldown_state['alerted_tracks'][track_id]

# ============================================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================================

def create_pipeline_state():
    """Create initial pipeline state"""
    return {
        'model': None,
        'track_histories': {},
        'audio_cache': {},
        'cooldown_state': {
            'alerted_tracks': {},
            'last_alert_time_by_class': {},
            'last_global_alert_time': -float('inf')
        },
        'alert_log': [],
        'frame_count': 0,
        'last_continuous_audio_time': -float('inf')
    }

def initialize_pipeline(model_path: str = YOLO_MODEL_PATH):
    """Initialize the vision assistance pipeline"""
    print("🚀 Initializing Vision Assistance Pipeline...")
    state = create_pipeline_state()
    state['model'] = load_yolo_model(model_path)
    print("✅ Vision Assistance Pipeline initialized successfully")
    return state

def process_frame(frame: np.ndarray, pipeline_state: Dict, timestamp: float = None) -> Tuple[np.ndarray, List[bytes], List[str]]:
    """Process a single frame and generate alerts"""
    frame_start_time = time.perf_counter()
    
    if timestamp is None:
        timestamp = time.time()

    pipeline_state['frame_count'] += 1

    # Step 1: Object detection and tracking
    yolo_start = time.perf_counter()
    detections = detect_and_track(pipeline_state['model'], frame)
    yolo_time = time.perf_counter() - yolo_start

    # Step 2: Update tracking histories
    active_track_ids = set()
    for detection in detections:
        track_id = detection['track_id']
        cx, cy = detection['center']
        active_track_ids.add(track_id)

        if track_id not in pipeline_state['track_histories']:
            pipeline_state['track_histories'][track_id] = deque(maxlen=HISTORY_FRAMES)
        pipeline_state['track_histories'][track_id].append((cx, cy))

    # Step 2.5: Calculate distances
    distance_start = time.perf_counter()
    for detection in detections:
        detection['distance'] = estimate_distance(detection['box'], detection['class_id'])
    distance_calc_time = time.perf_counter() - distance_start

    # Step 2.6: Calculate direction and motion
    direction_start = time.perf_counter()
    for detection in detections:
        track_id = detection['track_id']
        direction, motion = get_direction_motion(track_id, frame.shape[1], pipeline_state['track_histories'])
        detection['direction'] = direction
        detection['motion'] = motion
    direction_motion_time = time.perf_counter() - direction_start

    # Step 3: Generate alerts
    alert_start = time.perf_counter()
    frame_alerts = process_detections(detections, timestamp, frame.shape[1], pipeline_state['track_histories'], pipeline_state['cooldown_state'])
    alert_processing_time = time.perf_counter() - alert_start

    # Step 4: Generate audio
    audio_start = time.perf_counter()
    audio_data_list = []
    alert_texts = []
    
    for alert in frame_alerts:
        audio_segment = generate_tts_audio(alert.alert_text, pipeline_state['audio_cache'])
        if audio_segment:
            if ENABLE_LOCAL_AUDIO:
                threading.Thread(target=play_audio_locally, args=(audio_segment,), daemon=True).start()

            audio_bytes = export_audio_bytes(audio_segment)
            audio_data_list.append(audio_bytes)
            alert_texts.append(alert.alert_text)
            print(f"🔊 Alert: {alert.alert_text}")

        pipeline_state['alert_log'].append(alert)

    # Step 4.5: Continuous narration
    if (not frame_alerts and CONTINUOUS_AUDIO_MODE and should_provide_continuous_audio(timestamp, pipeline_state)):
        narration_text = generate_continuous_narration_from_detections(detections, frame.shape[1], pipeline_state['track_histories'])

        if narration_text:
            narration_audio = generate_tts_audio(narration_text, pipeline_state['audio_cache'])
            if narration_audio:
                if ENABLE_LOCAL_AUDIO:
                    threading.Thread(target=play_audio_locally, args=(narration_audio,), daemon=True).start()

                audio_bytes = export_audio_bytes(narration_audio)
                audio_data_list.append(audio_bytes)
                alert_texts.append(narration_text)
                print(f"🗣️  Narration: {narration_text}")

        pipeline_state['last_continuous_audio_time'] = timestamp
    
    audio_generation_time = time.perf_counter() - audio_start

    # Step 5: Annotate frame
    annotation_start = time.perf_counter()
    annotated_frame = annotate_frame(frame, detections, frame.shape[1], pipeline_state['track_histories'])
    annotation_time = time.perf_counter() - annotation_start

    # Step 6: Cleanup inactive tracks
    cleanup_inactive_tracks(active_track_ids, pipeline_state['track_histories'], pipeline_state['cooldown_state'])

    # Calculate total frame time
    total_frame_time = time.perf_counter() - frame_start_time

    # Log performance metrics
    if ENABLE_PERFORMANCE_LOGGING and performance_logger:
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            frame_count=pipeline_state['frame_count'],
            yolo_time=yolo_time,
            distance_calc_time=distance_calc_time,
            direction_motion_time=direction_motion_time,
            alert_processing_time=alert_processing_time,
            audio_generation_time=audio_generation_time,
            annotation_time=annotation_time,
            total_frame_time=total_frame_time,
            detections_count=len(detections),
            alerts_count=len(audio_data_list)
        )
        performance_logger.log_metrics(metrics)

    return annotated_frame, audio_data_list, alert_texts

# ============================================================================
# CAMERA INPUT FUNCTIONS (Browser-based via WebSocket)
# ============================================================================

# Camera functionality moved to browser-based WebSocket streaming
# No server-side camera access needed

# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time streaming"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.streaming_active = False
        self.streaming_task = None

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        if len(self.active_connections) >= MAX_WEBSOCKET_CLIENTS:
            await websocket.send_json({
                "error": "Maximum number of clients reached",
                "max_clients": MAX_WEBSOCKET_CLIENTS
            })
            await websocket.close()
            return False

        self.active_connections.append(websocket)
        print(f"🔗 WebSocket client connected. Total clients: {len(self.active_connections)}")
        return True

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"🔗 WebSocket client disconnected. Total clients: {len(self.active_connections)}")

        if not self.active_connections:
            self.stop_streaming()

    async def send_to_client(self, websocket: WebSocket, message: dict):
        """Send message to specific client with timeout"""
        try:
            await asyncio.wait_for(websocket.send_json(message), timeout=WEBSOCKET_SEND_TIMEOUT)
            return True
        except (asyncio.TimeoutError, Exception) as e:
            print(f"⚠️ Failed to send to client: {e}")
            self.disconnect(websocket)
        return False

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        disconnected = []

        for websocket in self.active_connections[:]:
            success = await self.send_to_client(websocket, message)
            if not success:
                disconnected.append(websocket)

        for websocket in disconnected:
            self.disconnect(websocket)

    def start_streaming(self):
        """Start continuous streaming task"""
        if not self.streaming_active:
            self.streaming_active = True
            self.streaming_task = asyncio.create_task(self.continuous_streaming())
            print("🎬 Started continuous WebSocket streaming")

    def stop_streaming(self):
        """Stop continuous streaming task"""
        if self.streaming_active:
            self.streaming_active = False
            if self.streaming_task:
                self.streaming_task.cancel()
            print("🎬 Stopped continuous WebSocket streaming")

    async def continuous_streaming(self):
        """Placeholder for future server-side streaming if needed"""
        print("🎥 Browser-based streaming active - no server-side streaming needed")
        
        # Keep connection alive but don't do continuous streaming
        while self.streaming_active and self.active_connections:
            await asyncio.sleep(1)

        print("🎥 Continuous streaming loop ended")

connection_manager = ConnectionManager()

# ============================================================================
# HTTP SERVER
# ============================================================================

pipeline_state = None
performance_logger = None

app = FastAPI(title="Vision Assistance API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline_state, performance_logger
    pipeline_state = initialize_pipeline()
    
    if ENABLE_PERFORMANCE_LOGGING:
        performance_logger = PerformanceLogger()
    
    print("🔗 WebSocket connection manager initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global connection_manager, performance_logger
    print("🛑 Server shutting down...")

    if performance_logger:
        performance_logger.flush_buffer()
        print("📊 Performance logs flushed")

    if connection_manager:
        connection_manager.stop_streaming()





@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline_state is not None,
        "model_loaded": pipeline_state['model'] is not None if pipeline_state else False
    }



@app.websocket("/ws/camera/stream")
async def websocket_camera_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time camera streaming"""
    global connection_manager

    connected = await connection_manager.connect(websocket)
    if not connected:
        return

    try:
        await connection_manager.send_to_client(websocket, {
            "type": "connected",
            "message": "WebSocket connected successfully",
            "timestamp": time.time(),
            "client_count": len(connection_manager.active_connections)
        })

        while True:
            try:
                data = await websocket.receive_json()

                if data.get("command") == "process_frame":
                    # Handle frame processing from browser camera
                    frame_data = data.get("frame_data")
                    if frame_data:
                        try:
                            # Decode base64 frame data
                            frame_bytes = base64.b64decode(frame_data)
                            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                # Process the frame
                                annotated_frame, audio_data_list, alert_texts = process_frame(frame, pipeline_state)
                                
                                # Encode annotated frame
                                _, buffer = cv2.imencode('.jpg', annotated_frame)
                                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                                
                                # Encode audio data
                                audio_data = []
                                for audio_bytes in audio_data_list:
                                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                                    audio_data.append(audio_b64)
                                
                                # Send processing result
                                await connection_manager.send_to_client(websocket, {
                                    "type": "processing_result",
                                    "frame_data": frame_b64,
                                    "audio_data": audio_data,
                                    "alert_texts": alert_texts,
                                    "frame_count": pipeline_state['frame_count'],
                                    "detections_count": len(detect_and_track(pipeline_state['model'], frame)),
                                    "alerts_count": len(audio_data_list),
                                    "timestamp": time.time()
                                })
                        except Exception as e:
                            await connection_manager.send_to_client(websocket, {
                                "type": "error",
                                "message": f"Frame processing error: {str(e)}",
                                "timestamp": time.time()
                            })

            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"⚠️ Error handling WebSocket message: {e}")
                break

    except WebSocketDisconnect:
        print("🔗 WebSocket client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        connection_manager.disconnect(websocket)

@app.get("/ws/status")
async def websocket_status():
    """Get WebSocket streaming status"""
    return {
        "success": True,
        "connected_clients": len(connection_manager.active_connections),
        "max_clients": MAX_WEBSOCKET_CLIENTS,
        "streaming_active": connection_manager.streaming_active
    }

@app.get("/performance/stats")
async def get_performance_stats():
    """Get performance statistics from logs"""
    if not performance_logger:
        raise HTTPException(status_code=503, detail="Performance logging not enabled")
    
    stats = performance_logger.get_summary_stats(last_n=100)
    if stats is None:
        raise HTTPException(status_code=404, detail="No performance data available")
    
    return {"success": True, "data": stats}

@app.post("/performance/flush")
async def flush_performance_logs():
    """Manually flush performance logs to file"""
    if not performance_logger:
        raise HTTPException(status_code=503, detail="Performance logging not enabled")
    
    performance_logger.flush_buffer()
    return {"success": True, "message": "Performance logs flushed"}

@app.get("/performance/download")
async def download_performance_logs():
    """Download performance log CSV file"""
    if not performance_logger:
        raise HTTPException(status_code=503, detail="Performance logging not enabled")
    
    if not os.path.exists(PERFORMANCE_LOG_FILE):
        raise HTTPException(status_code=404, detail="Performance log file not found")
    
    performance_logger.flush_buffer()
    
    def iterfile():
        with open(PERFORMANCE_LOG_FILE, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(
        iterfile(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={PERFORMANCE_LOG_FILE}"
        }
    )

@app.get("/")
def index():
    """Serve HTML client interface"""
    try:
        return HTMLResponse(open("websocket_camera_client.html", "r", encoding="utf-8").read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Vision Assistance API</h1><p>WebSocket client HTML file not found.</p>")

# ============================================================================
# MAIN SERVER ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the server"""
    print("🚀 Vision Assistance HTTP Server")
    print("=" * 50)
    print(f"Server will start on http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"YOLO Model Path: {YOLO_MODEL_PATH}")
    print(f"Performance Logging: {'Enabled' if ENABLE_PERFORMANCE_LOGGING else 'Disabled'}")
    if ENABLE_PERFORMANCE_LOGGING:
        print(f"Log File: {PERFORMANCE_LOG_FILE}")
    print("=" * 50)

    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info"
    )

if __name__ == "__main__":
    main()