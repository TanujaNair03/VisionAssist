"""
Vision Assistance System - Single File Modular Implementation
Real-time object detection with audio alerts via HTTP API

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
    # Accessories
    24, # backpack
    25, # umbrella
    26, # handbag
    27, # tie
    # Sports
    31, # snowboard
    34, # baseball bat
    35, # baseball glove
    36, # skateboard
    37, # surfboard
    38, # tennis racket
    # Kitchen & Food
    39, # bottle
    40, # wine glass
    41, # cup
    44, # spoon
    45, # bowl
    46, # banana
    47, # apple
    48, # sandwich
    49, # orange
    50, # broccoli
    51, # carrot
    52, # hot dog
    53, # pizza
    54, # donut
    55, # cake
    # Other
    73, # book
    74, # clock
    75, # vase
    77, # teddy bear
    78, # hair drier
    79, # toothbrush
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
ENABLE_LOCAL_AUDIO = False  # Set to True to play audio on server instead of sending to client
CONTINUOUS_AUDIO_MODE = True  # Provide ongoing narration of surroundings
CONTINUOUS_AUDIO_INTERVAL = 2.0  # Seconds between continuous updates
NARRATION_MAX_OBJECTS = 3  # Maximum objects to mention in narration

# Server Configuration
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8000

# WebSocket Streaming Configuration
STREAMING_FPS = 30
STREAMING_FRAME_DELAY = 1.0 / STREAMING_FPS  # seconds between frames
MAX_WEBSOCKET_CLIENTS = 5
WEBSOCKET_SEND_TIMEOUT = 1.0  # seconds

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
    
    # Run YOLO inference with tracking
    results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)
    
    detections = []
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        
        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            track_id = int(track_id)
            class_id = int(class_id)
            
            # Skip ignored classes
            if class_id in NOISY_CLASSES_TO_IGNORE:
                continue
            
            # Get class name
            class_name = model.names.get(class_id, f'Class{class_id}')
            
            # Calculate center point
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
    
    # Get known height for this class
    object_real_height = KNOWN_OBJECT_HEIGHTS.get(class_id, DEFAULT_OBJECT_HEIGHT)
    
    # Distance = (real_height * focal_length) / pixel_height
    estimated_distance = (object_real_height * FOCAL_LENGTH_PIXELS) / box_height_pixels
    
    return estimated_distance

def get_direction_motion(track_id: int, frame_width: int, track_histories: Dict) -> Tuple[str, str]:
    """Analyze object direction and motion based on tracking history"""
    direction_str = "Ahead"
    motion_str = "Static"
    
    if (track_id in track_histories and 
        len(track_histories[track_id]) == HISTORY_FRAMES):
        
        history = track_histories[track_id]
        oldest_pos, newest_pos = history[0], history[-1]
        
        # Calculate movement
        dx = newest_pos[0] - oldest_pos[0]
        dy = newest_pos[1] - oldest_pos[1]
        
        if (abs(dx) > MOVEMENT_THRESHOLD_PIXELS or 
            abs(dy) > MOVEMENT_THRESHOLD_PIXELS):
            motion_str = "Moving"
        
        # Determine direction relative to frame center
        frame_center_x = frame_width / 2
        relative_pos = (newest_pos[0] - frame_center_x) / frame_center_x
        
        if relative_pos > DIRECTION_THRESHOLD_RATIO:
            direction_str = "Right"
        elif relative_pos < -DIRECTION_THRESHOLD_RATIO:
            direction_str = "Left"
    
    return direction_str, motion_str

def annotate_frame(frame: np.ndarray, detections: List[Dict], 
                   frame_width: int, track_histories: Dict) -> np.ndarray:
    """Annotate frame with detection information"""
    annotated_frame = frame.copy()
    
    for detection in detections:
        box = detection['box']
        track_id = detection['track_id']
        class_name = detection['class_name']
        class_id = detection['class_id']
        
        x1, y1, x2, y2 = map(int, box)
        
        # Estimate distance
        distance = estimate_distance(box, class_id)
        
        # Get direction and motion
        direction, motion = get_direction_motion(track_id, frame_width, track_histories)
        
        # Create info text
        info_text = f"ID:{track_id} {class_name} {distance:.1f}m {direction}"
        if motion == "Moving":
            info_text += f" {motion}"
        
        # Draw bounding box and text
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, info_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_frame

# ============================================================================
# AUDIO ENGINE FUNCTIONS
# ============================================================================

def generate_tts_audio(text: str, audio_cache: Dict) -> Optional[AudioSegment]:
    """Generate TTS audio for given text with caching"""
    # Check cache first
    if text in audio_cache:
        return audio_cache[text]
    
    try:
        # Generate TTS
        tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=TTS_SLOW_SPEECH)
        
        # Use BytesIO for better performance
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Convert to AudioSegment
        audio_segment = AudioSegment.from_file(audio_buffer, format="mp3")
        
        # Cache the result with size management
        if len(audio_cache) >= AUDIO_CACHE_SIZE:
            # Remove oldest item (simple FIFO)
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
        # This requires the server to have audio output capabilities
        from pydub.playback import play
        play(audio_segment)
    except Exception as e:
        print(f"⚠️ Local audio playback failed: {e}")
        print("Make sure the server has audio output and required audio libraries")

# ============================================================================
# ALERT MANAGER FUNCTIONS
# ============================================================================

def should_alert(detection: Dict, current_time: float, frame_width: int,
                track_histories: Dict, cooldown_state: Dict) -> Optional[AlertEvent]:
    """Determine if an alert should be generated for a detection"""
    class_id = detection['class_id']
    track_id = detection['track_id']
    class_name = detection['class_name']
    box = detection['box']
    
    # Calculate distance
    distance = estimate_distance(box, class_id)
    
    # Get direction and motion
    direction, motion = get_direction_motion(track_id, frame_width, track_histories)
    
    # Check if object is within alert distance
    is_person = (class_id == 0)
    is_close_person = is_person and (distance < ALERT_DISTANCE_PERSON)
    is_close_object = (not is_person) and (distance < ALERT_DISTANCE_OBJECT)
    
    if not (is_close_person or is_close_object):
        return None
    
    # Check cooldown conditions
    if not is_cooldown_ready(track_id, class_id, current_time, cooldown_state):
        return None
    
    # Generate alert text
    alert_text = generate_alert_text(class_name, distance, direction, motion)
    
    # Create alert event
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
    
    # Update cooldown tracking
    update_cooldowns(track_id, class_id, current_time, cooldown_state)
    
    return alert_event

def is_cooldown_ready(track_id: int, class_id: int, current_time: float, 
                     cooldown_state: Dict) -> bool:
    """Check if all cooldown conditions are met"""
    
    # Check per-track cooldown
    last_track_alert = cooldown_state['alerted_tracks'].get(track_id, -float('inf'))
    if (current_time - last_track_alert) < ALERT_REPEAT_DELAY_SEC:
        return False
    
    # Check per-class cooldown
    last_class_alert = cooldown_state['last_alert_time_by_class'].get(class_id, -float('inf'))
    if (current_time - last_class_alert) < ALERT_CLASS_COOLDOWN_SEC:
        return False
    
    # Check global cooldown
    if (current_time - cooldown_state['last_global_alert_time']) < GLOBAL_ALERT_COOLDOWN:
        return False
    
    return True

def update_cooldowns(track_id: int, class_id: int, current_time: float, 
                    cooldown_state: Dict):
    """Update cooldown tracking"""
    cooldown_state['alerted_tracks'][track_id] = current_time
    cooldown_state['last_alert_time_by_class'][class_id] = current_time

def generate_alert_text(class_name: str, distance: float, 
                       direction: str, motion: str) -> str:
    """Generate contextual alert text"""
    dist_str = f"{distance:.0f}"
    
    # Base alert with object, distance, and direction
    if motion == "Moving":
        alert_base = f"{class_name} moving, about {dist_str} meters, {direction}."
    else:
        alert_base = f"{class_name}, about {dist_str} meters, {direction}."
    
    # Add caution prefix
    alert_text = f"Caution: {alert_base}"
    
    return alert_text

def process_detections(detections: List[Dict], current_time: float,
                      frame_width: int, track_histories: Dict, 
                      cooldown_state: Dict) -> List[AlertEvent]:
    """Process all detections and generate alerts"""
    potential_alerts = []
    
    # Find all potential alerts
    for detection in detections:
        alert_event = should_alert(detection, current_time, frame_width, 
                                  track_histories, cooldown_state)
        if alert_event:
            potential_alerts.append(alert_event)
    
    if not potential_alerts:
        return []
    
    # Prioritize alerts: persons first, then by distance
    potential_alerts.sort(key=lambda x: (0 if x.class_id == 0 else 1, x.distance))
    
    # Take the highest priority alert
    highest_priority = potential_alerts[0]
    
    # Final global cooldown check
    if (current_time - cooldown_state['last_global_alert_time']) >= GLOBAL_ALERT_COOLDOWN:
        cooldown_state['last_global_alert_time'] = current_time
        return [highest_priority]
    
    return []

def generate_continuous_narration(detections: List[Dict], frame_width: int, 
                                track_histories: Dict) -> str:
    """Generate continuous narration of the current scene"""
    if not detections:
        return "Area clear."
    
    # Sort detections by distance (closest first)
    detection_info = []
    for detection in detections:
        distance = estimate_distance(detection['box'], detection['class_id'])
        direction, motion = get_direction_motion(detection['track_id'], frame_width, track_histories)
        
        detection_info.append({
            'name': detection['class_name'],
            'distance': distance,
            'direction': direction,
            'motion': motion
        })
    
    # Sort by distance, take closest objects
    detection_info.sort(key=lambda x: x['distance'])
    top_objects = detection_info[:NARRATION_MAX_OBJECTS]
    
    # Build narration
    narration_parts = []
    
    if len(top_objects) == 1:
        obj = top_objects[0]
        if obj['motion'] == "Moving":
            narration = f"{obj['name']} moving {obj['direction']}, {obj['distance']:.0f} meters"
        else:
            narration = f"{obj['name']} {obj['direction']}, {obj['distance']:.0f} meters"
        return narration
    
    elif len(top_objects) == 2:
        obj1, obj2 = top_objects
        part1 = f"{obj1['name']} {obj1['direction']}, {obj1['distance']:.0f} meters"
        part2 = f"{obj2['name']} {obj2['direction']}, {obj2['distance']:.0f} meters"
        return f"{part1}. {part2}"
    
    else:  # 3 or more objects
        # Mention closest object in detail, then summarize others
        closest = top_objects[0]
        if closest['motion'] == "Moving":
            main_part = f"{closest['name']} moving {closest['direction']}, {closest['distance']:.0f} meters"
        else:
            main_part = f"{closest['name']} {closest['direction']}, {closest['distance']:.0f} meters"
        
        # Count other objects by direction
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

def should_provide_continuous_audio(current_time: float, pipeline_state: Dict) -> bool:
    """Determine if continuous audio should be provided"""
    if not CONTINUOUS_AUDIO_MODE:
        return False
    
    last_audio_time = pipeline_state.get('last_continuous_audio_time', -float('inf'))
    return (current_time - last_audio_time) >= CONTINUOUS_AUDIO_INTERVAL

def generate_continuous_narration_from_detections(detections: List[Dict], frame_width: int, 
                                                track_histories: Dict) -> str:
    """Generate continuous narration from detection data format"""
    if not detections:
        return "Area clear."
    
    # Convert detections to format needed for narration
    detection_info = []
    for detection in detections:
        # Calculate distance using bounding box
        bbox = detection.get('bbox', [0, 0, 100, 100])
        x1, y1, x2, y2 = bbox
        distance = estimate_distance([x1, y1, x2, y2], detection['class_id'])
        
        # Get direction and motion
        track_id = detection['track_id']
        direction = get_direction_from_position(detection['center'][0], frame_width)
        motion = get_motion_from_history(track_id, track_histories)
        
        detection_info.append({
            'name': detection['class_name'],
            'distance': distance,
            'direction': direction,
            'motion': motion
        })
    
    # Sort by distance, take closest objects
    detection_info.sort(key=lambda x: x['distance'])
    top_objects = detection_info[:NARRATION_MAX_OBJECTS]
    
    # Build narration
    if len(top_objects) == 1:
        obj = top_objects[0]
        if obj['motion'] == "Moving":
            narration = f"{obj['name']} moving {obj['direction']}, {obj['distance']:.0f} meters"
        else:
            narration = f"{obj['name']} {obj['direction']}, {obj['distance']:.0f} meters"
        return narration
    
    elif len(top_objects) == 2:
        obj1, obj2 = top_objects
        part1 = f"{obj1['name']} {obj1['direction']}, {obj1['distance']:.0f} meters"
        part2 = f"{obj2['name']} {obj2['direction']}, {obj2['distance']:.0f} meters"
        return f"{part1}. {part2}"
    
    else:  # 3 or more objects
        # Mention closest object in detail, then summarize others
        closest = top_objects[0]
        if closest['motion'] == "Moving":
            main_part = f"{closest['name']} moving {closest['direction']}, {closest['distance']:.0f} meters"
        else:
            main_part = f"{closest['name']} {closest['direction']}, {closest['distance']:.0f} meters"
        
        # Count other objects by direction
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
    if len(history) < 5:  # Need some history to determine motion
        return "Stationary"
    
    # Check if position has changed significantly
    recent_positions = history[-5:]
    x_positions = [pos[0] for pos in recent_positions]
    
    # Calculate movement range
    x_range = max(x_positions) - min(x_positions)
    movement_threshold = 30  # pixels
    
    return "Moving" if x_range > movement_threshold else "Stationary"

def cleanup_inactive_tracks(active_track_ids: Set[int], track_histories: Dict, 
                          cooldown_state: Dict):
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

def process_frame(frame: np.ndarray, pipeline_state: Dict, 
                 timestamp: float = None) -> Tuple[np.ndarray, List[bytes]]:
    """Process a single frame and generate alerts"""
    if timestamp is None:
        timestamp = time.time()
    
    pipeline_state['frame_count'] += 1
    
    # Step 1: Object detection and tracking
    detections = detect_and_track(pipeline_state['model'], frame)
    
    # Step 2: Update tracking histories
    active_track_ids = set()
    for detection in detections:
        track_id = detection['track_id']
        cx, cy = detection['center']
        
        active_track_ids.add(track_id)
        
        if track_id not in pipeline_state['track_histories']:
            pipeline_state['track_histories'][track_id] = deque(maxlen=HISTORY_FRAMES)
        pipeline_state['track_histories'][track_id].append((cx, cy))
    
    # Step 3: Generate alerts based on detections
    frame_alerts = process_detections(
        detections, timestamp, frame.shape[1], 
        pipeline_state['track_histories'], pipeline_state['cooldown_state']
    )
    
    # Step 4: Generate audio for alerts
    audio_data_list = []
    for alert in frame_alerts:
        audio_segment = generate_tts_audio(alert.alert_text, pipeline_state['audio_cache'])
        if audio_segment:
            # Option 1: Play audio locally on server
            if ENABLE_LOCAL_AUDIO:
                # Play audio on server (requires audio output)
                threading.Thread(target=play_audio_locally, args=(audio_segment,), daemon=True).start()
            
            # Option 2: Send audio data to client (always done for API compatibility)
            audio_bytes = export_audio_bytes(audio_segment)
            audio_data_list.append(audio_bytes)
            print(f"🔊 Alert: {alert.alert_text}")
        
        pipeline_state['alert_log'].append(alert)
    
    # Step 4.5: Generate continuous audio narration (only if no alerts and mode enabled)
    if (not frame_alerts and CONTINUOUS_AUDIO_MODE and 
        should_provide_continuous_audio(timestamp, pipeline_state)):
        
        narration_text = generate_continuous_narration_from_detections(
            detections, frame.shape[1], pipeline_state['track_histories']
        )
        
        if narration_text:
            narration_audio = generate_tts_audio(narration_text, pipeline_state['audio_cache'])
            if narration_audio:
                if ENABLE_LOCAL_AUDIO:
                    threading.Thread(target=play_audio_locally, args=(narration_audio,), daemon=True).start()
                
                audio_bytes = export_audio_bytes(narration_audio)
                audio_data_list.append(audio_bytes)
                print(f"🗣️  Narration: {narration_text}")
        
        pipeline_state['last_continuous_audio_time'] = timestamp
    
    # Step 5: Annotate frame with detection info
    annotated_frame = annotate_frame(
        frame, detections, frame.shape[1], pipeline_state['track_histories']
    )
    
    # Step 6: Cleanup inactive tracks
    cleanup_inactive_tracks(active_track_ids, pipeline_state['track_histories'], 
                           pipeline_state['cooldown_state'])
    
    return annotated_frame, audio_data_list

def get_pipeline_stats(pipeline_state: Dict) -> Dict:
    """Get comprehensive pipeline statistics"""
    current_time = time.time()
    
    # Count recent alerts (last 60 seconds)
    recent_alerts = [alert for alert in pipeline_state['alert_log'] 
                    if current_time - alert.timestamp < 60.0]
    
    # Count by class
    class_counts = {}
    for alert in recent_alerts:
        class_counts[alert.class_name] = class_counts.get(alert.class_name, 0) + 1
    
    return {
        'pipeline': {
            'frame_count': pipeline_state['frame_count'],
            'is_initialized': pipeline_state['model'] is not None
        },
        'alerts': {
            'total_alerts': len(pipeline_state['alert_log']),
            'recent_alerts_60s': len(recent_alerts),
            'tracked_objects': len(pipeline_state['cooldown_state']['alerted_tracks']),
            'class_counts_60s': class_counts,
            'last_global_alert': pipeline_state['cooldown_state']['last_global_alert_time']
        },
        'audio_cache': {
            'cached_items': len(pipeline_state['audio_cache']),
            'cache_limit': AUDIO_CACHE_SIZE,
            'cache_usage': len(pipeline_state['audio_cache']) / AUDIO_CACHE_SIZE
        },
        'detection_tracks': len(pipeline_state['track_histories'])
    }

def reset_pipeline(pipeline_state: Dict):
    """Reset pipeline state"""
    pipeline_state['frame_count'] = 0
    pipeline_state['alert_log'].clear()
    pipeline_state['audio_cache'].clear()
    pipeline_state['track_histories'].clear()
    pipeline_state['cooldown_state'] = {
        'alerted_tracks': {},
        'last_alert_time_by_class': {},
        'last_global_alert_time': -float('inf')
    }
    print("🔄 Pipeline state reset")

# ============================================================================
# CAMERA INPUT FUNCTIONS
# ============================================================================

def get_available_cameras():
    """Get list of available camera devices"""
    available_cameras = []
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'name': f'Camera {i}'
                })
            cap.release()
    return available_cameras

def capture_frame_from_camera(camera_index: int = 0):
    """Capture a single frame from camera"""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    
    try:
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        
        return frame
    finally:
        cap.release()

class CameraStream:
    """Manages continuous camera streaming"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.is_streaming = False
        
    def start_stream(self):
        """Start camera stream"""
        if self.cap is not None:
            self.stop_stream()
            
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        self.is_streaming = True
        print(f"📹 Camera {self.camera_index} streaming started")
        
    def stop_stream(self):
        """Stop camera stream"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_streaming = False
        print(f"📹 Camera {self.camera_index} streaming stopped")
        
    def capture_frame(self):
        """Capture frame from active stream"""
        if not self.is_streaming or self.cap is None:
            raise RuntimeError("Camera stream not active")
        
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from stream")
        
        return frame
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop_stream()

# Global camera stream
camera_stream = None

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
        
        # Stop streaming if no clients
        if not self.active_connections:
            self.stop_streaming()
            
    async def send_to_client(self, websocket: WebSocket, message: dict):
        """Send message to specific client with timeout"""
        try:
            await asyncio.wait_for(
                websocket.send_json(message), 
                timeout=WEBSOCKET_SEND_TIMEOUT
            )
            return True
        except (asyncio.TimeoutError, Exception) as e:
            print(f"⚠️ Failed to send to client: {e}")
            self.disconnect(websocket)
        return False
        
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
            
        # Send to all clients, remove failed connections
        disconnected = []
        
        for websocket in self.active_connections[:]:
            success = await self.send_to_client(websocket, message)
            if not success:
                disconnected.append(websocket)
        
        # Clean up failed connections
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
        """Main continuous streaming loop"""
        global camera_stream, pipeline_state
        
        print("🎥 Starting continuous camera streaming loop...")
        
        while self.streaming_active and self.active_connections:
            try:
                # Check if camera stream is active
                if camera_stream is None or not camera_stream.is_streaming:
                    await self.broadcast({
                        "type": "error",
                        "message": "Camera stream not active",
                        "timestamp": time.time()
                    })
                    await asyncio.sleep(1.0)
                    continue
                
                # Capture and process frame
                frame = camera_stream.capture_frame()
                annotated_frame, audio_data_list = process_frame(frame, pipeline_state)
                
                # Encode frame as JPEG for streaming
                ret, buffer = cv2.imencode('.jpg', annotated_frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Encode audio data
                    audio_data = []
                    for audio_bytes in audio_data_list:
                        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                        audio_data.append(audio_b64)
                    
                    # Broadcast frame and audio
                    message = {
                        "type": "frame_data",
                        "timestamp": time.time(),
                        "frame_count": pipeline_state['frame_count'],
                        "frame_data": frame_b64,
                        "audio_data": audio_data,
                        "alerts_count": len(audio_data_list),
                        "detections_count": len(detect_and_track(pipeline_state['model'], frame))
                    }
                    
                    await self.broadcast(message)
                
                # Control frame rate
                await asyncio.sleep(STREAMING_FRAME_DELAY)
                
            except Exception as e:
                print(f"❌ Error in streaming loop: {e}")
                await self.broadcast({
                    "type": "error",
                    "message": f"Streaming error: {str(e)}",
                    "timestamp": time.time()
                })
                await asyncio.sleep(1.0)
        
        print("🎥 Continuous streaming loop ended")

# Global connection manager
connection_manager = ConnectionManager()

# ============================================================================
# HTTP SERVER
# ============================================================================

# Global pipeline state
pipeline_state = None

# FastAPI app
app = FastAPI(title="Vision Assistance API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline_state, connection_manager
    pipeline_state = initialize_pipeline()
    print("🔗 WebSocket connection manager initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global connection_manager, camera_stream
    print("🛑 Server shutting down...")
    
    # Stop streaming and cleanup
    if connection_manager:
        connection_manager.stop_streaming()
    
    if camera_stream:
        camera_stream.stop_stream()

@app.post("/process_frame")
async def process_frame_endpoint(file: UploadFile = File(...)):
    """Process uploaded image file and return audio alerts"""
    try:
        # Read image file
        image_data = await file.read()
        
        # Decode image
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Process through pipeline
        annotated_frame, audio_data_list = process_frame(frame, pipeline_state)
        
        # Encode audio data as base64
        audio_data = []
        for audio_bytes in audio_data_list:
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            audio_data.append(audio_b64)
        
        return {
            "success": True,
            "timestamp": time.time(),
            "frame_count": pipeline_state['frame_count'],
            "detections_count": len(detect_and_track(pipeline_state['model'], frame)),
            "alerts_count": len(audio_data_list),
            "audio_data": audio_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_frame_b64")
async def process_frame_b64_endpoint(data: dict):
    """Process base64 encoded image and return audio alerts"""
    try:
        frame_b64 = data.get('frame')
        if not frame_b64:
            raise HTTPException(status_code=400, detail="No frame data provided")
        
        # Decode frame
        frame_data = base64.b64decode(frame_b64)
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode frame")
        
        # Process through pipeline
        annotated_frame, audio_data_list = process_frame(frame, pipeline_state)
        
        # Encode audio data as base64
        audio_data = []
        for audio_bytes in audio_data_list:
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            audio_data.append(audio_b64)
        
        return {
            "success": True,
            "timestamp": time.time(),
            "frame_count": pipeline_state['frame_count'],
            "detections_count": len(detect_and_track(pipeline_state['model'], frame)),
            "alerts_count": len(audio_data_list),
            "audio_data": audio_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats_endpoint():
    """Get pipeline statistics"""
    if pipeline_state is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    stats = get_pipeline_stats(pipeline_state)
    return {"success": True, "data": stats}

@app.post("/reset")
async def reset_pipeline_endpoint():
    """Reset pipeline state"""
    if pipeline_state is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    reset_pipeline(pipeline_state)
    return {"success": True, "message": "Pipeline state reset"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "pipeline_initialized": pipeline_state is not None,
        "model_loaded": pipeline_state['model'] is not None if pipeline_state else False
    }

@app.get("/cameras")
async def get_cameras():
    """Get available cameras"""
    try:
        cameras = get_available_cameras()
        return {"success": True, "cameras": cameras}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/camera/start/{camera_index}")
async def start_camera(camera_index: int):
    """Start camera stream"""
    global camera_stream
    try:
        if camera_stream is not None:
            camera_stream.stop_stream()
        
        camera_stream = CameraStream(camera_index)
        camera_stream.start_stream()
        
        return {
            "success": True, 
            "message": f"Camera {camera_index} started",
            "camera_index": camera_index
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/camera/stop")
async def stop_camera():
    """Stop camera stream"""
    global camera_stream
    try:
        if camera_stream is not None:
            camera_stream.stop_stream()
            camera_stream = None
        
        return {"success": True, "message": "Camera stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/camera/capture")
async def capture_from_camera():
    """Capture and process frame from active camera stream"""
    global camera_stream
    try:
        if camera_stream is None or not camera_stream.is_streaming:
            raise HTTPException(status_code=400, detail="No active camera stream. Start camera first.")
        
        # Capture frame from stream
        frame = camera_stream.capture_frame()
        
        # Process through pipeline
        annotated_frame, audio_data_list = process_frame(frame, pipeline_state)
        
        # Encode audio data as base64
        audio_data = []
        for audio_bytes in audio_data_list:
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            audio_data.append(audio_b64)
        
        return {
            "success": True,
            "timestamp": time.time(),
            "frame_count": pipeline_state['frame_count'],
            "detections_count": len(detect_and_track(pipeline_state['model'], frame)),
            "alerts_count": len(audio_data_list),
            "audio_data": audio_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/camera/capture_single/{camera_index}")
async def capture_single_frame(camera_index: int):
    """Capture and process a single frame from specified camera"""
    try:
        # Capture frame from camera
        frame = capture_frame_from_camera(camera_index)
        
        # Process through pipeline
        annotated_frame, audio_data_list = process_frame(frame, pipeline_state)
        
        # Encode audio data as base64
        audio_data = []
        for audio_bytes in audio_data_list:
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            audio_data.append(audio_b64)
        
        return {
            "success": True,
            "timestamp": time.time(),
            "frame_count": pipeline_state['frame_count'],
            "detections_count": len(detect_and_track(pipeline_state['model'], frame)),
            "alerts_count": len(audio_data_list),
            "audio_data": audio_data,
            "camera_index": camera_index
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/camera/status")
async def camera_status():
    """Get camera stream status"""
    global camera_stream
    if camera_stream is None:
        return {
            "success": True,
            "streaming": False,
            "camera_index": None
        }
    
    return {
        "success": True,
        "streaming": camera_stream.is_streaming,
        "camera_index": camera_stream.camera_index
    }

# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws/camera/stream")
async def websocket_camera_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time camera streaming"""
    global connection_manager, camera_stream
    
    # Try to connect client
    connected = await connection_manager.connect(websocket)
    if not connected:
        return
    
    try:
        # Send initial status
        await connection_manager.send_to_client(websocket, {
            "type": "connected",
            "message": "WebSocket connected successfully",
            "timestamp": time.time(),
            "camera_active": camera_stream is not None and camera_stream.is_streaming,
            "client_count": len(connection_manager.active_connections)
        })
        
        # Start streaming if this is the first client and camera is ready
        if (len(connection_manager.active_connections) == 1 and 
            camera_stream is not None and camera_stream.is_streaming):
            connection_manager.start_streaming()
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (for control commands)
                data = await websocket.receive_json()
                
                # Handle client commands
                if data.get("command") == "start_camera":
                    camera_index = data.get("camera_index", 0)
                    try:
                        if camera_stream is not None:
                            camera_stream.stop_stream()
                        
                        camera_stream = CameraStream(camera_index)
                        camera_stream.start_stream()
                        
                        await connection_manager.send_to_client(websocket, {
                            "type": "camera_started",
                            "camera_index": camera_index,
                            "timestamp": time.time()
                        })
                        
                        # Start streaming if not already active
                        if not connection_manager.streaming_active:
                            connection_manager.start_streaming()
                            
                    except Exception as e:
                        await connection_manager.send_to_client(websocket, {
                            "type": "error",
                            "message": f"Failed to start camera: {str(e)}",
                            "timestamp": time.time()
                        })
                
                elif data.get("command") == "stop_camera":
                    if camera_stream is not None:
                        camera_stream.stop_stream()
                        camera_stream = None
                    
                    connection_manager.stop_streaming()
                    
                    await connection_manager.send_to_client(websocket, {
                        "type": "camera_stopped",
                        "timestamp": time.time()
                    })
                
                elif data.get("command") == "get_stats":
                    if pipeline_state is not None:
                        stats = get_pipeline_stats(pipeline_state)
                        await connection_manager.send_to_client(websocket, {
                            "type": "stats",
                            "data": stats,
                            "timestamp": time.time()
                        })
                
                elif data.get("command") == "process_frame":
                    # Handle frame processing from client
                    frame_data = data.get("frame_data")
                    if frame_data and pipeline_state is not None:
                        try:
                            # Decode base64 frame
                            import base64
                            frame_bytes = base64.b64decode(frame_data)
                            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                # Process frame through pipeline
                                annotated_frame, audio_data_list = process_frame(frame, pipeline_state)
                                
                                # Encode annotated frame back to base64
                                ret, buffer = cv2.imencode('.jpg', annotated_frame, 
                                                         [cv2.IMWRITE_JPEG_QUALITY, 80])
                                if ret:
                                    annotated_frame_b64 = base64.b64encode(buffer).decode('utf-8')
                                    
                                    # Encode audio data
                                    audio_data = []
                                    for audio_bytes in audio_data_list:
                                        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                                        audio_data.append(audio_b64)
                                    
                                    # Send processing result back to client
                                    await connection_manager.send_to_client(websocket, {
                                        "type": "processing_result",
                                        "annotated_frame": annotated_frame_b64,
                                        "audio_data": audio_data,
                                        "detections_count": len(detect_and_track(pipeline_state['model'], frame)),
                                        "alerts_count": len(audio_data_list),
                                        "timestamp": time.time()
                                    })
                                else:
                                    await connection_manager.send_to_client(websocket, {
                                        "type": "error",
                                        "message": "Failed to encode processed frame",
                                        "timestamp": time.time()
                                    })
                            else:
                                await connection_manager.send_to_client(websocket, {
                                    "type": "error",
                                    "message": "Failed to decode frame data",
                                    "timestamp": time.time()
                                })
                        except Exception as e:
                            await connection_manager.send_to_client(websocket, {
                                "type": "error",
                                "message": f"Frame processing error: {str(e)}",
                                "timestamp": time.time()
                            })
                    else:
                        await connection_manager.send_to_client(websocket, {
                            "type": "error",
                            "message": "No frame data provided or pipeline not ready",
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
        "streaming_active": connection_manager.streaming_active,
        "streaming_fps": STREAMING_FPS,
        "camera_active": camera_stream is not None and camera_stream.is_streaming if camera_stream else False
    }


@app.get("/")
def index():
    return HTMLResponse(open("websocket_camera_client.html", "r", encoding="utf-8").read())
# ============================================================================
# MAIN SERVER ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the server"""
    print("🚀 Vision Assistance HTTP Server")
    print("=" * 50)
    print(f"Server will start on http://{SERVER_HOST}:{SERVER_PORT}")
    
    # Update YOLO model path if needed
    print(f"YOLO Model Path: {YOLO_MODEL_PATH}")
    print("Make sure to update YOLO_MODEL_PATH in the script if needed")
    
    # Start server
    uvicorn.run(
        "vision_assist_server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    main()
