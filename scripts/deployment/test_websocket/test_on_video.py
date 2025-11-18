# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "websockets>=10.0",
#   "opencv-python>=4.5.0",
#   "numpy>=1.21.0",
# ]
# ///



#!/usr/bin/env python3
"""
Video WebSocket Test Client
Tests vision assistance system with recorded video and saves audio packets continuously.
"""

import asyncio
import websockets
import json
import base64
import cv2
import os
import time
from datetime import datetime
import argparse

# Configuration
SERVER_URL = "ws://4.188.81.64/ws/camera/stream"
OUTPUT_DIR = "audio_output"
VIDEO_FPS = 5  # Frames per second to send
FRAME_QUALITY = 80  # JPEG quality (1-100)

def ensure_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created output directory: {OUTPUT_DIR}")

def save_audio_packet(audio_b64, frame_number, timestamp):
    """Save base64 audio data as MP3 file"""
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_b64)
        
        # Generate filename with frame number and timestamp
        time_str = datetime.fromtimestamp(timestamp).strftime("%H%M%S")
        filename = f"frame_{frame_number:04d}_{time_str}.mp3"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Save audio file
        with open(filepath, "wb") as f:
            f.write(audio_bytes)
        
        print(f"🔊 Saved audio: {filename} ({len(audio_bytes)} bytes)")
        return filepath
        
    except Exception as e:
        print(f"❌ Error saving audio for frame {frame_number}: {e}")
        return None

def load_video(video_path):
    """Load video file and return frames"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / video_fps
    
    print(f"📹 Video loaded: {os.path.basename(video_path)}")
    print(f"   📊 Frames: {frame_count}, FPS: {video_fps:.1f}, Duration: {duration:.1f}s")
    
    frames = []
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_number += 1
    
    cap.release()
    print(f"✅ Extracted {len(frames)} frames from video")
    return frames

def encode_frame(frame, quality=80):
    """Encode frame as base64 JPEG"""
    # Encode frame as JPEG
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ret:
        raise RuntimeError("Failed to encode frame as JPEG")
    
    # Convert to base64
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    return frame_b64

async def process_video_websocket(video_path, server_url=SERVER_URL, fps=5, quality=80):
    """Process video through WebSocket and save audio packets"""
    print(f"🚀 Starting video WebSocket test")
    print(f"📁 Video: {video_path}")
    print(f"🔗 Server: {server_url}")
    print(f"⚡ Processing FPS: {fps}")
    
    # Load video frames
    frames = load_video(video_path)
    
    # Create output directory
    ensure_output_directory()
    
    # Statistics
    total_frames = len(frames)
    processed_frames = 0
    total_audio_packets = 0
    start_time = time.time()
    
    try:
        print(f"\n🔗 Connecting to WebSocket...")
        async with websockets.connect(server_url) as websocket:
            print(f"✅ Connected to {server_url}")
            
            # Wait for connection confirmation
            welcome_msg = await websocket.recv()
            welcome_data = json.loads(welcome_msg)
            print(f"📨 Server says: {welcome_data.get('message', 'Connected')}")
            
            # Process frames at specified FPS
            frame_delay = 1.0 / fps
            
            for frame_idx, frame in enumerate(frames):
                frame_start_time = time.time()
                
                try:
                    # Encode frame
                    frame_b64 = encode_frame(frame, quality)
                    
                    # Send frame for processing
                    message = {
                        "command": "process_frame",
                        "frame_data": frame_b64
                    }
                    
                    print(f"📤 Sending frame {frame_idx + 1}/{total_frames}...")
                    await websocket.send(json.dumps(message))
                    
                    # Wait for response
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    result = json.loads(response)
                    
                    # Handle processing result
                    if result.get("type") == "processing_result":
                        detections = result.get("detections_count", 0)
                        alerts = result.get("alerts_count", 0)
                        
                        print(f"✅ Frame {frame_idx + 1}: {detections} detections, {alerts} alerts")
                        
                        # Save audio packets
                        audio_data = result.get("audio_data", [])
                        for audio_idx, audio_b64 in enumerate(audio_data):
                            save_audio_packet(audio_b64, frame_idx + 1, time.time())
                            total_audio_packets += 1
                        
                        processed_frames += 1
                    
                    elif result.get("type") == "error":
                        print(f"❌ Server error for frame {frame_idx + 1}: {result.get('message')}")
                    
                    else:
                        print(f"⚠️ Unexpected response type: {result.get('type')}")
                
                except asyncio.TimeoutError:
                    print(f"⏱️ Timeout waiting for frame {frame_idx + 1} response")
                except Exception as e:
                    print(f"❌ Error processing frame {frame_idx + 1}: {e}")
                
                # Control frame rate
                frame_elapsed = time.time() - frame_start_time
                if frame_elapsed < frame_delay:
                    await asyncio.sleep(frame_delay - frame_elapsed)
            
            print(f"\n🏁 Video processing completed!")
            
    except websockets.exceptions.ConnectionClosedException:
        print(f"❌ WebSocket connection closed unexpectedly")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    
    # Print final statistics
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n📊 Processing Summary:")
    print(f"   📹 Total frames: {total_frames}")
    print(f"   ✅ Processed frames: {processed_frames}")
    print(f"   🔊 Audio packets saved: {total_audio_packets}")
    print(f"   ⏱️ Processing time: {processing_time:.1f}s")
    print(f"   📈 Avg FPS: {processed_frames / processing_time:.1f}")
    print(f"   📁 Audio files saved in: {OUTPUT_DIR}/")

def find_video_file():
    """Find video file in current directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            return file
    
    return None

async def main():
    parser = argparse.ArgumentParser(description="Test WebSocket with recorded video")
    parser.add_argument("--video", "-v", help="Path to video file")
    parser.add_argument("--fps", "-f", type=float, default=VIDEO_FPS, 
                       help=f"Frames per second to process (default: {VIDEO_FPS})")
    parser.add_argument("--server", "-s", default=SERVER_URL,
                       help=f"WebSocket server URL (default: {SERVER_URL})")
    parser.add_argument("--quality", "-q", type=int, default=FRAME_QUALITY,
                       help=f"JPEG quality 1-100 (default: {FRAME_QUALITY})")
    
    args = parser.parse_args()
    
    # Find video file
    if args.video:
        video_path = args.video
    else:
        video_path = find_video_file()
        if not video_path:
            print("❌ No video file specified and none found in current directory")
            print("   Supported formats: .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm")
            print("   Usage: python test_on_video.py --video your_video.mp4")
            return 1
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    print("🎯 Video WebSocket Test Client")
    print("=" * 50)
    
    # Update settings and run
    try:
        await process_video_websocket(video_path, args.server, args.fps, args.quality)
        return 0
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))