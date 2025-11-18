
# 1. Deployment Instructions

Follow these steps to deploy the Vision Assistance Server on a VM or local Linux machine.

### **Step 1 — Upload Required Files**

Upload these files to the server:

* `vision_assist_server.py`
* `deploy.sh`
* `yolov8n_optuna_best.pt`
* Optional: `websocket_camera_client.html`, `websocket_live_test.py`

---

### **Step 2 — Make Script Executable**

```bash
chmod +x deploy.sh
```

---

### **Step 3 — Run Deployment**

```bash
./deploy.sh
```

The script automatically:

* Updates the system
* Installs Python and system dependencies
* Creates a virtual environment
* Installs all required Python packages
* Copies files to `/opt/vision-assist/`
* Creates a systemd service (`vision-assist.service`)
* Runs FastAPI on **port 80**
* Prints server and public URLs

---

### **Step 4 — Service Commands**

Check status:

```bash
sudo systemctl status vision-assist
```

Restart:

```bash
sudo systemctl restart vision-assist
```

Logs:

```bash
sudo journalctl -u vision-assist -f
```

After deployment access the server at:

```
http://<SERVER-IP>/
```

---

# 2. API Endpoint: `/process_frame`

### **Method**

`POST`

### **URL**

`/process_frame`

### **Purpose**

Processes a *single uploaded image* and returns:

* Number of detections
* Number of alerts generated
* Audio alerts (Base64 MP3 data)

---

## **Input Format**

### **Multipart Form Data**

Field:

```
file = <image_file>
```

Supported formats: `.jpg`, `.jpeg`, `.png`

### Example Request

```bash
curl -X POST -F "file=@road.jpg" http://SERVER/process_frame
```

---

##  **Output Format (JSON)**

### **Example Response**

```json
{
  "success": true,
  "timestamp": 1700000000,
  "frame_count": 52,
  "detections_count": 3,
  "alerts_count": 1,
  "audio_data": [
    "<base64_encoded_mp3>"
  ]
}
```

### **Field Description**

| Field              | Type         | Description                           |
| ------------------ | ------------ | ------------------------------------- |
| `success`          | bool         | Indicates successful processing       |
| `timestamp`        | float        | Server timestamp                      |
| `frame_count`      | int          | Total frames processed since startup  |
| `detections_count` | int          | Number of detected objects            |
| `alerts_count`     | int          | Audio alerts generated for this frame |
| `audio_data`       | list<string> | Base64 MP3 audio clips                |

---

# 3. WebSocket Endpoint: `/ws/camera/stream`

### **URL**

```
ws://SERVER-IP/ws/camera/stream
```

### **Purpose**

Provides:
✔ Real-time camera streaming
✔ Annotated frames (JPEG → Base64)
✔ Audio alerts (Base64 MP3)
✔ Continuous narration
✔ Ability for client to send commands (start camera, stop camera, get stats)

---

# 3.1  WebSocket – Client → Server Messages

These messages are **JSON commands the client can send**.

---

### **1. Start Camera**

```json
{
  "command": "start_camera",
  "camera_index": 0
}
```

### **2. Stop Camera**

```json
{
  "command": "stop_camera"
}
```

### **3. Request Stats**

```json
{
  "command": "get_stats"
}
```

### **4. Send Frame to Process**

(Client provides its own frame, instead of using camera.)

```json
{
  "command": "process_frame",
  "frame_data": "<base64_image>"
}
```

---

# 3.2 WebSocket – Server → Client Messages

Server continuously sends structured JSON packets.

---

### **1. Initial Connection Message**

```json
{
  "type": "connected",
  "message": "WebSocket connected successfully",
  "camera_active": true,
  "client_count": 1
}
```

---

### **2. Real-Time Frame Message**

Sent every frame (up to 30 FPS):

```json
{
  "type": "frame_data",
  "timestamp": 1700000000,
  "frame_count": 120,
  "frame_data": "<base64_jpeg>",
  "audio_data": ["<base64_mp3_audio>"],
  "alerts_count": 1,
  "detections_count": 3
}
```

---

### **frame_data**

Annotated frame: bounding boxes, distance, direction, motion.

### **audio_data**

Base64-encoded MP3s generated for the current frame
(alert or continuous narration).

---

### **3. Stats Response**

```json
{
  "type": "stats",
  "data": { ... }
}
```

---

### **4. Camera Status Messages**

Camera Started:

```json
{
  "type": "camera_started",
  "camera_index": 0
}
```

Camera Stopped:

```json
{
  "type": "camera_stopped"
}
```

---

### **5. Error Message**

```json
{
  "type": "error",
  "message": "Streaming error: <details>"
}
```
