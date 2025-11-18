#!/bin/bash

# Vision Assistance Server Deployment Script (Port 8000 Version)
# Backend runs on: 8000
# Frontend via Nginx + HTTPS on: 443
# WebSockets proxied via Nginx: wss://YOURDOMAIN/ws/camera/stream

echo "🚀 Vision Assistance Server Deployment - Backend on PORT 8000"

# -------------------------------------------------------------
# 1. System Update
# -------------------------------------------------------------
sudo apt update && sudo apt upgrade -y

# -------------------------------------------------------------
# 2. Install Python & system dependencies
# -------------------------------------------------------------
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    libgstreamer1.0-0 gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly libavcodec-dev libavformat-dev \
    libswscale-dev ffmpeg portaudio19-dev python3-dev pkg-config

# -------------------------------------------------------------
# 3. Create Application Directory
# -------------------------------------------------------------
sudo mkdir -p /opt/vision-assist
sudo chown $USER:$USER /opt/vision-assist
cd /opt/vision-assist

# -------------------------------------------------------------
# 4. Python Virtual Environment
# -------------------------------------------------------------
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# -------------------------------------------------------------
# 5. Python Dependencies
# -------------------------------------------------------------
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python-headless numpy gtts pydub fastapi \
    "uvicorn[standard]" websockets python-multipart Pillow requests

echo "✅ Python packages installed."

# -------------------------------------------------------------
# 6. Verify FFmpeg
# -------------------------------------------------------------
if ! command -v ffmpeg >/dev/null; then
    sudo apt install -y ffmpeg
fi

# -------------------------------------------------------------
# 7. Create systemd service for backend (PORT 8000)
# -------------------------------------------------------------
echo "⚙️ Creating systemd service on PORT 8000..."

sudo tee /etc/systemd/system/vision-assist.service >/dev/null <<EOF
[Unit]
Description=Vision Assistance Backend Server (FastAPI + WebSockets)
After=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/vision-assist
Environment=PATH=/opt/vision-assist/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/opt/vision-assist/venv/bin/uvicorn vision_assist_server:app --host 0.0.0.0 --port 8000 --log-level info
Restart=always
RestartSec=3
KillMode=mixed
KillSignal=SIGINT
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
EOF

# -------------------------------------------------------------
# 8. Nginx Frontend Folder (Optional)
# -------------------------------------------------------------
sudo mkdir -p /var/www/vision-assist
sudo cp /opt/vision-assist/*.html /var/www/vision-assist/ 2>/dev/null || true

# -------------------------------------------------------------
# 9. Copy project files
# -------------------------------------------------------------
cd "$OLDPWD" || cd ~
if [[ -f "vision_assist_server.py" && -f "yolov8n_optuna_best.pt" ]]; then
    cp vision_assist_server.py /opt/vision-assist/
    cp yolov8n_optuna_best.pt /opt/vision-assist/
    cp requirements.txt /opt/vision-assist/ 2>/dev/null || true

    cp websocket_camera_client.html /opt/vision-assist/ 2>/dev/null || true
    cp websocket_live_test.py /opt/vision-assist/ 2>/dev/null || true
    cp WEBSOCKET_STREAMING.md /opt/vision-assist/ 2>/dev/null || true

    sed -i "s|YOLO_MODEL_PATH = './yolov8n_optuna_best.pt'|YOLO_MODEL_PATH = '/opt/vision-assist/yolov8n_optuna_best.pt'|" /opt/vision-assist/vision_assist_server.py

    echo "📁 Backend + model copied to /opt/vision-assist"
fi

# -------------------------------------------------------------
# 10. Start Backend Service
# -------------------------------------------------------------
sudo systemctl daemon-reload
sudo systemctl enable vision-assist
sudo systemctl restart vision-assist

# -------------------------------------------------------------
# 11. Status check
# -------------------------------------------------------------
echo "📊 Service Status:"
sudo systemctl status vision-assist --no-pager -l

echo ""
echo "🔍 Testing backend..."
curl -s http://localhost:8000/health || echo "❌ Health check failed"

echo "=============================================================="
echo "🎉 Deployment Complete!"
echo ""
echo "Backend URL:"
echo "   🔗 http://$(curl -s ifconfig.me):8000"
echo ""
echo "When accessed through Nginx with HTTPS:"
echo "   🔗 https://4.188.81.64.nip.io/"
echo ""
echo "WebSocket:"
echo "   🔌 Local backend: ws://$(curl -s ifconfig.me):8000/ws/camera/stream"
echo "   🔌 Public HTTPS: wss://4.188.81.64.nip.io/ws/camera/stream"
echo ""
echo "Systemd Management:"
echo "   ➤ sudo systemctl restart vision-assist"
echo "   ➤ sudo journalctl -u vision-assist -f"
echo "=============================================================="

