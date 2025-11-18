# Install websocat first (on your VM)
curl -L https://github.com/vi/websocat/releases/latest/download/websocat.x86_64-unknown-linux-musl -o websocat
chmod +x websocat

# Test WebSocket connection
./websocat ws://4.188.81.64/ws/camera/stream
