#!/usr/bin/env python3
"""
Live camera stream — open in any browser on the same WiFi.
http://192.168.0.102:8080
"""

import io
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from picamera2 import Picamera2

FPS = 30
FRAME_INTERVAL = 1.0 / FPS

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720)}))
picam2.start()

frame_lock = threading.Lock()
latest_frame = b""
frame_event = threading.Event()


def capture_loop():
    global latest_frame
    while True:
        t0 = time.monotonic()
        buf = io.BytesIO()
        picam2.capture_file(buf, format="jpeg")
        with frame_lock:
            latest_frame = buf.getvalue()
        frame_event.set()
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, FRAME_INTERVAL - elapsed))


threading.Thread(target=capture_loop, daemon=True).start()


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    frame_event.wait()
                    frame_event.clear()
                    with frame_lock:
                        frame = latest_frame
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self.send_response(404)
            self.end_headers()


HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pi Camera</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #000; display: flex; flex-direction: column;
           align-items: center; justify-content: center; min-height: 100vh; }
    img  { max-width: 100%; max-height: 100vh; object-fit: contain; }
    p    { color: #888; font-family: monospace; margin-top: 8px; font-size: 12px; }
  </style>
</head>
<body>
  <img src="/stream" alt="camera feed">
  <p>Raspberry Pi 5 &mdash; OV5647 &mdash; 1280x720 &mdash; 30 fps</p>
</body>
</html>"""


if __name__ == "__main__":
    HOST = "0.0.0.0"
    PORT = 8080
    server = HTTPServer((HOST, PORT), StreamHandler)
    print(f"Camera stream running at 1280x720 @ 30 fps — open in browser:")
    print(f"  http://192.168.0.102:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
