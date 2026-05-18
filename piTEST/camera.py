#!/usr/bin/env python3
"""
Camera test — live MJPEG stream with FPS overlay.
Open in browser: http://<pi-ip>:8080
"""

import time
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
from picamera2 import Picamera2

WIDTH, HEIGHT = 1280, 720
TARGET_FPS    = 30
JPEG_QUALITY  = 85

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (WIDTH, HEIGHT)}))
picam2.start()

frame_lock   = threading.Lock()
latest_frame = b""
frame_event  = threading.Event()


def capture_loop():
    global latest_frame
    interval = 1.0 / TARGET_FPS
    last = time.monotonic()
    fps = 0.0
    while True:
        t0 = time.monotonic()

        raw = picam2.capture_array()
        img = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)

        now = time.monotonic()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now

        label = f"FPS: {fps:.1f}"
        cv2.putText(img, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)

        _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        with frame_lock:
            latest_frame = jpeg.tobytes()
        frame_event.set()

        time.sleep(max(0.0, interval - (time.monotonic() - t0)))


threading.Thread(target=capture_loop, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

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
                    frame_event.wait(); frame_event.clear()
                    with frame_lock: f = latest_frame
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404); self.end_headers()


HTML = """<!DOCTYPE html><html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pi Camera</title>
<style>*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh}
img{max-width:100%;max-height:95vh;object-fit:contain}
p{color:#0f0;font-family:monospace;margin-top:6px;font-size:13px}</style></head>
<body><img src="/stream"><p>Pi 5 &mdash; OV5647 &mdash; 1280x720</p></body></html>"""


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


if __name__ == "__main__":
    PORT = 8080
    print(f"Camera stream → http://{get_ip()}:{PORT}")
    print("Ctrl+C to stop.")
    srv = HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
