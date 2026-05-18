#!/usr/bin/env python3
"""
YOLO on Hailo-10H — live MJPEG stream with FPS overlay.
Open in browser: http://<pi-ip>:8080

Uses yolov8m_h10.hef (80 COCO classes, NMS on-chip, UINT8 input).
"""

import time
import socket
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
from picamera2 import Picamera2

from hailo_platform import (
    VDevice, HEF, FormatType, HailoSchedulingAlgorithm,
)

# ── Config ────────────────────────────────────────────────
HEF_PATH      = "/usr/share/hailo-models/yolov8m_h10.hef"
CAM_WIDTH     = 960
CAM_HEIGHT    = 540
INPUT_SIZE    = 640
CONF_THRESH   = 0.35
TARGET_FPS    = 30
JPEG_QUALITY  = 75
# ──────────────────────────────────────────────────────────

CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]
COLORS = [_PALETTE[i % len(_PALETTE)] for i in range(80)]

print(f"Loading {Path(HEF_PATH).name} ...")
hef = HEF(HEF_PATH)

params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
vdevice = VDevice(params)

infer_model = vdevice.create_infer_model(HEF_PATH)
infer_model.set_batch_size(1)
infer_model.input().set_format_type(FormatType.UINT8)
infer_model.output().set_format_type(FormatType.FLOAT32)
print("Hailo model ready.")

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT)}))
picam2.start()

frame_lock   = threading.Lock()
latest_frame = b""
frame_event  = threading.Event()


def parse_nms(output, scale_x, scale_y):
    """Parse Hailo NMS_BY_CLASS output → list of (x1,y1,x2,y2,conf,cls)."""
    dets = []
    # output is a list (per class) of arrays [N, 5] with (y1, x1, y2, x2, score) in [0,1]
    for cls_id, cls_boxes in enumerate(output):
        if cls_boxes is None or len(cls_boxes) == 0:
            continue
        for box in cls_boxes:
            y1n, x1n, y2n, x2n, score = box[:5]
            if score < CONF_THRESH:
                continue
            x1 = int(x1n * scale_x)
            y1 = int(y1n * scale_y)
            x2 = int(x2n * scale_x)
            y2 = int(y2n * scale_y)
            dets.append((x1, y1, x2, y2, float(score), cls_id))
    return dets


def inference_loop():
    global latest_frame
    interval = 1.0 / TARGET_FPS
    last = time.monotonic()
    fps  = 0.0

    with infer_model.configure() as configured:
        while True:
            t0 = time.monotonic()

            raw   = picam2.capture_array()
            frame = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)

            # Preprocess: letterbox-free resize, UINT8, RGB
            net_in = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            net_in = cv2.cvtColor(net_in, cv2.COLOR_BGR2RGB)
            net_in = np.ascontiguousarray(net_in)

            # Infer (synchronous)
            bindings = configured.create_bindings()
            bindings.input().set_buffer(net_in)
            out_buf = np.empty(infer_model.output().shape, dtype=np.float32)
            bindings.output().set_buffer(out_buf)
            configured.run([bindings], timeout=2000)

            # Hailo NMS output shape: variable. Parse via the layer info object.
            output_raw = bindings.output().get_buffer()
            dets = parse_nms(output_raw, CAM_WIDTH, CAM_HEIGHT)

            for x1, y1, x2, y2, conf, cls in dets:
                color = tuple(int(c) for c in COLORS[cls])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{CLASSES[cls]} {conf:.0%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now
            label = f"Hailo FPS: {fps:.1f}"
            cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            with frame_lock:
                latest_frame = jpeg.tobytes()
            frame_event.set()

            time.sleep(max(0.0, interval - (time.monotonic() - t0)))


threading.Thread(target=inference_loop, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers()
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
<title>Hailo YOLO Live</title>
<style>*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh}
img{max-width:100%;max-height:95vh;object-fit:contain}
p{color:#0f0;font-family:monospace;margin-top:6px;font-size:13px}</style></head>
<body><img src="/stream"><p>Pi 5 &mdash; Hailo-10H &mdash; YOLOv8m</p></body></html>"""


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


if __name__ == "__main__":
    PORT = 8080
    print(f"Hailo stream → http://{get_ip()}:{PORT}")
    print("Ctrl+C to stop.")
    srv = HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
