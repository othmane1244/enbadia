#!/usr/bin/env python3
"""
YOLO inference on live camera feed using onnxruntime (no torch/CUDA).
Open in browser: http://192.168.0.102:8080
"""

import io
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2

# ── Config ────────────────────────────────────────────────
MODEL_PATH  = Path(__file__).parent / "best.onnx"
FPS         = 15
WIDTH       = 1280
HEIGHT      = 720
INPUT_SIZE  = 640
CONF_THRESH = 0.4
IOU_THRESH  = 0.45
# ──────────────────────────────────────────────────────────

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Run export_model.py on your PC first, then transfer best.onnx to piTEST/"
    )

print(f"Loading model: {MODEL_PATH}")
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name  = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape   # [1, 3, 640, 640]
print("Model loaded.")

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (WIDTH, HEIGHT)}))
picam2.start()

frame_lock   = threading.Lock()
latest_frame = b""
frame_event  = threading.Event()


def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1]                          # BGR → RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))             # HWC → CHW
    return np.expand_dims(img, axis=0)             # → [1, 3, 640, 640]


def postprocess(outputs, orig_w, orig_h):
    """Parse YOLO output and return list of (x1,y1,x2,y2,conf,cls)."""
    preds = outputs[0][0]                          # [num_boxes, 4+num_classes]
    boxes, scores, class_ids = [], [], []

    sx = orig_w / INPUT_SIZE
    sy = orig_h / INPUT_SIZE

    for pred in preds.T:                           # ultralytics ONNX: [4+nc, 8400]
        confs = pred[4:]
        cls   = int(np.argmax(confs))
        conf  = float(confs[cls])
        if conf < CONF_THRESH:
            continue
        cx, cy, w, h = pred[:4]
        x1 = int((cx - w / 2) * sx)
        y1 = int((cy - h / 2) * sy)
        x2 = int((cx + w / 2) * sx)
        y2 = int((cy + h / 2) * sy)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(conf)
        class_ids.append(cls)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, IOU_THRESH)
    results = []
    for i in indices:
        x, y, w, h = boxes[i]
        results.append((x, y, x + w, y + h, scores[i], class_ids[i]))
    return results


def draw(frame, detections):
    for x1, y1, x2, y2, conf, cls in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"cls{cls} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return frame


def inference_loop():
    global latest_frame
    interval = 1.0 / FPS
    while True:
        t0 = time.monotonic()

        frame      = picam2.capture_array()
        blob       = preprocess(frame)
        outputs    = session.run(None, {input_name: blob})
        detections = postprocess(outputs, WIDTH, HEIGHT)
        annotated  = draw(frame, detections)

        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = jpeg.tobytes()
        frame_event.set()

        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, interval - elapsed))


threading.Thread(target=inference_loop, daemon=True).start()


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
  <title>YOLO Stream</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #000; display: flex; flex-direction: column;
           align-items: center; justify-content: center; min-height: 100vh; }
    img  { max-width: 100%; max-height: 95vh; object-fit: contain; }
    p    { color: #0f0; font-family: monospace; margin-top: 6px; font-size: 13px; }
  </style>
</head>
<body>
  <img src="/stream" alt="YOLO feed">
  <p>Raspberry Pi 5 &mdash; YOLO &mdash; onnxruntime CPU &mdash; 1280x720</p>
</body>
</html>"""


if __name__ == "__main__":
    PORT = 8080
    server = HTTPServer(("0.0.0.0", PORT), StreamHandler)
    print(f"YOLO stream running — open in browser:")
    print(f"  http://192.168.0.102:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
