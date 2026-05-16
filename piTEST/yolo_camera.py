#!/usr/bin/env python3
"""
YOLO on Pi camera — no torch, no CUDA.
Opens automatically at: http://192.168.0.102:8080
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
MODEL_FILE  = Path(__file__).parent / "yolov8n.onnx"
FPS         = 15
WIDTH, HEIGHT = 1280, 720
INPUT_SIZE  = 640
CONF        = 0.4
IOU         = 0.45
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

# ── Check model exists ────────────────────────────────────
if not MODEL_FILE.exists():
    print("ERROR: yolov8n.onnx not found.")
    print("Run this on your PC:")
    print('  python -c "from ultralytics import YOLO; YOLO(\'yolov8n.pt\').export(format=\'onnx\')"')
    print(f"  scp yolov8n.onnx ai@192.168.0.102:{MODEL_FILE}")
    exit(1)

# ── Load model ────────────────────────────────────────────
print("Loading model...")
session    = ort.InferenceSession(str(MODEL_FILE), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("Ready.")

# ── Camera ────────────────────────────────────────────────
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (WIDTH, HEIGHT)}))
picam2.start()

frame_lock   = threading.Lock()
latest_frame = b""
frame_event  = threading.Event()


def preprocess(img):
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(img, (2, 0, 1)), 0)


def postprocess(out, w, h):
    preds = out[0][0]
    sx, sy = w / INPUT_SIZE, h / INPUT_SIZE
    boxes, scores, ids = [], [], []
    for p in preds.T:
        confs = p[4:]
        cls   = int(np.argmax(confs))
        conf  = float(confs[cls])
        if conf < CONF:
            continue
        cx, cy, bw, bh = p[:4]
        x1 = int((cx - bw / 2) * sx)
        y1 = int((cy - bh / 2) * sy)
        boxes.append([x1, y1, int(bw * sx), int(bh * sy)])
        scores.append(conf)
        ids.append(cls)
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, CONF, IOU)
    return [(boxes[i][0], boxes[i][1],
             boxes[i][0]+boxes[i][2], boxes[i][1]+boxes[i][3],
             scores[i], ids[i]) for i in keep]


def inference_loop():
    global latest_frame
    interval = 1.0 / FPS
    while True:
        t0    = time.monotonic()
        frame = picam2.capture_array()
        dets  = postprocess(session.run(None, {input_name: preprocess(frame)}), WIDTH, HEIGHT)

        for x1, y1, x2, y2, conf, cls in dets:
            label = f"{CLASSES[cls]} {conf:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = jpeg.tobytes()
        frame_event.set()
        time.sleep(max(0.0, interval - (time.monotonic() - t0)))


threading.Thread(target=inference_loop, daemon=True).start()


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
            except (BrokenPipeError, ConnectionResetError): pass
        else:
            self.send_response(404); self.end_headers()


HTML = """<!DOCTYPE html><html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO Live</title>
<style>*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh}
img{max-width:100%;max-height:95vh;object-fit:contain}
p{color:#0f0;font-family:monospace;margin-top:6px;font-size:13px}</style></head>
<body><img src="/stream"><p>Pi 5 &mdash; YOLOv8n &mdash; onnxruntime &mdash; 1280x720</p></body></html>"""


if __name__ == "__main__":
    print("YOLO stream → http://192.168.0.102:8080")
    srv = HTTPServer(("0.0.0.0", 8080), Handler)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    finally: picam2.stop()
