#!/usr/bin/env python3
"""
YOLO test on Pi camera — onnxruntime CPU, MJPEG stream with FPS overlay.
Open in browser: http://<pi-ip>:8080

First run: ensure yolov8n.onnx is present in this folder.
  python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=320, simplify=True)"
  mv yolov8n.onnx piTEST/
"""

import time
import socket
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2

# ── Config ────────────────────────────────────────────────
MODEL_PATH   = Path(__file__).parent / "yolov8n.onnx"
CAM_WIDTH    = 960          # camera capture (smaller = faster preview)
CAM_HEIGHT   = 540
INPUT_SIZE   = 320          # network input — 320 is ~4x faster than 640 on Pi CPU
CONF_THRESH  = 0.35
IOU_THRESH   = 0.45
TARGET_FPS   = 30           # cap the inference loop
JPEG_QUALITY = 75
# ──────────────────────────────────────────────────────────

# Ultralytics palette (BGR) — hand-picked distinct colors, cycled per class.
_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]
COLORS = [_PALETTE[i % len(_PALETTE)] for i in range(80)]

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

if not MODEL_PATH.exists():
    print(f"ERROR: {MODEL_PATH.name} not found in {MODEL_PATH.parent}")
    print("Export with:")
    print(f"  python3 -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz={INPUT_SIZE}, simplify=True)\"")
    raise SystemExit(1)

print(f"Loading {MODEL_PATH.name} ...")
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4         # Pi 5 has 4 cores
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session    = ort.InferenceSession(str(MODEL_PATH), sess_options=sess_opts,
                                  providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print("Model ready.")

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT)}))
picam2.start()

frame_lock   = threading.Lock()
latest_frame = b""
frame_event  = threading.Event()


def preprocess(bgr):
    img = cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) * (1.0 / 255.0)
    img = np.transpose(img, (2, 0, 1))      # HWC → CHW
    return np.ascontiguousarray(img[None])  # [1,3,H,W]


def postprocess(outputs, orig_w, orig_h):
    preds = outputs[0][0]                                          # [4+nc, N]
    sx, sy = orig_w / INPUT_SIZE, orig_h / INPUT_SIZE

    confs     = preds[4:]                                          # [nc, N]
    class_ids = np.argmax(confs, axis=0)
    scores    = confs[class_ids, np.arange(confs.shape[1])]

    mask = scores >= CONF_THRESH
    if not mask.any():
        return []

    p  = preds[:4, mask]
    sc = scores[mask].tolist()
    ci = class_ids[mask]

    x1 = ((p[0] - p[2] / 2) * sx).astype(int)
    y1 = ((p[1] - p[3] / 2) * sy).astype(int)
    bw = (p[2] * sx).astype(int)
    bh = (p[3] * sy).astype(int)
    boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()

    keep = cv2.dnn.NMSBoxes(boxes, sc, CONF_THRESH, IOU_THRESH)
    return [(boxes[i][0], boxes[i][1],
             boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3],
             sc[i], int(ci[i])) for i in keep]


def inference_loop():
    global latest_frame
    interval = 1.0 / TARGET_FPS
    last = time.monotonic()
    fps  = 0.0
    while True:
        t0 = time.monotonic()

        raw   = picam2.capture_array()
        frame = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)

        blob = preprocess(frame)
        out  = session.run(None, {input_name: blob})
        dets = postprocess(out, CAM_WIDTH, CAM_HEIGHT)

        for x1, y1, x2, y2, conf, cls in dets:
            color = tuple(int(c) for c in COLORS[cls])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{CLASSES[cls]} {conf:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        now = time.monotonic()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        label = f"FPS: {fps:.1f}"
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
<title>YOLO Live</title>
<style>*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100vh}
img{max-width:100%;max-height:95vh;object-fit:contain}
p{color:#0f0;font-family:monospace;margin-top:6px;font-size:13px}</style></head>
<body><img src="/stream"><p>Pi 5 &mdash; YOLOv8n &mdash; onnxruntime CPU</p></body></html>"""


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


if __name__ == "__main__":
    PORT = 8081
    print(f"YOLO stream → http://{get_ip()}:{PORT}")
    print("Ctrl+C to stop.")
    srv = HTTPServer(("0.0.0.0", PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
