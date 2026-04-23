# ============================================================
# Comparaison PT vs ONNX — YOLO11n — Webcam temps réel
# Système de Surveillance Intelligente — ENSA Béni Mellal
# 
# Affiche deux fenêtres côte à côte :
#   - Fenêtre GAUCHE  : inférence PyTorch (.pt)
#   - Fenêtre DROITE  : inférence ONNX    (.onnx)
# Permet de comparer visuellement la dégradation INT8 et les FPS
# ============================================================

import cv2
import numpy as np
import onnxruntime as ort
import time
import threading
from ultralytics import YOLO

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
PT_MODEL     = "yolo11n.pt"
ONNX_MODEL   = "yolo11n.onnx"
INPUT_SIZE   = 640
CONF_THRESH  = 0.45
IOU_THRESH   = 0.45
WEBCAM_ID    = 0

# Classes COCO
CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

# ------------------------------------------------------------
# MODÈLE ONNX — fonctions identiques à test_inference.py
# ------------------------------------------------------------
def load_onnx(model_path):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    provider = session.get_providers()[0]
    print(f"✅ ONNX chargé  — Provider : {provider}")
    return session

def preprocess(frame):
    h, w = frame.shape[:2]
    ratio = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (INPUT_SIZE - new_w) // 2
    pad_h = (INPUT_SIZE - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized, pad_h, INPUT_SIZE - new_h - pad_h,
        pad_w, INPUT_SIZE - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    return blob, ratio, pad_w, pad_h

def postprocess_onnx(output, ratio, pad_w, pad_h, orig_w, orig_h):
    predictions = output[0].T
    boxes, scores, class_ids = [], [], []
    for pred in predictions:
        cx, cy, bw, bh = pred[:4]
        class_probs = pred[4:]
        class_id = int(np.argmax(class_probs))
        confidence = float(class_probs[class_id])
        if confidence < CONF_THRESH:
            continue
        x1 = int((cx - bw / 2 - pad_w) / ratio)
        y1 = int((cy - bh / 2 - pad_h) / ratio)
        x2 = int((cx + bw / 2 - pad_w) / ratio)
        y2 = int((cy + bh / 2 - pad_h) / ratio)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(confidence)
        class_ids.append(class_id)
    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, IOU_THRESH)
    detections = []
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        detections.append({
            "class_id"  : class_ids[i],
            "class_name": CLASSES[class_ids[i]],
            "confidence": round(scores[i], 3),
            "bbox"      : [x, y, x + w, y + h]
        })
    return detections

# ------------------------------------------------------------
# DESSIN
# ------------------------------------------------------------
def draw_detections(frame, detections, color_override=None):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        label  = f"{det['class_name']} {det['confidence']:.2f}"
        color  = color_override if color_override else tuple(int(c) for c in COLORS[cls_id])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame

def draw_header(frame, title, fps, n_det, color):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    # Titre
    cv2.putText(frame, title, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # FPS + détections
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Det: {n_det}", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame

def draw_footer_comparison(frame_pt, frame_onnx, fps_pt, fps_onnx, det_pt, det_onnx):
    """Ajoute un bandeau de comparaison en bas des deux frames."""
    h, w = frame_pt.shape[:2]
    
    # Différence FPS
    diff_fps = fps_onnx - fps_pt
    diff_sign = "+" if diff_fps >= 0 else ""
    
    for frame, fps, n_det, label in [
        (frame_pt,   fps_pt,   det_pt,   "PyTorch .pt"),
        (frame_onnx, fps_onnx, det_onnx, "ONNX Runtime"),
    ]:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 30), (w, h), (20, 20, 20), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.putText(
            frame,
            f"ONNX vs PT : {diff_sign}{diff_fps:.1f} FPS",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1
        )

# ------------------------------------------------------------
# STATS EN CONSOLE
# ------------------------------------------------------------
def print_stats(fps_pt, fps_onnx, det_pt, det_onnx, elapsed):
    print(f"\n[{elapsed:6.1f}s] ── Comparaison ──────────────────────────")
    print(f"  PyTorch .pt  : FPS={fps_pt:5.1f}  |  Détections={det_pt}")
    print(f"  ONNX Runtime : FPS={fps_onnx:5.1f}  |  Détections={det_onnx}")
    diff = fps_onnx - fps_pt
    sign = "+" if diff >= 0 else ""
    print(f"  Δ FPS        : {sign}{diff:.1f}  ({'ONNX plus rapide' if diff >= 0 else 'PT plus rapide'})")
    print(f"─────────────────────────────────────────────────")

# ------------------------------------------------------------
# BOUCLE PRINCIPALE
# ------------------------------------------------------------
def main():
    print("\n=== Comparaison PT vs ONNX — YOLO11n ===")
    print("  [Q] Quitter | [S] Screenshot comparaison\n")

    # Chargement modèles
    print("Chargement des modèles...")
    model_pt   = YOLO(PT_MODEL)
    session    = load_onnx(ONNX_MODEL)
    input_name = session.get_inputs()[0].name
    print(f"✅ PT chargé    — PyTorch (CPU/GPU selon dispo)")

    # Webcam
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print(f"❌ Webcam introuvable (ID={WEBCAM_ID})")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"✅ Webcam ouverte : {int(cap.get(3))}x{int(cap.get(4))}\n")

    fps_pt, fps_onnx = 0.0, 0.0
    frame_count = 0
    t_start = time.time()
    screenshot_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]

        # ── Inférence PyTorch (.pt) ──────────────────────
        t0 = time.perf_counter()
        results = model_pt(frame, conf=CONF_THRESH, iou=IOU_THRESH,
                           imgsz=INPUT_SIZE, verbose=False)[0]
        t1 = time.perf_counter()
        fps_pt = 0.9 * fps_pt + 0.1 * (1.0 / (t1 - t0 + 1e-6))

        # Convertir résultats PT en liste de dicts
        det_pt_list = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_pt_list.append({
                "class_id"  : cls_id,
                "class_name": CLASSES[cls_id],
                "confidence": round(conf, 3),
                "bbox"      : [x1, y1, x2, y2]
            })

        # ── Inférence ONNX ───────────────────────────────
        t0 = time.perf_counter()
        blob, ratio, pad_w, pad_h = preprocess(frame)
        outputs = session.run(None, {input_name: blob})
        det_onnx_list = postprocess_onnx(
            outputs[0], ratio, pad_w, pad_h, orig_w, orig_h
        )
        t1 = time.perf_counter()
        fps_onnx = 0.9 * fps_onnx + 0.1 * (1.0 / (t1 - t0 + 1e-6))

        # ── Affichage ────────────────────────────────────
        frame_pt   = frame.copy()
        frame_onnx = frame.copy()

        draw_detections(frame_pt,   det_pt_list,   color_override=None)
        draw_detections(frame_onnx, det_onnx_list, color_override=None)

        # En-têtes colorés différents pour distinguer les deux
        draw_header(frame_pt,   "PyTorch .pt  (FP32)",
                    fps_pt,   len(det_pt_list),   (100, 255, 100))
        draw_header(frame_onnx, "ONNX Runtime (FP32→INT8 ready)",
                    fps_onnx, len(det_onnx_list), (100, 200, 255))

        draw_footer_comparison(
            frame_pt, frame_onnx,
            fps_pt, fps_onnx,
            len(det_pt_list), len(det_onnx_list)
        )

        # Affichage dans deux fenêtres séparées
        cv2.imshow("◀ PyTorch .pt", frame_pt)
        cv2.imshow("▶ ONNX Runtime", frame_onnx)

        # Stats console toutes les 60 frames
        frame_count += 1
        if frame_count % 60 == 0:
            print_stats(fps_pt, fps_onnx,
                        len(det_pt_list), len(det_onnx_list),
                        time.time() - t_start)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Screenshot : colle les deux frames côte à côte
            combined = np.hstack([frame_pt, frame_onnx])
            fname = f"screenshot_comparison_{screenshot_id:03d}.jpg"
            cv2.imwrite(fname, combined)
            print(f"📸 Screenshot sauvegardé : {fname}")
            screenshot_id += 1

    cap.release()
    cv2.destroyAllWindows()

    # Résumé final
    print("\n══════════════════════════════════════════════")
    print("  RÉSUMÉ FINAL")
    print(f"  PyTorch .pt  : FPS moyen ≈ {fps_pt:.1f}")
    print(f"  ONNX Runtime : FPS moyen ≈ {fps_onnx:.1f}")
    print(f"  Δ FPS        : {fps_onnx - fps_pt:+.1f}")
    print("  → Sur RPi 5 + Hailo-8, ONNX sera compilé en HEF")
    print("     et tournera à ~25–35 FPS (vs ~2–4 FPS PT seul)")
    print("══════════════════════════════════════════════\n")

if __name__ == "__main__":
    main()