# ============================================================
# simulator.py — Simulateur pipeline RPi 5 sur PC
# Système de Surveillance Intelligente — ENSA Béni Mellal
#
# Remplace libcamera + HailoRT sur le PC de développement.
# Capture webcam → ONNX → POST /process_frame/ → alertes
#
# Sur RPi 5 réel : remplacer ce fichier par le pipeline
# libcamera → HailoRT → FastAPI client
# ============================================================

import cv2
import numpy as np
import onnxruntime as ort
import httpx
import asyncio
import time
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
ONNX_MODEL      = "yolo11n.onnx"
API_URL         = "http://127.0.0.1:8000/process_frame/"
CAMERA_ID       = "cam_01_simulation"
WEBCAM_ID       = 0
INPUT_SIZE      = 640
CONF_THRESH     = 0.45
IOU_THRESH      = 0.45
SEND_EVERY_N    = 1       # Envoyer 1 frame sur N à l'API (réduire charge réseau)
DISPLAY_WINDOW  = True    # Afficher la fenêtre de prévisualisation

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
# MODÈLE ONNX
# ------------------------------------------------------------
def load_model():
    # DirectML : GPU Windows universel (RTX 5060, CUDA 13.x+, AMD, Intel)
    # Pas de dépendance à la version CUDA — fonctionne via DirectX 12
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(ONNX_MODEL, providers=providers)
    provider = session.get_providers()[0]
    logger.info(f"✅ ONNX chargé — Provider : {provider}")
    if provider == "CPUExecutionProvider":
        logger.warning("⚠️  GPU non détecté — inférence sur CPU (vérifier onnxruntime-directml)")
    return session


# ------------------------------------------------------------
# PRÉ-TRAITEMENT (identique au pipeline RPi 5)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# POST-TRAITEMENT + FORMATAGE JSON pour l'API
# ------------------------------------------------------------
def postprocess_to_api_format(output, ratio, pad_w, pad_h, orig_w, orig_h):
    """
    Convertit la sortie ONNX en format FrameData attendu par l'API.
    Retourne une liste de dicts Detection compatibles Pydantic.
    """
    predictions = output[0].T
    boxes, scores, class_ids = [], [], []

    for pred in predictions:
        cx, cy, bw, bh = pred[:4]
        class_probs = pred[4:]
        class_id = int(np.argmax(class_probs))
        confidence = float(class_probs[class_id])
        if confidence < CONF_THRESH:
            continue
        x1 = max(0, int((cx - bw / 2 - pad_w) / ratio))
        y1 = max(0, int((cy - bh / 2 - pad_h) / ratio))
        x2 = min(orig_w, int((cx + bw / 2 - pad_w) / ratio))
        y2 = min(orig_h, int((cy + bh / 2 - pad_h) / ratio))
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
            "track_id":   int(i),                    
            "class_id":   int(class_ids[i]),       
            "class_name": CLASSES[int(class_ids[i])],
            "confidence": round(float(scores[i]), 3), 
            "bbox": {
                "x1": int(x),                       
                "y1": int(y),                        
                "x2": int(x + w),                    
                "y2": int(y + h),                   
            }
        })
    return detections

def prepare_detections(results) -> list[dict]:
    """Convertit les résultats YOLO en format API avec types Python natifs."""
    detections = []
    
    for box in results.boxes:
        # ✅ Convertir int32 → int natif Python
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else None
        
        detection = {
            "track_id": track_id,
            "class_id": class_id,
            "class_name": results.names[class_id],
            "confidence": confidence,
            "bbox": {
                "x1": x1,       # ✅ int natif (pas int32)
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        }
        detections.append(detection)
    
    return detections

# ------------------------------------------------------------
# AFFICHAGE
# ------------------------------------------------------------
def draw_frame(frame, detections, fps, api_alerts_count, frame_id):
    for det in detections:
        bb = det["bbox"]
        x1, y1, x2, y2 = bb["x1"], bb["y1"], bb["x2"], bb["y2"]
        cls_id = det["class_id"]
        label  = f"{det['class_name']} {det['confidence']:.2f}"
        color  = tuple(int(c) for c in COLORS[cls_id])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Det: {len(detections)}", (120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, f"Alertes API: {api_alerts_count}", (230, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    cv2.putText(frame, f"Frame: {frame_id}", (430, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
    cv2.putText(frame, "[SIMULATION PC — Remplace libcamera + HailoRT sur RPi 5]",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    return frame


# ------------------------------------------------------------
# ENVOI À L'API (async)
# ------------------------------------------------------------
async def send_to_api(client: httpx.AsyncClient, frame_data: dict) -> int:
    """
    Envoie un FrameData JSON à POST /process_frame/.
    Retourne le nombre d'alertes générées.
    """
    try:
        resp = await client.post(API_URL, json=frame_data, timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            n = len(data.get("alerts_generated", []))
            if n > 0:
                for alert in data["alerts_generated"]:
                    logger.warning(
                        f"🚨 ALERTE [{alert['alert_type']}] "
                        f"{alert['description'][:60]} "
                        f"(conf={alert['confidence_score']:.2f})"
                    )
            return n
        else:
            logger.error(f"API erreur {resp.status_code}")
            return 0
    except httpx.ConnectError:
        logger.warning("⚠️  API non joignable — lance uvicorn d'abord !")
        return 0
    except Exception as e:
        logger.error(f"Erreur envoi API : {e}")
        return 0


# ------------------------------------------------------------
# BOUCLE PRINCIPALE
# ------------------------------------------------------------
async def main():
    logger.info("=== Simulateur Pipeline RPi 5 ===")
    logger.info(f"  ONNX     : {ONNX_MODEL}")
    logger.info(f"  API      : {API_URL}")
    logger.info(f"  Caméra   : {CAMERA_ID}")
    logger.info("  [Q] pour quitter\n")

    session    = load_model()
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        logger.error(f"❌ Webcam introuvable (ID={WEBCAM_ID})")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logger.info(f"✅ Webcam ouverte : {int(cap.get(3))}x{int(cap.get(4))}")

    fps            = 0.0
    frame_id       = 0
    api_alerts_acc = 0

    async with httpx.AsyncClient() as client:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            orig_h, orig_w = frame.shape[:2]
            t0 = time.perf_counter()

            # ── Pipeline inférence ──
            blob, ratio, pad_w, pad_h = preprocess(frame)
            outputs     = session.run(None, {input_name: blob})
            detections  = postprocess_to_api_format(
                outputs[0], ratio, pad_w, pad_h, orig_w, orig_h
            )

            fps = 0.9 * fps + 0.1 * (1.0 / (time.perf_counter() - t0 + 1e-6))

            # ── Envoi à l'API (1 frame sur SEND_EVERY_N) ──
            if frame_id % SEND_EVERY_N == 0 and detections:
                frame_payload = {
                    "camera_id":  CAMERA_ID,
                    "frame_id":   frame_id,
                    "timestamp":  datetime.now(timezone.utc).isoformat(),
                    "fps":        round(fps, 2),
                    "detections": detections,
                }
                n_alerts = await send_to_api(client, frame_payload)
                api_alerts_acc += n_alerts

            # ── Affichage ──
            if DISPLAY_WINDOW:
                display = draw_frame(frame, detections, fps, api_alerts_acc, frame_id)
                cv2.imshow("Simulateur — RPi5 + Hailo-8 (PC)", display)

            frame_id += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"\n✅ Simulation terminée — {frame_id} frames, {api_alerts_acc} alertes")


if __name__ == "__main__":
    asyncio.run(main())
    