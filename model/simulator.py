# ============================================================
# simulator.py — Simulateur pipeline RPi 5 avec YOLO + vidéo
# Système de Surveillance Intelligente — ENSA Béni Mellal
#
# Capture webcam → YOLO11n → Annotations → POST /video/frame
# ============================================================

import cv2
import numpy as np
import asyncio
import httpx
import base64
import logging
from datetime import datetime, timezone
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# CONFIG
YOLO_MODEL = "yolo11n.pt"
API_URL = "http://127.0.0.1:8000/process_frame/"
VIDEO_FRAME_URL = "http://127.0.0.1:8000/video/frame"
CAMERA_ID = "cam_01_simulation"
WEBCAM_ID = 0
DISPLAY_WINDOW = True
SEND_EVERY_N = 1

# Couleurs YOLO par classe
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)

# État zones interdites (fictif - à mettre à jour depuis le backend)
FORBIDDEN_ZONES = [
    {"points": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.3), (0.1, 0.3)], "name": "Zone Entrée"}
]


async def send_video_frame(client: httpx.AsyncClient, frame: cv2.Mat, frame_id: int) -> bool:
    """Envoie une frame JPEG base64 au backend."""
    try:
        ret, jpeg_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            logger.error("Impossible d'encoder la frame en JPEG")
            return False
        
        base64_frame = base64.b64encode(jpeg_data.tobytes()).decode('utf-8')
        
        payload = {
            "frame_id": frame_id,
            "camera_id": CAMERA_ID,
            "data": base64_frame,
        }
        resp = await client.post(VIDEO_FRAME_URL, json=payload, timeout=1.0)
        return resp.status_code == 200
    except Exception as e:
        logger.error(f"Erreur envoi frame : {e}")
        return False


async def send_detections_to_api(client: httpx.AsyncClient, frame_data: dict) -> int:
    """Envoie les détections à l'API pour analyse comportementale."""
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
    except Exception as e:
        logger.error(f"Erreur envoi détections : {e}")
        return 0


def draw_annotations(frame, results, frame_id):
    """Dessine les détections YOLO et les zones interdites sur le frame."""
    h, w = frame.shape[:2]
    
    # Dessiner les zones interdites
    for zone in FORBIDDEN_ZONES:
        points = zone["points"]
        pts = np.array(
            [(int(p[0] * w), int(p[1] * h)) for p in points],
            dtype=np.int32
        )
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        
        # Remplissage semi-transparent
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        
        # Label zone
        center_x = int(np.mean([p[0] for p in points]) * w)
        center_y = int(np.mean([p[1] for p in points]) * h)
        cv2.putText(
            frame, f"🚫 {zone['name']}", (center_x - 50, center_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
    
    # Dessiner les détections
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = results[0].names[cls_id]
            
            # Couleur par classe
            color = tuple(int(c) for c in COLORS[cls_id])
            
            # BBox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label avec confiance
            label = f"{class_name} {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
    
    # Overlay info
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    
    cv2.putText(frame, f"Frame #{frame_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if results and len(results) > 0 and results[0].boxes is not None:
        n_det = len(results[0].boxes)
        cv2.putText(frame, f"Détections: {n_det}", (250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    
    return frame


async def main():
    logger.info("=== Simulateur YOLO + Vidéo ===")
    logger.info(f"  Modèle YOLO: {YOLO_MODEL}")
    logger.info(f"  API Détections: {API_URL}")
    logger.info(f"  Caméra: {CAMERA_ID}")
    logger.info("  [Q] pour quitter\n")

    # Charger le modèle YOLO
    try:
        model = YOLO(YOLO_MODEL)
        logger.info(f"✅ Modèle YOLO chargé: {YOLO_MODEL}")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle YOLO: {e}")
        logger.info("  Utilisation du mode vidéo sans détections...")
        model = None

    # Ouvrir la webcam
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        logger.error(f"❌ Webcam introuvable (ID={WEBCAM_ID})")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logger.info(f"✅ Webcam ouverte : {int(cap.get(3))}x{int(cap.get(4))}")

    frame_id = 0
    frames_sent = 0
    api_alerts = 0

    async with httpx.AsyncClient() as client:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            orig_h, orig_w = frame.shape[:2]

            # Inférence YOLO (si modèle disponible)
            results = None
            detections = []
            if model is not None:
                try:
                    results = model(frame, verbose=False)
                    
                    # Extraire les détections au format API
                    if results and len(results) > 0 and results[0].boxes is not None:
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            detections.append({
                                "track_id": None,
                                "class_id": cls_id,
                                "class_name": results[0].names[cls_id],
                                "confidence": confidence,
                                "bbox": {
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2,
                                }
                            })
                except Exception as e:
                    logger.warning(f"⚠️ Erreur inférence YOLO: {e}")

            # Dessiner les annotations
            display = draw_annotations(frame.copy(), results, frame_id)

            # Envoyer la frame annotée
            success = await send_video_frame(client, display, frame_id)
            if success:
                frames_sent += 1
                if frames_sent % 30 == 0:
                    logger.info(f"📹 {frames_sent} frames envoyées")

            # Envoyer les détections à l'API (1 sur SEND_EVERY_N)
            if frame_id % SEND_EVERY_N == 0 and detections:
                frame_payload = {
                    "camera_id": CAMERA_ID,
                    "frame_id": frame_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "fps": 30.0,
                    "detections": detections,
                }
                n_alerts = await send_detections_to_api(client, frame_payload)
                api_alerts += n_alerts

            # Afficher localement
            if DISPLAY_WINDOW:
                cv2.imshow("Simulateur YOLO", display)

            frame_id += 1

            # Attendre 33ms = ~30 FPS
            await asyncio.sleep(0.033)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"\n✅ Simulation terminée — {frame_id} frames, {frames_sent} envoyées, {api_alerts} alertes")


if __name__ == "__main__":
    asyncio.run(main())
