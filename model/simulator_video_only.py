# ============================================================
# simulator_video_only.py — Simulateur vidéo uniquement
# Envoie les frames vidéo vers le WebSocket sans inférence ONNX
# ============================================================

import cv2
import asyncio
import httpx
import base64
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# CONFIG
API_URL = "http://127.0.0.1:8000/process_frame/"
VIDEO_FRAME_URL = "http://127.0.0.1:8000/video/frame"
CAMERA_ID = "cam_01_simulation"
WEBCAM_ID = 0
DISPLAY_WINDOW = True


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


async def main():
    logger.info("=== Simulateur Vidéo (sans ONNX) ===")
    logger.info(f"  API Video   : {VIDEO_FRAME_URL}")
    logger.info(f"  Caméra      : {CAMERA_ID}")
    logger.info("  [Q] pour quitter\n")

    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        logger.error(f"❌ Webcam introuvable (ID={WEBCAM_ID})")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logger.info(f"✅ Webcam ouverte : {int(cap.get(3))}x{int(cap.get(4))}")

    frame_id = 0
    frames_sent = 0

    async with httpx.AsyncClient() as client:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Ajouter un texte sur la frame pour identifer que c'est un frame
            cv2.putText(frame, f"Frame #{frame_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"[SIMULATEUR VIDEO]", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Envoyer la frame
            success = await send_video_frame(client, frame, frame_id)
            if success:
                frames_sent += 1
                if frames_sent % 30 == 0:
                    logger.info(f"📹 {frames_sent} frames envoyées")

            # Afficher localement
            if DISPLAY_WINDOW:
                cv2.imshow("Simulateur Vidéo", frame)

            frame_id += 1

            # Attendre 33ms = ~30 FPS
            await asyncio.sleep(0.033)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"\n✅ Simulation terminée — {frame_id} frames, {frames_sent} envoyées")


if __name__ == "__main__":
    asyncio.run(main())
