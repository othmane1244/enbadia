# ============================================================
# main.py — Application FastAPI principale
# Système de Surveillance Intelligente — ENSA Béni Mellal
# Routes REST + WebSocket + gestion du cycle de vie
# ============================================================

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import FrameData, Alert, ProcessFrameResponse
from services import analyze_behavior
from database import manager, insert_alert, broadcast_alert, get_recent_alerts, get_local_buffer

# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# CYCLE DE VIE DE L'APPLICATION
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / Shutdown de l'application."""
    logger.info("🚀 Surveillance API démarrée")
    logger.info("   Endpoints disponibles :")
    logger.info("     POST /process_frame/    — recevoir détections")
    logger.info("     GET  /alerts/           — historique alertes")
    logger.info("     GET  /health/           — état du serveur")
    logger.info("     WS   /ws/alerts         — stream temps réel")
    yield
    logger.info("🛑 Surveillance API arrêtée")


# ------------------------------------------------------------
# INITIALISATION FASTAPI
# ------------------------------------------------------------
app = FastAPI(
    title="Surveillance IA — API Backend",
    description=(
        "API de surveillance intelligente par caméra.\n"
        "Reçoit les détections YOLO11n du RPi 5 + Hailo-8, "
        "analyse les comportements et génère des alertes en temps réel."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — autorise Next.js dashboard (Vercel) et dev local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compteur de frames traitées (stats)
_stats = {"frames_processed": 0, "alerts_total": 0, "start_time": time.time()}


# ------------------------------------------------------------
# ROUTES REST
# ------------------------------------------------------------

@app.get("/health/", tags=["Monitoring"])
async def health_check():
    """
    Vérifie que l'API est opérationnelle.
    Utilisé par le dashboard pour afficher l'état du RPi 5.
    """
    uptime = round(time.time() - _stats["start_time"], 1)
    return {
        "status":           "ok",
        "uptime_seconds":   uptime,
        "frames_processed": _stats["frames_processed"],
        "alerts_total":     _stats["alerts_total"],
        "ws_clients":       len(manager.active_connections),
    }


@app.post("/process_frame/", response_model=ProcessFrameResponse, tags=["Pipeline"])
async def process_frame(frame_data: FrameData):
    """
    Endpoint principal du pipeline de surveillance.

    Reçoit les détections d'un frame vidéo (depuis RPi 5 ou simulateur),
    analyse les comportements, génère les alertes et les pousse :
      1. Dans Supabase PostgreSQL (persistance)
      2. Via WebSocket aux clients dashboard connectés

    Simule le pipeline Edge-to-Cloud du document Phase 3.
    """
    t_start = time.perf_counter()

    # --- Analyse comportementale ---
    alerts: list[Alert] = analyze_behavior(frame_data)

    # --- Persistance + broadcast pour chaque alerte ---
    for alert in alerts:
        await insert_alert(alert)
        await broadcast_alert(alert)

    # --- Mise à jour des stats ---
    _stats["frames_processed"] += 1
    _stats["alerts_total"] += len(alerts)

    processing_ms = round((time.perf_counter() - t_start) * 1000, 2)

    logger.info(
        f"Frame {frame_data.frame_id} | cam={frame_data.camera_id} | "
        f"{len(frame_data.detections)} détections | "
        f"{len(alerts)} alertes | {processing_ms}ms"
    )

    return ProcessFrameResponse(
        frame_id            = frame_data.frame_id,
        camera_id           = frame_data.camera_id,
        detections_count    = len(frame_data.detections),
        alerts_generated    = alerts,
        processing_time_ms  = processing_ms,
    )


@app.get("/alerts/", tags=["Alertes"])
async def get_alerts(limit: int = 50):
    """
    Retourne les dernières alertes (Supabase ou buffer local).
    Utilisé par le dashboard pour l'historique.
    """
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit doit être entre 1 et 200")
    alerts = await get_recent_alerts(limit=limit)
    return {"count": len(alerts), "alerts": alerts}


@app.get("/alerts/buffer/", tags=["Debug"])
async def get_local_alerts():
    """
    Expose le buffer local d'alertes (mode simulation sans Supabase).
    Utile pour tester sans connexion cloud.
    """
    buf = get_local_buffer()
    return {"count": len(buf), "alerts": buf}


@app.delete("/alerts/buffer/", tags=["Debug"])
async def clear_local_buffer():
    """Vide le buffer local d'alertes (debug uniquement)."""
    from database import _local_alert_buffer
    count = len(_local_alert_buffer)
    _local_alert_buffer.clear()
    return {"cleared": count}


@app.get("/stats/", tags=["Monitoring"])
async def get_stats():
    """Statistiques de traitement en temps réel."""
    uptime = time.time() - _stats["start_time"]
    fps_avg = round(_stats["frames_processed"] / max(uptime, 1), 2)
    return {
        **_stats,
        "uptime_seconds": round(uptime, 1),
        "avg_fps":        fps_avg,
        "ws_clients":     len(manager.active_connections),
    }


# ------------------------------------------------------------
# WEBSOCKET — Stream alertes temps réel vers le dashboard
# ------------------------------------------------------------

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket pour le dashboard Next.js.
    Le client se connecte une fois et reçoit toutes les nouvelles
    alertes en temps réel sans polling.

    Simule la fonctionnalité Supabase Realtime côté serveur.
    """
    await manager.connect(websocket)
    logger.info(f"📡 Nouveau client WS connecté")

    # Message de bienvenue avec l'historique récent
    recent = await get_recent_alerts(limit=10)
    await websocket.send_json({
        "event":   "connected",
        "message": "Connecté au flux d'alertes en temps réel",
        "history": recent,
    })

    try:
        while True:
            # Maintient la connexion ouverte
            # Les alertes sont envoyées via manager.broadcast() dans process_frame()
            data = await websocket.receive_text()

            # Gestion des messages client (ex: marquer alerte comme résolue)
            if data == "ping":
                await websocket.send_json({"event": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("📡 Client WS déconnecté")