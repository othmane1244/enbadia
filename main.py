# ============================================================
# main.py — Application FastAPI principale
# Système de Surveillance Intelligente — ENSA Béni Mellal
# Routes REST + WebSocket + gestion du cycle de vie
# ============================================================

import time
import logging
import cv2
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import FrameData, Alert, ProcessFrameResponse, ZoneCreate, VideoFrame, Zone
from services import analyze_behavior
from database import manager, insert_alert, insert_zone, broadcast_alert, get_recent_alerts, get_local_buffer, fetch_zones
from watchdog import start_watchdog_task, stop_watchdog_task, get_watchdog_status
from report_generator import start_report_scheduler, stop_report_scheduler

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
# MANAGERS (alertes + vidéo)
# ------------------------------------------------------------
class ConnectionManager:
    """Gère les connexions WebSocket pour un type de flux."""
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, data: dict):
        """Envoie un message à tous les clients connectés."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Erreur broadcast : {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


# Video recording state
video_manager = ConnectionManager()  # Pour /ws/video
_video_writer = None
_video_output_path = None
_video_codec = cv2.VideoWriter_fourcc(*'mp4v')
_video_fps = 30
_video_frame_size = (1280, 720)


# ------------------------------------------------------------
# CYCLE DE VIE DE L'APPLICATION
# ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / Shutdown de l'application."""
    logger.info("🚀 Surveillance API démarrée")
    logger.info("   Endpoints disponibles :")
    logger.info("     POST /process_frame/    — recevoir détections")
    logger.info("     POST /video/frame       — recevoir frames vidéo")
    logger.info("     GET  /alerts/           — historique alertes")
    logger.info("     GET  /health/           — état du serveur")
    logger.info("     WS   /ws/alerts         — stream alertes temps réel")
    logger.info("     WS   /ws/video          — stream vidéo temps réel")
    
    # Charger les zones d'intrusion au démarrage
    logger.info("📍 Chargement des zones d'intrusion depuis Supabase...")
    zones_data = await fetch_zones()
    
    # Convertir au format Zone
    zones = []
    for z in zones_data:
        zone = Zone(
            zone_id=z.get("zone_id"),
            camera_id=z.get("camera_id", ""),
            name=z.get("name", "Zone Interdite"),
            points=z.get("points", []),
            active=True
        )
        zones.append(zone)
    
    app.state.zones = zones
    if zones:
        logger.info(f"✅ {len(zones)} zone(s) chargée(s) avec succès")
        for z in zones:
            logger.info(f"   - {z.name} ({z.camera_id})")
    else:
        logger.warning("⚠️  Aucune zone active trouvée dans Supabase")
    
    app.state.watchdog_task = start_watchdog_task()
    logger.info("🛡️ Watchdog lancé en arrière-plan")
    app.state.report_scheduler = start_report_scheduler()
    logger.info("🗓️ Scheduler de rapport lancé en arrière-plan")
    yield
    await stop_report_scheduler()
    await stop_watchdog_task()
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

    Utilise les zones d'intrusion chargées depuis Supabase au démarrage.
    """
    t_start = time.perf_counter()

    # --- Analyse comportementale avec zones dynamiques ---
    zones = getattr(app.state, 'zones', [])
    alerts: list[Alert] = analyze_behavior(frame_data, zones=zones)

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


@app.post("/zones/", tags=["Zones"])
async def create_zone(zone: ZoneCreate):
    """Crée une zone polygonale dans Supabase depuis le dashboard."""
    if not zone.camera_id.strip():
        raise HTTPException(status_code=400, detail="camera_id est requis")
    if len(zone.points) < 3:
        raise HTTPException(status_code=400, detail="Une zone doit contenir au moins 3 points")

    saved = await insert_zone(zone)
    if not saved:
        raise HTTPException(status_code=500, detail="Impossible d'enregistrer la zone")

    return {"ok": True, "camera_id": zone.camera_id, "points": len(zone.points)}


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


@app.get("/watchdog/status/", tags=["Monitoring"])
async def watchdog_status():
    """Retourne l'état système et du watchdog pour la page System."""
    return get_watchdog_status()


# ------------------------------------------------------------
# VIDEO — Streaming et enregistrement
# -------- -----------------------------------------------

@app.post("/video/frame", tags=["Video"])
async def receive_video_frame(frame: VideoFrame):
    """
    Reçoit une frame vidéo en JPEG base64 depuis le simulateur/backend.
    La broadcast à tous les clients WebSocket /ws/video.
    """
    await video_manager.broadcast({
        "event": "frame",
        "frame_id": frame.frame_id,
        "camera_id": frame.camera_id,
        "data": frame.data,  # Base64-encoded JPEG
    })
    return {"ok": True}


@app.post("/video/record/start/", tags=["Video"])
async def start_video_recording():
    """
    Démarre l'enregistrement vidéo via cv2.VideoWriter.
    Retourne le chemin du fichier sauvegardé.
    """
    global _video_writer, _video_output_path
    
    if _video_writer is not None:
        raise HTTPException(status_code=400, detail="Enregistrement déjà en cours")
    
    # Créer un dossier recordings s'il n'existe pas
    recordings_dir = Path("recordings")
    recordings_dir.mkdir(exist_ok=True)
    
    # Générer un nom de fichier avec timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _video_output_path = str(recordings_dir / f"video_{timestamp}.mp4")
    
    # Créer le VideoWriter
    _video_writer = cv2.VideoWriter(
        _video_output_path,
        _video_codec,
        _video_fps,
        _video_frame_size
    )
    
    if not _video_writer.isOpened():
        _video_writer = None
        raise HTTPException(status_code=500, detail="Impossible de démarrer l'enregistrement")
    
    logger.info(f"🎥 Enregistrement vidéo démarré : {_video_output_path}")
    return {"ok": True, "output_path": _video_output_path}


@app.post("/video/record/stop/", tags=["Video"])
async def stop_video_recording():
    """
    Arrête l'enregistrement vidéo et retourne le nom du fichier sauvegardé.
    """
    global _video_writer, _video_output_path
    
    if _video_writer is None:
        raise HTTPException(status_code=400, detail="Aucun enregistrement en cours")
    
    _video_writer.release()
    _video_writer = None
    
    output_name = Path(_video_output_path).name if _video_output_path else "unknown"
    logger.info(f"🎬 Enregistrement vidéo arrêté : {output_name}")
    
    return {"ok": True, "filename": output_name}


# -------- -----------------------------------------------
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


@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """
    WebSocket pour le stream vidéo en temps réel vers le dashboard.
    Le client reçoit chaque frame en JPEG base64 en temps réel.
    """
    await video_manager.connect(websocket)
    logger.info(f"📹 Nouveau client vidéo connecté ({len(video_manager.active_connections)} total)")

    # Message de bienvenue
    await websocket.send_json({
        "event": "connected",
        "message": "Connecté au flux vidéo en temps réel",
    })

    try:
        while True:
            # Maintient la connexion ouverte
            # Les frames sont envoyées via video_manager.broadcast() dans /video/frame
            data = await websocket.receive_text()

            # Gestion des messages client (ex: heartbeat)
            if data == "ping":
                await websocket.send_json({"event": "pong"})

    except WebSocketDisconnect:
        video_manager.disconnect(websocket)
        logger.info(f"📹 Client vidéo déconnecté ({len(video_manager.active_connections)} restants)")