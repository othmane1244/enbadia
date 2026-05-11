# ============================================================
# database.py — Couche Supabase + WebSocket broadcast
# Système de Surveillance Intelligente — ENSA Béni Mellal
# Gère la persistance PostgreSQL et les notifications temps réel
# ============================================================

from dotenv import load_dotenv
load_dotenv()  # ← DOIT être en tout premier, avant os.getenv()

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Set, Dict, Optional

from fastapi import WebSocket
from models import Alert, ZoneCreate
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# GESTIONNAIRE DE CONNEXIONS WEBSOCKET
# Garde la liste des clients dashboard connectés
# ------------------------------------------------------------

class ConnectionManager:
    """Gère les connexions WebSocket actives vers le dashboard."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.debug(f"WS connecté — total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.debug(f"WS déconnecté — total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Envoie un message JSON à tous les clients connectés."""
        if not self.active_connections:
            return

        dead = set()
        payload = json.dumps(message, default=str)

        for ws in self.active_connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)

        # Nettoyage des connexions mortes
        self.active_connections -= dead

# Instance globale partagée dans toute l'application
manager = ConnectionManager()


# ------------------------------------------------------------
# CLIENT SUPABASE
# ------------------------------------------------------------

def get_supabase_client():
    """
    Retourne le client Supabase si les variables d'env sont définies.
    En mode simulation (dev PC), retourne None et log un warning.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        logger.warning(
            "⚠️  SUPABASE_URL / SUPABASE_SERVICE_KEY non définis — "
            "mode simulation activé (pas de persistance cloud)"
        )
        return None

    try:
        from supabase import create_client
        client = create_client(url, key)
        logger.debug("✅ Supabase connecté")
        return client
    except ImportError:
        logger.warning("⚠️  Package supabase non installé — mode simulation")
        return None
    except Exception as e:
        logger.error(f"❌ Erreur connexion Supabase : {e}")
        return None


# Initialisation au démarrage
_supabase = get_supabase_client()

# Buffer local pour les alertes quand Supabase est absent
_local_alert_buffer: list[dict] = []


# ------------------------------------------------------------
# Telegram bot + cooldown pour prévenir les opérateurs
# ------------------------------------------------------------
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))
_last_telegram_sent: Dict[str, datetime] = {}


async def _should_send_telegram_for_alert(alert: Alert) -> bool:
    key = None
    try:
        if alert.detection_info and len(alert.detection_info) > 0:
            track = alert.detection_info[0]
            track_id = getattr(track, "track_id", None)
            if track_id is not None:
                key = f"track:{track_id}"
    except Exception:
        key = None

    if key is None:
        key = f"type:{alert.alert_type}:cam:{alert.camera_id}"

    last = _last_telegram_sent.get(key)
    if last is None:
        return True
    return (datetime.utcnow() - last) > timedelta(seconds=ALERT_COOLDOWN_SECONDS)


async def send_telegram_alert(alert: Alert) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.debug("Telegram non configuré (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID manquants)")
        return False

    try:
        ok = await _should_send_telegram_for_alert(alert)
        if not ok:
            logger.debug("🔕 Telegram cooldown actif — notification ignorée")
            return False

        when = alert.timestamp.isoformat()
        confidence = f"{alert.confidence_score:.2f}"
        dashboard_url = os.getenv("DASHBOARD_URL", "http://localhost:3000")
        link = f"{dashboard_url}/alerts/{alert.id}"
        text = (
            f"ALERTE: {alert.alert_type}\n"
            f"Caméra: {alert.camera_id}\n"
            f"Heure: {when}\n"
            f"Confiance: {confidence}\n"
            f"Détail: {alert.description}\n"
            f"Dash: {link}"
        )

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        import httpx
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, timeout=5.0)
            if r.status_code == 200:
                if alert.detection_info and len(alert.detection_info) > 0:
                    track = alert.detection_info[0]
                    track_id = getattr(track, "track_id", None)
                    key = f"track:{track_id}" if track_id is not None else f"type:{alert.alert_type}:cam:{alert.camera_id}"
                else:
                    key = f"type:{alert.alert_type}:cam:{alert.camera_id}"
                _last_telegram_sent[key] = datetime.utcnow()
                logger.debug("✅ Telegram envoyé")
                return True

            logger.error(f"❌ Erreur Telegram {r.status_code}: {r.text}")
            return False

    except Exception as e:
        logger.error(f"❌ Exception Telegram: {e}")
        return False


# ------------------------------------------------------------
# FONCTIONS PRINCIPALES
# ------------------------------------------------------------

async def insert_alert(alert: Alert) -> bool:
    """
    Insère une alerte dans Supabase PostgreSQL.
    Si Supabase indisponible, stocke en mémoire locale.
    Retourne True si succès, False sinon.
    """
    alert_dict = {
        "id":               alert.id,
        "created_at":       alert.timestamp.isoformat(),
        "camera_id":        alert.camera_id,
        "alert_type":       alert.alert_type,
        "description":      alert.description,
        "confidence_score": alert.confidence_score,
        "detection_info":   [d.model_dump() for d in alert.detection_info],
        "is_resolved":      alert.is_resolved,
    }

    if _supabase is None:
        # Mode simulation — stockage local
        _local_alert_buffer.append(alert_dict)
        logger.debug(
            f"[SIMULATION] Alerte sauvegardée localement "
            f"({len(_local_alert_buffer)} total) : {alert.alert_type}"
        )
        return True

    try:
        # Exécution en thread pour ne pas bloquer la boucle async
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: _supabase.table("alerts").insert(alert_dict).execute()
        )
        logger.debug(f"✅ Alerte insérée Supabase : {alert.alert_type} [{alert.id[:8]}]")
        return True

    except Exception as e:
        logger.error(f"❌ Erreur insert Supabase : {e}")
        _local_alert_buffer.append(alert_dict)
        return False


async def insert_zone(zone: ZoneCreate) -> bool:
    """Insère une zone polygonale dans Supabase."""
    zone_dict = {
        "camera_id": zone.camera_id,
        "zone_name": zone.zone_name,
        "points": zone.points,
        "active": zone.active,
        "updated_at": zone.updated_at.isoformat(),
    }

    if _supabase is None:
        logger.warning("⚠️  Supabase indisponible — zone non persistée")
        return False

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: _supabase.table("zones").insert(zone_dict).execute()
        )
        logger.debug(f"✅ Zone insérée Supabase : {zone.camera_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur insert zone Supabase : {e}")
        return False


async def fetch_zones(camera_id: str = None) -> list[dict]:
    """
    Récupère les zones actives depuis Supabase.
    Format retourné : [{"name": str, "points": [(x, y), ...], "camera_id": str}, ...]
    Chaque point est en coordonnées normalisées (0.0-1.0).
    Si camera_id est None, retourne toutes les zones actives.
    """
    if _supabase is None:
        logger.warning("⚠️  Supabase indisponible — aucune zone chargée")
        return []

    try:
        loop = asyncio.get_event_loop()
        
        def query_zones():
            q = _supabase.table("zones").select("*").eq("active", True)
            if camera_id:
                q = q.eq("camera_id", camera_id)
            return q.execute()
        
        response = await loop.run_in_executor(None, query_zones)
        zones = response.data if response.data else []
        
        formatted = []
        for z in zones:
            formatted.append({
                "name": z.get("zone_name", "Zone"),
                "points": z.get("points", []),
                "camera_id": z.get("camera_id", ""),
                "zone_id": z.get("id"),
            })
        
        logger.debug(f"✅ {len(formatted)} zone(s) chargée(s) de Supabase")
        return formatted
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération zones Supabase : {e}")
        return []


async def broadcast_alert(alert: Alert):
    """
    Envoie l'alerte à tous les clients WebSocket connectés.
    Appelé juste après insert_alert().
    """
    payload = {
        "event":     "new_alert",
        "alert":     alert.model_dump(),
        "timestamp": datetime.utcnow().isoformat(),
    }
    await manager.broadcast(payload)
    logger.debug(
        f"📡 Alerte broadcastée à {len(manager.active_connections)} client(s) WS"
    )
    try:
        await send_telegram_alert(alert)
    except Exception as e:
        logger.error(f"❌ Erreur envoi Telegram après broadcast: {e}")


async def get_recent_alerts(limit: int = 50) -> list[dict]:
    """
    Récupère les dernières alertes (Supabase ou buffer local).
    Utilisé par l'endpoint GET /alerts/
    """
    if _supabase is None:
        return list(reversed(_local_alert_buffer[-limit:]))

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: (
                _supabase.table("alerts")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
        )
        return result.data or []
    except Exception as e:
        logger.error(f"❌ Erreur lecture Supabase : {e}")
        return list(reversed(_local_alert_buffer[-limit:]))


def get_local_buffer() -> list[dict]:
    """Expose le buffer local pour debug / tests."""
    return _local_alert_buffer.copy()