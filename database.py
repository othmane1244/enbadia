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
from datetime import datetime
from typing import Set

from fastapi import WebSocket
from models import Alert
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
        logger.info(f"WS connecté — total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WS déconnecté — total: {len(self.active_connections)}")

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
        logger.info("✅ Supabase connecté")
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
        logger.info(
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
        logger.info(f"✅ Alerte insérée Supabase : {alert.alert_type} [{alert.id[:8]}]")
        return True

    except Exception as e:
        logger.error(f"❌ Erreur insert Supabase : {e}")
        _local_alert_buffer.append(alert_dict)
        return False


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
    logger.info(
        f"📡 Alerte broadcastée à {len(manager.active_connections)} client(s) WS"
    )


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