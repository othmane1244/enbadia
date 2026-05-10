# watchdog.py — Moniteur CPU / Disk + notifications Telegram + keepalive Supabase
# Vérifie l'utilisation CPU et disque toutes les 30s, envoie
# une alerte Telegram si un seuil critique est dépassé et ping
# Supabase toutes les 72 heures pour éviter la pause du plan gratuit.

import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

try:
    import psutil
except ImportError:
    psutil = None

try:
    import httpx
except ImportError:
    httpx = None

try:
    from supabase import create_client
except ImportError:
    create_client = None

CHECK_INTERVAL = int(os.getenv("WATCHDOG_INTERVAL_SECONDS", "30"))
CPU_THRESHOLD = float(os.getenv("WATCHDOG_CPU_THRESHOLD", "90"))
DISK_THRESHOLD = float(os.getenv("WATCHDOG_DISK_THRESHOLD", "90"))
SUPABASE_KEEPALIVE_INTERVAL_SECONDS = int(os.getenv("WATCHDOG_SUPABASE_INTERVAL_SECONDS", str(72 * 3600)))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

_watchdog_task: Optional[asyncio.Task] = None


def _get_disk_usage_path() -> str:
    """Retourne un chemin de volume compatible Windows/Linux pour psutil.disk_usage."""
    configured_path = os.getenv("WATCHDOG_DISK_PATH")
    if configured_path:
        return configured_path

    if os.name == "nt":
        drive = os.path.splitdrive(os.getcwd())[0]
        return drive + "\\" if drive else "C:\\"

    return "/"

async def _send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram non configuré pour watchdog")
        return False
    if httpx is None:
        logger.warning("httpx non installé — watchdog ne peut pas envoyer Telegram")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, timeout=5.0)
            return r.status_code == 200
    except Exception as e:
        logger.error(f"Erreur envoi Telegram watchdog: {e}")
        return False


async def _ping_supabase_keepalive() -> bool:
    """Envoie une requête minimale à Supabase pour maintenir l'activité du projet.

    Retourne True si le ping a été accepté par Supabase, False sinon.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        logger.debug("Supabase non configuré pour keepalive")
        return False
    if create_client is None:
        logger.warning("supabase non installé — keepalive Supabase ignoré")
        return False

    try:
        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        # Requête minimale et peu coûteuse, suffisante pour réveiller l'instance.
        result = client.table("alerts").select("id").limit(1).execute()
        logger.info("🔄 Keepalive Supabase envoyé")
        return result is not None
    except Exception as e:
        logger.error(f"Erreur keepalive Supabase: {e}")
        return False


def start_watchdog_task() -> asyncio.Task:
    """Démarre la boucle de surveillance en tâche de fond.

    Si la tâche existe déjà, elle est réutilisée.
    """
    global _watchdog_task
    if _watchdog_task is not None and not _watchdog_task.done():
        return _watchdog_task
    _watchdog_task = asyncio.create_task(monitor_loop(), name="watchdog-monitor-loop")
    return _watchdog_task


async def stop_watchdog_task() -> None:
    """Annule proprement la tâche de watchdog si elle tourne."""
    global _watchdog_task
    if _watchdog_task is None:
        return
    _watchdog_task.cancel()
    try:
        await _watchdog_task
    except asyncio.CancelledError:
        logger.info("Watchdog arrêté proprement")
    finally:
        _watchdog_task = None

async def monitor_loop():
    next_keepalive = datetime.now(timezone.utc)
    if psutil is None:
        logger.error("psutil non installé — surveillance CPU/disque désactivée")

    logger.info(f"Watchdog démarré — intervalle: {CHECK_INTERVAL}s")
    while True:
        now = datetime.now(timezone.utc)

        if psutil is not None:
            cpu = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage(_get_disk_usage_path()).percent
            text = (
                f"Watchdog: CPU {cpu:.1f}% | Disk {disk:.1f}% | "
                f"{now.isoformat()}"
            )
            logger.info(text)
            if cpu >= CPU_THRESHOLD or disk >= DISK_THRESHOLD:
                logger.warning("Seuil critique dépassé — envoi Telegram")
                await _send_telegram(f"CRITIQUE: {text}")

        if now >= next_keepalive:
            ok = await _ping_supabase_keepalive()
            if ok:
                next_keepalive = now + timedelta(seconds=SUPABASE_KEEPALIVE_INTERVAL_SECONDS)
            else:
                # En cas d'échec, on retentera à la prochaine itération
                next_keepalive = now + timedelta(seconds=min(3600, SUPABASE_KEEPALIVE_INTERVAL_SECONDS))

        await asyncio.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        asyncio.run(monitor_loop())
    except KeyboardInterrupt:
        logger.info("Watchdog arrêté par utilisateur")
