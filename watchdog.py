from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

WATCHDOG_INTERVAL_SECONDS = int(os.getenv("WATCHDOG_INTERVAL_SECONDS", "30"))
SUPABASE_KEEPALIVE_INTERVAL_SECONDS = int(os.getenv("SUPABASE_KEEPALIVE_INTERVAL_SECONDS", str(72 * 3600)))
CPU_ALERT_THRESHOLD = float(os.getenv("WATCHDOG_CPU_THRESHOLD", "90"))
DISK_ALERT_THRESHOLD = float(os.getenv("WATCHDOG_DISK_THRESHOLD", "90"))

_watchdog_task: Optional[asyncio.Task] = None
_last_supabase_ping: Optional[datetime] = None


def _get_disk_usage_path() -> str:
    if os.name == "nt":
        return str(Path.cwd().anchor or "C:\\")
    return "/"


async def _ping_supabase_keepalive() -> None:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        logger.debug("Supabase keepalive ignoré — variables manquantes")
        return

    try:
        from supabase import create_client
    except Exception:
        logger.warning("Package supabase manquant — keepalive ignoré")
        return

    try:
        client = create_client(url, key)
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.table("alerts").select("id").limit(1).execute(),
        )
        logger.info("🔄 Keepalive Supabase envoyé")
    except Exception as exc:
        logger.warning(f"Supabase keepalive échoué: {exc}")


async def _monitor_once() -> None:
    try:
        import psutil
    except Exception:
        logger.warning("psutil indisponible — watchdog CPU/disque désactivé")
        return

    cpu = psutil.cpu_percent(interval=None)
    disk = psutil.disk_usage(_get_disk_usage_path()).percent
    logger.info(f"Watchdog: CPU {cpu:.1f}% | Disk {disk:.1f}% | {datetime.now(timezone.utc).isoformat()}")

    if cpu >= CPU_ALERT_THRESHOLD:
        logger.warning(f"CPU élevé détecté: {cpu:.1f}%")
    if disk >= DISK_ALERT_THRESHOLD:
        logger.warning(f"Disque presque plein: {disk:.1f}%")


def _read_cpu_temperature_c() -> Optional[float]:
    try:
        import psutil
    except Exception:
        return None

    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:
        return None

    if not temps:
        return None

    preferred_sensors = (
        "coretemp",
        "cpu_thermal",
        "k10temp",
        "acpitz",
        "cpu-thermal",
    )

    for sensor_name in preferred_sensors:
        entries = temps.get(sensor_name)
        if not entries:
            continue
        for entry in entries:
            if entry.current is not None:
                return float(entry.current)

    for entries in temps.values():
        for entry in entries:
            if entry.current is not None:
                return float(entry.current)

    return None


def get_watchdog_status() -> dict:
    try:
        import psutil
    except Exception:
        return {
            "cpu_temperature_c": None,
            "ram_used_percent": None,
            "disk_used_percent": None,
            "watchdog_interval_seconds": WATCHDOG_INTERVAL_SECONDS,
            "supabase_keepalive_interval_seconds": SUPABASE_KEEPALIVE_INTERVAL_SECONDS,
            "last_supabase_ping": _last_supabase_ping.isoformat() if _last_supabase_ping else None,
        }

    return {
        "cpu_temperature_c": _read_cpu_temperature_c(),
        "ram_used_percent": float(psutil.virtual_memory().percent),
        "disk_used_percent": float(psutil.disk_usage(_get_disk_usage_path()).percent),
        "watchdog_interval_seconds": WATCHDOG_INTERVAL_SECONDS,
        "supabase_keepalive_interval_seconds": SUPABASE_KEEPALIVE_INTERVAL_SECONDS,
        "last_supabase_ping": _last_supabase_ping.isoformat() if _last_supabase_ping else None,
    }


async def _watchdog_loop() -> None:
    global _last_supabase_ping
    logger.info(f"Watchdog démarré — intervalle: {WATCHDOG_INTERVAL_SECONDS}s")
    while True:
        await _monitor_once()

        now = datetime.now(timezone.utc)
        if _last_supabase_ping is None or (now - _last_supabase_ping) >= timedelta(seconds=SUPABASE_KEEPALIVE_INTERVAL_SECONDS):
            await _ping_supabase_keepalive()
            _last_supabase_ping = now

        await asyncio.sleep(WATCHDOG_INTERVAL_SECONDS)


def start_watchdog_task() -> asyncio.Task:
    global _watchdog_task
    if _watchdog_task and not _watchdog_task.done():
        return _watchdog_task
    _watchdog_task = asyncio.create_task(_watchdog_loop(), name="watchdog-monitor-loop")
    return _watchdog_task


async def stop_watchdog_task() -> None:
    global _watchdog_task
    if not _watchdog_task:
        return
    _watchdog_task.cancel()
    try:
        await _watchdog_task
    except asyncio.CancelledError:
        logger.info("Watchdog arrêté proprement")
    finally:
        _watchdog_task = None