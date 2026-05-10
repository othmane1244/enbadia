from __future__ import annotations

import asyncio
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from database import get_supabase_client

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TELEGRAM_API_BASE = 'https://api.telegram.org'
SCHEDULER_TIMEZONE = timezone.utc

_scheduler: AsyncIOScheduler | None = None

TYPE_ORDER = [
    ('Chute', ('Chute', 'Fall')),
    ('Intrusion', ('Intrusion',)),
    ('Attroupement', ('Attroupement', 'Crowd')),
    ('Objet abandonné', ('Objet_Abandonne', 'Objet abandonné', 'Abandoned object')),
]


@dataclass
class ReportMetrics:
    report_date: date
    total_alerts: int
    type_counts: dict[str, int]
    peak_hour: Optional[int]
    peak_hour_count: int


def _normalize_alert_type(alert_type: str | None) -> str:
    if not alert_type:
        return 'Autre'

    normalized = alert_type.strip().lower().replace('-', '_').replace(' ', '_')
    mapping = {
        'chute': 'Chute',
        'fall': 'Chute',
        'intrusion': 'Intrusion',
        'attroupement': 'Attroupement',
        'crowd': 'Attroupement',
        'objet_abandonne': 'Objet abandonné',
        'objet_abandonné': 'Objet abandonné',
        'abandoned_object': 'Objet abandonné',
    }
    return mapping.get(normalized, 'Autre')


def _parse_created_at(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _day_bounds_utc(target_date: date) -> tuple[datetime, datetime]:
    start = datetime.combine(target_date, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


async def fetch_alerts_for_day(target_date: date) -> list[dict]:
    client = get_supabase_client()
    if client is None:
        logger.warning('Supabase indisponible — génération de rapport annulée')
        return []

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: client.table('alerts').select('*').order('created_at', desc=False).execute(),
        )
    except Exception as exc:
        logger.error(f'Impossible de récupérer les alertes Supabase: {exc}')
        return []

    start, end = _day_bounds_utc(target_date)
    daily_alerts: list[dict] = []

    for alert in result.data or []:
        created_at = _parse_created_at(alert.get('created_at'))
        if created_at is None:
            continue
        if start <= created_at < end:
            daily_alerts.append(alert)

    return daily_alerts


def build_report_metrics(target_date: date, alerts: Iterable[dict]) -> ReportMetrics:
    alerts_list = list(alerts)
    total_alerts = len(alerts_list)

    type_counts = Counter()
    hourly_counts = Counter()

    for alert in alerts_list:
        type_counts[_normalize_alert_type(alert.get('alert_type'))] += 1
        created_at = _parse_created_at(alert.get('created_at'))
        if created_at is not None:
            hourly_counts[created_at.hour] += 1

    peak_hour: Optional[int] = None
    peak_hour_count = 0
    if hourly_counts:
        peak_hour, peak_hour_count = max(hourly_counts.items(), key=lambda item: (item[1], -item[0]))

    return ReportMetrics(
        report_date=target_date,
        total_alerts=total_alerts,
        type_counts=dict(type_counts),
        peak_hour=peak_hour,
        peak_hour_count=peak_hour_count,
    )


def generate_pdf_report(metrics: ReportMetrics, output_path: Path) -> Path:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#111827'),
        spaceAfter=10,
    )
    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4b5563'),
        spaceAfter=18,
    )
    normal_style = ParagraphStyle(
        'ReportBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#111827'),
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"Rapport Surveillance — {metrics.report_date.isoformat()}",
    )

    rows = [['Type', 'Count', 'Pourcentage']]
    for label, _aliases in TYPE_ORDER:
        count = metrics.type_counts.get(label, 0)
        percentage = f"{(count / metrics.total_alerts * 100):.1f}%" if metrics.total_alerts else '0.0%'
        rows.append([label, str(count), percentage])

    table = Table(rows, colWidths=[90 * mm, 35 * mm, 35 * mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#111827')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('LEADING', (0, 0), (-1, -1), 13),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f9fafb'), colors.white]),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))

    peak_hour_text = 'N/A'
    if metrics.peak_hour is not None:
        peak_hour_text = f"{metrics.peak_hour:02d}:00 - {metrics.peak_hour:02d}:59 ({metrics.peak_hour_count} alertes)"

    story = [
        Paragraph(f"Rapport Surveillance — {metrics.report_date.isoformat()}", title_style),
        Paragraph(f"Généré automatiquement le {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", subtitle_style),
        Paragraph(f"Résumé total alertes : <b>{metrics.total_alerts}</b>", normal_style),
        Spacer(1, 10),
        table,
        Spacer(1, 12),
        Paragraph(f"Heure de pointe : <b>{peak_hour_text}</b>", normal_style),
    ]

    doc.build(story)
    return output_path


async def send_report_via_telegram(pdf_path: Path, metrics: ReportMetrics) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning('Telegram non configuré — envoi du rapport ignoré')
        return False

    url = f'{TELEGRAM_API_BASE}/bot{TELEGRAM_BOT_TOKEN}/sendDocument'
    caption = (
        f"Rapport Surveillance — {metrics.report_date.isoformat()}\n"
        f"Total alertes: {metrics.total_alerts}\n"
        f"Heure de pointe: "
        f"{f'{metrics.peak_hour:02d}:00' if metrics.peak_hour is not None else 'N/A'}"
    )

    try:
        async with httpx.AsyncClient() as client:
            with pdf_path.open('rb') as pdf_file:
                response = await client.post(
                    url,
                    data={
                        'chat_id': TELEGRAM_CHAT_ID,
                        'caption': caption,
                    },
                    files={
                        'document': (pdf_path.name, pdf_file, 'application/pdf'),
                    },
                    timeout=20.0,
                )

        if response.status_code == 200:
            logger.info(f'✅ Rapport PDF envoyé via Telegram: {pdf_path.name}')
            return True

        logger.error(f'❌ Erreur Telegram sendDocument {response.status_code}: {response.text}')
        return False
    except Exception as exc:
        logger.error(f'❌ Exception envoi rapport Telegram: {exc}')
        return False


async def generate_daily_report(target_date: Optional[date] = None) -> Optional[Path]:
    report_date = target_date or datetime.now(timezone.utc).date()
    alerts = await fetch_alerts_for_day(report_date)
    metrics = build_report_metrics(report_date, alerts)
    output_path = REPORTS_DIR / f'report_{report_date.isoformat()}.pdf'
    generate_pdf_report(metrics, output_path)
    await send_report_via_telegram(output_path, metrics)
    return output_path


async def _scheduled_nightly_report() -> None:
    # Exécuté à minuit UTC pour produire le rapport du jour qui vient de se terminer.
    target_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    logger.info(f'📝 Génération du rapport nocturne pour {target_date.isoformat()}')
    await generate_daily_report(target_date)


def start_report_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler and _scheduler.running:
        return _scheduler

    scheduler = AsyncIOScheduler(timezone=SCHEDULER_TIMEZONE)
    scheduler.add_job(
        _scheduled_nightly_report,
        CronTrigger(hour=0, minute=0, timezone=SCHEDULER_TIMEZONE),
        id='nightly_surveillance_report',
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    _scheduler = scheduler
    logger.info('🗓️ Scheduler de rapport lancé')
    return scheduler


async def stop_report_scheduler() -> None:
    global _scheduler
    if not _scheduler:
        return
    _scheduler.shutdown(wait=False)
    _scheduler = None
    logger.info('🛑 Scheduler de rapport arrêté')