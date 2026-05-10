# ============================================================
# services.py — Analyse comportementale
# Système de Surveillance Intelligente — ENSA Béni Mellal
# Détecte : Intrusion · Chute · Objet abandonné · Attroupement
# ============================================================

import logging
import math
from models import Detection, Alert, AlertType, FrameData

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# CONFIGURATION DES ZONES ET SEUILS
# ------------------------------------------------------------

# Zone d'intrusion interdite (en % de la frame 640x640)
# Format : (x1_pct, y1_pct, x2_pct, y2_pct)
ZONE_INTERDITE = (0.0, 0.0, 0.4, 1.0)   # 40% gauche de l'image

# Seuils comportementaux
FALL_RATIO_THRESHOLD     = 1.5    # Ratio w/h > 1.5 → personne allongée (plus strict pour réduire faux+)
CROWD_MIN_PERSONS        = 3     # Nombre min de personnes → attroupement
ABANDONED_OBJECT_DIST    = 120    # px — distance max objet/personne → "proche"
CONFIDENCE_MIN_DETECTION = 0.45   # Seuil confiance détection YOLO

# Classes COCO pertinentes pour la surveillance
PERSON_CLASS    = 0
BAG_CLASSES     = {24, 26, 28}    # backpack, handbag, suitcase
VEHICLE_CLASSES = {1, 2, 3, 5, 7} # bicycle, car, motorcycle, bus, truck

# Timer temporel par personne pour confirmer une chute avant alerte.
FALL_CONFIRMATION_SECONDS = 10
FALL_TRACK_TIMERS: dict[int, dict[str, object]] = {}


# ------------------------------------------------------------
# UTILITAIRES GÉOMÉTRIQUES
# ------------------------------------------------------------

def point_in_zone(
    cx: float, cy: float,
    zone: tuple[float, float, float, float],
    frame_w: int = 640, frame_h: int = 640
) -> bool:
    """Vérifie si un point (cx, cy) est dans la zone interdite."""
    zx1 = zone[0] * frame_w
    zy1 = zone[1] * frame_h
    zx2 = zone[2] * frame_w
    zy2 = zone[3] * frame_h
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Distance euclidienne entre deux points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ------------------------------------------------------------
# RÈGLES D'ANALYSE COMPORTEMENTALE
# ------------------------------------------------------------

def detect_intrusion(detections: list[Detection], camera_id: str) -> list[Alert]:
    """
    Règle 1 — Intrusion :
    Une personne dont le centre se trouve dans ZONE_INTERDITE.
    """
    alerts = []
    persons = [d for d in detections
               if d.class_id == PERSON_CLASS
               and d.confidence >= CONFIDENCE_MIN_DETECTION]

    for person in persons:
        cx, cy = person.bbox.center
        if point_in_zone(cx, cy, ZONE_INTERDITE):
            alerts.append(Alert(
                camera_id        = camera_id,
                alert_type       = AlertType.INTRUSION,
                description      = (
                    f"Personne détectée dans la zone interdite "
                    f"(centre: {cx:.0f},{cy:.0f}) "
                    f"[track_id={person.track_id}]"
                ),
                confidence_score = round(person.confidence, 3),
                detection_info   = [person],
            ))
            logger.warning(f"🚨 INTRUSION détectée — track_id={person.track_id}")

    return alerts


def detect_fall(
    detections: list[Detection],
    camera_id: str,
    frame_timestamp,
) -> list[Alert]:
    """
    Règle 2 — Chute possible :
    Une personne dont le ratio bbox w/h dépasse FALL_RATIO_THRESHOLD.
    Une personne debout a w/h ≈ 0.3–0.5.
    Une personne allongée a w/h > 1.2.
    """
    alerts = []
    persons = [d for d in detections
               if d.class_id == PERSON_CLASS
               and d.confidence >= CONFIDENCE_MIN_DETECTION]

    for person in persons:
        if person.track_id is None:
            logger.info("ℹ️ Chute ignorée — track_id manquant")
            continue

        ratio = person.bbox.aspect_ratio

        # Phase 1 strict : vérifier d'abord le ratio puis exiger
        # une confirmation de posture `Supine` fournie par T1.
        if ratio <= FALL_RATIO_THRESHOLD:
            FALL_TRACK_TIMERS.pop(person.track_id, None)
            continue

        posture_raw = person.posture_label
        posture_label = (posture_raw or "").strip().lower()
        posture_is_supine = posture_label == "supine"

        # Si aucune info de posture (N/A) — LOG uniquement, pas d'alerte
        if not posture_raw:
            FALL_TRACK_TIMERS.pop(person.track_id, None)
            logger.warning(
                f"⚠️ Posture N/A — Chute non confirmée, track_id={person.track_id} "
                f"ratio={ratio:.2f}"
            )
            continue

        # Si posture fournie mais pas 'Supine' — ignorer (non allongé)
        if not posture_is_supine:
            FALL_TRACK_TIMERS.pop(person.track_id, None)
            logger.info(
                f"ℹ️ Chute ignorée — track_id={person.track_id} "
                f"ratio={ratio:.2f} posture={person.posture_label}"
            )
            continue

        timer_state = FALL_TRACK_TIMERS.get(person.track_id)
        if timer_state is None:
            FALL_TRACK_TIMERS[person.track_id] = {
                "start_timestamp": frame_timestamp,
                "last_timestamp": frame_timestamp,
            }
            logger.info(
                f"⏱️ Timer chute démarré — track_id={person.track_id} "
                f"ratio={ratio:.2f} posture={person.posture_label}"
            )
            continue

        timer_state["last_timestamp"] = frame_timestamp
        start_timestamp = timer_state["start_timestamp"]
        elapsed_seconds = (frame_timestamp - start_timestamp).total_seconds()

        if elapsed_seconds < FALL_CONFIRMATION_SECONDS:
            logger.info(
                f"⏱️ Timer chute en cours — track_id={person.track_id} "
                f"{elapsed_seconds:.1f}s/{FALL_CONFIRMATION_SECONDS}s"
            )
            continue

        # Si on arrive ici : ratio > seuil, posture == Supine et timer >= 10s.
        conf = min(1.0, round(person.confidence * (ratio / 2.0), 3))
        alerts.append(Alert(
            camera_id        = camera_id,
            alert_type       = AlertType.CHUTE,
            description      = (
                f"Chute confirmée — ratio w/h={ratio:.2f} "
                f"(seuil={FALL_RATIO_THRESHOLD}) "
                f"[posture={person.posture_label or 'N/A'}] "
                f"[track_id={person.track_id}]"
            ),
            confidence_score = conf,
            detection_info   = [person],
        ))
        FALL_TRACK_TIMERS.pop(person.track_id, None)
        logger.warning(
            f"🚨 CHUTE confirmée — track_id={person.track_id} "
            f"ratio={ratio:.2f} durée={elapsed_seconds:.1f}s"
        )

    return alerts


def detect_abandoned_object(
    detections: list[Detection], camera_id: str
) -> list[Alert]:
    """
    Règle 3 — Objet abandonné :
    Un bagage (backpack/handbag/suitcase) détecté sans aucune personne
    à moins de ABANDONED_OBJECT_DIST pixels de son centre.

    Limitation : sans tracking persistant, cette règle génère
    des faux positifs si une personne sort du champ. À affiner
    avec un état temporel lors de l'implémentation RPi 5.
    """
    alerts = []
    bags    = [d for d in detections if d.class_id in BAG_CLASSES
               and d.confidence >= CONFIDENCE_MIN_DETECTION]
    persons = [d for d in detections if d.class_id == PERSON_CLASS
               and d.confidence >= CONFIDENCE_MIN_DETECTION]

    for bag in bags:
        bag_center = bag.bbox.center
        nearby_person = any(
            euclidean_distance(bag_center, p.bbox.center) < ABANDONED_OBJECT_DIST
            for p in persons
        )
        if not nearby_person:
            alerts.append(Alert(
                camera_id        = camera_id,
                alert_type       = AlertType.OBJET_ABANDONNE,
                description      = (
                    f"Objet abandonné : {bag.class_name} "
                    f"sans personne à moins de {ABANDONED_OBJECT_DIST}px "
                    f"[track_id={bag.track_id}]"
                ),
                confidence_score = round(bag.confidence * 0.85, 3),
                detection_info   = [bag],
            ))
            logger.warning(
                f"🚨 OBJET ABANDONNÉ — {bag.class_name} "
                f"track_id={bag.track_id}"
            )

    return alerts


def detect_crowd(detections: list[Detection], camera_id: str) -> list[Alert]:
    """
    Règle 4 — Attroupement :
    Plus de CROWD_MIN_PERSONS personnes détectées dans le même frame.
    """
    alerts = []
    persons = [d for d in detections
               if d.class_id == PERSON_CLASS
               and d.confidence >= CONFIDENCE_MIN_DETECTION]

    if len(persons) >= CROWD_MIN_PERSONS:
        avg_conf = round(
            sum(p.confidence for p in persons) / len(persons), 3
        )
        alerts.append(Alert(
            camera_id        = camera_id,
            alert_type       = AlertType.ATTROUPEMENT,
            description      = (
                f"Attroupement détecté : {len(persons)} personnes "
                f"(seuil={CROWD_MIN_PERSONS})"
            ),
            confidence_score = avg_conf,
            detection_info   = persons,
        ))
        logger.info(f"⚠️  ATTROUPEMENT — {len(persons)} personnes")

    return alerts



# ------------------------------------------------------------
# POINT D'ENTRÉE PRINCIPAL
# ------------------------------------------------------------

def analyze_behavior(frame_data: FrameData) -> list[Alert]:
    """
    Applique toutes les règles d'analyse sur un frame.
    Retourne la liste de toutes les alertes générées.

    Ordre d'analyse :
    1. Intrusion  (critique)
    2. Chute      (critique)
    3. Objet abandonné
    4. Attroupement
    """
    detections = frame_data.detections
    camera_id  = frame_data.camera_id
    all_alerts: list[Alert] = []

    if not detections:
        return all_alerts
    # 🔍 DEBUG : Afficher TOUTES les détections
    logger.info(f"📊 Total détections : {len(detections)}")
    for i, det in enumerate(detections):
        logger.info(
            f"  [{i}] class_id={det.class_id} ({det.class_name}) | "
            f"conf={det.confidence:.3f} | "
            f"track_id={det.track_id}"
        )

    # 🔍 DEBUG : Filtrer les personnes
    persons = [d for d in detections
               if d.class_id == PERSON_CLASS
               and d.confidence >= CONFIDENCE_MIN_DETECTION]
    logger.info(
        f"👥 Personnes filtrées : {len(persons)} "
        f"(confiance >= {CONFIDENCE_MIN_DETECTION})"
    )
    for p in persons:
        logger.info(f"  - track_id={p.track_id} conf={p.confidence:.3f}")


    all_alerts += detect_intrusion(detections, camera_id)
    all_alerts += detect_fall(detections, camera_id, frame_data.timestamp)
    all_alerts += detect_abandoned_object(detections, camera_id)
    all_alerts += detect_crowd(detections, camera_id)

    if all_alerts:
        logger.info(
            f"[Frame {frame_data.frame_id}] "
            f"{len(detections)} détections → "
            f"{len(all_alerts)} alerte(s)"
        )
    else:
        logger.warning(
            f"[Frame {frame_data.frame_id}] "
            f"{len(detections)} détections → "
            f"AUCUNE alerte générée ❌"
        )

    return all_alerts
