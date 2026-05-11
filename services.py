# ============================================================
# services.py — Analyse comportementale
# Système de Surveillance Intelligente — ENSA Béni Mellal
# Détecte : Intrusion · Chute · Objet abandonné · Attroupement
# ============================================================

import logging
import math
from datetime import datetime
from models import Detection, Alert, AlertType, FrameData, Zone

logger = logging.getLogger(__name__)

# Note: ZONE_INTERDITE removed — zones now loaded from Supabase

# Seuils comportementaux
FALL_RATIO_THRESHOLD     = 1.5    # Ratio w/h > 1.5 → personne allongée
CROWD_MIN_PERSONS        = 3     # Nombre min de personnes → attroupement
ABANDONED_OBJECT_DIST    = 120    # px — distance max objet/personne → "proche"
CONFIDENCE_MIN_DETECTION = 0.45   # Seuil confiance détection YOLO

# Classes COCO pertinentes pour la surveillance
PERSON_CLASS    = 0
BAG_CLASSES     = {24, 26, 28}    # backpack, handbag, suitcase
VEHICLE_CLASSES = {1, 2, 3, 5, 7} # bicycle, car, motorcycle, bus, truck


# ------------------------------------------------------------
# UTILITAIRES GÉOMÉTRIQUES
# ------------------------------------------------------------

def point_in_polygon(point_x: float, point_y: float, polygon_points: list[dict]) -> bool:
    """
    Vérifie si un point est à l'intérieur d'un polygone
    en utilisant l'algorithme Ray Casting.
    
    Les points du polygone sont en coordonnées normalisées (0.0-1.0).
    point_x, point_y sont aussi en coordonnées normalisées.
    
    Algorithme Ray Casting :
    On lance un rayon horizontal depuis le point vers la droite (+∞).
    Si le rayon traverse un nombre impair d'edges du polygone,
    le point est à l'intérieur.
    """
    if not polygon_points or len(polygon_points) < 3:
        return False
    
    # Normaliser les points du polygone
    vertices = []
    for p in polygon_points:
        if isinstance(p, dict):
            x = p.get('x', p.get('X', 0.0))
            y = p.get('y', p.get('Y', 0.0))
            vertices.append((float(x), float(y)))
        else:
            vertices.append((float(p[0]), float(p[1])))
    
    eps = 1e-9

    # Point sur un segment: considéré comme à l'intérieur (inclusif des bords/sommets)
    def _point_on_segment(px: float, py: float,
                          x1: float, y1: float,
                          x2: float, y2: float) -> bool:
        if (px < min(x1, x2) - eps or px > max(x1, x2) + eps or
                py < min(y1, y2) - eps or py > max(y1, y2) + eps):
            return False

        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        return abs(cross) <= eps

    # Ray casting algorithm (robuste)
    n = len(vertices)
    inside = False
    
    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]

        # Inclure explicitement les bords et sommets du polygone
        if _point_on_segment(point_x, point_y, p1x, p1y, p2x, p2y):
            return True
        
        # Vérifier si le rayon horizontal du point croise cet edge
        # Inclure la limite supérieure mais exclure la limite inférieure
        # pour éviter de compter deux fois les vertex
        if point_y > min(p1y, p2y):
            if point_y <= max(p1y, p2y):
                if point_x <= max(p1x, p2x):
                    # Calculer l'intersection x du rayon avec l'edge
                    if p1y != p2y:  # Edge n'est pas horizontal
                        xinters = (point_y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point_x <= xinters:
                            inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside


def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Distance euclidienne entre deux points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ------------------------------------------------------------
# RÈGLES D'ANALYSE COMPORTEMENTALE
# ------------------------------------------------------------

# Timers pour validation temporelle des chutes: {track_id: datetime_debut_supine}
_fall_timers: dict = {}

def detect_intrusion(detections: list[Detection], camera_id: str, zones: list[Zone] = None) -> list[Alert]:
    """
    Règle 1 — Intrusion :
    Une personne dont le centre se trouve dans une zone interdite.
    Utilise l'algorithme Ray Casting pour vérifier l'appartenance au polygone.
    
    Args:
        detections: Détections YOLO du frame
        camera_id: ID de la caméra
        zones: Zones polygonales interdites depuis Supabase (optionnel)
    """
    alerts = []
    
    # Si aucune zone, retourner
    if not zones:
        return alerts
    
    persons = [d for d in detections
               if d.class_id == PERSON_CLASS
               and d.confidence >= CONFIDENCE_MIN_DETECTION]

    # Frame dimensions used for normalization (pixels)
    frame_w = 1280.0
    frame_h = 720.0

    for person in persons:
        cx, cy = person.bbox.center
        # Normalize center coordinates to 0.0-1.0 before polygon check
        try:
            nx = float(cx) / frame_w
            ny = float(cy) / frame_h
        except Exception:
            nx = float(cx)
            ny = float(cy)
        
        # Vérifier dans chaque zone active
        for zone in zones:
            if not zone.active:
                continue
            
            # Ray Casting: vérifier si le centre est dans ce polygone
            if point_in_polygon(nx, ny, zone.points):
                alerts.append(Alert(
                    camera_id        = camera_id,
                    alert_type       = AlertType.INTRUSION,
                    description      = (
                        f"Personne détectée dans '{zone.name}' "
                        f"(centre: {int(cx)},{int(cy)}) "
                        f"[track_id={person.track_id}]"
                    ),
                    confidence_score = round(person.confidence, 3),
                    detection_info   = [person],
                ))
                logger.warning(
                    f"🚨 INTRUSION détectée dans '{zone.name}' "
                    f"— track_id={person.track_id}"
                )
                break  # Alerte une fois par personne même si dans plusieurs zones

    return alerts


def detect_fall(detections: list[Detection], camera_id: str) -> list[Alert]:
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
        track_id = person.track_id
        ratio = person.bbox.aspect_ratio
        posture = person.posture

        # Condition 1: ratio > FALL_RATIO_THRESHOLD
        if ratio <= FALL_RATIO_THRESHOLD:
            _fall_timers.pop(track_id, None)
            continue

        # Condition 2: posture == 'Supine'
        if posture != 'Supine':
            _fall_timers.pop(track_id, None)
            continue

        # Condition 3: timer > 5 secondes
        if track_id not in _fall_timers:
            _fall_timers[track_id] = datetime.now()
            continue

        duree = (datetime.now() - _fall_timers[track_id]).total_seconds()
        if duree >= 5:
            # Générer alerte chute
            conf = min(1.0, round(person.confidence * (ratio / 2.0), 3))
            alerts.append(Alert(
                camera_id        = camera_id,
                alert_type       = AlertType.CHUTE,
                description      = (
                    f"Chute confirmée — ratio w/h={ratio:.2f} "
                    f"(seuil={FALL_RATIO_THRESHOLD}) [track_id={track_id}]"
                ),
                confidence_score = conf,
                detection_info   = [person],
            ))
            logger.warning(f"🚨 CHUTE confirmée — track_id={track_id} duree={duree:.1f}s ratio={ratio:.2f}")
            # Reset timer après alerte pour éviter spam
            _fall_timers.pop(track_id, None)

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

def analyze_behavior(frame_data: FrameData, zones: list[Zone] = None) -> list[Alert]:
    """
    Applique toutes les règles d'analyse sur un frame.
    Retourne la liste de toutes les alertes générées.

    Ordre d'analyse :
    1. Intrusion  (critique) — utilise zones dynamiques de Supabase
    2. Chute      (critique)
    3. Objet abandonné
    4. Attroupement
    
    Args:
        frame_data: Données du frame avec détections
        zones: Zones polygonales d'intrusion depuis Supabase (optionnel)
    """
    detections = frame_data.detections
    camera_id  = frame_data.camera_id
    all_alerts: list[Alert] = []

    if not detections:
        return all_alerts
    # 🔍 DEBUG : Afficher TOUTES les détections
    logger.debug(f"📊 Total détections : {len(detections)}")
    for i, det in enumerate(detections):
        logger.debug(
            f"  [{i}] class_id={det.class_id} ({det.class_name}) | "
            f"conf={det.confidence:.3f} | "
            f"track_id={det.track_id}"
        )

    # 🔍 DEBUG : Filtrer les personnes
    persons = [d for d in detections
               if d.class_id == PERSON_CLASS
               and d.confidence >= CONFIDENCE_MIN_DETECTION]
    logger.debug(
        f"👥 Personnes filtrées : {len(persons)} "
        f"(confiance >= {CONFIDENCE_MIN_DETECTION})"
    )
    for p in persons:
        logger.debug(f"  - track_id={p.track_id} conf={p.confidence:.3f}")


    all_alerts += detect_intrusion(detections, camera_id, zones=zones)
    all_alerts += detect_fall(detections, camera_id)
    all_alerts += detect_abandoned_object(detections, camera_id)
    all_alerts += detect_crowd(detections, camera_id)

    if all_alerts:
        logger.info(
            f"[Frame {frame_data.frame_id}] "
            f"{len(detections)} détections → "
            f"{len(all_alerts)} alerte(s)"
        )
    else:
        logger.debug(
            f"[Frame {frame_data.frame_id}] "
            f"{len(detections)} détections → "
            f"AUCUNE alerte générée ❌"
        )

    return all_alerts
