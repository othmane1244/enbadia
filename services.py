# ============================================================
# services.py — Analyse comportementale
# Système de Surveillance Intelligente — ENSA Béni Mellal
# Détecte : Intrusion · Chute · Objet abandonné · Attroupement
# ============================================================

import logging
import math
from models import Detection, Alert, AlertType, FrameData, Zone

logger = logging.getLogger(__name__)

# Note: ZONE_INTERDITE removed — zones now loaded from Supabase

# Seuils comportementaux
FALL_RATIO_THRESHOLD     = 1.2    # Ratio w/h > 1.2 → personne allongée
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
    On lance un rayon horizontal depuis le point vers la droite.
    Si le rayon traverse un nombre impair d'edges du polygone,
    le point est à l'intérieur.
    """
    if not polygon_points or len(polygon_points) < 3:
        return False
    
    # Normaliser les points du polygone
    vertices = []
    for p in polygon_points:
        if isinstance(p, dict):
            vertices.append((p.get('x', 0.0), p.get('y', 0.0)))
        else:
            vertices.append((float(p[0]), float(p[1])))
    
    # Ray casting algorithm
    n = len(vertices)
    inside = False
    
    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]
        
        # Vérifier si le rayon horizontal du point traverse cet edge
        if point_y > min(p1y, p2y):
            if point_y <= max(p1y, p2y):
                if point_x <= max(p1x, p2x):
                    # Calculer l'intersection x du rayon avec l'edge
                    if p1y != p2y:
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

    for person in persons:
        cx, cy = person.bbox.center
        
        # Vérifier dans chaque zone active
        for zone in zones:
            if not zone.active:
                continue
            
            # Ray Casting: vérifier si le centre est dans ce polygone
            if point_in_polygon(cx, cy, zone.points):
                alerts.append(Alert(
                    camera_id        = camera_id,
                    alert_type       = AlertType.INTRUSION,
                    description      = (
                        f"Personne détectée dans '{zone.name}' "
                        f"(centre: {cx:.0f},{cy:.0f}) "
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
        ratio = person.bbox.aspect_ratio
        if ratio > FALL_RATIO_THRESHOLD:
            # Score de confiance pondéré par la conf YOLO et le ratio
            conf = min(1.0, round(person.confidence * (ratio / 2.0), 3))
            alerts.append(Alert(
                camera_id        = camera_id,
                alert_type       = AlertType.CHUTE,
                description      = (
                    f"Chute possible — ratio w/h={ratio:.2f} "
                    f"(seuil={FALL_RATIO_THRESHOLD}) "
                    f"[track_id={person.track_id}]"
                ),
                confidence_score = conf,
                detection_info   = [person],
            ))
            logger.warning(
                f"🚨 CHUTE possible — track_id={person.track_id} ratio={ratio:.2f}"
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
        logger.warning(
            f"[Frame {frame_data.frame_id}] "
            f"{len(detections)} détections → "
            f"AUCUNE alerte générée ❌"
        )

    return all_alerts
