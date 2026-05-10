# ============================================================
# models.py — Structures de données Pydantic
# Système de Surveillance Intelligente — ENSA Béni Mellal
# Validation automatique des données entrée/sortie de l'API
# ============================================================

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


# ============================================================
# ZONES — Géométrie de surveillance
# ============================================================

class Zone(BaseModel):
    """Zone polygonale de surveillance depuis Supabase."""
    zone_id: Optional[str] = Field(None, description="ID unique Supabase")
    camera_id: str = Field(..., description="ID caméra")
    name: str = Field(..., description="Nom de la zone")
    points: list[dict] = Field(default_factory=list, description="Points polygone")
    active: bool = Field(True, description="Zone active")


# ============================================================
# ENTRÉE — Ce que le pipeline envoie à l'API
# ============================================================

class BoundingBox(BaseModel):
    """Boîte englobante d'un objet détecté."""
    x1: int = Field(..., ge=0, description="Coin supérieur gauche X")
    y1: int = Field(..., ge=0, description="Coin supérieur gauche Y")
    x2: int = Field(..., ge=0, description="Coin inférieur droit X")
    y2: int = Field(..., ge=0, description="Coin inférieur droit Y")

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def aspect_ratio(self) -> float:
        """Ratio largeur/hauteur — utile pour détecter les chutes."""
        if self.height == 0:
            return 0.0
        return self.width / self.height


class Detection(BaseModel):
    """Un objet détecté dans un frame."""
    track_id:   Optional[int]   = Field(None, description="ID de tracking BoT-SORT")
    class_id:   int             = Field(..., ge=0, lt=80)
    class_name: str             = Field(..., min_length=1)
    confidence: float           = Field(..., ge=0.0, le=1.0)
    bbox:       BoundingBox


class FrameData(BaseModel):
    """Données complètes d'un frame vidéo traité."""
    camera_id:  str             = Field(..., description="ID caméra ex: cam_01")
    frame_id:   int             = Field(..., ge=0)
    timestamp:  datetime        = Field(default_factory=datetime.utcnow)
    fps:        float           = Field(..., ge=0.0)
    detections: list[Detection] = Field(default_factory=list)
    zones:      list[Zone]      = Field(default_factory=list, description="Zones d'intrusion")


# ------------------------------------------------------------
# SORTIE — Ce que l'API génère et envoie au dashboard
# ------------------------------------------------------------

class AlertType(str):
    INTRUSION         = "Intrusion"
    CHUTE             = "Chute"
    OBJET_ABANDONNE   = "Objet_Abandonne"
    ATTROUPEMENT      = "Attroupement"


class Alert(BaseModel):
    """Alerte générée par l'analyse comportementale."""
    id:               str            = Field(default_factory=lambda: str(uuid.uuid4()))
    camera_id:        str
    alert_type:       str
    description:      str
    confidence_score: float          = Field(..., ge=0.0, le=1.0)
    detection_info:   list[Detection]
    timestamp:        datetime       = Field(default_factory=datetime.utcnow)
    is_resolved:      bool           = False


class ProcessFrameResponse(BaseModel):
    """Réponse de l'endpoint POST /process_frame/"""
    frame_id:       int
    camera_id:      str
    detections_count: int
    alerts_generated: list[Alert]
    processing_time_ms: float


class ZoneCreate(BaseModel):
    """Données envoyées par le dashboard pour enregistrer une zone."""
    camera_id: str = Field(..., min_length=1)
    zone_name: str = Field(default='Zone Interdite', min_length=1)
    points: list[dict[str, float]] = Field(default_factory=list)
    active: bool = True
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class VideoFrame(BaseModel):
    """Frame vidéo encodée en JPEG base64 pour le WebSocket /ws/video."""
    frame_id: int
    camera_id: str
    data: str  # Base64-encoded JPEG