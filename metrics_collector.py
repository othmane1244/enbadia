"""
metrics_collector.py — Collecte et analyse des métriques système
Système de Surveillance Intelligente — ENSA Béni Mellal
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collecte les métriques de performance du pipeline."""
    
    def __init__(self, output_file: str = "metrics.json"):
        self.output_file = output_file
        self.frames = []  # list of {frame_id, timestamp_sent, timestamp_received, latency_ms}
        self.alerts = []  # list of {alert_type, timestamp, confidence, frame_id}
        self.start_time = None
        self.end_time = None
        self.frame_count = 0
        self.alert_count = 0
        
    def start(self):
        """Marquer le début de la collecte."""
        self.start_time = time.time()
        logger.info(f"📊 Métriques: collecte initiée à {datetime.now().isoformat()}")
    
    def end(self):
        """Marquer la fin de la collecte."""
        self.end_time = time.time()
        logger.info(f"📊 Métriques: collecte terminée à {datetime.now().isoformat()}")
    
    def record_frame_sent(self, frame_id: int) -> dict:
        """Enregistrer l'envoi d'un frame."""
        return {
            "frame_id": frame_id,
            "timestamp_sent": time.time(),
        }
    
    def record_frame_received(self, frame_data: dict):
        """Enregistrer la réception d'une réponse API pour un frame."""
        if not frame_data:
            return
        
        latency_ms = (time.time() - frame_data["timestamp_sent"]) * 1000
        self.frames.append({
            "frame_id": frame_data["frame_id"],
            "timestamp_sent": frame_data["timestamp_sent"],
            "timestamp_received": time.time(),
            "latency_ms": latency_ms,
        })
        self.frame_count += 1
    
    def record_alert(self, alert_type: str, confidence: float, frame_id: Optional[int] = None):
        """Enregistrer une alerte générée."""
        self.alerts.append({
            "alert_type": alert_type,
            "timestamp": time.time(),
            "confidence": confidence,
            "frame_id": frame_id,
        })
        self.alert_count += 1
    
    def get_statistics(self) -> dict:
        """Calculer les statistiques de performance."""
        if not self.frames or not self.start_time or not self.end_time:
            return {}
        
        total_time = self.end_time - self.start_time
        latencies = [f["latency_ms"] for f in self.frames]
        
        # Comptage par type d'alerte
        alert_types = {}
        for alert in self.alerts:
            atype = alert["alert_type"]
            alert_types[atype] = alert_types.get(atype, 0) + 1
        
        # Calcul moyenne confiance par type
        alert_confidences = {}
        for alert in self.alerts:
            atype = alert["alert_type"]
            if atype not in alert_confidences:
                alert_confidences[atype] = []
            alert_confidences[atype].append(alert["confidence"])
        
        avg_confidences = {
            atype: sum(confs) / len(confs)
            for atype, confs in alert_confidences.items()
        }
        
        stats = {
            "total_time_seconds": total_time,
            "frames_processed": self.frame_count,
            "alerts_generated": self.alert_count,
            "fps_average": self.frame_count / total_time if total_time > 0 else 0,
            "latency_ms": {
                "min": min(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0,
                "avg": sum(latencies) / len(latencies) if latencies else 0,
            },
            "alerts_by_type": alert_types,
            "avg_confidence_by_type": avg_confidences,
            "alert_rate": self.alert_count / self.frame_count if self.frame_count > 0 else 0,
        }
        
        return stats
    
    def save_to_file(self):
        """Sauvegarder les métriques en JSON."""
        stats = self.get_statistics()
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "frames": self.frames,
            "alerts": self.alerts,
        }
        
        with open(self.output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"💾 Métriques sauvegardées dans {self.output_file}")
        return output_data
    
    def print_summary(self):
        """Afficher un résumé des métriques."""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("📊 RÉSUMÉ DES MÉTRIQUES")
        print("="*70)
        print(f"Durée totale:        {stats.get('total_time_seconds', 0):.2f}s")
        print(f"Frames traitées:     {stats.get('frames_processed', 0)}")
        print(f"Alertes générées:    {stats.get('alerts_generated', 0)}")
        print(f"FPS moyen:           {stats.get('fps_average', 0):.2f}")
        print(f"Latence (ms):")
        print(f"  - Min:             {stats.get('latency_ms', {}).get('min', 0):.2f}")
        print(f"  - Avg:             {stats.get('latency_ms', {}).get('avg', 0):.2f}")
        print(f"  - Max:             {stats.get('latency_ms', {}).get('max', 0):.2f}")
        print(f"Taux alertes/frames: {stats.get('alert_rate', 0):.2%}")
        print("\nAlertes par type:")
        for atype, count in stats.get('alerts_by_type', {}).items():
            avg_conf = stats.get('avg_confidence_by_type', {}).get(atype, 0)
            print(f"  - {atype:20s}: {count:4d} (confiance avg: {avg_conf:.3f})")
        print("="*70 + "\n")


# Singleton global pour la collecte
_metrics = None

def get_metrics_collector() -> MetricsCollector:
    """Obtenir l'instance singleton du collecteur de métriques."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector("reports/metrics.json")
    return _metrics
