"""
Test temps réel - Détection de posture (Supine / Not_Supine)
Modèle: YOLOv12 fine-tuné sur dataset Posture
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import os

# ============================================================
# CONFIGURATION - Adaptez ce chemin si besoin
# ============================================================

# Option 1: Chemin direct (remplacez par votre chemin exact)
# MODEL_PATH = r"\runs_posture\posture_supine_yolo12\weights\best.pt"

# Option 2: Recherche automatique (décommentez si vous ne savez pas où est le modèle)
PROJECT_ROOT = Path(".")
candidates = list(PROJECT_ROOT.rglob("best.pt"))
if candidates:
    MODEL_PATH = str(candidates[0])
else:
    MODEL_PATH = None

# Seuil de confiance
CONF_THRESHOLD = 0.5

# ============================================================
# VÉRIFICATION MODÈLE
# ============================================================

if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    print("❌ Modèle introuvable!")
    print(f"Chemin cherché: {MODEL_PATH}")
    print("\nVeuillez vérifier le chemin dans la variable MODEL_PATH")
    exit()

print("=" * 60)
print("DÉTECTION POSTURE - TEMPS RÉEL")
print("=" * 60)

# Charger le modèle
model = YOLO(MODEL_PATH)
print(f"✅ Modèle chargé: {MODEL_PATH}")
print(f"Classes détectables: {model.names}")
print("-" * 60)
print("Commandes:")
print("  'q' = quitter")
print("  's' = sauvegarder une capture d'écran")
print("=" * 60)

# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Erreur: Impossible d'accéder à la webcam!")
    exit()

print("\n🎥 Webcam active...")
print("Placez-vous devant la caméra pour tester la posture")
print("-" * 60)

frame_count = 0
supine_alerts = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Erreur de capture webcam")
        break

    frame_count += 1

    # Prédiction
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    annotated = results[0].plot()

    # Analyse des détections
    supine_count = 0
    not_supine_count = 0

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])

        # Comptage
        if cls_name == 'Supine':
            supine_count += 1
        elif cls_name == 'Not_Supine':
            not_supine_count += 1

        # Alerte spéciale pour Supine (personne couchée)
        if cls_name == 'Supine' and conf > 0.6:
            supine_alerts += 1
            print(f"🛏️  [{frame_count}] SUPINE détecté! Confiance: {conf:.1%}")

            # Texte d'alerte en rouge
            cv2.putText(
                annotated,
                "!!! ALERTE: PERSONNE COUCHEE !!!",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),  # Rouge
                3
            )

    # Bandeau d'information en haut
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 45), (0, 0, 0), -1)

    info_text = f"Supine: {supine_count} | Not_Supine: {not_supine_count} | Frame: {frame_count}"
    cv2.putText(annotated, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Affichage
    cv2.imshow("Posture Detection - Supine/Not_Supine", annotated)

    # Gestion des touches
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\n👋 Arrêt demandé par l'utilisateur")
        break

    elif key == ord('s'):
        filename = f"capture_posture_{frame_count}.jpg"
        cv2.imwrite(filename, annotated)
        print(f"💾 Capture sauvegardée: {filename}")

# ============================================================
# FIN
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("=" * 60)
print("RÉSUMÉ DU TEST")
print("=" * 60)
print(f"🎞️  Frames analysées: {frame_count}")
print(f"🛏️  Alerts Supine: {supine_alerts}")
print("✅ Test terminé")
print("=" * 60)