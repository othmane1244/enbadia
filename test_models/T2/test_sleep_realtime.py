"""
Test temps réel - Détection personnes endormies
"""

import cv2
from ultralytics import YOLO

# ============================================================
# CHEMIN EXACT DE VOTRE MODÈLE
# ============================================================

MODEL_PATH = r"C:\Users\user\runs\detect\runs_sleep\sleep_yolo12\weights\best.pt"

# Vérifier que le fichier existe
import os
if not os.path.exists(MODEL_PATH):
    print(f"❌ Modèle introuvable: {MODEL_PATH}")
    print("Vérifiez le chemin!")
    exit()

print("=" * 60)
print("DÉTECTION TEMPS RÉEL - SLEEPING")
print("=" * 60)

# Charger le modèle
model = YOLO(MODEL_PATH)
print(f"✅ Modèle chargé!")
print(f"Classes: {model.names}")
print("-" * 60)

# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam non accessible!")
    exit()

print("\n🎥 Webcam active...")
print("Commandes: 'q' = quitter | 's' = sauvegarder")
print("-" * 60)

frame_count = 0
sleep_alerts = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Prédiction
    results = model(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()
    
    # Analyse - Focus sur sleeping
    sleeping_detected = False
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        
        if cls_name == 'sleeping' and conf > 0.6:
            sleeping_detected = True
            sleep_alerts += 1
            print(f"😴 [{frame_count}] PERSONNE ENDORMIE! Conf: {conf:.1%}")
            
            # Alerte rouge
            cv2.putText(annotated, "!!! ALERTE: PERSONNE ENDORMIE !!!", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 0, 255), 3)
    
    # Bandeau info
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (0,0,0), -1)
    cv2.putText(annotated, f"Frame: {frame_count} | Alerts: {sleep_alerts}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imshow("Sleep Detection YOLOv12", annotated)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"capture_sleep_{frame_count}.jpg"
        cv2.imwrite(filename, annotated)
        print(f"💾 Sauvegardé: {filename}")

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Terminé!")
print(f"Frames: {frame_count} | Alerts sleeping: {sleep_alerts}")