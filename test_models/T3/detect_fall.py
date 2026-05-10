from ultralytics import YOLO
import cv2

# Charger votre modèle fine-tuné
model = YOLO(r"C:\Users\user\runs\detect\runs\fall_detection_yolo122\weights\best.pt")

print("=" * 50)
print("Classes connues par le modèle:", model.names)
print("=" * 50)
print("Ce modèle ne connaît QUE 'fall'")
print("Il va détecter 'fall' sur TOUT ce qu'il voit !")
print("=" * 50)

# Démarrer la webcam
cap = cv2.VideoCapture(0)

print("\n🎥 Webcam active")
print("Testez avec :")
print("  - Votre visage")
print("  - Une main")
print("  - Un objet (téléphone, stylo)")
print("  - Une vraie personne")
print("\nAppuyez sur 'q' pour quitter")
print("-" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Prédiction avec seuil bas pour voir tous les faux positifs
    results = model(frame, conf=0.3, verbose=False)
    
    # Analyser les détections
    boxes = results[0].boxes
    if len(boxes) > 0:
        print(f"\n🎯 {len(boxes)} détection(s):")
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            
            # Coordonnées
            x, y, w, h = box.xywh[0]
            
            print(f"  [{i+1}] {class_name} | Conf: {conf:.2%} | Pos: ({int(x)}, {int(y)})")
    
    # Dessiner les résultats sur l'image
    annotated = results[0].plot()
    
    # Ajouter texte explicatif
    cv2.putText(annotated, f"Classes: {model.names}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, "Ce modèle ne connait que 'fall'", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Test Webcam - Fall Detection Only", annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Test terminé")