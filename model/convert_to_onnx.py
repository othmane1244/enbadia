from ultralytics import YOLO

# Télécharge automatiquement yolo11n.pt si absent
model = YOLO("yolo11n.pt")

# Export vers ONNX — paramètres compatibles Hailo DFC
model.export(
    format="onnx",
    imgsz=640,
    opset=11,          # Hailo DFC recommande opset 11
    simplify=True,     # Simplifie le graphe ONNX
    dynamic=False,     # Taille fixe obligatoire pour Hailo
    half=False,        # FP32 → sera quantifié en INT8 par le DFC
)

print("✅ Export terminé : yolo11n.onnx")