"""
Fine-tuning YOLOv12 - Détection de personnes endormies (Sleeping)
Dataset: smoking, eating, sleeping, phone
Fichier: data.yaml (train/val/test définis)
"""

import os
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml
import shutil
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

# Chemin vers votre data.yaml
DATA_YAML = Path("data/data.yaml")  # Adaptez si besoin

# Configuration modèle
MODEL_NAME = "yolo12n.pt"  # YOLOv12 Nano (léger et rapide)
# Alternatives: yolo12s.pt, yolo12m.pt

# Dossiers de sortie
RUNS_DIR = Path("runs_sleep")
WEIGHTS_DIR = Path("weights_sleep")

for d in [RUNS_DIR, WEIGHTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Hyperparamètres
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
LEARNING_RATE = 0.001
WORKERS = 0  # IMPORTANT: 0 sur Windows pour éviter les erreurs mémoire

# ============================================================
# ÉTAPE 1: VÉRIFICATION ENVIRONNEMENT
# ============================================================

def check_environment():
    print("=" * 60)
    print("VÉRIFICATION DE L'ENVIRONNEMENT")
    print("=" * 60)
    
    cuda = torch.cuda.is_available()
    print(f"CUDA disponible: {cuda}")
    
    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Installez: pip install ultralytics")
        raise
    
    print("=" * 60)
    return cuda

# ============================================================
# ÉTAPE 2: VÉRIFICATION DATASET
# ============================================================

def verify_dataset():
    print("\n" + "=" * 60)
    print("VÉRIFICATION DU DATASET")
    print("=" * 60)
    
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"{DATA_YAML} introuvable!")
    
    with open(DATA_YAML, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"Classes: {cfg.get('names')}")
    print(f"Nombre de classes: {cfg.get('nc')}")
    print(f"Train: {cfg.get('train')}")
    print(f"Val: {cfg.get('val')}")
    print(f"Test: {cfg.get('test')}")
    
    # Vérifier les images
    base_path = DATA_YAML.parent
    for split in ['train', 'val', 'test']:
        split_path = cfg.get(split, '')
        if not split_path:
            continue
            
        img_dir = base_path / split_path
        if img_dir.exists():
            num_img = len(list(img_dir.glob("*.*")))
            print(f"  {split}: {num_img} images")
        else:
            print(f"  {split}: ⚠️ Chemin non trouvé ({img_dir})")
    
    print("=" * 60)
    return cfg

# ============================================================
# ÉTAPE 3: ENTRAÎNEMENT (FINE-TUNING)
# ============================================================

def train_model(cfg):
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT YOLOv12 - SLEEPING DETECTION")
    print("=" * 60)
    
    # Charger le modèle pré-entraîné
    print(f"Chargement: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    
    # Configuration entraînement
    args = {
        "data": str(DATA_YAML),
        "project": str(RUNS_DIR),
        "name": "sleep_yolo12",
        
        # Paramètres principaux
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
        "imgsz": IMAGE_SIZE,
        
        # Optimisation
        "lr0": LEARNING_RATE,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        
        # Augmentations (adaptées pour activités humaines)
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.3,
        "shear": 2.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        
        # Système (CRITIQUE pour Windows)
        "workers": WORKERS,  # 0 = pas de multiprocessing
        "device": 0 if torch.cuda.is_available() else "cpu",
        "cache": True,
        
        # Sauvegarde et early stopping
        "save": True,
        "save_period": 10,
        "patience": 20,
        
        # Logging
        "verbose": True,
        "seed": 42,
    }
    
    print("Configuration:")
    for k, v in args.items():
        print(f"  {k}: {v}")
    
    print("\n🚀 Lancement de l'entraînement...")
    results = model.train(**args)
    
    print("\n✅ Entraînement terminé!")
    return results, model

# ============================================================
# ÉTAPE 4: VALIDATION
# ============================================================

def validate_model(model):
    print("\n" + "=" * 60)
    print("VALIDATION SUR JEU DE TEST")
    print("=" * 60)
    
    metrics = model.val(
        data=str(DATA_YAML),
        split='test',
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=0.25,
        iou=0.45,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0,        # Évite erreurs Windows
        plots=False       # Évite erreurs mémoire matplotlib
    )
    
    print(f"\n📊 Résultats globaux:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.p:.4f}")
    print(f"  Recall: {metrics.box.r:.4f}")
    
    # Résultats par classe
    if hasattr(metrics.box, 'ap50'):
        print(f"\n📊 Par classe (AP50):")
        # Les classes sont dans l'ordre du YAML: smoking, eating, sleeping, phone
        class_names = ['smoking', 'eating', 'sleeping', 'phone']
        for i, name in enumerate(class_names):
            if i < len(metrics.box.ap50):
                print(f"  {name}: {metrics.box.ap50[i]:.4f}")
    
    return metrics

# ============================================================
# ÉTAPE 5: EXPORTATION
# ============================================================

def export_model(model):
    print("\n" + "=" * 60)
    print("EXPORTATION DU MODÈLE")
    print("=" * 60)
    
    export_dir = WEIGHTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Copier best.pt
    best_path = Path(model.trainer.best) if hasattr(model, 'trainer') else None
    if best_path and best_path.exists():
        shutil.copy(best_path, export_dir / "best.pt")
        print(f"✅ best.pt: {export_dir / 'best.pt'}")
    
    # Export ONNX
    try:
        onnx = model.export(format="onnx", imgsz=IMAGE_SIZE)
        if Path(onnx).exists():
            shutil.move(onnx, export_dir / "best.onnx")
            print(f"✅ ONNX exporté")
    except Exception as e:
        print(f"⚠️ ONNX: {e}")
    
    return export_dir

# ============================================================
# ÉTAPE 6: INFÉRENCE TEMPS RÉEL (SLEEPING)
# ============================================================

def detect_sleep(model_path=None):
    """
    Détection temps réel avec focus sur la classe 'sleeping'
    """
    print("\n" + "=" * 60)
    print("DÉTECTION TEMPS RÉEL - FOCUS: SLEEPING")
    print("=" * 60)
    
    import cv2
    
    # Charger modèle
    if model_path and Path(model_path).exists():
        model = YOLO(model_path)
    else:
        # Chercher automatiquement
        candidates = list(RUNS_DIR.glob("**/best.pt"))
        if not candidates:
            raise FileNotFoundError("Aucun modèle trouvé!")
        model = YOLO(str(candidates[0]))
    
    print(f"Modèle chargé: {model.names}")
    print("Commandes: 'q' = quitter | 's' = sauvegarder")
    print("-" * 60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam non accessible!")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Prédiction
        results = model(frame, conf=0.5, verbose=False)
        annotated = results[0].plot()
        
        # Analyse des détections - Focus sleeping
        boxes = results[0].boxes
        sleep_detected = False
        
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Focus sur sleeping
            if cls_name == 'sleeping' and conf > 0.6:
                sleep_detected = True
                print(f"😴 [{frame_count}] SLEEPING détecté! Conf: {conf:.2%}")
                
                # Ajouter alerte visuelle
                cv2.putText(annotated, "ALERTE: PERSONNE ENDORMIE!", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, (0, 0, 255), 3)
        
        # Compteur de personnes endormies
        sleep_count = sum(1 for b in boxes if model.names[int(b.cls[0])] == 'sleeping')
        status_text = f"Sleeping: {sleep_count}"
        cv2.putText(annotated, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        cv2.imshow("Sleep Detection - YOLOv12", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_sleep_{frame_count}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"💾 Image sauvegardée: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Détection arrêtée")

# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     YOLOv12 Fine-tuning                                    ║
    ║     Classes: smoking | eating | sleeping | phone            ║
    ║     Focus: Détection de personnes endormies                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Étape 1
    check_environment()
    
    # Étape 2
    cfg = verify_dataset()
    
    # Étape 3
    results, model = train_model(cfg)
    
    # Étape 4
    metrics = validate_model(model)
    
    # Étape 5
    export_dir = export_model(model)
    
    print("\n" + "=" * 60)
    print("🎉 PIPELINE TERMINÉ!")
    print("=" * 60)
    print(f"📁 Modèle sauvegardé dans: {export_dir}")
    print(f"📊 mAP50: {metrics.box.map50:.4f}")
    
    # Lancer détection temps réel
    print("\n🚀 Lancer la détection webcam?")
    response = input("Appuyez sur Entrée pour continuer (ou 'n' pour skipper): ")
    
    if response.lower() != 'n':
        best_model = export_dir / "best.pt"
        detect_sleep(str(best_model))

if __name__ == "__main__":
    main()