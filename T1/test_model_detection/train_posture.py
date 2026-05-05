"""
Fine-tuning YOLOv12 - Détection de posture couchée (Supine)
Dataset: Posture (Roboflow) - 2 classes: Not_Supine, Supine
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

# Chemin vers votre nouveau dataset
DATA_YAML = Path("data/data.yaml")  # Adaptez si le chemin est différent

# Modèle de base
MODEL_NAME = "yolo12n.pt"  # YOLOv12 Nano

# Dossiers de sortie
RUNS_DIR = Path("runs_posture")
WEIGHTS_DIR = Path("weights_posture")

for d in [RUNS_DIR, WEIGHTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Hyperparamètres
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
LEARNING_RATE = 0.001
WORKERS = 5  # IMPORTANT: 0 sur Windows pour éviter les erreurs mémoire

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
# ÉTAPE 2: VÉRIFICATION DATASET POSTURE
# ============================================================

def verify_dataset():
    print("\n" + "=" * 60)
    print("VÉRIFICATION DU DATASET POSTURE")
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
    print(f"Source: {cfg.get('roboflow', {}).get('url', 'N/A')}")
    
    # Vérifier les images et labels
    base_path = DATA_YAML.parent
    for split in ['train', 'valid', 'test']:
        split_path = cfg.get(split, '')
        if not split_path:
            continue
            
        img_dir = base_path / split_path.replace('../', '')
        lbl_dir = (base_path / split_path.replace('../', '')).parent / "labels"
        
        if img_dir.exists():
            num_img = len(list(img_dir.glob("*.*")))
            print(f"  {split}/images: {num_img} images")
        else:
            print(f"  {split}/images: ⚠️ Non trouvé ({img_dir})")
            
        if lbl_dir.exists():
            num_lbl = len(list(lbl_dir.glob("*.txt")))
            print(f"  {split}/labels: {num_lbl} annotations")
        else:
            print(f"  {split}/labels: ⚠️ Non trouvé ({lbl_dir})")
    
    print("=" * 60)
    return cfg

# ============================================================
# ÉTAPE 3: ENTRAÎNEMENT (FINE-TUNING)
# ============================================================

def train_model(cfg):
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT YOLOv12 - POSTURE SUPINE")
    print("=" * 60)
    
    print(f"Chargement: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    
    # Configuration entraînement optimisée pour posture
    args = {
        "data": str(DATA_YAML.absolute()),  # Chemin absolu pour éviter les erreurs
        "project": str(RUNS_DIR),
        "name": "posture_supine_yolo12",
        
        # Paramètres principaux
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
        "imgsz": IMAGE_SIZE,
        
        # Optimisation
        "lr0": LEARNING_RATE,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        
        # Augmentations adaptées pour la posture corporelle
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 15.0,      # Rotation importante pour varier les angles de vue
        "translate": 0.1,
        "scale": 0.5,
        "shear": 5.0,         # Cisaillement pour simuler différentes perspectives
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        
        # Système (CRITIQUE pour Windows)
        "workers": WORKERS,   # 0 = pas de multiprocessing (évite erreur 1455)
        "device": 0 if torch.cuda.is_available() else "cpu",
        "cache": True,
        
        # Sauvegarde et early stopping
        "save": True,
        "save_period": 10,
        "patience": 20,
        
        # Logging (désactiver plots pour éviter erreurs mémoire)
        "plots": False,       # Évite les erreurs matplotlib/OpenCV
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
        data=str(DATA_YAML.absolute()),
        split='test',
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=0.25,
        iou=0.45,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0,        # Évite erreurs Windows
        plots=False       # Évite erreurs mémoire
    )
    
    print(f"\n📊 Résultats globaux:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.p:.4f}")
    print(f"  Recall: {metrics.box.r:.4f}")
    
    # Résultats par classe
    if hasattr(metrics.box, 'ap50'):
        print(f"\n📊 Par classe:")
        class_names = cfg.get('names', ['Not_Supine', 'Supine'])
        for i, name in enumerate(class_names):
            if i < len(metrics.box.ap50):
                print(f"  {name}: {metrics.box.ap50[i]:.4f}")
    
    return metrics

# ============================================================
# ÉTAPE 5: EXPORTATION DU MODÈLE
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
        print(f"✅ best.pt sauvegardé: {export_dir / 'best.pt'}")
    else:
        # Chercher le best.pt dans runs
        candidates = list(RUNS_DIR.rglob("best.pt"))
        if candidates:
            shutil.copy(candidates[0], export_dir / "best.pt")
            print(f"✅ best.pt trouvé et copié: {export_dir / 'best.pt'}")
    
    # Export ONNX
    try:
        onnx = model.export(format="onnx", imgsz=IMAGE_SIZE)
        if Path(onnx).exists():
            shutil.move(onnx, export_dir / "best.onnx")
            print(f"✅ Export ONNX réussi")
    except Exception as e:
        print(f"⚠️ Export ONNX échoué: {e}")
    
    return export_dir

# ============================================================
# ÉTAPE 6: INFÉRENCE TEMPS RÉEL - POSTURE
# ============================================================

def detect_posture(model_path=None):
    """
    Détection temps réel avec focus sur la posture Supine (couché sur le dos)
    """
    print("\n" + "=" * 60)
    print("DÉTECTION TEMPS RÉEL - POSTURE SUPINE")
    print("=" * 60)
    
    import cv2
    
    # Charger modèle
    if model_path and Path(model_path).exists():
        model = YOLO(model_path)
    else:
        # Chercher automatiquement
        candidates = list(RUNS_DIR.rglob("best.pt"))
        if not candidates:
            raise FileNotFoundError("Aucun modèle trouvé!")
        model = YOLO(str(candidates[0]))
    
    print(f"✅ Modèle chargé: {model.names}")
    print("Commandes: 'q' = quitter | 's' = sauvegarder")
    print("-" * 60)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam non accessible!")
    
    frame_count = 0
    supine_alerts = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Prédiction
        results = model(frame, conf=0.5, verbose=False)
        annotated = results[0].plot()
        
        # Analyse des détections - Focus Supine
        boxes = results[0].boxes
        supine_count = 0
        
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Focus sur Supine (personne couchée sur le dos)
            if cls_name == 'Supine' and conf > 0.6:
                supine_count += 1
                supine_alerts += 1
                print(f"🛏️  [{frame_count}] POSTURE SUPINE détectée! Conf: {conf:.1%}")
                
                # Alerte visuelle spéciale
                cv2.putText(annotated, "!!! ALERTE: PERSONNE COUCHEE !!!", 
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (0, 0, 255), 3)
            
            # Optionnel: afficher Not_Supine aussi
            elif cls_name == 'Not_Supine':
                pass  # Détection normale sans alerte
        
        # Bandeau d'information
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (0,0,0), -1)
        status = f"Posture Detection | Supine: {supine_count} | Frame: {frame_count}"
        cv2.putText(annotated, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.imshow("Posture Detection - YOLOv12", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_posture_{frame_count}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"💾 Capture sauvegardée: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Détection arrêtée")
    print(f"Total frames: {frame_count} | Alerts Supine: {supine_alerts}")

# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     YOLOv12 Fine-tuning - Posture Detection                ║
    ║     Classes: Not_Supine | Supine                            ║
    ║     Dataset: Roboflow Posture                               ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Étape 1: Vérifier environnement
    check_environment()
    
    # Étape 2: Vérifier dataset
    global cfg
    cfg = verify_dataset()
    
    # Étape 3: Entraînement
    results, model = train_model(cfg)
    
    # Étape 4: Validation
    metrics = validate_model(model)
    
    # Étape 5: Exportation
    export_dir = export_model(model)
    
    # Résumé
    print("\n" + "=" * 60)
    print("🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
    print("=" * 60)
    print(f"📁 Modèle exporté dans: {export_dir}")
    print(f"📊 mAP50: {metrics.box.map50:.4f}")
    print(f"📊 mAP50-95: {metrics.box.map:.4f}")
    
    # Lancer détection temps réel ?
    print("\n🚀 Lancer la détection webcam maintenant?")
    response = input("Appuyez sur Entrée pour continuer (ou 'n' pour arrêter): ")
    
    if response.lower() != 'n':
        best_model = export_dir / "best.pt"
        detect_posture(str(best_model))

if __name__ == "__main__":
    main()