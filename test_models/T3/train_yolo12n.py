"""
Fine-tuning YOLOv12 pour la détection de chutes (Fall Detection)
Dataset: Le2i Fall Detection Dataset (Roboflow)
Architecture: YOLOv12 Nano (yolo12n.pt)
"""

import os
import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import shutil
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

# Chemins du projet
PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Créer les répertoires nécessaires
for dir_path in [RUNS_DIR, WEIGHTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configuration du dataset
DATA_YAML_PATH = DATA_DIR / "data.yaml"

# Configuration du modèle
MODEL_NAME = "yolo12n.pt"  # YOLOv12 Nano
# Alternatives: yolo12s.pt, yolo12m.pt, yolo12l.pt, yolo12x.pt

# Hyperparamètres d'entraînement
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
LEARNING_RATE = 0.001
WORKERS = 8

# ============================================================
# ÉTAPE 1: VÉRIFICATION DE L'ENVIRONNEMENT
# ============================================================

def check_environment():
    """Vérifie la disponibilité du GPU et les dépendances."""
    print("=" * 60)
    print("VÉRIFICATION DE L'ENVIRONNEMENT")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA disponible: {cuda_available}")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}") 
        print(f"Mémoire GPU: {gpu_memory:.2f} GB")
        print(f"Nombre de GPUs: {torch.cuda.device_count()}")
    else:
        print("⚠️  Aucun GPU détecté - L'entraînement sera lent sur CPU")
    
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics non installé. Exécutez: pip install ultralytics")
        raise
    
    print("=" * 60)
    return cuda_available

# ============================================================
# ÉTAPE 2: VÉRIFICATION DU DATASET
# ============================================================

def verify_dataset():
    """Vérifie la structure et la validité du dataset."""
    print("\n" + "=" * 60)
    print("VÉRIFICATION DU DATASET LE2I")
    print("=" * 60)
    
    if not DATA_YAML_PATH.exists():
        raise FileNotFoundError(f"Fichier data.yaml non trouvé: {DATA_YAML_PATH}")
    
    # Lire la configuration du dataset
    with open(DATA_YAML_PATH, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Configuration du dataset:")
    print(f"  - Classes: {data_config.get('names', 'Non défini')}")
    print(f"  - Nombre de classes: {data_config.get('nc', 'Non défini')}")
    print(f"  - Chemin train: {data_config.get('train', 'Non défini')}")
    print(f"  - Chemin val: {data_config.get('val', 'Non défini')}")
    print(f"  - Chemin test: {data_config.get('test', 'Non défini')}")
    print(f"  - Source Roboflow: {data_config.get('roboflow', {}).get('url', 'Non défini')}")
    
    # Vérifier les répertoires
    splits = ['train', 'valid', 'test']
    for split in splits:
        # Construire le chemin absolu depuis data.yaml
        relative_path = data_config.get(split, f'../{split}/images')
        split_path = DATA_DIR / relative_path.replace('../', '')
        
        if not split_path.exists():
            print(f"⚠️  Répertoire {split} non trouvé à {split_path}")
            continue
            
        num_images = len(list(split_path.glob("*.*")))
        print(f"  - {split}/images: {num_images} images")
        
        # Vérifier les labels correspondants
        labels_path = split_path.parent / "labels"
        if labels_path.exists():
            num_labels = len(list(labels_path.glob("*.txt")))
            print(f"  - {split}/labels: {num_labels} annotations")
        else:
            print(f"  - {split}/labels: ❌ Non trouvé")
    
    print("=" * 60)
    return data_config

# ============================================================
# ÉTAPE 3: ENTRAÎNEMENT (FINE-TUNING)
# ============================================================

def train_model(data_config):
    """Lance l'entraînement du modèle YOLOv12."""
    print("\n" + "=" * 60)
    print("DÉBUT DE L'ENTRAÎNEMENT YOLOv12")
    print("=" * 60)
    
    # Charger le modèle pré-entraîné
    print(f"Chargement du modèle: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    
    # Configuration de l'entraînement optimisée pour une classe
    training_args = {
        # Chemins et données
        "data": str(DATA_YAML_PATH),
        "project": str(RUNS_DIR),
        "name": "fall_detection_yolo12",
        
        # Hyperparamètres principaux
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
        "imgsz": IMAGE_SIZE,
        
        # Optimisation
        "lr0": LEARNING_RATE,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        
        # Augmentations pour détection de chutes
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        
        # Configuration système
        "workers": WORKERS,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "cache": True,
        
        # Validation et sauvegarde
        "val": True,
        "save": True,
        "save_period": 10,
        "patience": 20,
        
        # Journalisation
        "verbose": True,
        "seed": 42,
    }
    
    print("Configuration d'entraînement:")
    for key, value in training_args.items():
        print(f"  - {key}: {value}")
    
    # Lancer l'entraînement
    print("\n🚀 Lancement de l'entraînement...")
    results = model.train(**training_args)
    
    print("\n✅ Entraînement terminé!")
    return results, model

# ============================================================
# ÉTAPE 4: VALIDATION
# ============================================================

def validate_model(model, data_config):
    """Valide le modèle entraîné sur le jeu de test."""
    print("\n" + "=" * 60)
    print("PHASE DE VALIDATION")
    print("=" * 60)
    
    # Valider sur le jeu de test
    metrics = model.val(
        data=str(DATA_YAML_PATH),
        split='test',
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=0.25,
        iou=0.45,
        device=0 if torch.cuda.is_available() else "cpu"
    )
    
    print("\n📊 Métriques de validation:")
    print(f"  - mAP50: {metrics.box.map50:.4f}")
    print(f"  - mAP50-95: {metrics.box.map:.4f}")
    print(f"  - mAP75: {metrics.box.map75:.4f}")
    print(f"  - Precision: {metrics.box.p:.4f}")
    print(f"  - Recall: {metrics.box.r:.4f}")
    
    return metrics

# ============================================================
# ÉTAPE 5: EXPORTATION DU MODÈLE
# ============================================================

def export_model(model):
    """Exporte le modèle dans différents formats."""
    print("\n" + "=" * 60)
    print("EXPORTATION DU MODÈLE")
    print("=" * 60)
    
    # Créer le répertoire de sortie
    export_dir = WEIGHTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Copier le meilleur modèle
    best_pt = Path(model.trainer.best) if hasattr(model, 'trainer') else None
    if best_pt and best_pt.exists():
        shutil.copy(best_pt, export_dir / "best.pt")
        print(f"✅ Modèle PyTorch sauvegardé: {export_dir / 'best.pt'}")
    
    # Export ONNX
    try:
        onnx_path = model.export(format="onnx", imgsz=IMAGE_SIZE)
        shutil.move(onnx_path, export_dir / "best.onnx")
        print(f"✅ Modèle ONNX exporté: {export_dir / 'best.onnx'}")
    except Exception as e:
        print(f"⚠️  Échec export ONNX: {e}")
    
    # Export TensorRT (si GPU NVIDIA disponible)
    if torch.cuda.is_available():
        try:
            engine_path = model.export(format="engine", imgsz=IMAGE_SIZE, half=True)
            shutil.move(engine_path, export_dir / "best.engine")
            print(f"✅ Modèle TensorRT exporté: {export_dir / 'best.engine'}")
        except Exception as e:
            print(f"⚠️  Échec export TensorRT: {e}")
    
    print(f"\n📁 Tous les modèles exportés dans: {export_dir}")
    return export_dir

# ============================================================
# ÉTAPE 6: FONCTION DE PRÉDICTION/INFÉRENCE
# ============================================================

def predict_fall(model_path, source, conf_threshold=0.5, save=True):
    """
    Fonction d'inférence pour détecter les chutes.
    
    Args:
        model_path: Chemin vers le modèle entraîné (.pt)
        source: Chemin vers image/vidéo/webcam (0 pour webcam)
        conf_threshold: Seuil de confiance
        save: Sauvegarder les résultats
    """
    print("\n" + "=" * 60)
    print("DÉTECTION DE CHUTE - INFÉRENCE")
    print("=" * 60)
    
    # Charger le modèle
    model = YOLO(model_path)
    
    # Lancer la prédiction
    results = model.predict(
        source=source,
        conf=conf_threshold,
        iou=0.45,
        imgsz=IMAGE_SIZE,
        save=save,
        show=True if source == 0 else False,
        verbose=True
    )
    
    # Analyser les résultats
    for result in results:
        boxes = result.boxes
        if boxes:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                
                print(f"Détection: {class_name} (confiance: {conf:.2f})")
                
                # Alerte spéciale pour les chutes
                if class_name.lower() == 'fall':
                    print("🚨 ALERTE: CHUTE DÉTECTÉE!")
    
    return results

# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

def main():
    """Pipeline complet de fine-tuning YOLOv12 pour la détection de chutes."""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     YOLOv12 - Fine-tuning pour Détection de Chutes         ║
    ║                    Dataset: Le2i (Roboflow)                  ║
    ║                    Classe unique: FALL                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Étape 1: Vérification environnement
    cuda_available = check_environment()
    
    # Étape 2: Vérification dataset
    data_config = verify_dataset()
    
    # Étape 3: Entraînement
    results, model = train_model(data_config)
    
    # Étape 4: Validation
    metrics = validate_model(model, data_config)
    
    # Étape 5: Exportation
    export_dir = export_model(model)
    
    print("\n" + "=" * 60)
    print("🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
    print("=" * 60)
    print(f"\n📊 Résumé:")
    print(f"  - Modèle entraîné: {MODEL_NAME}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - mAP50: {metrics.box.map50:.4f}")
    print(f"  - mAP50-95: {metrics.box.map:.4f}")
    print(f"\n📁 Modèles sauvegardés dans: {export_dir}")
    
    # Exemple d'utilisation pour l'inférence
    print("\n" + "=" * 60)
    print("EXEMPLE D'UTILISATION POUR L'INFÉRENCE:")
    print("=" * 60)
    print(f"""
    # Pour tester sur une image:
    predict_fall(
        model_path="{export_dir}/best.pt",
        source="chemin/vers/image.jpg",
        conf_threshold=0.5
    )
    
    # Pour tester sur une vidéo:
    predict_fall(
        model_path="{export_dir}/best.pt",
        source="chemin/vers/video.mp4",
        conf_threshold=0.5
    )
    
    # Pour webcam en temps réel:
    predict_fall(
        model_path="{export_dir}/best.pt",
        source=0,
        conf_threshold=0.5
    )
    """)

if __name__ == "__main__":
    main()