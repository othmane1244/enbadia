# Fine-Tuning 2 - Détection de Comportements (YOLOv12)

## 📋 Description du Projet

Ce projet implémente le **fine-tuning de YOLOv12** pour la détection de comportements humains en temps réel. Le modèle est entraîné pour reconnaître plusieurs classes de comportements : **smoking, eating, sleeping, phone**.

### Caractéristiques
- ✅ Modèle léger: YOLOv12 Nano (yolo12n.pt)
- ✅ Détection en temps réel via webcam
- ✅ Support GPU (CUDA) et CPU
- ✅ Fine-tuning sur dataset personnalisé
- ✅ Export de résultats annotés

---

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip (gestionnaire de paquets Python)
- (Optionnel) GPU compatible CUDA pour accélération

### Étapes d'installation

1. **Cloner le repository** (si pas encore fait):
```bash
git clone https://github.com/0xseljarida/Embedded-IA.git
cd Embedded-IA/T1/fine_tuning2
```

2. **Créer un environnement virtuel** (recommandé):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. **Installer les dépendances**:
```bash
pip install -r requirements.txt
```

4. **Vérifier l'installation**:
```bash
python -c "import torch; print('PyTorch:', torch.__version__); from ultralytics import YOLO; print('YOLO: OK')"
```

---

## 📂 Structure des Fichiers

```
fine_tuning2/
├── README.md                    # Ce fichier
├── requirements.txt             # Dépendances Python
├── data/                        # Dataset d'entraînement
│   ├── data.yaml               # Configuration YOLO
│   ├── train/                  # Images d'entraînement
│   │   ├── images/
│   │   └── labels/             # Annotations (format YOLO)
│   ├── val/                    # Validation
│   │   ├── images/
│   │   └── labels/
│   └── test/                   # Test
│       ├── images/
│       └── labels/
├── find_model.py               # Script pour localiser les modèles
├── train_sleep.py              # Script d'entraînement principal
├── test_sleep_realtime.py      # Détection temps réel (webcam)
├── yolo12n.pt                  # Modèle pré-entraîné YOLOv12 Nano
├── yolo26n.pt                  # Modèle alternatif (YOLOv26)
├── runs_sleep/                 # Résultats d'entraînement
└── weights_sleep/              # Modèles fine-tunés sauvegardés
```

---

## 🎯 Utilisation

### 1. **Entraîner le modèle**

Lancez l'entraînement avec la configuration définie:

```bash
python train_sleep.py
```

**Paramètres configurables dans `train_sleep.py`:**
- `EPOCHS`: Nombre d'epochs (défaut: 100)
- `BATCH_SIZE`: Taille des batchs (défaut: 16)
- `IMAGE_SIZE`: Résolution des images (défaut: 640)
- `LEARNING_RATE`: Taux d'apprentissage (défaut: 0.001)

### 2. **Détection en temps réel**

Lancez la détection via webcam:

```bash
python test_sleep_realtime.py
```

**Commandes clavier:**
- `q`: Quitter le programme
- `s`: Sauvegarder une capture

**Avant de lancer:** Vérifiez le chemin du modèle dans `test_sleep_realtime.py` (ligne `MODEL_PATH`)

### 3. **Localiser les modèles**

Pour trouver automatiquement les fichiers `.pt` disponibles:

```bash
python find_model.py
```

Cela affichera tous les modèles trouvés et leur taille en MB.

---

## 📊 Classes de Détection

Le modèle détecte les classes suivantes:

| Index | Classe | Description |
|-------|--------|-------------|
| 0 | **smoking** | Personne fumant |
| 1 | **eating** | Personne mangeant |
| 2 | **sleeping** | Personne endormie |
| 3 | **phone** | Personne au téléphone |

---

## 🔧 Configuration d'Entraînement

### Data.yaml

Le fichier `data/data.yaml` définit les chemins des données:

```yaml
path: /path/to/data
train: train/images
val: val/images
test: test/images

nc: 4  # Nombre de classes
names: ['smoking', 'eating', 'sleeping', 'phone']
```

### Ressources Recommandées

| Composant | Minimum | Recommandé |
|-----------|---------|-----------|
| GPU | 2GB VRAM | 6GB+ VRAM |
| CPU | Quad-core | 8-core i7+ |
| RAM | 8GB | 16GB+ |
| Stockage | 5GB | 20GB |

---

## 🐛 Résolution de Problèmes

### ❌ "CUDA out of memory"
→ Réduisez `BATCH_SIZE` de 16 à 8 dans `train_sleep.py`

### ❌ "Modèle introuvable" (test_sleep_realtime.py)
→ Mettez à jour `MODEL_PATH` avec le chemin exact vers `best.pt`
→ Utilisez `find_model.py` pour localiser le modèle

### ❌ "Webcam non accessible"
→ Vérifiez les permissions de caméra
→ Testez avec une autre application d'accès webcam

### ❌ Erreur mémoire Windows
→ `WORKERS` est défini à 0 dans `train_sleep.py` (déjà configuré)

---

## 📈 Résultats Attendus

Après entraînement, les fichiers suivants seront générés:

```
runs_sleep/
├── sleep_yolo12/
│   ├── weights/
│   │   ├── best.pt          # Meilleur modèle
│   │   └── last.pt          # Dernier checkpoint
│   ├── results.png
│   ├── confusion_matrix.png
│   └── ...
```

---

## 🤝 Dépendances Principales

- **torch**: Framework de deep learning
- **ultralytics**: Implémentation YOLOv12
- **opencv-python**: Traitement d'images (webcam, annotations)
- **PyYAML**: Parsing des fichiers de configuration
- **numpy, matplotlib**: Traitement et visualisation de données

Voir `requirements.txt` pour la liste complète.

---

## 📝 Notes

- Le modèle YOLOv12 Nano est optimisé pour la vitesse et la légèreté
- Pour meilleure précision, utilisez `yolo12s.pt` ou `yolo12m.pt`
- Les données d'entraînement doivent être au format YOLO (`.txt` annotations)
- GPU recommandé pour entraînement rapide (CPU possible mais lent)

---

## 📧 Support

Pour toute question ou amélioration, consultez:
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [Repository GitHub](https://github.com/0xseljarida/Embedded-IA)

---

**Dernière mise à jour**: Mai 2026  
**Branche**: test_models_detection  
**Statut**: ✅ Production
