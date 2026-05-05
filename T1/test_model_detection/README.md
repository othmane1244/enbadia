# Posture Detection - YOLOv12 Model

## Description
Ce dossier contient un modèle YOLOv12 fine-tuné pour la détection de posture en temps réel. Le modèle peut détecter deux classes : **Supine** et **Not_Supine**.

## Objectif
Permettre à vos collègues de tester le modèle de détection de posture sur leurs machines **sans avoir besoin de faire le fine-tuning** eux-mêmes.

## Contenu du dossier

- **test_posture_realtime.py** : Script principal pour tester la détection de posture en temps réel via webcam
- **train_posture.py** : Script d'entraînement (optionnel, pour le fine-tuning)
- **yolo12n.pt** : Modèle YOLO pré-entraîné de base
- **yolo26n.pt** : Modèle YOLO alternatif
- **data/** : Dossier contenant le dataset d'entraînement et de validation

## Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/0xseljarida/Embedded-IA.git
cd Embedded-IA/T1/test_model_detection
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

## Utilisation

### Tester la détection en temps réel
```bash
python test_posture_realtime.py
```

**Commandes disponibles :**
- `q` : Quitter l'application
- `s` : Sauvegarder une capture d'écran

### Entraîner le modèle (optionnel)
```bash
python train_posture.py
```

## Configuration

Vous pouvez modifier les paramètres suivants dans `test_posture_realtime.py` :

- **MODEL_PATH** : Chemin vers le modèle entraîné
- **CONF_THRESHOLD** : Seuil de confiance pour les prédictions (par défaut : 0.5)

## Modèle

- **Architecture** : YOLOv12
- **Dataset** : Roboflow Posture Dataset
- **Classes détectables** : 
  - `Supine` (allongé)
  - `Not_Supine` (pas allongé)

## Prérequis

- Python 3.8+
- Webcam (pour la détection en temps réel)
- Dépendances listées dans `requirements.txt`

## Support

Si vous avez des questions ou des problèmes :
1. Vérifiez que toutes les dépendances sont correctement installées
2. Assurez-vous que votre webcam fonctionne
3. Consultez les logs pour des détails sur les erreurs

## Auteur
Ce modèle et ce code ont été développés pour faciliter le déploiement de la détection de posture en temps réel.
