# Trois Simulateurs : Guide Comparatif

Le projet contient **trois fichiers simulateur** avec des objectifs différents. Ce document clarifie lequel utiliser et quand.

---

## 📋 Tableau Comparatif

| Feature | `simulator.py` | `simulator_video_only.py` | (Ancien ONNX) |
|---------|---|---|---|
| **Localisation** | `model/simulator.py` | `model/simulator_video_only.py` | ❌ Déprécié |
| **Inférence** | ✅ YOLO11n (ultralytics) | ❌ Aucune | ONNX (manquant) |
| **Détections** | Boîtes, confiance, classes | Non | ONNX |
| **Zones interdites** | ✅ Depuis Supabase | ❌ Non | - |
| **Streaming vidéo** | ✅ POST /video/frame | ✅ POST /video/frame | - |
| **API détections** | ✅ POST /process_frame/ | ❌ Non | - |
| **Affichage local** | ✅ OpenCV window | ✅ OpenCV window | - |
| **Dépendances** | ultralytics, Supabase | Aucune ML | onnxruntime |
| **Cas d'usage** | 🎯 **PRODUCTION** | 🔧 Tests vidéo | ❌ Non utilisé |
| **État** | ✅ Actif | ✅ Actif (fallback) | ❌ Archivé |

---

## 🚀 Quel Simulateur Utiliser ?

### **1. `simulator.py` — RECOMMANDÉ pour la production** ✅

**Quand l'utiliser :**
- 🎯 Environnement complet : détections YOLO + zones interdites + streaming vidéo
- 🔗 Supabase configuré et zones créées via le dashboard
- 📊 Besoin d'alertes comportementales (/process_frame/ API)
- 💻 Webcam disponible sur la machine de développement

**Commande :**
```bash
cd c:\Users\user\Documents\projet\PROJET\ JARIDA\Embedded-IA
python model/simulator.py
```

**Flux :**
```
Webcam → YOLO11n inférence → Dessine détections + zones Supabase
         → POST /video/frame (dashboard)
         → POST /process_frame/ (alertes API)
```

**Démarrage complet du système :**
```bash
# Terminal 1 : Backend FastAPI
python -m uvicorn main:app --reload

# Terminal 2 : Dashboard Next.js
cd dashboard && npm run dev

# Terminal 3 : Simulateur (dès que le backend est prêt)
python model/simulator.py
```

---

### **2. `simulator_video_only.py` — Fallback rapide** 🔧

**Quand l'utiliser :**
- 🧪 Test du streaming vidéo sans dépendance YOLO/Supabase
- 🔴 YOLO ne s'installe pas ou modèle manquant
- 🔴 Supabase non configuré
- ⚡ Vérification rapide du pipeline vidéo (cadre → base64 → WebSocket)

**Commande :**
```bash
python model/simulator_video_only.py
```

**Flux :**
```
Webcam → Encode JPEG base64 → POST /video/frame
         → Dashboard reçoit les frames (pas de détections)
```

**Avantages :**
- 🪶 Poids léger : ~50 lignes, zéro dépendances ML
- ⚡ Démarre instantanément
- 🐛 Isoler les problèmes de streaming vs détections

---

### **3. (Ancien Simulateur ONNX)** ❌

**⚠️ Déprécié — NE PAS UTILISER**

Raisons :
- 📦 Modèle `yolo11n.onnx` introuvable (26MB, non committé)
- 🔧 Setup complexe : onnxruntime + DirectML
- 🚀 YOLO11n ultralytics est plus rapide et plus simple
- ✅ Remplacé par `simulator.py`

---

## 🔄 Matrice de Dépannage

| Problème | Solution |
|----------|----------|
| "YOLO model not found" | `simulator_video_only.py` OU installer ultralytics : `pip install ultralytics` |
| "Supabase not connected" | Utiliser `simulator_video_only.py` (zones non affichées) OU configurer `.env` |
| "API détections non reçues" | Vérifier `/ws/video` et `/ws/alerts` connectés dans main.py |
| "Webcam introuvable (ID=0)" | Vérifier : `cv2.VideoCapture(0)` avec `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"` |
| "Frames arrivent sur dashboard" | ✅ Bon ! Streaming OK. Vérifier détections dans browser console. |

---

## 📝 Exemple de Flux Complet

**Démarrage 1 : Production complète (YOLO + Zones)**

```bash
# Terminal 1 — Backend
cd "c:\Users\user\Documents\projet\PROJET JARIDA\Embedded-IA"
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — Dashboard
cd dashboard
npm run dev
# → Accessible à http://localhost:3000

# Terminal 3 — Simulateur avec détections
python model/simulator.py
# → Logs : "✅ X zone(s) interdite(s) chargée(s) de Supabase"
# → Dashboard affiche video avec boîtes rouges (détections) + zones en rouge
```

**Résultat attendu :**
- ✅ Video stream en direct (30 FPS)
- ✅ Boîtes de détection colorées par classe
- ✅ Labels avec confiance (ex: "person 0.95")
- ✅ Zones interdites rouges du dashboard
- ✅ Alertes dans le dashboard si comportements détectés

---

**Démarrage 2 : Tests rapides (Vidéo uniquement)**

```bash
# Terminal 1 — Backend (minimal, juste /video/frame endpoint)
python -m uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Dashboard
npm run dev

# Terminal 3 — Simulateur léger (pas YOLO)
python model/simulator_video_only.py
# → Logs : "📹 30 frames envoyées"
# → Dashboard affiche video, pas de boîtes (pas d'inférence)
```

---

## 🎯 Checklist Avant de Lancer

- [ ] FastAPI backend en cours d'exécution (`python -m uvicorn main:app`)
- [ ] Dashboard Next.js démarré (`npm run dev`)
- [ ] `.env` configuré avec Supabase (pour `simulator.py`)
- [ ] Zones créées dans le dashboard (pour `simulator.py`)
- [ ] Webcam connectée et fonctionnelle
- [ ] Port 8000 et 3000/3001 disponibles

---

## 🔧 Fichiers Modifiés

- **model/simulator.py** → Intègre ultralytics YOLO + récupère zones Supabase
- **model/simulator_video_only.py** → Reste inchangé (fallback)
- **database.py** → Nouvelle fonction `fetch_zones()` pour Supabase
- **dashboard/components/VideosContent.tsx** → Layout corrigé pour les boutons

---

**Version:** 1.0 | **Date:** Mai 2026 | **État:** Production Ready ✅
