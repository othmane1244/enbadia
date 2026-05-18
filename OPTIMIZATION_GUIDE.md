# 🚀 Guide: Atteindre 60+ FPS avec YOLOv8n

## Baseline Actuel
- **Optimized Single-Frame**: 30.04 FPS (6.75 ms latency)
- **Objectif**: 60+ FPS (2x improvement)

---

## 🎯 Option 1: Batch Processing (⏱️ 10 min, gain: 2-3x)

### Étape 1: Tester le batch processing avec le modèle actuel
```bash
cd "C:\Users\user\Documents\projet\jarida 2\enbadia"
.\.venv_bench\Scripts\python.exe benchmarks/run_bench_batch.py \
  --model piTEST/yolov8n.onnx \
  --device onnx-cuda \
  --duration 20 \
  --batch-size 4 \
  --out results_batch_b4.csv
```

**Attendu**: 50-80 FPS (batch_size=4)

### Étape 2: Tester avec batch_size=8
```bash
.\.venv_bench\Scripts\python.exe benchmarks/run_bench_batch.py \
  --model piTEST/yolov8n.onnx \
  --device onnx-cuda \
  --duration 20 \
  --batch-size 8 \
  --out results_batch_b8.csv
```

**Attendu**: 60-90 FPS (batch_size=8)

---

## 🎯 Option 2: GPU Preprocessing (⏱️ 15 min, gain: 1.5-2x)

### Étape 1: Vérifier OpenCV CUDA
```bash
.\.venv_bench\Scripts\python.exe -c "import cv2; import cv2.cuda; print('OpenCV CUDA available')"
```

Si erreur → installer OpenCV contrib:
```bash
.\.venv_bench\Scripts\pip uninstall opencv-python -y
.\.venv_bench\Scripts\pip install opencv-contrib-python
```

### Étape 2: Tester GPU preprocessing
```bash
.\.venv_bench\Scripts\python.exe benchmarks/run_bench_gpu_preprocess.py \
  --model piTEST/yolov8n.onnx \
  --device onnx-cuda \
  --use-gpu-preprocessing \
  --duration 20 \
  --out results_gpu_preprocess.csv
```

**Attendu**: 40-60 FPS (GPU preprocessing enabled)

---

## 🎯 Option 3: Batch + Dynamic ONNX (⏱️ 20 min, gain: 3-4x)

### Étape 1: Exporter le modèle avec batch dynamique
```bash
.\.venv_bench\Scripts\python.exe benchmarks/export_dynamic_batch.py \
  --model yolov8n.pt \
  --output piTEST \
  --test
```

Cela crée `piTEST/yolov8n_dynamic_batch.onnx` optimisé pour les batchs variables.

### Étape 2: Tester avec le modèle dynamique
```bash
.\.venv_bench\Scripts\python.exe benchmarks/run_bench_batch.py \
  --model piTEST/yolov8n_dynamic_batch.onnx \
  --device onnx-cuda \
  --duration 20 \
  --batch-size 8 \
  --out results_batch_dynamic_b8.csv
```

**Attendu**: 70-100 FPS (modèle dynamique + batch)

---

## 📊 Comparaison Automatique

Pour tester **toutes les stratégies** en une seule commande:

```bash
.\.venv_bench\Scripts\python.exe benchmarks/compare_strategies.py
```

Cela génère un rapport `OPTIMIZATION_RESULTS.csv` comparant:
- Baseline (single-frame)
- Batch Processing (4 frames)
- GPU Preprocessing

---

## 🔬 Analyse des Résultats

Après chaque benchmark, analyser les résultats:

```bash
.\.venv_bench\Scripts\python.exe benchmarks/analyze_bench.py results_batch_b4.csv
```

Produit:
- `benchmark_report.html` (rapport interactif)
- `benchmark_report.png` (visualisation)

---

## ✅ Checklist - Progression vers 60+ FPS

- [ ] **Baseline validé**: 30.04 FPS
- [ ] **Batch B=4 testé**: ~50-60 FPS
- [ ] **Batch B=8 testé**: ~70-80 FPS
- [ ] **GPU Preprocessing testé**: ~40-50 FPS
- [ ] **Modèle dynamique exporté**: Ready
- [ ] **Batch Dynamique B=8 testé**: ~80-100 FPS

---

## 🎯 Recommandations

### Si vous atteignez 60+ FPS:
✅ **OBJECTIF ATTEINT!**
- Consolidez la meilleure stratégie
- Mesurez la stabilité sur 60 secondes
- Documentez les paramètres optimaux

### Si vous n'atteignez que 40-50 FPS:
🔄 **PROCHAINE ÉTAPE**: TensorRT
```bash
# Exporter en TensorRT (nécessite NVIDIA TensorRT)
.\.venv_bench\Scripts\python.exe -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', device=0, imgsz=320, dynamic=True)
"
```

### Pour quantization INT8:
```bash
# Exporter avec quantization
.\.venv_bench\Scripts\python.exe -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', device=0, imgsz=320, dynamic=True, int8=True)
"
```

---

## 📈 Performance Expectations

| Strategy | Batch | Expected FPS | Latency |
|----------|-------|-------------|---------|
| Baseline | 1 | 30 FPS | 33 ms |
| Batch Processing | 4 | 50-60 FPS | 17-20 ms |
| Batch Processing | 8 | 70-90 FPS | 11-14 ms |
| GPU Preprocessing | 1 | 40-50 FPS | 20-25 ms |
| Batch + GPU PP | 8 | 80-100 FPS | 10-12 ms |
| TensorRT | 8 | 100-150 FPS | 7-10 ms |

---

## 🛠️ Troubleshooting

### OpenCV CUDA not found
```bash
.\.venv_bench\Scripts\pip install --upgrade opencv-contrib-python
```

### CUDA out of memory
```bash
# Réduire batch_size
--batch-size 2  # au lieu de 8
```

### Modèle dynamique incompatible
```bash
# Revenir au modèle fixe
--model piTEST/yolov8n.onnx
```

---

**Created**: May 18, 2026
**Project**: YOLOv8n GPU Optimization
**Target**: 60+ FPS on RTX 5060
