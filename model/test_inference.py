# ============================================================
# Test d'inférence YOLO11n ONNX — Webcam temps réel
# Système de Surveillance Intelligente — ENSA Béni Mellal
# Simule le pipeline : libcamera → Hailo-8 NPU (sur PC via ONNX)
# ============================================================

import cv2
import numpy as np
import onnxruntime as ort
import time

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
ONNX_MODEL   = "yolo11n.onnx"
INPUT_SIZE   = 640          # Taille entrée modèle (640x640)
CONF_THRESH  = 0.45         # Seuil de confiance
IOU_THRESH   = 0.45         # Seuil NMS (Non-Maximum Suppression)
WEBCAM_ID    = 0            # 0 = webcam par défaut

# Classes COCO (80 classes)
CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

# Couleurs par classe (reproductibles)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

# ------------------------------------------------------------
# CHARGEMENT DU MODÈLE ONNX
# ------------------------------------------------------------
def load_model(model_path: str) -> ort.InferenceSession:
    # Priorité GPU (CUDA) → CPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    
    provider_used = session.get_providers()[0]
    print(f"✅ Modèle chargé : {model_path}")
    print(f"   Provider actif : {provider_used}")
    print(f"   Entrée  : {session.get_inputs()[0].name} {session.get_inputs()[0].shape}")
    print(f"   Sortie  : {session.get_outputs()[0].name} {session.get_outputs()[0].shape}")
    return session

# ------------------------------------------------------------
# PRÉ-TRAITEMENT (simule le pipeline CPU du RPi 5)
# ------------------------------------------------------------
def preprocess(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """
    Redimensionne et normalise le frame pour le modèle.
    Retourne : (blob, ratio, pad_w, pad_h)
    """
    h, w = frame.shape[:2]
    
    # Letterbox : resize avec padding pour garder le ratio
    ratio = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Padding centré (gris neutre)
    pad_w = (INPUT_SIZE - new_w) // 2
    pad_h = (INPUT_SIZE - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized, pad_h, INPUT_SIZE - new_h - pad_h,
        pad_w, INPUT_SIZE - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # BGR → RGB, HWC → NCHW, normalisation [0,1]
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)  # (1, 3, 640, 640)
    
    return blob, ratio, pad_w, pad_h

# ------------------------------------------------------------
# POST-TRAITEMENT : NMS (simule le CPU post-traitement RPi 5)
# ------------------------------------------------------------
def postprocess(
    output: np.ndarray,
    ratio: float,
    pad_w: int,
    pad_h: int,
    orig_w: int,
    orig_h: int
) -> list[dict]:
    """
    Décode les sorties YOLO11n et applique NMS.
    Sortie YOLO11n : (1, 84, 8400) → [cx, cy, w, h, cls0..cls79]
    """
    predictions = output[0]          # (84, 8400)
    predictions = predictions.T      # (8400, 84)
    
    boxes, scores, class_ids = [], [], []
    
    for pred in predictions:
        cx, cy, bw, bh = pred[:4]
        class_probs = pred[4:]
        class_id = int(np.argmax(class_probs))
        confidence = float(class_probs[class_id])
        
        if confidence < CONF_THRESH:
            continue
        
        # Convertir cx,cy,w,h → x1,y1,x2,y2 (coordonnées image originale)
        x1 = int((cx - bw / 2 - pad_w) / ratio)
        y1 = int((cy - bh / 2 - pad_h) / ratio)
        x2 = int((cx + bw / 2 - pad_w) / ratio)
        y2 = int((cy + bh / 2 - pad_h) / ratio)
        
        # Clamp aux dimensions de l'image
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)
        
        boxes.append([x1, y1, x2 - x1, y2 - y1])  # format xywh pour NMS
        scores.append(confidence)
        class_ids.append(class_id)
    
    if not boxes:
        return []
    
    # NMS OpenCV
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, IOU_THRESH)
    
    detections = []
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        detections.append({
            "class_id"  : class_ids[i],
            "class_name": CLASSES[class_ids[i]],
            "confidence": round(scores[i], 3),
            "bbox"      : [x, y, x + w, y + h]   # [x1, y1, x2, y2]
        })
    
    return detections

# ------------------------------------------------------------
# DESSIN DES DÉTECTIONS
# ------------------------------------------------------------
def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        label  = f"{det['class_name']} {det['confidence']:.2f}"
        color  = tuple(int(c) for c in COLORS[cls_id])
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Fond du label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame

# ------------------------------------------------------------
# OVERLAY INFO (FPS + nb détections)
# ------------------------------------------------------------
def draw_overlay(frame: np.ndarray, fps: float, n_det: int) -> np.ndarray:
    h, w = frame.shape[:2]
    
    # Bandeau supérieur semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Detections: {n_det}", (160, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, "YOLO11n ONNX | Simulation pipeline RPi5+Hailo-8",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return frame

# ------------------------------------------------------------
# BOUCLE PRINCIPALE
# ------------------------------------------------------------
def main():
    print("\n=== Surveillance IA — Test ONNX Webcam ===")
    print(f"   Modèle    : {ONNX_MODEL}")
    print(f"   Conf seuil: {CONF_THRESH}")
    print(f"   IOU seuil : {IOU_THRESH}")
    print("   Appuyer sur [Q] pour quitter\n")
    
    # Chargement modèle
    session = load_model(ONNX_MODEL)
    input_name = session.get_inputs()[0].name
    
    # Ouverture webcam
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la webcam (ID={WEBCAM_ID})")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"✅ Webcam ouverte : {int(cap.get(3))}x{int(cap.get(4))}")
    
    # Stats FPS
    fps        = 0.0
    frame_count = 0
    t_start    = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Impossible de lire le frame webcam")
            break
        
        orig_h, orig_w = frame.shape[:2]
        t0 = time.perf_counter()
        
        # --- Pipeline (simule RPi 5) ---
        # Étape 1 : Pré-traitement CPU
        blob, ratio, pad_w, pad_h = preprocess(frame)
        
        # Étape 2 : Inférence ONNX (→ Hailo-8 NPU sur RPi 5)
        outputs = session.run(None, {input_name: blob})
        
        # Étape 3 : Post-traitement + NMS CPU
        detections = postprocess(outputs[0], ratio, pad_w, pad_h, orig_w, orig_h)
        
        # Calcul FPS
        t1 = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / (t1 - t0 + 1e-6))  # lissage
        
        # --- Affichage ---
        frame = draw_detections(frame, detections)
        frame = draw_overlay(frame, fps, len(detections))
        
        cv2.imshow("Surveillance IA — YOLO11n ONNX (simulation RPi5+Hailo-8)", frame)
        
        # Log console toutes les 30 frames
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - t_start
            print(f"[{elapsed:6.1f}s] FPS: {fps:5.1f} | Détections: {len(detections)}")
            for d in detections:
                print(f"         → {d['class_name']:15s} conf={d['confidence']:.3f} bbox={d['bbox']}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test terminé.")

if __name__ == "__main__":
    main()