#!/usr/bin/env python3
"""Test different input resolutions for speed/accuracy tradeoff."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

try:
    import pynvml
    pynvml.nvmlInit()
except:
    pynvml = None

def benchmark_resolution(model_path, resolution, device='onnx-cuda', duration=10, webcam_idx=0):
    """Benchmark inference at given resolution."""
    
    # Load session
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'onnx-cuda' else ['CPUExecutionProvider']
    
    session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
    input_name = session.get_inputs()[0].name
    
    cap = cv2.VideoCapture(webcam_idx)
    if not cap.isOpened():
        print(f"Failed to open webcam {webcam_idx}")
        return None
    
    # Warmup
    for _ in range(3):
        ret, frame = cap.read()
        if ret:
            img = cv2.resize(frame, (resolution, resolution))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))[None]
            _ = session.run(None, {input_name: img})
    
    # Benchmark
    frames = 0
    infer_times = []
    start = time.time()
    
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        img = cv2.resize(frame, (resolution, resolution))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]
        
        t0 = time.time()
        _ = session.run(None, {input_name: img})
        t1 = time.time()
        
        infer_times.append((t1 - t0) * 1000)
        frames += 1
    
    cap.release()
    
    fps = frames / duration
    avg_infer = np.mean(infer_times)
    p95_infer = np.percentile(infer_times, 95)
    
    return {'fps': fps, 'infer_ms': avg_infer, 'p95_ms': p95_infer}

if __name__ == '__main__':
    model_path = Path('piTEST/yolov8n.onnx')
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)
    
    print("╔════════════════════════════════════════════╗")
    print("║   Resolution Speed/Accuracy Tradeoff Test   ║")
    print("╚════════════════════════════════════════════╝")
    print()
    
    resolutions = [192, 224, 256, 320, 416, 512]
    
    for res in resolutions:
        print(f"Testing {res}x{res}...", end=' ', flush=True)
        try:
            result = benchmark_resolution(model_path, res, duration=10)
            if result:
                print(f"✓ FPS: {result['fps']:.1f}  Infer: {result['infer_ms']:.2f}ms  P95: {result['p95_ms']:.2f}ms")
            else:
                print("✗ Failed")
        except Exception as e:
            print(f"✗ {e}")
    
    print()
    print("Recommendations:")
    print("  • 192x192: Fastest, good for edge devices, lower accuracy")
    print("  • 256x256: Good speed/accuracy balance")  
    print("  • 320x320: Default, best accuracy with CUDA")
    print("  • 512x512: Highest accuracy, slowest (overkill for most use cases)")
