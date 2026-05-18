#!/usr/bin/env python3
"""Batch processing simulation benchmark - test different batch sizes."""
import argparse
import time
from pathlib import Path
import csv

import cv2
import numpy as np
import onnxruntime as ort
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except:
    HAS_PYNVML = False


def load_session(model_path: Path, device: str):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'onnx-cuda' else ['CPUExecutionProvider']
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
    input_name = session.get_inputs()[0].name
    return session, input_name


def preprocess_frame(frame, input_size=320):
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img)


def benchmark_batch_simulation(model_path: Path, device: str, batch_size: int = 1, num_frames: int = 300):
    """Simulate batch processing by running inference on accumulated frames."""
    
    session, input_name = load_session(model_path, device)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return None
    
    # Read frames into memory first
    print(f"📸 Reading {num_frames} frames from webcam...")
    frames_list = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames_list.append(frame)
    cap.release()
    
    print(f"✓ Loaded {len(frames_list)} frames")
    
    # Simulate batching on stored frames
    print(f"\nBenchmarking batch_size={batch_size}...")
    
    total_infer_time = 0
    total_frames = 0
    
    for i in range(0, len(frames_list), batch_size):
        batch_frames = frames_list[i:i+batch_size]
        actual_batch_size = len(batch_frames)
        
        # Preprocess batch (sequential for current model limitation)
        preprocessed = [preprocess_frame(f, 320) for f in batch_frames]
        
        # Current model: batch_size=1 only, so run once per frame
        t0 = time.time()
        for blob in preprocessed:
            _ = session.run(None, {input_name: blob[None]})
        t1 = time.time()
        
        total_infer_time += (t1 - t0)
        total_frames += actual_batch_size
    
    fps = total_frames / total_infer_time if total_infer_time > 0 else 0
    avg_latency = (total_infer_time / total_frames) * 1000.0
    
    return {
        'batch_size': batch_size,
        'fps': fps,
        'avg_latency_ms': avg_latency,
        'total_frames': total_frames,
        'duration_s': total_infer_time
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='piTEST/yolov8n.onnx')
    p.add_argument('--device', choices=['onnx-cpu', 'onnx-cuda'], default='onnx-cuda')
    p.add_argument('--frames', type=int, default=300)
    args = p.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        exit(1)
    
    print("╔════════════════════════════════════════════════╗")
    print("║    Batch Processing Simulation Benchmark       ║")
    print("╚════════════════════════════════════════════════╝")
    print()
    
    batch_sizes = [1, 2, 4, 8]
    results = []
    
    for bs in batch_sizes:
        result = benchmark_batch_simulation(model_path, args.device, batch_size=bs, num_frames=args.frames)
        if result:
            results.append(result)
            print(f"  Batch {bs}: {result['fps']:.2f} FPS, {result['avg_latency_ms']:.2f}ms latency")
        print()
    
    if results:
        best = max(results, key=lambda x: x['fps'])
        print(f"🏆 Best: Batch size {best['batch_size']} → {best['fps']:.2f} FPS")
        print()
        print("Note: Current ONNX model has fixed batch=1.")
        print("      With dynamic batch model, expected 2-4x improvement per frame.")
