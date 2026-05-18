#!/usr/bin/env python3
"""
Optimized YOLOv8n benchmark with BATCH PROCESSING on GPU.
Accumulates 4-8 frames and sends them as a single batch to the GPU.
Expected: 60-80 FPS (2.5-3x improvement over single-frame inference).
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil
import tqdm

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except Exception:
    PYNVML_AVAILABLE = False


def get_gpu_stats():
    """Get GPU utilization and memory using pynvml (no subprocess)."""
    if not PYNVML_AVAILABLE:
        return 0, 0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return util.gpu, mem.used / 1024 / 1024  # GPU %, GPU MB
    except Exception:
        return 0, 0


def preprocess_frame(frame, size=(320, 320)):
    """Preprocess single frame to ONNX input format."""
    # Resize
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize [0, 255] -> [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    # Add batch dim and move to (B, C, H, W)
    blob = np.transpose(normalized, (2, 0, 1))  # (C, H, W)
    return blob


def preprocess_batch(frames, size=(320, 320)):
    """Preprocess multiple frames into a batch tensor."""
    blobs = []
    for frame in frames:
        blob = preprocess_frame(frame, size)
        blobs.append(blob)
    # Stack into batch: (B, C, H, W)
    batch = np.stack(blobs, axis=0).astype(np.float32)
    return batch


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n GPU Batch Benchmark")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--device", default="onnx-cuda", help="Device: onnx-cpu, onnx-cuda")
    parser.add_argument("--input", default="0", help="Input source: 0 (webcam) or path to video")
    parser.add_argument("--duration", type=int, default=30, help="Benchmark duration (seconds)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup duration (seconds)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (frames per GPU call)")
    parser.add_argument("--out", default="results_batch.csv", help="Output CSV file")
    args = parser.parse_args()

    # Load model
    print(f"Loading session... {args.device}")
    if args.device == "onnx-cuda":
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"
    
    session = ort.InferenceSession(args.model, providers=[provider])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input shape: {input_shape}, Provider: {provider}")

    # Input size from model (handle dynamic shapes)
    input_size = (320, 320)  # YOLOv8n standard size
    if len(input_shape) == 4:
        # Try to extract from shape (may contain None for dynamic dims)
        try:
            _, _, h, w = input_shape
            if h is not None and w is not None:
                input_size = (int(w), int(h))
        except (TypeError, ValueError):
            # Dynamic shapes use strings like 'height', 'width' - ignore
            pass

    print(f"Input size: {input_size}")
    print(f"Batch size: {args.batch_size}")

    # GPU availability
    gpu_available = PYNVML_AVAILABLE
    print(f"GPU monitoring: {'Available (pynvml)' if gpu_available else 'Unavailable'}")

    # Open input (webcam or video)
    if args.input == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print("Error: Cannot open input")
        return 1

    print(f"nvidia-smi available: {PYNVML_AVAILABLE}")

    # Warmup
    print(f"Warming up for {args.warmup}s...")
    start_warmup = time.monotonic()
    warmup_frames = 0
    frame_buffer = []
    
    while time.monotonic() - start_warmup < args.warmup:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_buffer.append(frame)
        warmup_frames += 1
        
        # Process batch
        if len(frame_buffer) == args.batch_size:
            batch = preprocess_batch(frame_buffer, input_size)
            _ = session.run(None, {input_name: batch})
            frame_buffer.clear()

    print(f"Benchmark running for {args.duration} seconds...")

    # Benchmark
    results = []
    start_time = time.monotonic()
    end_time = start_time + args.duration
    frame_idx = 0
    frame_buffer = []

    with tqdm.tqdm(total=args.duration, unit="s", desc="Running", smoothing=0.1) as pbar:
        while time.monotonic() < end_time:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_buffer.append(frame)

            # Process batch when full
            if len(frame_buffer) == args.batch_size:
                batch = preprocess_batch(frame_buffer, input_size)
                
                # Measure inference time
                t0 = time.perf_counter()
                outputs = session.run(None, {input_name: batch})
                infer_ms = (time.perf_counter() - t0) * 1000

                # Get system stats
                cpu_percent = psutil.cpu_percent(interval=0.01)
                mem_rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
                gpu_util, gpu_mem_mb = get_gpu_stats()

                # Record each frame in batch
                for i, preprocessed_frame in enumerate(frame_buffer):
                    # Note: All frames in batch share the same inference time
                    # In real scenario, you'd distribute latency or measure per-frame
                    results.append({
                        "timestamp": time.time(),
                        "frame_idx": frame_idx + i,
                        "infer_ms": infer_ms / args.batch_size,  # Amortized latency
                        "total_ms": (time.perf_counter() - t0) * 1000 / args.batch_size,
                        "cpu_percent": cpu_percent,
                        "mem_rss_mb": mem_rss_mb,
                        "gpu_util": gpu_util,
                        "gpu_mem_mb": gpu_mem_mb,
                    })

                frame_idx += args.batch_size
                frame_buffer.clear()

                # Update progress
                elapsed = time.monotonic() - start_time
                pbar.update(elapsed - pbar.n)

    cap.release()

    # Save results
    print(f"Benchmark finished. Results saved to {args.out}")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "frame_idx", "infer_ms", "total_ms",
            "cpu_percent", "mem_rss_mb", "gpu_util", "gpu_mem_mb"
        ])
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    if results:
        infer_times = [r["infer_ms"] for r in results]
        fps = len(results) / (results[-1]["timestamp"] - results[0]["timestamp"])
        print(f"\n{'='*50}")
        print(f"BATCH PROCESSING RESULTS (Batch Size: {args.batch_size})")
        print(f"{'='*50}")
        print(f"Frames Processed: {len(results)}")
        print(f"Effective FPS: {fps:.2f}")
        print(f"Avg Inference (amortized): {np.mean(infer_times):.2f} ms")
        print(f"Min/Max: {np.min(infer_times):.2f} / {np.max(infer_times):.2f} ms")
        print(f"Avg CPU: {np.mean([r['cpu_percent'] for r in results]):.1f}%")
        print(f"Avg GPU Mem: {np.mean([r['gpu_mem_mb'] for r in results]):.0f} MB")

    return 0


if __name__ == "__main__":
    exit(main())
