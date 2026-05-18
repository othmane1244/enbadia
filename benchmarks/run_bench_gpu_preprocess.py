#!/usr/bin/env python3
"""
GPU Preprocessing Benchmark - Move resize/normalize to GPU using OpenCV CUDA.
Requires: pip install opencv-contrib-python
Expected gain: 2-3x speedup by offloading preprocessing to GPU.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
import tqdm

try:
    import cv2.cuda as cuda
    CUDA_AVAILABLE = cuda.getCudaEnabledDeviceCount() > 0
    cv2_cuda = cv2.cuda
except (ImportError, AttributeError):
    CUDA_AVAILABLE = False
    cv2_cuda = None

import cv2

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except Exception:
    PYNVML_AVAILABLE = False


def get_gpu_stats():
    """Get GPU utilization and memory using pynvml."""
    if not PYNVML_AVAILABLE:
        return 0, 0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return util.gpu, mem.used / 1024 / 1024
    except Exception:
        return 0, 0


def preprocess_frame_cpu(frame, size=(320, 320)):
    """CPU preprocessing (baseline)."""
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    blob = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(blob, axis=0).astype(np.float32)


def preprocess_frame_gpu(frame, size=(320, 320)):
    """GPU preprocessing using OpenCV CUDA (if available)."""
    if not CUDA_AVAILABLE or cv2_cuda is None:
        return preprocess_frame_cpu(frame, size)
    
    try:
        # Upload to GPU
        gpu_frame = cv2_cuda.GpuMat()
        gpu_frame.upload(frame)
        
        # Resize on GPU
        gpu_resized = cv2_cuda.resize(gpu_frame, size)
        
        # Convert BGR to RGB on GPU (if available)
        # Note: cv2.cuda doesn't have cvtColor in all versions
        # Fallback to CPU if needed
        try:
            gpu_rgb = cv2_cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2RGB)
        except AttributeError:
            # Fallback: download and convert on CPU
            resized_cpu = gpu_resized.download()
            gpu_rgb = cv2_cuda.GpuMat()
            gpu_rgb.upload(cv2.cvtColor(resized_cpu, cv2.COLOR_BGR2RGB))
        
        # Normalize on GPU (using arithmetic operations)
        # Divide by 255
        gpu_normalized = cv2_cuda.multiply(gpu_rgb, 1.0 / 255.0)
        
        # Download result
        normalized_cpu = gpu_normalized.download()
        
        # Convert to float32 and transpose
        blob = normalized_cpu.astype(np.float32)
        blob = np.transpose(blob, (2, 0, 1))
        return np.expand_dims(blob, axis=0).astype(np.float32)
    
    except Exception as e:
        print(f"GPU preprocessing failed: {e}, falling back to CPU")
        return preprocess_frame_cpu(frame, size)


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n GPU Preprocessing Benchmark")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--device", default="onnx-cuda", help="Device: onnx-cpu, onnx-cuda")
    parser.add_argument("--input", default="0", help="Input source: 0 (webcam) or video path")
    parser.add_argument("--duration", type=int, default=30, help="Benchmark duration (seconds)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup duration (seconds)")
    parser.add_argument("--use-gpu-preprocessing", action="store_true", help="Use GPU preprocessing")
    parser.add_argument("--out", default="results_gpu_preprocess.csv", help="Output CSV file")
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

    # Input size (handle dynamic shapes)
    input_size = (320, 320)  # YOLOv8n standard size
    if len(input_shape) == 4:
        try:
            _, _, h, w = input_shape
            if h is not None and w is not None:
                input_size = (int(w), int(h))
        except (TypeError, ValueError):
            # Dynamic shapes - ignore
            pass

    print(f"Input size: {input_size}")
    print(f"GPU Preprocessing: {'ENABLED' if (args.use_gpu_preprocessing and CUDA_AVAILABLE) else 'DISABLED'}")
    if args.use_gpu_preprocessing:
        print(f"OpenCV CUDA available: {CUDA_AVAILABLE}")

    # Select preprocessing
    preprocess_fn = preprocess_frame_gpu if args.use_gpu_preprocessing else preprocess_frame_cpu

    # Open input
    if args.input == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print("Error: Cannot open input")
        return 1

    print(f"GPU monitoring: {'Available' if PYNVML_AVAILABLE else 'Unavailable'}")

    # Warmup
    print(f"Warming up for {args.warmup}s...")
    start_warmup = time.monotonic()
    
    while time.monotonic() - start_warmup < args.warmup:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        blob = preprocess_fn(frame, input_size)
        _ = session.run(None, {input_name: blob})

    print(f"Benchmark running for {args.duration} seconds...")

    # Benchmark
    results = []
    start_time = time.monotonic()
    end_time = start_time + args.duration
    frame_idx = 0

    with tqdm.tqdm(total=args.duration, unit="s", desc="Running", smoothing=0.1) as pbar:
        while time.monotonic() < end_time:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Total time (preprocess + inference)
            t0 = time.perf_counter()
            
            # Preprocess (CPU or GPU)
            blob = preprocess_fn(frame, input_size)
            preprocess_ms = (time.perf_counter() - t0) * 1000
            
            # Inference
            t1 = time.perf_counter()
            _ = session.run(None, {input_name: blob})
            infer_ms = (time.perf_counter() - t1) * 1000
            
            total_ms = (time.perf_counter() - t0) * 1000

            # Get stats
            cpu_percent = psutil.cpu_percent(interval=0.01)
            mem_rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
            gpu_util, gpu_mem_mb = get_gpu_stats()

            results.append({
                "timestamp": time.time(),
                "frame_idx": frame_idx,
                "preprocess_ms": preprocess_ms,
                "infer_ms": infer_ms,
                "total_ms": total_ms,
                "cpu_percent": cpu_percent,
                "mem_rss_mb": mem_rss_mb,
                "gpu_util": gpu_util,
                "gpu_mem_mb": gpu_mem_mb,
            })

            frame_idx += 1
            elapsed = time.monotonic() - start_time
            pbar.update(elapsed - pbar.n)

    cap.release()

    # Save results
    print(f"Benchmark finished. Results saved to {args.out}")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "frame_idx", "preprocess_ms", "infer_ms", "total_ms",
            "cpu_percent", "mem_rss_mb", "gpu_util", "gpu_mem_mb"
        ])
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    if results:
        infer_times = [r["infer_ms"] for r in results]
        preprocess_times = [r["preprocess_ms"] for r in results]
        total_times = [r["total_ms"] for r in results]
        fps = len(results) / (results[-1]["timestamp"] - results[0]["timestamp"])
        
        mode = "GPU Preprocessing" if args.use_gpu_preprocessing else "CPU Preprocessing"
        print(f"\n{'='*50}")
        print(f"{mode.upper()}")
        print(f"{'='*50}")
        print(f"Frames: {len(results)}")
        print(f"FPS: {fps:.2f}")
        print(f"Preprocess avg: {np.mean(preprocess_times):.2f} ms")
        print(f"Inference avg: {np.mean(infer_times):.2f} ms")
        print(f"Total avg: {np.mean(total_times):.2f} ms")
        print(f"P95 total: {np.percentile(total_times, 95):.2f} ms")

    return 0


if __name__ == "__main__":
    exit(main())
