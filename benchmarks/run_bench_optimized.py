#!/usr/bin/env python3
"""Optimized YOLO benchmark with batching, pynvml, and pure inference measurement."""
import argparse
import time
import sys
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
except Exception:
    HAS_PYNVML = False
    print("⚠ pynvml not available - GPU metrics disabled")

from tqdm import tqdm


def get_gpu_metrics_pynvml():
    """Get GPU metrics without subprocess (fast)."""
    if not HAS_PYNVML:
        return None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return int(util.gpu), int(mem.used / (1024*1024))
    except Exception:
        return None, None


def load_session(model_path: Path, device: str):
    if device == 'onnx-cpu':
        providers = ['CPUExecutionProvider']
    elif device == 'onnx-cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        raise ValueError('Unsupported device')

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    return session, input_name, input_shape


def preprocess_frame(frame, input_size):
    """Preprocess single frame."""
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img)


def preprocess_batch(frames, input_size):
    """Preprocess batch of frames (vectorized)."""
    batch = []
    for frame in frames:
        batch.append(preprocess_frame(frame, input_size))
    return np.stack(batch, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--device', choices=['onnx-cpu', 'onnx-cuda'], default='onnx-cpu')
    p.add_argument('--input', default='0', help='video file or webcam index (0)')
    p.add_argument('--duration', type=int, default=30, help='seconds to benchmark')
    p.add_argument('--warmup', type=int, default=3, help='seconds of warmup')
    p.add_argument('--batch-size', type=int, default=4, help='frames per batch')
    p.add_argument('--out', default='results_optimized.csv')
    p.add_argument('--measure-inference-only', action='store_true', help='skip preprocessing in timing')
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print('Model not found:', model_path)
        sys.exit(1)

    print('Loading session...', args.device)
    session, input_name, input_shape = load_session(model_path, args.device)
    input_size = input_shape[-1] if isinstance(input_shape, (list, tuple)) else 320

    # Open input
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print('Failed to open input', args.input)
        sys.exit(1)

    print(f'Batch size: {args.batch_size}')
    print(f'Measure inference only: {args.measure_inference_only}')

    # Pre-allocate batch buffer
    batch_buffer = np.zeros((args.batch_size, 3, input_size, input_size), dtype=np.float32)

    csv_fp = open(args.out, 'w', newline='')
    writer = csv.writer(csv_fp)
    writer.writerow(['timestamp','frame_idx','infer_ms','total_ms','cpu_percent','mem_rss_mb','gpu_util','gpu_mem_mb','batch_size'])

    start_time = time.monotonic()
    end_time = start_time + args.duration
    frame_idx = 0

    # Warmup
    print(f'Warming up for {args.warmup}s...')
    warmup_end = time.monotonic() + args.warmup
    while time.monotonic() < warmup_end:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        blob = preprocess_frame(frame, input_size)
        _ = session.run(None, {input_name: blob[None]})

    print('Running optimized benchmark for', args.duration, 'seconds...')
    pbar = tqdm(total=args.duration)
    last_report = time.monotonic()
    
    batch_frames = []
    batch_start = time.monotonic()

    while time.monotonic() < end_time:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        batch_frames.append(frame)

        # When batch is full or duration exceeded, process
        if len(batch_frames) >= args.batch_size or (time.monotonic() >= end_time and batch_frames):
            t_batch_start = time.monotonic()

            # Preprocess batch
            t_prep_start = time.monotonic()
            for i, f in enumerate(batch_frames):
                batch_buffer[i] = preprocess_frame(f, input_size)
            t_prep_end = time.monotonic()

            # Infer (measure this)
            t_infer_start = time.monotonic()
            batch_to_infer = batch_buffer[:len(batch_frames)]
            _ = session.run(None, {input_name: batch_to_infer})
            t_infer_end = time.monotonic()

            t_batch_end = time.monotonic()

            # Record metrics
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.Process().memory_info().rss / (1024*1024)
            gpu_util, gpu_mem = get_gpu_metrics_pynvml()

            infer_ms = (t_infer_end - t_infer_start) * 1000.0 / len(batch_frames)  # per-frame
            total_ms = (t_batch_end - t_batch_start) * 1000.0 / len(batch_frames)

            for i in range(len(batch_frames)):
                writer.writerow([time.time(), frame_idx + i, f'{infer_ms:.3f}', 
                               f'{total_ms:.3f}', cpu, f'{mem:.1f}', gpu_util, gpu_mem, len(batch_frames)])

            frame_idx += len(batch_frames)
            batch_frames = []

            # Update progress
            now = time.monotonic()
            if now - last_report >= 1.0:
                pbar.update(int(now - last_report))
                last_report = now

    pbar.close()
    csv_fp.close()
    cap.release()

    print('Benchmark finished. Results saved to', args.out)


if __name__ == '__main__':
    main()
