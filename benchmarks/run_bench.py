#!/usr/bin/env python3
"""Benchmark inference (ONNX) on CPU and GPU and collect metrics.

Usage examples:
  python run_bench.py --model path/to/model.onnx --device onnx-cpu --input 0 --duration 20 --out results_cpu.csv
  python run_bench.py --model path/to/model.onnx --device onnx-cuda --input 0 --duration 20 --out results_gpu.csv

The script captures per-frame inference time, CPU and memory usage, and (if available) GPU util/memory via nvidia-smi.
"""
import argparse
import time
import sys
import subprocess
from pathlib import Path
import csv

import cv2
import numpy as np
import onnxruntime as ort
import psutil
from tqdm import tqdm


def detect_nvidia_smi():
    try:
        subprocess.check_output(["nvidia-smi", "-h"], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False


def query_nvidia():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits"
        ])
        gpu_util, gpu_mem = out.decode().strip().split(',')
        return int(gpu_util), int(gpu_mem)
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
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.ascontiguousarray(img[None])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--device', choices=['onnx-cpu', 'onnx-cuda'], default='onnx-cpu')
    p.add_argument('--input', default='0', help='video file or webcam index (0)')
    p.add_argument('--duration', type=int, default=20, help='seconds to benchmark')
    p.add_argument('--warmup', type=int, default=3, help='seconds of warmup')
    p.add_argument('--out', default='results.csv')
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

    nvidia = detect_nvidia_smi()
    print('nvidia-smi available:', nvidia)

    csv_fp = open(args.out, 'w', newline='')
    writer = csv.writer(csv_fp)
    writer.writerow(['timestamp','frame_idx','infer_ms','total_ms','cpu_percent','mem_rss_mb','gpu_util','gpu_mem_mb'])

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
        _ = session.run(None, {input_name: blob})

    print('Running benchmark for', args.duration, 'seconds...')
    pbar = tqdm(total=args.duration)
    last_report = time.monotonic()
    while time.monotonic() < end_time:
        t0 = time.monotonic()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        blob = preprocess_frame(frame, input_size)
        t_infer0 = time.monotonic()
        _ = session.run(None, {input_name: blob})
        t_infer1 = time.monotonic()

        infer_ms = (t_infer1 - t_infer0) * 1000.0
        total_ms = (time.monotonic() - t0) * 1000.0

        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.Process().memory_info().rss / (1024*1024)
        gpu_util = None
        gpu_mem = None
        if nvidia:
            gu, gm = query_nvidia()
            gpu_util, gpu_mem = gu, gm

        writer.writerow([time.time(), frame_idx, f'{infer_ms:.3f}', f'{total_ms:.3f}', cpu, f'{mem:.1f}', gpu_util, gpu_mem])
        frame_idx += 1

        # update progress bar per second
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
