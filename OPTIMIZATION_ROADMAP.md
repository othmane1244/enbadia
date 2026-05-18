# 🚀 Path to 40+ FPS - Implementation Roadmap

## Current Status
- **Achieved**: 30.04 FPS (2.5x improvement from baseline)
- **Target**: 40+ FPS 
- **Time to Target**: 2-4 hours

---

## Phase 1: Dynamic Batch ONNX Export (30 min)

### Problem
Current ONNX model has fixed batch_size=1, preventing batch processing optimization.

### Solution
```bash
cd benchmarks
python export_dynamic_onnx.py
```

This creates `yolov8n_dynamic.onnx` with dynamic batch dimension.

### Verification
```python
import onnxruntime as ort
session = ort.InferenceSession('piTEST/yolov8n_dynamic.onnx')
input_shape = session.get_inputs()[0].shape
# Should show: [None, 3, 320, 320]  (None = dynamic batch)
```

---

## Phase 2: Batch Processing Benchmark (45 min)

### Create Optimized Batch Benchmark

**File**: `benchmarks/run_bench_batch.py`

```python
#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import csv
import cv2
import numpy as np
import onnxruntime as ort
import psutil
import pynvml

pynvml.nvmlInit()

def preprocess_frame(frame, size=320):
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='piTEST/yolov8n_dynamic.onnx')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--duration', type=int, default=30)
    args = parser.parse_args()
    
    # Load model (DYNAMIC BATCH!)
    session = ort.InferenceSession(
        args.model,
        sess_options=ort.SessionOptions(graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name
    
    cap = cv2.VideoCapture(0)
    
    # Warmup
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            blob = preprocess_frame(frame)
            _ = session.run(None, {input_name: blob[None]})
    
    # Benchmark
    csv_fp = open(f'results_gpu_batch{args.batch_size}.csv', 'w')
    writer = csv.writer(csv_fp)
    writer.writerow(['timestamp', 'frame_idx', 'infer_ms', 'total_ms', 'gpu_util'])
    
    frame_idx = 0
    start = time.time()
    batch_frames = []
    
    while time.time() - start < args.duration:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        batch_frames.append(frame)
        
        if len(batch_frames) >= args.batch_size:
            # Preprocess batch
            blobs = [preprocess_frame(f) for f in batch_frames]
            batch_blob = np.stack(blobs, axis=0)  # [batch_size, 3, 320, 320]
            
            # Single GPU call for entire batch!
            t0 = time.time()
            _ = session.run(None, {input_name: batch_blob})
            t1 = time.time()
            
            infer_ms_per_frame = ((t1 - t0) * 1000.0) / len(batch_frames)
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            
            for i in range(len(batch_frames)):
                writer.writerow([time.time(), frame_idx+i, infer_ms_per_frame, infer_ms_per_frame, gpu_util])
            
            frame_idx += len(batch_frames)
            batch_frames = []
    
    cap.release()
    csv_fp.close()
    print(f"Results saved to results_gpu_batch{args.batch_size}.csv")

if __name__ == '__main__':
    main()
```

### Run Tests
```bash
# Test different batch sizes
python benchmarks/run_bench_batch.py --batch-size 4 --duration 30
python benchmarks/run_bench_batch.py --batch-size 8 --duration 30
```

### Expected Results
- **Batch 4**: 35-45 FPS (better throughput, but some frames delayed)
- **Batch 8**: 45-60 FPS (maximum throughput, 8-frame latency)
- **GPU Util**: 60-85% (much higher than 11%)

---

## Phase 3: Async Preprocessing Thread (Optional, 30 min)

### Problem
While GPU processes frame N, CPU is idle waiting for frame N+1.

### Solution: Dual-threaded Pipeline
```
Thread 1 (Capture):  webcam → queue_frames
Thread 2 (Process):  queue_frames → preprocess → queue_blobs
Main (Infer):        queue_blobs → GPU → results
```

### Implementation Sketch
```python
from queue import Queue
from threading import Thread

frames_queue = Queue(maxsize=8)
blobs_queue = Queue(maxsize=8)

def capture_thread():
    while running:
        ret, frame = cap.read()
        if ret:
            frames_queue.put(frame)

def preprocess_thread():
    while running:
        frame = frames_queue.get()
        blob = preprocess_frame(frame)
        blobs_queue.put(blob)

# Start threads
Thread(target=capture_thread, daemon=True).start()
Thread(target=preprocess_thread, daemon=True).start()

# Main inference loop (never blocked)
while running:
    batch = [blobs_queue.get() for _ in range(batch_size)]
    results = session.run(None, {input_name: np.stack(batch)})
```

### Expected Improvement
- **Before**: CPU waits for GPU, then GPU waits for CPU
- **After**: CPU always preprocessing next batch while GPU runs
- **Gain**: +15-25% throughput

---

## Phase 4: TensorRT Export (Optional Max Performance, 1-2 hours)

### Why TensorRT?
- **Fused operations**: Combines multiple ops into single kernel
- **INT8 quantization**: 3-4x speedup with minimal accuracy loss
- **Dynamic shape optimization**: Auto-optimizes for RTX 5060

### Steps

1. **Check TensorRT installation status**
   ```bash
   python -c "import tensorrt; print(tensorrt.__version__)"
   ```

2. **Export (if TensorRT ready)**
   ```bash
   python benchmarks/export_tensorrt.py
   ```

3. **Run TensorRT benchmark**
   ```python
   import onnxruntime as ort
   session = ort.InferenceSession(
       'yolov8n.engine',  # TensorRT format
       providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider']
   )
   ```

### Expected Performance
| Format | FPS | GPU Util | Latency |
|--------|-----|----------|---------|
| ONNX (current) | 30 | 45% | 6.75ms |
| ONNX Batch4 | 40-45 | 70% | 8-10ms |
| TensorRT | 60-80 | 85% | 4-5ms |
| TensorRT INT8 | 80-100 | 90%+ | 3-4ms |

---

## Quick Decision Tree

```
START: Do you want 40+ FPS?
│
├─ YES, ASAP (need ~40 FPS)
│  └─ Use Phase 1+2 (Dynamic ONNX + Batch4)
│     └─ Expected: 40-50 FPS in 2 hours
│
├─ YES, MAX performance (need 60+ FPS)
│  └─ Use Phase 1+2+4 (Batch + TensorRT)
│     └─ Expected: 60-100 FPS in 4-6 hours
│
└─ MAYBE, want to experiment
   └─ Use Phase 1+2+3 (Batch + Async threads)
      └─ Expected: 45-60 FPS, understand bottlenecks
```

---

## Performance Validation

### Metrics to Track
1. **FPS**: Frames per second (main metric)
2. **GPU Util**: Should be 60-90% (not 11%)
3. **Latency**: P95 should be under 20ms for real-time
4. **Accuracy**: COCO mAP should not degrade

### Analysis Script
```python
import pandas as pd

df = pd.read_csv('results_gpu_batch4.csv')
print(f"FPS: {len(df) / (df.timestamp.max() - df.timestamp.min()):.2f}")
print(f"GPU Util: {df.gpu_util.mean():.1f}%")
print(f"Latency P95: {df.infer_ms.quantile(0.95):.2f}ms")
```

---

## Troubleshooting

### Issue: "Model expects batch=1"
**Solution**: Use `yolov8n_dynamic.onnx`, not `yolov8n.onnx`

### Issue: "Lower FPS with batch processing"
**Cause**: Model hasn't been re-exported, falls back to serial execution
**Solution**: Verify `session.get_inputs()[0].shape` shows `[None, 3, 320, 320]`

### Issue: "TensorRT not available"
**Cause**: Large installation (~2GB), may still be downloading
**Solution**: 
```bash
pip install tensorrt -q  # Quiet mode, will complete
# Wait 10-15 minutes for installation
```

### Issue: "GPU utilization still low (30-40%)"
**Cause**: Batch size too small, or preprocessing bottleneck
**Solution**: 
1. Increase batch size (8 or 16)
2. Use async preprocessing thread (Phase 3)

---

## Summary

| Phase | Effort | Expected FPS | GPU Util |
|-------|--------|--------------|----------|
| Current (30 FPS) | ✓ Done | 30.04 | 45% |
| Phase 1+2 | 1h | 40-50 | 70% |
| Phase 1+2+3 | 2h | 45-60 | 75% |
| Phase 1+2+4 | 4-6h | 60-100 | 85%+ |

**Recommended**: Phase 1+2 for quick 40+ FPS gain (1-2 hours, minimal risk)

---

**Next Command**: 
```bash
cd benchmarks && python export_dynamic_onnx.py
```

Good luck! 🚀
