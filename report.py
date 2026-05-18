import pandas as pd
import numpy as np

try:
    cpu = pd.read_csv("results_cpu.csv")
    gpu_v1 = pd.read_csv("results_gpu_cuda130_v2.csv")
    gpu_opt = pd.read_csv("results_gpu_optimized_b1.csv")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

def calc_stats(df):
    df["infer_ms"] = pd.to_numeric(df["infer_ms"], errors="coerce")
    df["gpu_util"] = pd.to_numeric(df["gpu_util"], errors="coerce")
    dur = df["timestamp"].max() - df["timestamp"].min()
    return {
        "frames": len(df),
        "fps": len(df) / dur if dur > 0 else 0,
        "latency": df["infer_ms"].mean(),
        "p95": df["infer_ms"].quantile(0.95),
        "gpu": df["gpu_util"].mean()
    }

c = calc_stats(cpu)
g1 = calc_stats(gpu_v1)
go = calc_stats(gpu_opt)

print("╔════════════════════════════════════════════════════════════╗")
print("║         YOLO BENCHMARK COMPARISON REPORT                   ║")
print("║         YOLOv8n ONNX (320x320) - Webcam Input              ║")
print("╚════════════════════════════════════════════════════════════╝")
print(f"\nCPU (Intel i7-13620H):")
print(f"  FPS: {c['fps']:6.2f} | Latency: {c['latency']:6.2f}ms | P95: {c['p95']:6.2f}ms")
print(f"\nGPU v1 (RTX 5060 + CUDA 13.0):")
print(f"  FPS: {g1['fps']:6.2f} | Latency: {g1['latency']:6.2f}ms | P95: {g1['p95']:6.2f}ms | Util: {g1['gpu']:5.1f}%")
print(f"\nGPU OPTIMIZED:")
print(f"  FPS: {go['fps']:6.2f} | Latency: {go['latency']:6.2f}ms | P95: {go['p95']:6.2f}ms | Util: {go['gpu']:5.1f}%")
print(f"\nSUMMARY:")
print(f"  CPU -> GPU v1: {g1['fps']/max(0.1, c['fps']):.2f}x speedup")
print(f"  GPU v1 -> OPT: {go['fps']/max(0.1, g1['fps']):.2f}x speedup")
print(f"  TOTAL SPEEDUP: {go['fps']/max(0.1, c['fps']):.2f}x")
