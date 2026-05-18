#!/usr/bin/env python3
"""Final benchmark report - CPU vs GPU CUDA 13.0"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
cpu = pd.read_csv('results_cpu.csv')
gpu = pd.read_csv('results_gpu_cuda130_v2.csv')

for df in [cpu, gpu]:
    df['infer_ms'] = pd.to_numeric(df['infer_ms'], errors='coerce')
    df['total_ms'] = pd.to_numeric(df['total_ms'], errors='coerce')

# Stats
cpu_fps = len(cpu) / (cpu['timestamp'].max() - cpu['timestamp'].min())
gpu_fps = len(gpu) / (gpu['timestamp'].max() - gpu['timestamp'].min())
cpu_lat = cpu['infer_ms'].mean()
gpu_lat = gpu['infer_ms'].mean()

print("\n" + "="*70)
print("  FINAL BENCHMARK REPORT: CPU vs GPU (CUDA 13.0)")
print("="*70)
print(f"\nCPU (Intel i7-13620H):")
print(f"  FPS:                  {cpu_fps:6.2f}")
print(f"  Inference Latency:    {cpu_lat:6.2f} ms")
print(f"  Min/Max:              {cpu['infer_ms'].min():.2f} / {cpu['infer_ms'].max():.2f} ms")
print(f"  P95:                  {cpu['infer_ms'].quantile(0.95):.2f} ms")
print(f"\nGPU (RTX 5060 - CUDA 13.0):")
print(f"  FPS:                  {gpu_fps:6.2f}")
print(f"  Inference Latency:    {gpu_lat:6.2f} ms")
print(f"  Min/Max:              {gpu['infer_ms'].min():.2f} / {gpu['infer_ms'].max():.2f} ms")
print(f"  P95:                  {gpu['infer_ms'].quantile(0.95):.2f} ms")

print(f"\n{'ACCELERATION':^70}")
print("-" * 70)
print(f"  FPS Speedup:          {gpu_fps/cpu_fps:.2f}x")
print(f"  Latency Improvement:  {cpu_lat/gpu_lat:.2f}x faster")
print(f"  Total Time Savings:   {(1 - gpu_lat/cpu_lat)*100:.1f}%")
print("="*70 + "\n")

# Generate plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Latency timeline
ax = axes[0, 0]
ax.plot(cpu['infer_ms'], label='CPU', alpha=0.7, linewidth=1.5, color='#1f77b4')
ax.plot(gpu['infer_ms'], label='GPU CUDA 13.0', alpha=0.7, linewidth=1.5, color='#ff7f0e')
ax.set_ylabel('Latency (ms)', fontsize=11)
ax.set_title('Inference Latency Over Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: FPS comparison
ax = axes[0, 1]
bars = ax.bar(['CPU', 'GPU'], [cpu_fps, gpu_fps], color=['#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('FPS', fontsize=11)
ax.set_title('FPS Comparison', fontsize=12, fontweight='bold')
for bar, val in zip(bars, [cpu_fps, gpu_fps]):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}', 
           ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Latency box plot
ax = axes[1, 0]
ax.boxplot([cpu['infer_ms'].dropna(), gpu['infer_ms'].dropna()], 
          labels=['CPU', 'GPU CUDA 13.0'], patch_artist=True)
ax.set_ylabel('Latency (ms)', fontsize=11)
ax.set_title('Latency Distribution', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Histogram
ax = axes[1, 1]
ax.hist(cpu['infer_ms'].dropna(), bins=25, alpha=0.6, label='CPU', color='#1f77b4')
ax.hist(gpu['infer_ms'].dropna(), bins=25, alpha=0.6, label='GPU CUDA 13.0', color='#ff7f0e')
ax.set_xlabel('Latency (ms)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Latency Distribution Histogram', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('benchmark_final_report.png', dpi=150, bbox_inches='tight')
print("✓ Report saved to benchmark_final_report.png\n")
