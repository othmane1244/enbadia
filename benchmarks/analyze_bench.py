#!/usr/bin/env python3
"""Comprehensive benchmark analysis and report generation."""
import sys
from pathlib import Path
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_and_analyze(csv_path: Path):
    """Load CSV and compute detailed statistics."""
    df = pd.read_csv(csv_path)
    df['infer_ms'] = pd.to_numeric(df['infer_ms'], errors='coerce')
    df['cpu_percent'] = pd.to_numeric(df['cpu_percent'], errors='coerce')
    df['mem_rss_mb'] = pd.to_numeric(df['mem_rss_mb'], errors='coerce')
    df['gpu_util'] = pd.to_numeric(df['gpu_util'], errors='coerce')
    df['gpu_mem_mb'] = pd.to_numeric(df['gpu_mem_mb'], errors='coerce')
    
    # Remove warmup frames
    duration = df['timestamp'].max() - df['timestamp'].min()
    fps = len(df) / duration
    
    infer = df['infer_ms'].dropna()
    
    return {
        'frames': len(df),
        'duration_s': duration,
        'fps': fps,
        'infer_ms': {
            'mean': infer.mean(),
            'median': infer.median(),
            'std': infer.std(),
            'min': infer.min(),
            'max': infer.max(),
            'p95': infer.quantile(0.95),
            'p99': infer.quantile(0.99),
        },
        'cpu_percent': df['cpu_percent'].mean(),
        'mem_rss_mb': df['mem_rss_mb'].mean(),
        'gpu_util': df['gpu_util'].mean() if df['gpu_util'].notna().any() else 0,
        'gpu_mem_mb': df['gpu_mem_mb'].mean() if df['gpu_mem_mb'].notna().any() else 0,
    }


def print_summary(label: str, stats: dict):
    """Print formatted summary."""
    print(f"\n{'='*60}")
    print(f"  {label.upper()}")
    print(f"{'='*60}")
    print(f"  Frames:       {stats['frames']:>6}")
    print(f"  Duration:     {stats['duration_s']:>6.1f} s")
    print(f"  FPS:          {stats['fps']:>6.2f}")
    print(f"\n  Inference Latency (ms):")
    print(f"    Mean:       {stats['infer_ms']['mean']:>6.2f}")
    print(f"    Median:     {stats['infer_ms']['median']:>6.2f}")
    print(f"    Std Dev:    {stats['infer_ms']['std']:>6.2f}")
    print(f"    Min:        {stats['infer_ms']['min']:>6.2f}")
    print(f"    Max:        {stats['infer_ms']['max']:>6.2f}")
    print(f"    P95:        {stats['infer_ms']['p95']:>6.2f}")
    print(f"    P99:        {stats['infer_ms']['p99']:>6.2f}")
    print(f"\n  System Resources:")
    print(f"    CPU (avg):  {stats['cpu_percent']:>6.1f}%")
    print(f"    RAM (avg):  {stats['mem_rss_mb']:>6.1f} MB")
    print(f"    GPU Util:   {stats['gpu_util']:>6.1f}%")
    print(f"    GPU Mem:    {stats['gpu_mem_mb']:>6.0f} MB")


def plot_comparison(csv_cpu: Path, stats_cpu: dict, csv_gpu: Path, stats_gpu: dict, out: Path = Path('benchmark_report.png')):
    """Create comprehensive comparison plots."""
    df_cpu = pd.read_csv(csv_cpu)
    df_gpu = pd.read_csv(csv_gpu)
    df_cpu['infer_ms'] = pd.to_numeric(df_cpu['infer_ms'], errors='coerce')
    df_gpu['infer_ms'] = pd.to_numeric(df_gpu['infer_ms'], errors='coerce')
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # 1. Latency timeline
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df_cpu['infer_ms'], label='CPU', alpha=0.7, linewidth=1.5)
    ax1.plot(df_gpu['infer_ms'], label='GPU', alpha=0.7, linewidth=1.5)
    ax1.set_ylabel('Latency (ms)', fontsize=11)
    ax1.set_title('Inference Latency Over Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. FPS comparison
    ax2 = fig.add_subplot(gs[0, 2])
    fps_vals = [stats_cpu['fps'], stats_gpu['fps']]
    colors = ['#1f77b4', '#ff7f0e']
    bars = ax2.bar(['CPU', 'GPU'], fps_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('FPS', fontsize=11)
    ax2.set_title('FPS Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(fps_vals) * 1.2)
    for bar, val in zip(bars, fps_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Latency distribution (box plot)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.boxplot([df_cpu['infer_ms'].dropna(), df_gpu['infer_ms'].dropna()], 
                labels=['CPU', 'GPU'], patch_artist=True)
    ax3.set_ylabel('Latency (ms)', fontsize=11)
    ax3.set_title('Latency Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Latency histogram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(df_cpu['infer_ms'].dropna(), bins=30, alpha=0.6, label='CPU', color='#1f77b4')
    ax4.hist(df_gpu['infer_ms'].dropna(), bins=30, alpha=0.6, label='GPU', color='#ff7f0e')
    ax4.set_xlabel('Latency (ms)', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Latency Histogram', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Percentile comparison
    ax5 = fig.add_subplot(gs[1, 2])
    percentiles = [50, 95, 99]
    cpu_perc = [stats_cpu['infer_ms']['median'], stats_cpu['infer_ms']['p95'], stats_cpu['infer_ms']['p99']]
    gpu_perc = [stats_gpu['infer_ms']['median'], stats_gpu['infer_ms']['p95'], stats_gpu['infer_ms']['p99']]
    x = np.arange(len(percentiles))
    width = 0.35
    ax5.bar(x - width/2, cpu_perc, width, label='CPU', color='#1f77b4', alpha=0.7, edgecolor='black')
    ax5.bar(x + width/2, gpu_perc, width, label='GPU', color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Latency (ms)', fontsize=11)
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'P{p}' for p in percentiles])
    ax5.set_title('Percentile Comparison', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. CPU utilization
    ax6 = fig.add_subplot(gs[2, 0])
    df_cpu['cpu_percent'] = pd.to_numeric(df_cpu['cpu_percent'], errors='coerce')
    df_gpu['cpu_percent'] = pd.to_numeric(df_gpu['cpu_percent'], errors='coerce')
    ax6.plot(df_cpu['cpu_percent'], label='CPU (bench)', alpha=0.6, linewidth=1)
    ax6.plot(df_gpu['cpu_percent'], label='GPU (bench)', alpha=0.6, linewidth=1)
    ax6.set_ylabel('CPU %', fontsize=11)
    ax6.set_xlabel('Frame Index', fontsize=11)
    ax6.set_title('CPU Utilization', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 100)
    
    # 7. Memory usage
    ax7 = fig.add_subplot(gs[2, 1])
    df_cpu['mem_rss_mb'] = pd.to_numeric(df_cpu['mem_rss_mb'], errors='coerce')
    df_gpu['mem_rss_mb'] = pd.to_numeric(df_gpu['mem_rss_mb'], errors='coerce')
    ax7.plot(df_cpu['mem_rss_mb'], label='CPU (RAM)', alpha=0.6, linewidth=1)
    ax7.plot(df_gpu['mem_rss_mb'], label='GPU (RAM)', alpha=0.6, linewidth=1)
    ax7.set_ylabel('Memory (MB)', fontsize=11)
    ax7.set_xlabel('Frame Index', fontsize=11)
    ax7.set_title('RAM Usage', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Summary table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    summary_text = f"""
    SUMMARY
    ━━━━━━━━━━━━━━━━━━
    CPU FPS:     {stats_cpu['fps']:.2f}
    GPU FPS:     {stats_gpu['fps']:.2f}
    
    CPU Lat:     {stats_cpu['infer_ms']['mean']:.2f} ms
    GPU Lat:     {stats_gpu['infer_ms']['mean']:.2f} ms
    
    Speedup:     {stats_cpu['infer_ms']['mean']/stats_gpu['infer_ms']['mean']:.2f}x
    
    CPU RAM:     {stats_cpu['mem_rss_mb']:.0f} MB
    GPU RAM:     {stats_gpu['mem_rss_mb']:.0f} MB
    """
    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n✓ Report saved to {out}")


def main():
    csv_cpu = Path('results_cpu.csv')
    csv_gpu = Path('results_gpu.csv')
    
    if not csv_cpu.exists():
        print("Error: results_cpu.csv not found")
        sys.exit(1)
    if not csv_gpu.exists():
        print("Error: results_gpu.csv not found")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("  BENCHMARK ANALYSIS - YOLOV8N INFERENCE")
    print("="*60)
    
    stats_cpu = load_and_analyze(csv_cpu)
    stats_gpu = load_and_analyze(csv_gpu)
    
    print_summary("CPU Results", stats_cpu)
    print_summary("GPU Results", stats_gpu)
    
    # Comparison
    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    speedup_fps = stats_gpu['fps'] / stats_cpu['fps']
    speedup_lat = stats_cpu['infer_ms']['mean'] / stats_gpu['infer_ms']['mean']
    print(f"  FPS Ratio:        {speedup_fps:>6.2f}x (GPU/CPU)")
    print(f"  Latency Ratio:    {speedup_lat:>6.2f}x (CPU/GPU)")
    print(f"  Memory Overhead:  {stats_gpu['mem_rss_mb'] - stats_cpu['mem_rss_mb']:>6.0f} MB (GPU)")
    
    # Generate plots
    plot_comparison(csv_cpu, stats_cpu, csv_gpu, stats_gpu, Path('benchmark_report.png'))
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
