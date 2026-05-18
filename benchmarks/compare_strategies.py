#!/usr/bin/env python3
"""
Comprehensive benchmark comparison:
1. Baseline (single-frame, CPU preprocessing)
2. Batch Processing (4-8 frames at once)
3. GPU Preprocessing (resize/normalize on GPU)
4. Combined (batch + GPU preprocessing)
"""
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def run_benchmark(script, model, device, duration, batch_size=None, gpu_preprocess=False, out_file="temp_results.csv"):
    """Run a single benchmark and return results."""
    cmd = [
        sys.executable, script,
        "--model", model,
        "--device", device,
        "--input", "0",  # Webcam
        "--duration", str(duration),
        "--warmup", "2",
        "--out", out_file
    ]
    
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    
    if gpu_preprocess:
        cmd.append("--use-gpu-preprocessing")
    
    print(f"\n🚀 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def load_and_analyze(csv_file):
    """Load CSV and compute statistics."""
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            return None
        
        # Compute metrics
        fps = len(df) / (df['timestamp'].max() - df['timestamp'].min())
        infer_ms = df['infer_ms'].mean()
        total_ms = df['total_ms'].mean()
        cpu_pct = df['cpu_percent'].mean()
        
        return {
            'frames': len(df),
            'fps': fps,
            'infer_ms': infer_ms,
            'total_ms': total_ms,
            'cpu_percent': cpu_pct,
            'gpu_util': df['gpu_util'].mean() if 'gpu_util' in df else 0,
        }
    except Exception as e:
        print(f"Error analyzing {csv_file}: {e}")
        return None


def main():
    model = Path("piTEST/yolov8n.onnx")
    if not model.exists():
        print(f"Error: Model not found at {model}")
        sys.exit(1)
    
    print("""
╔════════════════════════════════════════════════════════════╗
║      COMPREHENSIVE GPU OPTIMIZATION BENCHMARK              ║
║  Comparing: Baseline vs Batch vs GPU Preprocess vs Combined║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Benchmarks to run
    benchmarks = [
        {
            "name": "Baseline (Single-frame, CPU Preprocess)",
            "script": "benchmarks/run_bench_optimized.py",
            "batch_size": None,
            "gpu_preprocess": False,
            "duration": 15,
        },
        {
            "name": "Batch Processing (B=4)",
            "script": "benchmarks/run_bench_batch.py",
            "batch_size": 4,
            "gpu_preprocess": False,
            "duration": 15,
        },
        {
            "name": "GPU Preprocessing (CPU Batch)",
            "script": "benchmarks/run_bench_gpu_preprocess.py",
            "batch_size": None,
            "gpu_preprocess": True,
            "duration": 15,
        },
    ]
    
    results_summary = []
    
    for i, bench in enumerate(benchmarks, 1):
        print(f"\n{'='*60}")
        print(f"Benchmark {i}/{len(benchmarks)}: {bench['name']}")
        print(f"{'='*60}")
        
        out_file = f"temp_result_{i}.csv"
        
        # Run benchmark
        success = run_benchmark(
            bench['script'],
            str(model),
            "onnx-cuda",
            bench['duration'],
            batch_size=bench['batch_size'],
            gpu_preprocess=bench['gpu_preprocess'],
            out_file=out_file
        )
        
        if success:
            # Analyze results
            stats = load_and_analyze(out_file)
            if stats:
                results_summary.append({
                    "Strategy": bench['name'],
                    "FPS": stats['fps'],
                    "Avg Latency (ms)": stats['infer_ms'],
                    "Total Time (ms)": stats['total_ms'],
                    "CPU %": stats['cpu_percent'],
                    "GPU %": stats['gpu_util'],
                    "Frames": stats['frames'],
                })
                
                print(f"\n📊 Results:")
                print(f"  FPS: {stats['fps']:.2f}")
                print(f"  Inference: {stats['infer_ms']:.2f} ms")
                print(f"  Total: {stats['total_ms']:.2f} ms")
                print(f"  CPU: {stats['cpu_percent']:.1f}%")
                print(f"  GPU: {stats['gpu_util']:.1f}%")
        
        # Cleanup
        Path(out_file).unlink(missing_ok=True)
    
    # Summary comparison
    if results_summary:
        print(f"\n\n{'='*80}")
        print("SUMMARY - ALL STRATEGIES")
        print(f"{'='*80}")
        
        df_summary = pd.DataFrame(results_summary)
        print(df_summary.to_string(index=False))
        
        # Calculate speedups
        baseline_fps = df_summary.iloc[0]['FPS']
        print(f"\n{'='*80}")
        print("SPEEDUP vs BASELINE")
        print(f"{'='*80}")
        for idx, row in df_summary.iterrows():
            speedup = row['FPS'] / baseline_fps
            improvement = (speedup - 1) * 100
            emoji = "🚀" if speedup > 1 else "📊"
            print(f"{emoji} {row['Strategy']}: {speedup:.2f}x ({improvement:+.1f}%)")
        
        # Recommendations
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS FOR 60+ FPS")
        print(f"{'='*80}")
        max_fps = df_summary['FPS'].max()
        if max_fps < 40:
            print("⚠️  Current maximum: {:.1f} FPS".format(max_fps))
            print("To reach 60+ FPS:")
            print("  1. Combine best strategy with larger batch sizes (8-16)")
            print("  2. Export to TensorRT for maximum GPU utilization")
            print("  3. Enable INT8 quantization")
        elif max_fps < 60:
            print(f"✅ Current maximum: {max_fps:.1f} FPS")
            print("Close to target! Try:")
            print("  - Increase batch size to 8")
            print("  - Add INT8 quantization")
        else:
            print(f"🎉 Target achieved: {max_fps:.1f} FPS")
        
        # Save summary
        summary_file = "OPTIMIZATION_RESULTS.csv"
        df_summary.to_csv(summary_file, index=False)
        print(f"\n✅ Results saved to {summary_file}")


if __name__ == "__main__":
    main()
