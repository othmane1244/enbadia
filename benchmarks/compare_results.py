#!/usr/bin/env python3
"""Compare two benchmark CSVs and plot FPS/latency comparison."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def summarize(csv_path: Path):
    df = pd.read_csv(csv_path)
    df['infer_ms'] = pd.to_numeric(df['infer_ms'], errors='coerce')
    fps = len(df) / (df['timestamp'].max() - df['timestamp'].min())
    return {
        'frames': len(df),
        'duration_s': df['timestamp'].max() - df['timestamp'].min(),
        'fps': fps,
        'mean_ms': df['infer_ms'].mean(),
        'p95_ms': df['infer_ms'].quantile(0.95),
    }


def plot(csv1: Path, label1: str, csv2: Path = None, label2: str = None, out: Path = Path('compare.png')):
    df1 = pd.read_csv(csv1)
    s1 = summarize(csv1)

    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.title('Infer latency (ms)')
    plt.plot(df1['infer_ms'], label=label1)
    if csv2:
        df2 = pd.read_csv(csv2)
        plt.plot(df2['infer_ms'], label=label2)
    plt.ylabel('ms')
    plt.legend()

    plt.subplot(2,1,2)
    labels = [label1]
    fps_vals = [s1['fps']]
    if csv2:
        s2 = summarize(csv2)
        labels.append(label2)
        fps_vals.append(s2['fps'])
    plt.bar(labels, fps_vals)
    plt.ylabel('FPS')

    plt.tight_layout()
    plt.savefig(out)
    print('Saved comparison plot to', out)
    print('Summary:', label1, s1)
    if csv2:
        print('Summary:', label2, s2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv1')
    p.add_argument('label1')
    p.add_argument('--csv2')
    p.add_argument('--label2')
    p.add_argument('--out', default='compare.png')
    args = p.parse_args()
    plot(Path(args.csv1), args.label1, Path(args.csv2) if args.csv2 else None, args.label2, Path(args.out))


if __name__ == '__main__':
    main()
