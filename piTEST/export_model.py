#!/usr/bin/env python3
"""
Run this script on your PC (not the Pi).
It exports best.pt to best.onnx and transfers it to the Pi automatically.

Requirements on PC:
    pip install ultralytics paramiko

Usage:
    python export_model.py
"""

import os
import subprocess
import sys

# ── Config ────────────────────────────────────────────────
MODEL_PATH = "T1/test_model_detection/weights/best.pt"
IMG_SIZE   = 640
PI_USER    = "ai"
PI_IP      = "192.168.0.102"
PI_DEST    = "/home/ai/ai/piTEST/best.onnx"
# ──────────────────────────────────────────────────────────

def export():
    from ultralytics import YOLO
    print(f"Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print(f"Exporting to ONNX (imgsz={IMG_SIZE})...")
    path = model.export(format="onnx", imgsz=IMG_SIZE)
    print(f"Exported: {path}")
    return path

def transfer(onnx_path):
    dest = f"{PI_USER}@{PI_IP}:{PI_DEST}"
    print(f"Transferring to Pi — {dest}")
    result = subprocess.run(["scp", onnx_path, dest])
    if result.returncode == 0:
        print("Transfer complete.")
    else:
        print("SCP failed. Copy the file manually:")
        print(f"  scp {onnx_path} {dest}")

if __name__ == "__main__":
    onnx_path = export()
    transfer(onnx_path)
    print("\nDone! On the Pi run:")
    print("  source /opt/ai_venv/bin/activate")
    print("  cd /home/ai/ai")
    print("  python3 piTEST/yolo_stream.py")
