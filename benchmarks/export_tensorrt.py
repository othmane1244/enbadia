#!/usr/bin/env python3
"""Export YOLOv8n to TensorRT engine for maximum GPU performance."""
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
    print("Exporting YOLOv8n to TensorRT...")
    model = YOLO('yolov8n.pt')
    
    # Export to TensorRT (device=0 = GPU 0)
    engine_path = model.export(format='engine', device=0, half=True, imgsz=320)
    print(f"✅ TensorRT engine exported to: {engine_path}")
    
except Exception as e:
    print(f"❌ Export failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Ensure CUDA Toolkit is installed")
    print("  2. Ensure TensorRT is installed: pip install tensorrt")
    print("  3. Check GPU is available: nvidia-smi")
    sys.exit(1)
