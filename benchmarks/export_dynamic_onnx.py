#!/usr/bin/env python3
"""Re-export YOLOv8n ONNX with dynamic batch dimension for batch processing."""
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
    import onnx
    
    print("🔄 Exporting YOLOv8n with dynamic batch support...")
    model = YOLO('yolov8n.pt')
    
    # Export to ONNX with simplification
    onnx_path = model.export(
        format='onnx',
        imgsz=320,
        simplify=True,
        dynamic=True,  # Enable dynamic axes for batch dimension
        half=False,    # FP32 for compatibility
    )
    
    print(f"✓ Initial export: {onnx_path}")
    
    # Load and modify ONNX model to support dynamic batch
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    
    # Get input tensor
    input_tensor = graph.input[0]
    print(f"Original input shape: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
    
    # Modify batch dimension (first dim) to be dynamic (-1)
    input_tensor.type.tensor_type.shape.dim[0].dim_value = -1
    
    print(f"Modified input shape: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
    
    # Also modify outputs to have matching batch dimension
    for output in graph.output:
        if output.type.tensor_type.shape.dim:
            output.type.tensor_type.shape.dim[0].dim_value = -1
    
    # Save modified model
    dynamic_onnx = str(onnx_path).replace('.onnx', '_dynamic.onnx')
    onnx.save(onnx_model, dynamic_onnx)
    
    print(f"✅ Dynamic batch ONNX saved: {dynamic_onnx}")
    print()
    print("Now you can use batch processing:")
    print("  batch_blob = np.concatenate([preprocess(f) for f in frames], axis=0)")
    print("  results = session.run(None, {input_name: batch_blob})")
    
except Exception as e:
    print(f"❌ Export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
