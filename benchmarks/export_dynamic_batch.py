#!/usr/bin/env python3
"""
Export YOLOv8n model with dynamic batch dimension.
This enables inference with variable batch sizes (1, 2, 4, 8, etc.).
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import onnx
import numpy as np


def export_dynamic_batch(model_path="yolov8n.pt", output_dir="piTEST"):
    """Export YOLOv8n with dynamic batch."""
    print(f"Loading YOLOv8n from {model_path}...")
    model = YOLO(model_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Exporting ONNX with dynamic batch to {output_dir}...")
    
    # Export with dynamic batch
    # format: onnx
    # dynamic: True -> enables dynamic shapes (batch dimension becomes -1)
    # imgsz: 320 (standard YOLOv8n size)
    export_path = str(output_dir / "yolov8n_dynamic_batch.onnx")
    
    model.export(
        format="onnx",
        imgsz=320,
        dynamic=True,  # Enable dynamic shapes
        opset=14,       # ONNX opset version
        simplify=True,  # Simplify model
    )
    
    # Verify the exported model
    print("\n✅ Model exported. Verifying...")
    
    # Load and check the ONNX model
    onnx_model = onnx.load(str(list(output_dir.glob("*.onnx"))[0]))
    onnx.checker.check_model(onnx_model)
    
    # Print input/output shapes
    print("\nModel structure:")
    for input_tensor in onnx_model.graph.input:
        print(f"  Input: {input_tensor.name}")
        print(f"    Shape: {[d.dim_value if d.dim_value else 'dynamic' for d in input_tensor.type.tensor_type.shape.dim]}")
    
    for output_tensor in onnx_model.graph.output:
        print(f"  Output: {output_tensor.name}")
        print(f"    Shape: {[d.dim_value if d.dim_value else 'dynamic' for d in output_tensor.type.tensor_type.shape.dim]}")
    
    return export_path


def test_dynamic_batch(model_path, batch_sizes=[1, 2, 4, 8]):
    """Test inference with different batch sizes."""
    import onnxruntime as ort
    
    print(f"\n\nTesting inference with different batch sizes...")
    print("Loading session...")
    
    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    print(f"Model input shape: {input_shape}")
    
    for batch_size in batch_sizes:
        try:
            # Create dummy input: (batch, channels, height, width)
            dummy_input = np.random.randn(batch_size, 3, 320, 320).astype(np.float32)
            
            print(f"\n  Testing batch_size={batch_size}...", end=" ")
            outputs = session.run(None, {input_name: dummy_input})
            print(f"✅ Success! Output shape: {outputs[0].shape}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8n with dynamic batch")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model file")
    parser.add_argument("--output", default="piTEST", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Test with different batch sizes")
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model {args.model} not found")
        print("\nTo download YOLOv8n, run:")
        print("  from ultralytics import YOLO")
        print("  model = YOLO('yolov8n.pt')")
        return 1
    
    # Export
    export_path = export_dynamic_batch(args.model, args.output)
    print(f"\n✅ Exported to: {export_path}")
    
    # Test
    if args.test:
        test_dynamic_batch(export_path)


if __name__ == "__main__":
    exit(main())
