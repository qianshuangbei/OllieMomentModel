import os
import torch
import shutil
import coremltools as ct
from ultralytics import YOLO
from model.classifier import PoseClassifierV1

def export_pose_classifier(model_path, output_path, label_size=5, quantize=True):
    """
    Export PoseClassifierV1 to CoreML format with optional quantization
    """
    # Load model
    model = PoseClassifierV1(label_size)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 34, 4)
    
    # Export to TorchScript
    traced_model = torch.jit.trace(model, example_input)
    
    out = traced_model(example_input)
    print(f"Generated TorchScript model output shape: {out.shape}")
    
    # Convert to CoreML - pass configuration directly to convert function
    model_coreml = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape)],
        minimum_deployment_target=ct.target.iOS17,  # Adjust based on your needs
        compute_units=ct.ComputeUnit.ALL,  # Uses Neural Engine if available
        convert_to="mlprogram"  # Better performance than neuralnetwork
    )
    model_coreml.save(output_path)
    print(f"Exported pose classifier model Package to {output_path}")
    
    
    from coremltools.optimize.torch.quantization import PostTrainingQuantizer, \
    PostTrainingQuantizerConfig

    config = PostTrainingQuantizerConfig.from_dict(
        {
            "global_config": {
                "weight_dtype": "int8",
                "granularity": "per_block",
                "block_size": 128,
            },
            "module_type_configs": {
                torch.nn.Linear: None
            }
        }
    )
    quantizer = PostTrainingQuantizer(model, config)
    quantized_model = quantizer.compress()
    output_path = output_path.replace(".mlpackage", "_quantized.mlpackage")
    # Create example input
    example_input = torch.randn(1, 34, 4)
    
    # Export to TorchScript
    traced_model = torch.jit.trace(quantized_model, example_input)
    
    out = traced_model(example_input)
    print(f"Generated TorchScript model output shape: {out.shape}")
    
    # Convert to CoreML - pass configuration directly to convert function
    model_coreml = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape)],
        minimum_deployment_target=ct.target.iOS17,  # Adjust based on your needs
        compute_units=ct.ComputeUnit.ALL,  # Uses Neural Engine if available
        convert_to="mlprogram"  # Better performance than neuralnetwork
    )
    model_coreml.save(output_path)
    print(f"Exported pose classifier model Package to {output_path}")

def export_yolo(model_path, output_path):
    """
    Export YOLO model to CoreML format
    """
    # Load YOLO model
    model = YOLO(model_path)
    
    # Export to CoreML with INT8 quantization
    export_path = model.export(format="coreml", int8=True)
    print(f"Exported {model_path} to {export_path}")
    shutil.move(export_path, output_path)
    print(f"Exported YOLO model to {output_path}")

if __name__ == "__main__":
    # Example usage for pose classifier
    pose_model_path = "outputs/v3/checkpoints/model_best.pth"  # Adjust path as needed
    pose_output_path = "outputs/iosModel/cls.mlpackage"
    export_pose_classifier(pose_model_path, pose_output_path, 5)
    
    # Example usage for YOLO
    yolo_model_path = "yolo11n-pose.pt"  # Adjust path as needed
    yolo_output_path = "outputs/iosModel/yolo11npose.mlpackage"
    os.makedirs(os.path.dirname(yolo_output_path), exist_ok=True)
    export_yolo(yolo_model_path, yolo_output_path)

    yolo_model_path = "yolo11s-seg.pt"  # Adjust path as needed
    yolo_output_path = "outputs/iosModel/yolo11sseg.mlpackage"
    os.makedirs(os.path.dirname(yolo_output_path), exist_ok=True)
    export_yolo(yolo_model_path, yolo_output_path)
