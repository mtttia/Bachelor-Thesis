import torch
from l2cs import Pipeline
from pathlib import Path

def export_l2cs_to_onnx():
    
    gaze_pipeline = Pipeline(
        weights=Path("./L2CSNet_gaze360.pkl"), # pretrained model weight
        arch='ResNet50',
        device=torch.device('cpu')
    )
    
    model = gaze_pipeline.model
    
    dummy_input = torch.randn(1, 3, 448, 448)

    torch.onnx.export(
        model,
        dummy_input,
        "l2cs_gaze.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['pitch', 'yaw']
    )
    print("Model exported to l2cs_gaze.onnx")

if __name__ == "__main__":
    export_l2cs_to_onnx()

    