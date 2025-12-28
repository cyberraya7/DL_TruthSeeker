import torch
from model import IDSModel
import numpy as np
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

train = np.load('../data_process/train.npy')

print("Finished loading data")
model_path = 'luq_truthseeker.pth'
model = IDSModel(input_features=train.shape[1]-1, num_classes=len(config['class_names'])).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))



print("\nExporting final model to ONNX")
onnx_path = model_path.replace(".pth", ".onnx")

dummy_input = torch.randn(size=(1, train.shape[1]-1), device=device, dtype=torch.float32)

model.eval()
try:
    torch.onnx.export(
        model.to(device),
        dummy_input, onnx_path, verbose=False,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Final model exported to ONNX at: {onnx_path}")
except Exception as e:
    print(f"Error exporting model to ONNX: {e}")