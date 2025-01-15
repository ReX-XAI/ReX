#!/usr/bin/env python3
from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.Tensor import Tensor
import torch as tt
import torch.nn.functional as F
from rex_xai.input_data import Data
from rex_xai.prediction import from_pytorch_tensor
from monai.networks.nets import DenseNet121
from monai.transforms import Resize, LoadImage

# Load the sample 3D model #TODO: Add monai dependency for 3d extension
device = tt.device("cuda" if tt.cuda.is_available() else "cpu")
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
model.to(device)
model.eval()


def preprocess(path, shape, device, mode) -> Data:
    transform = Resize(spatial_size=(64, 64, 64))
    if type(path) == str:
        volume = LoadImage()(path)
        transformed_volume = transform(volume)
        data = Data(transformed_volume, shape, device, mode=mode, process=False)
    elif type(path) == Tensor:
        transformed_volume = transform(path)
        data = Data(transformed_volume, shape, device, mode=mode, process=False)
    else:
        raise ValueError("Invalid input type")
    data.mode = "voxel"
    data.model_shape = shape
    data.model_height = 64
    data.model_width = 64
    data.model_depth = 64
    data.transposed = True

    return data


def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        tensor = model(mutants)
        if raw:
            return F.softmax(tensor, dim=1)
        return from_pytorch_tensor(tensor)


def model_shape():
    return ["N", 64, 64, 64]

if __name__ == "__main__":
    print("Model loaded successfully!")