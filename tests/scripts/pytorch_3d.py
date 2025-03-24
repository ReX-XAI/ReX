#!/usr/bin/env python3
from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.Tensor import Tensor
import torch as tt
import torch.nn.functional as F
from rex_xai.input_data import Data
from rex_xai.prediction import from_pytorch_tensor
from monai.networks.nets import DenseNet121
from monai.transforms import Resize, LoadImage

# Load the sample 3D model
device = tt.device("cuda" if tt.cuda.is_available() else "cpu")
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)
model.to(device)
model.eval()


def preprocess(path, shape, device, mode) -> Data:
    """
    The preprocessing function is executed before the
    model is called. It is used to prepare the input.

    Args:
        path: str
            The path to the input data
        shape: tuple
            The shape of the input data
        device: str
            The device to use indicate where the data should be loaded: "cpu" or "cuda"
        mode: str
            The mode of the input data: "voxel" for 3D data

    Returns:
        Data: The input data object
            The input data object that contains the processed data
            and the metadata of the input data including the mode, model_height,
            model_width, model_depth, model_channels, model_order, and transposed.
    """
    transform = Resize(spatial_size=(64, 64, 64))
    if path is str:
        volume = LoadImage()(path)
        transformed_volume = transform(volume)
        data = Data(transformed_volume, shape, device, mode=mode, process=False)
    elif path is Tensor:
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
    """
    The prediction function calls the model itself and returns the output.

    Args:
        mutants: Data
            The input data object
        target: None
            The target label
        raw: bool
            A flag to indicate if the output should be raw or not
        binary_threshold: None
            The binary threshold value

    Returns:
        list[Prediction]
            A list of prediction objects which each contain the output tensor,
            the target label, the confidence of the label, the classification confidence,
            and the classification label.

    """
    with tt.no_grad():
        tensor = model(mutants)
        if raw:
            return F.softmax(tensor, dim=1)
        return from_pytorch_tensor(tensor)


def model_shape():
    return ["N", 64, 64, 64]
