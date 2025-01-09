#!/usr/bin/env python3

from torchvision.models import resnet50
from torchvision import transforms as T
import torch as tt
import torch.nn.functional as F
from PIL import Image  # type: ignore
from rex_xai.input_data import Data
from rex_xai.prediction import from_pytorch_tensor


model = resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()
model.to("cpu")


def preprocess(path, shape, device, mode) -> Data:
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(path).convert("RGB")
    data = Data(img, shape, device, mode=mode, process=False)
    data.data = transform(img).unsqueeze(0).to(device)  # type: ignore
    data.mode = "RGB"
    data.model_shape = shape
    data.model_height = 224
    data.model_width = 224
    data.model_channels = 3
    data.transposed = True
    data.model_order = "first"

    return data


def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        tensor = model(mutants)
        if raw:
            return F.softmax(tensor, dim=1)
        return from_pytorch_tensor(tensor)


def model_shape():
    return ["N", 3, 224, 224]
