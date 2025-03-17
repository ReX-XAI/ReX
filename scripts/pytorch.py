#!/usr/bin/env python3

import platform
from torchvision.models import resnet50
from torchvision import transforms as T
import torch as tt
import torch.nn.functional as F
from PIL import Image  # type: ignore
from rex_xai.input_data import Data
from rex_xai.prediction import from_pytorch_tensor


model = resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()

if platform.uname().system == "Darwin":
    model.to("mps")
else:
    model.to("cuda")


def preprocess(path, shape, device, mode) -> Data:
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # open the image with mode "RGB"
    img = Image.open(path).convert("RGB")
    # create a Data object
    data = Data(img, shape, device, mode='RGB')
    # manually set the data to the transformed image for model consumption
    data.data = transform(img).unsqueeze(0).to(device)  # type: ignore

    return data


def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        tensor = model(mutants)
        if raw:
            return F.softmax(tensor, dim=1)
        # from_pytorch_tensor consumes a tensor and converts it to a Prediction object
        # you can  alternatively use your own function here
        return from_pytorch_tensor(tensor, target=target)


def model_shape():
    return ["N", 3, 224, 224]
