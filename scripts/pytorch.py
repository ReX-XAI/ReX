#!/usr/bin/env python3

import platform
from torchvision.models import get_model, get_weight
from torchvision import transforms as T
import torch as tt
import torch.nn.functional as F
from PIL import Image  # type: ignore
from rex_xai.input.input_data import Data
from rex_xai.responsibility.prediction import from_pytorch_tensor
# from torchvision.models.weights import ResNet50_Weights


model = get_model('resnet50', weights="DEFAULT")
weights = get_weight("ResNet50_Weights.IMAGENET1K_V1")
model.eval()

if platform.uname().system == "Darwin":
    model.to("mps")
else:
    model.to("cuda")

def preprocess(path, shape, device, mode) -> Data:
    # open the image with mode "RGB"
    img = Image.open(path).convert("RGB")
    data = Data(img, shape, device, mode='RGB')
    # manually set the data to the transformed image for model consumption
    data.data = weights.transforms()(img).unsqueeze(0).to(device)  # type: ignore
    # make a copy
    original = Image.open(path).convert("RGB")
    original = T.functional.resize(original, (256, 256))
    original = T.functional.center_crop(original, 224)
    data.input = original

    return data


def prediction_function(mutants, target=None, raw=False, binary_threshold=False):
    with tt.no_grad(): # we don't use the grad and inference is faster without it
        tensor = model(mutants)
        if raw: # used when computing insertion/deletion curves
            return F.softmax(tensor, dim=1)
        # from_pytorch_tensor consumes a tensor and converts it to a Prediction object
        # you can  alternatively use your own function here
        return from_pytorch_tensor(tensor, target=target)


def model_shape():
    return ["N", 3, 224, 224]
