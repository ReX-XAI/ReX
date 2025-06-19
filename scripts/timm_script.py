#!/usr/bin/env python3

import platform

import timm
import torch as tt
from PIL import Image
from torchvision import transforms as T

# import torch.nn.functional as F
from rex_xai.input.input_data import Data

# from rex_xai.responsibility.prediction import from_pytorch_tensor

model = timm.create_model("levit_128.fb_dist_in1k", pretrained=True)
# model = timm.create_model("resnet50.a1_in1k", pretrained=True)
model.eval()

input_size = timm.data.resolve_data_config(model.pretrained_cfg)["input_size"]
model_shape = list(("N",) + input_size)

if platform.uname().system == "Darwin":
    if tt.mps.is_available():
        model.to("mps")
else:
    if tt.cuda.is_available():
        model.to("cuda")


data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)  # type: ignore
transform = timm.data.create_transform(**data_cfg)  # type: ignore

# print(transform) to get the transform. In this case, the bits we need are
# Resize(235) and center_crop (224, 224)


def preprocess(path, shape, device) -> Data:
    # open the image with mode "RGB"
    img = Image.open(path).convert("RGB")

    data = Data(img, shape, device, mode="RGB")

    data.data = transform(img).unsqueeze(0).to(device)  # type: ignore

    # make a copy for visualisation
    original = Image.open(path).convert("RGB").resize((235, 235))
    original = T.functional.center_crop(original, 224)
    data.input = original

    return data
