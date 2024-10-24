#!/usr/bin/env python3

import timm
from PIL import Image
import torch as tt
import torch.nn.functional as F
from rex_xai.input_data import Data
from rex_xai.prediction import from_pytorch_tensor


model = timm.create_model('resnet152.a2_in1k', pretrained=True).to("mps")
model.eval()


def preprocess(path, shape, device, mode) -> Data:
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg) #type: ignore
    transform = timm.data.create_transform(**data_cfg) #type: ignore

    img = Image.open(path).convert("RGB")
    data = Data(img, shape, device, mode=mode, process=False)
    data.data = transform(img).unsqueeze(0).to(device)
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
#
#
def model_shape():
    return ["N", 3, 224, 224]
#
#
