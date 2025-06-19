#!/usr/bin/env python3

import platform

from PIL import Image  # type: ignore
from torchvision import transforms as T
from torchvision.models import get_model

from rex_xai.input.input_data import Data

"""
a sample script for a torchvision model. This has its own custom transform and uses the 
default prediction function provided in `rex_xai/responsibility/prediction.py`.

The file must contain *at least* a variable called `model_shape` of type `List`.
It must also contain a function `preprocess(path, shape, device) -> Data` which
returns a ReX `Data` object (`rex_xai/input/input_data.py`).

If there is no `prediction_function(mutants, target=None, raw=False)` then 
ReX will use the default prediction function provided in `rex_xai/responsibility/prediction.py`.
ReX expects the model to be called simply `model`.

Alternatively, you can write your own `prediction_function` which return a list of `Prediction`
objects. 
"""

model = get_model("resnet50", weights="DEFAULT")
model.eval()

# you have to include this
model_shape = ["N", 3, 224, 224]

if platform.uname().system == "Darwin":
    model.to("mps")
else:
    model.to("cuda")

transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# you have to write this
def preprocess(path, shape, device) -> Data:
    # open the image with mode "RGB"
    img = Image.open(path).convert("RGB")

    data = Data(img, shape, device, mode="RGB")

    # manually set the data to the transformed image for model consumption
    data.data = transform(img).unsqueeze(0).to(device)  # type: ignore
    # make a copy
    original = Image.open(path).convert("RGB").resize((224, 224))
    data.input = original

    return data


# def prediction_function(mutants, target=None, raw=False):
#     with tt.no_grad(): # we don't use the grad and inference is faster without it
#         tensor = model(mutants)
#         if raw: # used when computing insertion/deletion curves
#             return F.softmax(tensor, dim=1)
#         # from_pytorch_tensor consumes a tensor and converts it to a Prediction object
#         # you can  alternatively use your own function here
#         return from_pytorch_tensor(tensor, target=target)
