#!/usr/bin/env python3

import platform

from PIL import Image  # type: ignore
from torchvision import transforms as T
from torchvision.models import get_model, get_weight

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

model = get_model("swin_s", weights="DEFAULT")
weights = get_weight("Swin_S_Weights.IMAGENET1K_V1")
model.eval()

# you have to include this
model_shape = ["N", 3, 224, 224]

if platform.uname().system == "Darwin":
    model.to("mps")
else:
    model.to("cuda")


# you have to write this
def preprocess(path, shape, device) -> Data:
    # open the image with mode "RGB"
    img = Image.open(path).convert("RGB")

    data = Data(img, shape, device, mode="RGB")

    data.data = weights.transforms()(img).unsqueeze(0).to(device)  # type: ignore

    # make a copy for visualisation
    original = Image.open(path).convert("RGB").resize((246, 246))
    original = T.functional.center_crop(original, 224)
    data.input = original

    return data
