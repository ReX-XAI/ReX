#!/usr/bin/env python3

from PIL import Image  # type: ignore
from torchvision import transforms as T
from torchvision.models import swin_v2_t

from rex_xai.input.input_data import Data

model = swin_v2_t(weights="DEFAULT")
model.eval()
model.to("cpu")


def preprocess(path, shape, device) -> Data:
    transform = T.Compose(
        [
            T.Resize((260, 260), T.InterpolationMode.BICUBIC),
            T.CenterCrop(256),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(path).convert("RGB")
    data = Data(img, shape, device)
    data.data = transform(img).unsqueeze(0).to(device)  # type: ignore

    return data


model_shape = ["N", 3, 256, 256]
