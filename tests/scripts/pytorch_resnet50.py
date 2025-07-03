#!/usr/bin/env python3

from PIL import Image  # type: ignore
from torchvision import transforms as T
from torchvision.models import resnet50

from rex_xai.input.input_data import Data

model = resnet50(weights="ResNet50_Weights.DEFAULT")
model.eval()

transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess(path, shape, device) -> Data:
    # open the image with mode "RGB"
    img = Image.open(path).convert("RGB")
    # create a Data object
    data = Data(img, shape, device, mode="RGB")
    # manually set the data to the transformed image for model consumption
    data.data = transform(img).unsqueeze(0).to(device)  # type: ignore
    original = Image.open(path).convert("RGB").resize((224, 224))
    data.input = original
    return data


model_shape = ("N", 3, 224, 224)
