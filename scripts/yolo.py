from PIL import Image
from ultralytics import YOLO
import torch as tt
from rex_xai.input.input_data import Data
from rex_xai.responsibility.prediction import Prediction
import numpy as np

# Load a model
model = YOLO("yolo11n.pt")
path = "tests/test_data/dog_hide.jpg" # Change this later

def preprocess(path, shape, device, mode) -> Data:
    img = Image.open(path).convert("RGB")
    tensor = tt.tensor(np.asarray(img), dtype=tt.float32).to(device)
    data = Data(img, shape, device, mode=mode, process=False)
    print(f"Data shape: {tensor.shape}")
    data.data = tensor
    data.mode = "RGB"
    data.model_shape = shape
    print(f"Proposed shape: {shape}")
    data.model_height = img.height
    data.model_width = img.width
    data.model_channels = 3
    data.transposed = False
    data.model_order = "last"
    data.mask_value = 0
    data.device = "cuda"
    return data

def yolo_result_to_pred(results, target):
    predictions = []
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        # print(f"Boxes: {boxes}")
        if len(boxes.cls) == 0:
            predictions.append(Prediction("NONE"))
            continue
        elif len(boxes.cls) == 1:
            label = result.names.get(boxes.cls.item())
            confidence = boxes.conf.item()
            box= boxes.xyxy
            prediction = Prediction(label, confidence, box, target)
            predictions.append(prediction)
        else:
            for i, box in enumerate(boxes):
                label = result.names.get(box.cls.item())
                print(f"There are multi class {label}")
                confidence = box.conf.item()
                prediction = Prediction(label, confidence, box, target)
                predictions.append(prediction)
    return predictions


def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        if len(mutants.shape) == 4:
            mutants = mutants.squeeze(0)
        tensor = model(mutants.cpu().numpy(), verbose=False)
        return yolo_result_to_pred(tensor, target)

def model_shape():
    img = Image.open(path)
    return ["N", img.height, img.width, 3]