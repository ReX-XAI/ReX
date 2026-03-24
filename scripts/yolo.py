from PIL import Image
from ultralytics import YOLO
import torch as tt
from rex_xai.input.input_data import Data
from rex_xai.responsibility.prediction import Prediction, Predictions
import numpy as np

# Load a model
model = YOLO("yolo11n.pt")
model.eval()


def preprocess(path, shape, device) -> Data:
    img = Image.open(path).convert("RGB")
    tensor = tt.tensor(np.asarray(img), dtype=tt.float32).to(device)
    data = Data(img, shape, device, process=False)
    data.data = tensor
    data.mode = "RGB"
    data.model_shape = shape
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
        if len(boxes.cls) == 0:
            predictions.append(Prediction("NONE", 0.0))
            continue
        else:
            for i, box in enumerate(boxes):
                label = result.names.get(box.cls.item())
                confidence = box.conf.item()
                box = box.xyxy.cpu().numpy()[0]
                prediction = Prediction(label, confidence, box, target)
                predictions.append(prediction)
    return Predictions(predictions)


def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        if len(mutants.shape) == 4:
            mutants = mutants.squeeze(0)
        tensor = model(mutants.cpu().numpy(), verbose=False)
        return yolo_result_to_pred(tensor, target)


model_shape = [1, "H", "W", 3]
