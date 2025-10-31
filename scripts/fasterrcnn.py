from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torch as tt
from rex_xai.input.input_data import Data
from rex_xai.responsibility.prediction import Prediction, Predictions
import torchvision.transforms.functional as F

# Load a model
model = fasterrcnn_resnet50_fpn_v2(pretrained=True).to("cuda")
model.eval()
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT


def preprocess(path, shape, device) -> Data:
    img = Image.open(path).convert("RGB")
    tensor = F.to_tensor(img).to(device)
    data = Data(img, shape, device, process=False)
    data.data = tensor
    data.mode = "RGB"
    data.model_shape = shape
    data.model_height = img.height
    data.model_width = img.width
    data.model_channels = 3
    data.transposed = False
    data.model_order = "first"
    data.mask_value = 0
    data.device = device
    return data


def faster_rcnn_result_to_pred(results, target):
    predictions = []
    preds = results[0]
    preds["boxes"] = preds["boxes"].cpu().numpy()
    preds["labels"] = preds["labels"].cpu().numpy()
    preds["scores"] = preds["scores"].cpu().numpy()
    if len(preds["boxes"]) == 0:
        predictions.append(Prediction("NONE", 0.0))
    else:
        for i, box in enumerate(preds["boxes"]):
            label = weights.meta["categories"][preds["labels"][i]]
            confidence = preds["scores"][i]
            box = box.tolist()
            prediction = Prediction(label, confidence, box, target)
            predictions.append(prediction)
        return Predictions(predictions)
    return Predictions((Prediction("NONE", 0.0)))


def prediction_function(mutants, target=None, raw=False, binary_threshold=None):
    with tt.no_grad():
        if len(mutants.shape) == 3:
            mutants = mutants.unsqueeze(0)
        tensor = model(mutants)
        return faster_rcnn_result_to_pred(tensor, target)


model_shape = [1, 3, "H", "W"]
