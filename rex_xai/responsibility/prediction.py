#!/usr/bin/env python3
import logging
from typing import List, Optional, Tuple, Iterator

import torch as tt
import torch.nn.functional as F
from numpy.typing import NDArray
import numpy as np


class Prediction:
    def __init__(
        self,
        pred=None,
        conf=None,
        box=None,
        target=None,
        target_confidence=None,
    ) -> None:
        self.classification: Optional[int] = pred
        self.confidence: Optional[float] = conf
        self.bounding_box: Optional[NDArray] = box
        self.target: Optional[int] = None if target is None else target.classification
        self.target_confidence: Optional[float] = target_confidence

    def __repr__(self) -> str:
        if self.bounding_box is None:
            if self.is_passing():
                return (
                    f"FOUND_CLASS: {self.classification}, CONF: {self.confidence:.5f}"
                )
            else:
                if self.target is None:
                    return f"FOUND_CLASS: {self.classification}, FOUND_CONF: {self.confidence:.5f}, TARGET_CLASS: n/a, TARGET_CONFIDENCE: n/a"
                else:
                    return f"FOUND_CLASS: {self.classification}, FOUND_CONF: {self.confidence:.5f}, TARGET_CLASS: {self.target}, TARGET_CONFIDENCE: {(self.target_confidence, '.5f')}"

        return f"CLASS: {self.classification}, CONF: {self.confidence:.5f}, TARGET_CLASS: {self.target}, TARGET_CONFIDENCE: {(self.target_confidence, '.5f')}, BOUNDING_BOX: {self.bounding_box}"

    def get_class(self):
        return self.classification

    def is_empty(self):
        return self.classification is None or self.confidence is None

    def is_passing(self):
        return self.target == self.classification

    def check_overlap(self, prediction: 'Prediction', percentage: float = 0.5) -> Tuple[float, bool]:
        """
        Compute IoU between this.prediction.bounding_box and another `prediction.bounding_box`.
        Accepts boxes in either (x1, y1, x2, y2) or (x, y, w, h) format.
        `percentage` can be a fraction in [0,1] (e.g. 0.5) or a percent in (0,100) (e.g. 50).
        Returns: (iou, iou >= threshold)
        """
        if self.bounding_box is None or prediction.bounding_box is None:
            return 0.0, True # no boxes to compare, consider as passing

        boxA = np.array(self.bounding_box, dtype=float)
        boxB = np.array(prediction.bounding_box, dtype=float)

        if percentage > 1:
            threshold = percentage / 100.0
        else:
            threshold = float(percentage)

        a = to_xyxy(boxA)
        b = to_xyxy(boxB)

        # intersection
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])

        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
        area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))

        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            iou = 0.0
        else:
            iou = inter_area / union_area

        return float(iou), (iou >= threshold)


class Predictions(List[Optional[Prediction]]):
    """
    A wrapper for a list of Prediction objects.
    This class provides easy access to the classifications, confidences and bounding boxes.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if args[0] is not None and isinstance(args[0], list) and type(args[0][0]) is Prediction:
            self._predictions: List[Prediction] = args[0]
            self.classifications = [
                p.classification if p is not None else None for p in self._predictions
            ]
            self.confidences = [
                p.confidence if p is not None else None for p in self._predictions
            ]
            self.bounding_boxes = [
                p.bounding_box if p is not None else None for p in self._predictions
            ]
        else:
            logging.warning("Predictions initialized without a list of Prediction objects.")
            self._predictions: List[Prediction] = []
            self.classifications: List[Optional[int]] = []
            self.confidences: List[Optional[float]] = []
            self.bounding_boxes: List[Optional[NDArray]] = []

    def __repr__(self) -> str:
        return f"Predictions({self._predictions})"

    def __getitem__(self, index) -> Prediction|None:
        return self._predictions[index]

    def __setitem__(self, index, value) -> None:
        self._predictions[index] = value

    def __len__(self) -> int:
        return len(self._predictions)

    def __iter__(self) -> Iterator[Prediction]:
        return iter(self._predictions)

    def append(self, value: Prediction):
        self._predictions.append(value)
        self.classifications.append(value.classification)
        self.confidences.append(value.confidence)
        self.bounding_boxes.append(value.bounding_box)
        return self



def from_pytorch_tensor(tensor, target=None) -> Predictions|List[Predictions]:
    """Convert a PyTorch tensor to a list of Predictions. If the batch size is 1, returns a single Predictions object.
    If the batch size is greater than 1, returns a list of Predictions objects."""
    softmax_tensor = F.softmax(tensor, dim=1)
    prediction_scores, pred_labels = tt.topk(softmax_tensor, 1)
    prediction: List[Prediction] = []
    batch_size = tensor.shape[0]
    if batch_size == 1:
        for i, (ps, pl) in enumerate(zip(prediction_scores, pred_labels)):
            p = Prediction(pl.item(), ps.item())
            if target is not None:
                p.target = target
                p.target_confidence = softmax_tensor[i, target[0].classification].item()
            prediction.append(p)
        return Predictions(prediction)
    else:
        # more than one batch
        predictions: List[Predictions] = []
        for i in range(batch_size):
            batch_pred = []
            p = Prediction(pred_labels[i].item(), prediction_scores[i].item())
            if target is not None:
                p.target = target
                p.target_confidence = softmax_tensor[i, target[0].classification].item()
            batch_pred.append(p)
            predictions.append(Predictions(batch_pred))
        return predictions


def default_prediction_function(model):
    def inner(mutants, target=None, raw=False):
        with tt.no_grad():
            tensor = model(mutants)
            if raw:
                return F.softmax(tensor, dim=1)
            return from_pytorch_tensor(tensor, target=target)

    return inner

def to_xyxy(box: np.ndarray) -> np.ndarray:
    if box.size != 4:
        raise ValueError("Bounding box must be length 4.")
    x0, y0, x1, y1 = box
    if (x1 <= x0) or (y1 <= y0):
        x, y, w, h = box
        return np.array([x, y, x + w, y + h], dtype=float)
    return np.array([x0, y0, x1, y1], dtype=float)