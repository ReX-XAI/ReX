#!/usr/bin/env python3

import torch as tt
import torch.nn.functional as F
from numpy.typing import NDArray
from typing import Optional, List


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
                return f"CLASS: {self.classification}, CONF: {self.confidence}"
            else:
                return f"CLASS: {self.classification}, CONF: {self.confidence}, TARGET_CLASS: {self.target}, TARGET_CONFIDENCE: {self.target_confidence}"

        return f"CLASS: {self.classification}, CONF: {self.confidence}, TARGET_CLASS: {self.target}, TARGET_CONFIDENCE: {self.target_confidence}, BOUNDING_BOX: {self.bounding_box}"

    def get_class(self):
        return self.classification

    def is_empty(self):
        return self.classification is None or self.confidence is None

    def is_passing(self):
        return self.target == self.classification


def from_pytorch_tensor(tensor, target=None, binary_threshold=None) -> List[Prediction]:
    # TODO get this to handle binary models
    softmax_tensor = F.softmax(tensor, dim=1)
    prediction_scores, pred_labels = tt.topk(softmax_tensor, 1)
    predictions = []
    for i, (ps, pl) in enumerate(zip(prediction_scores, pred_labels)):
        p = Prediction(pl.item(), ps.item())
        if target is not None:
            p.target = target
            p.target_confidence = softmax_tensor[i, target.classification].item()
        predictions.append(p)

    return predictions
