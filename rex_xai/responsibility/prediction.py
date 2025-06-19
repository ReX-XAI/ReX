#!/usr/bin/env python3

from typing import List, Optional

import torch as tt
import torch.nn.functional as F
from numpy.typing import NDArray


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


def from_pytorch_tensor(tensor, target=None) -> List[Prediction]:
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


def default_prediction_function(model):
    def inner(mutants, target=None, raw=False):
        with tt.no_grad():
            tensor = model(mutants)
            if raw:
                return F.softmax(tensor, dim=1)
            return from_pytorch_tensor(tensor, target=target)

    return inner
