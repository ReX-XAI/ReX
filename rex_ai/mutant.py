#!/usr/bin/env python
from typing import List, Optional
import numpy as np
import torch as tt
from PIL import Image  # type: ignore

try:
    from anytree.cachedsearch import find
except ImportError:
    from anytree.search import find

import matplotlib.pyplot as plt
from rex_ai.box import Box
from rex_ai.prediction import Prediction
from rex_ai.logger import logger
from rex_ai.input_data import Data
from rex_ai._utils import add_boundaries, set_boolean_mask_value

__combinations = [
    [
        0,
    ],
    [
        1,
    ],
    [
        2,
    ],
    [
        3,
    ],
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 3],
    [2, 3],
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3],
]


def _apply_to_data(mask, data: Data, masking_func):
    if isinstance(masking_func, (float, int)): 
        return tt.where(mask, data.data, masking_func)  #type: ignore
    if callable(masking_func):
        return masking_func(mask, data.data)

    logger.warning("applying default masking value of 0")
    return tt.where(mask, data.data, 0)  # type: ignore


def get_combinations():
    return __combinations


class Mutant:
    def __init__(self, data: Data, static, active, masking_func) -> None:
        self.shape = tuple(
            data.model_shape[1:]
        )  # the first element of shape is the batch information, so we drop that
        self.mode = data.mode
        self.channels: int = data.model_channels if data.model_channels is not None else 0
        self.order = data.model_order
        self.mask = tt.zeros(self.shape, dtype=tt.bool, device=data.device)
        self.static = static
        self.active = active
        self.prediction: Optional[Prediction] = None
        self.passing = False
        self.masking_func = masking_func
        self.depth = 0

    def __repr__(self) -> str:
        return f"ACTIVE: {self.active}, PREDICTION: {self.prediction}, PASSING: {self.passing}"

    def get_name(self):
        return self.active

    def update_status(self, target):
        if self.prediction is not None:
            if target.classification == self.prediction.classification:
                self.passing = True

    def get_length(self):
        return len(self.active.split("_"))

    def get_active_parts(self):
        return self.active.split("_")

    def area(self) -> int:
        """Return the total area *not* concealed by the mutant."""
        tensor = tt.count_nonzero(self.mask)
        if tensor.numel() == 0 or tensor is None:
            return 0
        else:
            return int(tensor.item()) // self.channels

    def predict(self, prediction_func, data, target):
        p = prediction_func(self.apply_to_data(data), target)
        self.prediction = p

        if data.classification == p.classification:
            self.passing = True

    def set_static_mask_regions(self, names, search_tree):
        for name in names:
            s = find(search_tree, lambda node: node.name == name)
            if s is not None:
                self.depth = max(self.depth, s.depth)
                self.set_mask_region_to_true(s)

    def set_active_mask_regions(self, boxes: List[Box]):
        for box in boxes:
            self.depth = max(self.depth, box.depth)
            self.set_mask_region_to_true(box)

    def set_mask_region_to_true(self, box: Box):
        set_boolean_mask_value(self.mask, self.mode, self.order, box)

    def apply_to_data(self, data: Data):
        return _apply_to_data(self.mask, data, self.masking_func)

    def save_mutant(self, data: Data, name=None, segs=None):
        colour = self.masking_func
        # image
        if data.mode in ("RGB", "L"):
            # an assumption here
            if isinstance(colour, float):
                colour *= 255
                colour = np.uint8(colour)
                print(f'casting {self.masking_func} to {colour}')

            m = np.array(
                data.input.resize((data.model_height, data.model_width))
            )
            mask = self.mask.squeeze().detach().cpu().numpy()
            if data.transposed:
                # if transposed, we have C * H * W, so change that to H * W * C
                m = np.where(mask, m.transpose((2, 0, 1)), colour)
                m = m.transpose((1, 2, 0))
            else:
                m = np.where(mask, m, colour)
            # draw on the segment_mask, if available
            if segs is not None:
                m = add_boundaries(m, segs)
            img = Image.fromarray(m, data.mode)
            if name is not None:
                img.save(name)
            else:
                img.save(f"{self.get_name()}.png")
        # spectral or time series data
        if data.mode == "spectral":
            m = self.apply_to_data(data)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(m[0][0])
            plt.savefig(f"{self.get_name()}.png")
        # 3d image
        if data.mode == "voxel":
            pass
