#!/usr/bin/env python
import numbers
from typing import List, Optional

import numpy as np
import torch as tt
from PIL import Image  # type: ignore

try:
    from anytree.cachedsearch import find
except ImportError:
    from anytree.search import find

import matplotlib.pyplot as plt

from rex_xai.input.input_data import Data
from rex_xai.mutants.box import Box
from rex_xai.responsibility.prediction import Prediction
from rex_xai.utils._utils import add_boundaries, set_boolean_mask_value, try_detach
from rex_xai.utils.logger import logger

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


def _apply_to_data(mask, data: Data):
    if callable(data.mask_value):
        return data.mask_value(mask, data.data)
    if isinstance(data.mask_value, numbers.Number):
        return tt.where(mask, data.data, data.mask_value)  # type: ignore

    print(data.mask_value)
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
        self.channels: int = (
            data.model_channels if data.model_channels is not None else 1
        )
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

    def get_active_boxes(self):
        return self.active.split("_")

    def area(self) -> int:
        """Return the total area *not* concealed by the mutant."""
        tensor = tt.count_nonzero(self.mask)
        if tensor.numel() == 0 or tensor is None:
            return 0
        else:
            return int(tensor.item()) // self.channels

    def set_static_mask_regions(self, names, search_tree):
        for name in names:
            box = find(search_tree, lambda node: node.name == name)
            if box is not None:
                self.depth = max(self.depth, box.depth)
                self.set_mask_region_to_true(box)

    def set_active_mask_regions(self, boxes: List[Box]):
        for box in boxes:
            self.depth = max(self.depth, box.depth)
            self.set_mask_region_to_true(box)

    def set_mask_region_to_true(self, box: Box):
        set_boolean_mask_value(self.mask, self.mode, self.order, box)

    def apply_to_data(self, data: Data):
        return _apply_to_data(self.mask, data)

    def save_mutant(self, data: Data, name=None, segs=None):
        if data.mode == "RGB":
            m = np.array(data.input)
            mask = try_detach(self.mask).squeeze()

            if data.transposed:
                # if transposed, we have C * H * W, so change that to H * W * C
                m = np.where(mask, m.transpose((2, 0, 1)), 0)
                m = m.transpose((1, 2, 0))
            else:
                mask = mask.transpose((1, 2, 0))
                m = np.where(mask, m, 0)
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
            ax.plot(m[0][0].detach().cpu().numpy())
            plt.savefig(f"{self.get_name()}.png")
        # 3d image
        if data.mode == "voxel":
            volume = self.apply_to_data(data).squeeze().detach().cpu().numpy()
            num_slices = min(volume.shape[0], 8)
            slice_indices_x = np.linspace(0, volume.shape[0] - 1, num_slices, dtype=int)
            slice_indices_y = np.linspace(0, volume.shape[1] - 1, num_slices, dtype=int)
            slice_indices_z = np.linspace(0, volume.shape[2] - 1, num_slices, dtype=int)

            fig, axes = plt.subplots(3, num_slices, figsize=(15, 6))
            for i, idx in enumerate(slice_indices_x):
                ax = axes[0, i]
                ax.imshow(volume[idx, :, :], cmap="gray")
                ax.set_title(f"X={idx}")
                ax.axis("off")

            for j, idy in enumerate(slice_indices_y):
                ax = axes[1, j]
                ax.imshow(volume[:, idy, :], cmap="gray")
                ax.set_title(f"Y={idy}")
                ax.axis("off")

            for z, idz in enumerate(slice_indices_z):
                ax = axes[2, z]
                ax.imshow(volume[:, :, idz], cmap="gray")
                ax.set_title(f"Z={idz}")
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(name or f"{self.get_name()}.png")
            plt.close(fig)
