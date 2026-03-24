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
from rex_xai.responsibility.prediction import Predictions
from rex_xai.utils._utils import (
    add_boundaries,
    set_boolean_mask_value,
    try_detach,
    try_rounding,
)
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
    def __init__(self, data: Data, static, active, masking_func, shape=None) -> None:
        self.matches = None
        if shape is not None:
            self.shape = shape
        else:
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
        self.predictions: Optional[Predictions] = None
        self.passing = False
        self.masking_func = masking_func
        self.depth = 0

    def __repr__(self) -> str:
        return f"ACTIVE: {self.active}, PREDICTION: {self.predictions}, PASSING: {self.passing}"

    def get_name(self):
        return self.active

    def update_status(
        self,
        targets: Predictions,
        iou_threshold: float = 0.5,
        conf_threshold: float = None,
        strict: bool = False,
    ):
        """Update the mutant's prediction and passing status based on the provided targets."""
        # update the mutant's prediction and passing status
        if self.predictions is not None:
            all_matching = []
            matches = []
            for target in targets:
                for pred in self.predictions:
                    if pred is not None and target is not None:
                        if conf_threshold is not None:
                            pred_conf = try_rounding(
                                pred.confidence, 4
                            )  # TODO: use args instead of hardcoding
                            target_conf = try_rounding(conf_threshold, 4)
                            if pred_conf < target_conf:
                                continue
                        if pred.classification == target.classification:
                            iou, is_matching = pred.check_overlap(
                                target, iou_threshold
                            )  # if no boxes, is_matching is True
                            if not is_matching:
                                continue
                            logger.debug(
                                f"Mutant {self.get_name()} prediction {pred} matches target {target} with IoU {iou:.3f}"
                            )
                            matches.append((pred, target, iou))
                            all_matching.append(True)
                        else:
                            all_matching.append(False)
            self.matches = matches
            if strict:
                self.passing = all(
                    all_matching
                )  # requires each target found to be matched with the original prediction
            else:
                self.passing = any(
                    all_matching
                )  # requires at least one target to be matched with the original prediction
        else:
            self.passing = False

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
            _, axes = plt.subplots(3, num_slices, figsize=(15, 6))

            for axis in range(3):  # 0=D, 1=H, 2=W
                slice_indices = np.linspace(
                    0, volume.shape[axis] - 1, num_slices, dtype=int
                )
                for i, slice_index in enumerate(slice_indices):
                    ax = axes[axis, i]
                    data_slice = np.take(volume, slice_index, axis=axis)

                    ax.imshow(data_slice, cmap="gray", vmin=0, vmax=1)
                    ax.set_title(f"Axis {axis}, Slice {slice_index}")
                    ax.axis("off")

            plt.tight_layout()
            plt.savefig(name or f"{self.get_name()}.png")


def filter_passing_mutants(
    mutants: List[Mutant], targets: Predictions, confidence_filter
) -> List[Mutant]:
    """Filter and return only the passing mutants from the provided list."""

    def passed_confidence(m: Mutant) -> bool:
        # if it didn't pass, discard immediately
        if not m.passing:
            return False

        # if there is exactly one prediction and one target, easy case
        if len(m.matches) == 1 and len(m.predictions) == 1 and len(targets) == 1:
            return (
                m.predictions[0].confidence >= targets[0].confidence * confidence_filter
            )

        # multiple case: check only matched pairs
        for pred, target, iou in m.matches:
            if pred.confidence < target.confidence * confidence_filter:
                return False

        return True

    return [m for m in mutants if passed_confidence(m)]
