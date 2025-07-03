#!/usr/bin/env python3

import importlib.metadata
from enum import Enum
from itertools import chain, combinations
from typing import Dict, List, Tuple, Union

import numpy as np
import torch as tt
from numpy.typing import NDArray
from skimage.segmentation import mark_boundaries

from rex_xai.mutants.box import Box
from rex_xai.responsibility.prediction import Prediction
from rex_xai.utils.logger import logger

Strategy = Enum("Strategy", ["Global", "Spatial", "MultiSpotlight", "Contrastive"])

Queue = Enum("Queue", ["Area", "All", "Intersection", "DC"])

SpatialSearch = Enum("SpatialSearch", ["NotFound", "Found"])

ResponsibilityStyle = Enum("ResponsibilityStyle", ["Additive", "Multiplicative"])


def update_mask_shape(batch_size: int, mask_shape) -> Tuple[int]:
    if isinstance(mask_shape, tuple):
        mask_shape = list(mask_shape)

    mask_shape[0] = batch_size
    return tuple(mask_shape)


class ReXPositions:
    def __init__(
        self,
        sufficient_position=None,
        contrastive_position=None,
        sufficiency_found=False,
        current_max_necessity=0.0,
    ) -> None:
        self.sufficient_position = sufficient_position
        self.contrastive_position = contrastive_position
        self.sufficiency_found = sufficiency_found
        self.current_max_necessity = current_max_necessity

    def __repr__(self) -> str:
        return f"suff: {self.sufficient_position}, con: {self.contrastive_position}, suff_found: {self.sufficiency_found}"

    def is_empty(self):
        return self.sufficient_position is None and self.contrastive_position is None


def try_rounding(n, rounding: int | None) -> float:
    if rounding is None:
        return n
    else:
        return round(n, rounding)


def find_required_prediction(
    target: int,
    threshold: float,
    insertion_predictions: List[Prediction],
    contrastive_completeness_threshold: float = 0.0,
    deletion_predictions: List[Prediction] | None = None,
    rounding=None,
    sufficiency_found=False,
):
    positions = ReXPositions(sufficiency_found=sufficiency_found)
    if deletion_predictions is None:
        for i, p in enumerate(insertion_predictions):
            local_confidence = try_rounding(p.confidence, rounding)
            threshold = try_rounding(threshold, rounding)
            if p.classification == target and local_confidence >= threshold:  # type: ignore
                positions.sufficient_position = i
                return positions
    else:
        for i in range(0, len(insertion_predictions)):
            local_confidence = try_rounding(
                insertion_predictions[i].confidence, rounding
            )
            threshold = try_rounding(threshold, rounding)
            contrastive_completeness_threshold = try_rounding(
                contrastive_completeness_threshold, rounding
            )

            # check for a sufficiency
            if (
                insertion_predictions[i].classification == target
                and local_confidence >= threshold  # type: ignore
                and not positions.sufficiency_found
            ):
                positions.sufficient_position = i
                positions.sufficiency_found = True

            # check for necessity above threshold
            if (
                insertion_predictions[i].classification == target
                and local_confidence >= contrastive_completeness_threshold
                and deletion_predictions[i].classification != target
            ):
                positions.contrastive_position = i
    return positions


def find_complete_prediction(
    target: int,
    target_confidence,
    insertion_predictions: List[Prediction],
    rounding=2,
):
    for i in range(0, len(insertion_predictions)):
        p = insertion_predictions[i]
        if (
            p.classification == target
            and round(p.confidence, rounding) == round(target_confidence, rounding)  # type: ignore
        ):
            return i
    return None


def match_resposnibility_style(s: str) -> ResponsibilityStyle:
    if s == "additive":
        return ResponsibilityStyle.Additive
    if s == "multiplicative":
        return ResponsibilityStyle.Multiplicative
    raise ReXTomlError(f"{s} is an unknown responsibility style")


def try_detach(t) -> np.ndarray:
    if isinstance(t, tt.Tensor):
        return t.detach().cpu().numpy()
    elif isinstance(t, np.ndarray):
        return t
    else:
        raise ReXDataError("trying to convert a non-array into a numpy array")


def one_d_permute(tensor):
    perm = tt.randperm(len(tensor))
    return tensor[perm], perm


def powerset(r, reverse=True):
    ps = list(chain.from_iterable(combinations(r, lim) for lim in range(1, len(r) + 1)))
    if reverse:
        return reversed(ps)
    else:
        return ps


def clause_area(clause, areas: Dict) -> int:
    tot = 0
    for c in clause:
        tot += areas[c]
    return tot


class ReXError(Exception):
    pass


class ReXTomlError(ReXError):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"ReXTomlError: {self.message}"


class ReXPathError(ReXError):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"ReXPathError: no such file exists at {self.message}"


class ReXScriptError(ReXError):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"ReXScriptError: {self.message}"


class ReXDataError(ReXError):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"ReXDataError: {self.message}"


class ReXMapError(ReXError):
    def __init__(self, message) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"ReXMapError: {self.message}"


def xlogx(ps):
    f = np.vectorize(_xlogx)
    return f(ps)


def _xlogx(p):
    if p == 0.0:
        return 0.0
    else:
        return p * np.log2(p)


def add_boundaries(
    img: Union[NDArray, tt.Tensor], segs: NDArray, colour=None
) -> NDArray:
    if colour is None:
        m = mark_boundaries(img, segs, mode="inner")
    else:
        m = mark_boundaries(img, segs, colour, mode="inner")
    m *= 255  # type: ignore
    m = m.astype(np.uint8)
    return m


def get_device(gpu: bool):
    if tt.backends.mps.is_available() and gpu:
        return tt.device("mps")
    if tt.device("cuda") and gpu:
        return tt.device("cuda")
    if gpu:
        logger.warning("gpu not available")
    return tt.device("cpu")


def get_map_locations(map, reverse=True):
    if isinstance(map, tt.Tensor):
        map = map.detach().cpu().numpy()
    coords = []
    for i, r in enumerate(np.nditer(map)):
        coords.append((r, np.unravel_index(i, map.shape)))
    coords = sorted(coords, reverse=reverse)
    return coords


def set_boolean_mask_value(
    tensor,
    mode,
    order,
    coords: Union[Box, Tuple[NDArray, NDArray]],
    val: bool = True,
):
    if isinstance(coords, Box):
        if mode in ("spectral", "tabular"):
            h = coords.col_start
            w = coords.col_stop
        elif mode == "voxel":
            h = slice(coords.row_start, coords.row_stop)
            w = slice(coords.col_start, coords.col_stop)
            d = slice(coords.depth_start, coords.depth_stop)
        else:
            h = slice(coords.row_start, coords.row_stop)
            w = slice(coords.col_start, coords.col_stop)
    else:
        if mode == "voxel":
            h = coords[0]
            w = coords[1]
            d = coords[2]  # type: ignore
        else:
            h = coords[0]
            w = coords[1]

    # three channels
    if mode == "RGB":
        # (C, H, W)
        if order == "first":
            tensor[:, h, w] = val
        # (H, W, C)
        else:
            tensor[h, w, :] = val
    elif mode in ("spectral", "tabular"):
        if len(tensor.shape) == 1:
            tensor[h:w] = val
        else:
            tensor[0, h:w] = val
    elif mode == "voxel":
        tensor[h, w, d] = val  # type: ignore
    else:
        raise ReXError("mode not recognised")


def ff(obj, fmt):
    """
    Like format(obj, fmt), but returns the string 'None' if obj is None.
    See the help for format() to see acceptable values for fmt.
    """
    return "None" if obj is None else format(obj, fmt)


def version():
    return importlib.metadata.version("rex-xai")
