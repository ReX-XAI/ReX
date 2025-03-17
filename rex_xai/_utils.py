#!/usr/bin/env python3

import importlib.metadata
from itertools import chain, combinations
from enum import Enum
from typing import Tuple, Union, Dict
from numpy.typing import NDArray
import torch as tt
import numpy as np
from skimage.segmentation import mark_boundaries
from rex_xai.logger import logger
from rex_xai.box import Box

Strategy = Enum("Strategy", ["Global", "Spatial", "MultiSpotlight", "Contrastive"])

Queue = Enum("Queue", ["Area", "All", "Intersection", "DC"])

SpatialSearch = Enum("SpatialSearch", ["NotFound", "Found"])


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
        m = mark_boundaries(img, segs, mode="thick")
    else:
        m = mark_boundaries(img, segs, colour, mode="thick")
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
            d = coords[2]
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
    elif mode == "L":
        if order == "first":
            # (1, H, W)
            tensor[0, h, w] = val
        else:
            tensor[h, w, :] = val
    elif mode in ("spectral", "tabular"):
        if len(tensor.shape) == 1:
            tensor[h:w] = val
        else:
            tensor[0, h:w] = val
    # elif mode == "tabular":

    elif mode == "voxel":
        tensor[h, w, d] = val
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
