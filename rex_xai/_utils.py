#!/usr/bin/env python3
from typing import Tuple, Union
from numpy.typing import NDArray
import torch as tt
import numpy as np
from skimage.segmentation import mark_boundaries
from rex_xai.logger import logger
from rex_xai.box import Box


def add_boundaries(img: Union[NDArray, tt.Tensor], segs: NDArray) -> NDArray:
    m = mark_boundaries(img, segs)
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
    coords = []
    for i, r in enumerate(np.nditer(map)):
        coords.append((r, np.unravel_index(i, map.shape)))
    coords = sorted(coords, reverse=reverse)
    return coords
    #


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
        else:
            h = slice(coords.row_start, coords.row_stop)
            w = slice(coords.col_start, coords.col_stop)
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
        tensor[0, h:w] = val
    elif mode == "voxel":
        logger.warning("not yet implemented")


def ff(obj, fmt):
    return "None" if obj is None else format(obj, fmt)
