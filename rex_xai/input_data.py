#!/usr/bin/env python
from typing import Optional
import numpy as np
import torch as tt

from enum import Enum

from rex_xai.occlusions import spectral_occlusion, context_occlusion
from rex_xai.prediction import Prediction
from rex_xai.logger import logger
from rex_xai._utils import ReXDataError

Setup = Enum("Setup", ["ONNXMPS", "ONNX", "PYTORCH"])


def _guess_mode(input):
    if hasattr(input, "mode"):
        return input.mode
    if hasattr(input, "shape"):
        if len(input.shape) == 4:
            return "voxel"
        else:
            return "spectral"


class Data:
    def __init__(self, input, model_shape, device, mode=None, process=True) -> None:
        self.input = input
        self.mode = None
        self.target: Optional[Prediction] = None
        self.device = device
        self.setup: Optional[Setup] = None

        self.mode = mode
        if mode is None:
            self.mode = _guess_mode(input)

        self.model_shape = model_shape
        height, width, channels, order, depth = self.__get_shape()
        self.model_height: Optional[int] = height
        self.model_width: Optional[int] = width
        self.model_depth: Optional[int] = depth
        self.model_channels: Optional[int] = channels if channels is not None else 1
        self.model_order = order
        self.mask_value = None
        self.background = None
        self.context = None

        if process:
            # RGB model but greyscale input so we convert greyscale to pseudo-RGB
            if self.model_channels == 3 and self.mode == "L":
                self.input = self.input.convert("RGB")
                self.mode = "RGB"
            elif self.mode == "RGB" and self.model_order == "first":
                self.transposed = True
            elif self.mode in ("tabular", "spectral"):
                self.data = self.input
                self.match_data_to_model_shape()
            elif self.mode == "voxel":
                self.data = self.input
            else:
                raise NotImplementedError

    def set_height(self, h: int):
        self.model_height = h

    def set_width(self, w: int):
        self.model_width = w

    def set_channels(self, c=None):
        self.model_channels = c

    def __repr__(self) -> str:
        data_info = f"Data: {self.mode}, {self.model_shape}, {self.model_height}, {self.model_width}, {self.model_channels}, {self.model_order}"
        if self.target is not None:
            target_info = repr(self.target)
            data_info = data_info + "\n\t Target:" + target_info
        return data_info

    def set_classification(self, cl):
        self.classification = cl

    def match_data_to_model_shape(self):
        """
        a PIL image has the from H * W * C, so
        if the model takes C * H * W we need to transpose self.data to
        get it into the correct form for the model to consume
        This function does *not* add in the batch channel at the beginning
        """
        assert self.data is not None
        if self.mode == "RGB" and self.model_order == "first":
            self.data = self.data.transpose(2, 0, 1)  # type: ignore
            self.transposed = True
        if self.mode == "L" and self.model_order == "first":
            self.data = self.data.transpose(1, 0)
            self.transposed = True
        if self.mode == "L" and self.model_order == "last":
            self.data = self.data.transpose(1, 0)
        if self.mode == "tabular" or self.mode == "spectral":
            self.data = self.generic_tab_preprocess()
        if self.mode == "voxel":
            pass
        self.data = self.try_unsqueeze()

    def generic_tab_preprocess(self):
        if isinstance(self.input, np.ndarray):
            self.data = self.input.astype("float32")
            arr = tt.from_numpy(self.data).to(self.device)
        else:
            arr = self.input
        for _ in range(len(self.model_shape) - len(arr.shape)):
            arr = arr.unsqueeze(0)
        return arr

    def load_data(self, astype="float32"):
        img = self.input.resize((self.model_height, self.model_width))
        img = np.array(img).astype(astype)
        self.data = img
        self.match_data_to_model_shape()
        self.data = tt.from_numpy(self.data).to(self.device)

    def _normalise(self, means, stds, astype, norm):
        assert self.data is not None

        normed_data = self.data
        if norm is not None:
            normed_data /= norm

        if self.model_order == "first" and self.model_channels == 3:
            if means is not None:
                for i, m in enumerate(means):
                    normed_data[:, i, :, :] = normed_data[:, i, :, :] - m
            if stds is not None:
                for i, s in enumerate(stds):
                    normed_data[:, i, :, :] = normed_data[:, i, :, :] / s

        if self.model_order == "last" and self.model_channels == 3:
            if means is not None:
                for i, m in enumerate(means):
                    normed_data[:, :, i] = normed_data[:, :, i] - m
            if stds is not None:
                for i, s in enumerate(stds):
                    normed_data[:, :, i] = normed_data[:, :, i] / s

        # greyscale
        if self.model_channels == 1:
            if means is not None:
                for i, m in enumerate(means):
                    normed_data[i] = normed_data[i] - m
            if stds is not None:
                for i, s in enumerate(stds):
                    normed_data[i] = normed_data[i] / s

        return normed_data

    def try_unsqueeze(self):
        out = self.data
        if self.model_order == "first":
            dim = 0
        else:
            dim = -1
        if isinstance(self.data, tt.Tensor):
            for _ in range(len(self.model_shape) - len(self.data.shape) - 1):
                out = tt.unsqueeze(out, dim=dim)  # type: ignore
            out = tt.unsqueeze(out, dim=0)  # type: ignore
        else:
            for _ in range(len(self.model_shape) - len(self.data.shape) - 1):  # type: ignore
                out = np.expand_dims(out, axis=dim)  # type: ignore
            out = np.expand_dims(out, axis=0)  # type: ignore
        return out

    def generic_image_preprocess(
        self,
        means=None,
        stds=None,
        astype="float32",
        norm: Optional[float] = 255.0,
    ):
        self.load_data(astype=astype)

        if self.mode == "RGB" and self.data is not None:
            self.data = self._normalise(means, stds, astype, norm)
            self.try_unsqueeze()
        if self.mode == "L":
            self.data = self._normalise(means, stds, astype, norm)

    def __get_shape(self):
        """returns height, width, channels, order, depth for the model"""
        if self.mode == "spectral":
            # an array of the form (h, w), so no channel info or order or depth
            if len(self.model_shape) == 2:
                return self.model_shape[0], self.model_shape[1], 1, None, None
            # an array of the form (batch, h, w), so no channel info or order or depth
            if len(self.model_shape) == 3:
                return self.model_shape[1], self.model_shape[2], 1, None, None
        if self.mode in ("RGB", "RGBA"):
            if len(self.model_shape) == 4:
                _, a, b, c = self.model_shape
                if a in (1, 3, 4):
                    return b, c, a, "first", None
                else:
                    return a, b, c, "last", None
        if self.mode == "voxel":
            if len(self.model_shape) == 4:
                batch, w, h, d = self.model_shape  # If batch is present
                return w, h, None, None, d
            else:
                w, h, d = self.model_shape
                return w, h, None, None, d

        raise ReXDataError(
            f"Incompatible 'mode' {self.mode}  and 'model_shape' ({self.model_shape}), cannot get valid shape of Data object so returning None"
        )

        # elif self.mode in ("RGB", "RGBA", "L") and len(self.model_shape) == 4:
        #     _, a, b, c = self.model_shape
        #     if a == 1 or a == 3 or a == 4:
        #         return b, c, a, "first"
        #     else:
        #         return a, b, c, "last"
        # elif self.mode == "voxel":
        #     pass
        # else:
        #     logger.warning("Incompatible 'mode' and 'model_shape', cannot get valid shape of Data object so returning None")
        #     return None, None, None, None

    def set_mask_value(self, m):
        assert self.data is not None
        # if m is a number, then if might still need to be normalised

        if m == "spectral" and self.mode != "spectral":
            logger.warning(
                "Mask value 'spectral' can only be used if mode is also 'spectral', using default mask value 0 instead"
            )
            m = 0

        match m:
            case int() | float() as m:
                self.mask_value = m
            case "min":
                self.mask_value = tt.min(self.data).item()  # type: ignore
            case "mean":
                self.mask_value = tt.mean(self.data).item()  # type: ignore
            case "spectral":
                self.mask_value = lambda m, d: spectral_occlusion(
                    m, d, device=self.device
                )
            case "context":
                self.mask_value = lambda m, d: context_occlusion(m, d, self.context)
                # TODO: Add args for noise and setting the context as currently only available through custom script
            case _:
                raise ValueError(
                    f"Invalid mask value {m}. Should be an integer, float, or one of 'min', 'mean', 'spectral'"
                )
