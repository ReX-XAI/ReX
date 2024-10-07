#!/usr/bin/env python
from typing import Optional
import numpy as np
import torch as tt

from enum import Enum

Setup = Enum("Setup", ["ONNXMPS", "ONNX", "PYTORCH"])


def _guess_mode(input):
    try:
        return input.mode
    except Exception:
        return "spectral"


class Data:
    def __init__(
        self, input, model_shape, device, mode=None, process=True
    ) -> None:
        self.input = input
        self.mode = None
        self.classification = None
        self.device = device
        self.setup: Optional[Setup] = None

        if process:
            if mode is None:
                self.mode = _guess_mode(input)
            else:
                self.mode = mode

        self.model_shape = model_shape
        height, width, channels, order = self.__get_shape()
        self.model_height: Optional[int] = height
        self.model_width: Optional[int] = width
        self.model_channels: Optional[int] = channels
        self.model_order = order

        if process:
            if self.mode == "tabular" or self.mode == "spectral":
                self.data = self.input
                self.match_to_model()
            else:
                self.data = None
            self.transposed = False

    def __repr__(self) -> str:
        return f"Data: {self.mode}, {self.model_shape}, {self.model_height}, {self.model_width}, {self.model_channels}, {self.model_order}"

    def set_classification(self, cl):
        self.classification = cl

    def match_to_model(self):
        """
        a PIL image has the from H * W * C, so
        if the model takes C * H * W we need to transpose self.data to
        get it into the correct form for the model to consume
        This function does *not* add in the batch channel at the beginning
        """
        if (
                self.mode == "RGB"
            and self.model_order == "first"
            and self.data is not None
        ):
            self.data = self.data.transpose(2, 0, 1)  # type: ignore
            self.data = self.unsqueeze()
            self.transposed = True
        if self.mode == "L":
            self.data = self.data.transpose(1, 0)
            self.data = self.unsqueeze()
            self.transposed = True
        if self.mode == "tabular" or self.mode == "spectral":
            self.generic_tab_preprocess()
        if self.mode == "voxel":
            pass

    def generic_tab_preprocess(self):
        arr = self.input.astype("float32")
        for _ in range(len(self.model_shape) - len(arr.shape)):
            arr = np.expand_dims(arr, axis=0)
        return arr

    def load_data(self, astype="float32"):
        img = self.input.resize((self.model_height, self.model_width))
        img = np.array(img).astype(astype)
        self.data = img
        self.match_to_model()
        self.data = tt.from_numpy(self.data).to(self.device)

    def _normalise(self, means, stds, astype, norm):
        assert self.data is not None

        normed_data = self.data
        if norm is not None:
            normed_data /= 255.0

        if self.model_order == "first" and self.model_channels == 3:
            if means is not None:
                for i, m in enumerate(means):
                    normed_data[:, i, :, :] = normed_data[:, i, :, :] - m
            if stds is not None:
                for i, s in enumerate(stds):
                    normed_data[:, i, :, :] = normed_data[:, i, :, :] / s

        # greyscale
        if self.model_order == "first" and self.model_channels == 1:
            if means is not None:
                for i, m in enumerate(means):
                    normed_data[i] = normed_data[i] - m
            if stds is not None:
                for i, s in enumerate(stds):
                    normed_data[i] = normed_data[i] / s

        if self.model_order == "last" and self.model_channels == 3:
            if means is not None:
                for i, m in enumerate(means):
                    normed_data[:, :, i] = normed_data[:, :, i] - m
            if stds is not None:
                for i, s in enumerate(stds):
                    normed_data[:, :, i] = normed_data[:, :, i] / s
        if self.model_order == "last" and self.model_channels == 1:
            pass

        return normed_data

    def unsqueeze(self):
        assert self.data is not None
        out = self.data
        if isinstance(self.data, tt.Tensor):
            for _ in range(len(self.model_shape) - len(self.data.shape)):
                out = tt.unsqueeze(out, dim=0)
        else:
            for _ in range(len(self.model_shape) - len(self.data.shape)):
                print(out.shape)
                out = np.expand_dims(out, axis=0)
        return out

    def generic_image_preprocess(
        self,
        means=None,
        stds=None,
        astype="float32",
        norm: Optional[float] = 1.0,
    ):
        self.load_data(astype=astype)

        if self.mode == "RGB" and self.data is not None:
            self.data = self._normalise(means, stds, astype, norm)
            self.unsqueeze()
        if self.mode == "L":
            self.data = self._normalise(means, stds, astype, norm)

    def __get_shape(self):
        if (self.mode == "tabular" or self.mode == "spectral") and len(
            self.model_shape
        ) == 3:
            return 1, self.model_shape[2], 1, None
        if self.mode in ("RGB", "RGBA", "L") and len(self.model_shape) == 4:
            _, a, b, c = self.model_shape
            if a == 1 or a == 3 or a == 4:
                return b, c, a, "first"
            else:
                return a, b, c, "last"
        if self.mode == "voxel":
            pass
        return None, None, None, None
