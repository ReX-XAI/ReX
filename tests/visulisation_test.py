#!/usr/bin/env python
import os

import numpy as np

from rex_xai.visualisation import voxel_plot, save_image
from rex_xai.config import CausalArgs
from rex_xai.input_data import Data

import torch as tt

voxel = np.random.rand(1, 64, 64, 64)
data_3d = Data(input=voxel, model_shape=[1, 64, 64, 64], device="cpu")
data_3d.mode = "voxel"
data_3d.data = voxel

def test_save_image_3d():
    # Explanation mask for the voxel data - random values of 0s and 1s
    explanation = tt.zeros(
        (1, 64, 64, 64), dtype=tt.bool, device="cpu"
        )
    explanation[0, 32:64, 32:64, 32:64] = 1
    args = CausalArgs()
    args.mode = "voxel"
    args.output = "test.png"
    save_image(explanation, data_3d, args, path=args.output)
    assert os.path.exists(args.output)
    assert os.path.getsize(args.output) > 0

    os.remove(args.output)