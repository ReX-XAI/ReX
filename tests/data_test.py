#!/usr/bin/env python

import numpy as np
from rex_xai.input_data import Data


tab = np.arange(0, 1, 999)
voxel = np.random.rand(1, 64, 64, 64)

def test_data():
    data = Data(input=tab, model_shape=[1, 999], device="cpu")
    assert data.model_shape == [1, 999]
    assert data.mode == "spectral"

def test_3D_data():
    data = Data(input=voxel, model_shape=[1, 64, 64, 64], device="cpu")
    assert data.model_shape == [1, 64, 64, 64]
    assert data.mode == "voxel"