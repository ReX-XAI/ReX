#!/usr/bin/env python

import numpy as np
from rex_xai.input_data import Data


tab = np.arange(0, 1, 999)

def test_data():
    data = Data(input=tab, model_shape=[1, 999], device="cpu")
    assert data.model_shape == [1, 999]
    assert data.mode == "spectral"
