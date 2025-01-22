#!/usr/bin/env python

import numpy as np
from rex_xai.input_data import Data
from rex_xai.explanation import load_and_preprocess_data


tab = np.arange(0, 1, 999)


def test_data():
    data = Data(input=tab, model_shape=[1, 999], device="cpu")
    assert data.model_shape == [1, 999]
    assert data.mode == "spectral"


def test_set_mask_value(args_custom, model_shape, cpu_device, caplog):
    data = load_and_preprocess_data(model_shape, cpu_device, args_custom)
    data.set_mask_value('spectral')
    
    assert caplog.records[0].msg == "Mask value 'spectral' can only be used if mode is also 'spectral', using default mask value 0 instead"
    assert data.mask_value == 0
