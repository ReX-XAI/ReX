#!/usr/bin/env python
import unittest
import numpy as np
from rex_ai.input_data import Data


tab = np.arange(0, 1, 999)


class TestData(unittest.TestCase):
    data = Data(tab, [1, 999])

    self.assertEqual(data.model_shape, [1, 999])
    self.assertEqual(data.mode, "tabular")
