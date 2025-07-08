#!/usr/bin/env python
from typing import Tuple

import numpy as np
import torch as tt
from scipy.integrate import simpson
from scipy.signal import periodogram
from scipy.stats import entropy

from rex_xai.explanation.explanation import Explanation
from rex_xai.mutants.mutant import _apply_to_data
from rex_xai.utils._utils import (
    get_map_locations,
    set_boolean_mask_value,
    try_detach,
    xlogx,
)


class Evaluation:
    # TODO does this need to be an object? Probably not...
    # TODO consider inheritance from Explanation object
    def __init__(self, explanation: Explanation) -> None:
        self.explanation = explanation

    def ratio(self) -> float:
        """Returns percentage of data required for sufficient explanation"""
        if self.explanation.necessity_mask is not None:
            mask = try_detach(self.explanation.necessity_mask)
        else:
            mask = try_detach(self.explanation.sufficiency_mask)

        try:
            return (
                tt.count_nonzero(mask)  # type: ignore
                / mask.size  # type: ignore
            ).item()
        except TypeError:
            return (
                np.count_nonzero(mask)  # type: ignore
                / mask.size  # type: ignore
            )

    def spectral_entropy(self) -> Tuple[float, float]:
        """
        This code is a simplified version of
        https://github.com/raphaelvallat/antropy/blob/master/src/antropy/entropy.py
        """
        _, psd = periodogram(self.explanation.target_map)
        psd_norm = psd / psd.sum()
        ent = -np.sum(xlogx(psd_norm))
        if len(psd_norm.shape) == 2:
            max_ent = np.log2(len(psd_norm[0]))
        else:
            max_ent = np.log2(len(psd_norm))
        return ent, max_ent

    def responsibility_entropy(self):
        flat_map = try_detach(self.explanation.target_map).ravel()
        uniform = np.ones(flat_map.shape)

        return entropy(flat_map, uniform, base=2)

    def robustness(self, lower=None, upper=None, repeats=2):
        if lower is None:
            lower = tt.min(self.explanation.data.data).item()  # type: ignore
        if upper is None:
            upper = tt.max(self.explanation.data.data).item()  # type: ignore

        mask_shape = self.explanation.sufficiency_mask.shape  # type: ignore

        robustness_shape = (self.explanation.args.batch_size,) + mask_shape

        good = 0
        bad = 0

        if self.explanation.necessity_mask is not None:
            mask = self.explanation.necessity_mask
        else:
            mask = self.explanation.sufficiency_mask

        for _ in (0, repeats):
            test_tensor = (
                tt.FloatTensor(*robustness_shape)
                .uniform_(lower, upper)
                .to(self.explanation.data.device)
            )

            assert mask is not None
            for j in range(0, self.explanation.args.batch_size):
                test_tensor[j] = tt.where(
                    mask,
                    self.explanation.data.data,  # type: ignore
                    test_tensor[j],
                )

            result = self.explanation.prediction_func(test_tensor)

            for p in result:
                if p.classification == self.explanation.data.target.classification:  # type: ignore
                    good += 1
                else:
                    bad += 1

        return (good, bad)

    def insertion_deletion_curve(self, prediction_func, normalise=False):
        # keep pyright happy...
        assert self.explanation.data.target is not None
        assert self.explanation.data.data is not None
        assert self.explanation.data.target.confidence is not None

        step = self.explanation.args.insertion_step
        ranking = get_map_locations(map=self.explanation.target_map)

        # initialise empty insertion and delection curves arrays
        insertion_curve = np.zeros(len(ranking) // step)
        deletion_curve = np.zeros(len(ranking) // step)

        # blank insertion mask
        insertion_mask = tt.zeros(
            self.explanation.data.data.squeeze(0).shape, dtype=tt.bool
        ).to(self.explanation.data.device)
        # dense deletion mask
        deletion_mask = tt.ones(
            self.explanation.data.data.squeeze(0).shape, dtype=tt.bool
        ).to(self.explanation.data.device)

        # edit the model shape to include a batch number, if necessary and make it a tuple
        model_shape = self.explanation.data.model_shape
        model_shape[0] = self.explanation.args.batch_size
        model_shape = tuple(model_shape)

        # initialise tensors for insertion and deletion mutants
        insertion_mutants = tt.empty(model_shape, dtype=tt.float32)
        deletion_mutants = tt.empty(model_shape, dtype=tt.float32)

        pointer = 0
        j = 0
        for i in range(0, len(ranking), step):
            chunk = ranking[i : i + step]
            for _, loc in chunk:
                # set insertion_mask values to true
                set_boolean_mask_value(
                    insertion_mask,
                    self.explanation.data.mode,
                    self.explanation.data.model_order,
                    loc,
                )
                # set deletion mask values to false
                set_boolean_mask_value(
                    deletion_mask,
                    self.explanation.data.mode,
                    self.explanation.data.model_order,
                    loc,
                    val=False,
                )
            insertion_mutants[j] = _apply_to_data(
                insertion_mask, self.explanation.data
            ).squeeze(0)
            deletion_mutants[j] = _apply_to_data(
                deletion_mask, self.explanation.data
            ).squeeze(0)
            j += 1

            if j >= self.explanation.args.batch_size - 1:
                insertion_update, deletion_update = self.__batch(
                    insertion_mutants, deletion_mutants, prediction_func
                )
                insertion_curve[pointer : pointer + len(insertion_update)] = (
                    insertion_update
                )
                deletion_curve[pointer : pointer + len(insertion_update)] = (
                    deletion_update
                )

                pointer += j
                j = 0
                insertion_mutants = tt.empty(model_shape, dtype=tt.float32)
                deletion_mutants = tt.empty(model_shape, dtype=tt.float32)

        remaining = len(insertion_curve) - pointer
        insertion_update, deletion_update = self.__batch(
            insertion_mutants, deletion_mutants, prediction_func
        )
        insertion_curve[pointer : pointer + remaining] = insertion_update[:remaining]
        deletion_curve[pointer : pointer + remaining] = deletion_update[:remaining]

        i_auc = simpson(insertion_curve, dx=step)
        d_auc = simpson(deletion_curve, dx=step)

        if normalise:
            const = self.explanation.data.target.confidence * len(ranking)
            i_auc /= const
            d_auc /= const

        return i_auc, d_auc

    def __batch(
        self,
        im,
        dm,
        prediction_func,
    ):
        ip = prediction_func(im.to(self.explanation.data.device), raw=True)
        ipe = [
            ip[p, self.explanation.data.target.classification].item()  # type: ignore
            for p in range(0, ip.shape[0])
        ]
        dp = prediction_func(dm.to(self.explanation.data.device), raw=True)
        dpe = [
            dp[p, self.explanation.data.target.classification].item()  # type: ignore
            for p in range(0, dm.shape[0])
        ]
        return ipe, dpe
