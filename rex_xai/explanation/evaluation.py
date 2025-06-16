#!/usr/bin/env python
from typing import Tuple

import numpy as np
import torch as tt
from scipy.integrate import simpson
from scipy.signal import periodogram
from scipy.stats import entropy

from rex_xai.explanation.explanation import Explanation
from rex_xai.mutants.mutant import _apply_to_data
from rex_xai.utils._utils import get_map_locations, set_boolean_mask_value, xlogx


class Evaluation:
    # TODO does this need to be an object? Probably not...
    # TODO consider inheritance from Explanation object
    def __init__(self, explanation: Explanation) -> None:
        self.explanation = explanation

    def ratio(self) -> float:
        """Returns percentage of data required for sufficient explanation"""
        final_mask = self.explanation.sufficiency_mask
        if isinstance(final_mask, tt.Tensor):
            final_mask = final_mask.detach().cpu().numpy()

        try:
            return (
                tt.count_nonzero(final_mask)  # type: ignore
                / final_mask.size  # type: ignore
            ).item()
        except TypeError:
            return (
                np.count_nonzero(final_mask)  # type: ignore
                / final_mask.size  # type: ignore
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
        flat_map = self.explanation.target_map.ravel()
        return entropy(flat_map, base=2)

    def insertion_deletion_curve(self, prediction_func, normalise=False):
        assert self.explanation.data.target is not None
        assert self.explanation.data.target.confidence is not None

        step = self.explanation.args.insertion_step
        ranking = get_map_locations(map=self.explanation.target_map)

        insertion_curve = np.zeros(len(ranking) // step)
        deletion_curve = np.zeros(len(ranking) // step)
        # insertion_curve = []
        # deletion_curve = []

        assert self.explanation.data.data is not None
        insertion_mask = tt.zeros(
            self.explanation.data.data.squeeze(0).shape, dtype=tt.bool
        ).to(self.explanation.data.device)
        deletion_mask = tt.ones(
            self.explanation.data.data.squeeze(0).shape, dtype=tt.bool
        ).to(self.explanation.data.device)

        model_shape = self.explanation.data.model_shape
        model_shape[0] = self.explanation.args.batch_size
        im = tt.empty(model_shape, dtype=tt.float32)
        dm = tt.empty(model_shape, dtype=tt.float32)

        j = 0
        for i in range(0, len(ranking), step):
            chunk = ranking[i : i + step]
            for _, loc in chunk:
                set_boolean_mask_value(
                    insertion_mask,
                    self.explanation.data.mode,
                    self.explanation.data.model_order,
                    loc,
                )
                set_boolean_mask_value(
                    deletion_mask,
                    self.explanation.data.mode,
                    self.explanation.data.model_order,
                    loc,
                    val=False,
                )
            im[j] = _apply_to_data(insertion_mask, self.explanation.data).squeeze(0)
            dm[j] = _apply_to_data(deletion_mask, self.explanation.data).squeeze(0)
            j += 1

            if j == self.explanation.args.batch_size:
                self.__batch(im, dm, prediction_func, insertion_curve, deletion_curve)
                im = tt.empty(
                    (self.explanation.args.batch_size, 3, 224, 224), dtype=tt.float32
                )
                dm = tt.empty(
                    (self.explanation.args.batch_size, 3, 224, 224), dtype=tt.float32
                )
                j = 0

        # TODO check this this is correct
        self.__batch(
            im[:j, :, :, :],
            dm[:j, :, :, :],
            prediction_func,
            insertion_curve,
            deletion_curve,
        )

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
        insertion_curve,
        deletion_curve,
    ):
        assert self.explanation.data.target is not None
        ip = prediction_func(im.to(self.explanation.data.device), raw=True)
        dp = prediction_func(dm.to(self.explanation.data.device), raw=True)
        for p in range(0, ip.shape[0]):
            insertion_curve[p] = ip[
                p, self.explanation.data.target.classification
            ].item()
            deletion_curve[p] = dp[
                p, self.explanation.data.target.classification
            ].item()  # type: ignore
