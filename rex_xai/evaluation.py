#!/usr/bin/env python
from typing import Tuple
import numpy as np
import torch as tt
from scipy.integrate import simpson
from scipy.signal import periodogram
from skimage.measure import shannon_entropy

from rex_xai.extraction import Explanation
from rex_xai._utils import get_map_locations
from rex_xai.mutant import _apply_to_data
from rex_xai._utils import set_boolean_mask_value, xlogx


class Evaluation:
    # TODO does this need to be an object? Probably not...
    # TODO consider inheritance from Explanation object
    def __init__(self, explanation: Explanation) -> None:
        self.explanation = explanation

    def ratio(self) -> float:
        """Returns percentage of data required for sufficient explanation"""
        if (
            self.explanation.explanation is None
            or self.explanation.data.model_channels is None
        ):
            raise ValueError("Invalid Explanation object")

        try:
            final_mask = self.explanation.final_mask.squeeze().item()  # type: ignore
        except Exception:
            final_mask = self.explanation.final_mask

        try:
            return (
                tt.count_nonzero(final_mask)  # type: ignore
                / final_mask.size  # type: ignore
                / self.explanation.data.model_channels
            ).item()
        except TypeError:
            return (
                np.count_nonzero(final_mask)  # type: ignore
                / final_mask.size  # type: ignore
                / self.explanation.data.model_channels
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

    def entropy_loss(self):
        img = np.array(self.explanation.data.input)
        assert self.explanation.explanation is not None
        exp = shannon_entropy(self.explanation.explanation.detach().cpu().numpy())

        return shannon_entropy(img), exp

    def insertion_deletion_curve(self, prediction_func, normalise=False):
        insertion_curve = []
        deletion_curve = []

        assert self.explanation.data.target is not None
        assert self.explanation.data.target.confidence is not None

        assert self.explanation.data.data is not None
        insertion_mask = tt.zeros(
            self.explanation.data.data.squeeze(0).shape, dtype=tt.bool
        ).to(self.explanation.data.device)
        deletion_mask = tt.ones(
            self.explanation.data.data.squeeze(0).shape, dtype=tt.bool
        ).to(self.explanation.data.device)
        im = []
        dm = []

        step = self.explanation.args.insertion_step
        ranking = get_map_locations(map=self.explanation.target_map)
        iters = len(ranking) // step

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
            im.append(
                _apply_to_data(insertion_mask, self.explanation.data, 0).squeeze(0)
            )
            dm.append(
                _apply_to_data(deletion_mask, self.explanation.data, 0).squeeze(0)
            )

            if len(im) == self.explanation.args.batch_size:
                self.__batch(im, dm, prediction_func, insertion_curve, deletion_curve)
                im = []
                dm = []

        if im != [] and dm != []:
            self.__batch(im, dm, prediction_func, insertion_curve, deletion_curve)

        i_auc = simpson(insertion_curve, dx=step)
        d_auc = simpson(deletion_curve, dx=step)

        if normalise:
            const = self.explanation.data.target.confidence * iters * step
            i_auc /= const
            d_auc /= const

        return i_auc, d_auc

    # # def sensitivity(self):
    #     #     pass

    # # def infidelity(self):
    #     #     pass

    def __batch(self, im, dm, prediction_func, insertion_curve, deletion_curve):
        assert self.explanation.data.target is not None
        ip = prediction_func(tt.stack(im).to(self.explanation.data.device), raw=True)
        dp = prediction_func(tt.stack(dm).to(self.explanation.data.device), raw=True)
        for p in range(0, ip.shape[0]):
            insertion_curve.append(
                ip[p, self.explanation.data.target.classification].item()
            )  # type: ignore
            deletion_curve.append(
                dp[p, self.explanation.data.target.classification].item()
            )  # type: ignore
