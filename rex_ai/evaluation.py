#!/usr/bin/env python
import numpy as np
import torch as tt

from rex_ai.logger import logger
from rex_ai.extraction import Explanation
from rex_ai._utils import get_map_locations
from rex_ai.mutant import _apply_to_data
from rex_ai._utils import set_boolean_mask_value
from scipy.integrate import simpson
from skimage.measure import shannon_entropy


class Evaluation:
    # TODO does this need to be an object? Probably not...
    def __init__(self, explanation: Explanation) -> None:
        self.explanation = explanation

    def ratio(self) -> float:
        """Returns percentage of data required for sufficient explanation"""
        assert self.explanation.explanation is not None
        try:
            return (
                tt.count_nonzero(self.explanation.explanation.squeeze()).item()
                / self.explanation.map.size
                / self.explanation.data.model_channels
            )
        except TypeError:
            return (
                np.count_nonzero(self.explanation.explanation)
                / self.explanation.map.size
                / self.explanation.data.model_channels
            )

    def entropy_loss(self):
        if self.explanation.data.mode in ("RGB", "L"):
            img = np.array(self.explanation.data.input)
            assert self.explanation.explanation is not None
            exp = shannon_entropy(
                self.explanation.explanation.detach().cpu().numpy()
            )

            return shannon_entropy(img), exp
        else:
            logger.warning(
                "entropy loss is not yet defined on this type of data"
            )

    def insertion_deletion_curve(self, prediction_func):
        insertion_curve = []
        deletion_curve = []

        assert self.explanation.args.target is not None
        assert self.explanation.args.target.confidence is not None

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
        ranking = get_map_locations(map=self.explanation.map)
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
                _apply_to_data(
                    insertion_mask, self.explanation.data, 0
                ).squeeze(0)
            )
            dm.append(
                _apply_to_data(deletion_mask, self.explanation.data, 0).squeeze(
                    0
                )
            )

            if len(im) == self.explanation.args.batch:
                self.__batch(
                    im, dm, prediction_func, insertion_curve, deletion_curve
                )
                im = []
                dm = []

        if im != [] and dm != []:
            self.__batch(
                im, dm, prediction_func, insertion_curve, deletion_curve
            )

        i_auc = simpson(insertion_curve, dx=step) / (
            (self.explanation.args.target.confidence) * iters * step
        )  # type: ignore
        d_auc = simpson(deletion_curve, dx=step) / (
            (self.explanation.args.target.confidence) * iters * step
        )  # type: ignore

        return i_auc, d_auc

    # # def sensitivity(self):
    #     #     pass

    # # def infidelity(self):
    #     #     pass

    def __batch(self, im, dm, prediction_func, insertion_curve, deletion_curve):
        assert self.explanation.args.target is not None
        ip = prediction_func(
            tt.stack(im).to(self.explanation.data.device), raw=True
        )
        dp = prediction_func(
            tt.stack(dm).to(self.explanation.data.device), raw=True
        )
        for p in range(0, ip.shape[0]):
            insertion_curve.append(
                ip[p, self.explanation.args.target.classification].item()
            )  # type: ignore
            deletion_curve.append(
                dp[p, self.explanation.args.target.classification].item()
            )  # type: ignore
