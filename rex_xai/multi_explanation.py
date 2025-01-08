#!/usr/bin/env python

"""generate multiple explanations from a responsibility landscape <pixel_ranking>"""

import numpy as np
import random
import torch as tt
from itertools import combinations
from PIL import Image

from rex_xai.extraction import Explanation
from rex_xai.distributions import random_coords, Distribution
from rex_xai.logger import logger
from rex_xai._utils import powerset, clause_area


class MultiExplanation(Explanation):
    def __init__(self, map, prediction_func, data, args, run_stats):
        super().__init__(map, prediction_func, data, args, run_stats)
        self.explanations = []

    def extract(self, method=None):
        target_map = self.maps.get(self.data.target.classification)
        if target_map is not None:
            self.maps = tt.from_numpy(target_map).to(self.data.device)
            self.blank()
            for i in range(0, self.args.spotlights):
                logger.info("spotlight number %d", i + 1)
                self.spotlight_search()
                self.explanations.append(self.explanation)
                self.blank()
            logger.info(
                "ReX has found a total of %d explanations via spotlight search",
                len(self.explanations),
            )

    def __dice(self, d1, d2):
        """calculates dice coefficient between two numpy arrays of the same dimensions"""
        d_sum = d1.sum() + d2.sum()
        if d_sum == 0:
            return 0
        intersection = tt.logical_and(d1, d2)
        return np.abs((2.0 * intersection.sum() / d_sum).item())

    def separate_by(self, dice_coefficient: float, reverse=True):
        exps = []
        sizes = dict()

        for i, exp in enumerate(self.explanations):
            size = tt.count_nonzero(exp)
            if size > 0:
                exps.append(i)
                sizes[i] = size

        clause_len = 0
        clauses = []

        perms = combinations(exps, 2)
        bad_pairs = set()
        for perm in perms:
            left, right = perm
            if (
                self.__dice(self.explanations[left], self.explanations[right])
                > dice_coefficient
            ):
                bad_pairs.add(perm)

        for s in powerset(exps, reverse=reverse):
            found = True
            for bp in bad_pairs:
                if bp[0] in s and bp[1] in s:
                    found = False
                    break
            if found:
                if len(s) >= clause_len:
                    clause_len = len(s)
                    clauses.append(s)
                else:
                    break

        clauses = sorted(clauses, key=lambda x: clause_area(x, sizes))
        return clauses

    def contrastive(self, clauses):
        for clause in clauses:
            logger.info(f"looking at {clause}")
            for part in powerset(clause, reverse=False):
                logger.debug(f"   examining {part}")
                mask = sum([self.explanations[x] for x in part])
                mask = mask.to(tt.bool)  # type: ignore
                d = tt.where(mask, 0, self.data.data)  # type: ignore
                p = self.prediction_func(d)[0]
                if p.classification != self.data.target.classification:
                    original = np.array(self.data.input.resize((224, 224)))
                    mask = mask.detach().cpu().numpy().transpose((1, 2, 0))
                    img = np.where(mask, 0, original)
                    img = Image.fromarray(img, "RGB")
                    img.save(
                        f"{self.data.target.classification}_to_{p.classification}.png"
                    )
                    return
                logger.debug(f"    {p}")

    def __random_step_from(self, origin, width, height, step=5):
        c, r = origin
        # flip a coin to move left (0) or right (1)
        c_dir = random.randint(0, 1)
        c = c - step if c_dir == 0 else c + step
        if c < 0:
            c = 0
        if c > width:
            c = width

        # flip a coin to move down (0) or up (1)
        r_dir = random.randint(0, 1)
        r = r - step if r_dir == 0 else r + step
        if r < 0:
            r = 0
        if r > height:
            r = height
        logger.debug(f"trying new location: moving from {origin} to {(c, r)}")
        return (c, r)

    def __random_location(self):
        assert self.data.model_width is not None
        assert self.data.model_height is not None
        origin = random_coords(
            Distribution.Uniform,
            self.data.model_width * self.data.model_height,
        )

        return np.unravel_index(
            origin, (self.data.model_height, self.data.model_width)
        )

    def spotlight_search(self, origin=None):
        if origin is None:
            centre = self.__random_location()
        else:
            centre = origin

        r = self._Explanation__spatial(centre=centre, expansion_limit=4)

        while r == -1:
            if self.args.spotlight_objective_function == "none":
                centre = self.__random_location()
            else:
                centre = self.__random_step_from(
                    centre,
                    self.data.model_height,
                    self.data.model_width,
                    step=self.args.spotlight_step,
                )
            r = self._Explanation__spatial(centre=centre, expansion_limit=4)
