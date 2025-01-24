#!/usr/bin/env python

"""generate multiple explanations from a responsibility landscape <pixel_ranking>"""

import os
import numpy as np

import torch as tt
from itertools import combinations

from rex_xai.extraction import Explanation
from rex_xai.distributions import random_coords, Distribution
from rex_xai.logger import logger
from rex_xai._utils import powerset, clause_area, SpatialSearch
from rex_xai.visualisation import save_multi_explanation


class MultiExplanation(Explanation):
    def __init__(self, map, prediction_func, data, args, run_stats):
        super().__init__(map, prediction_func, data, args, run_stats)
        self.explanations = []

    def save(self, path, mask=None, multi=None, multi_style=None, clauses=None):
        if multi_style is None:
            multi_style = self.args.multi_style
        if multi_style == "separate":
            logger.info("saving explanations in multiple different files")
            for i, mask in enumerate(self.explanations):
                if path is not None:
                    name, ext = os.path.splitext(path)
                    path = f"{name}_{i}{ext}"
                super().save(path, mask=mask)
        elif multi_style == "composite":
            logger.info("using composite style to save explanations")
            if clauses is None:
                clauses = range(0, len(self.explanations))
                save_multi_explanation(
                    self.explanations, self.data, self.args, clauses=clauses, path=path
                )
            else:
                for clause in clauses:
                    new_name = None
                    if path is not None:
                        name, ext = os.path.splitext(path)
                        new_name = f"{name}_{clause}{ext}"
                    save_multi_explanation(
                        self.explanations,
                        self.data,
                        self.args,
                        clauses=clause,
                        path=new_name,
                    )

    def extract(self, method=None):
        target_map = self.maps.get(self.data.target.classification)  # type: ignore
        if target_map is not None:
            self.maps = tt.from_numpy(target_map).to(self.data.device)
            self.blank()
            # we start with the global max explanation
            self._Explanation__global()
            if self.final_mask is not None:
                self.explanations.append(self.final_mask)

            self.blank()

            for i in range(0, self.args.spotlights - 1):
                logger.info("spotlight number %d", i + 1)
                self.spotlight_search()
                if self.final_mask is not None:
                    self.explanations.append(self.final_mask)
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
        logger.warn("not yet implemented")
        return
        # for clause in clauses:
        #     logger.info(f"looking at {clause}")
        #     for part in powerset(clause, reverse=False):
        #         logger.debug(f"   examining {part}")
        #         mask = sum([self.explanations[x] for x in part])
        #         mask = mask.to(tt.bool)  # type: ignore
        #         d = tt.where(mask, 0, self.data.data)  # type: ignore
        #         p = self.prediction_func(d)[0]
        #         if p.classification != self.data.target.classification:  # type: ignore
        #             logger.fatal("not yet finished")
        #             exit()
        # original = np.array(self.data.input.resize((224, 224)))
        # mask = mask.detach().cpu().numpy().transpose((1, 2, 0))
        # img = np.where(mask, 0, original)
        # img = Image.fromarray(img, "RGB")
        # img.save(
        #     f"{self.data.target.classification}_to_{p.classification}.png"
        # )
        # return
        # logger.debug(f"    {p}")

    def __random_step_from(self, origin, width, height, step=5):
        c, r = origin
        # flip a coin to move left (0) or right (1)
        c_dir = np.random.randint(0, 2)
        c = c - step if c_dir == 0 else c + step
        if c < 0:
            c = 0
        if c > width:
            c = width

        # flip a coin to move down (0) or up (1)
        r_dir = np.random.randint(0, 2)
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

        return np.unravel_index(origin, (self.data.model_height, self.data.model_width))

    def spotlight_search(self, origin=None):
        if origin is None:
            centre = self.__random_location()
        else:
            centre = origin

        ret, resp = self._Explanation__spatial(  # type: ignore
            centre=centre, expansion_limit=self.args.no_expansions
        )

        steps = 0
        while ret == SpatialSearch.NotFound and steps < self.args.max_spotlight_budget:
            steps += 1
            if self.args.spotlight_objective_function == "none":
                centre = self.__random_location()
                ret, resp = self._Explanation__spatial(  # type: ignore
                    centre=centre, expansion_limit=self.args.no_expansions
                )
            else:
                new_resp = 0.0
                while new_resp < resp:
                    centre = self.__random_step_from(
                        centre,
                        self.data.model_height,
                        self.data.model_width,
                        step=self.args.spotlight_step,
                    )
                    ret, new_resp = self._Explanation__spatial(  # type: ignore
                        centre=centre, expansion_limit=self.args.no_expansions
                    )
                    if ret == SpatialSearch.Found:
                        return
                ret, resp = self._Explanation__spatial(  # type: ignore
                    centre=centre, expansion_limit=self.args.no_expansions
                )
