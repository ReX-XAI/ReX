#!/usr/bin/env python
from __future__ import annotations

"""generate multiple explanations from a responsibility landscape <pixel_ranking>"""

import os
import re
from itertools import combinations
from typing import List

import numpy as np
import torch as tt

from rex_xai.explanation.explanation import Explanation
from rex_xai.mutants.distributions import Distribution, random_coords
from rex_xai.output.visualisation import (
    plot_image_grid,
    save_image,
    save_multi_explanation,
)
from rex_xai.utils._utils import (
    SpatialSearch,
    clause_area,
    powerset,
)
from rex_xai.utils.logger import logger


class MultiExplanation(Explanation):
    def __init__(self, maps, prediction_func, data, args, run_stats):
        super().__init__(maps, prediction_func, data, args, run_stats)
        self.explanations = []
        self.explanation_confidences = []

    def __repr__(self) -> str:
        pred_func = repr(self.prediction_func)
        match_func_name = re.search(r"(<function .+) at", pred_func)
        if match_func_name:
            pred_func = match_func_name.group(1) + " >"

        run_stats = {k: round(v, 5) for k, v in self.run_stats.items()}

        exp_text = (
            "MultiExplanation:"
            + f"\n\tCausalArgs: {type(self.args)}"
            + f"\n\tData: {self.data}"
            + f"\n\tprediction function: {pred_func}"
            + f"\n\tResponsibilityMaps: {self.maps}"
            + f"\n\trun statistics: {run_stats} (5 dp)"
        )

        if len(self.explanations) == 0:
            return (
                exp_text
                + f"\n\texplanations: {self.explanations}"
                + f"\n\texplanation confidences: {self.explanation_confidences}"
            )
        else:
            return (
                exp_text
                + f"\n\texplanations: {len(self.explanations)} explanations of {type(self.explanations[0])} and shape {self.explanations[0].shape}"
                + f"\n\texplanation confidences: {[round(x, ndigits=5) for x in self.explanation_confidences]} (5 dp)"
            )

    def save(
        self,
        path: str,
        mask=None,
        clauses: List[int] | None = None,
        multi_style: str | None = None,
    ):
        if multi_style is None:
            multi_style = self.args.multi_style
        if multi_style == "separate":
            logger.info("saving explanations in multiple different files")
            for i, mask in enumerate(self.explanations):
                name, ext = os.path.splitext(path)
                exp_path = f"{name}_{i}{ext}"
                super().save(exp_path, mask=mask)
        if multi_style == "composite":
            logger.info("using composite style to save explanations")
            if clauses is None:
                clause = range(0, len(self.explanations))
                save_multi_explanation(
                    self.explanations, self.data, self.args, clause=clause, path=path
                )
            else:
                name, ext = os.path.splitext(path)
                new_name = f"{name}_{clauses}{ext}"
                save_multi_explanation(
                    self.explanations,
                    self.data,
                    self.args,
                    clause=clauses,
                    path=new_name,
                )

    def show(self, multi_style=None, clauses=None):  # type: ignore
        if multi_style is None:
            multi_style = self.args.multi_style
        outs = []

        for mask in self.explanations:
            out = save_image(mask, self.data, self.args, path=None)
            outs.append(out)

        if multi_style == "separate":
            for mask in self.explanations:
                out = save_image(mask, self.data, self.args, path=None)
                outs.append(out)

        elif multi_style == "composite":
            if clauses is None:
                clause = tuple([i for i in range(len(self.explanations))])
                out = save_multi_explanation(
                    self.explanations, self.data, self.args, clause=clause, path=None
                )
                outs.append(out)
            else:
                for clause in clauses:
                    out = save_multi_explanation(
                        self.explanations,
                        self.data,
                        self.args,
                        clause=clause,
                        path=None,
                    )
                    outs.append(out)

        if len(outs) > 1:
            plot_image_grid(outs)
        else:
            return outs[0]

    def extract(self):
        self.blank()
        # we start with the global max explanation
        logger.info("spotlight number 1 (global max)")
        conf = self._Explanation__global()  # type: ignore
        if self.sufficiency_mask is not None:
            self.explanations.append(self.sufficiency_mask)
            self.explanation_confidences.append(conf)
        self.blank()

        for i in range(0, self.args.spotlights - 1):
            logger.info("spotlight number %d", i + 2)
            conf = self.spotlight_search()
            if self.sufficiency_mask is not None:
                self.explanations.append(self.sufficiency_mask)
                self.explanation_confidences.append(conf)
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

        if self.args.distribution == Distribution.Uniform:
            origin = random_coords(
                self.args.distribution,
                self.data.model_width * self.data.model_height,
                1,
            )
        else:
            origin = random_coords(
                self.args.distribution,
                self.data.model_width * self.data.model_height,
                1,
                self.args.distribution_args,
                self.data.model_height,
                self.data.model_width,
            )

        return np.unravel_index(origin, (self.data.model_height, self.data.model_width))  # type: ignore

    def spotlight_search(self, origin=None):
        if origin is None:
            centre = self.__random_location()
        else:
            centre = origin

        ret, resp, conf = self._Explanation__spatial(  # type: ignore
            centre=centre, expansion_limit=self.args.no_expansions
        )

        steps = 0
        while ret == SpatialSearch.NotFound and steps < self.args.max_spotlight_budget:
            if self.args.spotlight_objective_function == "none":
                centre = self.__random_location()
                ret, resp, conf = self._Explanation__spatial(  # type: ignore
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
                    ret, new_resp, conf = self._Explanation__spatial(  # type: ignore
                        centre=centre, expansion_limit=self.args.no_expansions
                    )
                    if ret == SpatialSearch.Found:
                        return conf
                ret, resp, conf = self._Explanation__spatial(  # type: ignore
                    centre=centre, expansion_limit=self.args.no_expansions
                )
            steps += 1
        return conf
