#!/usr/bin/env python

"""generate multiple explanations from a responsibility landscape <pixel_ranking>"""

import numpy as np

from numpy.typing import NDArray
from rex_xai.input_data import Data
from rex_xai.prediction import Prediction
from rex_xai.config import CausalArgs
from rex_xai.distributions import random_coords, Distribution


class MultiExplanation:
    def __init__(self, maps, data: Data, target: Prediction):
        self.maps = maps
        self.target = target
        self.data = data

    def __dice(self, d1: NDArray, d2: NDArray):
        """calculates dice coefficient between two numpy arrays of the same dimensions"""
        d_sum = d1.sum() + d2.sum()
        if d_sum == 0:
            return 0
        intersection = np.logical_and(d1, d2)
        return 2.0 * intersection.sum() / d_sum

    def separate_by(self, dice_coefficient: float):
        pass

    def spotlight_search(self, args: CausalArgs, coords=None):
        if coords is None:
            origin = random_coords(
                Distribution.Uniform,
                [self.data.model_width * self.data.model_height],
            )  # type: ignore
            return np.unravel_index(
                origin, (self.data.model_height, self.data.model_width)
            )  # type: ignore
        # pass


# from itertools import combinations
# import numpy as np
# from numpy.typing import NDArray
# from numpy.random import randn
# from numba import njit
# from tqdm import trange

# from rex_xai.config import CausalArgs

# from rex_xai.logger import logger

# # from rex_xai.ranking import linear_search, neighbours, spatial_search


# @njit
# def dice(im1, im2):
#     """calculates dice coefficient between two numpy arrays of the same dimensions"""
#     im_sum = im1.sum() + im2.sum()
#     if im_sum == 0:
#         return 0
#     intersection = np.logical_and(im1, im2)
#     return 2.0 * intersection.sum() / im_sum


# def extract(explanations):
#     """extract multiple explanations from a list of explanations"""
#     out = []
#     size = len(explanations)
#     areas = np.zeros(size)

#     combs = list(combinations(np.arange(0, size, 1), 2))
#     for comb in combs:
#         im1 = explanations[comb[0]]
#         im2 = explanations[comb[1]]
#         out.append((comb, dice(im1, im2)))
#         a1 = np.count_nonzero(im1)
#         a2 = np.count_nonzero(im2)
#         if areas[comb[0]] == 0.0:
#             areas[comb[0]] = a1
#         if areas[comb[1]] == 0.0:
#             areas[comb[1]] = a2

#     mat = np.zeros(size * size).reshape((size, size))
#     for x, y in out:
#         if y > 0.0:
#             if areas[x[0]] < areas[x[1]]:
#                 mat[x[0], x[1]] += 1
#             else:
#                 mat[x[1], x[0]] += 1

#     results = mat.sum(axis=0)

#     for v in np.argsort(results)[::-1]:
#         if np.count_nonzero(mat) == 0:
#             break
#         mat[v, :] = 0
#         mat[:, v] = 0
#         r = mat.sum(axis=0)
#         results -= r

#     final = np.where(results == 0)
#     return final, len(final[0]), areas


# def overlap(exp1, exp2):
#     """check overlap between two numpy arrays"""
#     gt = len(np.where(exp1 + exp2 > 0.0)[0])
#     ov = len(np.where(exp1 + exp2 == 2.0)[0])

#     return ov / gt


# def random_steps(r, c, step_size, lim_r, lim_c):
#     """takes random steps within a landscape"""
#     new_r = int(r + randn(1) * step_size)
#     while new_r < 0 or new_r > lim_r:
#         new_r = int(r + randn(1) * step_size)
#     new_c = int(c + randn(1) * step_size)
#     while new_c < 0 or new_c > lim_c:
#         new_c = int(c + randn(1) * step_size)

#     return new_r, new_c


# def multi_spotlight(
#     prediction_func,
#     args: CausalArgs,
#     pixel_ranking,
# ):
#     """launches multiple spotlight searches over a responsibility landscape"""
#     results = []
#     logger.info("find global maximum first")
#     exp1 = linear_search(
#         args.img_array,
#         prediction_func,
#         args.target,
#         pixel_ranking,
#         args.mask_value,
#         args.chunk_size,
#     )
#     results.append(exp1)
#     logger.info("starting spotlight search")
#     for _ in trange(args.spotlights - 1, disable=True):
#         results.append(
#             spotlight_search(
#                 prediction_func,
#                 args,
#                 pixel_ranking,
#             )
#         )
#     results = list(filter(lambda mat: mat is not None and np.sum(mat) > 0, results))
#     return results


# def spotlight_search(
#     prediction_func,
#     args: CausalArgs,
#     pixel_ranking: NDArray[np.float32],
#     r=None,
#     c=None,
#     total_steps_remaining=20,
# ):
#     """performs a spotlight search over a responsibility landscape"""
#     # if args.shape is None:
#     #     args.shape = Shape(args.img_array.shape)

#     if r is None:
#         r = np.random.randint(args.spatial_radius // 2, args.shape.width)
#     if c is None:
#         c = np.random.randint(args.spatial_radius // 2, args.shape.height)

#     # TODO this should be yet another hyperparameter
#     while total_steps_remaining > 0:
#         explanation = spatial_search(
#             prediction_func,
#             args,
#             pixel_ranking,
#             r,
#             c,
#         )
#         if explanation is None:
#             local = args.spotlight_objective_function(
#                 neighbours(
#                     args.shape,
#                     args.spatial_radius,
#                     r,
#                     c,
#                     pixel_ranking,
#                     0.0,
#                 )
#             )
#             attempts = args.spotlight_step * 2
#             new_r, new_c = random_steps(
#                 r, c, args.spotlight_step, args.shape.width, args.shape.height
#             )
#             near = args.spotlight_objective_function(
#                 neighbours(
#                     args.shape,
#                     args.spatial_radius,
#                     new_r,
#                     new_c,
#                     pixel_ranking,
#                     0.0,
#                 )
#             )
#             while local < near and attempts > 0:
#                 near = args.spotlight_objective_function(
#                     neighbours(
#                         args.shape,
#                         args.spatial_radius,
#                         new_r,
#                         new_c,
#                         pixel_ranking,
#                         0.0,
#                     )
#                 )
#                 new_r, new_c = random_steps(
#                     r,
#                     c,
#                     args.spotlight_step,
#                     args.shape.width,
#                     args.shape.height,
#                 )
#                 attempts -= 1
#             if attempts == 0:
#                 r, c = (
#                     np.random.randint(args.spatial_radius // 2, args.shape.width),
#                     np.random.randint(args.spatial_radius // 2, args.shape.height),
#                 )
#             else:
#                 r = new_r
#                 c = new_c
#         else:
#             return explanation

#         total_steps_remaining -= 1
#     return None
