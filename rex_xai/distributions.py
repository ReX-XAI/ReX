"""distributions module"""

from typing import Optional, Tuple
from enum import Enum
import numpy as np
# import torch as tt

from numba import njit
from scipy.stats import binom, betabinom
<<<<<<< HEAD:rex_xai/distributions.py
from rex_xai.logger import logger
=======
>>>>>>> dev:rex_ai/distributions.py

Distribution = Enum(
    "Distribution", ["Binomial", "Uniform", "BetaBinomial", "Adaptive"]
)


@njit
def _blend(dist, alpha, base):
    pmf = np.array([base.pmf(x) for x in range(0, len(dist))])
    blend = ((1.0 - alpha) * pmf) + (alpha * dist)
    blend /= np.sum(blend)
    return blend


def _2d_adaptive(
    map, args: Tuple[int, int, int, int], alpha=0.0, base=None
) -> int:
    # if the map exists and is not 0.0 everywhere...
    if map is not None and np.max(map) > 0.0:
        s = map[args[0] : args[1], args[2] : args[3]]
        sf = np.ndarray.flatten(s)
        # sf = np.max(sf) - sf
        sf /= np.sum(sf)

        # base = betabinom(0, len(sf), 1.1, 1.1)
        # if base is not None:
        #     print('blending')
        #     sf = _blend(alpha, base)
        pos = np.random.choice(np.arange(0, len(sf)), p=sf)
        return pos

    # if the map is empty or doesn't exist, return uniform
    return np.random.randint(1, (args[1] - args[0]) * (args[3] - args[2]))


def str2distribution(d: str) -> Distribution:
    """converts string into Distribution enum"""
    if d == "binom":
        return Distribution.Binomial
    if d == "uniform":
        return Distribution.Uniform
    if d == "betabinom":
        return Distribution.BetaBinomial
    if d == "adaptive":
        return Distribution.Adaptive
    return Distribution.Uniform


def random_coords(d: Optional[Distribution], *args, map=None) -> int:
    """generates random coordinates given a distribution and args"""
    if d == Distribution.Adaptive:
        return _2d_adaptive(map, args[0])
        # h, w = _2d_adaptive(map, args)
        # return h, w

    # try:
    if d == Distribution.Uniform:
        if args[0] < 2:
            return -1
        return np.random.randint(1, args[0])  # type: ignore

    if d == Distribution.Binomial:
        start, stop, *dist_args = args[0]
        return binom(stop - start - 1, dist_args).rvs() + start

    if d == Distribution.BetaBinomial:
        start, stop, *dist_args = args[0]
        stop -= 1
        alpha = dist_args[0][0]
        beta = dist_args[0][1]
        return betabinom(stop - start, alpha, beta).rvs() + start

    return -1

    # d is None or an option we don't recognise
    # return np.random.randint(start, stop)
    # except BaseException as e:
    #     logger.fatal("fatal error %s", e)
    #     return 0
    # sys.exit(-1)
