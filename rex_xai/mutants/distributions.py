"""distributions module"""

from typing import Optional, Tuple
from enum import Enum
import numpy as np
from scipy.stats import betabinom
from rex_xai.utils.logger import logger

Distribution = Enum("Distribution", ["Binomial", "Uniform", "BetaBinomial", "Adaptive"])


def _betabinom2d(height, width, alpha, beta):
    bx = betabinom(width, alpha, beta)
    by = betabinom(height, alpha, beta)

    w = np.array([bx.pmf(i) for i in range(0, width)])  # type: ignore
    h = np.array([by.pmf(i) for i in range(0, height)])  # type: ignore

    w = np.expand_dims(w, axis=0)
    h = np.expand_dims(h, axis=0)

    u = (h.T * w / np.sum(h.T * w)).ravel()
    p = np.random.choice(np.arange(0, len(u)), p=u)
    return p


def _blend(dist, alpha, base):
    pass
    # def random_coords(d: Optional[Distribution], *args, map=None) -> Optional[int]:


#     pmf = np.array([base.pmf(x) for x in range(0, len(dist))])
#     blend = ((1.0 - alpha) * pmf) + (alpha * dist)
#     blend /= np.sum(blend)
#     return blend
#
#
def _2d_adaptive(map, args: Tuple[int, int, int, int], alpha=0.0, base=None) -> int:
    return 0


#     # if the map exists and is not 0.0 everywhere...
#     if map is not None and np.max(map) > 0.0:
#         s = map[args[0] : args[1], args[2] : args[3]]
#         sf = np.ndarray.flatten(s)
#         # sf = np.max(sf) - sf
#         sf /= np.sum(sf)
#
#         # base = betabinom(0, len(sf), 1.1, 1.1)
#         # if base is not None:
#         #     sf = _blend(alpha, base)
#         pos = np.random.choice(np.arange(0, len(sf)), p=sf)
#         return pos
#
#     # if the map is empty or doesn't exist, return uniform
#     return np.random.randint(1, (args[1] - args[0]) * (args[3] - args[2]))


def str2distribution(d: str) -> Distribution:
    """converts string into Distribution enum"""
    if d == "binom":
        return Distribution.Binomial
    elif d == "uniform":
        return Distribution.Uniform
    elif d == "betabinom":
        return Distribution.BetaBinomial
    elif d == "adaptive":
        return Distribution.Adaptive
    else:
        logger.warning(
            "Invalid distribution '%s', reverting to uniform distribution",
            d,
        )
    return Distribution.Uniform


def random_coords(d: Optional[Distribution], *args, map=None):
    """generates random coordinates given a distribution and args"""

    try:
        if d == Distribution.Adaptive:
            return _2d_adaptive(map, args[0])

        if d == Distribution.Uniform:
            if args[1] == 1:
                return np.random.randint(1, args[0])
            else:
                return np.random.choice(args[0], args[1], replace=False)

        if d == Distribution.BetaBinomial:
            if args[1] == 1:
                return _betabinom2d(args[3], args[4], args[2][0], args[2][1])
            elif args[1] == 3:
                y = betabinom(args[0], *args[2])
                pmf = np.array([y.pmf(i) for i in range(0, args[0] + 1)])  # type:ignore
                return np.random.choice(args[0] + 1, 3, replace=False, p=pmf)
            else:
                pass

        return None
    except ValueError:
        return None
