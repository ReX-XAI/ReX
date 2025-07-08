#!/usr/bin/env python
from typing import List, Optional

import numpy as np

try:
    from anytree.cachedsearch import find
except ImportError:
    from anytree import find

from rex_xai.input.config import CausalArgs
from rex_xai.input.input_data import Data
from rex_xai.mutants.box import Box
from rex_xai.mutants.mutant import Mutant
from rex_xai.utils._utils import ResponsibilityStyle, ReXMapError
from rex_xai.utils.logger import logger


class ResponsibilityMaps:
    def __init__(self, style) -> None:
        self.maps = {}
        self.counts = {}
        self.style = style

    def __repr__(self) -> str:
        return str(self.counts)

    def get(self, k, increment=False):
        try:
            if increment:
                self.counts[k] += 1  # type: ignore
            return self.maps[k]
        except KeyError:
            return

    def new_map(self, k: int, height, width, depth=None):
        if depth is not None:
            if self.style == ResponsibilityStyle.Additive:
                self.maps[k] = np.zeros((height, width, depth), dtype="float32")
            else:
                self.maps[k] = np.ones((height, width, depth), dtype="float32")
            self.counts[k] = 1
        else:
            if self.style == "additive":
                self.maps[k] = np.zeros((height, width), dtype="float32")
            else:
                self.maps[k] = np.ones((height, width), dtype="float32")
            self.counts[k] = 1

    def items(self):
        return self.maps.items()

    def keys(self):
        return self.maps.keys()

    def len(self):
        return len(self.maps)

    def merge(self, maps):
        for k, v in maps.items():
            if np.max(v) == 0:
                pass
            if k in self.maps:
                if self.style == ResponsibilityStyle.Additive:
                    self.maps[k] += v
                else:
                    self.maps[k] *= v
            else:
                self.maps[k] = v

    def negative_responsibility(self, target):
        for k, v in self.maps.items():
            if k != target:
                logger.debug(
                    f"subtracting responsibility for class {k} from class {target}"
                )
                self.maps[target] = self.maps[target] - v  # type: ignore

    def responsibility(self, mutant: Mutant, args: CausalArgs):
        responsibility = np.zeros(4, dtype=np.float32)
        parts = mutant.get_active_boxes()
        r = 1 / len(parts)
        for p in parts:
            i = np.uint(p[-1])
            if (
                args.weighted
                and mutant.prediction is not None
                and mutant.prediction.confidence is not None
            ):
                responsibility[i] += r * mutant.prediction.confidence
            else:
                responsibility[i] += r
        return responsibility

    def update_maps(
        self, mutants: List[Mutant], args: CausalArgs, data: Data, search_tree
    ):
        """Update the different responsibility maps with all passing mutants <mutants>
        @params mutants: list of mutants
        @params args: causal args
        @params data: data
        @params search_tree: tree of boxes

        Mutates in place, does not return a value
        """

        for mutant in mutants:
            r = self.responsibility(mutant, args)

            k = None
            # check that there is a prediction value
            if mutant.prediction is not None:
                k = mutant.prediction.classification
            # if there's no prediction value, raise an exception
            if k is None:
                raise ReXMapError("the provided mutant has no known classification")
            # check if k has been seen before and has a map. If k is new, make a new map
            if k not in self.maps:
                self.new_map(k, data.model_height, data.model_width, data.model_depth)

            # get the responsibility map for k
            resp_map = self.get(k, increment=True)
            if resp_map is None:
                raise ValueError(
                    f"unable to open or generate a responsibility map for classification {k}"
                )

            # we only increment responsibility for active boxes, not static boxes
            for box_name in mutant.get_active_boxes():
                box: Optional[Box] = find(search_tree, lambda n: n.name == box_name)
                if box is not None and box.area() > 0:
                    index = np.uint(box_name[-1])
                    local_r = r[index]
                    if args.concentrate:
                        local_r *= box.depth

                    if data.mode == "spectral":
                        section = resp_map[0, box.col_start : box.col_stop]

                    elif data.mode == "RGB":
                        section = resp_map[
                            box.row_start : box.row_stop,
                            box.col_start : box.col_stop,
                        ]
                    elif data.mode == "voxel":
                        section = resp_map[
                            box.row_start : box.row_stop,
                            box.col_start : box.col_stop,
                            box.depth_start : box.depth_stop,
                        ]
                    else:
                        logger.warning("not yet implemented")
                        raise NotImplementedError

                    section += local_r
            self.maps[k] = resp_map

    def subset(self, id):
        m = self.maps.get(id)
        c = self.counts.get(id)
        self.maps = {id: m}
        self.counts = {id: c}
