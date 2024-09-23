#!/usr/bin/env python
import numpy as np
from typing import List, Optional

try:
    from anytree.cachedsearch import find
except ImportError:
    from anytree.search import find

from rex_ai.logger import logger
from rex_ai.config import CausalArgs
from rex_ai.mutant import Mutant
from rex_ai.input_data import Data
from rex_ai.box import Box


class ResponsibilityMaps:
    def __init__(self) -> None:
        self.maps = {}

    def get(self, k):
        try:
            return self.maps[k]
        except KeyError:
            return None

    def new_map(self, k: int, height, width):
        self.maps[k] = np.zeros((height, width), dtype="float32")

    def items(self):
        return self.maps.items()

    def keys(self):
        return self.maps.keys()

    def len(self):
        return len(self.maps)

    def merge(self, maps):
        for k, v in maps.items():
            if k in self.maps:
                self.maps[k] += v
            else:
                self.maps[k] = v

    def responsibility(self, mutant: Mutant, args: CausalArgs):
        responsibility = np.zeros(4, dtype=np.float32)
        parts = mutant.get_active_parts()
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
            for part in mutant.get_active_parts():
                k = mutant.prediction.classification  # type: ignore
                assert k is not None
                if k not in self.maps:
                    self.new_map(k, data.model_height, data.model_width)
                # index for the appropriate responsibility
                i = np.uint(part[-1])
                box: Optional[Box] = find(search_tree, lambda node: node.name == part)
                if box is not None:
                    concentration = 1.0
                    if args.concentrate:
                        concentration = box.area() / (data.model_width * data.model_height) #type: ignore
                    # first time we've seen a particular classification
                    # continue with map (may be blank)
                    if k not in self.maps:
                        self.new_map(k, data.model_height, data.model_width)
                    resp_map = self.get(k)
                    if resp_map is not None:
                        # NB: no segmentation data here, so just boxes
                        if data.mode in ("spectral", "tabular"):
                            resp_map[0, box.col_start : box.col_stop] += r[i]
                        elif data.mode in ("RGB", "L"):
                            if box.area() == 0:
                                pass
                            else:
                                resp_map[
                                    box.row_start : box.row_stop,
                                    box.col_start : box.col_stop,
                                ] += r[i] * concentration # * (box.area() / (224 * 224)) * mutant.depth
                        else:
                            logger.warning("not yet implemented for voxels")
                            pass
                        self.maps[k] = resp_map
                    else:
                        logger.fatal("unable to update responsibility maps")
                        exit(-1)
