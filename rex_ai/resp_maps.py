#!/usr/bin/env python
import numpy as np
from typing import List

import sys
try:
    from anytree.cachedsearch import find
except ImportError:
    pass

from rex_ai.config import CausalArgs
from rex_ai.mutant import Mutant
from rex_ai.input_data import Data
from rex_ai.logger import logger


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
            # r_good = np.where(r > 0)
            # r_bad = np.where(r == 0)
            k = None
            if mutant.prediction is not None:
                k = mutant.prediction.classification
            if k is None:
                logger.fatal("bad bad bad")
                sys.exit(-1)
            if k not in self.maps:
                self.new_map(k, data.model_height, data.model_width)
            resp_map = self.get(k)
            assert resp_map is not None
            parts = [np.uint(p[-1]) for p in mutant.get_active_parts()]
            print(r, parts, r[parts])

            # for p in mutant.get_active_parts():
            #
            #     # p is a string of the form e.g. 'R:3:2:1', we want
            #     # to get the last element of that string '1' and convert to 1
            #     # i = np.uint(p[-1])
            #     box: Optional[Box] = find(search_tree, lambda n: n.name == p)
            #     if box is not None:
            #         if box.area() == 0:
            #             pass
            #         else:
            #             section = resp_map[box.row_start : box.row_stop, box.col_start : box.col_stop]
            #             update = r[np.uint(p[-1])]






        # for mutant in mutants:
        #     r = self.responsibility(mutant, args)
        #     r_good = np.where(r > 0.0)
        #     r_bad = np.where(r == 0.0)
        #     print(r_good, r_bad)
        #     for part in mutant.get_active_parts():
        #         assert mutant.prediction is not None
        #         k = mutant.prediction.classification  
        #         assert k is not None
        #         if k not in self.maps:
        #             self.new_map(k, data.model_height, data.model_width)
        #         # index for the appropriate responsibility
        #         i = np.uint(part[-1])
        #         box: Optional[Box] = find(search_tree, lambda node: node.name == part)
        #         if box is not None:
        #             concentration = 1.0
        #             if args.concentrate:
        #                 concentration = box.area() / (data.model_width * data.model_height) #type: ignore
        #             # first time we've seen a particular classification
        #             # continue with map (may be blank)
        #             # if k not in self.maps:
        #             #     self.new_map(k, data.model_height, data.model_width)
        #             resp_map = self.get(k)
        #             if resp_map is not None:
        #                 # NB: no segmentation data here, so just boxes
        #                 if data.mode in ("spectral", "tabular"):
        #                     resp_map[0, box.col_start : box.col_stop] += r[i]
        #                 elif data.mode in ("RGB", "L"):
        #                     # TODO not sure why this should ever occur, but it does. Indicates a mistake somewhere
        #                     # else in the code
        #                     if box.area() == 0:
        #                         pass
        #                     else:
        #                         section = resp_map[box.row_start : box.row_stop, box.col_start : box.col_stop]
        #                         section += r[i] * np.min(section)
        #
        #                         # if r[i] == 0 and np.min(section) > 0:
        #                         #     section = 0
        #                         # else:
        #                         #     section += np.min(section) * r[i]
        #                         # print(np.min(section), np.mean(section), np.max(section))
        #                         # if np.min(section) == 0:
        #                         #     section += r[i]
        #                         # else:
        #                         #     # section += r[i]
        #                         #     section += np.min(section) * r[i] * concentration
        #                             # section += np.mean(section) * r[i] * concentration
        #
        #                         # print(np.min(section), np.max(section))
        #                         # section += np.min(section) * r[i]
        #                         # resp_map[
        #                         #     box.row_start : box.row_stop,
        #                         #     box.col_start : box.col_stop,
        #                         # ] += r[i] * concentration
        #                 else:
        #                     logger.warning("not yet implemented for voxels")
        #                     pass
        #                 self.maps[k] = resp_map
        #             else:
        #                 logger.fatal("unable to update responsibility maps")
        #                 exit(-1)
