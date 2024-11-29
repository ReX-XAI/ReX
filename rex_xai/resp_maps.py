#!/usr/bin/env python
import numpy as np
from typing import List

from typing import Optional
import sys

try:
    from anytree.cachedsearch import find
except ImportError:
    from anytree import find

from rex_xai.box import Box
from rex_xai.config import CausalArgs
from rex_xai.mutant import Mutant
from rex_xai.input_data import Data
from rex_xai.logger import logger


class ResponsibilityMaps:
    def __init__(self) -> None:
        self.maps = {}
        self.counts = {}

    def __repr__(self) -> str:
        return str(self.counts)

    def get(self, k):
        try:
            self.counts[k] += 1
            return self.maps[k]
        except KeyError:
            return None

    def new_map(self, k: int, height, width):
        self.maps[k] = np.zeros((height, width), dtype="float32")
        self.counts[k] = 1

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
            r_bad = np.where(r == 0)
            k = None
            if mutant.prediction is not None:
                k = mutant.prediction.classification
            if k is None:
                logger.fatal("this is no search classification, so exiting here")
                sys.exit(-1)
            if k not in self.maps:
                self.new_map(k, data.model_height, data.model_width)
            resp_map = self.get(k)
            assert resp_map is not None
            for box_name in mutant.get_active_boxes():
                box: Optional[Box] = find(search_tree, lambda n: n.name == box_name)
                if box is not None and box.area() > 0:
                    index = np.uint(box_name[-1])
                    section = resp_map[
                        box.row_start : box.row_stop,
                        box.col_start : box.col_stop,
                    ]
                    if all(section.shape):
                        if np.mean(section) == 0:
                            section += r[index]
                        else:
                            if args.concentrate:
                                section += np.mean(section) * r[index]
                            else:
                                section += r[index]
                    if args.concentrate:
                        for ind in r_bad:
                            for i in ind:
                                box_name = box_name[:-1] + str(i)
                                box: Optional[Box] = find(
                                    search_tree, lambda n: n.name == box_name
                                )
                                section = 0.001
            self.maps[k] = resp_map

    def subset(self, id):
        m = self.maps.get(id)
        c = self.counts.get(id)
        self.maps = {id: m}
        self.counts = {id: c}
