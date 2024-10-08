#!/usr/bin/env python
from typing import Optional

import torch as tt
import numpy as np

from rex_ai.resp_maps import ResponsibilityMaps
from rex_ai.visualisation import save_image, spectral_plot, surface_plot, heatmap_plot
from rex_ai.prediction import Prediction
from rex_ai.input_data import Data
from rex_ai.config import CausalArgs
from rex_ai.config import Strategy
from rex_ai.logger import logger
from rex_ai.mutant import _apply_to_data
from rex_ai._utils import get_map_locations, set_boolean_mask_value


class Explanation:
    def __init__(
        self,
        map,
        prediction_func,
        target: Prediction,
        data: Data,
        args: CausalArgs,
    ) -> None:
        self.map = map.get(target.classification)
        self.explanation: Optional[tt.Tensor] = None
        self.prediction_func = prediction_func
        self.target = target
        self.data = data
        self.args = args
        self.mask_func = args.mask_value

    def extract(self, method: Strategy):
        self.blank()
        if method == Strategy.Global:
            return self.__global()
        if method == Strategy.MultiSpotlight:
            logger.warning("not yet implemented, defaulting to global")
            return self.__global()
            # name = None
            # ext = None
            # if self.args.output is not None:
            #     name, ext = os.path.splitext(os.path.basename((self.args.output)))
            # # get default spatial with centre set at responsibility mass
            # self.__spatial()
            # self.save()
            # for i in range(0, 10):
            #     self.args.output = f"{name}_{i}{ext}" #type: ignore
            #     m = MultiExplanation(self.map, self.data, self.target)
            #     centre = m.spotlight_search(self.args)
            #     print(centre, self.target.classification)
            #     r = self.__spatial(centre=centre, expansion_limit=4)
            #     while r == -1:
            #         print("trying in new location")
            #         centre = m.spotlight_search(self.args)
            #         r = self.__spatial(centre=centre, expansion_limit=4)
            #     if r != -1:
            #         self.save()

        if method == Strategy.Spatial:
            if self.data.mode == "spectral":
                logger.warning(
                    "spatial search not yet implemented for spectral data, so defaulting to global search"
                )
                self.__global()
            else:
                self.__spatial()

    def blank(self):
        assert self.data.data is not None
        self.explanation = tt.zeros(
            self.data.data.shape, dtype=tt.bool, device=self.data.device
        )

    def set_to_true(self, coords, mask=None):
        if mask is not None:
            mask = set_boolean_mask_value(mask, self.data.mode, self.data.model_order, coords)

    def __global(self, map=None, wipe=False):
        if map is None:
            map = self.map
        ranking = get_map_locations(map)

        mutant = tt.zeros(self.data.model_shape[1:], dtype=tt.bool, device=self.data.device)
        masks = []

        for i in range(0, len(ranking), self.args.chunk_size):
            chunk = ranking[i : i + self.args.chunk_size]
            for _, loc in chunk:
                self.set_to_true(loc, mutant)
            d = _apply_to_data(mutant, self.data, self.mask_func).squeeze(0)
            masks.append(d.unsqueeze(0))
            if len(masks) == self.args.batch:
                preds = self.prediction_func(
                    tt.stack(masks).to(self.data.device)
                )
                for j, p in enumerate(preds):
                    if p.classification == self.target.classification:
                        logger.info("found an explanation of %f confidence", p.confidence)
                        # TODO yuk, yuk, yuk. Everywhere else, explanation is a boolean mask, but here we change to float32
                        self.explanation = masks[j]
                        return
                masks = []


    def __circle(self, centre, radius: int):
        Y, X = np.ogrid[: self.data.model_height, : self.data.model_width]

        dist_from_centre = np.sqrt((Y - centre[0]) ** 2 + (X - centre[1]) ** 2)

        # this produces a H * W mask which can be using in conjunction with np.where()
        mask = dist_from_centre <= radius

        return tt.from_numpy(mask).to(self.data.device)

    def __spatial(self, centre=None, expansion_limit=None) -> Optional[int]:
        # we don't have a search location to start from, so we try to isolate one
        if centre is None:
            centre = np.unravel_index(np.argmax(self.map), self.map.shape)

        start_radius = self.args.spatial_radius
        mask = tt.zeros(self.data.model_shape[1:], dtype=tt.bool, device=self.data.device)
        circle = self.__circle(centre, start_radius)
        if self.data.model_order == "first":
            mask[:, circle] = True
        else:
            mask[circle, :] = True

        expansions = 0
        cutoff = self.data.model_width * self.data.model_height * self.data.model_channels # type: ignore
        while tt.count_nonzero(mask) < cutoff:
            if expansion_limit is not None:
                if expansions >= expansion_limit:
                    logger.info(f"no explanation found after {expansion_limit} expansions")
                    return -1
            d = _apply_to_data(mask, self.data, self.mask_func)
            p = self.prediction_func(d)[0]
            if p.classification == self.target.classification:
                return self.__global(map=np.where(circle.detach().cpu().numpy(), self.map, 0))
            start_radius = int(start_radius * (1 + self.args.spatial_eta))
            circle = self.__circle(centre, start_radius)
            if self.data.model_order == "first":
                mask[:, circle] = True
            else:
                mask[circle, :] = True
            expansions += 1


    def save(self):
        # if it's an image
        if self.data.mode in ("RGB", "L"):
            save_image(self.explanation, self.data, self.args)
        # if it's a spectral array
        if self.data.mode == "spectral":
            spectral_plot(self.args, self.explanation, self.data, self.map, self.args.heatmap_colours)
        # if it's tabular data
        if self.data.mode == "tabular":
            pass
        if self.data.mode == "voxel":
            pass


    def heatmap_plot(self, maps: ResponsibilityMaps):
        if self.data.mode in ("RGB", "L"):
            if self.args.heatmap == "show":
                heatmap_plot(self.data, maps, self.args.heatmap_colours, self.target)
            else:
                heatmap_plot(self.data, maps, self.args.heatmap_colours, self.target, path=self.args.heatmap)



    def surface_plot(self, maps: ResponsibilityMaps):
        if self.data.mode == "RGB" or self.data.mode == "L":
            surface_plot(
                self.args,
                maps,
                self.target,
                destination=self.args.surface,
            )
        else:
            return NotImplementedError
