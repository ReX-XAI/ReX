#!/usr/bin/env python
from typing import Optional

import torch as tt
import numpy as np

from rex_xai import visualisation
from rex_xai.input_data import Data
from rex_xai.config import CausalArgs
from rex_xai.config import Strategy
from rex_xai.logger import logger
from rex_xai.mutant import _apply_to_data
from rex_xai._utils import get_map_locations, set_boolean_mask_value
from rex_xai.resp_maps import ResponsibilityMaps


class Explanation:
    def __init__(
        self,
        maps: ResponsibilityMaps,
        prediction_func,
        data: Data,
        args: CausalArgs,
        run_stats: dict,
        keep_all_maps=False,
    ) -> None:
        if data.target is None or data.target.classification is None:
            raise (
                ValueError(
                    "Data must have `target` defined to create an Explanation object!"
                )
            )

        if keep_all_maps:
            self.maps = maps
        else:
            maps.subset(data.target.classification)
            self.maps = maps

        self.target_map: np.ndarray = maps.get(data.target.classification)  # type: ignore
        if self.target_map is None:
            raise ValueError(
                f"No responsibility map found for target {data.target.classification}!"
            )

        self.explanation: Optional[tt.Tensor] = None
        self.final_mask = None
        self.prediction_func = prediction_func
        self.data = data
        self.args = args
        self.run_stats = run_stats

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
            mask = set_boolean_mask_value(
                mask, self.data.mode, self.data.model_order, coords
            )

    def __global(self, map=None, wipe=False):
        if map is None:
            map = self.target_map
        ranking = get_map_locations(map)

        mutant = tt.zeros(
            self.data.model_shape[1:], dtype=tt.bool, device=self.data.device
        )
        masks = []

        limit = 0
        for i in range(0, len(ranking), self.args.chunk_size):
            chunk = ranking[i : i + self.args.chunk_size]
            limit += self.args.chunk_size
            for _, loc in chunk:
                self.set_to_true(loc, mutant)
            d = _apply_to_data(mutant, self.data, self.data.mask_value).squeeze(0)
            masks.append(d)
            if len(masks) == self.args.batch:
                preds = self.prediction_func(tt.stack(masks).to(self.data.device))
                for j, p in enumerate(preds):
                    if p.classification == self.data.target.classification:  #  type: ignore
                        logger.info(
                            "found an explanation of %f confidence",
                            p.confidence,
                        )
                        self.explanation = masks[j]
                        self.final_mask = mutant.zero_()
                        for _, loc in ranking[:limit]:
                            self.set_to_true(loc, self.final_mask)
                        # np.save("test", self.final_mask.detach().cpu().numpy())
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
        map = self.target_map
        if centre is None:
            centre = np.unravel_index(np.argmax(map), map.shape)

        start_radius = self.args.spatial_radius
        mask = tt.zeros(
            self.data.model_shape[1:], dtype=tt.bool, device=self.data.device
        )
        circle = self.__circle(centre, start_radius)
        if self.data.model_order == "first":
            mask[:, circle] = True
        else:
            mask[circle, :] = True

        expansions = 0
        cutoff = (
            self.data.model_width * self.data.model_height * self.data.model_channels  # type: ignore
        )
        while tt.count_nonzero(mask) < cutoff:
            if expansion_limit is not None:
                if expansions >= expansion_limit:
                    logger.info(
                        f"no explanation found after {expansion_limit} expansions"
                    )
                    return -1
            d = _apply_to_data(mask, self.data, self.data.mask_value)
            p = self.prediction_func(d)[0]
            if p.classification == self.data.target.classification:  #  type: ignore
                return self.__global(
                    map=np.where(circle.detach().cpu().numpy(), map, 0)
                )
            start_radius = int(start_radius * (1 + self.args.spatial_eta))
            circle = self.__circle(centre, start_radius)
            if self.data.model_order == "first":
                mask[:, circle] = True
            else:
                mask[circle, :] = True
            expansions += 1

    def save(self, path):
        if self.data.mode in ("RGB", "L", "voxel"):
            if path is None:
                path = f"{self.data.target.classification}.png"  # type: ignore
            visualisation.save_image(self.explanation, self.data, self.args, path=path)
        if self.data.mode == "spectral":
            visualisation.spectral_plot(
                self.explanation,
                self.data,
                self.target_map,
                self.args.heatmap_colours,
                path=path,
            )
        if self.data.mode == "tabular":
            pass
        if self.data.mode == "voxel":
            pass

    def heatmap_plot(self, path=None):
        if self.data.mode in ("RGB", "L"):
            visualisation.heatmap_plot(
                self.data,
                self.target_map,
                self.args.heatmap_colours,
                path=path,
            )
        elif self.data.mode == "voxel":
            visualisation.voxel_plot(
                self.args,
                self.target_map,
                self.data,
                path=path,
            )
        else:
            return NotImplementedError

    def surface_plot(self, path=None):
        if self.data.mode in ("RGB", "L"):
            visualisation.surface_plot(
                self.args,
                self.target_map,
                self.data.target,
                path=path,
            )
        elif self.data.mode == "voxel":
            visualisation.voxel_plot(
                self.args,
                self.target_map,
                self.data.target,  #  type: ignore
                path=path,
            )
        else:
            return NotImplementedError

    def show(self, path=None):
        if self.data.mode in ("RGB", "L", "voxel"):
            out = visualisation.save_image(
                self.explanation, self.data, self.args, path=path
            )
            return out
        else:
            return NotImplementedError
