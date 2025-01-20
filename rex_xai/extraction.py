#!/usr/bin/env python
from typing import Optional, Tuple

import torch as tt

from rex_xai import visualisation
from rex_xai.input_data import Data
from rex_xai.config import CausalArgs
from rex_xai.config import Strategy
from rex_xai.logger import logger
from rex_xai.mutant import _apply_to_data
from rex_xai._utils import get_map_locations, set_boolean_mask_value, SpatialSearch
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

        self.target_map = tt.from_numpy(maps.get(data.target.classification)).to(
            data.device
        )
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
        if method == Strategy.Spatial:
            if self.data.mode == "spectral":
                logger.warning(
                    "spatial search not yet implemented for spectral data, so defaulting to global search"
                )
                self.__global()
            else:
                _ = self.__spatial()

        if isinstance(self.final_mask, tt.Tensor):
            self.final_mask = self.final_mask.detach().cpu().numpy()
        if isinstance(self.target_map, tt.Tensor):
            self.target_map = self.target_map.detach().cpu().numpy()

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
                            "found an explanation of %d with  %f confidence",
                            p.classification, p.confidence,
                        )
                        self.explanation = masks[j]
                        self.final_mask = mutant.zero_()
                        for _, loc in ranking[:limit]:
                            self.set_to_true(loc, self.final_mask)
                        return
                masks = []

    def __generate_circle_coordinates(self, centre, radius: int):
        Y, X = tt.meshgrid(
            tt.arange(0, self.data.model_height),
            tt.arange(0, self.data.model_width),
            indexing="ij",
        )

        dist_from_centre = tt.sqrt(
            (Y.to(self.data.device) - centre[0]) ** 2
            + (X.to(self.data.device) - centre[1]) ** 2
        )

        # this produces a H * W mask which can be using in conjunction with tt.where()
        circle_mask = dist_from_centre <= radius

        return circle_mask

    def __draw_circle(self, centre, start_radius=None):
        if start_radius is None:
            start_radius = self.args.spatial_radius
        mask = tt.zeros(
            self.data.model_shape[1:], dtype=tt.bool, device=self.data.device
        )
        circle_mask = self.__generate_circle_coordinates(centre, start_radius)
        if self.data.model_order == "first":
            mask[:, circle_mask] = True
        else:
            mask[circle_mask, :] = True
        return start_radius, circle_mask, mask

    def mean_masked_responsibility(self, mask):
        masked_responsibility = tt.where(mask, self.target_map, 0)  # type: ignore
        return self.args.spotlight_objective_function(masked_responsibility).item()

    def __spatial(
        self, centre=None, expansion_limit=None
    ) -> Optional[Tuple[SpatialSearch, float]]:
        # we don't have a search location to start from, so we try to isolate one
        map = self.target_map
        if centre is None:
            centre = tt.unravel_index(tt.argmax(map), map.shape)

        start_radius, circle, mask = self.__draw_circle(centre)

        if self.args.spotlight_objective_function is None:
            masked_responsibility = None
        else:
            masked_responsibility = self.mean_masked_responsibility(mask)

        expansions = 0
        cutoff = (
            self.data.model_width * self.data.model_height * self.data.model_channels  # type: ignore
        )
        while tt.count_nonzero(mask) < cutoff:
            if expansion_limit is not None:
                if expansions >= expansion_limit and expansion_limit > 1:
                    logger.info(
                        f"no explanation found after {expansion_limit} expansions"
                    )
                    return SpatialSearch.NotFound, masked_responsibility
            d = _apply_to_data(mask, self.data, self.data.mask_value)
            p = self.prediction_func(d)[0]
            if p.classification == self.data.target.classification:  #  type: ignore
                self.__global(map=tt.where(circle, map, 0))
                return SpatialSearch.Found, masked_responsibility
            start_radius = int(start_radius * (1 + self.args.spatial_eta))
            _, circle, _ = self.__draw_circle(centre, start_radius)
            if self.data.model_order == "first":
                mask[:, circle] = True
            else:
                mask[circle, :] = True
            expansions += 1

    def save(self, path, mask=None, multi=None, multi_style=""):
        if self.data.mode in ("RGB", "L"):
            if path is None:
                path = f"{self.data.target.classification}.png"  # type: ignore
            if mask is None:
                visualisation.save_image(self.explanation, self.data, self.args, path=path)
            else:
                visualisation.save_image(mask, self.data, self.args, path=path)
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
        else:
            return NotImplementedError

    def surface_plot(self, path=None):
        if self.data.mode in ("RGB", "L"):
            visualisation.surface_plot(
                self.args,
                self.target_map,
                self.data.target,  #  type: ignore
                path=path,
            )
        else:
            return NotImplementedError

    def show(self, path=None):
        if self.data.mode in ("RGB", "L"):
            out = visualisation.save_image(
                self.explanation, self.data, self.args, path=path
            )
            return out
        else:
            return NotImplementedError
