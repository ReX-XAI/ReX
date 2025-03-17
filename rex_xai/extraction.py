#!/usr/bin/env python
import re
from typing import Optional

import torch as tt

from rex_xai import visualisation
from rex_xai._utils import SpatialSearch, get_map_locations, set_boolean_mask_value
from rex_xai.config import CausalArgs, Strategy
from rex_xai.input_data import Data
from rex_xai.logger import logger
from rex_xai.mutant import _apply_to_data
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
        self.explanation_confidence = 0.0
        self.prediction_func = prediction_func
        self.data = data
        self.args = args
        self.run_stats = run_stats

    def __repr__(self) -> str:
        pred_func = repr(self.prediction_func)
        match_func_name = re.search(r"(<function .+) at", pred_func)
        if match_func_name:
            pred_func = match_func_name.group(1) + " >"

        run_stats = {k: round(v, 5) for k, v in self.run_stats.items()}

        exp_text = (
            "Explanation:"
            + f"\n\tCausalArgs: {type(self.args)}"
            + f"\n\tData: {self.data}"
            + f"\n\tprediction function: {pred_func}"
            + f"\n\tResponsibilityMaps: {self.maps}"
            + f"\n\trun statistics: {run_stats} (5 dp)"
        )

        if self.explanation is None or self.final_mask is None:
            return (
                exp_text
                + f"\n\texplanation: {self.explanation}"
                + f"\n\tfinal mask: {self.final_mask}"
                + f"\n\texplanation confidence: {self.explanation_confidence}"
            )
        else:
            return (
                exp_text
                + f"\n\texplanation: {type(self.explanation)} of shape {self.explanation.shape}"
                + f"\n\tfinal mask: {type(self.final_mask)} of shape {self.final_mask.shape}"
                + f"\n\texplanation confidence: {self.explanation_confidence:.5f} (5 dp)"
            )

    def extract(self, method: Strategy):
        self.blank()
        if method == Strategy.Global:
            self.__global()
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
        self.final_mask = None

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
            if len(masks) == self.args.batch_size:
                preds = self.prediction_func(tt.stack(masks).to(self.data.device))
                for j, p in enumerate(preds):
                    if (
                        p.classification == self.data.target.classification
                        and p.confidence
                        >= self.data.target.confidence
                        * self.args.minimum_confidence_threshold
                    ):  #  type: ignore
                        logger.info(
                            "found an explanation of %d with %f confidence",
                            p.classification,
                            p.confidence,
                        )
                        self.explanation = masks[j]
                        self.explanation_confidence = p.confidence
                        self.final_mask = mutant.zero_()
                        for _, loc in ranking[:limit]:
                            self.set_to_true(loc, self.final_mask)
                        return p.confidence
                masks = []

    def __generate_circle_coordinates(self, centre, radius: int):
        assert self.data.model_height is not None
        assert self.data.model_width is not None
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
            start_radius = self.args.spatial_initial_radius
        mask = tt.zeros(
            self.data.model_shape[1:], dtype=tt.bool, device=self.data.device
        )
        circle_mask = self.__generate_circle_coordinates(centre, start_radius)
        if self.data.model_order == "first":
            mask[:, circle_mask] = True
        else:
            mask[circle_mask, :] = True
        return start_radius, circle_mask, mask

    def compute_masked_responsibility(self, mask):
        try:
            masked_responsibility = tt.where(
                mask, self.target_map, self.data.mask_value
            )  # type: ignore
        except RuntimeError:
            masked_responsibility = tt.where(
                mask.permute((2, 0, 1)), self.target_map, self.data.mask_value
            )  # type: ignore
        except Exception as e:
            logger.fatal(e)
            exit()

        logger.debug("using %s", self.args.spotlight_objective_function)
        if self.args.spotlight_objective_function == "mean":
            return tt.mean(masked_responsibility).item()
        if self.args.spotlight_objective_function == "max":
            return tt.max(masked_responsibility).item()

        logger.warning(
            "unable to understand %s, so using mean for search",
            self.args.spotlight_objective_function,
        )
        return tt.mean(masked_responsibility).item()

    def __spatial(self, centre=None, expansion_limit=None):
        # we don't have a search location to start from, so we try to isolate one
        map = self.target_map
        if centre is None:
            centre = tt.unravel_index(tt.argmax(map), map.shape)  # type: ignore

        start_radius, circle, mask = self.__draw_circle(centre)

        if self.args.spotlight_objective_function == "none":
            masked_responsibility = None
        else:
            masked_responsibility = self.compute_masked_responsibility(mask)

        expansions = 0
        cutoff = (
            self.data.model_width * self.data.model_height * self.data.model_channels  # type: ignore
        )
        while tt.count_nonzero(mask) < cutoff:
            if expansion_limit is not None:
                if expansions >= expansion_limit and expansion_limit > 1:
                    logger.debug(
                        f"no explanation found after {expansion_limit} expansions"
                    )
                    return SpatialSearch.NotFound, masked_responsibility, None
            d = _apply_to_data(mask, self.data, self.data.mask_value)
            p = self.prediction_func(d)[0]
            if (
                p.classification == self.data.target.classification  # type: ignore
                and p.confidence
                >= self.data.target.confidence * self.args.minimum_confidence_threshold  # type: ignore
            ):
                conf = self.__global(map=tt.where(circle, map, 0))  # type: ignore
                return SpatialSearch.Found, masked_responsibility, conf
            start_radius = int(start_radius * (1 + self.args.spatial_radius_eta))
            _, circle, _ = self.__draw_circle(centre, start_radius)
            if self.data.model_order == "first":
                mask[:, circle] = True
            else:
                mask[circle, :] = True
            expansions += 1

    def save(self, path, mask=None, multi=None, multi_style="", clauses=None):
        # NOTE: the parameter multi_style="" is here simply to make overriding
        # the save function in MultiExplanation typecheck, same holds for clauses
        if self.data.mode in ("RGB", "L", "voxel"):
            if path is None:
                path = f"{self.data.target.classification}.png"  # type: ignore
            if mask is None:
                visualisation.save_image(
                    self.explanation, self.data, self.args, path=path
                )
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
                self.target_map,  # type: ignore
                self.data.target,  #  type: ignore
                path=path,
            )
        elif self.data.mode == "voxel":
            logger.warning("Surface plot not available for voxel data using voxel plot instead")
            visualisation.voxel_plot(
                self.args,
                self.target_map,
                self.data,
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
