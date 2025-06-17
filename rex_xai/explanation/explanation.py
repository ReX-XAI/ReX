#!/usr/bin/env python
from __future__ import annotations

import re

import torch as tt

from rex_xai.input.config import CausalArgs, Strategy
from rex_xai.input.input_data import Data
from rex_xai.mutants.mutant import _apply_to_data
from rex_xai.output import visualisation
from rex_xai.responsibility.resp_maps import ResponsibilityMaps
from rex_xai.utils._utils import (
    SpatialSearch,
    get_map_locations,
    set_boolean_mask_value,
    try_detach,
)
from rex_xai.utils.logger import logger


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

        self.sufficiency_mask = None
        self.sufficiency_confidence = 0.0
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

        return (
            exp_text
            + f"\n\tsufficiency mask: {self.sufficiency_mask}"
            + f"\n\texplanation confidence: {self.sufficiency_confidence}"
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
                _ = self.__global()
            else:
                _ = self.__spatial()

        self.sufficiency_mask = try_detach(self.sufficiency_mask)
        self.target_map = try_detach(self.target_map)

    def blank(self):
        assert self.data.data is not None
        self.sufficiency_mask = tt.zeros(
            self.data.data.shape, dtype=tt.bool, device=self.data.device
        )

    def set_to_true(self, coords, mask=None):
        if mask is not None:
            mask = set_boolean_mask_value(
                mask, self.data.mode, self.data.model_order, coords
            )

    def __global(self, map=None):
        if map is None:
            map = self.target_map
        ranking = get_map_locations(map)

        mutant = tt.zeros(
            self.data.model_shape[1:], dtype=tt.bool, device=self.data.device
        )
        masks = []
        tests = []

        limit = 0
        for i in range(0, len(ranking), self.args.chunk_size):
            chunk = ranking[i : i + self.args.chunk_size]
            limit += self.args.chunk_size
            for _, loc in chunk:
                self.set_to_true(loc, mutant)
            masks.append(mutant.detach().clone())
            tests.append(_apply_to_data(mutant, self.data).squeeze(0))
            if len(masks) == self.args.batch_size:
                preds = self.prediction_func(tt.stack(tests).to(self.data.device))
                for j, p in enumerate(preds):
                    if (
                        p.classification == self.data.target.classification  # type:ignore
                        and p.confidence
                        >= self.data.target.confidence  # type:ignore
                        * self.args.minimum_confidence_threshold
                    ):  #  type: ignore
                        logger.info(
                            "found an explanation of %d with %f confidence",
                            p.classification,
                            p.confidence,
                        )
                        self.sufficiency_confidence = p.confidence
                        self.sufficiency_mask = masks[j]
                        return p.confidence
                masks = []
                tests = []

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
                mask,
                self.target_map,  # type: ignore
                self.data.mask_value,  # type: ignore
            )  # type: ignore
        except RuntimeError:
            masked_responsibility = tt.where(
                mask.permute((2, 0, 1)),
                self.target_map,  # type: ignore
                self.data.mask_value,  # type: ignore
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
            d = _apply_to_data(mask, self.data)
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

    def contrastive(self):
        insertion_mask = tt.zeros(self.data.data.squeeze(0).shape, dtype=tt.bool).to(
            self.data.device
        )
        deletion_mask = tt.ones(self.data.data.squeeze(0).shape, dtype=tt.bool).to(
            self.data.device
        )

        ranking = get_map_locations(map=self.target_map)

        target_confidence = (
            self.args.minimum_confidence_threshold * self.data.target.confidence  # type: ignore
        )

        self.necessity_confidence = 0.0
        self.necessity_classification = None
        self.inverse_classification = None
        self.inverse_confidence = None

        step = self.args.chunk_size
        sufficient_found = False
        found = False

        i = 0
        while not found:
            chunk = ranking[i : i + step]
            for _, loc in chunk:
                set_boolean_mask_value(
                    insertion_mask,
                    self.data.mode,
                    self.data.model_order,
                    loc,
                )
                set_boolean_mask_value(
                    deletion_mask,
                    self.data.mode,
                    self.data.model_order,
                    loc,
                    val=False,
                )

            # mutants for insertion and deletion of pixels
            sufficient = self.prediction_func(_apply_to_data(insertion_mask, self.data))
            necessary = self.prediction_func(_apply_to_data(deletion_mask, self.data))

            assert self.data.target is not None
            # at the moment we do not have batching for this function, so len(sufficient) == 1 all the time
            for j in range(0, len(sufficient)):
                if (
                    not sufficient_found
                    and sufficient[j].classification == self.data.target.classification
                    and sufficient[j].confidence >= target_confidence
                ):
                    # sufficient explanation has been found
                    sufficient_found = True
                    logger.info(
                        "found sufficient explanation of class %d with confidence %f",
                        sufficient[j].classification,
                        sufficient[j].confidence,
                    )

                    # set sufficiency_mask and confidence
                    self.sufficiency_mask = insertion_mask.detach().clone()
                    self.sufficiency_confidence = sufficient[j].confidence

                    # we have found a sufficient explanation. If we are looking for a `complete` explanation,
                    # then we need to reset the `minimum_confidence_threshold` to 1 in order to find a core of
                    # enough confidence
                    if (
                        self.args.complete
                        and self.args.minimum_confidence_threshold < 1.0
                    ):
                        logger.info(
                            "setting the minimum confidence threshold to 1 in order to calculate a complete explanation."
                        )
                        target_confidence = self.data.target.confidence
                # if we get here, then we have already found a minimal, sufficient mask
                elif (
                    sufficient[j].classification == self.data.target.classification  # type: ignore
                    and necessary[j].classification != self.data.target.classification  # type: ignore
                    and sufficient[j].confidence >= target_confidence
                ):
                    self.necessity_classification = necessary[j].classification
                    self.inverse_classification = necessary[j].classification
                    self.inverse_confidence = necessary[j].confidence
                    self.necessity_mask = insertion_mask.detach().clone()
                    self.necessity_confidence = sufficient[j].confidence
                    logger.info(
                        "found contrastive explanation, changing %d to %d with confidence %.3f",
                        self.data.target.classification,
                        self.necessity_classification,
                        self.inverse_confidence,
                    )
                    # stop the loop here
                    found = True

            i += step

        # completeness
        if self.args.complete:
            # set a new target <completeness_confidence> which we need to bring as close to <target_confidence> as we can
            completeness_confidence = round(self.necessity_confidence, 2)
            step = 5  # hard coded fro the moment
            rounding = 2  # also hard coded for the moment
            len_ranking = len(ranking)
            target_confidence = round(self.data.target.confidence, rounding)  # type: ignore
            while completeness_confidence > target_confidence:  # type: ignore
                chunk = ranking[len_ranking - step : len_ranking]
                for _, loc in chunk:
                    set_boolean_mask_value(
                        insertion_mask,
                        self.data.mode,
                        self.data.model_order,
                        loc,
                    )
                sufficient = self.prediction_func(
                    _apply_to_data(insertion_mask, self.data)
                )
                completeness_confidence = round(sufficient[0].confidence, rounding)
                len_ranking -= step
                # if len_ranking is less than <i> then we are infringing on the necessity_mask. This should never happen...
                if len_ranking < i:
                    logger.warning(
                        "unable to find a minimal completeness explanation",
                        sufficient[0].confidence,
                        target_confidence,
                    )
                    break

            # subtract the necessity_mask from the completenss mask so that we can have it separate
            self.complete_mask = tt.logical_xor(
                insertion_mask.detach().clone(), self.necessity_mask
            )

            cp = self.prediction_func(_apply_to_data(self.complete_mask, self.data))[0]

            self.completeness_classification = cp.classification
            self.completeness_confidence = cp.confidence
            diff = self.necessity_confidence - self.data.target.confidence  # type: ignore
            direction = "increases" if diff < 0 else "reduces"
            logger.info(
                (
                    "found sufficient, necessary and complete explanation of class %d with confidence %.3f "
                    + "where sufficient and necessary explanation has confidence %.3f. "
                    + "Removing these pixels results in class %d with confidence %.3f."
                    + "The complete explanation %s the sufficient and necessary confidence by %.3f and has class %d "
                    + "with confidence %.3f."
                ),
                self.data.target.classification,  # type: ignore
                completeness_confidence,  # type: ignore
                self.necessity_confidence,  # type: ignore
                self.necessity_classification,  # type: ignore
                self.inverse_confidence,  # type: ignore
                direction,
                abs(diff),
                self.completeness_classification,
                self.completeness_confidence,
            )

    def save(self, path, mask=None):
        if self.data.mode in ("RGB", "voxel") and mask is None:
            if self.args.complete:
                visualisation.save_complete(self, self.data, self.args, path=path)
            else:
                visualisation.save_image(
                    self.sufficiency_mask,
                    self.data,
                    self.args,
                    path=path,
                    mask=self.sufficiency_mask,
                )

        if self.data.mode == "spectral":
            visualisation.spectral_plot(
                self.sufficiency_mask,
                self.data,
                self.target_map,
                self.args.heatmap_colours,
                path=path,
            )
        if self.data.mode == "tabular":
            pass

    def heatmap_plot(self, path=None):
        if self.target_map is not None:
            if self.data.mode == "RGB":
                visualisation.heatmap_plot(
                    self.data,
                    self.target_map,
                    self.args.heatmap_colours,
                    path=path,
                )
            elif self.data.mode == "voxel":
                visualisation.voxel_plot(
                    self.args,
                    self.target_map,  # type: ignore
                    self.data,
                    path=path,
                )
            else:
                return NotImplementedError

    def surface_plot(self, path=None):
        if self.data.mode == "RGB":
            visualisation.surface_plot(
                self.data.input,
                self.args,
                self.target_map,  # type: ignore
                self.data.target,  #  type: ignore
                path=path,
            )
        elif self.data.mode == "voxel":
            logger.warning(
                "Surface plot not available for voxel data using voxel plot instead"
            )
            visualisation.voxel_plot(
                self.args,
                self.target_map,  # type: ignore
                self.data,
                path=path,
            )
        else:
            return NotImplementedError

    def show(self, path=None):
        if self.data.mode in ("RGB", "voxel"):
            out = visualisation.save_image(
                self.sufficiency_mask, self.data, self.args, path=path
            )
            return out
        else:
            return NotImplementedError
