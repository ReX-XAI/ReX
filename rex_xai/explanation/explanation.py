#!/usr/bin/env python
from __future__ import annotations

import re
from typing import Dict

import torch as tt
from tqdm import tqdm

from rex_xai.input.config import CausalArgs, Strategy
from rex_xai.input.input_data import Data
from rex_xai.mutants.mutant import _apply_to_data
from rex_xai.output import visualisation
from rex_xai.responsibility.resp_maps import ResponsibilityMaps
from rex_xai.utils._utils import (
    ReXPositions,
    SpatialSearch,
    find_complete_prediction,
    find_required_prediction,
    get_map_locations,
    set_boolean_mask_value,
    update_mask_shape,
)
from rex_xai.utils.logger import logger


class Explanation:
    def __init__(
        self,
        maps: ResponsibilityMaps,
        prediction_func,
        data: Data,
        args: CausalArgs,
        run_stats: Dict[str, float],
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

        self.target_map: tt.Tensor | None = tt.from_numpy(
            maps.get(data.target.classification)
        ).to(data.device)
        if self.target_map is None:
            raise ValueError(
                f"No responsibility map found for target {data.target.classification}!"
            )

        self.sufficiency_mask: tt.Tensor | None = None
        self.sufficiency_confidence: float | None = None
        self.necessity_mask: tt.Tensor | None = None
        self.complete_mask: tt.Tensor | None = None
        self.prediction_func = prediction_func
        self.data: Data = data
        self.args: CausalArgs = args
        self.run_stats: Dict[str, float] = run_stats

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
            + f"\n\texplanation confidence: {self.sufficiency_confidence:.4f}"
        )

    def extract(self):
        self.blank()
        if self.args.strategy == Strategy.Global:
            self.__global()
        if self.args.strategy == Strategy.Contrastive:
            self.contrastive()
        if self.args.strategy == Strategy.Spatial:
            if self.data.mode == "spectral":
                logger.warning(
                    "spatial search not yet implemented for spectral data, so defaulting to global search"
                )
                _ = self.__global()
            else:
                _ = self.__spatial()

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

    def __build_insertion_mask(self, ranking, chunk_pointer, ind, mask, mask_memo):
        chunk = ranking[chunk_pointer : chunk_pointer + self.args.chunk_size]

        if chunk == []:
            return ind, chunk_pointer, True

        # if the chunk is not empty...
        for _, loc in chunk:
            set_boolean_mask_value(
                mask[ind],
                self.data.mode,
                self.data.model_order,
                loc,
            )

        if ind == 0 and mask_memo is not None:
            mask[ind] = tt.logical_or(mask_memo, mask[ind])
        if ind > 0:
            mask[ind] = tt.logical_or(mask[ind - 1], mask[ind])

        chunk_pointer += self.args.chunk_size
        ind += 1
        return ind, chunk_pointer, False

    def __global(self, map=None, rounding=4):
        # get responsibility map and ranking
        if map is None:
            map = self.target_map
        ranking = get_map_locations(map)

        mask_shape = update_mask_shape(self.args.batch_size, self.data.model_shape)
        insertion_mask = tt.zeros(mask_shape, dtype=tt.bool).to(self.data.device)
        insertion_memo = None

        target_confidence: float = round(
            self.data.target.confidence * self.args.minimum_confidence_threshold,  # type: ignore
            rounding,
        )
        chunk_pointer = 0
        ind = 0
        sufficient_found = False

        # main loop
        with tqdm(
            total=len(ranking) // self.args.chunk_size,
            desc="Extracting global explanation",
        ) as pbar:
            while not sufficient_found:
                ind, chunk_pointer, exhausted = self.__build_insertion_mask(
                    ranking,
                    chunk_pointer,
                    ind,
                    insertion_mask,
                    insertion_memo,
                )

                pbar.update(1)

                if ind == self.args.batch_size or exhausted:
                    if exhausted:
                        insertion_mask = insertion_mask[:ind]

                    sufficient = self.prediction_func(
                        _apply_to_data(insertion_mask, self.data)
                    )

                    positions: ReXPositions = find_required_prediction(
                        self.data.target.classification,  # type: ignore
                        target_confidence,
                        sufficient,
                        rounding=rounding,
                    )

                    if not positions.is_empty():
                        if not sufficient_found:
                            self.sufficiency_mask = (
                                insertion_mask[positions.sufficient_position]
                                .detach()
                                .clone()
                            )
                            self.sufficiency_confidence = sufficient[
                                positions.sufficient_position
                            ].confidence
                            sufficient_found = True
                            logger.info(
                                f"a sufficient explanation for {self.data.target.classification} found with confidence {self.sufficiency_confidence:.4f}"
                            )

                            if False not in tt.unique(self.sufficiency_mask):
                                logger.info(
                                    f"the entire input was required to get sufficiency at {self.sufficiency_confidence:.4f} confidence",
                                )

                            return self.sufficiency_confidence
                    ind = 0
                    insertion_memo = insertion_mask[-1]

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
        # TODO  rewrite to use batching
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

    def __build_contrastive_masks(
        self,
        ranking,
        ind,
        chunk_pointer,
        insertion_mask,
        insertion_memo,
        deletion_mask,
        deletion_memo,
    ):
        chunk = ranking[chunk_pointer : chunk_pointer + self.args.chunk_size]

        if chunk == []:
            return ind, chunk_pointer, True

        for _, loc in chunk:
            set_boolean_mask_value(
                insertion_mask[ind],
                self.data.mode,
                self.data.model_order,
                loc,
            )
            set_boolean_mask_value(
                deletion_mask[ind],
                self.data.mode,
                self.data.model_order,
                loc,
                val=False,
            )

        if ind == 0 and insertion_memo is not None and deletion_memo is not None:
            insertion_mask[ind] = tt.logical_or(insertion_memo, insertion_mask[ind])
            deletion_mask[ind] = tt.logical_and(deletion_memo, deletion_mask[ind])
        if ind > 0:
            insertion_mask[ind] = tt.logical_or(
                insertion_mask[ind - 1], insertion_mask[ind]
            )
            deletion_mask[ind] = tt.logical_and(
                deletion_mask[ind - 1], deletion_mask[ind]
            )

        chunk_pointer += self.args.chunk_size
        ind += 1

        return ind, chunk_pointer, False

    def __complete(
        self,
        ranking,
        insertion_mask,
        insertion_memo,
        mask_shape,
        starting_pointer,
        rounding=2,
    ):
        target_confidence = self.data.target.confidence  # type: ignore
        step = self.args.chunk_size

        # check that the confidences aren't already at the correct levels
        if round(self.necessity_confidence, rounding) == target_confidence:  # type: ignore
            logger.info("the sufficient and necessary explanation is already complete")
            self.complete_mask = None
            self.completeness_confidence = self.necessity_confidence
            return

        complete_explanation_found = False
        insertion_mask = insertion_mask.zero_()
        insertion_mask[0] = self.necessity_mask.detach().clone()  # type: ignore
        exhausted = False

        chunk_pointer = len(ranking)
        ind = 1

        with tqdm(
            total=(len(ranking) - starting_pointer) // self.args.chunk_size,
            desc="Completeness Explanation",
        ) as pbar:
            while not complete_explanation_found:
                chunk = ranking[chunk_pointer - step : chunk_pointer]

                if chunk == []:
                    logger.info("the entire image is required for completeness")
                    self.complete_mask = tt.logical_xor(
                        tt.ones(mask_shape[1:], dtype=tt.bool).to(self.data.device),
                        self.necessity_mask.detach().clone(),  # type: ignore
                    )
                    return

                for _, loc in chunk:
                    set_boolean_mask_value(
                        insertion_mask[ind],
                        self.data.mode,
                        self.data.model_order,
                        loc,
                    )

                pbar.update(1)
                if ind == 0 and insertion_memo is not None:
                    insertion_mask[ind] = tt.logical_or(
                        insertion_memo, insertion_mask[ind]
                    )
                else:
                    insertion_mask[ind] = tt.logical_or(
                        insertion_mask[ind - 1], insertion_mask[ind]
                    )

                ind += 1
                chunk_pointer -= step

                if chunk_pointer <= starting_pointer:
                    exhausted = True

                if ind == self.args.batch_size or exhausted:
                    if exhausted:
                        insertion_mask = insertion_mask[:ind]

                    complete_predictions = self.prediction_func(
                        _apply_to_data(insertion_mask, self.data)
                    )

                    position = find_complete_prediction(
                        self.data.target.classification,  # type: ignore
                        target_confidence,
                        complete_predictions,
                    )

                    if position is not None:
                        self.complete_mask = tt.logical_xor(
                            self.necessity_mask,  # type: ignore
                            insertion_mask[position],  # type: ignore
                        )
                        cp = self.prediction_func(
                            _apply_to_data(self.complete_mask, self.data)
                        )[0]
                        self.completeness_classification = cp.classification
                        self.completeness_confidence = cp.confidence
                        complete_explanation_found = True
                        logger.info(
                            "found complete explanation for class %d. The completeness mask is class %d with confidence %.2f of size %d",
                            self.data.target.classification,  # type: ignore
                            self.completeness_classification,
                            self.completeness_confidence,
                            tt.count_nonzero(self.complete_mask)  # type: ignore
                            // self.data.model_channels,
                        )
                    else:
                        ind = 0

    def contrastive(self):
        rounding = 4
        mask_shape = update_mask_shape(self.args.batch_size, self.data.model_shape)

        insertion_mask = tt.zeros(mask_shape, dtype=tt.bool).to(self.data.device)
        deletion_mask = tt.ones(mask_shape, dtype=tt.bool).to(self.data.device)

        ranking = get_map_locations(map=self.target_map)

        target_confidence: float = (
            self.args.minimum_confidence_threshold * self.data.target.confidence  # type: ignore
        )
        contrastive_completeness_threshold: float = self.data.target.confidence  # type: ignore

        self.sufficiency_confidence = None
        self.necessity_mask = None
        self.necessity_confidence = None
        # the class and confidence of the pixels after self.necessity_mask has been removed
        self.contrastive_classification = None
        self.contrastive_confidence = None

        sufficient_found = False
        contrastive_found = False

        insertion_memo = None
        deletion_memo = None
        chunk_pointer = 0
        ind = 0

        with tqdm(
            total=len(ranking) // self.args.batch_size,
            desc="Calculating Contrastive Explanation",
        ) as pbar:
            while not contrastive_found:
                ind, chunk_pointer, exhausted = self.__build_contrastive_masks(
                    ranking,
                    ind,
                    chunk_pointer,
                    insertion_mask,
                    insertion_memo,
                    deletion_mask,
                    deletion_memo,
                )

                pbar.update(1)
                # we have filled insertion_mask and deletion_mask with test cases, now time to test
                if ind == self.args.batch_size or exhausted:
                    if exhausted:
                        insertion_mask = insertion_mask[:ind]
                        deletion_mask = deletion_mask[:ind]

                    sufficient = self.prediction_func(
                        _apply_to_data(insertion_mask, self.data)
                    )
                    contrastive = self.prediction_func(
                        _apply_to_data(deletion_mask, self.data)
                    )

                    positions: ReXPositions = find_required_prediction(
                        self.data.target.classification,  # type: ignore
                        target_confidence,
                        sufficient,
                        contrastive_completeness_threshold,
                        contrastive,
                        rounding=rounding,
                        sufficiency_found=sufficient_found,
                    )

                    if positions.sufficient_position is not None:
                        if not sufficient_found:
                            sufficient_found = True
                            self.sufficiency_mask = (
                                insertion_mask[positions.sufficient_position]
                                .detach()
                                .clone()
                            )
                            self.sufficiency_confidence = sufficient[
                                positions.sufficient_position
                            ].confidence
                            logger.info(
                                "a sufficient explanation for %d found with confidence %.4f of size %d",
                                self.data.target.classification,  # type: ignore
                                self.sufficiency_confidence,
                                tt.count_nonzero(self.sufficiency_mask)  # type: ignore
                                // self.data.model_channels,
                            )

                            if False not in tt.unique(self.sufficiency_mask):
                                logger.info(
                                    "the entire input was required to get sufficiency at %.4f confidence, so there is no independent necessity or completeness",
                                    self.sufficiency_confidence,
                                )
                                self.args.strategy = Strategy.Global
                                self.args.complete = False
                                return

                    if positions.contrastive_position is not None:
                        contrastive_found = True
                        self.necessity_mask = (
                            insertion_mask[positions.contrastive_position]
                            .detach()
                            .clone()
                        )
                        self.necessity_confidence = sufficient[
                            positions.contrastive_position
                        ].confidence
                        self.contrastive_classification = contrastive[
                            positions.contrastive_position
                        ].classification
                        self.contrastive_confidence = contrastive[
                            positions.contrastive_position
                        ].confidence

                        logger.info(
                            "a contrastive explanation for %d (now %d) found with confidence %.3f of size %d",
                            self.data.target.classification,  # type: ignore
                            self.contrastive_classification,
                            self.necessity_confidence,
                            tt.count_nonzero(self.necessity_mask)  # type: ignore
                            // self.data.model_channels,
                        )

                        if tt.count_nonzero(self.sufficiency_mask) == tt.count_nonzero(  # type: ignore
                            self.necessity_mask
                        ):
                            logger.info(
                                "there is no difference between sufficiency and necessity on this input"
                            )

                        if False not in self.necessity_mask:
                            logger.info(
                                "the sufficient and necessery explanation is already complete"
                            )
                            self.args.complete = False

                        if self.args.complete:
                            return self.__complete(
                                ranking,
                                insertion_mask,
                                insertion_memo,
                                mask_shape,
                                chunk_pointer,
                            )
                        return

                    insertion_memo = insertion_mask[-1]
                    deletion_memo = deletion_mask[-1]
                    ind = 0

    def save(self, path, mask=None):
        if mask is not None:
            visualisation.save_image(
                mask,
                self.data,
                self.args,
                path=path,
            )
        assert self.sufficiency_mask is not None
        if self.data.mode in ("RGB", "voxel") and mask is None:
            if self.args.complete:
                visualisation.save_complete(self, self.data, self.args, path=path)

            elif self.args.strategy == Strategy.Contrastive:
                visualisation.save_contrastive(self, self.data, self.args, path=path)
            else:
                mask = (
                    self.sufficiency_mask
                    if self.necessity_mask is None
                    else self.necessity_mask
                )
                visualisation.save_image(
                    self.sufficiency_mask,
                    self.data,
                    self.args,
                    path=path,
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
        assert self.sufficiency_mask is not None
        if self.data.mode in ("RGB", "voxel"):
            out = visualisation.save_image(
                self.sufficiency_mask,
                self.data,
                self.args,
                path=path,
            )
            return out
        else:
            return NotImplementedError
