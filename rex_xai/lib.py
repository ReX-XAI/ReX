from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch as tt
from PIL import Image

from rex_xai.explanation.explanation import Explanation
from rex_xai.explanation.rex import calculate_responsibility, predict_target
from rex_xai.input.config import CausalArgs
from rex_xai.input.input_data import Data
from rex_xai.output import visualisation
from rex_xai.responsibility.prediction import default_prediction_function
from rex_xai.responsibility.resp_maps import ResponsibilityMaps
from rex_xai.utils._utils import Strategy


class ReX:
    def __init__(
        self,
        model,
        model_shape: List[int | str] | Tuple[int | str],
        device: str | tt.device,
        mode: str,
        prediction_function: Callable | None = None,
    ) -> None:
        self.args: CausalArgs = CausalArgs()
        self.model = model
        self.model_shape: List[int | str] | Tuple[int | str] = model_shape
        self.device: str | tt.device = device
        self.mode: str = mode
        self.explanation = None
        self.prediction_function = prediction_function
        self.data = None
        self.maps: ResponsibilityMaps | None = None
        self.stats: Dict[str, float] | None = None

        if self.prediction_function is None:
            self.get_default_prediction_function()

    def set_tabular_data(self, path: str, data=None):
        if self.mode == "spectral":
            self.args.path = path
            self.args.mode = "spectral"
            if data is not None:
                self.data = Data(
                    data,
                    data.shape,
                    self.device,
                    mode=self.args.mode,
                    process=True,
                )

    def set_input(self, input):
        if self.data is not None:
            self.data.input = input

    def set_transformed_data(self, transformed_data: tt.Tensor):
        if self.data is not None:
            self.data.data = transformed_data

    def set_rgb_image(self, path: str):
        if self.mode == "RGB":
            self.args.path = path
            self.args.mode = "RGB"
            img = Image.open(self.args.path).convert("RGB")
            self.data = Data(img, self.model_shape, self.device, mode="RGB")
            return img

    def show_target(self):
        if self.data is not None and self.data.target is not None:
            print(self.data.target)
        else:
            print("a target has not yet been set")

    def get_default_prediction_function(self):
        self.prediction_function = default_prediction_function(self.model)

    def set_target(self):
        if self.data is not None:
            self.data.target = predict_target(
                self.data, self.args, self.prediction_function
            )
        return self

    def set_prediction_function(self, function):
        self.prediction_function = function

    def calculate_responsibility(self, args: CausalArgs | None = None):
        if args is None:
            args = self.args
        if self.data is not None:
            if self.data.mask_value is None:
                self.data.set_mask_value(args.mask_value)
            maps, stats = calculate_responsibility(
                self.data, args, self.prediction_function
            )
            self.maps = maps
            self.stats = stats
            return self

    def generate_explanation_object(self):
        if self.data is not None and self.maps is not None and self.stats is not None:
            self.explanation = Explanation(
                self.maps,
                self.prediction_function,
                self.data,
                self.args,
                self.stats,
            )
        return self

    def extract_sufficient_explanation(self):
        if self.explanation is not None:
            self.explanation.extract()
        else:
            self.generate_explanation_object().explanation.extract()  # type: ignore

    def extract_contrastive_explanation(self):
        self.args.strategy = Strategy.Contrastive
        self.args.complete = False
        self.generate_explanation_object().explanation.extract()  # type: ignore

    def extract_complete_explanation(self):
        self.args.strategy = Strategy.Contrastive
        self.args.complete = True
        self.generate_explanation_object().explanation.extract()  # type: ignore

    def analyse(self):
        pass

    def rerun_with(self, new_args: CausalArgs):
        self.calculate_responsibility(args=new_args)

    def show(self):
        if self.explanation is not None and self.data is not None:
            if self.mode == "RGB":
                if self.args.complete:
                    return visualisation.save_complete(
                        self.explanation, self.data, self.args
                    )
                if self.args.strategy == Strategy.Contrastive:
                    return visualisation.save_contrastive(
                        self.explanation, self.data, self.args
                    )
                elif self.args.strategy == Strategy.Global:
                    return visualisation.save_image(
                        self.explanation.sufficiency_mask,  # type: ignore
                        self.data,
                        self.args,  # type: ignore
                    )
                else:
                    pass

            if self.mode == "spectral":
                visualisation.spectral_plot(
                    self.explanation.sufficiency_mask,
                    self.data,  # type: ignore
                    self.maps.get(self.data.target.classification),  # type: ignore
                    self.args.heatmap_colours,
                )
